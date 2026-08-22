"""
Base agent class for multi-agent system.
Provides common interface for all agents.
"""
import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable, Dict, Optional

from pydantic import BaseModel, ValidationError

from app.agents import grounding as _grounding
from app.agents import persona as _persona
from app.agents.actions import action_gateway
from app.services.call_metrics import (
    record_llm_failure,
    record_llm_rate_limited,
    record_llm_request,
    record_llm_retry,
    record_llm_timeout,
)
from app.services.groq_limiter import estimate_tokens
from app.services.groq_service import groq_service
from app.services.llm_errors import (
    LLMErrorKind,
    classify_llm_error,
    retry_after_seconds,
)
from app.tools.contract import (
    Effect,
    ErrorKind,
    ToolResult,
    coerce,
    effect_for_spec,
)

logger = logging.getLogger(__name__)

# Memory injected into an agent prompt is capped at this many characters.
#
# Was 20 000 (~5 000 tokens), which was sized against Groq's *context window* —
# the wrong constraint. The binding limit here is tokens per minute, not tokens
# per request, and the reasoning loop re-sends the whole prompt on every
# iteration, so a three-step turn transmitted 15 000 tokens of memory to answer
# one question. 6 000 chars matches `settings.memory_v2_budget_tokens`, which is
# the budget the v2 assembler already allocates against; the two numbers were
# doing the same job and disagreeing by 3×.
#
# Sections are priority-ordered upstream (profile facts first, résumé last), so
# a tighter cap costs the least useful content first. See
# `MemoryManager.format_context_for_prompt`.
_MAX_MEMORY_CHARS = 6_000

# Sentinels around the injected memory block, so the reasoning loop can drop it
# from follow-up iterations without re-deriving what it contained. Markers
# rather than a regex over the prose: the block's own text changes, and a split
# that silently stops matching would quietly restore the cost this removes.
_MEMORY_BLOCK_START = "<!--memory:start-->"
_MEMORY_BLOCK_END = "<!--memory:end-->"


def _without_memory_block(system_prompt: str) -> str:
    """
    The same prompt with the memory block replaced by a one-line reminder.

    Used from the second reasoning iteration onward. The model has already read
    the memory once in this exchange and the conversation carries forward; what
    it needs on a follow-up pass is the new tool observations, not another copy
    of the résumé. Prompts with no memory block are returned untouched.
    """
    start = system_prompt.find(_MEMORY_BLOCK_START)
    end = system_prompt.find(_MEMORY_BLOCK_END)
    if start == -1 or end == -1 or end < start:
        return system_prompt
    tail = system_prompt[end + len(_MEMORY_BLOCK_END):]
    return (
        system_prompt[:start]
        + "## Memory Context\n(Provided in full on the first step of this turn; "
        + "it has not changed. Work from it and from the tool observations below.)"
        + tail
    )

# Per-operation timeouts for the reasoning loop.
# Without these the loop hangs indefinitely if Groq or a tool is unresponsive.
#
# `_LLM_CALL_TIMEOUT` bounds a single HTTP attempt, applied inside `call_groq`.
# It used to be applied *around* the whole retry sequence, whose own worst case
# (2 × the 20 s client timeout, plus backoff) was longer than the budget — so
# the final attempt could never finish and the work of the earlier ones was
# discarded along with it.
_LLM_CALL_TIMEOUT = 30.0   # seconds per Groq HTTP attempt
_TOOL_CALL_TIMEOUT = 15.0  # seconds per tool call

# Attempts per logical completion, this layer only. Was 3, and sat underneath
# an SDK that was itself retrying, inside a reasoning loop, inside a reflect
# loop. Two is enough to ride out a single transient fault; anything beyond
# that is the amplification the audit measured.
_MAX_LLM_ATTEMPTS = 2

# What to wait after a 429 that arrived without a Retry-After header. Groq's
# token windows are minute-shaped, so a second-scale guess is optimistic —
# but not as optimistic as the 1 s the old loop used unconditionally.
_RATE_LIMIT_DEFAULT_WAIT = 5.0

# Longest we will sit out a rate limit before failing instead. Past this the
# caller's own deadline (a spoken turn's stall watchdog, the workflow ceiling)
# would expire mid-sleep anyway, and failing visibly beats holding a turn open
# to produce nothing.
_RATE_LIMIT_MAX_WAIT = 15.0


def _transient_backoff(attempt: int) -> float:
    """Exponential backoff for retryable faults: 1 s, 2 s, 4 s …"""
    return float(2 ** (attempt - 1))

# Strong references to detached background writes. asyncio only holds a weak
# reference to a running task, so without this a fire-and-forget task can be
# garbage-collected mid-execution and its exception never observed.
_background_tasks: set[asyncio.Task] = set()


def _spawn_background(coro, description: str) -> None:
    """Run a non-critical coroutine detached, keeping it alive and logged."""
    task = asyncio.create_task(coro, name=description)
    _background_tasks.add(task)

    def _on_done(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
        try:
            t.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("Background task '%s' failed: %s", description, exc)

    task.add_done_callback(_on_done)


# ─────────────────────────────────────────────────────────────
# What the tools actually produced
# ─────────────────────────────────────────────────────────────
#
# The honesty rules live in every specialist's system prompt: a tool that
# failed is not a tool that found nothing; never convert a failure into "you
# have no X on file". Those sentences are correct and they are also only
# instructions — the model may follow them, and a model shown an empty result
# is in precisely the state where it invents a plausible value instead.
#
# The classification itself now lives in `app.tools.contract`, which reads the
# same conventions these helpers used to inspect by hand. They remain as thin
# predicates over that adapter: several call sites and tests ask the yes/no
# question directly, and a `ToolResult` is a heavier answer than those need.
#
# Note the fixed READ effect below. These take a bare return value with no
# registry entry attached, so there is no declared effect to honour — and READ
# is the only assumption that keeps the lenient legacy interpretation these
# callers were written against. The consequential path always goes through
# `coerce` with the tool's real declared effect, where an unrecognisable
# result fails instead.


def _tool_reported_failure(result: Any) -> bool:
    """Whether the tool itself failed, as distinct from finding nothing."""
    return coerce(result, declared_effect=Effect.READ).is_error


def _tool_yielded_evidence(result: Any) -> bool:
    """
    Whether this tool actually returned data.

    An explicit `found: False` or `count: 0` is a successful lookup of an empty
    store — evidence of absence, not evidence. Anything else that came back
    non-empty counts.
    """
    return coerce(result, declared_effect=Effect.READ).yielded_evidence


def _assess_tool_outcomes(
    tools_used: list, tools_with_evidence: list, tools_errored: list
) -> str:
    """
    The answerability verdict for one reasoning loop.

    Mirrors `app.memory.answerability.assess`, expressed over tool outcomes
    rather than retrieved records — same three-way distinction, same reason for
    it: NO_DATA and TOOL_ERROR are both true statements and they are not
    interchangeable.
    """
    from app.memory.answerability import Answerability

    if not tools_used and not tools_with_evidence and not tools_errored:
        # Answered without looking anything up. Nothing to assert about the
        # store either way.
        #
        # The test is "no tool activity at all", deliberately not "tools_used is
        # empty". A tool that raises or times out never reaches the line that
        # records it as used — that append happens after the call returns — so
        # keying the guard on `tools_used` alone made TOOL_ERROR unreachable for
        # exactly the two failures it exists to describe, and a crashed lookup
        # reported as "nothing was looked up".
        return ""
    if tools_with_evidence and tools_errored:
        return Answerability.PARTIALLY_ANSWERABLE.value
    if tools_with_evidence:
        return Answerability.ANSWERABLE.value
    if tools_errored:
        return Answerability.TOOL_ERROR.value
    return Answerability.NO_DATA.value


# ─────────────────────────────────────────────────────────────
# What the user is told about a consequential action
# ─────────────────────────────────────────────────────────────
#
# The gateway made a confirmable tool unreachable from this loop. It did not,
# and could not, make the model stop *talking* as though it had reached one. So
# a turn that prepared an email came back as whatever prose the model chose,
# and "Done — I've sent it." was a sentence it was free to choose. The action
# was genuinely held, nothing was delivered, and the user was told the opposite.
#
# The system prompts say not to do this, in as many words, and that is the
# weakest possible form of the guarantee: an instruction the component under
# suspicion is asked to follow. What follows replaces it with an arrangement in
# which the model's account of a consequential action is simply not the thing
# that reaches the user.
#
# Two rules, and the first does almost all of the work:
#
#   * An action was held        → the answer *is* the gateway's preview. The
#                                 model's text is discarded, not inspected, so
#                                 whether it lied is not a question anyone has
#                                 to answer.
#
#   * Nothing was held, but the → the claim is false. Not doubtful: this loop
#     answer claims an act        cannot execute a confirmable tool, so a
#                                 completion inside it never happened.
#
# The second rule is narrowed to agents that actually carry a confirmable tool,
# and stands down when the gateway reported the action as already executed —
# which is the one case where "it has been sent" is true, because it was, on an
# earlier turn.


_HELD_ACTION_FOOTER = (
    'Reply "yes" to go ahead, or "no" to cancel. '
    "Nothing has been done yet."
)

_UNSUBSTANTIATED_CLAIM = (
    "I haven't done that. Anything like that I show you first and you say go — "
    "give me the details and I'll set it up."
)


def _render_held_actions(pending: list[ToolResult]) -> str:
    """
    The answer for a turn that prepared something and performed nothing.

    Built from the previews the gateway stored — the same text its content hash
    covers — so what the user reads and what an approval is bound to are one
    object. A model paraphrasing the recipient into something friendlier would
    break that correspondence, which is the other half of why its version is not
    used.
    """
    blocks: list[str] = []
    for held in pending:
        preview = (held.preview or "").strip()
        blocks.append(preview or f"Action prepared: {held.tool}")
    return "\n\n".join(blocks + [_HELD_ACTION_FOOTER])


def _offers_confirmable_tool(tools: Dict[str, Dict[str, Any]]) -> bool:
    """Whether any tool in this registry could have caused a held effect."""
    return any(
        action_gateway.requires_confirmation(effect_for_spec(spec, name))
        for name, spec in (tools or {}).items()
    )


def _reported_already_executed(results: list[ToolResult]) -> bool:
    """Whether the gateway said this exact action completed on an earlier turn.

    The one circumstance in which a completion claim is true, so the claim check
    stands down rather than contradicting the gateway.
    """
    return any(
        result.ok and dict(result.data or {}).get("already_executed")
        for result in results
    )


# ─────────────────────────────────────────────────────────────
# Pydantic models for structured LLM output validation
# ─────────────────────────────────────────────────────────────

class _ToolCallDecision(BaseModel):
    type: str        # must be "tool_call"
    tool: str
    tool_input: dict = {}


class _FinalDecision(BaseModel):
    type: str        # "final" or anything else
    content: str = ""
    is_complete: bool = True


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.groq_service = groq_service

    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    async def call_groq(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        _retries: int = _MAX_LLM_ATTEMPTS,
        **kwargs,
    ) -> str:
        """
        One logical completion, a bounded number of HTTP attempts.

        This is the only retry layer for Groq in the system, and it has to be:
        the SDK's own retry is off (`max_retries=0`), because two layers that
        cannot see each other multiply. What made the old version harmful was
        not that it retried but that it retried *blind* — a 1 s backoff applied
        equally to a 500, a connection reset and a 429, so a rate-limited
        request was re-sent into the same closed window that had just rejected
        it, three times, making the limit worse rather than waiting it out.

        So a rejection is classified before it is acted on:

        * **Rate limited (429)** — the window is closed and the provider says
          for how long. Wait that long, or don't retry at all if the wait is
          longer than the answer could still be useful for. Never sooner.
        * **Permanent (401/403/404/400)** — a bad key, a decommissioned model,
          a malformed request. No amount of retrying fixes any of these, and
          retrying spends quota to re-learn the same answer.
        * **Transient (5xx, timeout, connection)** — the case retrying was
          designed for. Short exponential backoff, and only here.

        `**kwargs` reaches `chat_completion` unchanged, which is what lets a
        caller ask for `response_format={"type": "json_object"}` — the planner
        needs it and previously had no way to pass it.

        The timeout is per attempt, applied here. Wrapping the whole retry
        sequence in one `wait_for` (which is what the reasoning loop did) meant
        the budget was consumed by earlier attempts and the last one could not
        run to completion — the work of every attempt was discarded together.
        """
        attempts = max(1, int(_retries))
        last_exc: Exception = RuntimeError("call_groq: no attempts made")

        record_llm_request(
            estimate_tokens(messages, max_tokens)
        )

        for attempt in range(1, attempts + 1):
            try:
                response = await asyncio.wait_for(
                    self.groq_service.chat_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    ),
                    timeout=_LLM_CALL_TIMEOUT,
                )
                return response["content"]

            except asyncio.TimeoutError as exc:
                last_exc = exc
                record_llm_timeout()
                wait = _transient_backoff(attempt)
                if attempt >= attempts:
                    break
                logger.warning(
                    "call_groq attempt %d/%d for agent '%s' timed out after %.0fs "
                    "— retrying in %.1fs",
                    attempt, attempts, self.name, _LLM_CALL_TIMEOUT, wait,
                )

            except Exception as exc:
                last_exc = exc
                kind = classify_llm_error(exc)

                if kind is LLMErrorKind.RATE_LIMITED:
                    record_llm_rate_limited()
                    wait = retry_after_seconds(exc)
                    if attempt >= attempts:
                        logger.error(
                            "call_groq rate limited on the final attempt (%d/%d) for "
                            "agent '%s'; provider asked for %.1fs. Not retrying.",
                            attempt, attempts, self.name,
                            wait if wait is not None else -1.0,
                        )
                        break
                    if wait is None:
                        wait = _RATE_LIMIT_DEFAULT_WAIT
                    if wait > _RATE_LIMIT_MAX_WAIT:
                        # Honouring it would outlive the caller's own deadline,
                        # and sleeping through that budget helps nobody: fail
                        # now, visibly, with the provider's own number attached.
                        logger.error(
                            "call_groq rate limited for agent '%s'; provider asked "
                            "for %.1fs, which exceeds the %.0fs ceiling. Failing "
                            "fast rather than holding the turn open.",
                            self.name, wait, _RATE_LIMIT_MAX_WAIT,
                        )
                        break
                    logger.warning(
                        "call_groq attempt %d/%d for agent '%s' was rate limited "
                        "(429); honouring Retry-After and waiting %.1fs",
                        attempt, attempts, self.name, wait,
                    )

                elif kind is LLMErrorKind.PERMANENT:
                    # Surfaced immediately and loudly. A rejected key or a
                    # decommissioned model is a deployment fault, and burning
                    # the remaining attempts on it only delays the report.
                    record_llm_failure()
                    logger.error(
                        "call_groq hit a permanent error for agent '%s' (no retry "
                        "will fix this): %s",
                        self.name, exc,
                    )
                    raise

                else:
                    record_llm_failure()
                    wait = _transient_backoff(attempt)
                    if attempt >= attempts:
                        break
                    logger.warning(
                        "call_groq attempt %d/%d failed for agent '%s': %s "
                        "— retrying in %.1fs",
                        attempt, attempts, self.name, exc, wait,
                    )

            record_llm_retry()
            await asyncio.sleep(wait)

        logger.error(
            "call_groq exhausted %d attempt(s) for agent '%s': %s",
            attempts, self.name, last_exc,
        )
        raise last_exc

    def _filter_tools_by_scope(
        self,
        tools: Dict[str, Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Drop tools the caller is not authorized to use.

        This is the authorization layer that route guards cannot provide: a
        guest and the owner reach the *same* agent through /agents/query, so
        without this a guest's conversation could still drive send_email or the
        ERP scraper. Unauthorized tools are removed from the registry entirely
        rather than refused on call, so the model is never even told they exist
        and cannot be talked into attempting them.

        A tool with no "scope" key is unrestricted (read-only helpers).
        """
        scopes = state.get("scopes")
        if scopes is None:
            # No scope context (internal/CLI invocation) — no restriction.
            return tools

        allowed: Dict[str, Dict[str, Any]] = {}
        denied: list[str] = []
        for name, spec in tools.items():
            required = spec.get("scope")
            if required is None or required in scopes:
                allowed[name] = spec
            else:
                denied.append(name)

        if denied:
            logger.info(
                "Agent '%s': withheld %d tool(s) outside caller scope: %s",
                self.name, len(denied), ", ".join(sorted(denied)),
            )
        return allowed

    def inject_memory_context(self, system_prompt: str, state: Dict[str, Any]) -> str:
        """
        Inject memory context into the system prompt.

        Fix 5: On truncation, log the original size and how many chars were dropped
        so operators can see the exact data-loss event rather than a silent cut.
        Memory sections are already priority-ordered (profile_facts first, resume last)
        in MemoryManager.format_context_for_prompt, so truncation only ever removes
        the lowest-priority tail content.
        """
        memory_prompt = state.get("memory_prompt") or ""
        original_len = len(memory_prompt)

        if original_len > _MAX_MEMORY_CHARS:
            memory_prompt = (
                memory_prompt[:_MAX_MEMORY_CHARS]
                + "\n[...memory truncated — low-priority sections omitted]"
            )
            dropped = original_len - _MAX_MEMORY_CHARS
            logger.warning(
                "Memory context truncated for agent '%s': %d→%d chars (%d dropped). "
                "Low-priority sections (resume tail) were cut. "
                "Profile facts and episodes are always safe (written first).",
                self.name, original_len, _MAX_MEMORY_CHARS, dropped,
            )

        # The clock, on every prompt. Without it a model cannot tell whether a
        # deadline has passed, cannot resolve "next Friday", and — as observed —
        # answers "what is today's date" by reporting that it has no real-time
        # access. It is one line, it is never stale, and it belongs to no
        # particular agent, so it is injected here rather than in each of them.
        try:
            from app.tools import time_tool

            clock_line = time_tool.current_context().prompt_line()
        except Exception as exc:  # a broken clock must not cost a turn
            logger.warning("Could not read the clock for prompt injection: %s", exc)
            clock_line = ""

        header = f"## Now\n{clock_line}\n\n" if clock_line else ""

        # The style contract goes on every prompt, from the one function both
        # execution paths build their prompt through — the tool-calling graph
        # and the streaming path used for voice. Applied here rather than in
        # six agents so "how the assistant speaks" has one definition; see
        # `app.agents.persona`.
        if memory_prompt:
            composed = (
                f"{header}"
                f"{_MEMORY_BLOCK_START}\n"
                f"## Memory Context\n{memory_prompt}\n\n"
                "Use this context to keep continuity with prior conversation and "
                "personal data when relevant. Do not mention memory sources explicitly "
                "unless the user asks. The date and time above come from the system "
                "clock, not from memory — use them, and never say you lack real-time "
                "access to the current date or time.\n"
                f"{_MEMORY_BLOCK_END}\n\n"
                f"{system_prompt}"
            )
        else:
            composed = f"{header}{system_prompt}" if header else system_prompt
        return _persona.apply(composed)

    async def execute_reasoning_loop(
        self,
        state: Dict[str, Any],
        base_system_prompt: str,
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Run a bounded reasoning loop with optional iterative tool usage.

        Expected model outputs (JSON):
        - Tool call: {"type":"tool_call","tool":"name","tool_input":{...}}
        - Final:     {"type":"final","content":"...","is_complete":true}

        Returns:
            {
                "final_answer": str,
                "iterations": int,
                "tools_used": [str],
                "trace": [str]
            }

        Fixes applied:
        - Fix 1: Injects inter_step_context so multi-step plans pass results forward
        - Fix 2: Injects reflect_failure_context so retries use a different strategy
        - Fix 3: Retrieves past tool insights before loop; saves successful outcomes after
        """
        # The registry as the agent declared it, before scope filtering. Used
        # only by the grounding check, which asks "could this agent have
        # satisfied the requirement" — and a tool withheld from this caller is
        # still a tool whose answer must not be invented in its place.
        declared_tools = set(tools or {})
        tools = self._filter_tools_by_scope(tools or {}, state)
        user_input = state.get("user_input", "")
        detected_intent = state.get("detected_intent", "")
        user_id = state.get("user_id", "")

        # ── Fix 3: Load past successful tool strategies from memory ──────────
        tool_hints_block = ""
        if user_id and tools:
            try:
                from app.memory.memory_manager import memory_manager as _mm
                past_insights = await _mm.get_tool_insights(
                    user_id=user_id,
                    agent_name=self.name,
                    tool_names=list(tools.keys()),
                    limit=4,
                )
                if past_insights:
                    hint_lines = ["Past successful tool approaches (from memory — reuse if relevant):"]
                    for h in past_insights:
                        hint_lines.append(
                            f"  - [{h['tool_name']}] inputs: {h['inputs_summary']} → {h['key_insight']}"
                        )
                    tool_hints_block = "\n".join(hint_lines)
            except Exception as _e:
                logger.debug("Tool memory retrieval skipped: %s", _e)

        tool_guide = "No external tools available."
        if tools:
            tool_lines = [
                f"- {name}: {info.get('description', '')}"
                for name, info in tools.items()
            ]
            tool_guide = "Available tools:\n" + "\n".join(tool_lines)
            if tool_hints_block:
                tool_guide = tool_guide + "\n\n" + tool_hints_block

        # ── Fix 2: Inject failure context from previous failed attempt ───────
        failure_injection = ""
        reflect_failure_context = state.get("reflect_failure_context") or ""
        if reflect_failure_context:
            failure_injection = (
                f"\n\nCRITICAL — PREVIOUS ATTEMPT FAILED:\n{reflect_failure_context}\n"
                "You MUST use a different strategy than before."
            )

        loop_instructions = f"""
You are in a reasoning loop with a strict maximum of {max_iterations} iterations.

{tool_guide}
{failure_injection}
CRITICAL OUTPUT RULE: Respond with ONLY a single raw JSON object. No explanation text, no markdown, no code fences, no preamble. The very first character of your response must be `{{` and the last must be `}}`.

Use ONE of these two shapes:

1) To call a tool:
{{"type":"tool_call","tool":"tool_name","tool_input":{{"key":"value"}}}}

2) To give the final answer:
{{"type":"final","content":"your complete response to the user","is_complete":true}}

Rules:
- Call a tool when you need real data (jobs, emails, schedules).
- After receiving tool observations, use them to write the final answer.
- Stop as soon as the answer is complete.
- Never exceed {max_iterations} iterations.
- Output ONLY JSON — no surrounding text whatsoever.
- If a tool above could answer this question and you have NOT yet received
  any tool observations for it, you MUST emit a tool_call now — never emit
  type=final on the first turn when a listed tool could answer the question.
  Guessing or inventing an answer instead of calling the tool is a hard
  failure, not a shortcut.
"""

        system_prompt = self.inject_memory_context(
            f"{base_system_prompt}\n\n{loop_instructions}",
            state,
        )

        observations: list[str] = []
        trace: list[str] = []
        tools_used: list[str] = []
        # Which tools actually produced data, and which failed. Kept apart on
        # purpose: a tool that returned nothing and a tool that raised look
        # identical downstream, and they license opposite statements. See
        # `app.memory.answerability`.
        tools_with_evidence: list[str] = []
        tools_errored: list[str] = []
        # Every call this loop made, typed. Returned so callers can reach the
        # effect, preview and idempotency key of what ran — the raw material a
        # confirmation gate will need, kept here rather than re-derived later.
        tool_results: list[ToolResult] = []
        # Actions held for the user's approval. Surfaced separately so a caller
        # can render the preview without scanning every result.
        pending_actions: list[ToolResult] = []
        final_answer = ""

        # Build recent conversation history for multi-turn context.
        #
        # Read through `persona.recent_turns` rather than off one field: the
        # web chat sends no `conversation_history`, so this list used to be
        # empty on every typed turn and each one was answered as if it were
        # the first. See that function for why there are two sources.
        raw_history = _persona.recent_turns(state)
        history_messages: list[dict] = [
            {"role": turn["role"], "content": str(turn.get("content", ""))[:400]}
            for turn in raw_history[-6:]
            if isinstance(turn, dict)
            and turn.get("role") in ("user", "assistant")
            and turn.get("content")
        ]

        # ── Fix 1: Prepend inter-step context to user query ──────────────────
        inter_step_context = state.get("inter_step_context") or ""
        effective_query = user_input
        if inter_step_context:
            effective_query = f"{inter_step_context}\n\nCurrent task: {user_input}"

        # Set when the loop stops because the provider is rate limiting us, so
        # the envelope can carry that fact to `reflect_node` — which must not
        # answer a closed rate-limit window by re-running the whole specialist.
        rate_limited = False

        # The same question for timeouts, which need counting rather than a
        # flag. A single slow call that the next iteration recovers from is
        # ordinary; a loop where *every* call timed out and none returned is a
        # provider that is not answering, and re-running the whole specialist
        # against it is the same amplification a rate limit causes — a timeout
        # is what queueing looks like from this side.
        llm_timeouts = 0
        llm_responses = 0

        # The prompt sent from the second iteration onward. The memory block is
        # unchanged between steps and the model has already read it, so
        # re-transmitting it every step multiplied the largest part of the
        # prompt by the iteration count for no added information.
        followup_system_prompt = _without_memory_block(system_prompt)

        for step in range(1, max_iterations + 1):
            observation_block = "\n".join(observations[-6:]) if observations else "(none)"

            messages = [
                {
                    "role": "system",
                    "content": system_prompt if step == 1 else followup_system_prompt,
                },
                *history_messages,
                {
                    "role": "user",
                    "content": (
                        f"Iteration: {step}/{max_iterations}\n"
                        f"Intent: {detected_intent}\n"
                        f"Query: {effective_query}\n"
                        f"Tool observations so far:\n{observation_block}"
                    ),
                },
            ]

            try:
                # No `wait_for` here: `call_groq` applies `_LLM_CALL_TIMEOUT` to
                # each HTTP attempt. Wrapping the retry sequence as well meant
                # the outer budget expired mid-sequence and discarded the work
                # of every attempt inside it.
                raw_response = await self.call_groq(
                    messages=messages, temperature=0.3, max_tokens=700
                )
            except asyncio.TimeoutError:
                llm_timeouts += 1
                logger.warning(
                    "LLM call timed out after %.0fs on step %d for agent '%s'",
                    _LLM_CALL_TIMEOUT, step, self.name,
                )
                observations.append(f"Step {step}: LLM call timed out.")
                trace.append(f"step_{step}: llm_timeout")
                continue
            except Exception as exc:
                # A rate limit ends the loop rather than costing another
                # iteration. Continuing would send the next step's request into
                # the same closed window, earning another rejection and making
                # the limit worse — the loop would be a cause of the outage it
                # is reacting to.
                #
                # Only a rate limit. Every other exception is re-raised, which
                # is what it did before: the agent's `execute` catches it and
                # stamps a *failed* envelope. Swallowing them here would turn a
                # provider outage into a "success" carrying the fallback
                # sentence, and `reflect_node` would never learn the turn had
                # failed at all.
                if classify_llm_error(exc) is LLMErrorKind.RATE_LIMITED:
                    rate_limited = True
                    logger.error(
                        "Agent '%s' stopped at step %d: provider is rate limiting. "
                        "Not spending the remaining %d iteration(s) on a closed window.",
                        self.name, step, max_iterations - step,
                    )
                    observations.append(f"Step {step}: rate limited by the model provider.")
                    trace.append(f"step_{step}: rate_limited")
                    break
                raise

            llm_responses += 1
            decision = self._parse_reasoning_decision(raw_response)

            if decision["type"] == "tool_call":
                tool_name = decision.get("tool", "")
                tool_input = decision.get("tool_input", {})

                tool_info = tools.get(tool_name)
                if not tool_info:
                    # Recorded as a typed failure as well as an observation: an
                    # out-of-scope tool is withheld from the registry entirely,
                    # so "unknown" here can mean "not permitted", and that must
                    # not read downstream as a lookup that found nothing.
                    unknown = ToolResult.failed(
                        f"unknown or unavailable tool '{tool_name}'",
                        kind=ErrorKind.UNKNOWN_TOOL,
                        tool=tool_name,
                    )
                    tool_results.append(unknown)
                    observations.append(f"Tool error: Unknown tool '{tool_name}'.")
                    trace.append(f"step_{step}: unknown_tool:{tool_name}")
                    continue

                tool_callable: Callable[..., Awaitable[Any]] = tool_info["callable"]
                declared_effect = effect_for_spec(tool_info, tool_name)

                # ── The gate ──────────────────────────────────────────────────
                # Placed before the try block that calls the tool, because that
                # is the whole point: a confirmable action must not be reachable
                # from this loop at all. `intercept` builds a preview and holds
                # the action; it never awaits `tool_callable`. The only code
                # that does is ActionGateway.confirm_and_execute, and that runs
                # after a valid token is presented.
                if action_gateway.requires_confirmation(declared_effect):
                    gated = await action_gateway.intercept(
                        tool=tool_name,
                        spec=tool_info,
                        arguments=tool_input,
                        owner_id=user_id,
                        conversation_id=state.get("session_id") or "",
                        effect=declared_effect,
                    )
                    tool_results.append(gated)
                    if gated.is_pending:
                        pending_actions.append(gated)
                    observations.append(
                        f"Tool {tool_name} observation: {gated.observation()}"
                    )
                    trace.append(f"step_{step}: gated:{tool_name}:{gated.status.value}")
                    if gated.is_error:
                        tools_errored.append(tool_name)
                    elif gated.ok:
                        # The already-executed notice. The action genuinely
                        # happened, earlier — so it counts as a call that ran.
                        tools_used.append(tool_name)
                        tools_with_evidence.append(tool_name)
                    # Deliberately not counted as used when pending: nothing ran.
                    continue

                try:
                    result = await asyncio.wait_for(
                        tool_callable(tool_input),
                        timeout=_TOOL_CALL_TIMEOUT,
                    )

                    # The single place a raw return value becomes typed. An
                    # unrecognisable result from a consequential tool becomes an
                    # error here rather than an optimistic success — see
                    # app/tools/contract.py.
                    tool_result = coerce(
                        result,
                        tool=tool_name,
                        declared_effect=declared_effect,
                        tool_input=tool_input,
                    )
                    tool_results.append(tool_result)

                    # Legacy dict results keep their exact previous observation
                    # text, so migrating a tool to the contract is what changes
                    # what the model sees — never this change on its own.
                    summarized = (
                        self._summarize_tool_result(result)
                        if tool_result.adapted
                        else tool_result.observation()
                    )
                    observations.append(f"Tool {tool_name} observation: {summarized}")
                    tools_used.append(tool_name)
                    if tool_result.is_error:
                        tools_errored.append(tool_name)
                    elif tool_result.yielded_evidence:
                        tools_with_evidence.append(tool_name)
                    trace.append(f"step_{step}: tool_call:{tool_name}")

                    # ── Fix 3: Save successful tool outcome to memory ─────────
                    # Gated on an actual success. This block records the call as
                    # `outcome_quality="good"` and replays it to later turns as
                    # a strategy worth reusing — which, for a call that errored
                    # or found nothing, teaches the agent to repeat a approach
                    # that did not work. Before the contract there was no
                    # reliable way to tell here; now there is.
                    if user_id and tool_result.ok:
                        try:
                            from app.memory.memory_manager import memory_manager as _mm
                            inputs_summary = json.dumps(tool_input, default=str)[:300]
                            key_insight = summarized[:300]
                            _spawn_background(
                                _mm.save_tool_outcome(
                                    user_id=user_id,
                                    agent_name=self.name,
                                    tool_name=tool_name,
                                    inputs_summary=inputs_summary,
                                    outcome_quality="good",
                                    key_insight=key_insight,
                                ),
                                f"save-tool-outcome-{self.name}-{tool_name}",
                            )
                        except Exception as _e:
                            logger.debug("Tool memory save skipped: %s", _e)

                except asyncio.TimeoutError:
                    # Retryable, and deliberately so: a timeout says nothing
                    # about whether the call will succeed next time. It also
                    # says nothing about whether the effect already happened,
                    # which is why the key is carried on the result.
                    tool_results.append(ToolResult.failed(
                        f"timed out after {_TOOL_CALL_TIMEOUT:.0f}s",
                        kind=ErrorKind.TIMEOUT,
                        effect=declared_effect,
                        retryable=True,
                        tool=tool_name,
                    ))
                    observations.append(f"Tool {tool_name} timed out after {_TOOL_CALL_TIMEOUT:.0f}s.")
                    tools_errored.append(tool_name)
                    trace.append(f"step_{step}: tool_timeout:{tool_name}")
                    logger.warning(
                        "Tool '%s' timed out after %.0fs in agent '%s'",
                        tool_name, _TOOL_CALL_TIMEOUT, self.name,
                    )
                except Exception as e:
                    tool_results.append(ToolResult.failed(
                        str(e),
                        kind=ErrorKind.EXCEPTION,
                        effect=declared_effect,
                        tool=tool_name,
                    ))
                    observations.append(f"Tool {tool_name} failed: {str(e)}")
                    tools_errored.append(tool_name)
                    trace.append(f"step_{step}: tool_error:{tool_name}")
                    logger.warning(
                        "Tool '%s' raised an exception in agent '%s': %s",
                        tool_name, self.name, e,
                    )

                continue

            # Final answer (or raw fallback)
            final_answer = (decision.get("content") or raw_response or "").strip()
            trace.append(f"step_{step}: final")
            if final_answer:
                break

        if not final_answer:
            final_answer = (
                "I could not complete your request confidently. "
                "Please try again with more detail."
            )

        # ── The model's account of a consequential action is not delivered ────
        # See the note at the top of this module. `answer_source` is returned so
        # callers and tests can assert *why* an answer reads the way it does
        # without matching on its words.
        # ── A personal fact must come from a lookup ──────────────────────────
        #
        # Ordered after the held-action substitution and before the claim
        # check, because the three answer different questions and only one can
        # apply. A turn that prepared an action is not a retrieval turn; a
        # retrieval turn that never called its tool has nothing to claim.
        #
        # `required_tools` was decided deterministically at the routing edge and
        # travels on the state — the model is not consulted about what this
        # question needed, only about how to phrase the answer once it has been
        # looked up. See `app.agents.grounding`.
        grounded_answer, grounding_verdict = _grounding.enforce(
            state.get("query_category"),
            final_answer,
            tools_used=tools_used,
            tools_with_evidence=tools_with_evidence,
            tools_errored=tools_errored,
            available_tools=declared_tools,
        )

        answer_source = "model"
        claimable = (
            _offers_confirmable_tool(tools)
            and not _reported_already_executed(tool_results)
        )
        if pending_actions:
            final_answer = _render_held_actions(pending_actions)
            answer_source = "gateway_preview"
        elif grounded_answer is not final_answer:
            logger.error(
                "Agent '%s' answered a %s question with grounding=%s "
                "(used=%s errored=%s); the model's answer was not delivered.",
                self.name, state.get("query_category"), grounding_verdict.value,
                tools_used, tools_errored,
            )
            final_answer = grounded_answer
            answer_source = f"grounding_{grounding_verdict.value}"
        elif claimable:
            # Imported here rather than at module scope: `confirmable_tools`
            # reaches the email sender and the memory manager, both of which
            # sit above this module.
            from app.agents.confirmable_tools import claims_consequential_completion

            fabricated = claims_consequential_completion(final_answer)
            if fabricated:
                logger.error(
                    "Agent '%s' claimed a %s completion with no execution behind "
                    "it; the claim was not delivered.", self.name, fabricated,
                )
                final_answer = _UNSUBSTANTIATED_CLAIM
                answer_source = "claim_suppressed"

        # Every agent's `execute` reads this off the loop; writing it here as
        # well means an agent that forgets to copy it still cannot deliver a
        # turn whose held actions are invisible to the layers above.
        state["pending_actions"] = pending_actions

        # Same reasoning, for the same reason. `reflect_node` decides whether a
        # turn is worth retrying, and a turn whose required tool was never
        # called is the most retryable failure there is — the tool exists, the
        # data is there, the model simply did not look. Written here rather
        # than left to each agent so a specialist that forgets to copy it
        # cannot silently cost the user a recoverable turn.
        state["grounding"] = grounding_verdict.value

        # Same again, for `reflect_node`. A turn that ended because the provider
        # is rate limiting is the one failure that must never be retried: the
        # retry lands in the same closed window, and re-running a specialist is
        # the single most expensive thing this system can do in response to
        # being told it is doing too much.
        state["rate_limited"] = rate_limited

        # Every LLM call timed out and none returned. Distinct from a rate
        # limit (the provider answered, refusing) and from an ordinary poor
        # answer (the model returned something), and it is its own entry in the
        # outcome vocabulary rather than being folded into either.
        llm_unreachable = llm_timeouts > 0 and llm_responses == 0
        state["llm_unreachable"] = llm_unreachable
        if llm_unreachable:
            logger.error(
                "Agent '%s': all %d LLM attempt(s) timed out and none returned. "
                "Reporting the turn as unanswerable rather than retrying into a "
                "provider that is not responding.",
                self.name, llm_timeouts,
            )

        return {
            "final_answer": final_answer,
            "answer_source": answer_source,
            "grounding": grounding_verdict.value,
            "rate_limited": rate_limited,
            "llm_unreachable": llm_unreachable,
            "iterations": len(trace),
            "tools_used": list(dict.fromkeys(tools_used)),   # deduplicated, order preserved
            "trace": trace,
            # Deterministic verdict on what the tools actually produced. The
            # honesty rules in the system prompt say the right thing; this is
            # the same claim derived from observed tool results rather than
            # from the model's willingness to follow an instruction.
            "answerability": _assess_tool_outcomes(
                tools_used, tools_with_evidence, tools_errored
            ),
            "tools_with_evidence": list(dict.fromkeys(tools_with_evidence)),
            "tools_errored": list(dict.fromkeys(tools_errored)),
            "tool_results": tool_results,
            "pending_actions": pending_actions,
        }

    def _parse_reasoning_decision(self, response: str) -> Dict[str, Any]:
        """
        Parse model decision — tool call vs final answer.

        Two strategies (in order):
        1. Parse the raw response directly as JSON.
        2. Extract the first {...} substring (handles accidental preamble or
           markdown wrapping that the model emits despite instructions).

        When a candidate parses as JSON it is validated with Pydantic.
        If neither strategy yields valid JSON the full response is returned as
        a final answer and a warning is logged so LLM misbehaviour is visible.
        """
        text = (response or "").strip()
        if not text:
            return {"type": "final", "content": ""}

        candidates = [text]
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match and brace_match.group(0) != text:
            candidates.append(brace_match.group(0))

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue

            decision_type = str(payload.get("type", "")).strip().lower()
            # Accept alternate key spellings the model sometimes emits
            tool_name = str(
                payload.get("tool") or payload.get("tool_name") or ""
            ).strip()

            if decision_type == "tool_call" and tool_name:
                try:
                    validated = _ToolCallDecision(
                        type="tool_call",
                        tool=tool_name,
                        tool_input=payload.get("tool_input")
                            or payload.get("parameters")
                            or {},
                    )
                    return validated.model_dump()
                except ValidationError as exc:
                    logger.warning(
                        "Tool-call Pydantic validation failed for agent '%s': %s",
                        self.name, exc,
                    )
                    continue

            content = str(
                payload.get("content")
                or payload.get("answer")
                or payload.get("final_answer")
                or ""
            )
            return {"type": "final", "content": content or text}

        # Neither strategy produced valid JSON — warn so it shows in logs
        logger.warning(
            "Agent '%s' received non-JSON LLM output (first 120 chars): %.120s",
            self.name, text,
        )
        return {"type": "final", "content": text}

    def _compute_confidence(
        self,
        final_answer: str,
        tools_used: list,
        iterations: int,
        max_iterations: int,
        was_retry: bool = False,
    ) -> float:
        """
        Unified semantic confidence scoring matrix (Fix 4).
        Replaces every agent's arbitrary word-count threshold with a single
        principled function that all agents call identically.

        Signals:
          Base score      0.50  — we produced some answer
          Tool used      +0.25  — agent retrieved real data, not just reasoning
          Substantial    +0.10  — answer has >30 words (actually covers the question)
          Detailed       +0.05  — answer has >80 words (thorough coverage)
          Max-iter pen   -0.10  — used all available iterations (struggled to complete)
          Retry penalty  -0.10  — this was a retry after a prior failure (reflect loop)

        Scale: 0.0 (no answer) → 0.95 (max; never 1.0 — perfect certainty is never warranted).
        """
        if not final_answer:
            return 0.0

        score = 0.50

        if tools_used:
            score += 0.25

        word_count = len(final_answer.split())
        if word_count > 30:
            score += 0.10
        if word_count > 80:
            score += 0.05

        if iterations >= max_iterations:
            score -= 0.10

        if was_retry:
            score -= 0.10

        return round(max(0.0, min(0.95, score)), 2)

    def _summarize_tool_result(self, result: Any, max_chars: int = 1200) -> str:
        """Compact tool result for loop observations while keeping salient signal."""
        try:
            payload = result

            if isinstance(result, dict):
                payload = dict(result)

                if isinstance(payload.get("results"), list):
                    trimmed = []
                    for item in payload["results"][:3]:
                        if isinstance(item, dict):
                            trimmed.append(
                                {
                                    "title": item.get("title"),
                                    "url": item.get("url"),
                                    "snippet": item.get("snippet") or item.get("content"),
                                    "rank_score": item.get("rank_score") or item.get("score"),
                                }
                            )
                        else:
                            trimmed.append(item)
                    payload["results"] = trimmed

            text = json.dumps(payload, ensure_ascii=True, default=str)
        except Exception:
            text = str(result)

        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text
