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

from app.services.groq_service import groq_service

logger = logging.getLogger(__name__)

# Memory injected into every agent prompt is capped at this many characters
# (~5 000 tokens at 4 chars/token) to prevent overflowing Groq's context window.
_MAX_MEMORY_CHARS = 20_000

# Per-operation timeouts for the reasoning loop.
# Without these the loop hangs indefinitely if Groq or a tool is unresponsive.
_LLM_CALL_TIMEOUT = 30.0   # seconds per Groq call
_TOOL_CALL_TIMEOUT = 15.0  # seconds per tool call

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
        _retries: int = 3,
    ) -> str:
        """
        Call Groq with exponential-backoff retry (1 s → 2 s between attempts).
        Raises the last exception if all retries are exhausted.
        """
        last_exc: Exception = RuntimeError("call_groq: no attempts made")
        for attempt in range(1, _retries + 1):
            try:
                response = await self.groq_service.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response["content"]
            except Exception as exc:
                last_exc = exc
                if attempt < _retries:
                    wait = 2 ** (attempt - 1)   # 1 s, 2 s
                    logger.warning(
                        "call_groq attempt %d/%d failed for agent '%s': %s — retrying in %ds",
                        attempt, _retries, self.name, exc, wait,
                    )
                    await asyncio.sleep(wait)
        logger.error(
            "call_groq exhausted %d retries for agent '%s': %s",
            _retries, self.name, last_exc,
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

        if memory_prompt:
            return (
                f"## Memory Context\n{memory_prompt}\n\n"
                "Use this context to keep continuity with prior conversation and "
                "personal data when relevant. Do not mention memory sources explicitly "
                "unless the user asks.\n\n"
                f"{system_prompt}"
            )
        return system_prompt

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
"""

        system_prompt = self.inject_memory_context(
            f"{base_system_prompt}\n\n{loop_instructions}",
            state,
        )

        observations: list[str] = []
        trace: list[str] = []
        tools_used: list[str] = []
        final_answer = ""

        # Build recent conversation history for multi-turn context.
        raw_history = state.get("conversation_history") or []
        history_messages: list[dict] = [
            {"role": turn["role"], "content": str(turn.get("content", ""))[:400]}
            for turn in raw_history[-6:]
            if turn.get("role") in ("user", "assistant") and turn.get("content")
        ]

        # ── Fix 1: Prepend inter-step context to user query ──────────────────
        inter_step_context = state.get("inter_step_context") or ""
        effective_query = user_input
        if inter_step_context:
            effective_query = f"{inter_step_context}\n\nCurrent task: {user_input}"

        for step in range(1, max_iterations + 1):
            observation_block = "\n".join(observations[-6:]) if observations else "(none)"

            messages = [
                {"role": "system", "content": system_prompt},
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
                raw_response = await asyncio.wait_for(
                    self.call_groq(messages=messages, temperature=0.3, max_tokens=700),
                    timeout=_LLM_CALL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "LLM call timed out after %.0fs on step %d for agent '%s'",
                    _LLM_CALL_TIMEOUT, step, self.name,
                )
                observations.append(f"Step {step}: LLM call timed out.")
                trace.append(f"step_{step}: llm_timeout")
                continue

            decision = self._parse_reasoning_decision(raw_response)

            if decision["type"] == "tool_call":
                tool_name = decision.get("tool", "")
                tool_input = decision.get("tool_input", {})

                tool_info = tools.get(tool_name)
                if not tool_info:
                    observations.append(f"Tool error: Unknown tool '{tool_name}'.")
                    trace.append(f"step_{step}: unknown_tool:{tool_name}")
                    continue

                tool_callable: Callable[..., Awaitable[Any]] = tool_info["callable"]

                try:
                    result = await asyncio.wait_for(
                        tool_callable(tool_input),
                        timeout=_TOOL_CALL_TIMEOUT,
                    )
                    summarized = self._summarize_tool_result(result)
                    observations.append(f"Tool {tool_name} observation: {summarized}")
                    tools_used.append(tool_name)
                    trace.append(f"step_{step}: tool_call:{tool_name}")

                    # ── Fix 3: Save successful tool outcome to memory ─────────
                    if user_id:
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
                    observations.append(f"Tool {tool_name} timed out after {_TOOL_CALL_TIMEOUT:.0f}s.")
                    trace.append(f"step_{step}: tool_timeout:{tool_name}")
                    logger.warning(
                        "Tool '%s' timed out after %.0fs in agent '%s'",
                        tool_name, _TOOL_CALL_TIMEOUT, self.name,
                    )
                except Exception as e:
                    observations.append(f"Tool {tool_name} failed: {str(e)}")
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

        return {
            "final_answer": final_answer,
            "iterations": len(trace),
            "tools_used": list(dict.fromkeys(tools_used)),   # deduplicated, order preserved
            "trace": trace,
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
