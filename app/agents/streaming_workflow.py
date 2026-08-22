"""
Streaming workflow for low-latency voice responses.
Streams LLM tokens as they arrive instead of waiting for full response.

Fix 7: This module is now wired to the /stream WebSocket route in agent_routes.py.
The workflow skips tool-calling (trades completeness for sub-second first-token latency)
which is the right trade-off for voice/streaming UI contexts.
"""
import asyncio
import uuid
import logging
from typing import Dict, Any, AsyncGenerator, FrozenSet, Optional
from app.agents import confirmation, query_intent
from app.agents.agent_profiles import get_capabilities
from app.agents.confirmable_tools import claims_completion_in_stream
from app.agents.workflow import decide_route, parallel_init_node, run_workflow
from app.agents.job_agent import job_agent
from app.agents.email_agent import email_agent
from app.agents.academic_agent import academic_agent
from app.agents.profile_agent import profile_agent
from app.services.groq_service import groq_service

logger = logging.getLogger(__name__)

# Ceiling on memory retrieval + planning. Past this the assistant answers from
# the transcript alone, which is far better than saying nothing at all.
_INIT_TIMEOUT_SECONDS = 4.0


def get_agent_by_name(agent_name: str):
    """
    Helper function to get agent instance by name.

    Args:
        agent_name: Name of the agent (job, email, academic, profile)

    Returns:
        Agent instance or None if not found
    """
    agents = {
        "job": job_agent,
        "email": email_agent,
        "academic": academic_agent,
        "profile": profile_agent
    }
    return agents.get(agent_name)


def _get_agent_system_prompt(agent, state: Dict[str, Any]) -> str:
    """
    Build the system prompt for an agent in the streaming (tool-free) path.

    Capability text comes from the shared agent profile registry so this path
    and the tool-calling workflow always describe the same agent the same way.

    Args:
        agent: Agent instance
        state: Workflow state with memory

    Returns:
        System prompt with memory context injected
    """
    base_prompt = get_capabilities(agent.name, default=agent.description)
    return agent.inject_memory_context(base_prompt, state)


_NO_IDENTITY = (
    "I don't know who you are on this connection yet, so I can't check that "
    "against your profile."
)

# What is said when the escalation itself fails. Never a model's words: the
# state in which the tools are unreachable is precisely the state in which a
# model asked to cover for them invents the answer. The job-match wording is
# kept verbatim because a comparison that did not run is a different thing to
# report than a lookup that did not run.
_ESCALATION_FAILED = "Couldn't look that up just now. Try me again in a second."
_ESCALATION_FAILED_BY_CATEGORY = {
    "JOB_MATCH": "Couldn't finish that comparison. Try me again in a second.",
}

# The tool workflow returned, but with nothing in it. Saying nothing at all is
# not an option on a spoken turn, and letting the tool-free path have a second
# go would be the silent fallback this module exists to remove.
_ESCALATION_EMPTY = (
    "I looked that up but didn't get an answer back. Please try again."
)


def _prefetched_from(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The initialization results this turn already has, packaged for handover.

    Only the fields `parallel_init_node` would otherwise recompute, and only
    when retrieval actually produced something: `memory_prompt` is the key
    `_adopt_prefetched` requires, because an absent one is indistinguishable
    from a failed retrieval and adopting that would answer a personal question
    with no memory and no sign that anything went wrong. When init timed out or
    failed, this returns a dict without it and the tool workflow initializes
    normally — the slower path, correctly chosen.
    """
    handover: Dict[str, Any] = {
        "query_category": state.get("query_category"),
        "selected_agent": state.get("selected_agent"),
        "detected_intent": state.get("detected_intent"),
        "planner_confidence": state.get("planner_confidence"),
        "execution_plan": state.get("execution_plan") or [],
    }
    memory_prompt = state.get("memory_prompt")
    if memory_prompt:
        handover["memory_prompt"] = memory_prompt
        handover["memory_context"] = state.get("memory_context") or {}
    return handover


async def _escalate_to_tools(
    *,
    category: str,
    reason: str,
    user_input: str,
    user_id: str,
    session_id: str,
    conversation_history: list,
    output_mode: str,
    scopes: Optional[FrozenSet[str]],
    memory_owner_id: Optional[str],
    memory_visibilities: Optional[list],
    request_id: str,
    skip_user_ingest: bool = False,
    prefetched: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Hand a turn that cannot be answered here to the tool-calling workflow.

    Emits the same `metadata` / `complete` events a streamed turn would, so
    every consumer — the LiveKit worker's chunker, the `/agents/stream`
    WebSocket, the caption pump — is unaffected. What changes is only that the
    text was produced by tools rather than by a language model.

    Deliberately yields no `token` events. The tool workflow has no token
    stream, and `speech_token_stream` already speaks a `complete` that never
    streamed, so the reply is heard exactly as any other non-streaming answer.

    `reason` is which of the two rules fired — the category is unanswerable
    without tools at all, or it is answerable but retrieval did not deliver.
    It is carried into the metadata rather than inferred later, because the two
    look identical from outside and license different follow-up questions.

    `skip_user_ingest` suppresses re-storing the user's message. It is set on
    the second rule only, where `parallel_init_node` has already run and
    already stored it; ingesting twice would put the same utterance into the
    thread twice and hand the next turn a duplicated transcript.

    Without a caller identity there is nothing to look anything up against, and
    the honest answer is to say so — falling through to an ungrounded reply is
    the failure this whole function exists to prevent.
    """
    agent_hint = query_intent.escalation_agent_for(category, text=user_input)

    if not user_id:
        logger.warning(
            "Refusing a %s turn with no caller identity (request=%s)",
            category, request_id,
        )
        yield {
            "type": "metadata",
            "selected_agent": agent_hint,
            "detected_intent": category,
            "execution_path": [],
            "route": "tool_required",
            "escalation_reason": reason,
        }
        yield {
            "type": "complete",
            "display_text": _NO_IDENTITY,
            "speech_text": _NO_IDENTITY,
            "agent": agent_hint,
            "success": False,
            "error": "missing_user_id",
            "route": "tool_required",
        }
        return

    logger.info(
        "Escalating a %s turn to the tool workflow (reason=%s, request=%s)",
        category, reason, request_id,
    )

    def _failure(message: str, error: str):
        return (
            {
                "type": "metadata",
                "selected_agent": agent_hint,
                "detected_intent": category,
                "execution_path": [],
                "route": "tool_required",
                "escalation_reason": reason,
                "tools_used": [],
            },
            {
                "type": "complete",
                "display_text": message,
                "speech_text": message,
                "agent": agent_hint,
                "user_id": user_id,
                "session_id": session_id,
                "success": False,
                "error": error,
                "route": "tool_required",
            },
        )

    try:
        result = await run_workflow(
            user_input=user_input,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history,
            output_mode=output_mode,
            scopes=scopes,
            memory_owner_id=memory_owner_id,
            memory_visibilities=memory_visibilities,
            skip_user_ingest=skip_user_ingest,
            # Initialization this path has already paid for, handed over rather
            # than repeated. `skip_user_ingest` suppressed the duplicate *write*
            # but not the duplicate planner call or the duplicate retrieval
            # fan-out — so an escalated turn ran `parallel_init_node` twice and
            # paid for two full inits to answer one question.
            #
            # None when the escalation happened before init ran (the
            # unconditional rules, which fire on the category alone), and
            # `parallel_init_node` then initializes normally. See
            # `_adopt_prefetched`.
            prefetched=prefetched,
        )
    except Exception as exc:
        logger.error("Escalated %s turn failed: %s", category, exc)
        for event in _failure(
            _ESCALATION_FAILED_BY_CATEGORY.get(category, _ESCALATION_FAILED),
            str(exc),
        ):
            yield event
        return

    envelope = result.get("task_result") or {}
    display_text = result.get("display_text") or ""
    speech_text = result.get("speech_text") or display_text

    if not display_text.strip():
        # An empty envelope is a failure wearing a success's clothes. Reporting
        # it as one keeps the caller from treating silence as an answer.
        logger.error("Escalated %s turn produced no text (request=%s)",
                     category, request_id)
        for event in _failure(_ESCALATION_EMPTY, "empty_tool_result"):
            yield event
        return

    yield {
        "type": "metadata",
        # The agent that actually ran, not the planner's guess at one.
        "selected_agent": envelope.get("agent") or result.get("selected_agent")
                          or agent_hint,
        "detected_intent": result.get("query_category") or category,
        "execution_path": result.get("execution_path", []),
        "planner_confidence": result.get("planner_confidence"),
        "route": "tool_required",
        "escalation_reason": reason,
        "tools_used": list(envelope.get("evidence") or []),
        # What the tools actually observed — ANSWERABLE, NO_DATA, TOOL_ERROR.
        # Carried out so a client can tell "nothing on file" from "couldn't
        # read it", which is the distinction the whole escalation protects.
        "answerability": result.get("answerability") or "",
    }
    yield {
        "type": "complete",
        "display_text": display_text,
        "speech_text": speech_text,
        "agent": envelope.get("agent") or agent_hint,
        "user_id": user_id,
        "session_id": session_id,
        "success": result.get("error") is None,
        "route": "tool_required",
    }


async def run_streaming_workflow(
    user_input: str,
    user_id: str = None,
    session_id: str = None,
    conversation_history: list = None,
    output_mode: str = "user",
    scopes: Optional[FrozenSet[str]] = None,
    memory_owner_id: Optional[str] = None,
    memory_visibilities: Optional[list] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run workflow with true LLM token streaming for low-latency voice/UI.

    Fix 7: This function is now the backend for the /stream WebSocket route.
    It trades tool-calling completeness for sub-second first-token latency,
    which is the correct trade-off for streaming voice interactions.

    Yields:
        {"type": "metadata", "selected_agent": ..., "detected_intent": ..., ...}
        {"type": "token",    "token": "...", "accumulated": "..."}   ← live tokens
        {"type": "complete", "display_text": ..., "speech_text": ..., "success": bool}
    """
    if not session_id:
        session_id = f"session_{(user_id or 'anon').strip().lower()}"

    # Full initial state — all AgentState fields initialised so memory_node
    # and planner_node can read/write without KeyError or missing-key warnings.
    state: Dict[str, Any] = {
        "user_input": user_input,
        "user_id": (user_id or "").strip().lower(),
        "session_id": session_id,
        "conversation_history": conversation_history or [],
        "output_mode": output_mode,
        "scopes": scopes,
        "memory_owner_id": memory_owner_id,
        "memory_visibilities": memory_visibilities,
        # This path is the first to see the utterance, so it ingests normally.
        "skip_user_ingest": False,
        "memory_context": None,
        "memory_prompt": None,
        "detected_intent": None,
        "selected_agent": None,
        "planner_confidence": None,
        "needs_clarification": None,
        "clarification_question": None,
        "clarification_reason": None,
        "query_category": None,
        "memory_sources": None,
        "profile_intent": None,
        "followup_subject": None,
        "route": None,
        "required_tools": [],
        "schedule_intent": None,
        "task_result": None,
        "agent_reasoning": None,
        "iteration_count": 0,
        "reflect_outcome": None,
        "execution_plan": [],
        "current_step_index": 0,
        "step_results": {},
        "inter_step_context": None,
        "reflect_failure_context": None,
        "display_text": None,
        "speech_text": None,
        "current_agent": None,
        "execution_path": [],
        "request_id": uuid.uuid4().hex,
        "error": None,
    }

    # ── Step 0: Escalate the categories that cannot be answered here ─────────
    #
    # This workflow has no tools. For most questions that is a latency trade —
    # the reply is composed from retrieved memory and is honest about what it
    # does not know. For a few categories it is not a trade at all: answering
    # them from a prompt produces a *false statement*, because the true answer
    # is a computation the model cannot perform and will approximate instead.
    #
    # JOB_MATCH is the case that forced this. Answered here, a spoken "do I
    # match this job?" becomes a model reading a résumé blob and deciding for
    # itself how well the user matches — the unevidenced claim `app.matching`
    # exists to make unrepresentable, arriving through the one door that
    # bypassed it.
    #
    # The check is deliberately placed *before* `parallel_init_node` rather than
    # after `decide_route`, for two reasons. Classification is pure — it reads
    # the utterance and the turns already in hand, touching no store — so this
    # costs microseconds and no round trip. And escalating after init would run
    # memory retrieval twice, storing the user's message twice with it.
    #
    # Why here rather than in `hybrid_router`: that router is one of two ways in.
    # `/agents/stream` calls this function directly and never consults it, so a
    # fix there would leave a second door open. This is the door they share.
    #
    # ACTION_REQUEST is the third rule and the one with an effect attached. A
    # send answered here is not a wrong answer, it is a claimed act: the path
    # holds no tools, so the model cannot send and will report that it did.
    # Escalation hands the turn to the real registry, where `send_email` meets
    # the gateway and comes back as a preview rather than as prose.
    decision = query_intent.classify(
        user_input,
        has_context=bool(conversation_history),
        history=conversation_history,
    )

    # ── Step 0a: a reply to an action already held for approval ──────────────
    #
    # Strictly ahead of the escalation below, and the ordering is load-bearing.
    # "send it" and "cancel" are simultaneously affirmations and action-shaped
    # sentences; asked in the other order, the escalation would read a reply to
    # a pending action as a request to prepare a second one, and the user's
    # "yes" would mint a fresh action instead of approving the one they read.
    #
    # It stays cheap on ordinary turns: `should_intercept` matches the closed
    # set of decision words first and only asks the store when one hits, so the
    # overwhelming majority of turns pay a regex and no query.
    replying_to_pending = await confirmation.should_intercept(state)

    hard_reason = (
        None
        if replying_to_pending
        else query_intent.escalation_reason(decision.category, text=user_input)
    )
    if hard_reason:
        async for event in _escalate_to_tools(
            category=decision.category.value,
            reason=hard_reason,
            user_input=user_input,
            user_id=state["user_id"],
            session_id=session_id,
            conversation_history=conversation_history or [],
            output_mode=output_mode,
            scopes=scopes,
            memory_owner_id=memory_owner_id,
            memory_visibilities=memory_visibilities,
            request_id=state["request_id"],
        ):
            yield event
        return

    # ── Step 1+2: Memory retrieval and planning, concurrently ────────────────
    # These are independent — the planner classifies intent from the raw
    # utterance and does not read memory — so running them in series simply
    # added one round trip's worth of silence before the first token. The
    # timeout matters just as much: memory touches Postgres, Cohere, and
    # Qdrant, and any one of them hanging used to stall the whole spoken turn
    # forever with no audio and no error.
    init_ok = True
    try:
        state = await asyncio.wait_for(
            parallel_init_node(state), timeout=_INIT_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        init_ok = False
        logger.warning(
            "Streaming workflow init exceeded %.1fs; answering without memory context",
            _INIT_TIMEOUT_SECONDS,
        )
    except Exception as e:
        init_ok = False
        logger.warning("Streaming workflow init failed: %s", e)

    state.setdefault("memory_prompt", "")

    # Whether there is anything to answer a personal question *from*. The
    # memory prompt is where every retrieval safeguard is expressed — the
    # status hints, the "treat skills as unknown, not absent" line, the refusal
    # policy — so an empty one does not mean "the user has no data", it means
    # none of those safeguards are in the prompt. A model handed a personal
    # question with none of them is exactly the situation this phase is about.
    memory_grounded = init_ok and bool((state.get("memory_prompt") or "").strip())

    # The transcript is a separate signal, and it has to be. It is passed in by
    # the caller and rendered into the messages below, so it survives a memory
    # outage untouched — which is why "tell me more about that" and "go ahead"
    # keep streaming through one, while "what did I just tell you" spoken as
    # the very first turn of a conversation does not.
    history_grounded = bool(state.get("conversation_history"))
    if not state.get("selected_agent"):
        state["selected_agent"] = "profile"
    if not state.get("detected_intent"):
        state["detected_intent"] = user_input

    # ── Source-aware routing, identical to the tool-calling path ─────────────
    # This path used to read `needs_clarification` straight off the planner,
    # which meant every policy the graph applied — answer-first, one question
    # per conversation, the clock instead of memory — was silently absent from
    # spoken turns. One decision function now serves both.
    route = await decide_route(state)

    if route == "temporal":
        from app.tools import time_tool

        answer = time_tool.answer_temporal_query(user_input)
        from app.memory import provenance as _prov

        _prov.record(
            state.get("session_id") or "",
            category=state.get("query_category") or "",
            sources=["temporal_tool"],
            agent="temporal",
            tools=["current_datetime"],
            question=user_input,
        )
        yield {
            "type": "metadata",
            "selected_agent": "temporal",
            "detected_intent": state.get("query_category"),
            "execution_path": state.get("execution_path", []),
        }
        yield {
            "type": "complete",
            "display_text": answer,
            "speech_text": answer,
            "agent": "temporal",
            "success": True,
        }
        return

    if route == "confirm_action":
        # The same resolver the graph node calls — deliberately not a spoken
        # variant. A confirmation mechanism that differs between text and voice
        # is two mechanisms, and the weaker one becomes the way in.
        from app.agents import confirmation as _confirmation

        outcome = await _confirmation.resolve(state)
        yield {
            "type": "metadata",
            "selected_agent": "confirm_action",
            "detected_intent": state.get("query_category"),
            "execution_path": state.get("execution_path", []),
        }
        yield {
            "type": "complete",
            "display_text": outcome.text,
            "speech_text": outcome.text,
            "agent": "confirm_action",
            "success": True,
        }
        return

    if route == "provenance":
        # Read the record of the previous turn. Same answer the graph path
        # gives, for the same reason: a spoken "how did you know?" must not be
        # the one path where a model reconstructs its own sourcing.
        from app.memory import provenance

        answer = provenance.explain_last(state.get("session_id") or "")
        yield {
            "type": "metadata",
            "selected_agent": "provenance",
            "detected_intent": state.get("query_category"),
            "execution_path": state.get("execution_path", []),
        }
        yield {
            "type": "complete",
            "display_text": answer,
            "speech_text": answer,
            "agent": "provenance",
            "success": True,
        }
        return

    if route == "clarification":
        question = state.get("clarification_question") or "Could you clarify what you need help with?"
        yield {
            "type": "metadata",
            "selected_agent": "clarification",
            "detected_intent": state.get("detected_intent"),
            "execution_path": state.get("execution_path", []),
        }
        yield {
            "type": "complete",
            "display_text": question,
            "speech_text": question,
            "agent": "clarification",
            "success": True,
        }
        return

    if route in ("job", "email", "academic", "profile"):
        state["selected_agent"] = route

    # ── The second gate: stream only what retrieval actually grounded ────────
    #
    # Placed here rather than beside the first one because it is asking a
    # question the first one cannot: not "what kind of question is this" but
    # "did the sources that answer it arrive". Both are needed. The first stops
    # categories whose answers are unreachable from any prompt; this one stops
    # the far more common failure of a reachable answer that simply did not
    # arrive — Qdrant slow, Postgres slow, the 4-second ceiling hit — leaving a
    # personal question, an empty context block and a model willing to fill it.
    #
    # After the deterministic routes and before any token: `temporal`,
    # `provenance`, `confirm_action` and `clarification` have already returned
    # above, so a confirmation reply can never be pulled through here. That
    # ordering is load-bearing — the confirmation gateway must stay the first
    # thing that sees a "yes".
    #
    # `state["query_category"]` rather than the step-0 `decision`: this is what
    # `decide_route` concluded, and it is the value every other consumer of
    # this turn — logs, provenance, the episode record — will report.
    grounding_reason = query_intent.escalation_reason(
        state.get("query_category"),
        text=user_input,
        memory_grounded=memory_grounded,
        history_grounded=history_grounded,
    )
    if grounding_reason:
        logger.warning(
            "Refusing to stream a %s turn without grounded sources "
            "(init_ok=%s, memory=%s, history=%s, request=%s)",
            state.get("query_category"), init_ok, memory_grounded,
            history_grounded, state["request_id"],
        )
        async for event in _escalate_to_tools(
            category=str(state.get("query_category") or decision.category.value),
            reason=grounding_reason,
            user_input=user_input,
            user_id=state["user_id"],
            session_id=session_id,
            conversation_history=conversation_history or [],
            output_mode=output_mode,
            scopes=scopes,
            memory_owner_id=memory_owner_id,
            memory_visibilities=memory_visibilities,
            request_id=state["request_id"],
            # `parallel_init_node` already stored this utterance; ingesting it
            # again would duplicate it in the thread.
            skip_user_ingest=True,
            # …and already retrieved and planned. Handed over so the tool
            # workflow adopts it instead of running a second full init.
            prefetched=_prefetched_from(state),
        ):
            yield event
        return

    # ── Emit metadata (agent selected, intent detected) ──────────────────────
    yield {
        "type": "metadata",
        "selected_agent": state.get("selected_agent"),
        "detected_intent": state.get("detected_intent"),
        "execution_path": state.get("execution_path", []),
        "planner_confidence": state.get("planner_confidence"),
    }

    # ── Step 3: Stream the agent's response ──────────────────────────────────
    agent = get_agent_by_name(state.get("selected_agent", "profile"))
    if not agent:
        yield {
            "type": "complete",
            "display_text": "I couldn't process your request.",
            "speech_text": "I couldn't process your request.",
            "error": "Agent not found",
            "success": False,
        }
        return

    system_prompt = _get_agent_system_prompt(agent, state)

    # Include recent conversation history for multi-turn continuity
    raw_history = state.get("conversation_history") or []
    history_messages = [
        {"role": turn["role"], "content": str(turn.get("content", ""))[:400]}
        for turn in raw_history[-4:]
        if turn.get("role") in ("user", "assistant") and turn.get("content")
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        *history_messages,
        {
            "role": "user",
            "content": (
                f"Intent: {state.get('detected_intent', 'general')}\n\n"
                f"Query: {user_input}"
            ),
        },
    ]

    accumulated_text = ""
    token_index = 0
    fabricated: Optional[str] = None

    # ── Why this generator is held in a variable and closed explicitly ───────
    #
    # It holds a Groq limiter slot for the lifetime of the iteration, and this
    # loop has three ways to leave without exhausting it: the `break` below when
    # a reply fabricates a completion, an exception, and cancellation — which is
    # what a voice barge-in does on every interruption.
    #
    # Abandoning an async generator suspended at a yield does not run its
    # `finally`. CPython does finalize it eventually, via a garbage-collection
    # hook that schedules `aclose()` on the loop, so the slot is not leaked
    # permanently — but "eventually" is not a property to depend on for the
    # single gate every Groq request in the process passes through. Four
    # interrupted turns holding slots that GC has not yet reclaimed is a stall
    # with no error anywhere.
    #
    # `aclose()` in a `finally` makes the release a control-flow event.
    token_stream = groq_service.stream_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    try:
        async for token in token_stream:
            accumulated_text += token

            # ── The last line of defence ─────────────────────────────────────
            #
            # This path holds no tools. `ActionGateway.confirm_and_execute` is
            # the only code in the system that invokes a confirmable tool, and
            # it is not reachable from here — so a reply forming the words "I've
            # sent it" is not making a claim that might be wrong, it is making
            # one that is false. That is a fact about the architecture, not a
            # judgement about the sentence, which is what makes acting on it
            # safe.
            #
            # Checked per token and *before* the token is yielded, so the
            # clause that completes the claim is never emitted. The turn is then
            # handed to the tool workflow, which prepares the action properly
            # and answers with the gateway's preview — the user gets what they
            # asked for, held for their approval, rather than a lie about it.
            #
            # The windowed form, because this runs on every token of every
            # streamed reply and the whole point of this path is latency.
            fabricated = claims_completion_in_stream(accumulated_text)
            if fabricated:
                break

            token_index += 1
            yield {
                "type": "token",
                "token": token,
                "index": token_index,
                "accumulated": accumulated_text,
            }

        if fabricated:
            logger.error(
                "Tool-free reply claimed a %s completion it could not have "
                "performed; escalating to the tool workflow (request=%s)",
                fabricated, state["request_id"],
            )
            async for event in _escalate_to_tools(
                category=str(state.get("query_category") or decision.category.value),
                reason=query_intent.ESCALATION_ACTION_REQUEST,
                user_input=user_input,
                user_id=state["user_id"],
                session_id=session_id,
                conversation_history=conversation_history or [],
                output_mode=output_mode,
                scopes=scopes,
                memory_owner_id=memory_owner_id,
                memory_visibilities=memory_visibilities,
                request_id=state["request_id"],
                # `parallel_init_node` already stored this utterance.
                skip_user_ingest=True,
                # …and already retrieved and planned. See above.
                prefetched=_prefetched_from(state),
            ):
                yield event
            return

        # ── Persist the streamed response to memory ───────────────────────
        try:
            from app.memory.memory_manager import memory_manager as _mm
            await _mm.on_agent_response(
                user_id=state["user_id"],
                session_id=session_id,
                agent_response=accumulated_text,
                metadata={
                    "agent": agent.name,
                    "intent": state.get("detected_intent"),
                    "streaming": True,
                },
            )
        except Exception as _e:
            logger.warning("Streaming workflow memory save failed: %s", _e)

        # Record where this answer came from, so a spoken "how did you know?"
        # on the very next turn reads a fact instead of reconstructing one.
        try:
            from app.memory import provenance

            provenance.record(
                session_id or "",
                category=state.get("query_category") or "",
                sources=state.get("memory_sources") or [],
                agent=agent.name,
                tools=[],
                question=user_input,
            )
        except Exception as _e:
            logger.debug("Could not record streaming provenance: %s", _e)

        yield {
            "type": "complete",
            "display_text": accumulated_text,
            "speech_text": accumulated_text,
            "agent": agent.name,
            "user_id": state["user_id"],
            "session_id": session_id,
            "success": True,
        }

    except Exception as e:
        logger.error("Streaming workflow LLM error: %s", e)
        yield {
            "type": "complete",
            "display_text": "I encountered an error processing your request.",
            "speech_text": "I encountered an error.",
            "error": str(e),
            "success": False,
        }
    finally:
        # Returns the limiter slot on every exit path — the `break` above, an
        # exception, or cancellation. See the note at the top of this block.
        # Closing an already-exhausted generator is a no-op, so the normal
        # completion path pays nothing for this.
        await token_stream.aclose()
