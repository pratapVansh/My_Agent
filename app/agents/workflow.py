"""
LangGraph workflow with Phase 1 upgrades:
- Confidence-gated routing (clarification node when planner confidence < 0.6)
- Plan-Execute-Reflect loop (max 3 iterations, retries failed tasks)
- Structured TaskEnvelope flowing through all agents
"""
import logging
import re
import uuid
from typing import Dict, Any, FrozenSet, Literal, Optional
import asyncio

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
from app.agents import (
    clarification_policy,
    confirmation,
    grounding,
    profile_intent,
    query_intent,
)
from app.agents.state import AgentState
from app.agents.planner_agent import planner_agent
from app.agents.job_agent import job_agent
from app.agents.email_agent import email_agent
from app.agents.academic_agent import academic_agent
from app.agents.profile_agent import profile_agent
from app.agents.response_agent import response_agent
from app.memory.memory_manager import memory_manager
from app.memory import provenance
from app.memory.sources import QueryCategory, sources_for
from app.services import call_metrics
from app.services.debug_logger import log_step
from app.services.llm_errors import LLMErrorKind, classify_llm_error
from app.services.langsmith_service import traceable
from app.config import settings

# Reflect passes per specialist, per plan step.
#
# Was 3. Each pass re-runs the *whole* specialist — its own reasoning loop of
# up to three LLM calls — so the ceiling was nine calls for one plan step
# before the planner and the response agent were counted. Two still recovers
# the failure this loop exists for (a required tool the first attempt did not
# call, which the injected failure context reliably fixes on the retry) at half
# the worst-case cost.
MAX_ITERATIONS = 2


def _precategorise(state: AgentState) -> Optional[str]:
    """
    Classify the query before retrieval, so context is assembled for it.

    Classification is deterministic and costs microseconds, which is what makes
    it affordable to run here *and* again at the routing edge rather than
    caching a value that could go stale. The reason it must run first is that
    the prompt is assembled during retrieval: without a category, every question
    receives every source, and a question about a CGPA arrives buried in project
    summaries and résumé excerpts.
    """
    try:
        decision = query_intent.classify(
            state.get("user_input", "") or "",
            has_context=_has_conversation_context(state),
            history=_conversation_turns(state),
        )
        state["query_category"] = decision.category.value
        return decision.category.value
    except Exception as exc:
        # Never fail a turn over context scoping — an unscoped prompt is the
        # previous behaviour and is merely noisier, not broken.
        logger.warning("Query pre-categorisation failed: %s", exc)
        return None


# ─────────────────────────────────────────────
# Memory retrieval node
# ─────────────────────────────────────────────
@traceable(name="memory_node", run_type="chain", tags=["workflow", "memory"])
async def memory_node(state: AgentState) -> AgentState:
    """
    Retrieval for the sequential branch (`parallel_workflow_enabled=false`).

    Category scoping and retrieval-scope safety are applied here exactly as in
    `parallel_init_node`, so the flag changes timing rather than behaviour.

    One deliberate exception: `state["prefetched"]` is *not* adopted on this
    path. A handover only arrives from `_escalate_to_tools`, and honouring it
    here without the matching skip in `planner_node` would reuse the retrieval
    while still paying for the planner — a half-applied optimisation that is
    harder to reason about than not applying it at all. The parallel branch is
    the default and does adopt it.
    """
    user_id = state.get("user_id")
    session_id = state.get("session_id", "default_session")
    user_input = state.get("user_input", "")

    if not user_id:
        state["error"] = "Missing required user_id in workflow state"
        state["memory_context"] = {}
        state["memory_prompt"] = ""
        return state

    category = _precategorise(state)

    try:
        if not state.get("skip_user_ingest"):
            await memory_manager.on_user_input(
                user_id=user_id, session_id=session_id, user_message=user_input
            )
        memory_context, memory_prompt = await memory_manager.build_memory_prompt(
            user_id=user_id, session_id=session_id, query=user_input,
            memory_owner_id=state.get("memory_owner_id"),
            visibilities=state.get("memory_visibilities"),
            category=category,
        )
        state["memory_context"] = memory_context
        state["memory_prompt"] = memory_prompt
    except Exception as e:
        logger.error("Memory retrieval error: %s", e)
        state["memory_context"] = {}
        state["memory_prompt"] = ""

    return state


# ─────────────────────────────────────────────
# Parallel init node (memory + planner in parallel)
# ─────────────────────────────────────────────
@traceable(name="parallel_init_node", run_type="chain", tags=["workflow", "optimization"])
async def parallel_init_node(state: AgentState) -> AgentState:
    """
    Run memory retrieval and planner in parallel.
    Planner uses user_input only (no memory) for speed.
    Memory result is then available for the specialist agent.
    """
    user_id = state.get("user_id")
    session_id = state.get("session_id", "default_session")
    user_input = state.get("user_input", "")

    if not user_id:
        state["error"] = "Missing required user_id in workflow state"
        state["memory_context"] = {}
        state["memory_prompt"] = ""
        return state

    # ── Work a caller already paid for ───────────────────────────────────────
    # The streaming path retrieves memory and runs the planner, then escalates
    # a turn that needs tools to this workflow — which used to run both again
    # from scratch. `skip_user_ingest` suppressed the duplicate *write* but
    # nothing suppressed the duplicate planner call or the duplicate retrieval
    # fan-out, so an escalated spoken turn paid for its initialization twice.
    if _adopt_prefetched(state):
        await _settle_route(state)
        return state

    category = _precategorise(state)

    # ── Is the planner load-bearing on this turn? ────────────────────────────
    # `_precategorise` has already run and is deterministic, so for a handful
    # of categories the answer is known before the model is asked: the route is
    # owned by the category and terminates without a specialist, which means
    # every field the planner produces is discarded by `agent_for`. Asking
    # anyway cost a full 120B call to answer "what time is it" from a clock.
    planner_needed = query_intent.planner_is_load_bearing(category)

    try:
        async def memory_task():
            try:
                # Suppressed when the caller has already stored this utterance
                # — see `run_workflow(skip_user_ingest=...)`.
                #
                # Still awaited, deliberately. The expensive half of this call
                # (the Cohere embedding and the Qdrant upsert) is detached
                # inside `on_user_input` itself; what remains is two Postgres
                # writes that are ordering-critical — `on_agent_response` reads
                # the user's message back out of the store to pair the two
                # halves of the turn, so detaching the whole call here would
                # race that read and pair the reply with nothing.
                if not state.get("skip_user_ingest"):
                    await memory_manager.on_user_input(
                        user_id=user_id, session_id=session_id, user_message=user_input
                    )
                return await memory_manager.build_memory_prompt(
                    user_id=user_id, session_id=session_id, query=user_input,
                    memory_owner_id=state.get("memory_owner_id"),
                    visibilities=state.get("memory_visibilities"),
                    category=category,
                )
            except Exception as e:
                logger.error("Memory retrieval error: %s", e)
                return {}, ""

        async def planner_task():
            if not planner_needed:
                logger.debug(
                    "Skipping the planner for category %s — its route is already "
                    "decided and its output would be discarded.", category,
                )
                return {}
            try:
                with call_metrics.phase("planner"):
                    return await planner_agent.execute(state)
            except Exception as e:
                logger.error("Planner error: %s", e)
                return state

        with call_metrics.phase("init"):
            (memory_context, memory_prompt), planner_state = await asyncio.gather(
                memory_task(), planner_task()
            )

        state["memory_context"] = memory_context
        state["memory_prompt"] = memory_prompt

        # Merge planner results
        for key in ["selected_agent", "detected_intent", "agent_reasoning",
                    "execution_path", "error", "planner_confidence",
                    "needs_clarification", "clarification_question"]:
            if key in planner_state:
                state[key] = planner_state[key]

    except Exception as e:
        logger.error("Parallel init error: %s", e)
        state["memory_context"] = {}
        state["memory_prompt"] = ""
        state["error"] = f"Initialization error: {str(e)}"

    await _settle_route(state)
    return state


# Fields a caller may hand over. Anything absent is simply not adopted, so a
# partial handover degrades to computing the rest rather than to a broken turn.
_PREFETCHABLE_FIELDS = (
    "memory_context",
    "memory_prompt",
    "query_category",
    "selected_agent",
    "detected_intent",
    "planner_confidence",
    "needs_clarification",
    "clarification_question",
    "execution_plan",
)


def _adopt_prefetched(state: AgentState) -> bool:
    """
    Take over initialization a caller already performed. True if it was used.

    Requires the memory prompt to be present, because that is the field whose
    absence is indistinguishable from "retrieval returned nothing" — adopting a
    handover without it would silently answer a personal question with no
    memory and no way to tell that had happened. Everything else is optional.
    """
    prefetched = state.get("prefetched")
    if not isinstance(prefetched, dict) or "memory_prompt" not in prefetched:
        return False

    for key in _PREFETCHABLE_FIELDS:
        if key in prefetched:
            state[key] = prefetched[key]

    # Planner defaults the escalating caller may not have set, so the graph
    # downstream reads the same shape it would from a normal init.
    state.setdefault("execution_plan", [])
    state["current_step_index"] = state.get("current_step_index") or 0
    state["step_results"] = state.get("step_results") or {}

    logger.info(
        "Adopted prefetched initialization (category=%s, agent=%s) — planner and "
        "retrieval not repeated.",
        state.get("query_category"), state.get("selected_agent"),
    )
    return True


async def _settle_route(state: AgentState) -> str:
    """
    Make the routing decision here, in a node, where its writes survive.

    This is the whole fix for a class of bugs that all had the same shape. The
    decision was being made in `route_after_init`, which LangGraph treats as a
    conditional edge: it returns a route and its state mutations are dropped.
    So `profile_intent`, `memory_sources`, `followup_subject` and the chosen
    specialist were computed correctly and then thrown away, and three separate
    downstream mechanisms silently degraded — the profile agent's deterministic
    tool directive arrived empty, reflect retried the planner's stale choice
    rather than the agent that failed, and nothing downstream could tell which
    tools the turn was obliged to call.

    Running it from the node costs one extra classification on the turns that
    reach a specialist, which is microseconds of pure function, and buys a
    decision that the rest of the graph can actually read.
    """
    try:
        return await decide_route(state)
    except Exception as exc:
        # Routing must never be the thing that fails a turn. Falling through to
        # `route_after_init` reproduces the previous behaviour exactly.
        logger.warning("Could not settle the route in-node: %s", exc)
        return ""


# ─────────────────────────────────────────────
# Clarification node — fires when planner confidence < 0.6
# ─────────────────────────────────────────────
@traceable(name="clarification_node", run_type="chain", tags=["workflow", "clarification"])
async def clarification_node(state: AgentState) -> AgentState:
    """
    Converts the clarification question into a display/speech response
    without routing to any specialist agent.
    """
    question = state.get("clarification_question") or "Could you clarify what you need help with?"
    confidence = state.get("planner_confidence", 0.0)

    log_step("CLARIFICATION", {"confidence": confidence, "question": question})

    # Pack as a minimal task_result so response_agent can render it
    state["task_result"] = {
        "agent": "planner",
        "result": {"content": question},
        "status": "clarification",
        "confidence": confidence,
        "evidence": [],
        "next_actions": [],
        "goal": "clarify user intent",
        "inputs": {},
        "constraints": {},
        "task_id": "clarification",
    }
    state["display_text"] = question
    state["speech_text"] = question
    state["current_agent"] = "clarification"
    if state.get("execution_path") is not None:
        state["execution_path"].append("clarification")

    return state


# ─────────────────────────────────────────────
# Specialist agent nodes
# ─────────────────────────────────────────────
@traceable(name="planner_node", run_type="chain", tags=["workflow", "planner"])
async def planner_node(state: AgentState) -> AgentState:
    state = await planner_agent.execute(state)
    # The sequential branch settles its route here for the same reason the
    # parallel one does — see `_settle_route`. Keeping both branches identical
    # is what stops `parallel_workflow_enabled` from changing behaviour rather
    # than only timing.
    await _settle_route(state)
    return state


@traceable(name="job_node", run_type="chain", tags=["workflow", "job"])
async def job_node(state: AgentState) -> AgentState:
    return await job_agent.execute(state)


@traceable(name="email_node", run_type="chain", tags=["workflow", "email"])
async def email_node(state: AgentState) -> AgentState:
    return await email_agent.execute(state)


@traceable(name="academic_node", run_type="chain", tags=["workflow", "academic"])
async def academic_node(state: AgentState) -> AgentState:
    return await academic_agent.execute(state)


@traceable(name="profile_node", run_type="chain", tags=["workflow", "profile"])
async def profile_node(state: AgentState) -> AgentState:
    return await profile_agent.execute(state)


@traceable(name="response_node", run_type="chain", tags=["workflow", "response"])
async def response_node(state: AgentState) -> AgentState:
    result = await response_agent.execute(state)
    _record_provenance(result)
    return result


def _record_provenance(state: AgentState) -> None:
    """
    Write down where this turn's answer came from.

    Called on every path that produces a final answer, because "how did you
    know?" can follow any of them — a date from the clock as easily as a CGPA
    from a résumé. The tools recorded are the ones the agent reported actually
    calling (`evidence`), not the ones it was offered; that difference is the
    entire value of the record.
    """
    envelope = state.get("task_result") or {}
    conversation_id = state.get("session_id") or ""
    if not conversation_id:
        return

    # A turn that failed produced no answer, so it has no provenance — and
    # recording an empty one would erase the record of the last answer that
    # *did* succeed. A live run against an exhausted LLM quota did exactly
    # that: an errored turn overwrote a good record with `sources=()`, so the
    # following "how did you know?" reported nothing known about an answer the
    # user had actually received.
    #
    # The test is the envelope's own status, deliberately NOT `state["error"]`.
    # A conditional-edge function in LangGraph returns a route, not state, so
    # the `state["error"] = None` that `decide_route` performs before sending a
    # turn to `temporal`/`degraded`/`provenance` never reaches the node — those
    # nodes run with the error flag still set precisely *because* they are the
    # paths that answer correctly in spite of it. Reading it here suppressed
    # the record for every degraded-mode answer, which a live run caught: the
    # clock answered, and the next "how did you know?" claimed no record.
    status = envelope.get("status")
    if not envelope or status == "failed":
        logger.debug("Not recording provenance for a failed turn (status=%s)", status)
        return

    category = state.get("query_category") or ""

    # `memory_sources` is written by `decide_route`, which is a conditional-edge
    # function — so like the error flag above, it never reaches the node. Rather
    # than depend on that propagation, derive the sources from the category
    # itself: the precedence table is the definition of which stores answer a
    # category, so deriving it here yields the same list `decide_route` would
    # have written. Without this the record kept `sources=()` and the clock's
    # provenance read "an unrecorded source" for an answer that came, provably,
    # from the clock.
    sources = list(state.get("memory_sources") or [])
    if not sources and category:
        try:
            sources = [s.value for s in sources_for(QueryCategory(category))]
        except ValueError:
            sources = []

    try:
        provenance.record(
            conversation_id,
            category=category,
            sources=sources,
            agent=envelope.get("agent") or state.get("current_agent") or "",
            tools=[str(e) for e in (envelope.get("evidence") or [])],
            answerability=state.get("answerability") or "",
            question=state.get("user_input") or "",
        )
    except Exception as exc:  # never let bookkeeping break a delivered answer
        logger.debug("Could not record provenance: %s", exc)


@traceable(name="confirm_action_node", run_type="chain", tags=["workflow", "confirm"])
async def confirm_action_node(state: AgentState) -> AgentState:
    """
    Approve or cancel an action the user was shown.

    No model call. The decision was already made deterministically by
    `confirmation.detect`, and the execution — if any — happens inside
    `ActionGateway.confirm_and_execute`, which is the only code in the system
    that invokes a confirmable tool. This node holds no tool callable and could
    not run one if it tried.
    """
    outcome = await confirmation.resolve(state)

    log_step("confirm_action", {
        **outcome.summary(),
        "intent": QueryCategory.CONFIRM_ACTION.value,
    })

    state["task_result"] = {
        "agent": "confirm_action",
        "result": {"content": outcome.text},
        "status": "success",
        "confidence": 1.0,
        "evidence": ["action_gateway"],
        "next_actions": [],
        "goal": "approve or cancel a held action",
        "inputs": {"user_input": state.get("user_input", "")},
        "constraints": {},
        "task_id": "confirm_action",
    }
    state["display_text"] = outcome.text
    state["speech_text"] = outcome.text
    state["current_agent"] = "confirm_action"
    if state.get("execution_path") is not None:
        state["execution_path"].append("confirm_action")

    _record_provenance(state)
    return state


@traceable(name="provenance_node", run_type="chain", tags=["workflow", "provenance"])
async def provenance_node(state: AgentState) -> AgentState:
    """
    Answer "how did you know that?" from the record of the previous turn.

    No model call and no retrieval — for the same reason `temporal_node` makes
    neither. The question has exactly one true answer, it was written down when
    the answer was given, and asking a model to reconstruct it produces a
    fluent account of a lookup that may never have happened.

    When nothing was recorded, this says so. An unrecorded provenance is a
    NO_DATA about the assistant's own behaviour, and inventing one is the same
    class of error as inventing a CGPA.
    """
    conversation_id = state.get("session_id") or ""
    entry = provenance.last(conversation_id)
    answer = entry.explain() if entry else provenance.NO_RECORD

    log_step("memory_query", {
        "intent": QueryCategory.PROVENANCE_QUERY.value,
        "sources": ["conversation_current"],
        "retrieval": False,
        "answerability": "ANSWERABLE" if entry else "NO_DATA",
        "clarification": False,
        **({"explains": entry.summary()} if entry else {}),
    })

    state["task_result"] = {
        "agent": "provenance",
        "result": {"content": answer},
        "status": "success" if entry else "partial",
        "confidence": 1.0 if entry else 0.5,
        "evidence": ["answer_provenance"],
        "next_actions": [],
        "goal": "explain where the previous answer came from",
        "inputs": {"user_input": state.get("user_input", "")},
        "constraints": {},
        "task_id": "provenance",
    }
    state["display_text"] = answer
    state["speech_text"] = answer
    state["current_agent"] = "provenance"
    if state.get("execution_path") is not None:
        state["execution_path"].append("provenance")

    # Deliberately NOT recorded as this turn's provenance: an explanation is not
    # itself an answer anyone asks the origin of, and overwriting the record
    # would make a second "how did you know?" explain the explanation.
    return state


# Categories whose answer is a stored value read back verbatim, with no
# inference. These stay available when the language model does not.
_DEGRADED_ANSWERABLE = frozenset({
    QueryCategory.PROFILE_IDENTITY,
    QueryCategory.EXPLICIT_MEMORY,
})


@traceable(name="degraded_node", run_type="chain", tags=["workflow", "degraded"])
async def degraded_node(state: AgentState) -> AgentState:
    """
    Answer from stored records when the language model is unavailable.

    Reads keys and reports them. It composes no prose beyond a fixed template
    and never touches a model, so it cannot hallucinate — which is the whole
    reason it is allowed to run in exactly the situation where the normal
    safeguards (tool observations, reflect) are unavailable.

    When nothing is stored it says so plainly *and* says the model is down, so
    the user can tell "I have no record of this" apart from "I couldn't think
    right now". Collapsing those two is the failure this system is most careful
    about everywhere else.
    """
    from app.memory import identity as identity_module

    user_id = state.get("user_id") or ""
    category = state.get("query_category") or ""
    answer = ""

    try:
        facts = await memory_manager.get_profile_facts(user_id=user_id)
    except Exception as exc:
        logger.error("Degraded lookup failed for user=%s: %s", user_id, exc)
        facts = []

    by_key = {str(f.get("key", "")): f.get("value") for f in facts if f.get("value")}

    if category == QueryCategory.PROFILE_IDENTITY.value:
        name = by_key.get(identity_module.CANONICAL_NAME_KEY)
        if not name:
            try:
                resume = await memory_manager.retrieve_resume(user_id=user_id)
                name = (resume or {}).get("name")
            except Exception as exc:
                logger.warning("Degraded résumé lookup failed: %s", exc)
        answer = (
            f"Your name is {name}."
            if name else
            "I don't have your name on file."
        )

    elif category == QueryCategory.EXPLICIT_MEMORY.value:
        remembered = []
        for key, value in by_key.items():
            if not identity_module.is_explicit_memory_key(key):
                continue
            label = key.replace("remembered_", "").replace("_", " ").strip()
            # "the name Devasi" rather than "name: Devasi" — this text is read
            # aloud on the voice path, where a colon is a stumble.
            remembered.append(f"the {label} {value}" if label else str(value))

        answer = (
            "You asked me to remember " + ", and ".join(remembered) + "."
            if remembered else
            "You haven't asked me to remember anything yet."
        )

    if not answer:
        answer = "I couldn't complete that request."

    answer += (
        " (I'm running on stored records only right now — my language model is "
        "temporarily unavailable, so I can't go further than what's on file.)"
    )

    log_step("memory_query", {
        "intent": category,
        "sources": state.get("memory_sources") or [],
        "retrieval": True,
        "degraded": True,
        "reason": (state.get("degraded_reason") or "")[:120],
    })

    state["task_result"] = {
        "agent": "degraded",
        "result": {"content": answer},
        "status": "partial",
        "confidence": 0.6,
        "evidence": ["profile_facts"],
        "next_actions": [],
        "goal": "answer from stored records without a model",
        "inputs": {"user_input": state.get("user_input", "")},
        "constraints": {},
        "task_id": "degraded",
    }
    state["display_text"] = answer
    state["speech_text"] = answer
    state["current_agent"] = "degraded"
    if state.get("execution_path") is not None:
        state["execution_path"].append("degraded")

    # Bypasses `response`, so it records its own provenance — "how did you know?"
    # follows a degraded answer as readily as any other.
    _record_provenance(state)
    return state


@traceable(name="temporal_node", run_type="chain", tags=["workflow", "temporal"])
async def temporal_node(state: AgentState) -> AgentState:
    """
    Answer "what is today's date?" from the clock.

    No model call and no retrieval. The question is deterministic, the answer is
    arithmetic, and the previous behaviour — routing it into memory, finding
    nothing, and reporting no real-time access — was wrong in a way no amount of
    better retrieval could fix. Memory is for what was true; the clock is for
    what is true now.
    """
    from app.tools import time_tool

    query = state.get("user_input", "") or ""
    answer = time_tool.answer_temporal_query(query)

    log_step("memory_query", {
        "intent": QueryCategory.TEMPORAL_CURRENT.value,
        "sources": ["temporal_tool"],
        "retrieval": False,
        "answerability": "ANSWERABLE",
        "clarification": False,
    })

    state["task_result"] = {
        "agent": "temporal",
        "result": {"content": answer},
        "status": "success",
        "confidence": 1.0,
        "evidence": ["current_datetime"],
        "next_actions": [],
        "goal": "report the current date/time",
        "inputs": {"user_input": query},
        "constraints": {},
        "task_id": "temporal",
    }
    state["display_text"] = answer
    state["speech_text"] = answer
    state["current_agent"] = "temporal"
    if state.get("execution_path") is not None:
        state["execution_path"].append("temporal")

    # Bypasses `response`, so it records its own provenance.
    _record_provenance(state)
    return state


# ─────────────────────────────────────────────
# Reflect node — validates output, decides retry/next_step/done
# ─────────────────────────────────────────────
@traceable(name="reflect_node", run_type="chain", tags=["workflow", "reflect"])
async def reflect_node(state: AgentState) -> AgentState:
    """
    Plan-Execute-Reflect step:
    1. Check if task succeeded (TaskEnvelope.status)
    2. If failed AND iterations remaining → mark for retry WITH failure context (Fix 2)
    3. If succeeded AND more plan steps remain → advance to next step (Fix 1)
    4. Otherwise → mark done
    """
    task_result = state.get("task_result") or {}
    status = task_result.get("status", "success")
    iteration_count = state.get("iteration_count") or 0
    iteration_count += 1
    state["iteration_count"] = iteration_count

    confidence = task_result.get("confidence", 0.0)

    # ── What "failed" means ──────────────────────────────────────────────────
    #
    # Every specialist computes `status = "success" if final_answer else
    # "failed"`, and the reasoning loop substitutes a fallback sentence when the
    # model produces nothing — so `final_answer` is never empty and the envelope
    # was effectively always "success". The retry branch below could therefore
    # only be reached by an agent raising, which made the entire learn-from-
    # failure mechanism unreachable for the failure that actually happens: a
    # tool that errored.
    #
    # `answerability` is the loop's own verdict on what the tools returned, and
    # TOOL_ERROR means every tool it called failed. That is a failed attempt by
    # any honest reading, and it is precisely the case a different strategy
    # might recover from — a retry may pick different arguments or a different
    # tool. NO_DATA deliberately does not qualify: an empty store is an answer,
    # and retrying it just asks the same empty store again.
    answerability = state.get("answerability") or ""
    if status != "failed" and answerability == "TOOL_ERROR":
        logger.info(
            "Reflect: treating a TOOL_ERROR turn as failed so a retry can use a "
            "different strategy (agent=%s)", task_result.get("agent"),
        )
        status = "failed"

    # ── The tool was never called ────────────────────────────────────────────
    #
    # `grounding == "skipped"` means the category required a tool, the agent
    # held it, and the model answered without it — so `grounding.enforce`
    # replaced the answer with a refusal. That refusal is correct and must
    # stay; what was missing is that the turn ended there. It is the *most*
    # recoverable failure in the system: the tool exists and the data is
    # sitting behind it, the model simply did not look. A retry re-enters the
    # loop with `reflect_failure_context` injected, which is exactly the
    # signal that makes the second attempt call the tool.
    #
    # Measured against `openai/gpt-oss-120b`, the first attempt skips a
    # required tool often enough that without this the user sees "please ask
    # again" on a question the agent could answer.
    #
    # Only SKIPPED. NO_DATA is an answer (the store really is empty) and
    # FAILED is already covered by TOOL_ERROR above — retrying either just
    # asks the same empty store or the same broken tool a second time.
    if status != "failed" and (state.get("grounding") or "") == "skipped":
        logger.info(
            "Reflect: required tool was never called (grounding=skipped); "
            "retrying so the tool actually runs (agent=%s)",
            task_result.get("agent"),
        )
        status = "failed"

    # ── The one failure that must never be retried ───────────────────────────
    #
    # A retry here re-runs the entire specialist — a fresh reasoning loop, up
    # to `max_iterations` more model calls. When the reason the turn failed is
    # that the provider is rate limiting us, that is the most expensive
    # possible response to being told we are asking for too much: the retry
    # lands in the same closed window, fails the same way, and each rejected
    # request still counts against the account. The loop becomes a cause of the
    # outage it is reacting to.
    #
    # A turn where every LLM attempt timed out and none returned is excluded
    # for the same reason: a timeout is what queueing looks like from this
    # side, and queueing is what rate limiting looks like before the provider
    # gets round to saying so. Re-running a specialist against a provider that
    # is not answering produces more unanswered calls. One slow call that a
    # later iteration recovered from is *not* this case — see
    # `llm_unreachable`, which requires that nothing came back at all.
    #
    # This is a *retry* gate only. The turn still routes to the response agent,
    # still reports honestly, and the NO_DATA / TOOL_ERROR distinction below is
    # untouched — an empty store and a broken tool still mean what they meant.
    envelope = state.get("task_result") or {}
    provider_exhausted = (
        bool(state.get("rate_limited"))
        or envelope.get("rate_limited") is True
        or bool(state.get("llm_unreachable"))
        or envelope.get("llm_unreachable") is True
    )
    if not provider_exhausted:
        error_text = str(state.get("error") or "")
        if error_text:
            provider_exhausted = (
                classify_llm_error(RuntimeError(error_text)) is LLMErrorKind.RATE_LIMITED
            )

    log_step("REFLECT", {
        "status": status,
        "answerability": answerability,
        "confidence": confidence,
        "iteration": iteration_count,
        "max": MAX_ITERATIONS,
        "provider_exhausted": provider_exhausted,
    })

    if status == "failed" and provider_exhausted:
        logger.error(
            "Reflect: not retrying a turn the model provider did not serve "
            "(agent=%s, rate_limited=%s, llm_unreachable=%s). Re-running the "
            "specialist would issue the same calls into the same conditions.",
            task_result.get("agent"),
            bool(state.get("rate_limited")) or envelope.get("rate_limited") is True,
            bool(state.get("llm_unreachable")) or envelope.get("llm_unreachable") is True,
        )
        state["reflect_outcome"] = "done"
        state["reflect_failure_context"] = None
        if state.get("execution_path") is not None:
            state["execution_path"].append(f"reflect_{iteration_count}_rate_limited")
        return state

    # ── Fix 2: Reflect learns from failures ─────────────────────────────────
    if status == "failed" and iteration_count < MAX_ITERATIONS:
        # Build a rich failure context so the retry attempt uses a different strategy
        result_content = task_result.get("result", {}).get("content", "")
        tools_tried = task_result.get("evidence", [])
        agent_name = task_result.get("agent", state.get("selected_agent", "agent"))

        failure_lines = [
            f"Your previous attempt as the '{agent_name}' agent FAILED.",
        ]
        if result_content:
            failure_lines.append(f"It returned: \"{result_content[:250]}\"")
        if tools_tried:
            failure_lines.append(f"Tools you already tried: {', '.join(tools_tried)}.")
        failure_lines.append(
            "You MUST try a different approach: use different tool parameters, "
            "try an alternative tool, or reason without tools if appropriate."
        )

        state["reflect_failure_context"] = " ".join(failure_lines)
        state["reflect_outcome"] = "retry"
        logger.warning(
            "Reflect: task failed (iteration %d/%d), retrying with failure context injected.",
            iteration_count, MAX_ITERATIONS,
        )
        if state.get("execution_path") is not None:
            state["execution_path"].append(f"reflect_{iteration_count}_retry")
        return state

    # Clear stale failure context on success
    state["reflect_failure_context"] = None

    # ── Fix 1: Multi-step plan advancement ──────────────────────────────────
    execution_plan = state.get("execution_plan") or []
    current_step_index = state.get("current_step_index") or 0

    # Some answers are finished when their agent returns, and a second step
    # would not extend them — it would *replace* them, because each step's
    # envelope overwrites the last. A live run showed a JOB_MATCH turn produce
    # the evidence-backed match report and then advance to the profile agent,
    # which paraphrased it into free prose; the rendered answer, its source ids
    # and its structured verdict were all lost. The planner cannot prevent this
    # because it never learns that routing overrode its agent choice, so the
    # decision is made here from the category instead.
    if query_intent.is_single_step(state.get("query_category")):
        if len(execution_plan) > 1:
            logger.info(
                "Ignoring a %d-step plan for a single-step category (%s): the "
                "answer is already complete.",
                len(execution_plan), state.get("query_category"),
            )
        state["reflect_outcome"] = "done"
        if state.get("execution_path") is not None:
            state["execution_path"].append(f"reflect_{iteration_count}_done")
        return state

    if status != "failed" and len(execution_plan) > 1 and current_step_index < len(execution_plan) - 1:
        # Current step succeeded and there are more steps to run
        step_results = state.get("step_results") or {}
        completed_step = execution_plan[current_step_index]
        result_summary = (task_result.get("result", {}).get("content", "") or "")[:400]
        step_results[str(current_step_index + 1)] = {
            "step": completed_step["step"],
            "agent": completed_step["agent"],
            "goal": completed_step["goal"],
            "summary": result_summary,
        }
        state["step_results"] = step_results

        # Advance to next step
        next_index = current_step_index + 1
        state["current_step_index"] = next_index
        next_step = execution_plan[next_index]
        state["selected_agent"] = next_step["agent"]
        state["detected_intent"] = next_step["goal"]

        # Build inter-step context so the next agent has full prior results
        context_lines = ["Results from previous steps:"]
        for k, v in step_results.items():
            context_lines.append(
                f"  Step {v['step']} ({v['agent']}): {v['goal']} → {v['summary']}"
            )
        state["inter_step_context"] = "\n".join(context_lines)

        # Reset iteration counter for the new step
        state["iteration_count"] = 0
        state["reflect_outcome"] = "next_step"

        log_step("PLAN_ADVANCE", {
            "completed_step": current_step_index + 1,
            "next_step": next_index + 1,
            "next_agent": next_step["agent"],
            "total_steps": len(execution_plan),
        })
        if state.get("execution_path") is not None:
            state["execution_path"].append(f"reflect_{iteration_count}_next_step_{next_index + 1}")
        return state

    # ── All steps done ───────────────────────────────────────────────────────
    state["reflect_outcome"] = "done"
    if status == "failed":
        logger.warning(
            "Reflect: task failed after %d iteration(s), returning best answer.",
            iteration_count,
        )
    if state.get("execution_path") is not None:
        state["execution_path"].append(f"reflect_{iteration_count}_done")
    return state


# ─────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────
def _is_self_referential(query: str) -> bool:
    """
    Whether the query asks about the user themselves or this conversation.

    Delegates to the deterministic classifier so the workflow and the profile
    agent agree on what counts as a personal question.
    """
    return profile_intent.is_self_referential(query)


def _conversation_turns(state: AgentState) -> list:
    """
    Recent turns, from wherever this path put them.

    The two workflow entry points populate different fields — the graph carries
    `conversation_history`, the memory node fills `memory_context.chat_history`
    — and a follow-up must resolve its subject from whichever is present. Both
    are consulted so "tell me more about that" works identically on the typed
    and spoken paths.
    """
    turns: list = []
    context = state.get("memory_context") or {}
    if isinstance(context, dict):
        history = context.get("chat_history")
        if isinstance(history, list):
            turns.extend(history)
    direct = state.get("conversation_history")
    if isinstance(direct, list):
        turns.extend(direct)
    return turns


def _has_conversation_context(state: AgentState) -> bool:
    """Whether earlier turns exist for a follow-up to refer back to."""
    return bool(_conversation_turns(state))


# The keys `MemoryManager.retrieve_context` actually returns. Kept in one place
# so a rename cannot silently turn this check into "no memory, always clarify".
_MEMORY_CONTEXT_KEYS = ("profile_facts", "episodes", "chat_history", "long_term", "preferences")


def _has_memory_signal(state: AgentState) -> bool:
    """Whether any memory source returned material for this turn."""
    context = state.get("memory_context") or {}
    if not isinstance(context, dict):
        return bool(context)
    for key in _MEMORY_CONTEXT_KEYS:
        value = context.get(key)
        if isinstance(value, str):
            if value.strip():
                return True
        elif value:
            return True
    # The assembled prompt is the ground truth about what the agent will see —
    # if anything reached it, the turn is answerable from memory.
    return bool((state.get("memory_prompt") or "").strip())


async def decide_route(state: AgentState) -> str:
    """
    The single routing decision, shared by the graph and the streaming path.

    **Call this from a node, not from a conditional edge.** Everything it writes
    to `state` — the category, the sources, the required tools, the specialist
    it picked — is load-bearing downstream, and a LangGraph conditional-edge
    function returns a route rather than state, so writes made from one are
    discarded. That is not a theoretical hazard: it is why the profile agent's
    "ROUTING DECISION (already made deterministically — follow it)" block
    arrived with the decision missing from it in seven cases out of eight, and
    why a failed job agent was retried as the profile agent.
    `route_after_init` therefore *reads* `state["route"]` instead of computing
    it, and the nodes that feed it call this first.

    Four things happen here, in this order, and the order is the design:

    1. **Categorise deterministically.** Before the planner's opinion is
       consulted, the query is classified by shape into one of the categories in
       `app.memory.sources`. This is what separates "what is today's date"
       (the clock) from "what is my current CPI" (an education chunk) — two
       questions that both contain a now-marker and that the planner, scoring
       intent from text alone, cannot tell apart.

    2. **Record the sources.** The category names an ordered list of stores.
       Carrying it on the state means the agent, the prompt assembler and the
       logs all agree on where the answer was supposed to come from.

    3. **Pick the specialist from the category, not only the planner.** A
       personal question goes to profile whatever the planner concluded, because
       the planner cannot see the store it would otherwise ask the user to
       substitute for.

    4. **Gate clarification last.** Asking is permitted only for an action
       missing a parameter no store holds, only once per conversation, and never
       after the user has asked to be questioned less.
    """
    query = state.get("user_input", "") or ""
    conversation_id = state.get("session_id") or ""
    turns = _conversation_turns(state)

    # ── 0. A reply to an action already held for approval ────────────────────
    # Ahead of classification, the planner, and every specialist — which is the
    # point. "Yes" is a word a language model will happily reinterpret as a
    # request to do something, and the one turn where that reinterpretation is
    # unrecoverable is the turn that approves an irreversible action. So the
    # decision is made here, deterministically, before any model output can
    # influence it.
    #
    # Gated on an action actually being pending for this caller in this
    # conversation: with nothing outstanding, "yes" is ordinary conversation
    # and falls straight through to normal routing.
    # Asynchronous because the answer now lives in Postgres rather than in this
    # process — which is what lets a "yes" be answered by a worker that never
    # saw the action proposed. LangGraph resolves coroutine path functions
    # under `ainvoke`, which is how this graph is always run.
    if await confirmation.should_intercept(state):
        state["query_category"] = QueryCategory.CONFIRM_ACTION.value
        state["memory_sources"] = [
            s.value for s in sources_for(QueryCategory.CONFIRM_ACTION)
        ]
        state["required_tools"] = []
        state["route"] = "confirm_action"
        log_step("memory_query", {
            "intent": QueryCategory.CONFIRM_ACTION.value,
            "retrieval": False,
            "agent": "confirm_action",
            "pending": len(await confirmation.pending_for_state(state)),
        })
        return "confirm_action"

    decision = query_intent.classify(query, has_context=bool(turns), history=turns)

    if state.get("error"):
        # Degraded mode. An upstream failure — almost always the LLM being rate
        # limited or down — used to turn *every* question into an error page,
        # including ones whose answer is a database row. A memory system that
        # cannot say the user's name because a language model is unavailable is
        # not a memory system; the name is stored, and reading it back needs no
        # inference at all.
        #
        # Only categories with a deterministic answer qualify. Anything needing
        # synthesis (experience, projects, general knowledge) genuinely depends
        # on what just broke and still short-circuits, because a half-answer
        # invented from fragments is worse than an honest failure.
        state["query_category"] = decision.category.value
        state["memory_sources"] = [s.value for s in decision.sources]

        if decision.category is QueryCategory.TEMPORAL_CURRENT:
            logger.info(
                "Answering a date/time question from the clock despite an "
                "upstream failure: %s", state.get("error"),
            )
            state["error"] = None
            state["route"] = "temporal"
            return "temporal"

        if decision.category is QueryCategory.PROVENANCE_QUERY:
            # Reads a record and formats it — no model, no retrieval, nothing
            # that the upstream failure could have broken. Found by a live run
            # against an exhausted LLM quota, where "how did you know?" came
            # back as a planner error despite needing no planner at all.
            logger.info(
                "Explaining the previous answer's source from the record "
                "despite an upstream failure: %s", state.get("error"),
            )
            state["error"] = None
            state["route"] = "provenance"
            return "provenance"

        if decision.category in _DEGRADED_ANSWERABLE:
            logger.info(
                "Answering %s from stored records in degraded mode: %s",
                decision.category.value, state.get("error"),
            )
            state["degraded_reason"] = str(state.get("error"))
            state["error"] = None
            state["route"] = "degraded"
            return "degraded"

        state["route"] = "response"
        return "response"

    state["query_category"] = decision.category.value
    state["memory_sources"] = [source.value for source in decision.sources]
    state["followup_subject"] = decision.subject
    if decision.profile_intent:
        # Kept under the legacy key: the profile agent's tool mapping and the
        # existing routing tests both read it.
        state["profile_intent"] = decision.profile_intent

    # Which tools this category may not answer without. Written here so the
    # specialist is handed the requirement rather than asked to infer it from a
    # tool list, and so `execute_reasoning_loop` can check afterwards whether
    # one of them actually ran. See `app.agents.grounding`.
    #
    # SCHEDULE_TEMPORAL is refined by its sub-intent, because that one category
    # spans the timetable, the attendance log, the exam list and personal plans.
    # Without the refinement, "what classes do I have tomorrow" counted as
    # grounded once `get_plans` had run.
    subintent = (
        query_intent.schedule_intent(query)
        if decision.category is QueryCategory.SCHEDULE_TEMPORAL else None
    )
    state["schedule_intent"] = subintent
    state["required_tools"] = list(
        grounding.required_tools(decision.category, subintent=subintent)
    )

    # A standing request for fewer questions is recorded before any decision is
    # made, so it applies to the very turn it was expressed in.
    clarification_policy.note_user_message(conversation_id, query)

    # The planner's plan shape is an input, not just an output: a compound
    # request it decomposed into several steps owns its own first step. See
    # `query_intent.agent_for`.
    plan_steps = len(state.get("execution_plan") or [])

    selected = query_intent.agent_for(
        decision, state.get("selected_agent"), plan_steps=plan_steps,
    )

    log_step("memory_query", {
        "query_chars": len(query),
        **decision.summary(),
        "retrieval": decision.requires_retrieval,
        "agent": selected,
        "required_tools": state["required_tools"],
    })

    if state.get("needs_clarification"):
        verdict = clarification_policy.evaluate(conversation_id, decision.category)
        state["clarification_reason"] = verdict.reason
        if verdict.allowed:
            attempts = clarification_policy.record_attempt(conversation_id)
            log_step("clarification", {
                **verdict.summary(),
                "attempts": attempts,
                "intent": decision.category.value,
            })
            state["route"] = "clarification"
            return "clarification"

        log_step("CLARIFICATION SUPPRESSED", {
            **verdict.summary(),
            "intent": decision.category.value,
            "agent": selected,
            "confidence": state.get("planner_confidence", 0.0),
        })
        state["needs_clarification"] = False
        state["clarification_question"] = ""

    # The specialist that will actually run, recorded as such.
    #
    # `selected_agent` used to hold only the planner's opinion, and this
    # function overrules that opinion routinely — so `route_after_reflect`,
    # which reads the field to decide who to retry, was retrying whoever the
    # planner had originally named. A JOB_MATCH turn whose job agent failed was
    # retried by the profile agent, which holds no `match_job` and no posting,
    # while being handed a failure context that said the job agent had failed.
    state["selected_agent"] = selected
    state["route"] = selected
    return selected


async def route_after_init(
    state: AgentState,
) -> Literal["clarification", "temporal", "degraded", "provenance", "confirm_action", "job", "email", "academic", "profile", "response"]:
    """
    Graph edge for the routing decision — a *reader*. See `decide_route`.

    The decision was already made and recorded by the node that ran before
    this, which is the only place its writes survive. Recomputing here is the
    fallback for a node that could not settle it (and for `memory_node`, which
    predates this arrangement); the answer is the same either way, but the
    state it leaves behind is not.
    """
    route = state.get("route") or await decide_route(state)
    if route not in (
        "clarification", "temporal", "degraded", "provenance", "confirm_action",
        "job", "email", "academic", "profile", "response",
    ):
        return "profile"
    return route


def route_after_reflect(state: AgentState) -> Literal["job", "email", "academic", "profile", "response"]:
    """
    Route from reflect:
    - "retry"     → the specialist that actually ran and failed
    - "next_step" → next specialist in execution plan (Fix 1)
    - "done"      → response agent

    On a retry the agent is taken from the envelope first, because the envelope
    is stamped by whichever specialist produced it. `selected_agent` is now
    written by `decide_route` too, so the two normally agree — but the envelope
    is the record of what happened rather than of what was decided, and a retry
    must follow what happened. Retrying a different specialist than the one
    that failed is not a retry; it hands a job-match failure to an agent with no
    matcher, together with a failure context describing someone else's attempt.
    """
    outcome = state.get("reflect_outcome", "done")
    if outcome == "retry":
        envelope = state.get("task_result") or {}
        ran = envelope.get("agent") or state.get("current_agent")
        if ran in ("job", "email", "academic", "profile"):
            return ran

    if outcome in ("retry", "next_step"):
        selected = state.get("selected_agent", "profile")
        if selected in ("job", "email", "academic", "profile"):
            return selected
    return "response"


# ─────────────────────────────────────────────
# Build the workflow graph
# ─────────────────────────────────────────────
def create_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("job", job_node)
    workflow.add_node("email", email_node)
    workflow.add_node("academic", academic_node)
    workflow.add_node("profile", profile_node)
    workflow.add_node("response", response_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("temporal", temporal_node)
    workflow.add_node("degraded", degraded_node)
    workflow.add_node("provenance", provenance_node)
    workflow.add_node("confirm_action", confirm_action_node)

    if settings.parallel_workflow_enabled:
        workflow.add_node("parallel_init", parallel_init_node)
        workflow.set_entry_point("parallel_init")

        workflow.add_conditional_edges(
            "parallel_init",
            route_after_init,
            {
                "clarification": "clarification",
                "temporal": "temporal",
                "degraded": "degraded",
                "provenance": "provenance",
                "confirm_action": "confirm_action",
                "job": "job",
                "email": "email",
                "academic": "academic",
                "profile": "profile",
                "response": "response",
            }
        )
    else:
        workflow.add_node("memory", memory_node)
        workflow.add_node("planner", planner_node)
        workflow.set_entry_point("memory")
        workflow.add_edge("memory", "planner")

        workflow.add_conditional_edges(
            "planner",
            route_after_init,
            {
                "clarification": "clarification",
                "temporal": "temporal",
                "degraded": "degraded",
                "provenance": "provenance",
                "confirm_action": "confirm_action",
                "job": "job",
                "email": "email",
                "academic": "academic",
                "profile": "profile",
                "response": "response",
            }
        )

    # Clarification and the clock both bypass reflect — neither produced a task
    # there is anything to reflect on, and both already hold a final answer.
    workflow.add_edge("clarification", END)
    workflow.add_edge("temporal", END)
    workflow.add_edge("degraded", END)
    # Reads a record and formats it; there is nothing to reflect on.
    workflow.add_edge("provenance", END)
    # Already executed (or refused) a decided action — a reflect retry here
    # would be a second attempt at something the user approved once.
    workflow.add_edge("confirm_action", END)

    # All specialists → reflect
    workflow.add_edge("job", "reflect")
    workflow.add_edge("email", "reflect")
    workflow.add_edge("academic", "reflect")
    workflow.add_edge("profile", "reflect")

    # Reflect → retry specialist OR response
    workflow.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "job": "job",
            "email": "email",
            "academic": "academic",
            "profile": "profile",
            "response": "response",
        }
    )

    workflow.add_edge("response", END)

    return workflow.compile()


# Singleton workflow instance
multi_agent_workflow = create_workflow()


@traceable(name="run_workflow", run_type="chain", tags=["workflow", "entrypoint"])
async def run_workflow(
    user_input: str,
    user_id: str,
    session_id: str = None,
    conversation_history: list = None,
    output_mode: str = "user",
    scopes: Optional[FrozenSet[str]] = None,
    memory_owner_id: Optional[str] = None,
    memory_visibilities: Optional[list] = None,
    skip_user_ingest: bool = False,
    timeout_seconds: Optional[float] = None,
    prefetched: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute the multi-agent workflow.

    Args:
        scopes: Capabilities of the authenticated caller. Specialist agents
            filter their tools against this set, so a restricted caller cannot
            reach privileged tools by asking for them. None means unrestricted
            and must only be used for internal/CLI invocation.
        skip_user_ingest: Do not store the user's message again. Set only by a
            caller that has already ingested this exact utterance — the
            streaming workflow, when it escalates a turn *after* its own
            `parallel_init_node` has run. Retrieval still happens; what is
            suppressed is the write, so the thread does not end up holding the
            same sentence twice.
        timeout_seconds: Overall deadline for this run. Defaults to
            `settings.workflow_timeout_seconds`; the voice worker passes the
            shorter `voice_workflow_timeout_seconds` because a spoken turn's
            caller has stopped listening long before the typed ceiling.
        prefetched: Initialization results a caller has already paid for —
            memory context, prompt, category, routing. Supplied by the
            streaming workflow when it escalates a turn to tools, so the
            planner call and the whole retrieval fan-out are not repeated. See
            `parallel_init_node`.

    Returns:
        Final state with display_text and speech_text
    """
    if not user_id:
        raise ValueError("user_id is required for workflow execution")

    user_id = user_id.strip().lower()

    if not session_id:
        session_id = f"session_{user_id}"

    log_step("USER QUERY", {"user_id": user_id, "query": user_input})

    initial_state: AgentState = {
        "user_input": user_input,
        "user_id": user_id,
        "session_id": session_id,
        "output_mode": output_mode,
        "scopes": scopes,
        "memory_owner_id": memory_owner_id,
        "memory_visibilities": memory_visibilities,
        "conversation_history": conversation_history or [],
        "skip_user_ingest": skip_user_ingest,
        "prefetched": prefetched or None,
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
        # Fix 1: multi-step plan fields
        "execution_plan": [],
        "current_step_index": 0,
        "step_results": {},
        "inter_step_context": None,
        # Fix 2: reflect learning
        "reflect_failure_context": None,
        "display_text": None,
        "speech_text": None,
        "current_agent": None,
        "execution_path": [],
        "request_id": uuid.uuid4().hex,
        "error": None,
    }

    # Overall deadline. Reflect retries × reasoning iterations × per-call
    # timeouts can otherwise stack into several minutes of work continuing
    # server-side long after the client has given up waiting.
    deadline = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else settings.workflow_timeout_seconds
    )

    # Everything this turn costs, counted rather than estimated. The audit that
    # motivated the retry and retrieval work could only be written by reading
    # the code and multiplying the layers together by hand; these counters make
    # the same numbers observable per turn. See `app.services.call_metrics`.
    metrics_scope = call_metrics.turn(initial_state["request_id"])
    turn_metrics = metrics_scope.__enter__()
    try:
        final_state = await asyncio.wait_for(
            multi_agent_workflow.ainvoke(initial_state),
            timeout=deadline,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Workflow timed out after %.0fs (user=%s, request_id=%s)",
            deadline, user_id, initial_state["request_id"],
        )
        timeout_message = (
            "That request took too long to complete. "
            "Please try again, or break it into smaller steps."
        )
        # Closed here as well as on the success path: a timed-out turn is
        # exactly the one whose cost is worth reading, and leaving the scope
        # open would attribute the next turn's calls to this one.
        metrics_scope.__exit__(None, None, None)
        log_step("TURN_COST", turn_metrics.as_dict())
        logger.info("Turn cost (timed out): %s", turn_metrics.as_dict())
        return {
            **initial_state,
            "display_text": timeout_message,
            "speech_text": timeout_message,
            "error": "workflow_timeout",
        }
    else:
        metrics_scope.__exit__(None, None, None)
        log_step("TURN_COST", turn_metrics.as_dict())
        logger.info("Turn cost: %s", turn_metrics.as_dict())
    finally:
        # Idempotent: `_current.reset` has already run on both paths above.
        # This catches an unexpected exception escaping the graph, so the
        # context var is never left pointing at a finished turn.
        if call_metrics.current() is turn_metrics:
            metrics_scope.__exit__(None, None, None)

    # ── Who actually answered ────────────────────────────────────────────────
    # `selected_agent` holds the planner's choice, and the planner is routinely
    # overruled: `decide_route` picks the specialist from the deterministic
    # category, and its state writes never reach the nodes because a LangGraph
    # conditional-edge function returns a route rather than state. So a
    # JOB_MATCH turn ran the job agent while this field still said "profile" —
    # and that field is what the logs, the episode record and the client
    # metadata all report.
    #
    # The envelope knows: every specialist stamps its own name on the result it
    # produced, as do `temporal`, `provenance`, `degraded` and `confirm_action`.
    # Corrected after the graph rather than inside it, so nothing the routing or
    # reflect logic reads mid-run is disturbed.
    envelope_agent = (final_state.get("task_result") or {}).get("agent")
    if envelope_agent and envelope_agent != final_state.get("selected_agent"):
        logger.debug(
            "Reporting the agent that ran (%s) rather than the planner's "
            "choice (%s)", envelope_agent, final_state.get("selected_agent"),
        )
        final_state["selected_agent"] = envelope_agent

    # Save agent response to memory
    try:
        if final_state.get("display_text"):
            await memory_manager.on_agent_response(
                user_id=user_id,
                session_id=session_id,
                agent_response=final_state["display_text"],
                metadata={
                    "agent": final_state.get("selected_agent"),
                    "intent": final_state.get("detected_intent"),
                    "confidence": final_state.get("planner_confidence"),
                }
            )
            log_step("FINAL RESPONSE", {
                "request_id": final_state.get("request_id"),
                "agent": final_state.get("selected_agent"),
                "intent": final_state.get("detected_intent"),
                "confidence": final_state.get("planner_confidence"),
                "iterations": final_state.get("iteration_count", 0),
                "execution_path": final_state.get("execution_path"),
                "display_text": (final_state.get("display_text") or "")[:300],
            })
    except Exception as e:
        logger.error("Error saving agent response: %s", e)

    # Fix 10: Episode memory policy — write on EVERY turn (lightweight) + every 10th turn (rich LLM summary).
    # Previously: only every 10 turns → sessions with < 10 turns had ZERO cross-session context.
    # Now: every turn stores a lightweight episode (no LLM call, ~0ms overhead).
    #       Every 10th turn additionally stores a richer multi-turn LLM-summarized episode.
    try:
        if final_state.get("display_text"):
            turn_count = await memory_manager.get_session_turn_count(user_id, session_id)
            outcome = "failed" if final_state.get("error") else "success"

            # ── Lightweight episode — written every single turn ───────────────
            # Uses raw input/output directly; no extra LLM call; never skipped.
            await memory_manager.store_episode(
                user_id=user_id,
                session_id=session_id,
                user_summary=user_input[:150],
                agent_summary=(final_state.get("display_text") or "")[:150],
                agent_used=final_state.get("selected_agent"),
                intent=final_state.get("detected_intent"),
                outcome=outcome,
            )
            log_step("EPISODE_LIGHTWEIGHT", {"turn": turn_count, "user_id": user_id})

            # ── Rich episode — written every 10th turn (LLM-summarized) ──────
            # Condenses recent chat history into a single coherent paragraph so
            # future sessions have a high-quality multi-turn context snapshot.
            if turn_count > 0 and turn_count % 10 == 0:
                recent_history = (final_state.get("memory_context") or {}).get("chat_history", [])
                if recent_history:
                    lines = [
                        f"{'User' if m['role'] == 'user' else 'Agent'}: {m['content'][:120]}"
                        for m in recent_history[-10:]
                        if m.get("content")
                    ]
                    conversation_block = "\n".join(lines)
                    from app.services.groq_service import groq_service
                    summary_response = await groq_service.chat_completion(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Summarize the following conversation in 1-2 sentences each for "
                                    "the user's intent and the agent's outcome. "
                                    "Format: USER_SUMMARY: ... | AGENT_SUMMARY: ..."
                                ),
                            },
                            {"role": "user", "content": conversation_block},
                        ],
                        temperature=0.3,
                        max_tokens=120,
                    )
                    raw = summary_response.get("content", "")
                    user_summary = (
                        raw.split("USER_SUMMARY:")[-1].split("|")[0].strip()[:298]
                        if "USER_SUMMARY:" in raw else user_input[:150]
                    )
                    agent_summary = (
                        raw.split("AGENT_SUMMARY:")[-1].strip()[:298]
                        if "AGENT_SUMMARY:" in raw else (final_state.get("display_text") or "")[:150]
                    )
                else:
                    user_summary = user_input[:150]
                    agent_summary = (final_state.get("display_text") or "")[:150]

                await memory_manager.store_episode(
                    user_id=user_id,
                    session_id=session_id,
                    user_summary=user_summary,
                    agent_summary=agent_summary,
                    agent_used=final_state.get("selected_agent"),
                    intent=final_state.get("detected_intent"),
                    outcome=outcome,
                )
                log_step("EPISODE_RICH", {"turn": turn_count, "user_id": user_id})
    except Exception as e:
        logger.error("Error storing episode: %s", e)

    return final_state
