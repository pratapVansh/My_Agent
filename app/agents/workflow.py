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
from app.agents import clarification_policy, profile_intent, query_intent
from app.agents.state import AgentState
from app.agents.planner_agent import planner_agent
from app.agents.job_agent import job_agent
from app.agents.email_agent import email_agent
from app.agents.academic_agent import academic_agent
from app.agents.profile_agent import profile_agent
from app.agents.response_agent import response_agent
from app.memory.memory_manager import memory_manager
from app.memory.sources import QueryCategory
from app.services.debug_logger import log_step
from app.services.langsmith_service import traceable
from app.config import settings

MAX_ITERATIONS = 3


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

    category = _precategorise(state)

    try:
        async def memory_task():
            try:
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
            try:
                return await planner_agent.execute(state)
            except Exception as e:
                logger.error("Planner error: %s", e)
                return state

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

    return state


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
    return await planner_agent.execute(state)


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
    return await response_agent.execute(state)


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

    log_step("REFLECT", {
        "status": status,
        "confidence": confidence,
        "iteration": iteration_count,
        "max": MAX_ITERATIONS,
    })

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


def decide_route(state: AgentState) -> str:
    """
    The single routing decision, shared by the graph and the streaming path.

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
            return "temporal"

        if decision.category in _DEGRADED_ANSWERABLE:
            logger.info(
                "Answering %s from stored records in degraded mode: %s",
                decision.category.value, state.get("error"),
            )
            state["degraded_reason"] = str(state.get("error"))
            state["error"] = None
            return "degraded"

        return "response"

    state["query_category"] = decision.category.value
    state["memory_sources"] = [source.value for source in decision.sources]
    state["followup_subject"] = decision.subject
    if decision.profile_intent:
        # Kept under the legacy key: the profile agent's tool mapping and the
        # existing routing tests both read it.
        state["profile_intent"] = decision.profile_intent

    # A standing request for fewer questions is recorded before any decision is
    # made, so it applies to the very turn it was expressed in.
    clarification_policy.note_user_message(conversation_id, query)

    selected = query_intent.agent_for(decision, state.get("selected_agent"))

    log_step("memory_query", {
        "query_chars": len(query),
        **decision.summary(),
        "retrieval": decision.requires_retrieval,
        "agent": selected,
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
            return "clarification"

        log_step("CLARIFICATION SUPPRESSED", {
            **verdict.summary(),
            "intent": decision.category.value,
            "agent": selected,
            "confidence": state.get("planner_confidence", 0.0),
        })
        state["needs_clarification"] = False
        state["clarification_question"] = ""

    return selected


def route_after_init(
    state: AgentState,
) -> Literal["clarification", "temporal", "degraded", "job", "email", "academic", "profile", "response"]:
    """Graph edge for the routing decision. See `decide_route`."""
    route = decide_route(state)
    if route not in (
        "clarification", "temporal", "degraded", "job", "email", "academic",
        "profile", "response",
    ):
        return "profile"
    return route


def route_after_reflect(state: AgentState) -> Literal["job", "email", "academic", "profile", "response"]:
    """
    Route from reflect:
    - "retry"     → same specialist (failure context injected, Fix 2)
    - "next_step" → next specialist in execution plan (Fix 1)
    - "done"      → response agent
    """
    outcome = state.get("reflect_outcome", "done")
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
) -> Dict[str, Any]:
    """
    Execute the multi-agent workflow.

    Args:
        scopes: Capabilities of the authenticated caller. Specialist agents
            filter their tools against this set, so a restricted caller cannot
            reach privileged tools by asking for them. None means unrestricted
            and must only be used for internal/CLI invocation.

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
    try:
        final_state = await asyncio.wait_for(
            multi_agent_workflow.ainvoke(initial_state),
            timeout=settings.workflow_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Workflow timed out after %.0fs (user=%s, request_id=%s)",
            settings.workflow_timeout_seconds, user_id, initial_state["request_id"],
        )
        timeout_message = (
            "That request took too long to complete. "
            "Please try again, or break it into smaller steps."
        )
        return {
            **initial_state,
            "display_text": timeout_message,
            "speech_text": timeout_message,
            "error": "workflow_timeout",
        }

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
