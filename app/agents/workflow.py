"""
LangGraph workflow with Phase 1 upgrades:
- Confidence-gated routing (clarification node when planner confidence < 0.6)
- Plan-Execute-Reflect loop (max 3 iterations, retries failed tasks)
- Structured TaskEnvelope flowing through all agents
"""
import logging
import uuid
from typing import Dict, Any, Literal
import asyncio

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
from app.agents.state import AgentState
from app.agents.planner_agent import planner_agent
from app.agents.job_agent import job_agent
from app.agents.email_agent import email_agent
from app.agents.academic_agent import academic_agent
from app.agents.profile_agent import profile_agent
from app.agents.response_agent import response_agent
from app.memory.memory_manager import memory_manager
from app.services.debug_logger import log_step
from app.services.langsmith_service import traceable
from app.config import settings

MAX_ITERATIONS = 3


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

    try:
        await memory_manager.on_user_input(
            user_id=user_id, session_id=session_id, user_message=user_input
        )
        memory_context = await memory_manager.retrieve_context(
            user_id=user_id, session_id=session_id, query=user_input
        )
        state["memory_context"] = memory_context
        state["memory_prompt"] = memory_manager.format_context_for_prompt(memory_context)
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

    try:
        async def memory_task():
            try:
                await memory_manager.on_user_input(
                    user_id=user_id, session_id=session_id, user_message=user_input
                )
                ctx = await memory_manager.retrieve_context(
                    user_id=user_id, session_id=session_id, query=user_input
                )
                return ctx, memory_manager.format_context_for_prompt(ctx)
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
def route_after_init(state: AgentState) -> Literal["clarification", "job", "email", "academic", "profile", "response"]:
    """Route from parallel_init: clarify OR dispatch to specialist."""
    if state.get("error"):
        return "response"
    if state.get("needs_clarification"):
        return "clarification"
    selected = state.get("selected_agent", "profile")
    if selected in ("job", "email", "academic", "profile"):
        return selected
    return "profile"


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

    if settings.parallel_workflow_enabled:
        workflow.add_node("parallel_init", parallel_init_node)
        workflow.set_entry_point("parallel_init")

        workflow.add_conditional_edges(
            "parallel_init",
            route_after_init,
            {
                "clarification": "clarification",
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
                "job": "job",
                "email": "email",
                "academic": "academic",
                "profile": "profile",
                "response": "response",
            }
        )

    # Clarification bypasses reflect — goes directly to END
    workflow.add_edge("clarification", END)

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
    output_mode: str = "user"
) -> Dict[str, Any]:
    """
    Execute the multi-agent workflow.

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
        "conversation_history": conversation_history or [],
        "memory_context": None,
        "memory_prompt": None,
        "detected_intent": None,
        "selected_agent": None,
        "planner_confidence": None,
        "needs_clarification": None,
        "clarification_question": None,
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

    final_state = await multi_agent_workflow.ainvoke(initial_state)

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
