"""
Streaming workflow for low-latency voice responses.
Streams LLM tokens as they arrive instead of waiting for full response.

Fix 7: This module is now wired to the /stream WebSocket route in agent_routes.py.
The workflow skips tool-calling (trades completeness for sub-second first-token latency)
which is the right trade-off for voice/streaming UI contexts.
"""
import uuid
import logging
from typing import Dict, Any, AsyncGenerator
from app.agents.workflow import memory_node, planner_node
from app.agents.job_agent import job_agent
from app.agents.email_agent import email_agent
from app.agents.academic_agent import academic_agent
from app.agents.profile_agent import profile_agent
from app.services.groq_service import groq_service

logger = logging.getLogger(__name__)


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
    Get appropriate system prompt for each agent type with memory context.

    Args:
        agent: Agent instance
        state: Workflow state with memory

    Returns:
        System prompt with memory context injected
    """
    # Agent-specific prompts
    agent_prompts = {
        "job": """You are a job search and career advisor assistant.

Your capabilities:
- Help users search and find relevant jobs
- Provide application guidance and tips
- Offer career advice and development suggestions
- Assist with resume/interview preparation

Provide practical, actionable advice tailored to the user's query.
Be concise but comprehensive.""",

        "email": """You are an email management and composition assistant.

Your capabilities:
- Draft professional emails
- Manage email organization
- Compose responses and follow-ups
- Schedule meetings via email

Write clear, professional, context-appropriate emails.""",

        "academic": """You are an academic tracking and planning assistant.

Your capabilities:
- Track attendance records
- Manage timetables and schedules
- Provide academic planning advice
- Help with course management

Provide accurate, helpful academic guidance.""",

        "profile": """You are a profile management and general assistance agent.

Your capabilities:
- Help with user profile information
- Manage preferences and settings
- Handle general queries that don't fit other categories
- Provide friendly, helpful responses

Be helpful, conversational, and adapt to the user's needs.
For general queries, provide useful information or assistance."""
    }

    # Get base prompt for this agent
    base_prompt = agent_prompts.get(agent.name, agent.description)

    # Inject memory context
    return agent.inject_memory_context(base_prompt, state)


async def run_streaming_workflow(
    user_input: str,
    user_id: str = None,
    session_id: str = None,
    conversation_history: list = None,
    output_mode: str = "user"
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

    # ── Step 1: Memory retrieval ─────────────────────────────────────────────
    try:
        state = await memory_node(state)
    except Exception as e:
        logger.warning("Streaming workflow memory_node failed: %s", e)

    # ── Step 2: Intent detection & routing ───────────────────────────────────
    try:
        state = await planner_node(state)
    except Exception as e:
        logger.warning("Streaming workflow planner_node failed: %s", e)
        state["selected_agent"] = "profile"
        state["detected_intent"] = user_input

    # Handle clarification — emit as a complete event immediately
    if state.get("needs_clarification"):
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

    try:
        async for token in groq_service.stream_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        ):
            accumulated_text += token
            token_index += 1
            yield {
                "type": "token",
                "token": token,
                "index": token_index,
                "accumulated": accumulated_text,
            }

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
