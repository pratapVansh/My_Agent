"""
Streaming workflow for low-latency voice responses.
Streams LLM tokens as they arrive instead of waiting for full response.
"""
from typing import Dict, Any, AsyncGenerator
from app.agents.workflow import memory_node, planner_node
from app.agents.job_agent import job_agent
from app.agents.email_agent import email_agent
from app.agents.academic_agent import academic_agent
from app.agents.profile_agent import profile_agent
from app.services.groq_service import groq_service


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
    Run workflow with streaming response generation.

    Yields dictionary chunks with types:
    - "metadata": Agent selection and routing info (sent first)
    - "token": Individual text tokens as they arrive
    - "complete": Final state with full text

    Args:
        user_input: User query
        user_id: User identifier
        session_id: Session identifier
        conversation_history: Previous messages
        output_mode: Output formatting mode

    Yields:
        Chunks of the workflow execution
    """
    # Initialize state
    state = {
        "user_input": user_input,
        "user_id": user_id,
        "session_id": session_id,
        "conversation_history": conversation_history or [],
        "output_mode": output_mode,
        "execution_path": []
    }

    # Step 1: Memory retrieval (parallel with planning)
    state = await memory_node(state)

    # Step 2: Plan and route
    state = await planner_node(state)

    # Send metadata early (agent selection, execution path)
    yield {
        "type": "metadata",
        "selected_agent": state.get("selected_agent"),
        "detected_intent": state.get("detected_intent"),
        "execution_path": state.get("execution_path", [])
    }

    # Step 3: Get the selected agent
    agent = get_agent_by_name(state.get("selected_agent", "profile"))

    # Step 4: Stream the agent's response
    if not agent:
        yield {
            "type": "complete",
            "display_text": "I couldn't process your request.",
            "speech_text": "I couldn't process your request.",
            "error": "Agent not found"
        }
        return

    # Build system prompt with memory injection
    system_prompt = _get_agent_system_prompt(agent, state)

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Intent: {state.get('detected_intent', 'general')}\n\nQuery: {user_input}"}
    ]

    # Stream the response
    accumulated_text = ""

    try:
        async for token in groq_service.stream_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        ):
            accumulated_text += token

            # Yield each token immediately
            yield {
                "type": "token",
                "token": token,
                "accumulated": accumulated_text
            }

        # Send final complete response
        yield {
            "type": "complete",
            "display_text": accumulated_text,
            "speech_text": accumulated_text,
            "agent": agent.name,
            "success": True
        }

    except Exception as e:
        yield {
            "type": "complete",
            "display_text": "I encountered an error processing your request.",
            "speech_text": "I encountered an error.",
            "error": str(e),
            "success": False
        }
