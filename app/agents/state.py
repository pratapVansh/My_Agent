"""
State schema for LangGraph workflow.
Defines the structure of data flowing through the multi-agent system.
"""
from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict):
    """State passed between agents in the workflow."""

    # Input
    user_input: str  # Original user query
    conversation_history: Optional[List[Dict[str, str]]]  # Chat history
    user_id: Optional[str]  # User identifier
    session_id: Optional[str]  # Session identifier
    output_mode: Optional[str]  # user or recruiter mode

    # Memory context
    memory_context: Optional[Dict[str, Any]]  # Retrieved memory context
    memory_prompt: Optional[str]  # Formatted memory for prompt injection

    # Planner outputs
    detected_intent: Optional[str]  # Intent classification
    selected_agent: Optional[str]  # Which task agent to use

    # Task agent outputs
    task_result: Optional[Dict[str, Any]]  # Result from task agent
    agent_reasoning: Optional[str]  # Agent's thought process

    # Response agent output
    display_text: Optional[str]  # Text for UI display
    speech_text: Optional[str]  # Text for voice output

    # Metadata
    current_agent: Optional[str]  # Current agent executing
    execution_path: Optional[List[str]]  # Track agent flow
    error: Optional[str]  # Error messages if any
