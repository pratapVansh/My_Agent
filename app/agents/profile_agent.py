"""
Profile Agent - Handles user profile and preference operations.
Specializes in profile management and personalization.
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent


class ProfileAgent(BaseAgent):
    """
    Profile Agent handles:
    - User profile queries
    - Preference management
    - Personal information requests
    - General queries (default fallback)
    """

    def __init__(self):
        super().__init__(
            name="profile",
            description="User profile operations and general assistance"
        )

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process profile-related and general requests.

        Args:
            state: Workflow state with user_input and detected_intent

        Returns:
            Updated state with task_result
        """
        user_input = state["user_input"]
        intent = state.get("detected_intent", "")

        base_system_prompt = """You are a profile management and general assistance agent.

Your capabilities:
- Help with user profile information
- Manage preferences and settings
- Handle general queries that don't fit other categories
- Provide friendly, helpful responses

Be helpful, conversational, and adapt to the user's needs.
For general queries, provide useful information or assistance."""

        try:
            loop_result = await self.execute_reasoning_loop(
                state=state,
                base_system_prompt=base_system_prompt,
                tools={},
                max_iterations=3,
            )

            state["task_result"] = {
                "agent": self.name,
                "content": loop_result["final_answer"],
                "success": True,
                "tools_used": loop_result["tools_used"],
                "reasoning_trace": loop_result["trace"],
            }
            state["agent_reasoning"] = (
                f"Processed profile/general query: {intent}. "
                f"iterations={loop_result['iterations']}, tools={loop_result['tools_used']}"
            )
            state["current_agent"] = self.name

            if state.get("execution_path"):
                state["execution_path"].append(self.name)

            return state

        except Exception as e:
            state["error"] = f"Profile agent error: {str(e)}"
            state["task_result"] = {
                "agent": self.name,
                "content": "I encountered an error processing your query.",
                "success": False
            }
            return state


# Singleton instance
profile_agent = ProfileAgent()
