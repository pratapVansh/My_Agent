"""
Email Agent - Handles email-related tasks.
Specializes in email composition, management, and organization.
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.tools.email_draft_tool import email_draft_tool


class EmailAgent(BaseAgent):
    """
    Email Agent handles:
    - Email composition and drafting
    - Email organization and categorization
    - Email response suggestions
    - Email management advice
    """

    def __init__(self):
        super().__init__(
            name="email",
            description="Email composition, management, and organization"
        )

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process email-related requests.

        Args:
            state: Workflow state with user_input and detected_intent

        Returns:
            Updated state with task_result
        """
        user_input = state["user_input"]
        intent = state.get("detected_intent", "")

        base_system_prompt = """You are an email management and composition agent.

Your capabilities:
- Draft professional emails for various purposes
- Suggest email organization strategies
- Help compose responses to emails
- Provide email etiquette guidance

Be professional, clear, and actionable in your responses.
Format emails properly with subject lines, greetings, body, and signatures when drafting."""

        user_id = state.get("user_id", "")

        async def tool_email_draft(tool_input: Dict[str, Any]):
            query = str(tool_input.get("query") or user_input)
            tone = str(tool_input.get("tone") or "professional")
            recipient_name = str(tool_input.get("recipient_name") or "")
            return await email_draft_tool.draft_email(
                user_id=user_id,
                query=query,
                tone=tone,
                recipient_name=recipient_name,
            )

        tools = {
            "email_draft": {
                "description": "Generate a structured personalized email draft.",
                "callable": tool_email_draft,
            }
        }

        try:
            loop_result = await self.execute_reasoning_loop(
                state=state,
                base_system_prompt=base_system_prompt,
                tools=tools,
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
                f"Processed email-related query: {intent}. "
                f"iterations={loop_result['iterations']}, tools={loop_result['tools_used']}"
            )
            state["current_agent"] = self.name

            if state.get("execution_path"):
                state["execution_path"].append(self.name)

            return state

        except Exception as e:
            state["error"] = f"Email agent error: {str(e)}"
            state["task_result"] = {
                "agent": self.name,
                "content": "I encountered an error processing your email request.",
                "success": False
            }
            return state


# Singleton instance
email_agent = EmailAgent()
