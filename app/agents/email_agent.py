"""
Email Agent - Handles email-related tasks.
Specializes in email composition, management, and organization.
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent


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

        system_prompt = self.inject_memory_context(base_system_prompt, state)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Intent: {intent}\n\nQuery: {user_input}"}
        ]

        try:
            response = await self.call_groq(
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )

            state["task_result"] = {
                "agent": self.name,
                "content": response,
                "success": True
            }
            state["agent_reasoning"] = f"Processed email-related query: {intent}"
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
