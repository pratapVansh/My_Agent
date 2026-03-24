"""
Academic Agent - Handles academic and educational tasks.
Specializes in research, study assistance, and learning support.
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent


class AcademicAgent(BaseAgent):
    """
    Academic Agent handles:
    - Research assistance
    - Study help and learning strategies
    - Assignment guidance
    - Academic writing support
    """

    def __init__(self):
        super().__init__(
            name="academic",
            description="Academic research, study help, and learning support"
        )

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process academic-related requests.

        Args:
            state: Workflow state with user_input and detected_intent

        Returns:
            Updated state with task_result
        """
        user_input = state["user_input"]
        intent = state.get("detected_intent", "")

        base_system_prompt = """You are an academic and educational assistant.

Your capabilities:
- Help with research and information gathering
- Provide study strategies and learning techniques
- Offer guidance on assignments and projects
- Assist with academic writing and citations
- Explain complex concepts clearly

Focus on helping users learn and understand rather than just providing answers.
Be educational, supportive, and encouraging."""

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
            state["agent_reasoning"] = f"Processed academic query: {intent}"
            state["current_agent"] = self.name

            if state.get("execution_path"):
                state["execution_path"].append(self.name)

            return state

        except Exception as e:
            state["error"] = f"Academic agent error: {str(e)}"
            state["task_result"] = {
                "agent": self.name,
                "content": "I encountered an error processing your academic query.",
                "success": False
            }
            return state


# Singleton instance
academic_agent = AcademicAgent()
