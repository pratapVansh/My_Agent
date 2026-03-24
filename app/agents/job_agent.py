"""
Job Agent - Handles job-related queries.
Specializes in job search, applications, and career advice.
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent


class JobAgent(BaseAgent):
    """
    Job Agent handles:
    - Job search queries
    - Application assistance
    - Career advice
    - Resume/interview preparation
    """

    def __init__(self):
        super().__init__(
            name="job",
            description="Job search, applications, and career guidance"
        )

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process job-related requests with memory context.

        Args:
            state: Workflow state with user_input, detected_intent, and memory_prompt

        Returns:
            Updated state with task_result
        """
        user_input = state["user_input"]
        intent = state.get("detected_intent", "")

        base_system_prompt = """You are a job search and career advisor assistant.

Your capabilities:
- Help users search and find relevant jobs
- Provide application guidance and tips
- Offer career advice and development suggestions
- Assist with resume/interview preparation

Provide practical, actionable advice tailored to the user's query.
Be concise but comprehensive."""

        # Inject memory context
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
            state["agent_reasoning"] = f"Processed job-related query: {intent}"
            state["current_agent"] = self.name

            if state.get("execution_path"):
                state["execution_path"].append(self.name)

            return state

        except Exception as e:
            state["error"] = f"Job agent error: {str(e)}"
            state["task_result"] = {
                "agent": self.name,
                "content": "I encountered an error processing your job-related query.",
                "success": False
            }
            return state


# Singleton instance
job_agent = JobAgent()
