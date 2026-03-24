"""
Job Agent - Handles job-related queries.
Specializes in job search, applications, and career advice.
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.tools.job_search_tool import job_search_tool


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

        user_id = state.get("user_id", "")

        async def tool_job_search(tool_input: Dict[str, Any]):
            query = str(tool_input.get("query") or user_input)
            location = tool_input.get("location")
            max_results = int(tool_input.get("max_results", 6))
            min_score = float(tool_input.get("min_score", 0.2))
            return await job_search_tool.search_jobs(
                user_id=user_id,
                query=query,
                location=location,
                max_results=max_results,
                min_score=min_score,
            )

        tools = {
            "job_search": {
                "description": "Search current job opportunities and rank them.",
                "callable": tool_job_search,
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
                f"Processed job-related query: {intent}. "
                f"iterations={loop_result['iterations']}, tools={loop_result['tools_used']}"
            )
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
