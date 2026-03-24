"""
Academic Agent - Handles academic and educational tasks.
Specializes in research, study assistance, and learning support.
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.tools.timetable_tool import timetable_tool


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

        user_id = state.get("user_id", "")

        async def tool_class_suggestions(tool_input: Dict[str, Any]):
            day_of_week = tool_input.get("day_of_week")
            low_attendance_threshold = float(tool_input.get("low_attendance_threshold", 75.0))
            return await timetable_tool.suggest_classes(
                user_id=user_id,
                day_of_week=day_of_week,
                low_attendance_threshold=low_attendance_threshold,
            )

        tools = {
            "class_suggestions": {
                "description": "Suggest classes using timetable and attendance context.",
                "callable": tool_class_suggestions,
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
                f"Processed academic query: {intent}. "
                f"iterations={loop_result['iterations']}, tools={loop_result['tools_used']}"
            )
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
