"""
Planner Agent - Detects intent and routes to appropriate task agent.
Controls the workflow by analyzing user input and selecting the right specialist.
"""
from typing import Dict, Any
import json
from app.agents.base_agent import BaseAgent
from app.services.langsmith_service import traceable


class PlannerAgent(BaseAgent):
    """
    Planner Agent responsible for:
    - Detecting user intent from input
    - Routing to the appropriate task agent
    - Controlling multi-step execution flow
    """

    def __init__(self):
        super().__init__(
            name="planner",
            description="Detects intent and routes requests to specialized agents"
        )
        self.available_agents = {
            "job": "Job-related queries (search, applications, career advice)",
            "email": "Email management (compose, send, organize)",
            "academic": "Academic tasks (research, study help, assignments)",
            "profile": "User profile operations (view, update, preferences)"
        }

    @traceable(name="planner_decision", run_type="chain", tags=["planner", "decision"])
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user input and determine routing.

        Args:
            state: Current workflow state with user_input

        Returns:
            Updated state with detected_intent and selected_agent
        """
        user_input = state["user_input"]

        # Build prompt for intent detection
        system_prompt = self.inject_memory_context(self._build_system_prompt(), state)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        try:
            # Call Groq with low latency settings
            response = await self.call_groq(
                messages=messages,
                temperature=0.3,  # Lower temp for more consistent routing
                max_tokens=256  # Small response for fast routing
            )

            # Parse response
            routing_decision = self._parse_response(response)

            # Update state
            state["detected_intent"] = routing_decision["intent"]
            state["selected_agent"] = routing_decision["agent"]
            state["current_agent"] = "planner"

            if "execution_path" not in state or state["execution_path"] is None:
                state["execution_path"] = []
            state["execution_path"].append("planner")

            return state

        except Exception as e:
            state["error"] = f"Planner error: {str(e)}"
            state["selected_agent"] = "response"  # Fallback to response
            return state

    def _build_system_prompt(self) -> str:
        """Build system prompt with available agents and routing rules."""
        agents_desc = "\n".join([
            f"- {name}: {desc}"
            for name, desc in self.available_agents.items()
        ])

        return f"""You are a routing agent that analyzes user queries and selects the best specialist agent.

Available agents:
{agents_desc}

Analyze the user's query and respond with a JSON object containing:
1. "intent": A brief description of what the user wants
2. "agent": The name of the best agent to handle this (job/email/academic/profile)

Rules:
- Choose exactly ONE agent
- Base decision on query content and context
- If unclear, default to "profile" for general queries
- Respond ONLY with valid JSON

Example response:
{{"intent": "user wants to search for jobs", "agent": "job"}}"""

    def _parse_response(self, response: str) -> Dict[str, str]:
        """
        Parse LLM response to extract routing decision.

        Args:
            response: LLM output

        Returns:
            Dictionary with intent and agent
        """
        try:
            # Try to parse as JSON
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean.split("```json")[1].split("```")[0].strip()
            elif response_clean.startswith("```"):
                response_clean = response_clean.split("```")[1].split("```")[0].strip()

            parsed = json.loads(response_clean)

            # Validate agent selection
            agent = parsed.get("agent", "profile").lower()
            if agent not in self.available_agents:
                agent = "profile"  # Default fallback

            return {
                "intent": parsed.get("intent", "general query"),
                "agent": agent
            }
        except Exception:
            # Fallback parsing if JSON fails
            return {
                "intent": "general query",
                "agent": "profile"
            }


# Singleton instance
planner_agent = PlannerAgent()
