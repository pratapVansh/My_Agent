"""
Planner Agent - Detects intent, scores confidence, and routes to appropriate task agent.
If confidence < 0.6, raises a clarifying question instead of routing blindly.
"""
from typing import Dict, Any
import json
import logging
from app.agents.base_agent import BaseAgent
from app.services.langsmith_service import traceable

logger = logging.getLogger(__name__)

# Guidance given to the model: below this it should ask a clarifying question
# itself, via needs_clarification in its JSON response.
CONFIDENCE_THRESHOLD = 0.6

# Server-side safety net. Deliberately lower than CONFIDENCE_THRESHOLD: it only
# overrides the model when it reported very low confidence yet still asked to
# route, so ordinary 0.4–0.6 routing decisions are left to the model's judgment.
FORCE_CLARIFICATION_BELOW = 0.35

# How many prior turns the planner sees. Routing a follow-up like "email him
# about that" is impossible without recent context, and every specialist
# already receives history — the planner was the only component flying blind.
_PLANNER_HISTORY_TURNS = 4
_PLANNER_HISTORY_CHARS = 300


class PlannerAgent(BaseAgent):
    """
    Planner Agent responsible for:
    - Detecting user intent from input
    - Scoring routing confidence (0.0–1.0)
    - Routing to the appropriate task agent when confident
    - Asking a clarifying question when not confident enough
    """

    def __init__(self):
        super().__init__(
            name="planner",
            description="Detects intent and routes requests to specialized agents"
        )
        self.available_agents = {
            "job": "Job-related queries (search, applications, career advice)",
            "email": "Email management (compose, draft, organize)",
            "academic": "Academic tasks (timetable, attendance, study help, exams, daily plans, reminders, tasks)",
            "profile": "User profile, resume, skills, projects, general queries"
        }

    @traceable(name="planner_decision", run_type="chain", tags=["planner", "decision"])
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state["user_input"]

        system_prompt = self.inject_memory_context(self._build_system_prompt(), state)
        messages = [
            {"role": "system", "content": system_prompt},
            *self._recent_history_messages(state),
            {"role": "user", "content": user_input},
        ]

        try:
            response = await self.call_groq(
                messages=messages,
                temperature=0.2,
                max_tokens=500
            )

            routing = self._parse_response(response)

            state["detected_intent"] = routing["intent"]
            state["selected_agent"] = routing["agent"]
            state["planner_confidence"] = routing["confidence"]
            state["needs_clarification"] = routing["needs_clarification"]
            state["clarification_question"] = routing.get("clarification_question", "")
            state["current_agent"] = "planner"

            # Fix 1: store the execution plan from planner
            state["execution_plan"] = routing.get("execution_plan", [])
            state["current_step_index"] = 0
            state["step_results"] = {}
            state["inter_step_context"] = None

            if "execution_path" not in state or state["execution_path"] is None:
                state["execution_path"] = []
            state["execution_path"].append("planner")

            return state

        except Exception as e:
            logger.error("Planner execution failed, defaulting to profile agent: %s", e, exc_info=True)
            state["error"] = f"Planner error: {str(e)}"
            state["selected_agent"] = "profile"
            state["planner_confidence"] = 0.5
            state["needs_clarification"] = False
            state["execution_plan"] = []
            state["current_step_index"] = 0
            state["step_results"] = {}
            state["inter_step_context"] = None
            return state

    def _recent_history_messages(self, state: Dict[str, Any]) -> list:
        """Recent conversation turns, trimmed, for context-aware routing."""
        raw_history = state.get("conversation_history") or []
        return [
            {"role": turn["role"], "content": str(turn.get("content", ""))[:_PLANNER_HISTORY_CHARS]}
            for turn in raw_history[-_PLANNER_HISTORY_TURNS:]
            if turn.get("role") in ("user", "assistant") and turn.get("content")
        ]

    def _build_system_prompt(self) -> str:
        agents_desc = "\n".join([
            f"- {name}: {desc}"
            for name, desc in self.available_agents.items()
        ])

        return f"""You are a planning agent that decomposes user requests into an ordered execution plan.

Available agents:
{agents_desc}

Analyze the user query and respond with a JSON object:
{{
  "intent": "brief description of the overall goal",
  "agent": "first agent to run (job|email|academic|profile)",
  "confidence": 0.0-1.0,
  "needs_clarification": true|false,
  "clarification_question": "question if unclear, else empty string",
  "execution_plan": [
    {{"step": 1, "agent": "agent_name", "goal": "specific goal for this step"}},
    {{"step": 2, "agent": "agent_name", "goal": "specific goal using results from step 1"}}
  ]
}}

Rules for execution_plan:
- ALWAYS include at least 1 step, even for simple tasks.
- Add MORE steps when the user asks to do multiple distinct things in one message, e.g.:
  * "search for jobs AND draft a cover letter" → step1: job(search), step2: email(draft)
  * "find ML jobs in NYC then send me an email summary" → step1: job(search), step2: email(send summary)
  * "check my attendance and email my professor" → step1: academic(attendance), step2: email(draft to professor)
- Single-step examples (only 1 step needed):
  * "find software jobs" → step1: job(search)
  * "what's my timetable?" → step1: academic(timetable)
  * "draft an email to HR" → step1: email(draft)
- In multi-step plans, the goal for step N+1 should reference "results from step N" so the agent knows to use prior output.
- "agent" at the top level must equal the agent of step 1.

Confidence scoring:
- 0.9–1.0: Query clearly maps to a specific plan
- 0.7–0.89: Likely correct, minor ambiguity
- 0.6–0.69: Could be planned differently
- Below 0.6: Set needs_clarification=true

Additional rules:
- If needs_clarification is true, write a short natural clarifying question.
- If needs_clarification is false, clarification_question must be "".
- Default to "profile" for general personal questions.
- Respond ONLY with valid JSON, no other text.

Example — single step:
{{"intent": "find software jobs", "agent": "job", "confidence": 0.95, "needs_clarification": false, "clarification_question": "", "execution_plan": [{{"step": 1, "agent": "job", "goal": "Search for software engineering job listings"}}]}}

Example — multi step:
{{"intent": "search jobs and draft cover letter", "agent": "job", "confidence": 0.9, "needs_clarification": false, "clarification_question": "", "execution_plan": [{{"step": 1, "agent": "job", "goal": "Search for relevant job listings"}}, {{"step": 2, "agent": "email", "goal": "Draft a tailored cover letter using the job search results from step 1"}}]}}"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            clean = response.strip()
            if clean.startswith("```json"):
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif clean.startswith("```"):
                clean = clean.split("```")[1].split("```")[0].strip()

            parsed = json.loads(clean)

            agent = parsed.get("agent", "profile").lower()
            if agent not in self.available_agents:
                agent = "profile"

            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            needs_clarification = bool(parsed.get("needs_clarification", False))

            # Safety-net: only override for extreme low-confidence cases
            if not needs_clarification and confidence < FORCE_CLARIFICATION_BELOW:
                logger.info(
                    "Forcing clarification: planner reported confidence %.2f (below %.2f)",
                    confidence, FORCE_CLARIFICATION_BELOW,
                )
                needs_clarification = True

            clarification_question = str(parsed.get("clarification_question", "")).strip()
            if needs_clarification and not clarification_question:
                clarification_question = "Could you clarify what you need help with?"

            # Parse execution_plan — validate each step has required fields
            raw_plan = parsed.get("execution_plan", [])
            execution_plan = []
            if isinstance(raw_plan, list):
                for item in raw_plan:
                    if isinstance(item, dict):
                        step_agent = str(item.get("agent", "profile")).lower()
                        if step_agent not in self.available_agents:
                            step_agent = "profile"
                        execution_plan.append({
                            "step": int(item.get("step", len(execution_plan) + 1)),
                            "agent": step_agent,
                            "goal": str(item.get("goal", "")),
                        })

            # Always ensure at least one step exists
            if not execution_plan:
                execution_plan = [{"step": 1, "agent": agent, "goal": parsed.get("intent", "complete the user request")}]

            # Ensure `agent` matches the first step's agent
            agent = execution_plan[0]["agent"]

            return {
                "intent": parsed.get("intent", "general query"),
                "agent": agent,
                "confidence": confidence,
                "needs_clarification": needs_clarification,
                "clarification_question": clarification_question,
                "execution_plan": execution_plan,
            }

        except Exception as exc:
            # Logged (not silent) so a drift in the routing model's output
            # format shows up as misrouting warnings instead of an invisible
            # slide into always choosing the default agent.
            logger.warning(
                "Planner returned unparseable routing output (%s); defaulting to profile. "
                "First 200 chars: %.200s",
                exc, response,
            )
            return {
                "intent": "general query",
                "agent": "profile",
                "confidence": 0.5,
                "needs_clarification": False,
                "clarification_question": "",
                "execution_plan": [{"step": 1, "agent": "profile", "goal": "handle the user request"}],
            }


# Singleton instance
planner_agent = PlannerAgent()
