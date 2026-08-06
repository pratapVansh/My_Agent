"""
Profile Agent - Handles user profile queries and general assistance.
Returns a TaskEnvelope for structured coordination.
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.agents.state import make_envelope
from app.auth.models import Scope
from app.memory.long_term_memory_qdrant import long_term_memory_qdrant
from app.memory.memory_manager import memory_manager


class ProfileAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="profile",
            description="User profile, resume, skills, projects, and general assistance"
        )

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state["user_input"]
        intent = state.get("detected_intent", "")
        user_id = state.get("user_id", "")

        base_system_prompt = """You are a personal profile assistant. You help users understand their own professional profile.

You have structured tools to retrieve the user's resume, skills, and projects from memory.

Your capabilities:
1. **get_profile_summary** — fetch a complete overview: name, skills, projects, and experience.
2. **get_skills** — retrieve the user's technical skills and proficiency levels.
3. **get_projects** — retrieve the user's projects with descriptions and technologies.
4. **get_resume** — retrieve the user's full resume text.
5. **get_strengths** — analyze the user's profile and identify their top strengths.
6. **remember_preference** — save a user preference or fact (e.g. "I prefer concise answers").
7. **forget_preference** — delete a stored preference by key.
8. **list_my_memories** — show all stored preferences and profile facts.

Response templates (use these formats for common queries):

For "summarize my experience" or "who am I":
  Use get_profile_summary, then reply:
  "Here's your profile summary:
  **Name:** [name]
  **Key Skills:** [top 5 skills]
  **Projects:** [2-3 project names]
  **Experience:** [brief from resume]"

For "what are my strengths" or "what am I good at":
  Use get_strengths, then list 3-5 concrete strengths backed by evidence from skills/projects.

For "what projects have I worked on":
  Use get_projects, then list each project with name, description, and technologies.

For "what skills do I have" or "what technologies do I know":
  Use get_skills, then group by category if possible (e.g. Languages, Frameworks, Tools).

For "remember that I prefer X" or "save this preference":
  Extract the key (e.g. "preferred_tone") and value (e.g. "concise"), call remember_preference.
  Confirm: "Got it! I've saved that you prefer concise answers."

For "forget my preference for X" or "don't remember X":
  Call forget_preference with the key. Confirm deletion.

For "what do you know about me" or "show my preferences":
  Call list_my_memories, then present as a clean list.

Rules:
- Use ONLY information from tool results. Do NOT invent or hallucinate.
- If a specific piece of information is not found, say "I don't have that information in your profile."
- Always call at least one tool before answering profile questions.
- For general (non-profile) questions, answer directly without tools."""

        # ── Tool implementations ───────────────────────────────────────────

        def _safe_list(data) -> list:
            """Normalize retrieve_* results — handle 'NO_DATA' string sentinel."""
            if not data or data == "NO_DATA":
                return []
            return data if isinstance(data, list) else []

        async def tool_get_profile_summary(tool_input: Dict[str, Any]):
            """Fetch all profile sections and return a structured summary."""
            resume_data = await long_term_memory_qdrant.retrieve_resume(user_id=user_id)
            skills_data = _safe_list(await long_term_memory_qdrant.retrieve_skills(user_id=user_id, limit=10))
            projects_data = _safe_list(await long_term_memory_qdrant.retrieve_projects(user_id=user_id, limit=5))

            name = None
            resume_snippet = ""
            if resume_data:
                name = resume_data.get("name")
                content = resume_data.get("content", "")
                resume_snippet = content[:400] if content else ""

            skills_list = [s.get("content", "") for s in skills_data if s.get("content")]
            projects_list = [
                {
                    "content": p.get("content", ""),
                    "metadata": p.get("metadata", {}),
                }
                for p in projects_data if p.get("content")
            ]

            return {
                "success": True,
                "name": name,
                "resume_snippet": resume_snippet,
                "skills": skills_list[:8],
                "projects": projects_list[:5],
                "has_data": bool(resume_data or skills_list or projects_list),
            }

        async def tool_get_skills(tool_input: Dict[str, Any]):
            """Retrieve user's skills from vector memory."""
            limit = int(tool_input.get("limit", 15))
            skills_data = _safe_list(await long_term_memory_qdrant.retrieve_skills(user_id=user_id, limit=limit))
            if not skills_data:
                return {"success": True, "count": 0, "skills": [], "message": "No skills found in your profile."}
            skills = [s.get("content", "") for s in skills_data if s.get("content")]
            return {"success": True, "count": len(skills), "skills": skills}

        async def tool_get_projects(tool_input: Dict[str, Any]):
            """Retrieve user's projects from vector memory."""
            limit = int(tool_input.get("limit", 10))
            projects_data = _safe_list(await long_term_memory_qdrant.retrieve_projects(user_id=user_id, limit=limit))
            if not projects_data:
                return {"success": True, "count": 0, "projects": [], "message": "No projects found in your profile."}
            projects = [
                {"content": p.get("content", ""), "metadata": p.get("metadata", {})}
                for p in projects_data if p.get("content")
            ]
            return {"success": True, "count": len(projects), "projects": projects}

        async def tool_get_resume(tool_input: Dict[str, Any]):
            """Retrieve the user's full resume."""
            resume_data = await long_term_memory_qdrant.retrieve_resume(user_id=user_id)
            if not resume_data or not resume_data.get("content"):
                return {"success": True, "found": False, "message": "No resume found in your profile."}
            return {
                "success": True,
                "found": True,
                "name": resume_data.get("name"),
                "content": resume_data.get("content", "")[:1500],
            }

        async def tool_get_strengths(tool_input: Dict[str, Any]):
            """Analyze profile and identify key strengths using LLM."""
            skills_data = _safe_list(await long_term_memory_qdrant.retrieve_skills(user_id=user_id, limit=15))
            projects_data = _safe_list(await long_term_memory_qdrant.retrieve_projects(user_id=user_id, limit=5))
            resume_data = await long_term_memory_qdrant.retrieve_resume(user_id=user_id)

            skills_text = "\n".join(
                f"- {s.get('content', '')}" for s in skills_data if s.get("content")
            ) or "No skills data."
            projects_text = "\n".join(
                f"- {p.get('content', '')[:200]}" for p in projects_data if p.get("content")
            ) or "No projects data."
            resume_snippet = ""
            if resume_data and resume_data.get("content"):
                resume_snippet = resume_data["content"][:500]

            prompt = (
                "Based on the following profile data, identify 4–5 key professional strengths. "
                "Be specific and cite evidence from the data (e.g., 'Strong in Python — demonstrated across 3 projects').\n\n"
                f"Skills:\n{skills_text}\n\n"
                f"Projects:\n{projects_text}\n\n"
                f"Resume excerpt:\n{resume_snippet}\n\n"
                "List the strengths as bullet points. No preamble."
            )

            if not (skills_data or projects_data or resume_data):
                return {"success": False, "message": "No profile data found to analyze strengths."}

            strengths = await self.call_groq(
                messages=[
                    {"role": "system", "content": "You are a career coach. Analyze profile data and output strengths only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            return {"success": True, "strengths": strengths}

        async def tool_remember_preference(tool_input: Dict[str, Any]):
            """Save a user preference or fact to profile memory."""
            key = str(tool_input.get("key", "")).strip().lower().replace(" ", "_")
            value = str(tool_input.get("value", "")).strip()
            if not key or not value:
                return {"success": False, "reason": "Both 'key' and 'value' are required."}
            source = str(tool_input.get("source", "explicit"))
            confidence = float(tool_input.get("confidence", 1.0))
            record_id = await memory_manager.save_profile_fact(
                user_id=user_id,
                key=key,
                value=value,
                source=source,
                confidence=confidence,
                consent_level="explicit",
            )
            return {"success": True, "id": record_id, "key": key, "value": value,
                    "message": f"Saved: {key} = {value}"}

        async def tool_forget_preference(tool_input: Dict[str, Any]):
            """Delete a specific profile fact by key."""
            key = str(tool_input.get("key", "")).strip().lower().replace(" ", "_")
            if not key:
                return {"success": False, "reason": "'key' is required."}
            deleted = await memory_manager.forget_profile_fact(user_id=user_id, key=key)
            if deleted:
                return {"success": True, "message": f"Forgotten: {key}"}
            return {"success": False, "message": f"No memory found for key '{key}'."}

        async def tool_list_my_memories(tool_input: Dict[str, Any]):
            """Return all stored profile facts for this user."""
            facts = await memory_manager.get_profile_facts(user_id=user_id)
            if not facts:
                return {"success": True, "count": 0, "facts": [],
                        "message": "No preferences or profile facts saved yet."}
            return {"success": True, "count": len(facts), "facts": facts}

        tools = {
            "get_profile_summary": {
                "description": (
                    "Fetch a complete profile overview including name, skills, projects, and resume snippet. "
                    "Use for 'summarize my experience', 'who am I', 'tell me about myself'. Args: none."
                ),
                "callable": tool_get_profile_summary,
                "scope": Scope.PROFILE_READ.value,
            },
            "get_skills": {
                "description": (
                    "Retrieve the user's technical skills and technologies. "
                    "Use for 'what skills do I have', 'what technologies do I know'. Args: limit (int, default 15)."
                ),
                "callable": tool_get_skills,
                "scope": Scope.PROFILE_READ.value,
            },
            "get_projects": {
                "description": (
                    "Retrieve the user's projects with descriptions. "
                    "Use for 'what projects have I worked on', 'show my portfolio'. Args: limit (int, default 10)."
                ),
                "callable": tool_get_projects,
                "scope": Scope.PROFILE_READ.value,
            },
            "get_resume": {
                "description": (
                    "Retrieve the user's full resume text. "
                    "Use for 'show my resume', 'what's in my resume'. Args: none."
                ),
                "callable": tool_get_resume,
                "scope": Scope.PROFILE_READ.value,
            },
            "get_strengths": {
                "description": (
                    "Analyze the user's profile and identify their top professional strengths. "
                    "Use for 'what are my strengths', 'what am I good at', 'what are my strong points'. Args: none."
                ),
                "callable": tool_get_strengths,
                "scope": Scope.PROFILE_READ.value,
            },
            "remember_preference": {
                "description": (
                    "Save a user preference or fact to long-term profile memory. "
                    "Use when user says 'remember that I prefer X', 'save this', 'my name is X'. "
                    "Args: key (str, e.g. 'preferred_tone'), value (str, e.g. 'concise'), "
                    "source (str: explicit|inferred, default explicit), confidence (float 0–1, default 1.0)."
                ),
                "callable": tool_remember_preference,
                "scope": Scope.PROFILE_WRITE.value,
            },
            "forget_preference": {
                "description": (
                    "Delete a specific stored preference by key. "
                    "Use when user says 'forget my preference for X', 'don't remember X'. "
                    "Args: key (str)."
                ),
                "callable": tool_forget_preference,
                "scope": Scope.PROFILE_WRITE.value,
            },
            "list_my_memories": {
                "description": (
                    "List all stored profile facts and preferences. "
                    "Use for 'what do you know about me', 'show my preferences', 'what have I saved'. "
                    "Args: none."
                ),
                "callable": tool_list_my_memories,
                "scope": Scope.PROFILE_WRITE.value,
            },
        }

        try:
            loop_result = await self.execute_reasoning_loop(
                state=state,
                base_system_prompt=base_system_prompt,
                tools=tools,
                max_iterations=3,
            )

            final_answer = loop_result["final_answer"]
            tools_used = loop_result["tools_used"]

            confidence = self._compute_confidence(
                final_answer=final_answer,
                tools_used=tools_used,
                iterations=loop_result["iterations"],
                max_iterations=3,
                was_retry=bool(state.get("reflect_failure_context")),
            )
            status = "success" if final_answer else "failed"

            envelope = make_envelope(
                agent=self.name,
                goal=intent or user_input,
                inputs={"user_input": user_input, "intent": intent},
                result_content=final_answer,
                status=status,
                confidence=confidence,
                tools_used=tools_used,
                next_actions=["get_skills", "get_projects"] if not tools_used else [],
            )

            state["task_result"] = envelope
            state["agent_reasoning"] = (
                f"Profile query processed. intent={intent}, "
                f"iterations={loop_result['iterations']}, tools={tools_used}, "
                f"confidence={confidence:.2f}"
            )
            state["current_agent"] = self.name
            if state.get("execution_path") is not None:
                state["execution_path"].append(self.name)

            return state

        except Exception as e:
            envelope = make_envelope(
                agent=self.name,
                goal=intent or user_input,
                inputs={"user_input": user_input},
                result_content="I encountered an error processing your profile query.",
                status="failed",
                confidence=0.0,
            )
            state["task_result"] = envelope
            state["error"] = f"Profile agent error: {str(e)}"
            return state


# Singleton instance
profile_agent = ProfileAgent()
