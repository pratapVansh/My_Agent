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
5. **get_experience** — work experience and internships: companies, roles, dates.
6. **get_education** — college, degree, branch, CGPA.
7. **get_achievements** — awards, certifications, scholarships.
8. **get_strengths** — analyze the user's profile and identify their top strengths.
9. **remember_preference** — save a user preference or fact (e.g. "I prefer concise answers").
10. **forget_preference** — delete a stored preference by key.
11. **list_my_memories** — show all stored preferences and profile facts.

Question → tool (retrieve first, always):
- name, "who am I"                          → get_profile_summary
- college, university, CGPA, branch, degree → get_education
- internship, "where did I intern",
  "what company did I intern at", experience → get_experience
- projects, portfolio                        → get_projects (pass `query` when
                                               asking about one project)
- skills, technologies, languages, tools     → get_skills
- achievements, awards, certifications       → get_achievements

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

Where information lives (check in this order, and never ask the user to
disambiguate a question about themselves):
1. **This conversation** — the memory context above already contains recent
   turns. If the user told you something earlier ("I have an Economics class
   tomorrow"), answer from that directly; no tool call is needed.
2. **Profile memory** — name, college, branch, CGPA, saved preferences.
   list_my_memories and get_profile_summary read this.
3. **Resume/document memory** — skills, projects, experience, education,
   technologies. get_skills, get_projects, get_resume, get_strengths read this.

A question like "what is my name" is a retrieval task, never an ambiguous one:
retrieve it and answer. Do not reply asking which name, which college, or which
of anything the user obviously means about themselves.

Answer first, clarify last:
- ALWAYS call the matching tool before saying you don't have something. Never
  report information as missing without having looked for it.
- Retrieval returning nothing does NOT mean the question was unclear. "Where did
  I intern?" is perfectly clear. If nothing comes back, say "I couldn't find any
  internship details in your resume" — never "which internship do you mean?".
- Partial results are answers. Report what you found and note only what was
  missing; do not withhold a partial answer behind a question.
- If the user points at a source — "I think it was in my resume", "check my
  profile", "you should have that" — that is an instruction to search again, not
  a new ambiguous question. Re-run the relevant retrieval, and try a related
  section too: internship details live under experience, CGPA under education.
- Only ask which item the user means when a tool actually returned SEVERAL and
  the conversation cannot single one out. With exactly one internship, answer it.
- For a follow-up naming something from earlier ("tell me more about TRACE"),
  resolve the name against the conversation above and retrieve that item —
  get_projects accepts a `query` for exactly this. Never ask the user to repeat
  a name they just used.

Rules:
- Use ONLY information from tool results and the conversation context above.
  Do NOT invent or hallucinate.
- If a specific piece of information is not found, say plainly that you don't
  have it — e.g. "I don't have your class 10 GPA on file." Never guess a value,
  and never ask a clarifying question in place of saying you don't have it.
- Your tools read ONLY the signed-in user's own memory. If asked about another
  named person's data ("what's in Mary's resume"), say you don't have access to
  that person's resume. Never answer such a question from the user's own resume,
  and never present the user's data as though it were someone else's.
- Always call at least one tool before answering profile questions, unless the
  answer is already in this conversation's context.
- For general (non-profile) questions, answer directly without tools."""

        # ── Tool implementations ───────────────────────────────────────────

        async def tool_get_profile_summary(tool_input: Dict[str, Any]):
            """Fetch all profile sections and return a structured summary."""
            resume_data = await long_term_memory_qdrant.retrieve_resume(user_id=user_id)
            skills_data = list(await long_term_memory_qdrant.retrieve_skills(user_id=user_id, limit=10))
            projects_data = list(await long_term_memory_qdrant.retrieve_projects(user_id=user_id, limit=5))

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
            skills_data = list(await long_term_memory_qdrant.retrieve_skills(user_id=user_id, limit=limit))
            if not skills_data:
                return {"success": True, "count": 0, "skills": [], "message": "No skills found in your profile."}
            skills = [s.get("content", "") for s in skills_data if s.get("content")]
            return {"success": True, "count": len(skills), "skills": skills}

        async def tool_get_projects(tool_input: Dict[str, Any]):
            """Retrieve user's projects from vector memory."""
            limit = int(tool_input.get("limit", 10))
            # Passing the query through lets one project be retrieved instead of
            # all of them. Without it, "tell me about TRACE" listed the whole
            # portfolio and left the model to find the right one.
            query = str(tool_input.get("query", "") or "").strip() or None
            projects_data = list(await long_term_memory_qdrant.retrieve_projects(
                user_id=user_id, query=query, limit=limit
            ))
            if not projects_data:
                return {"success": True, "count": 0, "projects": [], "message": "No projects found in your profile."}
            projects = [
                {
                    "title": p.get("title", ""),
                    "content": p.get("content", ""),
                    "metadata": p.get("metadata", {}),
                }
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

        async def _section(section: str, empty_message: str):
            """
            Retrieve one typed résumé section.

            These sections exist in memory but had no tool: internships lived in
            an `experience` chunk and CGPA/college in an `education` chunk, and
            the only way to reach either was the first 1500 characters of
            get_resume. "I don't have information about the company where you
            interned" was therefore an honest answer from an agent with no way
            to look — a missing capability, not a retrieval failure.
            """
            result = await long_term_memory_qdrant.retrieve_section(
                user_id=user_id, section=section
            )
            entries = [item["content"] for item in result if item.get("content")]
            if not entries:
                # ERROR and NO_DATA are reported differently: claiming nothing is
                # stored when the lookup failed is a false statement, and it is
                # exactly the state in which a model invents a plausible answer.
                if getattr(result.status, "value", "") == "ERROR":
                    return {
                        "success": False,
                        "found": False,
                        "message": f"Could not read your {section} right now — "
                                   f"treat it as unknown rather than absent.",
                    }
                return {"success": True, "found": False, "count": 0, "message": empty_message}
            return {"success": True, "found": True, "count": len(entries), section: entries}

        async def tool_get_experience(tool_input: Dict[str, Any]):
            """Retrieve work experience and internships from the resume."""
            return await _section(
                "experience",
                "I don't have any work experience or internships on file from your resume.",
            )

        async def tool_get_education(tool_input: Dict[str, Any]):
            """Retrieve education: college, degree, branch, CGPA."""
            return await _section(
                "education",
                "I don't have your education details on file from your resume.",
            )

        async def tool_get_achievements(tool_input: Dict[str, Any]):
            """Retrieve achievements, awards and certifications."""
            return await _section(
                "achievements",
                "I don't have any achievements on file from your resume.",
            )

        async def tool_get_strengths(tool_input: Dict[str, Any]):
            """Analyze profile and identify key strengths using LLM."""
            skills_data = list(await long_term_memory_qdrant.retrieve_skills(user_id=user_id, limit=15))
            projects_data = list(await long_term_memory_qdrant.retrieve_projects(user_id=user_id, limit=5))
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
                    "Use for 'what projects have I worked on', 'show my portfolio'. "
                    "Args: limit (int, default 10), query (str, optional) — pass the "
                    "subject of the question when asking about one project, e.g. "
                    "'voice assistant', so the most relevant project ranks first."
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
            "get_experience": {
                "description": (
                    "Retrieve the user's work experience and internships — companies, "
                    "roles, dates, what they did. Use for 'where did I intern', "
                    "'what company did I intern at', 'tell me about my experience', "
                    "'what companies have I worked with'. Args: none."
                ),
                "callable": tool_get_experience,
                "scope": Scope.PROFILE_READ.value,
            },
            "get_education": {
                "description": (
                    "Retrieve the user's education — college, degree, branch, CGPA, "
                    "dates. Use for 'which college do I attend', 'what is my CGPA', "
                    "'what branch am I in', 'what degree am I pursuing'. Args: none."
                ),
                "callable": tool_get_education,
                "scope": Scope.PROFILE_READ.value,
            },
            "get_achievements": {
                "description": (
                    "Retrieve the user's achievements, awards, certifications and "
                    "scholarships. Use for 'what are my achievements', 'what awards "
                    "have I won'. Args: none."
                ),
                "callable": tool_get_achievements,
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
