"""
Job Agent - Phase 2 upgrade.
New capabilities:
- Skills-aware job search (matches against user's Qdrant skills)
- Bookmark jobs to PostgreSQL (save_job_bookmark tool)
- View saved jobs (get_bookmarked_jobs tool)
- Deduplication: already-bookmarked URLs are skipped in search results
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.agents.state import make_envelope
from app.auth.models import Scope
from app.tools.job_search_tool import job_search_tool
from app.tools.email_draft_tool import email_draft_tool
from app.memory.short_term_memory import short_term_memory
from app.memory.memory_manager import memory_manager
from app.domain.email import email_repository
from app.domain.jobs import jobs_repository


class JobAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="job",
            description="Job search, bookmarking, and career guidance — skills-aware"
        )

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state["user_input"]
        intent = state.get("detected_intent", "")
        user_id = state.get("user_id", "")

        base_system_prompt = """You are a job search and career advisor assistant with access to the user's skills profile.

Your capabilities:
1. **Search jobs** — use job_search to find relevant openings. Results include skills matched and exclude already-bookmarked jobs.
2. **Bookmark a job** — use save_job_bookmark to save a job. Always bookmark the top result automatically unless told not to.
3. **Show saved jobs** — use get_bookmarked_jobs to list previously saved listings.
4. **Draft application email** — use draft_application_email after finding a job to write a personalized application email.

When presenting job results: show title, company, match score, and skills matched. List top 3–5.
When the user says "apply" or "write an email for this job", call draft_application_email with the job details.
For career advice (no tool needed): give practical guidance based on the user's profile in memory."""

        # ── Tool definitions ──────────────────────────────────────────────
        # Use a dict so the nested closure can update the reference without
        # needing nonlocal or in-place mutation of a bare list.
        ctx: Dict[str, Any] = {"job_results": []}

        async def tool_job_search(tool_input: Dict[str, Any]):
            query = str(tool_input.get("query") or user_input)
            location = tool_input.get("location")
            max_results = int(tool_input.get("max_results", 6))
            min_score = float(tool_input.get("min_score", 0.2))
            result = await job_search_tool.search_jobs(
                user_id=user_id,
                query=query,
                location=location,
                max_results=max_results,
                min_score=min_score,
            )
            # Capture raw results for rich frontend cards
            if result.get("results"):
                ctx["job_results"] = result["results"][:5]
            return result

        async def tool_save_bookmark(tool_input: Dict[str, Any]):
            title = str(tool_input.get("title", "Unknown Job"))
            url = str(tool_input.get("url", ""))
            if not url:
                return {"success": False, "reason": "url is required"}
            result = await jobs_repository.save_bookmark(
                user_id=user_id,
                title=title,
                url=url,
                company=tool_input.get("company"),
                snippet=tool_input.get("snippet"),
                rank_score=float(tool_input.get("rank_score", 0.0)),
                search_query=tool_input.get("search_query"),
                skills_matched=tool_input.get("skills_matched", []),
            )
            if result == "already_saved":
                return {"success": True, "status": "already_saved", "title": title}
            return {"success": True, "status": "saved", "id": result, "title": title}

        async def tool_get_bookmarks(tool_input: Dict[str, Any]):
            limit = int(tool_input.get("limit", 10))
            bookmarks = await jobs_repository.get_bookmarks(user_id=user_id, limit=limit)
            if not bookmarks:
                return {"success": True, "count": 0, "bookmarks": [], "message": "No saved jobs yet."}
            return {"success": True, "count": len(bookmarks), "bookmarks": bookmarks}

        async def tool_draft_application_email(tool_input: Dict[str, Any]):
            job_title = str(tool_input.get("job_title", ""))
            company = str(tool_input.get("company", ""))
            job_url = str(tool_input.get("job_url", ""))
            job_description = str(tool_input.get("job_description", ""))
            recipient = f"Hiring Manager at {company}" if company else "Hiring Manager"
            query = (
                f"Write a professional job application email for the position of "
                f"'{job_title}' at '{company}'. "
                f"Job URL: {job_url}. "
                f"Job description: {job_description}"
            )
            draft_result = await email_draft_tool.draft_email(
                user_id=user_id,
                query=query,
                tone="professional",
                recipient_name=recipient,
            )
            # Auto-save the draft to PostgreSQL
            draft = draft_result.get("draft", {})
            try:
                draft_id = await email_repository.save_draft(
                    user_id=user_id,
                    subject=draft.get("subject", f"Application for {job_title}"),
                    body=draft.get("body", ""),
                    recipient_name=recipient,
                    tone="professional",
                    greeting=draft.get("greeting"),
                    closing=draft.get("closing"),
                    signature=draft.get("signature"),
                    context={"job_title": job_title, "company": company, "job_url": job_url},
                )
                draft_result["draft_id"] = draft_id
                draft_result["saved"] = True
            except Exception:
                draft_result["saved"] = False
            return draft_result

        tools = {
            "job_search": {
                "description": (
                    "Search current job openings. Automatically uses the user's skills to "
                    "find better matches and skips already-bookmarked jobs. "
                    "Args: query (str), location (str, optional), max_results (int, default 6)."
                ),
                "callable": tool_job_search,
                "scope": Scope.JOBS_SEARCH.value,
            },
            "save_job_bookmark": {
                "description": (
                    "Save a job to the user's bookmarks. "
                    "Args: title (str), url (str), company (str), snippet (str), "
                    "rank_score (float), skills_matched (list of str)."
                ),
                "callable": tool_save_bookmark,
                "scope": Scope.JOBS_WRITE.value,
            },
            "get_bookmarked_jobs": {
                "description": "Show all previously saved/bookmarked jobs. Args: limit (int, default 10).",
                "callable": tool_get_bookmarks,
                "scope": Scope.JOBS_SEARCH.value,
            },
            "draft_application_email": {
                "description": (
                    "Draft a personalized job application email using the user's profile, "
                    "then save it automatically. "
                    "Args: job_title (str), company (str), job_url (str), job_description (str)."
                ),
                "callable": tool_draft_application_email,
                "scope": Scope.EMAIL_DRAFT.value,
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

            next_actions = []
            if "job_search" in tools_used:
                next_actions.append("save_job_bookmark")
            if "save_job_bookmark" in tools_used:
                next_actions.append("draft_application_email")

            envelope = make_envelope(
                agent=self.name,
                goal=intent or user_input,
                inputs={"user_input": user_input, "intent": intent},
                result_content=final_answer,
                status=status,
                confidence=confidence,
                tools_used=tools_used,
                next_actions=next_actions,
            )

            # Attach raw job results for rich frontend cards
            if ctx["job_results"]:
                envelope["result"]["jobs"] = ctx["job_results"]

            state["task_result"] = envelope
            state["agent_reasoning"] = (
                f"Job query processed. intent={intent}, "
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
                result_content="I encountered an error processing your job-related query.",
                status="failed",
                confidence=0.0,
            )
            state["task_result"] = envelope
            state["error"] = f"Job agent error: {str(e)}"
            return state


# Singleton instance
job_agent = JobAgent()
