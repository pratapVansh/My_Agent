"""
Job Agent - Phase 2 upgrade.
New capabilities:
- Skills-aware job search (matches against user's Qdrant skills)
- Bookmark jobs to PostgreSQL (save_job_bookmark tool)
- View saved jobs (get_bookmarked_jobs tool)
- Deduplication: already-bookmarked URLs are skipped in search results
- Evidence-based job matching (match_job), explained from the evidence table

## Why `match_job` writes its own answer

Every other tool here hands the model data and lets it compose a reply. This
one does not, and the exception is deliberate.

"How well do I match this?" is the single question in the system where a fluent
answer is most dangerous. The model has the posting in front of it, the
requirement list names a dozen technologies, and the cheapest way to sound
useful is to affirm them. That produces a paragraph telling the user — who may
forward it to a recruiter — that they know things their resume never mentions.

So the final answer for a match turn is `app.matching.explain.render(report)`:
prose composed from the verdict table, in Python, where every positive sentence
is backed by a stored chunk id and every gap was gated on
`CandidateProfile.may_assert_gaps`. The model's job shrinks to deciding *which*
posting to check, which is a routing decision it is good at, rather than what
the candidate knows, which it has no way to know.

This mirrors what `provenance_node` and `temporal_node` already do in the
workflow: a question with exactly one true answer is not asked of a model.
"""
from typing import Any, Dict, List, Optional

from app.agents.base_agent import BaseAgent
from app.agents.state import make_envelope
from app.auth.models import Scope
from app.candidate import build_candidate_profile
from app.matching import (
    extract_requirements,
    match_requirements,
    render,
    render_summary,
)
from app.tools.contract import Effect, ToolResult
from app.tools.job_search_tool import job_search_tool
from app.tools.email_draft_tool import email_draft_tool
from app.memory.short_term_memory import short_term_memory
from app.memory.memory_manager import memory_manager
from app.domain.email import email_repository
from app.domain.jobs import jobs_repository

# Budget for the rendered report inside a tool observation. `ToolResult.
# observation` truncates the whole JSON payload at 1200 characters and escapes
# non-ASCII to six-character sequences, so the report is fitted well below that
# — a cut there could land mid-source-id and emit an identifier that looks real.
_MATCH_OBSERVATION_CHARS = 800


def _job_from_results(
    tool_input: Dict[str, Any], results: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Resolve which already-found job the user means.

    Refuses to guess between several, for the same reason
    `confirmation.resolve` refuses to guess between two pending actions:
    silently matching against the wrong posting produces a confident,
    thoroughly evidenced answer to a question nobody asked.
    """
    if not results:
        return None

    url = str(tool_input.get("job_url") or "").strip()
    if url:
        return next(
            (r for r in results if str(r.get("url", "")).strip() == url), None
        )

    index = tool_input.get("job_index")
    if index is not None:
        try:
            position = int(index)
        except (TypeError, ValueError):
            return None
        if 1 <= position <= len(results):
            return results[position - 1]
        return None

    # "this job" is unambiguous only when there is exactly one.
    return results[0] if len(results) == 1 else None


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
5. **Assess a match** — use match_job whenever the user asks how well they match, fit, or qualify for a role.

## Assessing a match — read this before answering one
You do NOT judge how well the user matches a job. match_job does, by comparing
the posting's requirements against evidence in the user's stored resume, and the
system replaces your final answer with its report. This is enforced outside your
control, so there is nothing to gain by writing your own assessment.

What this means for you:
- NEVER state that the user has a skill, a qualification, or a number of years
  of experience. You have no way to know, and match_job does.
- NEVER produce a match percentage or a verdict of your own. The score is
  computed deterministically.
- Call match_job with `job_description` when the user pasted a posting, or with
  `job_index` (1-based) / `job_url` to reuse a result from an earlier
  job_search in this conversation. Do not re-type the user's skills into it —
  it reads them from their profile itself.
- If match_job reports it has no job description, ask the user to paste the
  posting. Do not guess at the requirements.

When presenting job results: show title, company, match score, and skills matched. List top 3–5.
When the user says "apply" or "write an email for this job", call draft_application_email with the job details.
For career advice (no tool needed): give practical guidance based on the user's profile in memory."""

        # ── Tool definitions ──────────────────────────────────────────────
        # Use a dict so the nested closure can update the reference without
        # needing nonlocal or in-place mutation of a bare list.
        ctx: Dict[str, Any] = {"job_results": [], "match_report": None}

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

        async def tool_match_job(tool_input: Dict[str, Any]) -> ToolResult:
            """
            Compare a posting against the user's evidenced profile.

            The candidate side is read from `user_id` — the signed-in caller —
            and never from `tool_input`. That is the "no second source of
            truth" rule made structural: a model that puts a skill list in its
            arguments cannot thereby introduce a qualification, because nothing
            here looks at one.
            """
            description = str(tool_input.get("job_description") or "").strip()
            title = str(
                tool_input.get("title") or tool_input.get("job_title") or ""
            ).strip()
            source = str(tool_input.get("job_url") or "").strip()

            if not description:
                chosen = _job_from_results(tool_input, ctx["job_results"])
                if chosen:
                    description = (
                        f"{chosen.get('title', '')}\n{chosen.get('snippet', '')}"
                    )
                    title = title or str(chosen.get("title", ""))
                    source = source or str(chosen.get("url", ""))

            if not description:
                return ToolResult.no_data(
                    "No job description is available. Ask the user to paste the "
                    "posting, or run job_search first and pass job_index.",
                    tool="match_job",
                )

            requirements = extract_requirements(
                description, title=title, source=source
            )
            if not requirements:
                return ToolResult.no_data(
                    "No recognisable requirements could be read from that job "
                    "description.",
                    tool="match_job",
                )

            # The one source of truth for what the candidate can prove.
            profile = await build_candidate_profile(user_id)
            report = match_requirements(requirements, profile)
            ctx["match_report"] = report

            return ToolResult.success(
                {
                    "report": render(report, limit=_MATCH_OBSERVATION_CHARS),
                    "score": report.score,
                    "band": report.band.value,
                    "requirements_read": len(requirements),
                    "evidenced": len(report.matched),
                    "partial": len(report.partial),
                    "not_evidenced": len(report.missing),
                    "not_checked": len(report.unknown),
                    "profile_degraded": report.profile_degraded,
                },
                tool="match_job",
            )

        tools = {
            "match_job": {
                "description": (
                    "Assess how well the user matches a job, from evidence in "
                    "their stored resume. Use for 'how well do I match this', "
                    "'am I a good fit', 'do I qualify'. Produces the score and "
                    "the explanation itself — never write your own. "
                    "Args: job_description (str, the posting text), title (str, "
                    "optional), or job_index (int, 1-based, to reuse a result "
                    "from an earlier job_search) or job_url (str)."
                ),
                "callable": tool_match_job,
                # Reads the posting and the user's own stored profile. Changes
                # nothing anywhere.
                "effect": Effect.READ,
                "scope": Scope.JOBS_SEARCH.value,
            },
            "job_search": {
                "description": (
                    "Search current job openings. Automatically uses the user's skills to "
                    "find better matches and skips already-bookmarked jobs. "
                    "Args: query (str), location (str, optional), max_results (int, default 6)."
                ),
                "callable": tool_job_search,
                "effect": Effect.READ,
                "scope": Scope.JOBS_SEARCH.value,
            },
            "save_job_bookmark": {
                "description": (
                    "Save a job to the user's bookmarks. "
                    "Args: title (str), url (str), company (str), snippet (str), "
                    "rank_score (float), skills_matched (list of str)."
                ),
                "callable": tool_save_bookmark,
                "effect": Effect.LOCAL_WRITE,
                "scope": Scope.JOBS_WRITE.value,
            },
            "get_bookmarked_jobs": {
                "description": "Show all previously saved/bookmarked jobs. Args: limit (int, default 10).",
                "callable": tool_get_bookmarks,
                "effect": Effect.READ,
                "scope": Scope.JOBS_SEARCH.value,
            },
            "draft_application_email": {
                "description": (
                    "Draft a personalized job application email using the user's profile, "
                    "then save it automatically. "
                    "Args: job_title (str), company (str), job_url (str), job_description (str)."
                ),
                "callable": tool_draft_application_email,
                # Composes and stores a draft. Nothing leaves the system —
                # sending is a separate, confirmable step.
                "effect": Effect.LOCAL_WRITE,
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

            # ── The match answer is rendered, not generated ───────────────────
            # See the module docstring. When a report exists it *is* the answer:
            # every positive sentence in it carries a stored chunk id, and every
            # gap passed the `may_assert_gaps` gate. Whatever the model wrote is
            # discarded rather than merged, because a merge would put an
            # unverified sentence next to verified ones and there would be no
            # way for a reader to tell which was which.
            match_report = ctx.get("match_report")
            if match_report is not None:
                final_answer = render(match_report)

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
                next_actions.append("match_job")
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

            # The structured verdict, for a UI that wants to render the table
            # rather than the prose. Carries the same numbers the text does —
            # both come from the one report, so they cannot disagree.
            if match_report is not None:
                envelope["result"]["match"] = {
                    "title": match_report.title,
                    "score": match_report.score,
                    "band": match_report.band.value,
                    "required_coverage": match_report.required_coverage,
                    "preferred_coverage": match_report.preferred_coverage,
                    "degraded": match_report.profile_degraded,
                    "summary": render_summary(match_report),
                    "evidence_ids": list(match_report.evidence_ids()),
                    "requirements": [m.summary() for m in match_report.matches],
                }

            state["task_result"] = envelope
            # The observed verdict on what the tools produced. Computed by the
            # loop for every agent; surfacing it here is what lets provenance
            # and the response layer tell "no jobs matched" apart from "the
            # job board was unreachable".
            state["answerability"] = loop_result.get("answerability") or ""
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
