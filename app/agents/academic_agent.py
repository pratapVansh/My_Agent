"""
Academic Agent - Handles academic and educational tasks.
Returns a TaskEnvelope for structured coordination.
"""
from typing import Dict, Any
from datetime import datetime
from app.agents.base_agent import BaseAgent
from app.agents.state import make_envelope
from app.auth.models import Scope
from app.tools.contract import Effect
from app.tools import time_tool
from app.tools.timetable_tool import timetable_tool, TimetableInput
from app.memory.memory_manager import memory_manager
from app.domain.academic import academic_repository


def _render_schedule(tool_results) -> "str | None":
    """
    The deterministic answer for a turn that read the timetable, or None.

    Reads the *last* schedule result rather than the first: a loop that called
    `get_schedule` twice — for "today and tomorrow" — should answer about what
    it looked at most recently, and a first-call-wins rule would silently drop
    the second half of the question. Turns that touched no schedule tool return
    None and keep the model's text.
    """
    from app.tools import schedule_query

    for result in reversed(list(tool_results)):
        if getattr(result, "tool", "") not in ("get_schedule", "get_next_class"):
            continue
        payload = getattr(result, "raw", None)
        if not isinstance(payload, dict):
            # The contract wrapped a non-dict return. Nothing to render from,
            # and inventing a schedule sentence here would be the exact failure
            # this function exists to prevent.
            continue
        rendered = schedule_query.render_from_tool_result(payload)
        if rendered:
            return rendered
    return None


class AcademicAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="academic",
            description="Academic research, study help, timetable and attendance support"
        )

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state["user_input"]
        intent = state.get("detected_intent", "")
        user_id = state.get("user_id", "")

        base_system_prompt = """You are an academic assistant with full control over the user's schedule, attendance, and exams.

Your capabilities:
0. **get_schedule** — the user's actual classes for a day, range, subject or time.
   This is THE tool for "what classes do I have tomorrow", "what's my schedule
   on Friday", "when is Generative AI", "do I have a class at 10am".
   **Pass the user's own word for the day** ("today", "tomorrow", "Friday") in
   `when`. Never compute a date or a weekday yourself — the tool does that
   against the real clock, and your arithmetic is not the source of truth.
0b. **get_next_class** — the next class from right now. Takes no arguments.
1. **class_suggestions** — show timetable with attendance risk flags (< 75% = high risk).
2. **add_timetable_entry** — add a new class to the schedule via voice (e.g. "Add Physics Monday 9am to 11am Room 204").
3. **mark_attendance** — record today's (or any date's) attendance for a subject.
4. **get_attendance_summary** — show per-subject attendance percentages.
5. **add_exam** — add an upcoming exam/test/quiz to the schedule.
6. **get_upcoming_exams** — list all upcoming exams sorted by date.
7. **generate_study_schedule** — create a study plan based on exams and available slots.
8. **save_plan** — save a personal daily task or reminder (e.g. "remind me to study NLP tomorrow").
9. **get_plans** — retrieve tasks/plans for today, tomorrow, or any date.
10. **mark_plan_done** — mark a task as completed.

Day names: Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6.
When the user says "today", use today's date. When they say "mark me present/absent", call mark_attendance.
Always confirm after storing data."""

        # ── Helper parsers ────────────────────────────────────────────────────

        _DAY_MAP = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
            "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
        }

        def _parse_day(val) -> int:
            if isinstance(val, int):
                return max(0, min(6, val))
            s = str(val).strip().lower()
            return _DAY_MAP.get(s, 0)

        def _parse_time_str(val: str):
            if not val:
                return None
            val = val.strip().lower().replace(" ", "")
            for fmt in ("%H:%M", "%H:%M:%S", "%I:%M%p", "%I%p"):
                try:
                    return datetime.strptime(val, fmt).time()
                except Exception:
                    continue
            return None

        def _parse_date_str(val: str):
            """
            Resolve a date phrase against the assistant's configured timezone.

            Every `date.today()` here used to be the *host's* date while
            `time_tool` used the configured one. In a datacentre those are
            different dates for several hours a day, so an attendance record
            marked at 11pm local landed on the wrong day — the exact error
            `time_tool`'s own docstring warns about. There is one clock now.
            """
            if not val:
                return time_tool.today()
            val = val.strip().lower()
            relative = time_tool.resolve_relative_day(val)
            if relative is not None:
                return relative
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    return datetime.strptime(val, fmt).date()
                except Exception:
                    continue
            return time_tool.today()

        # ── Tool implementations ───────────────────────────────────────────

        async def tool_get_schedule(tool_input: Dict[str, Any]):
            """
            The classes on a day, read from the stored timetable.

            `when` is passed through verbatim — the user's own word. Every date
            calculation happens inside the tool against the configured clock, so
            the model is never asked what "tomorrow" means.
            """
            return await timetable_tool.get_schedule(
                user_id=user_id,
                when=str(tool_input.get("when") or "today"),
                subject=(str(tool_input["subject"]).strip()
                         if tool_input.get("subject") else None),
                at_time=(str(tool_input["at_time"]).strip()
                         if tool_input.get("at_time") else None),
            )

        async def tool_get_next_class(tool_input: Dict[str, Any]):
            return await timetable_tool.get_next_class(user_id=user_id)

        async def tool_class_suggestions(tool_input: Dict[str, Any]):
            day_of_week = tool_input.get("day_of_week")
            low_attendance_threshold = float(tool_input.get("low_attendance_threshold", 75.0))
            return await timetable_tool.suggest_classes(
                user_id=user_id,
                day_of_week=day_of_week,
                low_attendance_threshold=low_attendance_threshold,
            )

        async def tool_add_timetable_entry(tool_input: Dict[str, Any]):
            subject = str(tool_input.get("subject", "")).strip()
            if not subject:
                return {"success": False, "reason": "subject is required"}
            day = _parse_day(tool_input.get("day_of_week", 0))
            start_t = _parse_time_str(str(tool_input.get("start_time", "09:00")))
            end_t = _parse_time_str(str(tool_input.get("end_time", "10:00")))
            if not start_t or not end_t:
                return {"success": False, "reason": "Could not parse start_time or end_time"}
            entry = TimetableInput(
                day_of_week=day,
                start_time=start_t.strftime("%H:%M"),
                end_time=end_t.strftime("%H:%M"),
                subject=subject,
                location=tool_input.get("location"),
                instructor=tool_input.get("instructor"),
            )
            result = await timetable_tool.store_timetable(user_id=user_id, entries=[entry])
            return {**result, "message": f"Added {subject} on day {day} from {start_t} to {end_t}."}

        async def tool_mark_attendance(tool_input: Dict[str, Any]):
            subject = str(tool_input.get("subject", "")).strip()
            if not subject:
                return {"success": False, "reason": "subject is required"}
            status = str(tool_input.get("status", "present")).lower()
            if status not in ("present", "absent", "late"):
                status = "present"
            rec_date = _parse_date_str(str(tool_input.get("date", "today")))
            notes = tool_input.get("notes")
            rec_id = await academic_repository.store_attendance(
                user_id=user_id,
                date=rec_date,
                subject=subject,
                status=status,
                notes=notes,
            )
            return {
                "success": True,
                "id": rec_id,
                "message": f"Marked {status} for {subject} on {rec_date.isoformat()}.",
            }

        async def tool_get_attendance_summary(tool_input: Dict[str, Any]):
            records = await academic_repository.retrieve_attendance(user_id=user_id)
            if not records:
                return {"success": True, "summary": [], "message": "No attendance records found."}
            from collections import defaultdict
            by_subject: Dict[str, Dict[str, float]] = defaultdict(lambda: {"attended": 0.0, "total": 0.0})
            for rec in records:
                subj = rec.get("subject", "Unknown")
                status = (rec.get("status") or "").lower()
                by_subject[subj]["total"] += 1.0
                if status == "present":
                    by_subject[subj]["attended"] += 1.0
                elif status == "late":
                    by_subject[subj]["attended"] += 0.5
            summary = []
            for subj, agg in sorted(by_subject.items()):
                pct = (agg["attended"] / agg["total"] * 100) if agg["total"] > 0 else 100.0
                summary.append({
                    "subject": subj,
                    "attended": int(agg["attended"]),
                    "total": int(agg["total"]),
                    "percentage": round(pct, 1),
                    "risk": "high" if pct < 75 else ("medium" if pct < 85 else "low"),
                })
            summary.sort(key=lambda x: x["percentage"])
            return {"success": True, "summary": summary}

        async def tool_add_exam(tool_input: Dict[str, Any]):
            subject = str(tool_input.get("subject", "")).strip()
            if not subject:
                return {"success": False, "reason": "subject is required"}
            exam_date = _parse_date_str(str(tool_input.get("exam_date", "")))
            start_t_str = tool_input.get("start_time")
            start_t = _parse_time_str(str(start_t_str)) if start_t_str else None
            exam_id = await academic_repository.store_exam(
                user_id=user_id,
                subject=subject,
                exam_date=exam_date,
                start_time=start_t,
                location=tool_input.get("location"),
                exam_type=tool_input.get("exam_type", "exam"),
                notes=tool_input.get("notes"),
            )
            return {
                "success": True,
                "exam_id": exam_id,
                "message": f"Added {tool_input.get('exam_type','exam')} for {subject} on {exam_date.isoformat()}.",
            }

        async def tool_get_upcoming_exams(tool_input: Dict[str, Any]):
            exams = await academic_repository.retrieve_exams(user_id=user_id, upcoming_only=True)
            if not exams:
                return {"success": True, "count": 0, "exams": [], "message": "No upcoming exams."}
            return {"success": True, "count": len(exams), "exams": exams}

        async def tool_generate_study_schedule(tool_input: Dict[str, Any]):
            """LLM-based study plan from exams + timetable context."""
            exams = await academic_repository.retrieve_exams(user_id=user_id, upcoming_only=True)
            timetable = await academic_repository.retrieve_timetable(user_id=user_id)
            attendance_records = await academic_repository.retrieve_attendance(user_id=user_id)

            # Build attendance risk map
            from collections import defaultdict
            by_subject: Dict[str, Dict[str, float]] = defaultdict(lambda: {"attended": 0.0, "total": 0.0})
            for rec in attendance_records:
                subj = rec.get("subject", "")
                status = (rec.get("status") or "").lower()
                by_subject[subj]["total"] += 1.0
                if status == "present":
                    by_subject[subj]["attended"] += 1.0
                elif status == "late":
                    by_subject[subj]["attended"] += 0.5
            risk_map = {}
            for subj, agg in by_subject.items():
                pct = (agg["attended"] / agg["total"] * 100) if agg["total"] > 0 else 100.0
                risk_map[subj] = round(pct, 1)

            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            timetable_text = "\n".join(
                f"  {days[e.get('day_of_week', 0)]}: {e.get('subject')} {e.get('start_time')}–{e.get('end_time')}"
                for e in timetable
            ) or "No timetable set."
            exams_text = "\n".join(
                f"  {e.get('exam_date')} — {e.get('subject')} ({e.get('exam_type','exam')})"
                for e in exams
            ) or "No upcoming exams."
            risk_text = "\n".join(
                f"  {s}: {p}% attendance" for s, p in sorted(risk_map.items(), key=lambda x: x[1])
            ) or "No attendance data."

            prompt = (
                f"Create a concise weekly study schedule for a student.\n\n"
                f"Upcoming exams:\n{exams_text}\n\n"
                f"Class timetable:\n{timetable_text}\n\n"
                f"Attendance risk (lower % = higher risk):\n{risk_text}\n\n"
                f"Today: {time_tool.today().isoformat()}\n\n"
                f"Rules: Prioritise subjects with low attendance and near exams. "
                f"Schedule 1–2 hour study blocks in free time slots. "
                f"Output as a plain-text day-by-day schedule."
            )
            schedule = await self.call_groq(
                messages=[
                    {"role": "system", "content": "You are an academic planning assistant. Output only the schedule — no preamble."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=600,
            )
            return {"success": True, "schedule": schedule}

        async def tool_save_plan(tool_input: Dict[str, Any]):
            title = str(tool_input.get("title", "")).strip()
            if not title:
                return {"success": False, "reason": "title is required"}
            plan_date = _parse_date_str(str(tool_input.get("date", "today")))
            priority = str(tool_input.get("priority", "medium")).lower()
            if priority not in ("high", "medium", "low"):
                priority = "medium"
            plan_id = await academic_repository.store_plan(
                user_id=user_id,
                plan_date=plan_date,
                title=title,
                description=tool_input.get("description"),
                priority=priority,
            )
            return {
                "success": True,
                "plan_id": plan_id,
                "message": f"Saved plan '{title}' for {plan_date.isoformat()} (priority: {priority}).",
            }

        async def tool_get_plans(tool_input: Dict[str, Any]):
            date_str = str(tool_input.get("date", "")).strip()
            include_done = bool(tool_input.get("include_done", False))
            if date_str and date_str not in ("all", "upcoming"):
                plan_date = _parse_date_str(date_str)
            else:
                plan_date = None  # returns all upcoming
            plans = await academic_repository.retrieve_plans(
                user_id=user_id,
                plan_date=plan_date,
                include_done=include_done,
            )
            if not plans:
                label = plan_date.isoformat() if plan_date else "upcoming days"
                return {"success": True, "plans": [], "message": f"No plans found for {label}."}
            return {"success": True, "count": len(plans), "plans": plans}

        async def tool_mark_plan_done(tool_input: Dict[str, Any]):
            plan_id = str(tool_input.get("plan_id", "")).strip()
            if not plan_id:
                return {"success": False, "reason": "plan_id is required"}
            updated = await academic_repository.mark_plan_done(plan_id=plan_id, user_id=user_id)
            return {
                "success": updated,
                "message": "Marked as done." if updated else "Plan not found.",
            }

        tools = {
            "get_schedule": {
                "description": (
                    "THE tool for any question about which classes the user has. "
                    "Pass the user's own words for the day in `when` — 'today', "
                    "'tomorrow', 'Friday', 'this week', or a date like "
                    "'2026-08-14'. Do NOT work out the date yourself; the tool "
                    "resolves it. "
                    "Args: when (str, default 'today'), subject (str, optional — "
                    "filters to one subject), at_time (str, optional e.g. "
                    "'10:00' or '10 am')."
                ),
                "callable": tool_get_schedule,
                "effect": Effect.READ,
                "scope": Scope.TIMETABLE_READ.value,
            },
            "get_next_class": {
                "description": (
                    "The user's next upcoming class, from the current time "
                    "onwards. Use for 'what is my next class' / 'what do I have "
                    "next'. Args: none — the tool reads the clock itself."
                ),
                "callable": tool_get_next_class,
                "effect": Effect.READ,
                "scope": Scope.TIMETABLE_READ.value,
            },
            "class_suggestions": {
                "description": (
                    "Show timetable with per-subject attendance risk flags. "
                    "Args: day_of_week (int 0–6, optional), low_attendance_threshold (float, default 75)."
                ),
                "callable": tool_class_suggestions,
                "effect": Effect.READ,
                "scope": Scope.TIMETABLE_READ.value,
            },
            "add_timetable_entry": {
                "description": (
                    "Add a new class/lecture to the timetable via voice. "
                    "Args: subject (str), day_of_week (int or name e.g. 'Monday'), "
                    "start_time (str e.g. '09:00'), end_time (str), location (str), instructor (str)."
                ),
                "callable": tool_add_timetable_entry,
                "effect": Effect.LOCAL_WRITE,
                "scope": Scope.TIMETABLE_WRITE.value,
            },
            "mark_attendance": {
                "description": (
                    "Record attendance for a subject on a specific date. "
                    "Args: subject (str), status (present|absent|late), date (str e.g. 'today' or '2024-03-15'), notes (str)."
                ),
                "callable": tool_mark_attendance,
                "effect": Effect.LOCAL_WRITE,
                "scope": Scope.ATTENDANCE_WRITE.value,
            },
            "get_attendance_summary": {
                "description": "Show per-subject attendance percentage and risk level. Args: none.",
                "callable": tool_get_attendance_summary,
                "effect": Effect.READ,
                "scope": Scope.ATTENDANCE_READ.value,
            },
            "add_exam": {
                "description": (
                    "Add an exam, test, or quiz to the schedule. "
                    "Args: subject (str), exam_date (str e.g. '2024-04-20'), "
                    "start_time (str, optional), location (str), exam_type (exam|midterm|final|quiz|assignment), notes (str)."
                ),
                "callable": tool_add_exam,
                "effect": Effect.LOCAL_WRITE,
                "scope": Scope.TIMETABLE_WRITE.value,
            },
            "get_upcoming_exams": {
                "description": "List all upcoming exams sorted by date. Args: none.",
                "callable": tool_get_upcoming_exams,
                "effect": Effect.READ,
                "scope": Scope.TIMETABLE_READ.value,
            },
            "generate_study_schedule": {
                "description": (
                    "Generate a personalised weekly study schedule based on upcoming exams, "
                    "timetable, and attendance risk. Args: none."
                ),
                "callable": tool_generate_study_schedule,
                # Reads the timetable and composes a plan; storing it is
                # save_plan's job, not this one's.
                "effect": Effect.READ,
                "scope": Scope.TIMETABLE_READ.value,
            },
            "save_plan": {
                "description": (
                    "Save a daily plan, task, or reminder for a specific date. "
                    "Args: title (str), date (str e.g. 'today', 'tomorrow', '2024-04-10'), "
                    "description (str, optional), priority (high|medium|low)."
                ),
                "callable": tool_save_plan,
                "effect": Effect.LOCAL_WRITE,
                "scope": Scope.TIMETABLE_WRITE.value,
            },
            "get_plans": {
                "description": (
                    "Retrieve plans/tasks for a specific date or all upcoming days. "
                    "Args: date (str e.g. 'today', 'tomorrow', '2024-04-10', or 'upcoming'), "
                    "include_done (bool, default false)."
                ),
                "callable": tool_get_plans,
                "effect": Effect.READ,
                "scope": Scope.TIMETABLE_READ.value,
            },
            "mark_plan_done": {
                "description": (
                    "Mark a plan/task as completed. "
                    "Args: plan_id (str — from get_plans response)."
                ),
                "callable": tool_mark_plan_done,
                "effect": Effect.LOCAL_WRITE,
                "scope": Scope.TIMETABLE_WRITE.value,
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

            # ── The timetable answers for itself ──────────────────────────────
            #
            # When a schedule tool ran, the reply is rendered from the rows it
            # returned rather than from the model's account of them. Same
            # arrangement as a held action's preview: the model's version is not
            # checked against the data, it is simply not the thing delivered.
            #
            # A timetable answer is a list of times with subjects attached, and
            # every word of paraphrase drift is a class the user might turn up
            # late to — or miss entirely. There is nothing here a model adds.
            #
            # `grounding` has already guaranteed that *a* schedule tool ran for
            # a timetable question; this decides what the user reads when it did.
            rendered = _render_schedule(loop_result.get("tool_results") or [])
            if rendered is not None:
                final_answer = rendered

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
                next_actions=["check_attendance", "view_timetable"] if tools_used else [],
            )

            state["task_result"] = envelope
            # See job_agent: the loop computes this for every agent, and it is
            # what separates "no classes today" from "the timetable store is
            # unreachable".
            state["answerability"] = loop_result.get("answerability") or ""
            state["agent_reasoning"] = (
                f"Academic query processed. intent={intent}, "
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
                result_content="I encountered an error processing your academic query.",
                status="failed",
                confidence=0.0,
            )
            state["task_result"] = envelope
            state["error"] = f"Academic agent error: {str(e)}"
            return state


# Singleton instance
academic_agent = AcademicAgent()
