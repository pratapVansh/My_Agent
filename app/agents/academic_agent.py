"""
Academic Agent - Handles academic and educational tasks.
Returns a TaskEnvelope for structured coordination.
"""
from typing import Dict, Any
from datetime import datetime, date as date_cls, time as time_cls
from app.agents.base_agent import BaseAgent
from app.agents.state import make_envelope
from app.tools.timetable_tool import timetable_tool, TimetableInput
from app.memory.memory_manager import memory_manager


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
            if not val:
                return date_cls.today()
            val = val.strip().lower()
            if val in ("today", "now"):
                return date_cls.today()
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    return datetime.strptime(val, fmt).date()
                except Exception:
                    continue
            return date_cls.today()

        # ── Tool implementations ───────────────────────────────────────────

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
            rec_id = await memory_manager.store_attendance(
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
            records = await memory_manager.retrieve_attendance(user_id=user_id)
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
            exam_id = await memory_manager.store_exam(
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
            exams = await memory_manager.retrieve_exams(user_id=user_id, upcoming_only=True)
            if not exams:
                return {"success": True, "count": 0, "exams": [], "message": "No upcoming exams."}
            return {"success": True, "count": len(exams), "exams": exams}

        async def tool_generate_study_schedule(tool_input: Dict[str, Any]):
            """LLM-based study plan from exams + timetable context."""
            exams = await memory_manager.retrieve_exams(user_id=user_id, upcoming_only=True)
            timetable = await memory_manager.retrieve_timetable(user_id=user_id)
            attendance_records = await memory_manager.retrieve_attendance(user_id=user_id)

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
                f"Today: {date_cls.today().isoformat()}\n\n"
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
            plan_id = await memory_manager.store_plan(
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
            plans = await memory_manager.retrieve_plans(
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
            updated = await memory_manager.mark_plan_done(plan_id=plan_id, user_id=user_id)
            return {
                "success": updated,
                "message": "Marked as done." if updated else "Plan not found.",
            }

        tools = {
            "class_suggestions": {
                "description": (
                    "Show timetable with per-subject attendance risk flags. "
                    "Args: day_of_week (int 0–6, optional), low_attendance_threshold (float, default 75)."
                ),
                "callable": tool_class_suggestions,
            },
            "add_timetable_entry": {
                "description": (
                    "Add a new class/lecture to the timetable via voice. "
                    "Args: subject (str), day_of_week (int or name e.g. 'Monday'), "
                    "start_time (str e.g. '09:00'), end_time (str), location (str), instructor (str)."
                ),
                "callable": tool_add_timetable_entry,
            },
            "mark_attendance": {
                "description": (
                    "Record attendance for a subject on a specific date. "
                    "Args: subject (str), status (present|absent|late), date (str e.g. 'today' or '2024-03-15'), notes (str)."
                ),
                "callable": tool_mark_attendance,
            },
            "get_attendance_summary": {
                "description": "Show per-subject attendance percentage and risk level. Args: none.",
                "callable": tool_get_attendance_summary,
            },
            "add_exam": {
                "description": (
                    "Add an exam, test, or quiz to the schedule. "
                    "Args: subject (str), exam_date (str e.g. '2024-04-20'), "
                    "start_time (str, optional), location (str), exam_type (exam|midterm|final|quiz|assignment), notes (str)."
                ),
                "callable": tool_add_exam,
            },
            "get_upcoming_exams": {
                "description": "List all upcoming exams sorted by date. Args: none.",
                "callable": tool_get_upcoming_exams,
            },
            "generate_study_schedule": {
                "description": (
                    "Generate a personalised weekly study schedule based on upcoming exams, "
                    "timetable, and attendance risk. Args: none."
                ),
                "callable": tool_generate_study_schedule,
            },
            "save_plan": {
                "description": (
                    "Save a daily plan, task, or reminder for a specific date. "
                    "Args: title (str), date (str e.g. 'today', 'tomorrow', '2024-04-10'), "
                    "description (str, optional), priority (high|medium|low)."
                ),
                "callable": tool_save_plan,
            },
            "get_plans": {
                "description": (
                    "Retrieve plans/tasks for a specific date or all upcoming days. "
                    "Args: date (str e.g. 'today', 'tomorrow', '2024-04-10', or 'upcoming'), "
                    "include_done (bool, default false)."
                ),
                "callable": tool_get_plans,
            },
            "mark_plan_done": {
                "description": (
                    "Mark a plan/task as completed. "
                    "Args: plan_id (str — from get_plans response)."
                ),
                "callable": tool_mark_plan_done,
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
                next_actions=["check_attendance", "view_timetable"] if tools_used else [],
            )

            state["task_result"] = envelope
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
