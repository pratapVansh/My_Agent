"""
Timetable tool for storing user input and suggesting classes based on attendance risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Dict, List, Optional

from app.memory.memory_manager import memory_manager
from app.services.langsmith_service import traceable


@dataclass
class TimetableInput:
    day_of_week: int
    start_time: str
    end_time: str
    subject: str
    location: Optional[str] = None
    instructor: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TimetableTool:
    """Store timetable entries and suggest priority classes."""

    @traceable(name="tool_timetable_store", run_type="tool", tags=["tool", "timetable"])
    async def store_timetable(
        self,
        user_id: str,
        entries: List[TimetableInput],
    ) -> Dict[str, Any]:
        stored_ids: List[str] = []
        for entry in entries:
            start_t = self._parse_time(entry.start_time)
            end_t = self._parse_time(entry.end_time)
            if start_t is None or end_t is None:
                continue
            if entry.day_of_week < 0 or entry.day_of_week > 6:
                continue

            rec_id = await memory_manager.store_timetable_entry(
                user_id=user_id,
                day_of_week=entry.day_of_week,
                start_time=start_t,
                end_time=end_t,
                subject=entry.subject,
                location=entry.location,
                instructor=entry.instructor,
                metadata=entry.metadata,
            )
            stored_ids.append(rec_id)

        return {
            "tool": "timetable_store",
            "success": True,
            "user_id": user_id,
            "stored_count": len(stored_ids),
            "stored_ids": stored_ids,
        }

    @traceable(name="tool_timetable_suggest", run_type="tool", tags=["tool", "attendance", "timetable"])
    async def suggest_classes(
        self,
        user_id: str,
        day_of_week: Optional[int] = None,
        low_attendance_threshold: float = 75.0,
    ) -> Dict[str, Any]:
        attendance_records = await memory_manager.retrieve_attendance(user_id=user_id)
        timetable_entries = await memory_manager.retrieve_timetable(user_id=user_id, day_of_week=day_of_week)

        attendance_map = self._attendance_percentage_by_subject(attendance_records)

        suggestions: List[Dict[str, Any]] = []
        for item in timetable_entries:
            subject = item.get("subject", "")
            percentage = attendance_map.get(subject, 100.0)
            risk = "low"
            if percentage < low_attendance_threshold:
                risk = "high"
            elif percentage < low_attendance_threshold + 10:
                risk = "medium"

            suggestions.append(
                {
                    "subject": subject,
                    "day_of_week": item.get("day_of_week"),
                    "start_time": item.get("start_time"),
                    "end_time": item.get("end_time"),
                    "location": item.get("location"),
                    "attendance_percentage": round(percentage, 2),
                    "risk": risk,
                    "suggestion_reason": (
                        "Low attendance priority"
                        if percentage < low_attendance_threshold
                        else "Regular schedule"
                    ),
                }
            )

        suggestions.sort(
            key=lambda x: (x["attendance_percentage"], x["day_of_week"], x["start_time"])
        )

        return {
            "tool": "class_suggestions",
            "success": True,
            "user_id": user_id,
            "day_of_week": day_of_week,
            "low_attendance_threshold": low_attendance_threshold,
            "suggestions": suggestions,
        }

    def _parse_time(self, value: str) -> Optional[time]:
        if not value:
            return None
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).time()
            except Exception:
                continue
        return None

    def _attendance_percentage_by_subject(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        by_subject: Dict[str, Dict[str, float]] = {}

        for rec in records:
            subject = rec.get("subject", "")
            status = (rec.get("status", "") or "").lower()
            if not subject:
                continue

            if subject not in by_subject:
                by_subject[subject] = {"attended": 0.0, "total": 0.0}

            by_subject[subject]["total"] += 1.0
            if status == "present":
                by_subject[subject]["attended"] += 1.0
            elif status == "late":
                by_subject[subject]["attended"] += 0.5

        out: Dict[str, float] = {}
        for subject, agg in by_subject.items():
            if agg["total"] <= 0:
                out[subject] = 100.0
            else:
                out[subject] = (agg["attended"] / agg["total"]) * 100.0
        return out


timetable_tool = TimetableTool()
