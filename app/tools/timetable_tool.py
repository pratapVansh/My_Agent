"""
Timetable tool for storing user input and suggesting classes based on attendance risk.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.memory.memory_manager import memory_manager
from app.domain.academic import academic_repository
from app.services.langsmith_service import traceable

logger = logging.getLogger(__name__)


@dataclass
class TimetableInput:
    day_of_week: int
    start_time: str
    end_time: str
    subject: str
    location: Optional[str] = None
    instructor: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def _parse_time_str(value: str) -> Optional[time]:
    if not value:
        return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).time()
        except Exception:
            continue
    return None


def validate_entries(
    entries: List[TimetableInput],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Parse and validate raw timetable entries without touching the database.

    Pure, so it can run once before a caller decides whether to write anything
    at all — the check an atomic replace runs before it starts a transaction,
    and the same check a one-at-a-time store runs for a single entry. Kept in
    one place so the two paths cannot silently disagree on what a valid entry
    is.

    Returns `(valid, skipped)`. Each `valid` row is a plain dict with its time
    strings already parsed into `time` objects, ready to build either an ORM
    row or a `store_timetable_entry` call. Rejected rows are reported rather
    than dropped silently: a parse failure across a whole import otherwise
    looks like a valid but nearly-empty timetable.
    """
    valid: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for entry in entries:
        start_t = _parse_time_str(entry.start_time)
        end_t = _parse_time_str(entry.end_time)
        if start_t is None or end_t is None:
            skipped.append({
                "reason": "unparseable_time",
                "subject": str(entry.subject)[:80],
                "start_time": str(entry.start_time)[:20],
                "end_time": str(entry.end_time)[:20],
            })
            continue
        if entry.day_of_week < 0 or entry.day_of_week > 6:
            skipped.append({
                "reason": "invalid_day_of_week",
                "subject": str(entry.subject)[:80],
                "day_of_week": str(entry.day_of_week),
            })
            continue

        valid.append({
            "day_of_week": entry.day_of_week,
            "start_time": start_t,
            "end_time": end_t,
            "subject": entry.subject,
            "location": entry.location,
            "instructor": entry.instructor,
            "metadata": entry.metadata,
        })

    return valid, skipped


class TimetableTool:
    """Store timetable entries and suggest priority classes."""

    @traceable(name="tool_timetable_store", run_type="tool", tags=["tool", "timetable"])
    async def store_timetable(
        self,
        user_id: str,
        entries: List[TimetableInput],
        schedule_upload_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        valid, skipped = validate_entries(entries)
        stored_ids: List[str] = []

        for row in valid:
            rec_id = await academic_repository.store_timetable_entry(
                user_id=user_id,
                day_of_week=row["day_of_week"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                subject=row["subject"],
                location=row["location"],
                instructor=row["instructor"],
                metadata=row["metadata"],
                schedule_upload_id=schedule_upload_id,
            )
            stored_ids.append(rec_id)

        if skipped:
            logger.warning(
                "Timetable store for user=%s skipped %d/%d entries (first reason: %s)",
                user_id, len(skipped), len(entries), skipped[0]["reason"],
            )

        return {
            "tool": "timetable_store",
            "success": True,
            "user_id": user_id,
            "stored_count": len(stored_ids),
            "skipped_count": len(skipped),
            "skipped": skipped[:10],
            "stored_ids": stored_ids,
        }

    @traceable(name="tool_timetable_suggest", run_type="tool", tags=["tool", "attendance", "timetable"])
    async def suggest_classes(
        self,
        user_id: str,
        day_of_week: Optional[int] = None,
        low_attendance_threshold: float = 75.0,
    ) -> Dict[str, Any]:
        attendance_records = await academic_repository.retrieve_attendance(user_id=user_id)
        timetable_entries = await academic_repository.retrieve_timetable(user_id=user_id, day_of_week=day_of_week)

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

    # ── Reading the schedule ─────────────────────────────────────────────────
    #
    # `suggest_classes` above answers "which classes should I prioritise", sorts
    # by attendance risk, and takes a `day_of_week` integer. It was the only read
    # path, which meant "what have I got tomorrow" required a model to compute
    # today's date, add a day, derive the weekday and convert it to 0–6 — four
    # arithmetic steps in the one component that cannot do arithmetic. The two
    # methods below are the schedule question instead of the attendance one:
    # time-ordered, date-resolved in Python, and honest about the difference
    # between an empty day and an absent timetable.

    @traceable(name="tool_get_schedule", run_type="tool", tags=["tool", "timetable"])
    async def get_schedule(
        self,
        user_id: str,
        *,
        when: str = "today",
        subject: Optional[str] = None,
        at_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Classes for a resolved day or range, time-ordered.

        `when` is the user's own word — "today", "tomorrow", "friday",
        "2026-08-14", "this week". It is resolved here, deterministically,
        against the configured timezone. An unrecognised phrase is refused
        rather than defaulted to today: answering confidently about the wrong
        day is the failure this whole path exists to prevent.
        """
        from app.tools import schedule_query

        window = schedule_query.resolve_when(when)
        if window is None:
            return {
                "tool": "get_schedule",
                "success": False,
                "reason": "unresolved_day",
                "requested": when,
                "message": (
                    f"I couldn't work out which day '{when}' refers to. Try "
                    "'today', 'tomorrow', a weekday, or a date."
                ),
            }

        # Absent timetable and empty day are different answers, so the question
        # is asked before the rows are read rather than inferred from their
        # absence afterwards.
        try:
            has_any = await academic_repository.has_timetable(user_id=user_id)
        except Exception as exc:
            logger.error("Timetable availability check failed for %s: %s", user_id, exc)
            raise

        rows = await self._rows_for_window(user_id, window)

        if subject:
            rows = _filter_by_subject(rows, subject)
        wanted_time = self._parse_time(at_time) if at_time else None
        if at_time and wanted_time is None:
            from app.tools import schedule_query as _sq
            wanted_time = _sq.resolve_at_time(at_time)
        if wanted_time is not None:
            rows = [r for r in rows if _covers(r, wanted_time)]

        return {
            "tool": "get_schedule",
            "success": True,
            "user_id": user_id,
            "window": window.summary(),
            "has_timetable": has_any,
            "count": len(rows),
            "classes": rows,
            "filters": {
                "subject": subject,
                "at_time": wanted_time.strftime("%H:%M") if wanted_time else None,
            },
        }

    @traceable(name="tool_get_next_class", run_type="tool", tags=["tool", "timetable"])
    async def get_next_class(self, user_id: str, *, days_ahead: int = 7) -> Dict[str, Any]:
        """
        The next class starting after the current moment.

        Searches today from now, then forward day by day. Bounded at a week
        because a timetable is a weekly recurrence: if nothing is found in seven
        days there is nothing to find, and scanning further would loop forever
        on an empty schedule.
        """
        from app.tools import schedule_query, time_tool

        moment = time_tool.now()
        today = moment.date()
        current = moment.time()

        try:
            has_any = await academic_repository.has_timetable(user_id=user_id)
        except Exception as exc:
            logger.error("Timetable availability check failed for %s: %s", user_id, exc)
            raise

        for offset in range(days_ahead + 1):
            target = today + timedelta(days=offset)
            rows = await self._rows_for_date(user_id, target)
            if offset == 0:
                rows = [r for r in rows if _start_of(r) > current]
            if rows:
                nxt = rows[0]
                return {
                    "tool": "get_next_class",
                    "success": True,
                    "user_id": user_id,
                    "has_timetable": has_any,
                    "count": 1,
                    "found": True,
                    "asked_at": moment.strftime("%Y-%m-%d %H:%M"),
                    "is_today": offset == 0,
                    "label": schedule_query._describe(target, today=today),
                    "classes": [nxt],
                }

        return {
            "tool": "get_next_class",
            "success": True,
            "user_id": user_id,
            "has_timetable": has_any,
            "count": 0,
            "found": False,
            "asked_at": moment.strftime("%Y-%m-%d %H:%M"),
            "classes": [],
        }

    async def _rows_for_window(self, user_id: str, window) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for day in window.dates():
            rows.extend(await self._rows_for_date(user_id, day))
        return rows

    async def _rows_for_date(self, user_id: str, target) -> List[Dict[str, Any]]:
        """
        One day's classes, stamped with the actual date and their provenance.

        The store holds a weekly recurrence keyed by `day_of_week`; the date is
        attached here so the answer can say "Friday 14 August" rather than
        "day 4", and so a range query can group by real dates.
        """
        from app.tools import schedule_query

        entries = await academic_repository.retrieve_timetable(
            user_id=user_id, day_of_week=target.weekday()
        )
        rows = []
        for entry in entries:
            meta = entry.get("metadata") or {}
            rows.append({
                "id": entry.get("id"),
                "date": target.isoformat(),
                "weekday": schedule_query.weekday_name(target.weekday()),
                "day_of_week": entry.get("day_of_week"),
                "start_time": (entry.get("start_time") or "")[:5],
                "end_time": (entry.get("end_time") or "")[:5],
                "subject": entry.get("subject"),
                "subject_code": meta.get("subject_code"),
                "session_type": meta.get("session_type") or "LECTURE",
                "room": entry.get("location"),
                "faculty": entry.get("instructor"),
                "note": meta.get("note"),
                # Provenance travels with the row, so the answer can name the
                # document without anyone reconstructing the link later.
                "schedule_upload_id": entry.get("schedule_upload_id"),
                "source_title": meta.get("source_title"),
                "source_page": meta.get("source_page"),
            })
        rows.sort(key=lambda r: r["start_time"])
        return rows

    def _parse_time(self, value: str) -> Optional[time]:
        return _parse_time_str(value)

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


# ── Matching helpers ─────────────────────────────────────────────────────────

def _start_of(row: Dict[str, Any]) -> time:
    raw = (row.get("start_time") or "00:00")[:5]
    hour, _, minute = raw.partition(":")
    return time(int(hour or 0), int(minute or 0))


def _end_of(row: Dict[str, Any]) -> time:
    raw = (row.get("end_time") or "23:59")[:5]
    hour, _, minute = raw.partition(":")
    return time(int(hour or 0), int(minute or 0))


def _covers(row: Dict[str, Any], moment: time) -> bool:
    """Whether a class is running at this time. End-exclusive: a 10:30 class
    starting is the answer to "at 10:30", not the 09:30 one finishing."""
    return _start_of(row) <= moment < _end_of(row)


def _filter_by_subject(rows: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Match a subject the way a person names it, without inventing aliases.

    Three passes over what the timetable *actually stores* — the code, the full
    name, and the individual words of the name. "DBMS" matches nothing here and
    correctly returns nothing, because this timetable has no such subject;
    inventing an alias to make it match would be the fabrication this whole
    feature is built to avoid. "DIP", "digital image", and "image processing"
    all match, because those strings are in the stored name.
    """
    needle = (query or "").strip().lower()
    if not needle:
        return rows

    exact_code = [
        r for r in rows if (r.get("subject_code") or "").lower() == needle
    ]
    if exact_code:
        return exact_code

    substring = [
        r for r in rows
        if needle in (r.get("subject") or "").lower()
        or needle in (r.get("subject_code") or "").lower()
    ]
    if substring:
        return substring

    # Every significant word of the query appears in the stored name.
    words = [w for w in re.split(r"[^a-z0-9]+", needle) if len(w) > 2]
    if not words:
        return []
    return [
        r for r in rows
        if all(w in (r.get("subject") or "").lower() for w in words)
    ]


timetable_tool = TimetableTool()
