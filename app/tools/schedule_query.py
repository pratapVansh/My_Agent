"""
Turning "tomorrow" into a date, without asking a model.

The existing route to the timetable was `class_suggestions(day_of_week: int)`.
To answer "what classes do I have tomorrow" a model had to work out today's
date, add a day, derive the weekday, and convert it to 0–6 — four arithmetic
steps, in a component that cannot do arithmetic reliably and has no clock. It
also had to do them from a prompt line that says what today is, which it may or
may not read. That is the whole reason this module exists: the model's job is
reduced to passing the user's own word through, and every calculation happens
here in Python against `time_tool`.

One clock. `time_tool` reads the configured `assistant_timezone`; the academic
agent used bare `date.today()`, which is the host's. In a datacentre those are
different dates for several hours a day, and "what have I got tomorrow" asked
at 11pm is exactly when the difference shows up. Everything here goes through
`time_tool` and nothing calls `date.today()`.

Resolution is total or it is nothing: an unrecognised phrase returns `None` and
the caller reports that it could not tell which day was meant. It never falls
back to "today", because silently answering about the wrong day is worse than
asking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, time, timedelta
from typing import List, Optional, Tuple

from app.tools import time_tool

_WEEKDAY_NAMES: Tuple[str, ...] = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
)

_WEEKDAY_LOOKUP = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}

# Phrases that name a span rather than a day.
_THIS_WEEK_RE = re.compile(r"\bthis\s+week\b|\bwhole\s+week\b|\bweekly\b|\bmy\s+week\b")
_NEXT_WEEK_RE = re.compile(r"\bnext\s+week\b")

_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DMY_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")

# "at 10", "at 10 am", "10:30", "10 o'clock"
_AT_TIME_RE = re.compile(
    r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.|o'?clock)?\b",
    re.IGNORECASE,
)


def weekday_name(day_of_week: int) -> str:
    return _WEEKDAY_NAMES[day_of_week % 7]


# ── The resolved window ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScheduleWindow:
    """A resolved span of dates, and how to describe it back to the user."""

    start: date
    end: date
    label: str
    """Human phrasing for the answer — "tomorrow, Friday 14 August"."""

    kind: str = "day"
    """`day` or `range`. A range never says "you have no classes tomorrow"."""

    def dates(self) -> List[date]:
        span = (self.end - self.start).days
        return [self.start + timedelta(days=offset) for offset in range(span + 1)]

    def summary(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "label": self.label,
            "kind": self.kind,
        }


def _describe(target: date, *, today: date) -> str:
    """"tomorrow, Friday 14 August" — the relative word first, because that is
    what the user said and it is how they will check the answer."""
    weekday = weekday_name(target.weekday())
    stamp = target.strftime("%d %B").lstrip("0")
    delta = (target - today).days
    if delta == 0:
        return f"today, {weekday} {stamp}"
    if delta == 1:
        return f"tomorrow, {weekday} {stamp}"
    if delta == -1:
        return f"yesterday, {weekday} {stamp}"
    return f"{weekday} {stamp}"


def _day(target: date, *, today: date) -> ScheduleWindow:
    return ScheduleWindow(target, target, _describe(target, today=today), "day")


def next_weekday_on_or_after(reference: date, day_of_week: int) -> date:
    """
    The next date falling on `day_of_week`, counting today as eligible.

    "What do I have on Monday?" asked *on* a Monday means today. Skipping to
    next week would be a different question, and one the user did not ask.
    """
    ahead = (day_of_week - reference.weekday()) % 7
    return reference + timedelta(days=ahead)


def resolve_when(text: str, *, today: Optional[date] = None) -> Optional[ScheduleWindow]:
    """
    Resolve a phrase to dates, or return None.

    Order matters. "next week" is checked before the weekday names so that
    phrase is not consumed by a stray "mon" inside it, and explicit dates are
    checked before relative words because a sentence carrying both means the
    date.
    """
    reference = today or time_tool.today()
    lowered = (text or "").strip().lower()
    if not lowered:
        return None

    # ── Explicit dates ───────────────────────────────────────────────────────
    iso = _ISO_RE.search(lowered)
    if iso:
        try:
            target = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None
        return _day(target, today=reference)

    dmy = _DMY_RE.search(lowered)
    if dmy:
        try:
            target = date(int(dmy.group(3)), int(dmy.group(2)), int(dmy.group(1)))
        except ValueError:
            return None
        return _day(target, today=reference)

    # ── Spans ────────────────────────────────────────────────────────────────
    if _NEXT_WEEK_RE.search(lowered):
        monday = reference + timedelta(days=(7 - reference.weekday()))
        return ScheduleWindow(monday, monday + timedelta(days=6), "next week", "range")

    if _THIS_WEEK_RE.search(lowered):
        monday = reference - timedelta(days=reference.weekday())
        return ScheduleWindow(monday, monday + timedelta(days=6), "this week", "range")

    # ── Relative days ────────────────────────────────────────────────────────
    # `time_tool` already owns this vocabulary; asking it keeps "tomorrow"
    # meaning one thing across the whole application.
    relative = time_tool.resolve_relative_day(lowered, reference=reference)
    if relative is not None:
        return _day(relative, today=reference)

    # ── Named weekdays ───────────────────────────────────────────────────────
    for word, index in _WEEKDAY_LOOKUP.items():
        if re.search(rf"\b{word}\b", lowered):
            return _day(next_weekday_on_or_after(reference, index), today=reference)

    return None


def resolve_at_time(text: str) -> Optional[time]:
    """
    Pull "10 AM" out of a question, or return None.

    Deliberately conservative. A bare number with no am/pm and no colon is not
    treated as a time — "do I have 2 classes tomorrow" must not become a query
    about 02:00. Only an explicit meridiem, an explicit `HH:MM`, or "o'clock"
    counts.
    """
    lowered = (text or "").lower()
    for match in _AT_TIME_RE.finditer(lowered):
        hour_raw, minute_raw, meridiem = match.group(1), match.group(2), match.group(3)
        if not meridiem and minute_raw is None:
            continue
        try:
            hour = int(hour_raw)
        except ValueError:
            continue
        if not 0 <= hour <= 23:
            continue
        minute = int(minute_raw) if minute_raw else 0
        if not 0 <= minute <= 59:
            continue
        if meridiem:
            flag = meridiem.replace(".", "")
            if flag.startswith("p") and hour < 12:
                hour += 12
            elif flag.startswith("a") and hour == 12:
                hour = 0
        return time(hour=hour, minute=minute)
    return None


# ── Rendering ────────────────────────────────────────────────────────────────
#
# Deterministic, so what the user reads is the rows themselves. The reasoning
# loop is still free to phrase around this, but `grounding` guarantees it cannot
# replace it with something the tool did not return.

def _clock(value: str) -> str:
    """`"09:30"` → `"9:30 am"`. Spoken answers should not read out "zero nine"."""
    try:
        hour, minute = (int(part) for part in value.split(":")[:2])
    except (ValueError, IndexError):
        return value
    suffix = "am" if hour < 12 else "pm"
    display = hour % 12 or 12
    return f"{display}:{minute:02d} {suffix}"


def merge_adjacent(rows: List[dict]) -> List[dict]:
    """
    Collapse back-to-back slots of the same class into one block.

    The grid stores one row per printed cell, which is the right unit for
    provenance and the wrong one for reading aloud: Thursday's Generative AI
    occupies two consecutive cells and is one two-hour class to the person
    attending it. Merging only ever joins rows that agree on subject, session
    type and room, and only when one ends exactly where the next begins.
    """
    if not rows:
        return []

    ordered = sorted(rows, key=lambda r: (r.get("date", ""), r.get("start_time", "")))
    merged: List[dict] = [dict(ordered[0])]

    for row in ordered[1:]:
        previous = merged[-1]
        same_class = (
            row.get("date") == previous.get("date")
            and row.get("subject") == previous.get("subject")
            and row.get("session_type") == previous.get("session_type")
            and row.get("room") == previous.get("room")
            and previous.get("end_time") == row.get("start_time")
        )
        if same_class:
            previous["end_time"] = row.get("end_time")
            previous.setdefault("merged_ids", [previous.get("id")])
            previous["merged_ids"].append(row.get("id"))
        else:
            merged.append(dict(row))
    return merged


def render_day(label: str, rows: List[dict]) -> str:
    """The answer for one day. Empty is an answer, not a failure."""
    if not rows:
        return f"You have no classes {label}."

    # `.capitalize()` would lowercase the rest — "Tomorrow, friday 14 august".
    heading = label[:1].upper() + label[1:]
    lines = [f"{heading}, you have:"]
    for row in merge_adjacent(rows):
        line = (
            f"  {_clock(row['start_time'])}–{_clock(row['end_time'])} — "
            f"{row['subject']}"
        )
        if row.get("session_type") == "LAB":
            line += " (lab)"
        extras = []
        if row.get("room"):
            extras.append(row["room"])
        if row.get("faculty"):
            extras.append(row["faculty"])
        if extras:
            line += f" · {' · '.join(extras)}"
        lines.append(line)
    return "\n".join(lines)


def render_range(label: str, rows: List[dict]) -> str:
    """The answer for a span. Grouped by day, days without classes named."""
    if not rows:
        return f"You have no classes {label}."

    by_date: dict = {}
    for row in rows:
        by_date.setdefault(row.get("date"), []).append(row)

    blocks = [f"Your schedule for {label}:"]
    for day in sorted(by_date):
        heading = _WEEKDAY_NAMES[date.fromisoformat(day).weekday()]
        blocks.append("")
        blocks.append(f"{heading} ({day}):")
        for row in merge_adjacent(by_date[day]):
            line = (
                f"  {_clock(row['start_time'])}–{_clock(row['end_time'])} — "
                f"{row['subject']}"
            )
            if row.get("session_type") == "LAB":
                line += " (lab)"
            blocks.append(line)
    return "\n".join(blocks)


# ── The answer ───────────────────────────────────────────────────────────────

NO_TIMETABLE = (
    "I don't have your timetable yet. Send it over and I can answer from it."
)


def render_from_tool_result(payload: dict) -> Optional[str]:
    """
    Build the answer from a schedule tool's own output, or None.

    Returning the rendered rows rather than letting the model restate them is
    the same arrangement the action gateway uses for a held action: the model's
    version is not checked, it is not asked for. A timetable answer is a list of
    facts with times attached, and a paraphrase of it is strictly worse — every
    word of drift is a class the user might miss.

    None means "this payload is not a schedule answer", and the caller leaves
    the model's text alone.
    """
    if not isinstance(payload, dict):
        return None

    tool = payload.get("tool")
    if tool not in ("get_schedule", "get_next_class"):
        return None

    # The tool could not work out which day was meant. Its own message says so
    # precisely; anything generated here would be vaguer.
    if payload.get("success") is False:
        return payload.get("message") or None

    # No timetable at all is a different statement from an empty day, and
    # collapsing them tells a user with no uploaded schedule that they are free.
    if not payload.get("has_timetable"):
        return NO_TIMETABLE

    rows = payload.get("classes") or []

    if tool == "get_next_class":
        if not payload.get("found"):
            return "You have no more classes scheduled this week."
        row = merge_adjacent(rows)[0]
        label = payload.get("label") or "next"
        line = (
            f"Your next class is {row['subject']} "
            f"{label}, {_clock(row['start_time'])}–{_clock(row['end_time'])}"
        )
        if row.get("session_type") == "LAB":
            line += " (lab)"
        extras = [x for x in (row.get("room"), row.get("faculty")) if x]
        if extras:
            line += f" · {' · '.join(extras)}"
        return line + "."

    window = payload.get("window") or {}
    label = window.get("label") or "that day"
    filters = payload.get("filters") or {}

    if filters.get("subject") and not rows:
        return (
            f"I don't see {filters['subject']} on your timetable {label}."
        )
    if filters.get("at_time") and not rows:
        return f"You have no class at {_clock(filters['at_time'])} {label}."

    if window.get("kind") == "range":
        return render_range(label, rows)
    return render_day(label, rows)


__all__ = [
    "NO_TIMETABLE",
    "ScheduleWindow",
    "render_from_tool_result",
    "merge_adjacent",
    "next_weekday_on_or_after",
    "render_day",
    "render_range",
    "resolve_at_time",
    "resolve_when",
    "weekday_name",
]
