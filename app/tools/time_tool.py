"""
The clock.

"What is today's date?" was answered with "I don't have real-time access",
which was true of the memory system and irrelevant to the question. Nothing in
the assistant could read a clock: the academic agent parsed `"today"` into a
`date` internally, but that capability was private to timetable tools and no
route reached it from a plain question.

Memory must never answer this. A stored sentence saying "today is the 9th" was
true when it was written and is false now, and it will be retrieved by
similarity for years. Current state comes from a capability, not from recall —
that separation is the point of the module, not the two lines of arithmetic.

Timezone comes from configuration rather than the host clock's locale, because
the host is a datacentre in another country. A server in UTC telling a user in
Asia/Kolkata that it is still yesterday is the same class of error as answering
from memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Relative day words the assistant is expected to resolve without asking.
_RELATIVE_DAYS: Dict[str, int] = {
    "today": 0, "tonight": 0, "now": 0, "currently": 0,
    "tomorrow": 1, "tmrw": 1, "next day": 1,
    "yesterday": -1,
    "day after tomorrow": 2, "overmorrow": 2,
    "day before yesterday": -2,
}

_WEEKDAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
)


def _zone():
    """The assistant's configured timezone, or the host's if unset/invalid."""
    from app.config import settings

    name = (getattr(settings, "assistant_timezone", "") or "").strip()
    if not name:
        return None
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception as exc:  # unknown zone, or no tzdata on this platform
        logger.warning(
            "Unknown assistant_timezone %r (%s); falling back to host local time.",
            name, exc,
        )
        return None


def now() -> datetime:
    """The current moment in the assistant's timezone."""
    zone = _zone()
    return datetime.now(zone) if zone else datetime.now()


def today() -> date:
    """The current date in the assistant's timezone."""
    return now().date()


def resolve_relative_day(phrase: str, *, reference: Optional[date] = None) -> Optional[date]:
    """
    Turn "today"/"tomorrow"/"yesterday" into a date.

    Returns None for anything it does not recognise, so a caller can fall back
    to real date parsing rather than silently answering about the wrong day.
    """
    key = (phrase or "").strip().lower()
    if not key:
        return None
    base = reference or today()

    # Longest match first: "day after tomorrow" must not match "tomorrow".
    for word in sorted(_RELATIVE_DAYS, key=len, reverse=True):
        if word in key:
            return base + timedelta(days=_RELATIVE_DAYS[word])
    return None


@dataclass(frozen=True)
class TemporalContext:
    """The current moment, in the forms an answer or a prompt needs."""

    moment: datetime
    timezone_name: str

    @property
    def date(self) -> date:
        return self.moment.date()

    @property
    def weekday(self) -> str:
        return _WEEKDAYS[self.moment.weekday()]

    def date_sentence(self) -> str:
        """A complete answer to "what is today's date?"."""
        return (
            f"Today is {self.weekday}, "
            f"{self.moment.strftime('%d %B %Y').lstrip('0')}."
        )

    def time_sentence(self) -> str:
        """A complete answer to "what time is it?"."""
        clock = self.moment.strftime("%I:%M %p").lstrip("0")
        zone = f" ({self.timezone_name})" if self.timezone_name else ""
        return f"It's {clock}{zone} on {self.weekday}, {self.moment.strftime('%d %B %Y').lstrip('0')}."

    def prompt_line(self) -> str:
        """
        One line injected into agent prompts.

        Every agent gets this, not just the ones answering clock questions: a
        model that does not know today's date cannot resolve "next Friday",
        cannot tell whether a deadline has passed, and — as observed — cannot
        tell that a résumé's "expected 2027" is in the future.
        """
        return (
            f"Current date and time: {self.weekday}, "
            f"{self.moment.strftime('%Y-%m-%d %H:%M')}"
            f"{f' {self.timezone_name}' if self.timezone_name else ''}."
        )

    def summary(self) -> Dict[str, str]:
        return {
            "date": self.date.isoformat(),
            "weekday": self.weekday,
            "time": self.moment.strftime("%H:%M"),
            "timezone": self.timezone_name,
        }


def current_context() -> TemporalContext:
    """The clock, packaged for prompts and for direct answers."""
    moment = now()
    zone = moment.tzinfo
    name = ""
    if zone is not None:
        name = getattr(zone, "key", "") or str(zone)
    return TemporalContext(moment=moment, timezone_name=name)


def answer_temporal_query(query: str) -> str:
    """
    Answer a date/time question directly, with no model call.

    A deterministic question deserves a deterministic answer: routing "what is
    the date" through an LLM adds latency and a chance of being wrong about
    something arithmetic.
    """
    context = current_context()
    text = (query or "").lower()

    wants_time = any(word in text for word in ("time", "clock", "hour", "o'clock"))
    wants_date = any(word in text for word in ("date", "day", "today", "month", "year"))

    if wants_time and not wants_date:
        return context.time_sentence()
    if wants_time and wants_date:
        return context.time_sentence()

    if "tomorrow" in text:
        target = context.date + timedelta(days=1)
        return f"Tomorrow is {_WEEKDAYS[target.weekday()]}, {target.strftime('%d %B %Y').lstrip('0')}."
    if "yesterday" in text:
        target = context.date - timedelta(days=1)
        return f"Yesterday was {_WEEKDAYS[target.weekday()]}, {target.strftime('%d %B %Y').lstrip('0')}."

    return context.date_sentence()


def as_tool_result(query: str = "") -> Dict[str, object]:
    """Tool-shaped result, for agents that reach the clock through their loop."""
    context = current_context()
    return {
        "success": True,
        "answer": answer_temporal_query(query),
        **context.summary(),
    }
