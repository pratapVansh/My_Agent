"""
How much of the turn's budget is left, readable from anywhere inside it.

Every waiting layer in this system had its own ceiling and none of them knew
about the others. A spoken turn is given 35 s by
`settings.voice_workflow_timeout_seconds`; inside it the Groq limiter would
wait up to 20 s for token budget, and `BaseAgent.call_groq` would honour a
`Retry-After` of up to 15 s. Two of those in sequence is 35 s of pure waiting
inside a 35 s budget — the turn was guaranteed to expire *while sleeping*, and
the sleep produced nothing. Raising the ceiling would only move the number; the
defect is that the sub-budgets were never bounded by the parent.

So the parent publishes its deadline once and the layers below read it. Nobody
sleeps past the moment their answer stops being wanted, and a wait that cannot
finish in time is abandoned immediately rather than entered and then cut off.

A `ContextVar`, for the same reason `app.services.call_metrics` is one: the
callers are forty deep and threading a parameter through all of them would be a
larger change than the bug. `asyncio.create_task` copies the context, so work a
turn spawns inherits the turn's deadline. Outside a turn — the background
memory worker, a script, a health probe — nothing is set and every helper here
reports "no deadline", which is the previous behaviour exactly.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

_deadline: ContextVar[Optional[float]] = ContextVar("turn_deadline", default=None)

# Never hand back the last sliver of the budget. A wait that finishes exactly on
# the deadline leaves no time to do anything with what it waited for, so the
# call it was protecting would be cancelled anyway — with the wait already paid.
_RESERVE_SECONDS = 0.25


@contextmanager
def budget(seconds: Optional[float]):
    """Publish a deadline `seconds` from now for the duration of the block."""
    if seconds is None or seconds <= 0:
        token = _deadline.set(None)
    else:
        token = _deadline.set(time.monotonic() + float(seconds))
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining() -> Optional[float]:
    """
    Seconds left in the turn, or None when no deadline is in force.

    Can be zero or negative: the caller is past its deadline and should stop
    rather than start something new.
    """
    at = _deadline.get()
    if at is None:
        return None
    return at - time.monotonic()


def clamp(seconds: float) -> float:
    """
    The longest it is worth sleeping for `seconds`, given the turn's budget.

    Returns 0.0 when there is no useful time left, which callers read as "do
    not wait at all" — either proceed immediately or fail now. Without a
    deadline in force the request is granted unchanged.
    """
    left = remaining()
    if left is None:
        return max(0.0, float(seconds))
    return max(0.0, min(float(seconds), left - _RESERVE_SECONDS))


def exhausted() -> bool:
    """True when a deadline is in force and there is no usable time left."""
    left = remaining()
    return left is not None and left <= _RESERVE_SECONDS


__all__ = ["budget", "clamp", "exhausted", "remaining"]
