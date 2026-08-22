"""
Per-turn accounting for external calls.

The audit that motivated these changes could only be written by reading the
code and multiplying the retry layers together by hand. That is not a thing
anyone should have to do twice, so the counts are now measured rather than
derived: how many logical LLM calls a turn made, how many HTTP attempts those
became, how many were rate-limited, how many embeddings and Qdrant operations
it took, and how long the phases ran.

A `ContextVar` rather than a parameter threaded through forty call sites.
`asyncio.create_task` copies the current context, so work a turn spawns is
counted against the turn that spawned it — including the detached memory
writes, which is the honest attribution: they are that turn's cost even though
the user does not wait for them.

Outside a turn (the background memory worker, a script, a health probe) the
context var is unset and every helper is a no-op. Instrumentation must never be
the reason something fails, so nothing here raises.
"""
from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    """External-call counts and phase timings for one logical turn."""

    request_id: str = ""

    # Groq
    llm_logical_calls: int = 0      # times the application asked for a completion
    llm_http_attempts: int = 0      # times an HTTP request actually went out
    llm_retries: int = 0            # attempts that were retries of a previous one
    llm_rate_limited: int = 0       # attempts rejected with 429
    llm_timeouts: int = 0
    llm_failures: int = 0
    llm_estimated_tokens: int = 0

    # Cohere
    embed_calls: int = 0            # API calls actually issued
    embed_cache_hits: int = 0       # served from the 60s LRU
    embed_coalesced: int = 0        # joined an in-flight call instead of issuing one

    # Qdrant
    qdrant_ops: int = 0

    # Phase timings, seconds
    timings: Dict[str, float] = field(default_factory=dict)

    def time(self, phase: str, seconds: float) -> None:
        self.timings[phase] = round(self.timings.get(phase, 0.0) + seconds, 4)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "llm_logical_calls": self.llm_logical_calls,
            "llm_http_attempts": self.llm_http_attempts,
            "llm_retries": self.llm_retries,
            "llm_rate_limited": self.llm_rate_limited,
            "llm_timeouts": self.llm_timeouts,
            "llm_failures": self.llm_failures,
            "llm_estimated_tokens": self.llm_estimated_tokens,
            "embed_calls": self.embed_calls,
            "embed_cache_hits": self.embed_cache_hits,
            "embed_coalesced": self.embed_coalesced,
            "qdrant_ops": self.qdrant_ops,
            **{f"t_{k}": v for k, v in sorted(self.timings.items())},
        }


_current: contextvars.ContextVar[Optional[TurnMetrics]] = contextvars.ContextVar(
    "turn_metrics", default=None
)


def current() -> Optional[TurnMetrics]:
    """The metrics for the turn in progress, or None outside one."""
    return _current.get()


@contextmanager
def turn(request_id: str = ""):
    """Open a measurement scope. Yields the `TurnMetrics` being filled in."""
    metrics = TurnMetrics(request_id=request_id)
    token = _current.set(metrics)
    started = time.monotonic()
    try:
        yield metrics
    finally:
        metrics.time("total", time.monotonic() - started)
        _current.reset(token)


@contextmanager
def phase(name: str):
    """Time one phase of the turn (`retrieval`, `planner`, `router`, ...)."""
    started = time.monotonic()
    try:
        yield
    finally:
        metrics = _current.get()
        if metrics is not None:
            metrics.time(name, time.monotonic() - started)


# ── Counters ─────────────────────────────────────────────────────────────
# Each is a no-op outside a turn scope.

def _bump(field_name: str, amount: int = 1) -> None:
    metrics = _current.get()
    if metrics is None:
        return
    try:
        setattr(metrics, field_name, getattr(metrics, field_name) + amount)
    except Exception:  # pragma: no cover — accounting must never fail a turn
        logger.debug("Could not record metric %s", field_name)


def record_llm_request(estimated_tokens: int = 0) -> None:
    """One logical completion request, before any HTTP attempt."""
    _bump("llm_logical_calls")
    if estimated_tokens:
        _bump("llm_estimated_tokens", estimated_tokens)


def record_groq_call(model: Optional[str] = None, stream: bool = False) -> None:
    """One HTTP request actually issued to Groq."""
    _bump("llm_http_attempts")


def record_llm_retry() -> None:
    _bump("llm_retries")


def record_llm_rate_limited() -> None:
    _bump("llm_rate_limited")


def record_llm_timeout() -> None:
    _bump("llm_timeouts")


def record_llm_failure() -> None:
    _bump("llm_failures")


def record_embed_call(count: int = 1) -> None:
    _bump("embed_calls", count)


def record_embed_cache_hit() -> None:
    _bump("embed_cache_hits")


def record_embed_coalesced() -> None:
    _bump("embed_coalesced")


def record_qdrant_op(count: int = 1) -> None:
    _bump("qdrant_ops", count)
