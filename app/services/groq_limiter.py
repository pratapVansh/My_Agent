"""
Back-pressure in front of Groq.

Every other guard in this system limits the wrong thing. The HTTP rate limiter
counts requests per client per minute, but one request fans out into a planner
call, up to three reasoning iterations and a reflect retry. The per-call
timeouts bound how long a single call may take, not how many run at once. And
the background memory worker calls the same account on its own schedule with no
user attached to it at all.

So the account sees bursts, answers 429, and the retry layers turn each 429
into more requests. This module is the missing piece: one gate, shared by every
Groq entry point, that converts a burst into a queue.

Two independent constraints, both required:

* **Concurrency** — a semaphore. Bounds how many requests are open at once,
  which is what stops a fan-out from arriving as a spike.
* **Token budget** — a leaky bucket refilled continuously at TPM/60 per second.
  Concurrency alone does not bound *volume*: four concurrent 8,000-token
  requests exceed a 24,000 TPM budget however politely they queue.

The token estimate is deliberately crude — characters ÷ 4, plus the reply
allowance. It does not need to be right, it needs to be *conservative and
cheap*: `settings.groq_tokens_per_minute` is set below the account's real limit
precisely so the estimate's error is absorbed by headroom rather than by a 429.

Not a retry layer. This admits or delays a request; it never re-sends one.
Retry lives in `BaseAgent.call_groq`, and it lives there alone.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from app.config import settings
from app.services import deadline as _deadline

logger = logging.getLogger(__name__)

# Rough characters-per-token for English prompt text. Groq's tokenizer is not
# this, and does not need to be — see the module docstring on why the estimate
# may be wrong so long as it is not wildly optimistic.
_CHARS_PER_TOKEN = 4

# A request is never allowed to want more than this share of one minute's
# budget. Without the clamp, a single oversized prompt could ask for more than
# the bucket can ever hold and wait forever for capacity that cannot arrive.
_MAX_SHARE_OF_BUDGET = 0.5

# Ceiling on how long a caller waits for token capacity before being let
# through anyway. Blocking indefinitely would convert a budget overrun into a
# hang, and a hang is worse than a 429: the 429 is visible and retryable, the
# hang consumes the caller's own deadline in silence.
#
# It is a ceiling and no longer the only bound. A spoken turn is given 35 s in
# total, and 20 s of it spent here left the retry layer below with a budget it
# could not honour — the turn expired while sleeping, having sent nothing. The
# wait is now also clamped to whatever `app.services.deadline` says is left, so
# a caller with 4 s to live waits 4 s and then proceeds or fails, visibly.
_MAX_WAIT_SECONDS = 20.0


def estimate_tokens(
    messages: Optional[List[Dict[str, Any]]] = None,
    max_tokens: Optional[int] = None,
) -> int:
    """
    Approximate the token cost of one request: prompt in, allowance out.

    `max_tokens` is counted in full even though most replies are shorter. The
    budget exists to stop the account being oversubscribed, and reserving the
    worst case is the only way an estimate made *before* the call can do that.
    """
    prompt_chars = 0
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            prompt_chars += len(content)
        elif content is not None:
            # Multimodal / structured content: fall back to its repr length
            # rather than skipping it, so it is never counted as free.
            prompt_chars += len(str(content))

    prompt_tokens = prompt_chars // _CHARS_PER_TOKEN
    completion_tokens = int(max_tokens) if max_tokens else 0
    return max(1, prompt_tokens + completion_tokens)


class Reservation:
    """
    What one admitted request holds, and how it gives back what it did not use.

    The estimate made *before* a call has to reserve `max_tokens` in full,
    because the reply's real length is not knowable yet and a budget that
    under-reserves is not a budget. But the reply is almost never `max_tokens`
    long — a reasoning step allowed 700 typically spends 120 — and until now the
    difference was never returned. Every request therefore drew its worst case
    against a minute's budget permanently, and the bucket ran empty against
    spending that had not happened.

    Groq reports the truth in `usage.total_tokens`. `settle` puts the
    difference back, so the budget tracks the account rather than the ceiling.
    Reserving high and settling low is the only ordering that is safe in both
    directions: the account is never oversubscribed while a call is in flight,
    and it is never under-used once the call has landed.

    Settling is optional. A stream, or a call that raised, simply never settles
    and keeps its full reservation — the conservative outcome, and the one the
    limiter had before.
    """

    __slots__ = ("_limiter", "_reserved", "_settled")

    def __init__(self, limiter: "GroqLimiter", reserved: float) -> None:
        self._limiter = limiter
        self._reserved = float(reserved)
        self._settled = False

    @property
    def reserved(self) -> float:
        return self._reserved

    async def settle(self, actual_tokens: Optional[int]) -> None:
        """Return the reserved-but-unspent difference to the bucket. Once."""
        if self._settled or not actual_tokens or actual_tokens <= 0:
            return
        self._settled = True
        refund = self._reserved - float(actual_tokens)
        if refund > 0:
            await self._limiter._refund(refund)


class GroqLimiter:
    """One shared gate: bounded concurrency plus a token-per-minute budget."""

    def __init__(
        self,
        max_concurrency: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
    ) -> None:
        self._configured_concurrency = max_concurrency
        self._configured_tpm = tokens_per_minute

        self._semaphore: Optional[asyncio.Semaphore] = None
        self._token_lock: Optional[asyncio.Lock] = None
        # The loop the primitives above were created on. asyncio primitives
        # bind to a loop on first use, and the test suite runs each test on a
        # fresh one — so a module-level singleton has to notice the change and
        # rebuild rather than raise "attached to a different loop".
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._available_tokens: float = float(self.tokens_per_minute)
        self._last_refill: float = time.monotonic()

        # Observability. `run_workflow` reads these to report per-turn cost —
        # see `app.services.call_metrics`.
        self.admitted: int = 0
        self.delayed: int = 0
        self.total_wait_seconds: float = 0.0
        # Requests let through with the budget still short — the ones the
        # provider is entitled to answer with a 429. A non-zero count is the
        # signal that the turn is asking for more than the tier grants, and it
        # is the number to look at before touching any timeout.
        self.admitted_over_budget: int = 0
        # Reserved-minus-actual, returned to the bucket after the fact. See
        # `settle`.
        self.refunded_tokens: float = 0.0

    # ── Configuration ────────────────────────────────────────────────────

    @property
    def max_concurrency(self) -> int:
        if self._configured_concurrency is not None:
            return max(1, self._configured_concurrency)
        return max(1, settings.groq_max_concurrency)

    @property
    def tokens_per_minute(self) -> int:
        if self._configured_tpm is not None:
            return max(1, self._configured_tpm)
        return max(1, settings.groq_tokens_per_minute)

    # ── Loop-safe primitive construction ─────────────────────────────────

    def _primitives(self) -> tuple[asyncio.Semaphore, asyncio.Lock]:
        """The semaphore and lock for the *currently running* loop."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop or self._semaphore is None or self._token_lock is None:
            self._loop = loop
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
            self._token_lock = asyncio.Lock()
            # A new loop means a new run: start it with a full bucket rather
            # than whatever balance the previous one happened to leave behind.
            self._available_tokens = float(self.tokens_per_minute)
            self._last_refill = time.monotonic()
        return self._semaphore, self._token_lock

    # ── Token bucket ─────────────────────────────────────────────────────

    def _refill_locked(self) -> None:
        """Add the tokens that have accrued since the last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        if elapsed <= 0:
            return
        capacity = float(self.tokens_per_minute)
        self._available_tokens = min(
            capacity, self._available_tokens + elapsed * (capacity / 60.0)
        )

    async def _spend(self, tokens: int) -> float:
        """
        Wait until `tokens` of budget are available, then spend them.

        Returns the amount actually reserved, which is what `settle` needs in
        order to give the unused part back — it is not always `tokens`, because
        an oversized request is clamped below.
        """
        _, token_lock = self._primitives()
        capacity = float(self.tokens_per_minute)
        # Clamp before waiting. A request larger than the whole bucket would
        # otherwise wait for capacity that can never exist.
        want = min(float(tokens), capacity * _MAX_SHARE_OF_BUDGET)

        # Two bounds, and the shorter wins. The first is this module's own
        # patience; the second is how long the turn will still be listening.
        # Waiting past the caller's deadline is not back-pressure, it is a
        # stall that produces neither an answer nor an error in time to matter.
        budget_seconds = _deadline.clamp(_MAX_WAIT_SECONDS)
        deadline = time.monotonic() + budget_seconds
        waited = 0.0

        while True:
            async with token_lock:
                self._refill_locked()
                if self._available_tokens >= want:
                    self._available_tokens -= want
                    if waited > 0:
                        self.delayed += 1
                        self.total_wait_seconds += waited
                    return want
                shortfall = want - self._available_tokens
                sleep_for = shortfall / (capacity / 60.0)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Let it through rather than hang. The provider's own limiter
                # is the backstop, and a visible 429 beats a silent stall.
                async with token_lock:
                    self._available_tokens = max(0.0, self._available_tokens - want)
                logger.warning(
                    "Groq token budget still short after %.1fs (want ~%d tokens, "
                    "budget %d/min) — admitting anyway to avoid stalling the "
                    "caller. This is the 429 the limiter could not prevent: the "
                    "turn is asking for more than the tier grants, and the fix "
                    "is fewer or smaller calls, not a longer wait.",
                    budget_seconds, int(want), self.tokens_per_minute,
                )
                self.delayed += 1
                self.admitted_over_budget += 1
                self.total_wait_seconds += budget_seconds
                return want

            # Cap each nap so a raised budget or a released reservation is
            # noticed promptly rather than slept through.
            nap = max(0.01, min(sleep_for, remaining, 1.0))
            waited += nap
            await asyncio.sleep(nap)

    async def _refund(self, tokens: float) -> None:
        """
        Give unspent budget back, never exceeding the bucket's capacity.

        Capped for the same reason `_refill_locked` caps: a bucket holding more
        than a minute's worth would let the next burst spend a minute it has
        not lived through yet.
        """
        if tokens <= 0:
            return
        _, token_lock = self._primitives()
        capacity = float(self.tokens_per_minute)
        async with token_lock:
            self._refill_locked()
            before = self._available_tokens
            self._available_tokens = min(capacity, before + tokens)
            self.refunded_tokens += self._available_tokens - before

    # ── Public API ───────────────────────────────────────────────────────

    @asynccontextmanager
    async def reserve(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Hold a slot for one Groq request.

        Streaming and non-streaming both pass through here. For a stream the
        slot is held for the lifetime of the *iteration*, not just the initial
        response, because that is how long the connection is actually open.

        Yields a `Reservation`. A caller that learns the request's real cost —
        a non-streaming completion does, from `usage.total_tokens` — should
        `await` its `settle`, which returns the over-reserved difference to the
        budget. Ignoring the handle is safe and keeps the full reservation.
        """
        if not settings.groq_limiter_enabled:
            yield Reservation(self, 0.0)
            return

        semaphore, _ = self._primitives()
        tokens = estimate_tokens(messages, max_tokens)

        reserved = await self._spend(tokens)
        await semaphore.acquire()
        self.admitted += 1
        try:
            yield Reservation(self, reserved)
        finally:
            semaphore.release()

    def snapshot(self) -> Dict[str, Any]:
        """Current counters, for logging and tests."""
        return {
            "admitted": self.admitted,
            "delayed": self.delayed,
            "total_wait_seconds": round(self.total_wait_seconds, 3),
            "admitted_over_budget": self.admitted_over_budget,
            "refunded_tokens": round(self.refunded_tokens, 1),
            "available_tokens": round(self._available_tokens, 1),
            "max_concurrency": self.max_concurrency,
            "tokens_per_minute": self.tokens_per_minute,
        }

    def reset(self) -> None:
        """Drop all state. Tests only."""
        self._semaphore = None
        self._token_lock = None
        self._loop = None
        self._available_tokens = float(self.tokens_per_minute)
        self._last_refill = time.monotonic()
        self.admitted = 0
        self.delayed = 0
        self.total_wait_seconds = 0.0
        self.admitted_over_budget = 0
        self.refunded_tokens = 0.0


# The shared gate. Every Groq entry point uses this one instance — a limiter
# per caller would bound each caller and none of the account.
groq_limiter = GroqLimiter()
