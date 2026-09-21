"""
The retry stack, and the gate in front of it.

These tests exist because the audit's central finding could not be observed
from any single place in the code: one logical completion became up to 66 HTTP
requests only when the SDK's retry, `call_groq`'s retry, the reasoning loop and
the reflect loop were multiplied together. Each layer looked reasonable alone.

So what is asserted here is the *count*, not the shape. A test that checks
"there is backoff" would have passed against the broken version.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from app.agents import base_agent
from app.agents.base_agent import BaseAgent
from app.config import settings
from app.services import call_metrics
from app.services.groq_limiter import GroqLimiter, estimate_tokens
from app.services.llm_errors import (
    LLMErrorKind,
    classify_llm_error,
    retry_after_seconds,
)


class _Agent(BaseAgent):
    """Minimal concrete agent — `call_groq` is what is under test."""

    def __init__(self):
        super().__init__(name="probe", description="test agent")

    async def execute(self, state):  # pragma: no cover — never called
        return state


@pytest.fixture
def agent():
    return _Agent()


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Record what the retry loop *would* have waited, without waiting."""
    slept: list[float] = []

    async def _sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(base_agent.asyncio, "sleep", _sleep)
    return slept


class _Response:
    """Just enough of an httpx response for the classifier to read."""

    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}


class _ApiError(Exception):
    def __init__(self, status_code: int, headers: dict | None = None, message: str = "boom"):
        super().__init__(message)
        self.status_code = status_code
        self.response = _Response(status_code, headers)


# ═══════════════════════════════════════════════════════════════════════════
# A · 429 is not an ordinary failure
# ═══════════════════════════════════════════════════════════════════════════

def test_a_429_is_classified_as_rate_limited():
    assert classify_llm_error(_ApiError(429)) is LLMErrorKind.RATE_LIMITED


def test_a_500_is_transient_and_a_401_is_permanent():
    assert classify_llm_error(_ApiError(503)) is LLMErrorKind.TRANSIENT
    assert classify_llm_error(_ApiError(401)) is LLMErrorKind.PERMANENT


def test_a_stringified_rate_limit_is_still_recognised():
    """
    Anything that has been repackaged on the way up loses its status code.

    `groq_service` no longer does that, but wrappers and fakes still can, and
    misreading one of these as transient is what produced the 1-second retry
    into a closed window.
    """
    assert classify_llm_error(
        RuntimeError("Groq API error: 429 rate limited")
    ) is LLMErrorKind.RATE_LIMITED


async def test_a_permanent_error_is_not_retried_at_all(agent, monkeypatch):
    """A rejected key is a deployment fault; spending attempts re-learns it."""
    attempts = {"n": 0}

    async def rejected(**kwargs):
        attempts["n"] += 1
        raise _ApiError(401, message="invalid api key")

    monkeypatch.setattr(agent.groq_service, "chat_completion", rejected)

    with pytest.raises(Exception):
        await agent.call_groq(messages=[{"role": "user", "content": "hi"}])

    assert attempts["n"] == 1, "a permanent error must fail on the first attempt"


# ═══════════════════════════════════════════════════════════════════════════
# B · Retry-After is honoured
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"retry-after": "12"}, 12.0),
        ({"retry-after": "6.573s"}, 6.573),
        ({"retry-after": "250ms"}, 0.25),
        ({"x-ratelimit-reset-tokens": "7.66s"}, 7.66),
    ],
)
def test_retry_after_is_read_from_the_response_headers(headers, expected):
    assert retry_after_seconds(_ApiError(429, headers)) == pytest.approx(expected)


def test_retry_after_falls_back_to_the_message_body():
    """Groq states the wait in the body of a 429 even when the header is absent."""
    exc = RuntimeError("Rate limit reached. Please try again in 6.5s.")
    assert retry_after_seconds(exc) == pytest.approx(6.5)


def test_no_guidance_reads_as_none_rather_than_zero():
    """
    None and 0.0 are different answers and the caller treats them differently.

    Collapsing them would make "the provider said nothing" mean "retry now",
    which is exactly the behaviour being removed.
    """
    assert retry_after_seconds(_ApiError(429)) is None


async def test_the_retry_waits_for_the_window_the_provider_named(
    agent, monkeypatch, _no_real_sleeping
):
    attempts = {"n": 0}

    async def limited_then_ok(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _ApiError(429, {"retry-after": "9"})
        return {"content": "recovered"}

    monkeypatch.setattr(agent.groq_service, "chat_completion", limited_then_ok)

    result = await agent.call_groq(messages=[{"role": "user", "content": "hi"}])

    assert result == "recovered"
    assert _no_real_sleeping == [9.0], (
        "the retry must wait the window the provider named, not a fixed backoff"
    )


async def test_a_transient_failure_uses_backoff_not_the_rate_limit_wait(
    agent, monkeypatch, _no_real_sleeping
):
    attempts = {"n": 0}

    async def flaky(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _ApiError(503)
        return {"content": "recovered"}

    monkeypatch.setattr(agent.groq_service, "chat_completion", flaky)

    assert await agent.call_groq(messages=[]) == "recovered"
    assert _no_real_sleeping == [1.0]


async def test_an_impossibly_long_window_fails_fast_instead_of_sleeping(
    agent, monkeypatch, _no_real_sleeping
):
    """
    Honouring a 10-minute Retry-After would outlive every caller's deadline.

    Sleeping through it produces nothing and holds the turn open; failing now
    at least lets the voice path report and the user ask again.
    """
    async def limited(**kwargs):
        raise _ApiError(429, {"retry-after": "600"})

    monkeypatch.setattr(agent.groq_service, "chat_completion", limited)

    with pytest.raises(Exception):
        await agent.call_groq(messages=[])

    assert _no_real_sleeping == [], "must not sleep past the ceiling"


# ═══════════════════════════════════════════════════════════════════════════
# C · Retry lives in exactly one layer
# ═══════════════════════════════════════════════════════════════════════════

def test_the_groq_sdk_does_not_retry_on_its_own(monkeypatch):
    """
    The SDK's retry is invisible to `call_groq` and multiplies with it.

    Asserted on the constructed client rather than on the constant, so a change
    to how the client is built cannot quietly reintroduce the amplification.
    """
    from app.services.groq_service import GroqService

    monkeypatch.setattr(settings, "groq_api_key", "gsk_test_key_not_real")
    service = GroqService()
    assert service.client.max_retries == 0


async def test_one_logical_call_is_bounded_to_two_http_attempts(agent, monkeypatch):
    attempts = {"n": 0}

    async def always_limited(**kwargs):
        attempts["n"] += 1
        raise _ApiError(429, {"retry-after": "1"})

    monkeypatch.setattr(agent.groq_service, "chat_completion", always_limited)

    with pytest.raises(Exception):
        await agent.call_groq(messages=[])

    assert attempts["n"] == base_agent._MAX_LLM_ATTEMPTS == 2


async def test_the_rate_limit_is_never_swallowed(agent, monkeypatch):
    """
    The caller must be able to tell a rate limit from an empty answer.

    `reflect_node` decides whether to re-run an entire specialist off this
    distinction, and a swallowed 429 would look to it like an ordinary poor
    answer worth retrying — the single most expensive possible response.
    """
    async def always_limited(**kwargs):
        raise _ApiError(429)

    monkeypatch.setattr(agent.groq_service, "chat_completion", always_limited)

    with pytest.raises(Exception) as raised:
        await agent.call_groq(messages=[])

    assert classify_llm_error(raised.value) is LLMErrorKind.RATE_LIMITED


async def test_the_timeout_applies_per_attempt(agent, monkeypatch):
    """
    Both attempts must get a full budget.

    The old arrangement wrapped the whole retry sequence in one `wait_for`, so
    a slow first attempt consumed the budget and the second could never run to
    completion — the work of both was discarded together.
    """
    monkeypatch.setattr(base_agent, "_LLM_CALL_TIMEOUT", 0.05)
    attempts = {"n": 0}

    async def slow_then_fast(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            # An Event that is never set, rather than `asyncio.sleep`: the
            # fixture above patches `sleep` on the shared asyncio module, so a
            # sleeping first attempt would return instantly and never time out.
            await asyncio.Event().wait()
        return {"content": "second attempt ran"}

    monkeypatch.setattr(agent.groq_service, "chat_completion", slow_then_fast)

    assert await agent.call_groq(messages=[]) == "second attempt ran"
    assert attempts["n"] == 2


async def test_call_options_reach_the_provider(agent, monkeypatch):
    """`response_format` had no way through; the planner needs it."""
    seen = {}

    async def capture(**kwargs):
        seen.update(kwargs)
        return {"content": "{}"}

    monkeypatch.setattr(agent.groq_service, "chat_completion", capture)

    await agent.call_groq(
        messages=[], response_format={"type": "json_object"}
    )

    assert seen["response_format"] == {"type": "json_object"}


# ═══════════════════════════════════════════════════════════════════════════
# D · The limiter
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_limiter_bounds_concurrency():
    limiter = GroqLimiter(max_concurrency=2, tokens_per_minute=10_000_000)
    live = {"now": 0, "peak": 0}

    async def one_call():
        async with limiter.reserve(messages=[{"role": "user", "content": "x"}]):
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(0.02)
            live["now"] -= 1

    await asyncio.gather(*(one_call() for _ in range(10)))

    assert live["peak"] <= 2, f"limiter admitted {live['peak']} concurrent requests"


async def test_the_token_budget_delays_rather_than_rejecting():
    """
    Back-pressure means queueing, not failing.

    A limiter that rejected would just move the 429 in-process; the point is
    that the request still happens, later.
    """
    # 6000 tokens/min refills at 100/second. The bucket is emptied directly
    # rather than by issuing hundreds of requests to drain it, so the wait
    # under test is short and exact: a ~50-token request needs 0.5s of refill.
    limiter = GroqLimiter(max_concurrency=8, tokens_per_minute=6000)
    message = [{"role": "user", "content": "x" * 200}]  # ~50 tokens

    # Prime the loop-bound primitives, then empty the bucket.
    async with limiter.reserve(message):
        pass
    limiter._available_tokens = 0.0
    limiter._last_refill = time.monotonic()

    started = time.monotonic()
    async with limiter.reserve(message):
        pass
    elapsed = time.monotonic() - started

    assert 0.2 < elapsed < 5.0, f"expected a short refill wait, waited {elapsed:.2f}s"
    assert limiter.snapshot()["delayed"] >= 1


async def test_an_oversized_request_is_clamped_rather_than_deadlocked():
    """
    A prompt larger than the whole bucket must not wait for capacity that
    cannot exist.

    Unclamped, this request would want 100k tokens from a bucket that holds
    60k — a condition no amount of refilling satisfies, so it would block until
    the 20s ceiling let it through. Clamped, it is funded immediately.
    """
    limiter = GroqLimiter(max_concurrency=4, tokens_per_minute=60_000)
    huge = [{"role": "user", "content": "x" * 400_000}]  # ~100k tokens

    started = time.monotonic()
    async with limiter.reserve(huge):
        pass

    assert time.monotonic() - started < 1.0


async def test_the_limiter_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(settings, "groq_limiter_enabled", False)
    limiter = GroqLimiter(max_concurrency=1, tokens_per_minute=1)

    async def one_call():
        async with limiter.reserve([{"role": "user", "content": "x" * 10_000}]):
            return True

    assert all(await asyncio.gather(*(one_call() for _ in range(5))))


def test_the_token_estimate_counts_the_prompt_and_the_reply_allowance():
    tokens = estimate_tokens([{"role": "user", "content": "x" * 400}], max_tokens=500)
    assert tokens == 100 + 500


def test_the_token_estimate_matches_the_providers_own_accounting():
    """
    Pinned against a real 429 rather than against a guess.

    A live probe of the planner call produced, from Groq:

        "Limit 8000, Used 5486, Requested 3523"

    and this estimator returned 3,526 for the same request. The agreement
    matters because the budget is enforced *before* the call: an estimator that
    ran low would let the account be oversubscribed and reintroduce the 429s
    the limiter exists to avoid.

    Groq counts `max_tokens` in full in "Requested", which is why this
    estimator does too.
    """
    planner_sized_prompt = [{"role": "system", "content": "x" * 8_087},
                            {"role": "user", "content": "what are my skills"}]
    estimated = estimate_tokens(planner_sized_prompt, max_tokens=1500)

    assert 3_400 <= estimated <= 3_700, estimated
    # Never under-estimate: headroom absorbs error, a shortfall does not.
    assert estimated >= 3_523


def test_the_configured_budget_does_not_exceed_the_known_account_limit():
    """
    A budget above the real TPM is inert, not conservative.

    At 24,000 against an 8,000 account, every request passed the gate and was
    then rejected by Groq — the limiter was doing nothing at all, which is
    exactly what the live probe showed.
    """
    assert settings.groq_tokens_per_minute <= 8_000, (
        "GROQ_TOKENS_PER_MINUTE is above the measured free-tier limit; the "
        "limiter will not engage before the provider rejects."
    )


def test_the_limiter_survives_a_new_event_loop():
    """
    The instance is a module singleton and the suite runs a loop per test.

    asyncio primitives bind to a loop on first use, so without the rebuild in
    `_primitives` the second loop to touch the shared limiter would raise
    "attached to a different loop" — in production that is the first request
    after any loop restart.
    """
    limiter = GroqLimiter(max_concurrency=1, tokens_per_minute=10_000_000)

    async def use_it():
        async with limiter.reserve([{"role": "user", "content": "x"}]):
            return True

    assert asyncio.run(use_it())
    assert asyncio.run(use_it())


# ═══════════════════════════════════════════════════════════════════════════
# Measurement
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_turn_metrics_separate_logical_calls_from_http_attempts(
    agent, monkeypatch, _no_real_sleeping
):
    """
    The two numbers the audit had to derive by hand are now measured.

    One logical call that took two HTTP attempts is the exact shape the old
    stack hid, so the counters have to be able to express it.
    """
    async def limited_then_ok(**kwargs):
        if not hasattr(limited_then_ok, "called"):
            limited_then_ok.called = True
            raise _ApiError(429, {"retry-after": "1"})
        return {"content": "ok"}

    monkeypatch.setattr(agent.groq_service, "chat_completion", limited_then_ok)

    with call_metrics.turn("req-1") as metrics:
        await agent.call_groq(messages=[{"role": "user", "content": "hello"}])

    assert metrics.llm_logical_calls == 1
    assert metrics.llm_rate_limited == 1
    assert metrics.llm_retries == 1


def test_metrics_are_inert_outside_a_turn():
    """The background worker and scripts call the same code paths."""
    assert call_metrics.current() is None
    call_metrics.record_llm_retry()
    call_metrics.record_qdrant_op()
