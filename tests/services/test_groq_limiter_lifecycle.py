"""
The limiter's slot must come back. Every time.

A semaphore in front of every Groq call is a single point of failure of a
particular kind: it does not fail loudly. A leaked slot does not raise, it
narrows the gate, and the symptom is the whole application getting slower and
eventually hanging with no error anywhere. The four slots are consumed one
barge-in at a time.

So these tests are about the release path rather than the acquire path, and
specifically about the three ways a caller leaves without finishing:
cancellation, an exception, and abandoning a stream part-way — which is what a
voice barge-in does on every interruption.
"""
from __future__ import annotations

import asyncio
import gc

import pytest

from app.config import settings
from app.services.groq_limiter import GroqLimiter


def _free_slots(limiter: GroqLimiter) -> int:
    """How many slots the semaphore currently has available."""
    semaphore, _ = limiter._semaphore, limiter._token_lock
    assert semaphore is not None, "primitives not built yet"
    return semaphore._value


@pytest.fixture
def limiter():
    # A budget large enough that the token bucket is never the thing under
    # test here — these tests are about the semaphore.
    return GroqLimiter(max_concurrency=2, tokens_per_minute=10_000_000)


MSG = [{"role": "user", "content": "hello"}]


# ── Part 3 · cancellation must not consume a slot ────────────────────────

async def test_a_cancelled_holder_releases_its_slot(limiter):
    """A barge-in cancels the turn's task while the request is open."""
    entered = asyncio.Event()

    async def holder():
        async with limiter.reserve(MSG):
            entered.set()
            await asyncio.Event().wait()   # never completes

    task = asyncio.create_task(holder())
    await entered.wait()
    assert _free_slots(limiter) == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _free_slots(limiter) == 2, "cancellation leaked a slot"


async def test_cancellation_while_queued_leaves_the_gate_intact(limiter):
    """
    The waiter never got a slot, so it must not release one it does not hold.

    Over-releasing is the mirror-image bug and is worse: it silently raises the
    concurrency limit rather than lowering it.
    """
    holding = asyncio.Event()
    release = asyncio.Event()

    async def holder():
        async with limiter.reserve(MSG):
            holding.set()
            await release.wait()

    holders = [asyncio.create_task(holder()) for _ in range(2)]
    await holding.wait()
    await asyncio.sleep(0.01)

    async def queued():
        async with limiter.reserve(MSG):
            pass

    waiter = asyncio.create_task(queued())
    await asyncio.sleep(0.02)
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)

    release.set()
    await asyncio.gather(*holders)

    assert _free_slots(limiter) == 2


async def test_an_exception_releases_the_slot(limiter):
    with pytest.raises(RuntimeError):
        async with limiter.reserve(MSG):
            raise RuntimeError("provider exploded")

    assert _free_slots(limiter) == 2


async def test_repeated_cancellation_does_not_narrow_the_gate(limiter):
    """
    The failure mode this guards is cumulative and silent.

    One leak per interrupted voice turn is invisible until the fourth, at which
    point every Groq call in the process blocks forever with no error raised
    anywhere.
    """
    for _ in range(20):
        started = asyncio.Event()

        async def holder():
            async with limiter.reserve(MSG):
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(holder())
        await started.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert _free_slots(limiter) == 2

    # And the gate still admits work.
    async with limiter.reserve(MSG):
        pass


async def test_the_limiter_does_not_deadlock_under_load(limiter):
    """Fifty callers, two slots, a mix of failures — all must drain."""
    async def worker(i: int):
        try:
            async with limiter.reserve(MSG):
                await asyncio.sleep(0.001)
                if i % 5 == 0:
                    raise ValueError("simulated failure")
        except ValueError:
            pass

    await asyncio.wait_for(
        asyncio.gather(*(worker(i) for i in range(50))), timeout=10.0
    )

    assert _free_slots(limiter) == 2


# ── Part 3 · streaming releases its slot ─────────────────────────────────

async def test_a_fully_consumed_stream_releases_its_slot(limiter):
    async def streamer():
        async with limiter.reserve(MSG):
            for token in "abc":
                yield token

    assert [t async for t in streamer()] == ["a", "b", "c"]
    assert _free_slots(limiter) == 2


async def test_an_abandoned_stream_releases_its_slot(limiter):
    """
    The barge-in case, and the one most likely to leak.

    `streaming_workflow` breaks out of `async for token in
    stream_chat_completion(...)` when a reply starts fabricating a completion.
    Breaking abandons the async generator while it is suspended at a yield —
    inside the `async with` that holds the slot. Nothing runs the `finally`
    until the generator is finalized.
    """
    async def streamer():
        async with limiter.reserve(MSG):
            for token in "abcdefgh":
                yield token

    agen = streamer()
    async for token in agen:
        if token == "c":
            break

    # Deterministic finalization, which is what production must also do.
    await agen.aclose()
    assert _free_slots(limiter) == 2


async def test_an_abandoned_stream_without_aclose_is_a_leak_until_finalized(limiter):
    """
    Documents *why* the explicit close above matters.

    Without it the slot comes back only when the generator is finalized, which
    is a garbage-collection event rather than a control-flow one. This test
    asserts the slot is eventually returned; it deliberately does not assert
    that it is returned promptly, because it is not.
    """
    async def streamer():
        async with limiter.reserve(MSG):
            for token in "abcdefgh":
                yield token

    async def consume():
        async for token in streamer():
            if token == "c":
                break

    await consume()
    gc.collect()
    await asyncio.sleep(0.05)   # let the finalizer's scheduled aclose run

    assert _free_slots(limiter) == 2


async def test_a_cancelled_stream_consumer_releases_the_slot(limiter):
    """A voice turn cancelled mid-stream by the watchdog or a barge-in."""
    started = asyncio.Event()

    async def streamer():
        async with limiter.reserve(MSG):
            started.set()
            while True:
                yield "token"
                await asyncio.sleep(0.01)

    async def consume():
        agen = streamer()
        try:
            async for _ in agen:
                pass
        finally:
            await agen.aclose()

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert _free_slots(limiter) == 2


# ── Part 4 · the token bucket ────────────────────────────────────────────

async def test_cancellation_while_waiting_for_tokens_is_clean():
    """
    A caller cancelled during the refill wait holds nothing and must leave
    nothing behind — including the token lock, which every other caller needs.
    """
    limiter = GroqLimiter(max_concurrency=4, tokens_per_minute=600)
    msg = [{"role": "user", "content": "x" * 800}]  # ~200 tokens

    async with limiter.reserve(msg):
        pass
    limiter._available_tokens = 0.0

    waiter = asyncio.create_task(limiter.reserve(msg).__aenter__())
    await asyncio.sleep(0.05)
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)

    assert not limiter._token_lock.locked(), "the token lock was left held"

    # A later caller still works once the bucket has refilled.
    limiter._available_tokens = float(limiter.tokens_per_minute)
    await asyncio.wait_for(limiter.reserve(msg).__aenter__(), timeout=2.0)


async def test_a_request_at_the_budget_boundary_is_admitted():
    """Exactly half the bucket is the clamp boundary; it must not wait forever."""
    limiter = GroqLimiter(max_concurrency=4, tokens_per_minute=1000)
    # 500 tokens = exactly _MAX_SHARE_OF_BUDGET of a 1000-token bucket.
    msg = [{"role": "user", "content": "x" * 2000}]

    await asyncio.wait_for(limiter.reserve(msg).__aenter__(), timeout=3.0)


async def test_a_request_above_the_budget_still_completes():
    """Clamped, not rejected, and not blocked on impossible capacity."""
    limiter = GroqLimiter(max_concurrency=4, tokens_per_minute=1000)
    msg = [{"role": "user", "content": "x" * 100_000}]  # ~25k tokens

    await asyncio.wait_for(limiter.reserve(msg).__aenter__(), timeout=3.0)


async def test_the_bucket_refills_over_time():
    limiter = GroqLimiter(max_concurrency=4, tokens_per_minute=6000)  # 100/s
    limiter._primitives()
    limiter._available_tokens = 0.0
    limiter._last_refill = asyncio.get_running_loop().time() - 0  # anchor

    import time as _t
    limiter._last_refill = _t.monotonic() - 1.0   # one second has passed

    async with limiter._token_lock:
        limiter._refill_locked()

    assert 90 <= limiter._available_tokens <= 110, limiter._available_tokens


async def test_the_bucket_never_exceeds_its_capacity():
    """A long idle period must not bank unlimited burst capacity."""
    import time as _t

    limiter = GroqLimiter(max_concurrency=4, tokens_per_minute=6000)
    limiter._primitives()
    limiter._available_tokens = 0.0
    limiter._last_refill = _t.monotonic() - 3600.0   # idle for an hour

    async with limiter._token_lock:
        limiter._refill_locked()

    assert limiter._available_tokens == float(limiter.tokens_per_minute)


async def test_back_pressure_stays_bounded(monkeypatch):
    """
    The admission ceiling is what stops back-pressure becoming a hang.

    A request that cannot be funded within the ceiling is let through so the
    provider's own limiter can answer — a visible 429 is recoverable, a silent
    multi-minute stall is not.
    """
    from app.services import groq_limiter as gl

    monkeypatch.setattr(gl, "_MAX_WAIT_SECONDS", 0.3)
    limiter = GroqLimiter(max_concurrency=4, tokens_per_minute=60)
    msg = [{"role": "user", "content": "x" * 400}]

    async with limiter.reserve(msg):
        pass
    limiter._available_tokens = 0.0

    import time as _t
    started = _t.monotonic()
    async with limiter.reserve(msg):
        pass
    elapsed = _t.monotonic() - started

    assert elapsed < 2.0, f"admission ceiling did not bound the wait ({elapsed:.1f}s)"
    assert limiter.snapshot()["delayed"] >= 1


async def test_the_streaming_workflow_closes_its_token_stream(monkeypatch):
    """
    The production consumer, not a stand-in.

    `run_streaming_workflow` breaks out of the token loop when a tool-free
    reply claims a completion it could not have performed. That break abandons
    the generator holding the limiter slot; this asserts the generator is
    closed on the way out rather than left to garbage collection.
    """
    from app.agents import streaming_workflow as sw

    closed = {"n": 0}

    async def instrumented_stream(**kwargs):
        try:
            # Long enough that the consumer must break rather than exhaust it.
            for _ in range(500):
                yield "I have sent the email. "
        finally:
            closed["n"] += 1

    monkeypatch.setattr(sw.groq_service, "stream_chat_completion", instrumented_stream)

    async def _init(state):
        state["memory_prompt"] = "ctx"
        state["selected_agent"] = "profile"
        state["query_category"] = "SMALL_TALK"
        return state

    async def _escalated(**kwargs):
        if False:
            yield {}

    monkeypatch.setattr(sw, "parallel_init_node", _init)
    monkeypatch.setattr(sw, "_escalate_to_tools", _escalated)

    events = []
    async for event in sw.run_streaming_workflow(
        user_input="say something nice",
        user_id="vansh",
        session_id="stream-close-1",
    ):
        events.append(event)

    assert closed["n"] >= 1, "the token stream was never closed"


async def test_disabled_limiter_holds_no_slot(limiter, monkeypatch):
    monkeypatch.setattr(settings, "groq_limiter_enabled", False)

    async with limiter.reserve(MSG):
        pass

    # Primitives were never built, so there is nothing to leak.
    assert limiter._semaphore is None or _free_slots(limiter) == 2
