"""
Why a working spoken turn used to be cancelled at exactly twenty seconds.

`watch_for_stall` is a good design and is deliberately left as it is. It cancels
a turn that has stopped making *progress* rather than one that has simply been
running a while, because a wall-clock deadline cuts a long spoken answer
mid-sentence — which is precisely what it must not do.

But it can only see what `progress.touch()` reports. The streaming branch
touches on every token; the tool branch was a single opaque `await
run_workflow(...)` with one touch before it and one after. So a tool turn
reported liveness twice, and anything in between that outlived
`voice_turn_stall_seconds` was cancelled as stalled while the workflow was
still working through a 120-second budget of its own.

Two deadlines, never reconciled, and the shorter one had no way to tell work
from silence. The fix is to make the work visible, not to make the watchdog
more patient — raising the threshold would have hidden the bug and made
genuinely frozen turns take longer to notice.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from app.config import settings
from app.livekit_worker import TurnProgress, watch_for_stall


# ── The bug, reproduced against the real watchdog ────────────────────────

async def test_an_opaque_await_is_cancelled_once_the_stall_window_passes(monkeypatch):
    """
    The old tool branch, exactly: touch, long await, touch.

    This is what shipped, and it is what the production logs recorded as a turn
    dying at twenty seconds.
    """
    monkeypatch.setattr(settings, "voice_turn_stall_seconds", 0.2)
    monkeypatch.setattr(settings, "voice_turn_max_seconds", 600.0)

    progress = TurnProgress()
    completed = {"ok": False}

    async def turn_body():
        progress.touch()
        await asyncio.sleep(1.0)   # run_workflow, making real progress
        progress.touch()
        completed["ok"] = True

    turn = asyncio.create_task(turn_body())
    guard = asyncio.create_task(watch_for_stall("vansh", progress, turn))
    await asyncio.gather(turn, guard, return_exceptions=True)

    assert turn.cancelled(), "expected the historical false cancellation"
    assert not completed["ok"]


# ── G · The heartbeat ────────────────────────────────────────────────────

async def test_a_heartbeat_keeps_a_working_turn_alive(monkeypatch):
    """
    The same turn, with liveness asserted while the workflow runs.

    The stall threshold is unchanged — what changed is that the work is now
    visible to the thing watching for its absence.
    """
    monkeypatch.setattr(settings, "voice_turn_stall_seconds", 0.2)
    monkeypatch.setattr(settings, "voice_turn_max_seconds", 600.0)

    progress = TurnProgress()
    completed = {"ok": False}
    published: list[dict] = []

    async def turn_body():
        progress.touch()

        async def heartbeat():
            while True:
                await asyncio.sleep(0.05)
                progress.touch()
                published.append({"type": "working"})

        beat = asyncio.create_task(heartbeat())
        try:
            await asyncio.sleep(1.0)   # run_workflow, making real progress
        finally:
            beat.cancel()
        progress.touch()
        completed["ok"] = True

    turn = asyncio.create_task(turn_body())
    guard = asyncio.create_task(watch_for_stall("vansh", progress, turn))
    await asyncio.gather(turn, guard, return_exceptions=True)

    assert completed["ok"], "a turn making progress must not be cancelled"
    assert not turn.cancelled()
    assert published, "the browser should have been told work is in progress"


async def test_the_watchdog_still_catches_a_genuinely_frozen_turn(monkeypatch):
    """
    The heartbeat must not blind the watchdog.

    A heartbeat that outlived its turn — or one driven by a timer rather than
    by the turn's own task — would assert liveness for work that had stopped,
    which is worse than the bug it fixes.
    """
    monkeypatch.setattr(settings, "voice_turn_stall_seconds", 0.2)
    monkeypatch.setattr(settings, "voice_turn_max_seconds", 600.0)

    progress = TurnProgress()

    async def frozen_turn():
        progress.touch()
        await asyncio.Event().wait()   # never completes, never touches

    turn = asyncio.create_task(frozen_turn())
    guard = asyncio.create_task(watch_for_stall("vansh", progress, turn))
    await asyncio.gather(turn, guard, return_exceptions=True)

    assert turn.cancelled()
    assert progress.abort_reason == "stalled"


async def test_the_heartbeat_dies_with_its_turn():
    """
    Cancelled in a `finally`, so a barge-in or a watchdog cancellation takes it
    down too. A leaked heartbeat is a task that keeps touching a `TurnProgress`
    for a turn that no longer exists.
    """
    beats = {"n": 0}

    async def heartbeat():
        while True:
            await asyncio.sleep(0.02)
            beats["n"] += 1

    async def turn_body():
        beat = asyncio.create_task(heartbeat())
        try:
            await asyncio.sleep(5.0)
        finally:
            beat.cancel()
        return beat

    turn = asyncio.create_task(turn_body())
    await asyncio.sleep(0.1)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    seen = beats["n"]
    await asyncio.sleep(0.1)
    assert beats["n"] == seen, "the heartbeat outlived the turn it belonged to"


def test_the_worker_wires_a_heartbeat_around_the_tool_workflow():
    """
    Guards the call site itself.

    The behaviour above is provable in isolation; this is what stops the tool
    branch quietly reverting to a bare `await run_workflow(...)`.
    """
    import inspect

    from app import livekit_worker

    source = inspect.getsource(livekit_worker)
    assert "workflow_heartbeat" in source
    assert "_TOOL_TURN_HEARTBEAT_SECONDS" in source
    assert livekit_worker._TOOL_TURN_HEARTBEAT_SECONDS < settings.voice_turn_stall_seconds


# ── H · The two deadlines agree ──────────────────────────────────────────

def test_a_spoken_turn_has_its_own_shorter_deadline():
    """
    120 seconds is a typed turn's budget. A caller waiting on audio has stopped
    listening long before that, and the stall watchdog would have given up
    first anyway — so the workflow was allowed to keep working server-side for
    a turn nobody could still receive.
    """
    assert settings.voice_workflow_timeout_seconds < settings.workflow_timeout_seconds
    assert settings.voice_workflow_timeout_seconds >= settings.voice_turn_stall_seconds


async def test_run_workflow_honours_a_caller_supplied_deadline(monkeypatch):
    """The voice worker passes its own; the default must stay the typed one."""
    from app.agents import workflow

    seen = {}

    class _Graph:
        async def ainvoke(self, state):
            await asyncio.sleep(5.0)
            return state

    async def _fake_wait_for(coro, timeout):
        seen["timeout"] = timeout
        coro.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(workflow, "multi_agent_workflow", _Graph())
    monkeypatch.setattr(workflow.asyncio, "wait_for", _fake_wait_for)

    result = await workflow.run_workflow(
        user_input="what is my timetable",
        user_id="vansh",
        session_id="voice-1",
        timeout_seconds=settings.voice_workflow_timeout_seconds,
    )

    assert seen["timeout"] == settings.voice_workflow_timeout_seconds
    assert result["error"] == "workflow_timeout"


async def test_the_typed_path_keeps_the_longer_deadline(monkeypatch):
    from app.agents import workflow

    seen = {}

    class _Graph:
        async def ainvoke(self, state):
            return state

    async def _fake_wait_for(coro, timeout):
        seen["timeout"] = timeout
        return await coro

    monkeypatch.setattr(workflow, "multi_agent_workflow", _Graph())
    monkeypatch.setattr(workflow.asyncio, "wait_for", _fake_wait_for)
    monkeypatch.setattr(workflow.memory_manager, "on_agent_response", _noop)
    monkeypatch.setattr(workflow.memory_manager, "get_session_turn_count", _zero)

    await workflow.run_workflow(
        user_input="what is my timetable", user_id="vansh", session_id="text-1",
    )

    assert seen["timeout"] == settings.workflow_timeout_seconds


async def _noop(**kwargs):
    return None


async def _zero(*args, **kwargs):
    return 0
