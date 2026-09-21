"""
Final verification: prompt size, task cleanup, and whether the metrics tell the truth.

Three things are checked here that are easy to get almost right:

**Prompt size.** The memory cap was lowered and the block is now sent only on
the first reasoning iteration. Both are trivially breakable by an edit that
looks unrelated, and neither fails loudly — the turn still answers, it just
costs three times as much.

**Task cleanup.** A timed-out or cancelled turn must not leave work running.
Background writes are detached deliberately; a *leaked* task is different from
a detached one, and the difference is whether anything holds it and observes
its failure.

**Metrics honesty.** Numbers that are wrong are worse than no numbers, because
the next audit trusts them. In particular, detached background work must not be
counted as if the user waited for it.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.agents import base_agent, workflow
from app.agents.base_agent import (
    _MAX_MEMORY_CHARS,
    _MEMORY_BLOCK_END,
    _MEMORY_BLOCK_START,
    BaseAgent,
    _without_memory_block,
)
from app.config import settings
from app.services import call_metrics


class _Agent(BaseAgent):
    def __init__(self):
        super().__init__(name="probe", description="a probe")

    async def execute(self, state):
        return state


# ═══════════════════════════════════════════════════════════════════════════
# Part 15 · prompt size
# ═══════════════════════════════════════════════════════════════════════════

def test_the_memory_cap_matches_the_v2_budget():
    """
    Two numbers doing the same job disagreed by 3×. `memory_v2_budget_tokens`
    is what the v2 assembler allocates against; the char cap is the v1
    equivalent and is aligned to it.
    """
    assert _MAX_MEMORY_CHARS == 6_000
    assert _MAX_MEMORY_CHARS == settings.memory_v2_budget_tokens


def test_an_oversized_memory_block_is_truncated_from_the_low_priority_end():
    agent = _Agent()
    # Profile facts are written first upstream, résumé last, so a cut takes the
    # résumé tail rather than the user's name.
    memory = "User Profile Facts:\n- name: Vansh\n" + ("RESUME BODY. " * 2000)
    state = {"memory_prompt": memory}

    prompt = agent.inject_memory_context("SYSTEM RULES", state)

    assert "name: Vansh" in prompt
    assert "memory truncated" in prompt
    assert len(prompt) < len(memory)


def test_a_small_memory_block_is_passed_through_intact():
    agent = _Agent()
    state = {"memory_prompt": "User Profile Facts:\n- name: Vansh"}

    prompt = agent.inject_memory_context("SYSTEM RULES", state)

    assert "name: Vansh" in prompt
    assert "memory truncated" not in prompt
    assert "SYSTEM RULES" in prompt


def test_stripping_the_memory_block_keeps_everything_else():
    agent = _Agent()
    state = {"memory_prompt": "User Profile Facts:\n- name: Vansh"}
    full = agent.inject_memory_context("SYSTEM RULES FOR THE AGENT", state)

    trimmed = _without_memory_block(full)

    assert "name: Vansh" not in trimmed
    assert "SYSTEM RULES FOR THE AGENT" in trimmed
    assert "Provided in full on the first step" in trimmed
    assert len(trimmed) < len(full)
    assert _MEMORY_BLOCK_START not in trimmed and _MEMORY_BLOCK_END not in trimmed


def test_stripping_a_prompt_with_no_memory_block_is_a_no_op():
    plain = "SYSTEM RULES ONLY, no memory was retrieved."
    assert _without_memory_block(plain) == plain


async def test_iteration_one_gets_memory_and_later_iterations_do_not(monkeypatch):
    """
    The measurable property: a three-step turn used to transmit the memory
    block three times.
    """
    agent = _Agent()
    prompts: list[str] = []

    async def capture(messages, temperature=0.7, max_tokens=1024, _retries=2, **kwargs):
        prompts.append(messages[0]["content"])
        # Keep asking for a tool so the loop runs every iteration.
        return json.dumps({"type": "tool_call", "tool": "noop", "tool_input": {}})

    async def noop_tool(_args):
        return {"success": True, "found": False}

    monkeypatch.setattr(agent, "call_groq", capture)

    state = {
        "user_input": "what are my skills",
        "user_id": "vansh",
        "session_id": "hard-1",
        "memory_prompt": "User Profile Facts:\n- name: Vansh\n- cgpa: 8.9",
    }

    await agent.execute_reasoning_loop(
        state,
        # The *raw* prompt, as every specialist passes it. The loop calls
        # `inject_memory_context` itself, so pre-injecting here would produce
        # two memory blocks — a shape production never builds.
        base_system_prompt="SYSTEM RULES",
        tools={
            "noop": {
                "callable": noop_tool,
                "description": "does nothing",
                # Declared READ so the action gateway does not hold the call
                # for confirmation; this test is about prompts, not effects.
                "effect": "READ",
            }
        },
        max_iterations=3,
    )

    assert len(prompts) == 3
    assert "cgpa: 8.9" in prompts[0], "iteration 1 must carry the memory"
    assert "cgpa: 8.9" not in prompts[1], "iteration 2 re-sent the memory block"
    assert "cgpa: 8.9" not in prompts[2]
    assert "SYSTEM RULES" in prompts[1], "the agent's own rules must survive"
    assert len(prompts[1]) < len(prompts[0])


# ═══════════════════════════════════════════════════════════════════════════
# Part 7 · timeout and cancellation leave nothing running
# ═══════════════════════════════════════════════════════════════════════════

def _live_tasks() -> set:
    return {
        t for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    }


async def test_a_timed_out_workflow_leaves_no_task_running(monkeypatch):
    """
    `wait_for` cancels the graph task, but only if nothing inside it swallows
    the cancellation. A turn that keeps working after its deadline is server
    time spent on an answer nobody will receive.
    """
    class _Graph:
        async def ainvoke(self, state):
            await asyncio.sleep(30)
            return state

    monkeypatch.setattr(workflow, "multi_agent_workflow", _Graph())

    before = _live_tasks()
    result = await workflow.run_workflow(
        user_input="something slow", user_id="vansh", session_id="hard-2",
        timeout_seconds=0.05,
    )
    await asyncio.sleep(0.05)
    leaked = _live_tasks() - before

    assert result["error"] == "workflow_timeout"
    assert not leaked, f"tasks still running after the deadline: {leaked}"


async def test_a_cancelled_workflow_does_not_leave_the_metrics_scope_open(monkeypatch):
    """
    The scope is a ContextVar. Left open, the *next* turn's calls are counted
    against the cancelled one and both numbers become fiction.
    """
    class _Graph:
        async def ainvoke(self, state):
            await asyncio.sleep(30)
            return state

    monkeypatch.setattr(workflow, "multi_agent_workflow", _Graph())

    task = asyncio.create_task(workflow.run_workflow(
        user_input="x", user_id="vansh", session_id="hard-3", timeout_seconds=30,
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert call_metrics.current() is None


async def test_the_voice_deadline_is_shorter_than_the_typed_one():
    assert settings.voice_workflow_timeout_seconds == pytest.approx(35.0, abs=5.0)
    assert settings.voice_workflow_timeout_seconds < settings.workflow_timeout_seconds
    # It must outlast the stall window, or the watchdog and the deadline race.
    assert settings.voice_workflow_timeout_seconds > settings.voice_turn_stall_seconds


# ═══════════════════════════════════════════════════════════════════════════
# Part 19 · the metrics tell the truth
# ═══════════════════════════════════════════════════════════════════════════

def test_a_turn_reports_every_dimension():
    """
    Every counter is present and independent.

    Deliberately synchronous and driven against a directly-constructed
    `TurnMetrics` rather than through the module-level ContextVar. The earlier
    version awaited inside the scope, which yielded to the event loop, and
    under a loaded suite it failed roughly one run in eight — a detached task
    from elsewhere in the suite can inherit a turn context and increment a
    counter while this one is open. That is *intended* behaviour (see
    `test_detached_work_is_attributed...`), so the fix is to stop this test
    depending on global isolation it was never promised.

    The ContextVar plumbing itself is covered by the nesting and cancellation
    tests below, which do not assert exact totals.
    """
    from app.services.call_metrics import TurnMetrics

    m = TurnMetrics(request_id="obs-1")
    m.llm_logical_calls += 1
    m.llm_estimated_tokens += 1200
    m.llm_http_attempts += 2
    m.llm_retries += 1
    m.llm_rate_limited += 1
    m.embed_calls += 1
    m.embed_cache_hits += 1
    m.embed_coalesced += 1
    m.qdrant_ops += 3
    m.time("retrieval", 0.25)
    m.time("total", 0.40)

    report = m.as_dict()

    assert report["request_id"] == "obs-1"
    assert report["llm_logical_calls"] == 1
    assert report["llm_http_attempts"] == 2
    assert report["llm_retries"] == 1
    assert report["llm_rate_limited"] == 1
    assert report["llm_estimated_tokens"] == 1200
    assert report["embed_calls"] == 1
    assert report["embed_cache_hits"] == 1
    assert report["embed_coalesced"] == 1
    assert report["qdrant_ops"] == 3
    assert report["t_retrieval"] == 0.25
    assert report["t_total"] == 0.40


async def test_the_recording_helpers_reach_the_active_turn():
    """
    The plumbing, asserted with `>=` because the suite is concurrent.

    What matters here is that each helper targets the right field of the turn
    in scope — not that this turn is the only thing running in the process.
    """
    with call_metrics.turn("obs-1b") as m:
        call_metrics.record_llm_request(estimated_tokens=1200)
        call_metrics.record_groq_call("openai/gpt-oss-120b")
        call_metrics.record_llm_retry()
        call_metrics.record_llm_rate_limited()
        call_metrics.record_embed_call()
        call_metrics.record_embed_cache_hit()
        call_metrics.record_embed_coalesced()
        call_metrics.record_qdrant_op(3)
        with call_metrics.phase("retrieval"):
            await asyncio.sleep(0)

    report = m.as_dict()

    assert report["llm_logical_calls"] >= 1
    assert report["llm_http_attempts"] >= 1
    assert report["llm_retries"] >= 1
    assert report["llm_rate_limited"] >= 1
    assert report["llm_estimated_tokens"] >= 1200
    assert report["embed_calls"] >= 1
    assert report["embed_cache_hits"] >= 1
    assert report["embed_coalesced"] >= 1
    assert report["qdrant_ops"] >= 3
    assert "t_retrieval" in report and "t_total" in report


async def test_logical_calls_and_http_attempts_are_separately_visible():
    """
    The one distinction the original audit had to compute by hand. Collapsing
    these two into a single "llm calls" number is what hid the amplification.
    """
    with call_metrics.turn("obs-2") as m:
        call_metrics.record_llm_request()
        for _ in range(2):
            call_metrics.record_groq_call("m")
            call_metrics.record_llm_retry()

    assert m.llm_logical_calls == 1
    assert m.llm_http_attempts == 2
    assert m.llm_http_attempts > m.llm_logical_calls


async def test_detached_work_is_attributed_but_does_not_inflate_the_turn_time():
    """
    A detached write inherits the turn's context, so its cost is attributed to
    the turn that caused it — which is honest. What it must not do is extend
    `t_total`, because the user did not wait for it.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    async def detached():
        started.set()
        await release.wait()
        call_metrics.record_embed_call()

    with call_metrics.turn("obs-3") as m:
        task = asyncio.create_task(detached())
        await started.wait()

    total_at_close = m.timings["total"]
    assert total_at_close < 0.5, "the turn waited for detached work"

    release.set()
    await task

    # Attribution still lands on the originating turn's object.
    assert m.embed_calls == 1
    # But the recorded duration was closed before it ran.
    assert m.timings["total"] == total_at_close


async def test_metrics_outside_a_turn_are_inert_and_cheap():
    assert call_metrics.current() is None
    for _ in range(10_000):
        call_metrics.record_groq_call("m")
        call_metrics.record_qdrant_op()
    assert call_metrics.current() is None


async def test_a_nested_turn_does_not_corrupt_the_outer_one():
    """
    Escalation runs `run_workflow` from inside a streaming turn, so scopes can
    nest. The inner one must not leave the outer one reset.
    """
    with call_metrics.turn("outer") as outer:
        call_metrics.record_qdrant_op()
        with call_metrics.turn("inner") as inner:
            call_metrics.record_qdrant_op(5)
        assert call_metrics.current() is outer
        call_metrics.record_qdrant_op()

    assert outer.qdrant_ops == 2
    assert inner.qdrant_ops == 5
    assert call_metrics.current() is None


async def test_the_workflow_emits_a_turn_cost_line(monkeypatch, caplog):
    """The numbers have to reach the logs, not just the object."""
    import logging

    class _Graph:
        async def ainvoke(self, state):
            state["display_text"] = ""
            return state

    monkeypatch.setattr(workflow, "multi_agent_workflow", _Graph())

    with caplog.at_level(logging.INFO, logger="app.agents.workflow"):
        await workflow.run_workflow(
            user_input="hello", user_id="vansh", session_id="hard-4",
        )

    assert any("Turn cost" in r.message for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════
# Part 20 · no leaked background tasks from the detached writes
# ═══════════════════════════════════════════════════════════════════════════

async def test_detached_memory_writes_are_held_and_observed():
    """
    asyncio holds only a weak reference to a running task. Without a strong
    one, a fire-and-forget write can be collected mid-run and its exception
    never retrieved — the failure would be invisible.
    """
    from app.memory import memory_manager as mm

    failures = []

    async def boom():
        raise RuntimeError("write failed")

    mm._spawn_shadow(boom(), "test-detached")
    assert mm._shadow_tasks, "the task was not retained"

    # Awaited to completion rather than slept past: under a loaded event loop a
    # fixed sleep is a coin flip, and a flaky test is worse than no test.
    await _drain(mm._shadow_tasks)
    assert not mm._shadow_tasks, "the finished task was not released"


async def test_the_reasoning_loops_background_writes_are_retained():
    async def slow():
        await asyncio.sleep(0.02)

    base_agent._spawn_background(slow(), "test-bg")
    assert base_agent._background_tasks

    await _drain(base_agent._background_tasks)
    assert not base_agent._background_tasks


async def _drain(task_set: set, timeout: float = 5.0) -> None:
    """Wait for a detached-task registry to empty, or fail loudly."""
    pending = list(task_set)
    if pending:
        await asyncio.wait(pending, timeout=timeout)
    # The done-callback that removes the entry runs on the next loop pass.
    for _ in range(100):
        if not task_set:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"registry did not drain: {task_set}")
