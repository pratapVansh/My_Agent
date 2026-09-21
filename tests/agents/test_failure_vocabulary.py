"""
Five outcomes, five meanings, and one of them must never cost a retry.

The reflect loop is the most expensive decision in the system: a retry re-runs
an entire specialist, including a fresh reasoning loop. So what it may retry
has to be decided from *why* the turn failed, and that requires the reasons to
stay distinguishable all the way up:

    SUCCESS      the model answered            → done
    NO_DATA      the store is empty            → done; an empty store is an answer
    TOOL_ERROR   the tool broke                → retry; a different approach may work
    TIMEOUT      nothing came back             → done; the provider is not answering
    RATE_LIMIT   the provider refused          → done; retrying makes it worse

The two at the bottom are the ones this pass added, and they are the two where
retrying is not merely wasteful but actively harmful — a retry into a closed
window earns another rejection, and each rejection still counts against the
account.

Collapsing any pair here would be a silent regression: the turn would still
answer, just at several times the cost, under exactly the conditions where cost
is what already went wrong.
"""
from __future__ import annotations

import asyncio
import json
import logging

import pytest

from app.agents import base_agent, workflow
from app.agents.base_agent import BaseAgent
from app.agents.planner_agent import planner_agent
from app.agents.workflow import reflect_node
from app.memory.sources import QueryCategory


def _state(**extra) -> dict:
    state = {
        "task_result": {"status": "failed", "agent": "job", "result": {"content": ""}},
        "iteration_count": 0,
        "execution_path": [],
        "execution_plan": [],
        "current_step_index": 0,
        "query_category": QueryCategory.JOB_SEARCH.value,
    }
    state.update(extra)
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Part 14 · the five outcomes stay distinct
# ═══════════════════════════════════════════════════════════════════════════

async def test_success_is_done():
    state = await reflect_node(_state(
        task_result={"status": "success", "agent": "job", "result": {"content": "Found 3 roles."}},
        answerability="OK",
    ))
    assert state["reflect_outcome"] == "done"


async def test_no_data_is_an_answer_not_a_failure():
    """Retrying an empty store just asks the same empty store again."""
    state = await reflect_node(_state(
        task_result={"status": "success", "agent": "profile", "result": {"content": "Nothing stored."}},
        answerability="NO_DATA",
    ))
    assert state["reflect_outcome"] == "done"
    assert not state.get("reflect_failure_context")


async def test_tool_error_is_retried():
    """The failure the loop exists for: a different approach may recover it."""
    state = await reflect_node(_state(
        task_result={"status": "success", "agent": "job", "result": {"content": "x"}},
        answerability="TOOL_ERROR",
    ))
    assert state["reflect_outcome"] == "retry"
    assert "different approach" in state["reflect_failure_context"]


async def test_a_skipped_required_tool_is_retried():
    """The most recoverable failure there is — the tool exists, nothing looked."""
    state = await reflect_node(_state(
        task_result={"status": "success", "agent": "profile", "result": {"content": "x"}},
        grounding="skipped",
    ))
    assert state["reflect_outcome"] == "retry"


async def test_rate_limit_is_not_retried():
    state = await reflect_node(_state(rate_limited=True))
    assert state["reflect_outcome"] == "done"
    assert "rate_limited" in " ".join(state["execution_path"])


async def test_an_unreachable_provider_is_not_retried():
    """
    Every LLM attempt timed out and none returned.

    This was the gap: the code's own comment claimed timeouts were excluded
    from retry and the gate only tested for rate limits, so a provider that had
    stopped answering was met with a full second specialist run — more calls,
    into the same conditions.
    """
    state = await reflect_node(_state(
        task_result={"status": "success", "agent": "job", "result": {"content": "x"}},
        grounding="skipped",          # would normally trigger a retry
        llm_unreachable=True,
    ))
    assert state["reflect_outcome"] == "done"


async def test_an_unreachable_provider_on_the_envelope_also_stops_the_retry():
    state = await reflect_node(_state(
        task_result={
            "status": "failed", "agent": "job",
            "result": {"content": ""}, "llm_unreachable": True,
        },
    ))
    assert state["reflect_outcome"] == "done"


async def test_a_recovered_timeout_is_still_retryable():
    """
    The gate must be narrow. One slow call that a later iteration recovered
    from is an ordinary turn, and blocking its retry would trade one
    amplification bug for a correctness one.
    """
    state = await reflect_node(_state(
        task_result={"status": "success", "agent": "job", "result": {"content": "x"}},
        grounding="skipped",
        llm_unreachable=False,
    ))
    assert state["reflect_outcome"] == "retry"


async def test_the_retry_budget_is_two_passes():
    assert workflow.MAX_ITERATIONS == 2
    assert (await reflect_node(_state(iteration_count=1)))["reflect_outcome"] == "done"
    assert (await reflect_node(_state(iteration_count=0)))["reflect_outcome"] == "retry"


# ── The loop sets the flags the gate reads ───────────────────────────────

class _Agent(BaseAgent):
    def __init__(self):
        super().__init__(name="probe", description="")

    async def execute(self, state):
        return state


async def test_the_loop_marks_a_turn_whose_every_call_timed_out(monkeypatch):
    monkeypatch.setattr(base_agent, "_LLM_CALL_TIMEOUT", 0.02)

    async def never_returns(**kwargs):
        await asyncio.Event().wait()

    agent = _Agent()
    monkeypatch.setattr(agent.groq_service, "chat_completion", never_returns)

    state = {"user_input": "find jobs", "user_id": "vansh", "session_id": "fv-1"}
    result = await agent.execute_reasoning_loop(
        state, base_system_prompt="You are a probe.", tools={}, max_iterations=2,
    )

    assert result["llm_unreachable"] is True
    assert state["llm_unreachable"] is True
    assert result["rate_limited"] is False


async def test_a_loop_that_got_an_answer_is_not_marked_unreachable(monkeypatch):
    monkeypatch.setattr(base_agent, "_LLM_CALL_TIMEOUT", 0.05)
    calls = {"n": 0}

    async def slow_then_answers(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.Event().wait()
        return {"content": json.dumps({"type": "final", "content": "Here you go."})}

    agent = _Agent()
    monkeypatch.setattr(agent.groq_service, "chat_completion", slow_then_answers)

    result = await agent.execute_reasoning_loop(
        {"user_input": "find jobs", "user_id": "vansh", "session_id": "fv-2"},
        base_system_prompt="You are a probe.", tools={}, max_iterations=3,
    )

    assert result["llm_unreachable"] is False
    assert result["final_answer"] == "Here you go."


async def test_a_rate_limited_loop_stops_early_and_is_marked(monkeypatch):
    class _Limited(Exception):
        status_code = 429

    calls = {"n": 0}

    async def limited(**kwargs):
        calls["n"] += 1
        raise _Limited()

    agent = _Agent()
    monkeypatch.setattr(agent.groq_service, "chat_completion", limited)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(base_agent.asyncio, "sleep", _no_sleep)

    result = await agent.execute_reasoning_loop(
        {"user_input": "find jobs", "user_id": "vansh", "session_id": "fv-3"},
        base_system_prompt="You are a probe.", tools={}, max_iterations=3,
    )

    assert result["rate_limited"] is True
    # Two attempts inside call_groq on step 1, then the loop stops — it does
    # not spend the remaining two iterations on a closed window.
    assert calls["n"] == base_agent._MAX_LLM_ATTEMPTS == 2


async def test_a_genuine_error_still_propagates(monkeypatch):
    """
    Not swallowed. The agent's `execute` catches this and stamps a *failed*
    envelope; converting it to a success carrying the fallback sentence would
    hide a provider outage behind a polite non-answer.
    """
    async def broken(**kwargs):
        raise ValueError("malformed request")

    agent = _Agent()
    monkeypatch.setattr(agent.groq_service, "chat_completion", broken)

    with pytest.raises(ValueError):
        await agent.execute_reasoning_loop(
            {"user_input": "x", "user_id": "vansh", "session_id": "fv-4"},
            base_system_prompt="p", tools={}, max_iterations=2,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Part 8 · the planner under every response shape
# ═══════════════════════════════════════════════════════════════════════════

def _planner_state(text: str = "find me a job") -> dict:
    return {"user_input": text, "user_id": "vansh", "session_id": "fv-p", "execution_path": []}


def _install(monkeypatch, response, counter=None):
    async def reply(messages, temperature=0.7, max_tokens=1024, _retries=2, **kwargs):
        if counter is not None:
            counter["n"] += 1
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(planner_agent, "call_groq", reply)


@pytest.mark.parametrize(
    "label, response",
    [
        ("empty string", ""),
        ("whitespace only", "   \n  "),
        ("truncated json", '{"intent": "find jobs", "agent": "job", "confid'),
        ("not json at all", "Sure! I think you want a job search."),
        ("json array", '["job"]'),
        ("json null", "null"),
        ("fenced but truncated", '```json\n{"agent": "job"'),
    ],
)
async def test_a_malformed_planner_response_is_survivable(monkeypatch, label, response):
    """
    Deterministic fallback, one call, no crash.

    The fallback routes to profile, which may be the wrong specialist — that is
    a known cost and is why it logs at ERROR. What it must never do is raise, or
    spend a second model call trying to repair the first.
    """
    calls = {"n": 0}
    _install(monkeypatch, response, calls)

    result = await planner_agent.execute(_planner_state())

    assert calls["n"] == 1, "the planner must not re-ask the model to fix its output"
    assert result["selected_agent"] == "profile"
    assert result["needs_clarification"] is False
    assert result["execution_plan"], "a plan must always exist"


async def test_unexpected_fields_are_ignored_not_fatal(monkeypatch):
    _install(monkeypatch, json.dumps({
        "intent": "find jobs",
        "agent": "job",
        "confidence": 0.9,
        "needs_clarification": False,
        "clarification_question": "",
        "execution_plan": [{"step": 1, "agent": "job", "goal": "search"}],
        "surprise_field": {"nested": [1, 2, 3]},
        "another": None,
    }))

    result = await planner_agent.execute(_planner_state())
    assert result["selected_agent"] == "job"


async def test_an_unknown_agent_name_is_coerced(monkeypatch):
    _install(monkeypatch, json.dumps({
        "intent": "x", "agent": "quantum_specialist", "confidence": 0.9,
        "needs_clarification": False, "clarification_question": "",
        "execution_plan": [{"step": 1, "agent": "quantum_specialist", "goal": "g"}],
    }))

    result = await planner_agent.execute(_planner_state())
    assert result["selected_agent"] == "profile"


async def test_an_out_of_range_confidence_is_clamped(monkeypatch):
    _install(monkeypatch, json.dumps({
        "intent": "x", "agent": "job", "confidence": 47.5,
        "needs_clarification": False, "clarification_question": "",
        "execution_plan": [{"step": 1, "agent": "job", "goal": "g"}],
    }))

    result = await planner_agent.execute(_planner_state())
    assert 0.0 <= result["planner_confidence"] <= 1.0


async def test_a_rate_limited_planner_falls_back_without_crashing(monkeypatch, caplog):
    class _Limited(Exception):
        status_code = 429

    _install(monkeypatch, _Limited())

    with caplog.at_level(logging.ERROR, logger="app.agents.planner_agent"):
        result = await planner_agent.execute(_planner_state())

    assert result["selected_agent"] == "profile"
    assert "Planner error" in result["error"]
    assert any("Planner execution failed" in r.message for r in caplog.records)


async def test_a_timed_out_planner_falls_back_without_crashing(monkeypatch):
    """
    Note the shape difference, which is real and is safe.

    A *parse* failure synthesises a one-step plan; an *exception* leaves
    `execution_plan` empty. Every consumer guards with `or []` and only indexes
    when `len(plan) > 1`, so an empty plan routes as a single step rather than
    breaking — which is the right reading of "the planner produced nothing".
    Asserted here so the difference is recorded rather than discovered.
    """
    _install(monkeypatch, asyncio.TimeoutError())

    result = await planner_agent.execute(_planner_state())

    assert result["selected_agent"] == "profile"
    assert result["error"]
    assert result["execution_plan"] == []
    assert result["current_step_index"] == 0

    # And an empty plan is inert everywhere downstream.
    reflected = await reflect_node(_state(
        execution_plan=result["execution_plan"],
        task_result={"status": "success", "agent": "profile", "result": {"content": "x"}},
    ))
    assert reflected["reflect_outcome"] == "done"


async def test_the_planner_error_reaches_degraded_routing(monkeypatch):
    """
    The fallback sets `error`, and `decide_route` reads it to answer
    deterministic categories from stored records instead of failing the turn.
    That contract must survive the planner's failure handling.
    """
    class _Limited(Exception):
        status_code = 429

    _install(monkeypatch, _Limited())
    result = await planner_agent.execute(_planner_state("what is today's date"))

    assert result["error"]
    route = await workflow.decide_route({
        "user_input": "what is today's date",
        "session_id": "fv-deg",
        "error": result["error"],
        "conversation_history": [],
    })
    assert route == "temporal", "a clock question must survive a planner outage"
