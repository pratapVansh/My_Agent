"""
What one simple question is allowed to cost.

The symptom was a 429 on an 8,000 TPM account and a spoken turn expiring at its
35-second ceiling. The cause was not the ceiling. "What is my name?" — a
question whose answer is one row in the user's own record — made three model
calls and reserved roughly 13,000 tokens:

    planner            ~4,100   routing a question whose route `agent_for`
                                already owned outright
    reasoning step 1   ~4,700   asking the model whether to call the tool the
                                routing edge had already named as mandatory
    reasoning step 2   ~3,200   reading the result back

and when step 1 answered that predetermined question wrongly — no tool call —
`grounding.enforce` replaced the answer and `reflect_node` re-ran the entire
specialist, doubling it.

Two of those three calls were asking for an answer the system already had. The
tests here pin the counts, because the counts were the defect: every version of
this code produced a correct sentence, and the waste was invisible from it.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from app.agents import base_agent, prefetch
from app.agents.base_agent import BaseAgent
from app.memory.sources import QueryCategory
from app.services import deadline
from app.services.groq_limiter import GroqLimiter
from app.tools.contract import Effect


# ═══════════════════════════════════════════════════════════════════════════
# 1 · The lookup runs before the model is asked whether to run it
# ═══════════════════════════════════════════════════════════════════════════

class _Probe(BaseAgent):
    def __init__(self):
        super().__init__(name="probe", description="test agent")

    async def execute(self, state):  # pragma: no cover — never called
        return state


def _read_tool(name, ran, payload):
    async def _call(tool_input):
        ran.append((name, dict(tool_input)))
        return payload
    return {
        "description": f"{name} description",
        "callable": _call,
        "effect": Effect.READ,
    }


def _final_answer(text):
    return json.dumps({"type": "final", "content": text, "is_complete": True})


def _state(**over):
    base = {
        "user_input": "what is my name",
        "user_id": "vansh",
        "session_id": "budget",
        "query_category": QueryCategory.PROFILE_IDENTITY.value,
        "required_tools": ["get_identity", "get_profile_summary"],
        "conversation_history": [],
        "memory_prompt": "",
    }
    base.update(over)
    return base


async def test_the_required_lookup_runs_before_the_first_completion():
    """
    One completion, not two, and the tool ran anyway.

    The saving is the first call outright: the model used to spend a whole
    completion emitting a tool_call for `get_identity` — the only answer the
    routing edge would have accepted, and one it had already written into the
    prompt as a directive.
    """
    agent = _Probe()
    ran: list = []
    calls: list = []

    async def _final(messages, **kwargs):
        calls.append(messages)
        return _final_answer("You're Vansh.")

    agent.call_groq = _final

    result = await agent.execute_reasoning_loop(
        state=_state(),
        base_system_prompt="You are a profile assistant.",
        tools={"get_identity": _read_tool(
            "get_identity", ran, {"success": True, "found": True, "name": "Vansh"}
        )},
    )

    assert [t for t, _ in ran] == ["get_identity"]
    assert len(calls) == 1
    assert result["grounding"] == "satisfied"
    assert result["tools_used"] == ["get_identity"]
    assert result["final_answer"] == "You're Vansh."


async def test_the_model_is_told_the_lookup_is_already_done():
    """
    The prompt otherwise contains a rule that would buy the second call back.

    `execute_reasoning_loop` tells the model it MUST emit a tool_call before
    its first final answer whenever a listed tool could answer the question —
    the rule that stops it inventing a CGPA. On a pre-grounded turn, obeying
    that rule costs exactly the completion this mechanism removes, so the note
    that overrides it has to actually reach the prompt.
    """
    agent = _Probe()
    ran: list = []
    seen: list = []

    async def _final(messages, **kwargs):
        seen.append("\n".join(str(m.get("content", "")) for m in messages))
        return _final_answer("You're Vansh.")

    agent.call_groq = _final
    await agent.execute_reasoning_loop(
        state=_state(),
        base_system_prompt="You are a profile assistant.",
        tools={"get_identity": _read_tool("get_identity", ran, {"name": "Vansh"})},
    )

    prompt = seen[0]
    assert "ALREADY DONE FOR YOU" in prompt
    assert "Do NOT call get_identity again" in prompt
    # And the observation it is being told to read is actually there.
    assert "Tool get_identity observation" in prompt


async def test_a_failed_lookup_falls_through_to_the_next_candidate():
    """
    `PROFILE_EDUCATION` may be answered by `get_education` or by `get_resume`.
    When the first errors the requirement is still outstanding, so the second
    is tried — which is what the model would have done, without the call.
    """
    agent = _Probe()
    ran: list = []

    async def _boom(tool_input):
        ran.append(("get_education", tool_input))
        raise RuntimeError("store unavailable")

    async def _final(messages, **kwargs):
        return _final_answer("CGPA 8.8.")

    agent.call_groq = _final
    result = await agent.execute_reasoning_loop(
        state=_state(
            user_input="what is my cgpa",
            query_category=QueryCategory.PROFILE_EDUCATION.value,
            required_tools=["get_education", "get_resume"],
        ),
        base_system_prompt="p",
        tools={
            "get_education": {"description": "d", "callable": _boom,
                              "effect": Effect.READ},
            "get_resume": _read_tool(
                "get_resume", ran, {"success": True, "found": True,
                                    "content": "CGPA 8.80"}
            ),
        },
    )

    assert [t for t, _ in ran] == ["get_education", "get_resume"]
    assert result["grounding"] == "satisfied"


async def test_an_empty_store_is_not_asked_a_second_time():
    """
    A lookup that ran and found nothing has answered the question.

    Falling through to the next candidate here is how "I have no résumé on
    file" gets invented from a store that was never supposed to hold it — and
    it would spend a second round trip to do it.
    """
    agent = _Probe()
    ran: list = []

    async def _final(messages, **kwargs):
        return _final_answer("Nothing on file.")

    agent.call_groq = _final
    result = await agent.execute_reasoning_loop(
        state=_state(
            user_input="what is my cgpa",
            query_category=QueryCategory.PROFILE_EDUCATION.value,
            required_tools=["get_education", "get_resume"],
        ),
        base_system_prompt="p",
        tools={
            "get_education": _read_tool(
                "get_education", ran, {"success": True, "found": False}
            ),
            "get_resume": _read_tool("get_resume", ran, {"found": True}),
        },
    )

    assert [t for t, _ in ran] == ["get_education"]
    assert result["grounding"] == "no_data"


async def test_a_tool_the_agent_does_not_hold_is_not_prefetched():
    agent = _Probe()
    ran: list = []

    async def _final(messages, **kwargs):
        return _final_answer("ok")

    agent.call_groq = _final
    await agent.execute_reasoning_loop(
        state=_state(required_tools=["match_job"]),
        base_system_prompt="p",
        tools={"get_identity": _read_tool("get_identity", ran, {"name": "V"})},
    )
    assert ran == []


async def test_a_write_is_never_run_before_the_model():
    """
    The safety floor. `ZERO_ARGUMENT_LOOKUPS` contains only reads, so this can
    only fire if someone adds a consequential tool to it — and the point is
    that doing so must not become a way to execute an action with no model, no
    gateway and no user in the loop.
    """
    agent = _Probe()
    ran: list = []

    async def _final(messages, **kwargs):
        return _final_answer("ok")

    agent.call_groq = _final
    spec = _read_tool("get_identity", ran, {"name": "V"})
    spec["effect"] = Effect.EXTERNAL_WRITE

    await agent.execute_reasoning_loop(
        state=_state(), base_system_prompt="p", tools={"get_identity": spec},
    )
    assert ran == []


def test_a_parameterised_lookup_is_not_prefetchable():
    """
    `get_schedule` answers SCHEDULE_TEMPORAL and defaults `when` to today, so a
    no-argument call on "what classes do I have tomorrow" would return the
    wrong day *and* satisfy the grounding check with it. A prefetch that can
    seed a confidently wrong observation is worse than the call it saves.
    """
    assert "get_schedule" not in prefetch.ZERO_ARGUMENT_LOOKUPS
    assert "match_job" not in prefetch.ZERO_ARGUMENT_LOOKUPS
    assert "recall_explicit_memory" not in prefetch.ZERO_ARGUMENT_LOOKUPS

    assert prefetch.plan(
        {"required_tools": ["get_schedule"]}, ["get_schedule"]
    ) is None


def test_job_search_is_prefetched_so_the_model_cannot_skip_it():
    """
    `job_search` is prefetchable even though it takes a `query`, because its
    wrapper defaults that query to the user's own utterance — the zero-argument
    call is the sentence the model would have passed, not a guess at it.

    Live, 4 of 5 JOB_SEARCH turns never emitted the call at all: the model
    answered from memory, `grounding` discarded the answer as ungrounded, and
    the turn ended on a refusal that invited a retry which failed the same way.
    Running the lookup first makes that failure unreachable by construction.
    """
    assert "job_search" in prefetch.ZERO_ARGUMENT_LOOKUPS

    planned = prefetch.plan({"required_tools": ["job_search"]}, ["job_search"])
    assert planned == ("job_search", {})


async def test_a_job_search_turn_runs_the_lookup_before_the_first_completion():
    """The whole point: the tool has already run when the model is first asked."""
    agent = _Probe()
    ran: list = []
    tools_at_first_call: list = []

    async def _final(messages, **kwargs):
        tools_at_first_call.append([t for t, _ in ran])
        return _final_answer("here are some roles")

    agent.call_groq = _final
    await agent.execute_reasoning_loop(
        state=_state(required_tools=["job_search"]),
        base_system_prompt="p",
        tools={"job_search": _read_tool("job_search", ran, {"results": [{"title": "Dev"}]})},
    )

    # Ran before the model was asked anything, with no arguments.
    assert [t for t, _ in ran] == ["job_search"]
    assert ran[0][1] == {}
    assert tools_at_first_call[0] == ["job_search"]


def test_a_follow_up_subject_narrows_the_prefetched_lookup():
    """
    "Tell me more about TRACE" had its subject resolved deterministically at
    the routing edge. Passing it retrieves the one project rather than the
    whole portfolio — a better observation, and a much smaller one.
    """
    planned = prefetch.plan(
        {"required_tools": ["get_projects"], "followup_subject": "TRACE"},
        ["get_projects"],
    )
    assert planned == ("get_projects", {"query": "TRACE"})


# ═══════════════════════════════════════════════════════════════════════════
# 2 · A turn's sub-waits may not outlive the turn
# ═══════════════════════════════════════════════════════════════════════════

def test_no_deadline_means_no_clamping():
    """Outside a turn — the background worker, a script — nothing changes."""
    assert deadline.remaining() is None
    assert deadline.clamp(20.0) == 20.0
    assert deadline.exhausted() is False


def test_a_wait_is_cut_to_what_the_turn_has_left():
    with deadline.budget(2.0):
        assert 0 < deadline.clamp(20.0) <= 2.0
        assert deadline.exhausted() is False


def test_an_expired_turn_grants_no_wait_at_all():
    with deadline.budget(0.001):
        time.sleep(0.02)
        assert deadline.clamp(20.0) == 0.0
        assert deadline.exhausted() is True


async def test_the_retry_does_not_sleep_through_the_callers_deadline(monkeypatch):
    """
    The 35-second spoken turn, in one test.

    `_RATE_LIMIT_MAX_WAIT` is 15 s and the limiter would wait up to 20 s for
    token budget — 35 s of sleeping inside a 35 s budget, which expires having
    sent nothing. Honouring a Retry-After longer than the turn has left is not
    patience, it is a guaranteed timeout with the provider's blessing.
    """
    slept: list[float] = []

    async def _sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(base_agent.asyncio, "sleep", _sleep)

    class _RateLimited(Exception):
        status_code = 429

    agent = _Probe()

    async def _always_limited(**kwargs):
        raise _RateLimited("rate_limit_exceeded")

    agent.groq_service = type(
        "S", (), {"chat_completion": staticmethod(_always_limited)}
    )()

    with deadline.budget(0.5):
        with pytest.raises(Exception):
            await agent.call_groq(messages=[{"role": "user", "content": "hi"}])

    assert slept == [], (
        "the retry slept past a deadline that leaves no time to use the answer"
    )


async def test_a_request_is_not_sent_after_the_turn_has_expired():
    """
    Issuing it would spend the account's budget on an answer nobody is left to
    receive — and on a rate-limited account that is the most expensive kind of
    request there is.
    """
    sent = {"n": 0}
    agent = _Probe()

    async def _chat(**kwargs):
        sent["n"] += 1
        return {"content": "hi"}

    agent.groq_service = type("S", (), {"chat_completion": staticmethod(_chat)})()

    with deadline.budget(0.001):
        time.sleep(0.02)
        with pytest.raises(asyncio.TimeoutError):
            await agent.call_groq(messages=[{"role": "user", "content": "hi"}])

    assert sent["n"] == 0


async def test_the_limiter_waits_no_longer_than_the_turn_has_left():
    """
    20 s of back-pressure inside a sub-second turn is not back-pressure. The
    request is admitted (or fails) in time for the caller to do something about
    it, and the over-budget admission is counted so the cause stays visible.
    """
    limiter = GroqLimiter(max_concurrency=4, tokens_per_minute=100)
    # One pass first, so the loop-bound primitives exist. Building them resets
    # the bucket to full, which would undo the line below.
    async with limiter.reserve([{"role": "user", "content": "x"}]):
        pass
    limiter._available_tokens = 0.0

    started = time.monotonic()
    with deadline.budget(0.6):
        async with limiter.reserve([{"role": "user", "content": "x" * 400}]):
            pass
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, elapsed
    assert limiter.snapshot()["admitted_over_budget"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3 · The budget is reconciled against what the call actually cost
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_unused_reply_allowance_is_returned():
    """
    Reserving `max_tokens` in full is the only honest estimate *before* a call.
    Keeping it afterwards is not: a reasoning step allowed 700 typically spends
    120, and the other 580 were being held against the minute's budget for a
    reply that was never generated.
    """
    limiter = GroqLimiter(max_concurrency=4, tokens_per_minute=10_000)
    limiter._available_tokens = 10_000.0

    async with limiter.reserve([{"role": "user", "content": "x" * 400}], 700) as r:
        assert r.reserved == 800          # 100 prompt + 700 allowance
        after_reserve = limiter._available_tokens
        await r.settle(220)               # what it really cost

    assert after_reserve == pytest.approx(9_200, abs=1)
    assert limiter._available_tokens == pytest.approx(9_780, abs=1)


async def test_settling_twice_refunds_once():
    limiter = GroqLimiter(max_concurrency=4, tokens_per_minute=10_000)
    limiter._available_tokens = 10_000.0

    async with limiter.reserve([{"role": "user", "content": "x" * 400}], 700) as r:
        await r.settle(200)
        await r.settle(200)

    assert limiter._available_tokens == pytest.approx(9_800, abs=1)


async def test_an_unsettled_reservation_keeps_its_full_estimate():
    """A stream, or a call that raised, stays conservative — as it always was."""
    limiter = GroqLimiter(max_concurrency=4, tokens_per_minute=10_000)
    limiter._available_tokens = 10_000.0

    async with limiter.reserve([{"role": "user", "content": "x" * 400}], 700):
        pass

    assert limiter._available_tokens == pytest.approx(9_200, abs=1)


async def test_a_refund_cannot_overfill_the_bucket():
    """
    Same cap as the refill, for the same reason: a bucket holding more than a
    minute's worth would let the next burst spend a minute it has not lived
    through yet.
    """
    limiter = GroqLimiter(max_concurrency=4, tokens_per_minute=1_000)
    limiter._available_tokens = 1_000.0

    await limiter._refund(5_000)
    assert limiter._available_tokens == 1_000.0
