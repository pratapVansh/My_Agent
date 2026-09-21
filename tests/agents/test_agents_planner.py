"""
The planner: routing, confidence, clarification, and what it does when it dies.

The planner is the only agent whose entire output is a routing decision, which
makes it the one place a model failure is silently survivable — a bad parse
falls back to `profile` and the turn continues. That fallback is correct and it
is also exactly what hides a drifting output format, so most of what follows
pins the fallback rather than the happy path.

Worth stating plainly: the planner's choice is advisory. `decide_route` in the
workflow overrides it for any category that owns its agent, and the tests at
the bottom of this file assert that, because "the planner said email" must not
be a way to reach the email agent with a question about the user's CGPA.
"""
from __future__ import annotations

import pytest

from app.agents import base_agent
from app.agents.planner_agent import (
    FORCE_CLARIFICATION_BELOW,
    PlannerAgent,
)
from tests.support import ScriptedLLM, drive, plan, state


@pytest.fixture
def planner():
    return PlannerAgent()


# ═══════════════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("agent_name", ["profile", "job", "email", "academic"])
async def test_the_planner_routes_to_each_available_agent(planner, agent_name):
    result, _ = await drive(
        planner, [plan(agent=agent_name, intent="do the thing")],
        state("do the thing"),
    )
    assert result["selected_agent"] == agent_name
    assert result["detected_intent"] == "do the thing"
    assert result["current_agent"] == "planner"


async def test_an_unknown_agent_falls_back_to_profile(planner):
    """A hallucinated agent name must not produce an unroutable state."""
    result, _ = await drive(
        planner, [plan(agent="quantum_butler")], state("do something"),
    )
    assert result["selected_agent"] == "profile"


async def test_the_execution_plan_is_captured(planner):
    result, _ = await drive(
        planner,
        [plan(agent="job", steps=[
            {"step": 1, "agent": "job", "goal": "find jobs"},
            {"step": 2, "agent": "email", "goal": "draft a cover letter"},
        ])],
        state("find jobs and write a cover letter"),
    )
    assert len(result["execution_plan"]) == 2
    assert result["execution_plan"][1]["agent"] == "email"
    assert result["current_step_index"] == 0


async def test_the_agent_is_forced_to_match_the_first_plan_step(planner):
    """
    A plan whose first step disagrees with the chosen agent would route the
    turn to one specialist and hand it another's goal.
    """
    result, _ = await drive(
        planner,
        [plan(agent="email", steps=[{"step": 1, "agent": "academic", "goal": "timetable"}])],
        state("what classes do I have"),
    )
    assert result["selected_agent"] == "academic"


# ═══════════════════════════════════════════════════════════════════════════
# Ambiguity and clarification
# ═══════════════════════════════════════════════════════════════════════════

async def test_an_ambiguous_request_can_ask_for_clarification(planner):
    result, _ = await drive(
        planner,
        [plan(agent="profile", confidence=0.4, needs_clarification=True,
              clarification_question="Which project did you mean?")],
        state("tell me about it"),
    )
    assert result["needs_clarification"] is True
    assert result["clarification_question"] == "Which project did you mean?"


async def test_clarification_without_a_question_still_gets_one(planner):
    result, _ = await drive(
        planner,
        [plan(needs_clarification=True, clarification_question="")],
        state("hmm"),
    )
    assert result["clarification_question"].strip() != ""


async def test_very_low_confidence_forces_clarification(planner):
    """
    The server-side safety net. Set below the model's own threshold so ordinary
    0.4–0.6 routing is left to its judgment and only the extreme case is
    overridden.
    """
    result, _ = await drive(
        planner,
        [plan(confidence=FORCE_CLARIFICATION_BELOW - 0.1, needs_clarification=False)],
        state("uhh"),
    )
    assert result["needs_clarification"] is True


async def test_ordinary_low_confidence_is_left_alone(planner):
    result, _ = await drive(
        planner,
        [plan(confidence=0.5, needs_clarification=False)],
        state("find me a job"),
    )
    assert result["needs_clarification"] is False


async def test_confidence_is_clamped(planner):
    result, _ = await drive(planner, [plan(confidence=9.5)], state("x"))
    assert 0.0 <= result["planner_confidence"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Failure
# ═══════════════════════════════════════════════════════════════════════════

async def test_unparseable_output_falls_back_to_profile(planner):
    """Drift in the routing model's format must not take the turn down."""
    result, _ = await drive(planner, ["this is not JSON at all"], state("hello"))
    assert result["selected_agent"] == "profile"
    assert result["needs_clarification"] is False
    assert result["execution_plan"]


async def test_json_wrapped_in_code_fences_is_still_parsed(planner):
    fenced = "```json\n" + plan(agent="job") + "\n```"
    result, _ = await drive(planner, [fenced], state("find jobs"))
    assert result["selected_agent"] == "job"


async def test_a_model_outage_leaves_a_routable_state(planner):
    """
    The live failure seen against an exhausted quota: the planner raises, and
    the turn must still be routable rather than dead. `error` is set so the
    workflow can choose its degraded path.
    """
    llm = ScriptedLLM([], fail_after=0)
    result, _ = await drive(planner, llm, state("what is my name"))

    assert result["selected_agent"] == "profile"
    assert result["error"]
    assert "Planner error" in result["error"]
    assert result["execution_plan"] == []


async def test_the_planner_retries_a_transient_failure(planner, monkeypatch):
    """
    `call_groq` retries with backoff, and this drives the *real* retry rather
    than substituting it: the fake sits at the `groq_service` boundary, below
    the retry loop, so a first-attempt failure has to be genuinely recovered.
    """
    attempts = {"n": 0}

    async def flaky_completion(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("Groq API error: 503 service unavailable")
        return {"content": plan(agent="academic")}

    monkeypatch.setattr(planner.groq_service, "chat_completion", flaky_completion)
    # Retry backoff is 1s then 2s; nothing here needs to actually wait.
    monkeypatch.setattr("app.agents.base_agent.asyncio.sleep", _no_sleep)

    result = await planner.execute(state("what classes do I have"))

    assert attempts["n"] == 2, "the first failure should have been retried"
    assert result["selected_agent"] == "academic"
    assert not result.get("error")


async def test_the_planner_gives_up_after_exhausting_retries(planner, monkeypatch):
    """
    Two attempts, not three, and the second one is the last.

    The count is the point of the change, not an incidental detail. This layer
    used to make three attempts *underneath* an SDK that was itself retrying,
    inside a reasoning loop, inside a reflect loop — so one logical call could
    become dozens of HTTP requests, each of which made a rate limit worse. See
    `_MAX_LLM_ATTEMPTS`.
    """
    attempts = {"n": 0}

    async def always_failing(**kwargs):
        attempts["n"] += 1
        raise RuntimeError("Groq API error: 429 rate limited")

    monkeypatch.setattr(planner.groq_service, "chat_completion", always_failing)
    monkeypatch.setattr("app.agents.base_agent.asyncio.sleep", _no_sleep)

    result = await planner.execute(state("what is my name"))

    assert attempts["n"] == base_agent._MAX_LLM_ATTEMPTS == 2
    assert result["selected_agent"] == "profile"
    assert "Planner error" in result["error"]


async def _no_sleep(_seconds):
    return None


# ═══════════════════════════════════════════════════════════════════════════
# The planner does not get the last word
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_personal_question_reaches_profile_whatever_the_planner_said():
    """
    The planner scores intent from text alone and cannot see the store it would
    otherwise ask the user to substitute for. `decide_route` overrides it.
    """
    from app.agents.workflow import decide_route

    routed = await decide_route(state("what is my CGPA?", selected_agent="email"))
    assert routed == "profile"


async def test_a_timetable_question_reaches_academic_whatever_the_planner_said():
    from app.agents.workflow import decide_route

    routed = await decide_route(state("what classes do I have today?", selected_agent="job"))
    assert routed == "academic"
