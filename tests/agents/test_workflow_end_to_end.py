"""
The whole path: a sentence in, a decision, an agent, a tool, an answer out.

The per-agent files test specialists in isolation. This tests the seams between
them — routing, the terminal nodes that answer without a model, the reflect
loop, and the confirmation path — because every bug found live so far has lived
in a seam rather than in a component.

Two of those live findings are pinned here permanently: LangGraph discards
state written by a conditional-edge function, and a turn that fails must not
overwrite the provenance of the last one that succeeded. Both looked fine in
unit tests and both were wrong in a running conversation.
"""
from __future__ import annotations

import pytest

from app.agents.actions import action_gateway
from app.agents.confirmation import resolve
from app.agents.email_agent import EmailAgent
from app.agents.profile_agent import ProfileAgent
from app.agents.workflow import (
    confirm_action_node,
    create_workflow,
    decide_route,
    provenance_node,
    temporal_node,
)
from app.memory import provenance
from app.memory.sources import QueryCategory
from tests.support import (
    drive,
    final,
    state,
    stub_services,
    tool_call,
)


@pytest.fixture(autouse=True)
def _clean():
    action_gateway.reset()
    provenance.reset()
    yield
    action_gateway.reset()
    provenance.reset()


@pytest.fixture
def services(monkeypatch):
    return stub_services(monkeypatch)


# ═══════════════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query,expected", [
    ("what is my name?", "profile"),
    ("what are my skills?", "profile"),
    ("what is my current CPI?", "profile"),
    ("what classes do I have today?", "academic"),
    ("am I free tomorrow?", "academic"),
    ("what is today's date?", "temporal"),
    ("how did you know?", "provenance"),
])
async def test_queries_reach_their_owning_route(query, expected):
    assert await decide_route(state(query)) == expected


def test_the_graph_registers_every_route_it_can_return():
    """
    A route with no node is a crash at runtime and nothing at import time.
    This is the check that would have caught it.
    """
    graph = create_workflow()
    nodes = set(getattr(graph, "nodes", {}) or {})
    for route in (
        "profile", "job", "email", "academic", "temporal", "degraded",
        "provenance", "confirm_action", "clarification", "response",
    ):
        assert route in nodes, f"route '{route}' has no node"


# ═══════════════════════════════════════════════════════════════════════════
# Terminal nodes — answers with no model
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_clock_answers_without_a_model():
    s = state("what is today's date?")
    s["query_category"] = QueryCategory.TEMPORAL_CURRENT.value
    await temporal_node(s)

    assert s["display_text"]
    assert s["current_agent"] == "temporal"
    assert s["task_result"]["status"] == "success"


async def test_the_clock_answers_even_when_the_model_is_down():
    """Found live against an exhausted quota: this route needs no model."""
    routed = await decide_route(state("what is today's date?", error="Groq 429"))
    assert routed == "temporal"


async def test_provenance_explains_the_previous_answer():
    s = state("what is today's date?")
    s["query_category"] = QueryCategory.TEMPORAL_CURRENT.value
    await temporal_node(s)

    asked = state("how did you know?")
    await provenance_node(asked)

    assert "clock" in asked["display_text"]
    assert "résumé" not in asked["display_text"]


async def test_provenance_admits_when_nothing_was_recorded():
    asked = state("how did you know?", session_id="fresh")
    await provenance_node(asked)
    assert asked["display_text"] == provenance.NO_RECORD


async def test_a_failed_turn_does_not_erase_the_previous_provenance():
    """Found live. A failure must not wipe the record of a real answer."""
    from app.agents.workflow import _record_provenance

    good = state("what is today's date?")
    good["query_category"] = QueryCategory.TEMPORAL_CURRENT.value
    await temporal_node(good)
    assert "clock" in provenance.explain_last(good["session_id"])

    failed = state("what are my skills?", error="Groq 429")
    failed["task_result"] = {"agent": "profile", "evidence": [], "status": "failed"}
    _record_provenance(failed)

    assert "clock" in provenance.explain_last(good["session_id"])


async def test_sources_are_derived_when_the_router_could_not_supply_them():
    """
    Found live: LangGraph discards state written by a conditional-edge
    function, so `memory_sources` never reaches the node. Deriving it from the
    category is what keeps the provenance record truthful.
    """
    from app.agents.workflow import _record_provenance

    s = state("what is today's date?")
    s["query_category"] = QueryCategory.TEMPORAL_CURRENT.value
    s["memory_sources"] = []            # what the node actually sees
    s["task_result"] = {"agent": "temporal", "evidence": ["current_datetime"],
                        "status": "success"}
    _record_provenance(s)

    assert "clock" in provenance.explain_last(s["session_id"])


# ═══════════════════════════════════════════════════════════════════════════
# NO_DATA and TOOL_ERROR reach the response layer
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_no_data_path_end_to_end(services):
    agent = ProfileAgent()
    result, _ = await drive(
        agent, [tool_call("get_skills"), final("I don't have skills on file.")],
        state("what are my skills?"),
    )
    assert result["answerability"] == "NO_DATA"
    assert result["task_result"]["status"] == "success"


async def test_the_tool_error_path_end_to_end(monkeypatch):
    async def broken(*a, **kw):
        raise RuntimeError("qdrant unreachable")

    stub_services(monkeypatch, retrieve_skills=broken)

    result, _ = await drive(
        ProfileAgent(),
        [tool_call("get_skills"), final("I couldn't look that up right now.")],
        state("what are my skills?"),
    )
    assert result["answerability"] == "TOOL_ERROR"


async def test_no_data_and_tool_error_are_never_the_same_verdict(monkeypatch):
    """
    The distinction the whole memory layer protects, asserted at the top of the
    stack: an empty store and a broken one must not produce one verdict.
    """
    empty = stub_services(monkeypatch)
    a, _ = await drive(ProfileAgent(), [tool_call("get_skills"), final("x")],
                       state("skills?"))

    async def broken(*args, **kw):
        raise RuntimeError("down")

    monkeypatch.setattr(
        "app.memory.long_term_memory_qdrant.long_term_memory_qdrant.retrieve_skills",
        broken, raising=False,
    )
    b, _ = await drive(ProfileAgent(), [tool_call("get_skills"), final("x")],
                       state("skills?"))

    assert a["answerability"] != b["answerability"]
    assert {a["answerability"], b["answerability"]} == {"NO_DATA", "TOOL_ERROR"}


# ═══════════════════════════════════════════════════════════════════════════
# Clarification
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_personal_question_never_reaches_clarification():
    """
    Asking the user to disambiguate their own CGPA is a failure to look, not a
    clarification — so the planner's request is suppressed.
    """
    s = state("what is my CGPA?", needs_clarification=True,
              clarification_question="Which CGPA?")
    assert await decide_route(s) == "profile"
    assert s["needs_clarification"] is False


async def test_an_underspecified_action_may_clarify_once():
    from app.agents import clarification_policy

    clarification_policy.reset()
    s = state("send this to him", needs_clarification=True,
              clarification_question="Send what, to whom?")
    assert await decide_route(s) == "clarification"
    clarification_policy.reset()


async def test_the_clarification_budget_is_one_per_conversation():
    from app.agents import clarification_policy

    clarification_policy.reset()
    first = state("send this to him", needs_clarification=True,
                  clarification_question="Who?")
    assert await decide_route(first) == "clarification"

    second = state("schedule a meeting", needs_clarification=True,
                   clarification_question="When?")
    assert await decide_route(second) != "clarification"
    clarification_policy.reset()


# ═══════════════════════════════════════════════════════════════════════════
# The confirmation path, whole
# ═══════════════════════════════════════════════════════════════════════════

async def test_prepare_route_confirm_execute(services):
    """
    The acceptance criterion of T2+T3, through the real agent and the real
    routing: two turns, one email.
    """
    # Turn 1 — the model asks to send.
    prepared, _ = await drive(
        EmailAgent(),
        [tool_call("send_email", to_email="alice@example.com",
                   subject="Hi", body="Hello."),
         final("Ready for your approval.")],
        state("send this email to alice"),
    )
    assert services.emails_sent == 0
    assert prepared["pending_actions"]

    # Turn 2 — an ambiguous reply changes nothing.
    assert await decide_route(state("okay")) == "confirm_action"
    ambiguous = state("okay")
    await confirm_action_node(ambiguous)
    assert services.emails_sent == 0

    # Turn 3 — an explicit yes sends exactly once.
    assert await decide_route(state("yes")) == "confirm_action"
    approved = state("yes")
    await confirm_action_node(approved)
    assert services.emails_sent == 1
    assert "alice@example.com" in approved["display_text"]

    # Turn 4 — nothing pending, so "yes" is ordinary conversation again.
    assert await decide_route(state("yes")) != "confirm_action"


async def test_a_question_asked_while_an_action_is_pending_still_routes_normally(services):
    await drive(
        EmailAgent(),
        [tool_call("send_email", to_email="a@b.com", subject="s", body="b"), final("...")],
        state("send it"),
    )
    assert await decide_route(state("what is my CGPA?")) == "profile"
    assert services.emails_sent == 0


# ═══════════════════════════════════════════════════════════════════════════
# Reflect / retry
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_retry_can_recover_within_one_turn(monkeypatch):
    attempts = {"n": 0}

    async def flaky(*a, **kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")
        return [{"content": "Python, FastAPI"}]

    stub_services(monkeypatch, retrieve_skills=flaky)

    result, _ = await drive(
        ProfileAgent(),
        [tool_call("get_skills"), tool_call("get_skills"), final("Python and FastAPI.")],
        state("what are my skills?"),
    )
    assert attempts["n"] == 2
    assert result["answerability"] == "PARTIALLY_ANSWERABLE"


async def test_a_retry_cannot_duplicate_an_external_action(services):
    """
    The invariant that makes retrying safe at all: reads may repeat freely,
    external writes may not.
    """
    await drive(
        EmailAgent(),
        [tool_call("send_email", to_email="a@b.com", subject="s", body="b"), final("...")],
        state("send it"),
    )
    await resolve(state("yes"))
    assert services.emails_sent == 1

    for _ in range(3):
        await drive(
            EmailAgent(),
            [tool_call("send_email", to_email="a@b.com", subject="s", body="b"),
             final("...")],
            state("send it"),
        )
    assert services.emails_sent == 1
