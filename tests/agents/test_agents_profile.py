"""
The profile agent — the one that answers questions about the user.

Its registry is the largest in the system (thirteen tools) and the one most
load-bearing for correctness rather than safety: nothing here sends anything,
but everything here is a claim about the user's own data, and the failure mode
is a confident wrong answer rather than a visible error.

So these tests are mostly about the two distinctions the memory work
established, now asserted at the agent boundary where they are actually
consumed:

  * **canonical identity vs. a remembered name** — separate tools, separate
    keys, and `get_identity` must never be able to return the other one
  * **nothing on file vs. the lookup failed** — the same empty list, opposite
    statements, now carried by the tool contract into `answerability`

The registry is the real one throughout. A test that built its own would prove
the loop can call a tool and nothing about whether `get_identity` still exists
or still reads the canonical key.
"""
from __future__ import annotations

import pytest

from app.agents.profile_agent import ProfileAgent
from app.tools.contract import Effect
from tests.support import (
    ScriptedLLM,
    capture_registry,
    drive,
    final,
    state,
    stub_services,
    tool_call,
)


@pytest.fixture
def agent():
    return ProfileAgent()


@pytest.fixture
def services(monkeypatch):
    return stub_services(monkeypatch)


# ═══════════════════════════════════════════════════════════════════════════
# The registry itself
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_real_registry_exposes_the_expected_tools(agent, services):
    tools = await capture_registry(agent)
    for name in (
        "get_identity", "recall_explicit_memory", "get_resume", "get_skills",
        "get_projects", "get_education", "get_experience", "get_achievements",
        "remember_preference", "forget_preference", "list_my_memories",
    ):
        assert name in tools, f"{name} is no longer registered"


async def test_only_forget_preference_is_destructive(agent, services):
    """
    A read-only agent with one irreversible tool. If that ratio changes, it
    should change deliberately.
    """
    tools = await capture_registry(agent)
    destructive = [
        name for name, spec in tools.items()
        if spec.get("effect") is Effect.DESTRUCTIVE
    ]
    assert destructive == ["forget_preference"]
    assert tools["forget_preference"].get("preview") is not None


# ═══════════════════════════════════════════════════════════════════════════
# Identity
# ═══════════════════════════════════════════════════════════════════════════

async def test_identity_is_read_from_the_canonical_key(agent, monkeypatch):
    seen = {}

    async def facts(user_id=None, key=None, **kw):
        seen["key"] = key
        return [{"key": key, "value": "Vansh Pratap Singh"}] if key else []

    stub_services(monkeypatch, get_profile_facts=facts)

    result, _ = await drive(
        agent,
        [tool_call("get_identity"), final("Your name is Vansh Pratap Singh.")],
        state("what is my name"),
    )

    from app.memory.identity import CANONICAL_NAME_KEY
    assert seen["key"] == CANONICAL_NAME_KEY
    assert "Vansh Pratap Singh" in result["task_result"]["result"]["content"]


async def test_identity_falls_back_to_the_resume(agent, monkeypatch):
    stub_services(
        monkeypatch,
        retrieve_resume={"name": "Vansh Pratap Singh", "content": "..."},
    )
    result, llm = await drive(
        agent, [tool_call("get_identity"), final("Your name is Vansh.")],
        state("what is my name"),
    )
    assert "resume" in " ".join(llm.observations()).lower()
    assert result["task_result"]["status"] == "success"


async def test_a_missing_name_is_reported_as_missing_not_invented(agent, services):
    """NO_DATA reaches the model as an explicit emptiness, not an empty dict."""
    result, llm = await drive(
        agent, [tool_call("get_identity"), final("I don't have your name on file.")],
        state("what is my name"),
    )
    observed = " ".join(llm.observations())
    assert "found" in observed or "don't have" in observed
    assert result["answerability"] in ("NO_DATA", "ANSWERABLE")


async def test_a_remembered_name_is_a_different_tool(agent, monkeypatch):
    """
    The bug this whole subsystem exists to prevent, at the agent boundary:
    "what did I ask you to remember" must not reach the canonical key.
    """
    async def facts(user_id=None, key=None, **kw):
        return [{"key": "remembered_name", "value": "Devasi"}]

    stub_services(monkeypatch, get_profile_facts=facts)

    _, llm = await drive(
        agent,
        [tool_call("recall_explicit_memory"), final("You asked me to remember Devasi.")],
        state("what name did I ask you to remember"),
    )
    assert "Devasi" in " ".join(llm.observations())


# ═══════════════════════════════════════════════════════════════════════════
# Résumé sections
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tool,section", [
    ("get_education", "education"),
    ("get_experience", "experience"),
    ("get_achievements", "achievements"),
])
async def test_section_tools_request_their_own_section(agent, monkeypatch, tool, section):
    """
    Each section has its own tool because CGPA lived in an `education` chunk
    that nothing could reach. This asserts the routing to the right section.
    """
    requested = {}

    async def retrieve_section(user_id=None, section=None, **kw):
        requested["section"] = section
        return [{"content": f"{section} content here"}]

    stub_services(monkeypatch, retrieve_section=retrieve_section)

    await drive(agent, [tool_call(tool), final("ok")], state(f"tell me my {section}"))
    assert requested["section"] == section


async def test_cpi_comes_back_from_the_education_section(agent, monkeypatch):
    async def retrieve_section(user_id=None, section=None, **kw):
        return [{"content": "B.Tech IT, RGIPT, CGPA: 8.80 / 10"}]

    stub_services(monkeypatch, retrieve_section=retrieve_section)

    result, llm = await drive(
        agent, [tool_call("get_education"), final("Your CPI is 8.80.")],
        state("what is my current CPI"),
    )
    assert "8.80" in " ".join(llm.observations())
    assert result["answerability"] == "ANSWERABLE"


async def test_projects_pass_the_query_through_for_ranking(agent, monkeypatch):
    """
    Without the query, "tell me about TRACE" listed the whole portfolio and
    left the model to find the right one.
    """
    seen = {}

    async def retrieve_projects(user_id=None, query=None, limit=None, **kw):
        seen["query"] = query
        return [{"title": "TRACE", "content": "a retrieval benchmark"}]

    stub_services(monkeypatch, retrieve_projects=retrieve_projects)

    await drive(
        agent,
        [tool_call("get_projects", query="TRACE"), final("TRACE is your benchmark.")],
        state("tell me about TRACE"),
    )
    assert seen["query"] == "TRACE"


async def test_the_resume_tool_returns_the_document(agent, monkeypatch):
    stub_services(
        monkeypatch,
        retrieve_resume={"name": "Vansh", "content": "Full resume text " * 50},
    )
    result, llm = await drive(
        agent, [tool_call("get_resume"), final("Here is your resume.")],
        state("show me my resume"),
    )
    assert "Full resume text" in " ".join(llm.observations())
    assert result["task_result"]["status"] == "success"


async def test_skills_are_retrieved(agent, monkeypatch):
    stub_services(
        monkeypatch,
        retrieve_skills=[{"content": "Python, FastAPI, Qdrant"}],
    )
    _, llm = await drive(
        agent, [tool_call("get_skills"), final("Python, FastAPI, Qdrant.")],
        state("what are my skills"),
    )
    assert "Python" in " ".join(llm.observations())


# ═══════════════════════════════════════════════════════════════════════════
# Memory writes
# ═══════════════════════════════════════════════════════════════════════════

async def test_remembering_a_preference_writes_it(agent, monkeypatch):
    recorded = stub_services(monkeypatch)
    await drive(
        agent,
        [tool_call("remember_preference", key="preferred_tone", value="concise"),
         final("Saved.")],
        state("remember that I prefer concise answers"),
    )
    assert recorded.saved_facts
    assert recorded.saved_facts[0]["key"] == "preferred_tone"


async def test_forgetting_a_preference_is_gated_not_executed(agent, monkeypatch):
    """
    The DESTRUCTIVE tool. The loop may request it; only a confirmation runs it.
    """
    from app.agents.actions import action_gateway

    action_gateway.reset()
    recorded = stub_services(
        monkeypatch,
        get_profile_facts=_facts_returning("preferred_tone", "concise"),
    )

    result, _ = await drive(
        agent,
        [tool_call("forget_preference", key="preferred_tone"), final("...")],
        state("forget my tone preference"),
    )

    assert recorded.forgotten_keys == [], "deletion ran without confirmation"
    pending = await action_gateway.pending_for(state()["session_id"], state()["user_id"])
    assert len(pending) == 1
    assert "preferred_tone" in pending[0].preview
    action_gateway.reset()


def _facts_returning(key, value):
    async def facts(user_id=None, key_arg=None, **kw):
        return [{"key": key, "value": value}]
    return facts


# ═══════════════════════════════════════════════════════════════════════════
# Failure
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_failing_lookup_is_a_tool_error_not_an_empty_store(agent, monkeypatch):
    """
    The distinction that licenses opposite statements. A raised exception must
    not become "you have no skills on file".
    """
    async def broken(*a, **kw):
        raise RuntimeError("qdrant unreachable")

    stub_services(monkeypatch, retrieve_skills=broken)

    result, _ = await drive(
        agent, [tool_call("get_skills"), final("I couldn't look that up.")],
        state("what are my skills"),
    )
    assert result["answerability"] == "TOOL_ERROR"


async def test_an_empty_store_is_no_data(agent, services):
    result, _ = await drive(
        agent, [tool_call("get_skills"), final("I don't have skills on file.")],
        state("what are my skills"),
    )
    assert result["answerability"] == "NO_DATA"


async def test_an_agent_level_exception_produces_a_failed_envelope(agent, monkeypatch):
    stub_services(monkeypatch)
    llm = ScriptedLLM([], fail_after=0)
    result, _ = await drive(agent, llm, state("what are my skills"))

    assert result["task_result"]["status"] == "failed"
    assert result["error"]


async def test_answering_without_a_tool_asserts_nothing_about_the_store(agent, services):
    result, _ = await drive(agent, [final("Hello!")], state("hi"))
    assert result["answerability"] == ""
    assert result["task_result"]["evidence"] == []
