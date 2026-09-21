"""
The job agent — search, ranking, bookmarking, and application drafting.

This is the agent the Jarvis roadmap expands next, so these tests are as much a
baseline for that work as a check on today's behaviour. Two things they pin
deliberately:

  * **"Apply" drafts; it does not apply.** `draft_application_email` composes
    and stores. Nothing in this agent submits anything, and nothing in it sends
    the draft — that remains the email agent's gated tool.
  * **A search that finds nothing is not a search that failed.** Tavily returning
    an empty list is NO_DATA; Tavily raising is TOOL_ERROR. The agent must not
    collapse them, because "there are no backend jobs in Delhi" and "I couldn't
    reach the job board" are different statements about the world.
"""
from __future__ import annotations

import pytest

from app.agents.job_agent import JobAgent
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
    return JobAgent()


@pytest.fixture
def services(monkeypatch):
    return stub_services(monkeypatch)


def _results(*titles):
    return {
        "tool": "job_search", "success": True, "user_skills": ["Python", "FastAPI"],
        "total_candidates": len(titles), "total_filtered": len(titles),
        "results": [
            {
                "title": title, "company": "Acme", "url": f"https://x.test/{i}",
                "snippet": "…", "rank_score": 0.9 - i * 0.1,
                "skills_matched": ["Python"],
            }
            for i, title in enumerate(titles)
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_real_registry_exposes_the_expected_tools(agent, services):
    tools = await capture_registry(agent)
    assert set(tools) == {
        "job_search", "save_job_bookmark", "get_bookmarked_jobs",
        "draft_application_email", "match_job",
    }


async def test_nothing_in_the_job_agent_is_consequential(agent, services):
    """
    Searching and bookmarking touch only the user's own data. If a tool here
    ever becomes EXTERNAL_WRITE — an actual application submission — it should
    be a deliberate change that breaks this test.
    """
    tools = await capture_registry(agent)
    for name, spec in tools.items():
        assert spec["effect"] in (Effect.READ, Effect.LOCAL_WRITE), name


# ═══════════════════════════════════════════════════════════════════════════
# Search
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_search_returns_ranked_results(agent, monkeypatch):
    stub_services(monkeypatch, search_jobs=_results("Backend Engineer", "Platform Engineer"))

    result, llm = await drive(
        agent,
        [tool_call("job_search", query="backend engineer"),
         final("I found 2 roles.")],
        state("find me backend jobs"),
    )

    observed = " ".join(llm.observations())
    assert "Backend Engineer" in observed
    assert result["answerability"] == "ANSWERABLE"


async def test_search_arguments_reach_the_tool(agent, monkeypatch):
    seen = {}

    async def search(user_id=None, query=None, location=None, max_results=None, **kw):
        seen.update({"query": query, "location": location, "max_results": max_results})
        return _results("Backend Engineer")

    stub_services(monkeypatch, search_jobs=search)

    await drive(
        agent,
        [tool_call("job_search", query="backend", location="Bangalore", max_results=3),
         final("ok")],
        state("backend jobs in Bangalore"),
    )
    assert seen["query"] == "backend"
    assert seen["location"] == "Bangalore"
    assert seen["max_results"] == 3


async def test_raw_results_are_attached_for_the_frontend(agent, monkeypatch):
    """The envelope carries structured results, not only prose."""
    stub_services(monkeypatch, search_jobs=_results("A", "B", "C"))

    result, _ = await drive(
        agent, [tool_call("job_search", query="x"), final("Found 3.")],
        state("find jobs"),
    )
    jobs = result["task_result"]["result"].get("jobs")
    assert jobs and len(jobs) == 3
    assert jobs[0]["title"] == "A"


async def test_no_results_is_no_data_not_an_error(agent, services):
    result, _ = await drive(
        agent,
        [tool_call("job_search", query="underwater basket weaving"),
         final("I didn't find any matching roles.")],
        state("find underwater basket weaving jobs"),
    )
    assert result["answerability"] == "NO_DATA"
    assert result["task_result"]["status"] == "success"


async def test_a_failing_search_is_a_tool_error(agent, monkeypatch):
    async def broken(*a, **kw):
        raise RuntimeError("tavily unreachable")

    stub_services(monkeypatch, search_jobs=broken)

    result, _ = await drive(
        agent, [tool_call("job_search", query="x"), final("I couldn't search right now.")],
        state("find jobs"),
    )
    assert result["answerability"] == "TOOL_ERROR"


async def test_a_malformed_search_result_does_not_become_evidence(agent, monkeypatch):
    """
    The fail-safe rule for a READ tool: an unrecognisable shape is tolerated
    rather than refused, but a *falsy* one must not be reported as findings.
    """
    stub_services(monkeypatch, search_jobs={"success": True, "results": []})

    result, _ = await drive(
        agent, [tool_call("job_search", query="x"), final("Nothing found.")],
        state("find jobs"),
    )
    assert result["answerability"] == "NO_DATA"


# ═══════════════════════════════════════════════════════════════════════════
# Bookmarks
# ═══════════════════════════════════════════════════════════════════════════

async def test_bookmarking_a_job_writes_it(agent, monkeypatch):
    recorded = stub_services(monkeypatch)

    await drive(
        agent,
        [tool_call("save_job_bookmark", title="Backend Engineer",
                   url="https://x.test/1", company="Acme"),
         final("Saved.")],
        state("save that job"),
    )
    assert recorded.bookmarks == [{"title": "Backend Engineer", "url": "https://x.test/1"}]


async def test_a_bookmark_without_a_url_is_refused(agent, services):
    """The tool's own precondition — asserted so a refactor cannot drop it."""
    result, _ = await drive(
        agent,
        [tool_call("save_job_bookmark", title="No URL"), final("I need the link.")],
        state("save that"),
    )
    assert result["answerability"] == "TOOL_ERROR"


async def test_listing_bookmarks_with_none_saved_is_no_data(agent, services):
    result, _ = await drive(
        agent, [tool_call("get_bookmarked_jobs"), final("No saved jobs yet.")],
        state("show my saved jobs"),
    )
    assert result["answerability"] == "NO_DATA"


# ═══════════════════════════════════════════════════════════════════════════
# Application drafting
# ═══════════════════════════════════════════════════════════════════════════

async def test_applying_drafts_but_does_not_send(agent, monkeypatch):
    """
    "Apply to this job" prepares. The agent has no path to delivery — sending
    lives behind the email agent's gated tool.
    """
    recorded = stub_services(monkeypatch)

    result, _ = await drive(
        agent,
        [tool_call("draft_application_email", job_title="Backend Engineer",
                   company="Acme", job_url="https://x.test/1"),
         final("I've drafted your application.")],
        state("apply to that job"),
    )

    assert recorded.saved_drafts, "the draft should have been stored"
    assert recorded.emails_sent == 0, "the job agent must never send"
    assert result["task_result"]["status"] == "success"


async def test_a_draft_failure_does_not_claim_success(agent, monkeypatch):
    async def broken(*a, **kw):
        raise RuntimeError("groq down")

    stub_services(monkeypatch, draft_email=broken)

    result, _ = await drive(
        agent,
        [tool_call("draft_application_email", job_title="X", company="Y"),
         final("I couldn't draft that.")],
        state("apply"),
    )
    assert result["answerability"] == "TOOL_ERROR"


# ═══════════════════════════════════════════════════════════════════════════
# Retry
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_retried_search_can_succeed_on_the_second_attempt(agent, monkeypatch):
    """
    The reflect loop's shape: fail, then try differently. Read-only, so
    repeating is harmless — which is exactly why the effect matters.
    """
    attempts = {"n": 0}

    async def flaky(*a, **kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")
        return _results("Backend Engineer")

    stub_services(monkeypatch, search_jobs=flaky)

    result, _ = await drive(
        agent,
        [tool_call("job_search", query="backend"),
         tool_call("job_search", query="backend engineer"),
         final("Found one after retrying.")],
        state("find jobs"),
    )

    assert attempts["n"] == 2
    # One call failed and one produced evidence — partial, not clean.
    assert result["answerability"] == "PARTIALLY_ANSWERABLE"


async def test_an_agent_level_failure_produces_a_failed_envelope(agent, monkeypatch):
    stub_services(monkeypatch)
    result, _ = await drive(agent, ScriptedLLM([], fail_after=0), state("find jobs"))

    assert result["task_result"]["status"] == "failed"
    assert result["task_result"]["confidence"] == 0.0
    assert "Job agent error" in result["error"]


# ═══════════════════════════════════════════════════════════════════════════
# The Tavily boundary
#
# Everything above stubs `search_jobs` wholesale, which is right for testing
# the agent but means the provider call itself has never been exercised. These
# drive the real `job_search_tool` against a faked Tavily client, because the
# distinction that matters most here — an empty board versus an unreachable one
# — is decided inside that method and nowhere else.
# ═══════════════════════════════════════════════════════════════════════════

import asyncio  # noqa: E402

from tavily.errors import InvalidAPIKeyError, UsageLimitExceededError  # noqa: E402

from app.config import settings  # noqa: E402
from app.domain.jobs import jobs_repository  # noqa: E402
from app.tools.job_search_tool import JobSearchTool  # noqa: E402


class _FakeTavilyClient:
    """Stands in for `tavily.AsyncTavilyClient`. Returns, or raises."""

    payload: object = {"results": []}

    def __init__(self, api_key=None):
        self.api_key = api_key

    async def search(self, **kwargs):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


@pytest.fixture
def tavily(monkeypatch):
    """A configured key plus a fake client; set `tavily.payload` per test."""
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-real-key-123")
    import tavily as tavily_pkg
    monkeypatch.setattr(tavily_pkg, "AsyncTavilyClient", _FakeTavilyClient)
    return _FakeTavilyClient


@pytest.fixture
def tool(monkeypatch):
    """The real search tool with its memory and Postgres reads stubbed out."""
    instance = JobSearchTool()

    async def _no_skills(*a, **kw):
        return []

    async def _no_seen(*a, **kw):
        return set()

    monkeypatch.setattr(instance, "_fetch_user_skills", _no_skills)
    monkeypatch.setattr(jobs_repository, "get_bookmarked_urls", _no_seen)
    return instance


async def test_a_mocked_tavily_response_becomes_ranked_results(tool, tavily):
    tavily.payload = {
        "results": [
            {
                "title": "Backend Engineer at Acme - LinkedIn",
                "url": "https://linkedin.com/jobs/1",
                "content": "Python and FastAPI experience required.",
                "score": 0.9,
            },
            {
                "title": "Junior Developer | Globex - Indeed",
                "url": "https://indeed.com/jobs/2",
                "content": "Entry level role.",
                "score": 0.6,
            },
        ]
    }

    result = await tool.search_jobs(user_id="owner", query="backend", max_results=5)

    assert result["success"] is True
    assert [job["title"] for job in result["results"]] == [
        "Backend Engineer at Acme", "Junior Developer | Globex",
    ]
    assert result["results"][0]["company"] == "Acme"
    # Ranked, not merely passed through.
    assert result["results"][0]["rank_score"] >= result["results"][1]["rank_score"]


async def test_a_placeholder_api_key_refuses_instead_of_searching(tool, monkeypatch):
    """
    The failure this test exists for: an unfilled .env used to produce an empty
    result list, which the agent then reported as "no jobs found" — a claim
    about the job market derived from a missing credential.
    """
    monkeypatch.setattr(settings, "tavily_api_key", "your_tavily_api_key_here")
    called = False

    async def _must_not_run(*a, **kw):
        nonlocal called
        called = True
        return [], None

    monkeypatch.setattr(tool, "_search_with_tavily", _must_not_run)

    result = await tool.search_jobs(user_id="owner", query="backend")

    assert result["success"] is False
    assert "TAVILY_API_KEY" in result["error"]
    assert not called, "no search should be attempted without a usable key"


async def test_an_empty_board_is_not_an_error(tool, tavily):
    """Tavily answering with nothing is a fact about the board, not a failure."""
    tavily.payload = {"results": []}

    result = await tool.search_jobs(user_id="owner", query="underwater basket weaving")

    assert result["success"] is True
    assert result["results"] == []
    assert "error" not in result


@pytest.mark.parametrize("raised, expected", [
    (UsageLimitExceededError("quota"), "rate limited"),
    (InvalidAPIKeyError(), "rejected TAVILY_API_KEY"),
    (asyncio.TimeoutError(), "timed out"),
])
async def test_a_provider_failure_is_reported_as_a_failure(tool, tavily, raised, expected):
    """Never an empty list: each of these would otherwise read as 'no jobs'."""
    tavily.payload = raised

    result = await tool.search_jobs(user_id="owner", query="backend")

    assert result["success"] is False
    assert expected in result["error"]


async def test_a_skipped_search_reports_the_missing_key_not_a_generic_refusal(
    agent, services, monkeypatch
):
    """
    The gap found by running this live: the model answered a JOB_SEARCH turn
    without calling anything, grounding replaced its answer with the generic
    "let me check job listings again", and the user was never told the reason
    was an unset key sitting in their own .env.

    Grounding cannot know that — it only sees that no tool ran. The agent can.
    """
    monkeypatch.setattr(settings, "tavily_api_key", "your_tavily_api_key_here")

    result, _ = await drive(
        agent,
        [final("Here are some roles you might like!")],
        state("find me backend jobs", query_category="JOB_SEARCH"),
    )

    answer = result["task_result"]["result"]["content"]
    assert "TAVILY_API_KEY" in answer
    assert "check" not in answer.lower() or "TAVILY_API_KEY" in answer


async def test_a_configured_key_leaves_the_grounding_refusal_alone(
    agent, services, monkeypatch
):
    """The substitution is about configuration, not about covering for a skip."""
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-real-key-123")

    result, _ = await drive(
        agent,
        [final("Here are some roles you might like!")],
        state("find me backend jobs", query_category="JOB_SEARCH"),
    )

    answer = result["task_result"]["result"]["content"]
    assert "TAVILY_API_KEY" not in answer


async def test_a_search_that_errored_on_the_missing_key_reports_it(
    agent, monkeypatch
):
    """
    The case seen live, and the more common of the two: the model *did* call
    job_search, the tool refused for want of a key, and grounding replaced the
    answer with "try me again in a second" — advice that can never work.
    """
    monkeypatch.setattr(settings, "tavily_api_key", "your_tavily_api_key_here")
    stub_services(monkeypatch)
    # The real tool, so its own refusal is what reaches the agent.
    monkeypatch.setattr(
        "app.agents.job_agent.job_search_tool", JobSearchTool(), raising=False
    )

    result, _ = await drive(
        agent,
        [tool_call("job_search", query="backend"), final("Here are some roles!")],
        state("find me backend jobs", query_category="JOB_SEARCH"),
    )

    answer = result["task_result"]["result"]["content"]
    assert "TAVILY_API_KEY" in answer
    assert "again in a second" not in answer
