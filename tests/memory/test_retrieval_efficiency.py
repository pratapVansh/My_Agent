"""
What one question is allowed to cost the memory layer.

The audit measured up to eight Cohere calls and eight Qdrant operations for a
single simple question, six of the embeddings being the *same sentence* with
the same input type. None of that was a bug in the memory design — the design
is sound and is deliberately left alone here. It was a scheduling problem: a
cache consulted concurrently by callers who all missed before any of them
wrote, and a category filter applied one step after the cost had been paid.

These tests assert the counts. A test that checked "retrieval returns the right
sections" would have passed against the broken version too.
"""
from __future__ import annotations

import asyncio

import pytest

from app.memory.memory_cache import memory_cache
from app.memory.memory_manager import MemoryManager
from app.memory.retrieval_result import RetrievalResult
from app.memory.sources import QueryCategory
from app.services.cohere_service import CohereService


# ═══════════════════════════════════════════════════════════════════════════
# K · One query, one embedding — even under concurrency
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def cohere():
    return CohereService()


def _fake_embed_response(calls: list, delay: float = 0.02):
    """A client double that records every call and is slow enough to overlap."""
    class _Response:
        embeddings = [[0.1] * 1024]

    class _Client:
        async def embed(self, texts, model, input_type):
            calls.append((tuple(texts), input_type))
            await asyncio.sleep(delay)
            return _Response()

    return _Client()


async def test_concurrent_identical_queries_make_one_api_call(cohere, monkeypatch):
    """
    The exact shape of the production waste.

    `retrieve_preferences`, `retrieve_skills`, `retrieve_projects`, two résumé
    fallbacks and the shadow engine all embed the same query under one
    `asyncio.gather`. The 60s LRU could not help: every one of them reached the
    lookup before the first had written its result.
    """
    calls: list = []
    monkeypatch.setattr(cohere, "_client", _fake_embed_response(calls))

    results = await asyncio.gather(
        *(cohere.embed_text("what is my CGPA", input_type="search_query") for _ in range(6))
    )

    assert len(calls) == 1, f"expected one API call, made {len(calls)}"
    assert all(r == results[0] for r in results), "every caller gets the same vector"


async def test_a_failed_embedding_is_not_cached(cohere, monkeypatch):
    """
    A failure must not poison the key for the next 60 seconds.

    Only the in-flight entry is cleared on error; the LRU is never written. The
    next caller retries against a provider that may since have recovered.
    """
    attempts = {"n": 0}

    class _Client:
        async def embed(self, texts, model, input_type):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("cohere unavailable")

            class _Response:
                embeddings = [[0.2] * 1024]

            return _Response()

    monkeypatch.setattr(cohere, "_client", _Client())
    # The wrapper's own backoff is not what is under test here.
    monkeypatch.setattr(cohere, "_retry_with_backoff", lambda fn, **kw: fn())

    with pytest.raises(RuntimeError):
        await cohere.embed_text("flaky", input_type="search_query")

    assert await cohere.embed_text("flaky", input_type="search_query") == [0.2] * 1024
    assert attempts["n"] == 2


async def test_waiters_fail_together_rather_than_hanging(cohere, monkeypatch):
    """A future nobody resolves is worse than an error everybody sees."""
    class _Client:
        async def embed(self, texts, model, input_type):
            await asyncio.sleep(0.02)
            raise RuntimeError("cohere unavailable")

    monkeypatch.setattr(cohere, "_client", _Client())
    monkeypatch.setattr(cohere, "_retry_with_backoff", lambda fn, **kw: fn())

    outcomes = await asyncio.gather(
        *(cohere.embed_text("doomed", input_type="search_query") for _ in range(4)),
        return_exceptions=True,
    )

    assert len(outcomes) == 4
    assert all(isinstance(o, Exception) for o in outcomes)


async def test_the_in_flight_map_does_not_leak(cohere, monkeypatch):
    calls: list = []
    monkeypatch.setattr(cohere, "_client", _fake_embed_response(calls))

    await asyncio.gather(
        *(cohere.embed_text("clean up", input_type="search_query") for _ in range(3))
    )

    assert cohere._in_flight == {}


async def test_document_embeddings_are_left_alone(cohere, monkeypatch):
    """
    Writes carry different text each time and are not coalesced.

    Sharing a result between two *document* embeds would be a correctness bug,
    not an optimisation — so the single-flight path is deliberately keyed to
    search_query only.
    """
    calls: list = []
    monkeypatch.setattr(cohere, "_client", _fake_embed_response(calls))

    await asyncio.gather(
        *(cohere.embed_text("stored fact", input_type="search_document") for _ in range(3))
    )

    assert len(calls) == 3


# ═══════════════════════════════════════════════════════════════════════════
# L · Retrieval fetches only what the question can use
# ═══════════════════════════════════════════════════════════════════════════

class _RecordingManager(MemoryManager):
    """A manager whose five retrieval legs record rather than call out."""

    def __init__(self):
        super().__init__()
        self.touched: list[str] = []
        self.search_sections = None

        outer = self

        class _ShortTerm:
            async def get_recent_context(self, **kw):
                outer.touched.append("chat")
                return []

            async def get_profile_facts(self, **kw):
                outer.touched.append("profile_facts")
                return []

            async def get_recent_episodes(self, **kw):
                outer.touched.append("episodes")
                return []

        class _Smart:
            async def retrieve_preferences(self, **kw):
                outer.touched.append("preferences")
                return []

        class _LongTerm:
            async def search_all(self, user_id, query, limit=5, sections=None):
                outer.touched.append("long_term")
                outer.search_sections = sections
                return {"resume": {}, "skills": [], "projects": []}

        self.short_term = _ShortTerm()
        self.smart = _Smart()
        self.long_term = _LongTerm()


@pytest.fixture
def manager():
    memory_cache.clear()
    yield _RecordingManager()
    memory_cache.clear()


async def test_a_clock_question_does_not_touch_the_vector_store(manager):
    """
    TEMPORAL_CURRENT renders chat history and nothing else.

    It used to pay for two semantic searches, a résumé scroll walked to
    exhaustion, and up to two fallbacks — then render none of them.
    """
    await manager.retrieve_context(
        user_id="vansh",
        session_id="s1",
        query="what time is it",
        category=QueryCategory.TEMPORAL_CURRENT.value,
    )

    assert "long_term" not in manager.touched
    assert "preferences" not in manager.touched
    assert "episodes" not in manager.touched
    assert "chat" in manager.touched


async def test_a_skills_question_still_reaches_the_vector_store(manager):
    """The filter must narrow the fan-out, not disable it."""
    await manager.retrieve_context(
        user_id="vansh",
        session_id="s1",
        query="what are my skills",
        category=QueryCategory.PROFILE_SKILLS.value,
    )

    assert "long_term" in manager.touched


async def test_the_section_set_reaches_search_all(manager):
    """
    `search_all` skips the individual lookups whose section is excluded.

    Without this the category filter would stop at the front door and
    `retrieve_projects` would still run for a question that cannot render it.
    """
    await manager.retrieve_context(
        user_id="vansh",
        session_id="s1",
        query="what are my skills",
        category=QueryCategory.PROFILE_SKILLS.value,
    )

    assert manager.search_sections is not None
    assert "skills" in manager.search_sections
    assert "projects" not in manager.search_sections


async def test_no_category_still_fetches_everything(manager):
    """The historical behaviour is preserved when nothing narrows it."""
    await manager.retrieve_context(user_id="vansh", session_id="s1", query="anything")

    for leg in ("chat", "preferences", "long_term", "profile_facts", "episodes"):
        assert leg in manager.touched


async def test_search_all_skips_the_fallback_for_an_excluded_section():
    """
    A skipped lookup must not look like an empty one.

    `search_all` falls back to a résumé search when a dedicated collection
    returns NO_DATA — a second embedding and a second Qdrant query each. If
    skipping a section produced NO_DATA, the fallback would fire for it and
    reinstate exactly the cost the section filter exists to avoid.

    The skills fallback *does* fire here and should: that lookup genuinely ran
    and genuinely found nothing, which is the case the fallback is for.
    """
    from app.memory.long_term_memory_qdrant import LongTermMemoryQdrant

    store = LongTermMemoryQdrant()
    called: list[str] = []
    fallbacks: list[str] = []

    async def _skills(*a, **kw):
        called.append("skills")
        return RetrievalResult.no_data()

    async def _projects(*a, **kw):
        called.append("projects")
        return RetrievalResult.no_data()

    async def _resume(*a, **kw):
        called.append("resume")
        return None

    async def _fallback(user_id, query, semantic_type, limit):
        fallbacks.append(semantic_type)
        return []

    store.retrieve_skills = _skills
    store.retrieve_projects = _projects
    store.retrieve_resume = _resume
    store._fallback_resume_search = _fallback

    await store.search_all("vansh", "what are my skills", sections={"skills"})

    assert called == ["skills"], f"only the wanted lookup should run, ran {called}"
    assert fallbacks == ["skills"], (
        f"the excluded section must not fall back, fell back for {fallbacks}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# N · The cache cannot serve one caller's context to another
# ═══════════════════════════════════════════════════════════════════════════

def test_owner_and_guest_scopes_produce_different_cache_keys():
    """
    `RetrievalScope` exists because "whose memory" and "at what visibility" are
    different questions from "which user is asking". A key naming only the
    caller cannot tell an owner's read from a guest's, so the first owner turn
    to populate the cache would serve owner-private context to the next guest
    out of it. The scope parameter existed and no caller passed one.
    """
    owner = MemoryManager._retrieval_scope_key("vansh", ["private", "public"], "PROFILE_SKILLS")
    guest = MemoryManager._retrieval_scope_key("vansh", ["public"], "PROFILE_SKILLS")

    assert owner != guest


def test_the_category_is_part_of_the_cache_identity():
    """
    Two questions with the same text now fetch different sources.

    Without the category in the key, a TEMPORAL turn's context — chat history
    only — could be served to a PROFILE_SKILLS turn as though it were complete,
    and the skills section would be silently missing rather than retrieved.
    """
    a = MemoryManager._retrieval_scope_key("vansh", ["public"], "TEMPORAL_CURRENT")
    b = MemoryManager._retrieval_scope_key("vansh", ["public"], "PROFILE_SKILLS")

    assert a != b


async def test_a_guest_does_not_read_an_owner_cached_entry(manager, monkeypatch):
    """End to end through the real cache, not just the key function."""
    owner_context = {
        "chat_history": [], "preferences": [{"memory": "owner-private"}],
        "long_term": {}, "profile_facts": [], "episodes": [],
    }
    memory_cache.set(
        "vansh", owner_context, "who am I",
        scope=MemoryManager._retrieval_scope_key("vansh", ["private", "public"], "PROFILE_GENERAL"),
    )

    guest_view = await manager.retrieve_context(
        user_id="vansh",
        session_id="s1",
        query="who am I",
        category=QueryCategory.PROFILE_GENERAL.value,
        memory_owner_id="vansh",
        visibilities=["public"],
    )

    assert guest_view["preferences"] != [{"memory": "owner-private"}]
