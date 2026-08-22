"""
What each kind of question costs the memory layer.

The section filter existed before this work but was applied by
`format_context_for_prompt` — after retrieval had already paid for every
source. This measures the cost per category at the point it is now incurred,
and it measures it in the units that were actually running out: Cohere calls
and Qdrant operations.

The tests are paired on purpose. For every "this must be skipped" there is a
"this must still be fetched", because the failure mode of an over-aggressive
filter is not slowness, it is an answer that quietly omits what it needed —
which is far worse than the cost it saves.
"""
from __future__ import annotations

import pytest

from app.memory.memory_cache import memory_cache
from app.memory.memory_manager import MemoryManager
from app.memory.sources import QueryCategory


class _CountingManager(MemoryManager):
    """Counts Cohere and Qdrant work per retrieval, by leg."""

    def __init__(self):
        super().__init__()
        self.legs: list[str] = []
        self.embeddings = 0
        self.qdrant_ops = 0
        self.sections_seen = None
        outer = self

        class _ShortTerm:
            async def get_recent_context(self, **kw):
                outer.legs.append("chat_history")
                return [{"role": "user", "content": "hi"}]

            async def get_profile_facts(self, **kw):
                outer.legs.append("profile_facts")
                return [{"key": "name", "value": "Vansh"}]

            async def get_recent_episodes(self, **kw):
                outer.legs.append("episodes")
                return []

        class _Smart:
            async def retrieve_preferences(self, user_id, query=None, limit=5):
                outer.legs.append("preferences")
                if query:
                    outer.embeddings += 1      # embed_text(search_query)
                    outer.qdrant_ops += 1      # query_points
                return []

        class _LongTerm:
            async def search_all(self, user_id, query, limit=5, sections=None):
                outer.legs.append("long_term")
                outer.sections_seen = sections
                allowed = sections
                if allowed is None or "skills" in allowed:
                    outer.legs.append("skills_lookup")
                    outer.embeddings += 1
                    outer.qdrant_ops += 1
                if allowed is None or "projects" in allowed:
                    outer.legs.append("projects_lookup")
                    outer.embeddings += 1
                    outer.qdrant_ops += 1
                if allowed is None or "resume" in allowed:
                    outer.legs.append("resume_scroll")
                    outer.qdrant_ops += 1      # scroll, no embedding
                return {"resume": {}, "skills": [], "projects": []}

        self.short_term = _ShortTerm()
        self.smart = _Smart()
        self.long_term = _LongTerm()


@pytest.fixture
def manager():
    memory_cache.clear()
    yield _CountingManager()
    memory_cache.clear()


async def _retrieve(manager, category: str, query: str = "what is my college"):
    return await manager.retrieve_context(
        user_id="vansh", session_id="cost-1", query=query, category=category,
    )


# ── The question from the brief ──────────────────────────────────────────

async def test_what_is_my_college_skips_projects_skills_and_episodes(manager):
    """
    "What is my college?" is PROFILE_EDUCATION — answered from the résumé's
    education section. It has no use for a semantic search over projects or
    skills, and no use for episodic memory.
    """
    await _retrieve(manager, QueryCategory.PROFILE_EDUCATION.value)

    assert "projects_lookup" not in manager.legs
    assert "skills_lookup" not in manager.legs
    assert "episodes" not in manager.legs
    assert "preferences" not in manager.legs

    # And it still reads what it needs.
    assert "resume_scroll" in manager.legs
    assert "profile_facts" in manager.legs

    assert manager.embeddings == 0, "an education lookup needs no embedding"
    assert manager.qdrant_ops == 1, "one résumé scroll, nothing else"


# ── Per-category cost table ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "category, expect_embeddings, expect_qdrant, must_include, must_exclude",
    [
        # A clock question needs the thread and nothing else.
        (QueryCategory.TEMPORAL_CURRENT.value, 0, 0,
         ["chat_history"], ["long_term", "preferences", "episodes", "profile_facts"]),

        # Education: résumé section only.
        (QueryCategory.PROFILE_EDUCATION.value, 0, 1,
         ["resume_scroll", "profile_facts"], ["skills_lookup", "projects_lookup"]),

        # Skills: the skills collection plus the résumé, never projects.
        (QueryCategory.PROFILE_SKILLS.value, 1, 2,
         ["skills_lookup", "resume_scroll"], ["projects_lookup", "episodes"]),

        # Projects: the mirror image.
        (QueryCategory.PROFILE_PROJECTS.value, 1, 2,
         ["projects_lookup", "resume_scroll"], ["skills_lookup", "episodes"]),

        # Episodic memory reads episodes, not the résumé.
        (QueryCategory.EPISODIC_MEMORY.value, 0, 0,
         ["episodes", "chat_history"], ["long_term", "skills_lookup"]),

        # General knowledge needs stated preferences and the thread — not a CV.
        (QueryCategory.GENERAL_KNOWLEDGE.value, 1, 1,
         ["preferences", "chat_history"], ["long_term", "skills_lookup", "episodes"]),

        # Small talk is the cheapest turn in the system.
        (QueryCategory.SMALL_TALK.value, 0, 0,
         ["chat_history", "profile_facts"], ["long_term", "preferences", "episodes"]),

        # A résumé question is the one that legitimately costs the most.
        (QueryCategory.DOCUMENT_RESUME.value, 2, 3,
         ["skills_lookup", "projects_lookup", "resume_scroll"], ["preferences"]),
    ],
)
async def test_retrieval_cost_per_category(
    manager, category, expect_embeddings, expect_qdrant, must_include, must_exclude
):
    await _retrieve(manager, category)

    for leg in must_include:
        assert leg in manager.legs, f"{category} must fetch {leg}, got {manager.legs}"
    for leg in must_exclude:
        assert leg not in manager.legs, f"{category} must not fetch {leg}"

    assert manager.embeddings == expect_embeddings, (
        f"{category}: expected {expect_embeddings} embeddings, made {manager.embeddings}"
    )
    assert manager.qdrant_ops == expect_qdrant, (
        f"{category}: expected {expect_qdrant} Qdrant ops, made {manager.qdrant_ops}"
    )


async def test_an_unknown_category_falls_back_to_fetching_everything(manager):
    """
    The safe default. An unrecognised category must not silently answer from a
    narrowed context — it reverts to the historical behaviour.
    """
    await _retrieve(manager, "SOMETHING_NEW_AND_UNMAPPED")

    for leg in ("chat_history", "preferences", "long_term", "profile_facts", "episodes"):
        assert leg in manager.legs


async def test_no_category_fetches_everything(manager):
    await manager.retrieve_context(
        user_id="vansh", session_id="cost-2", query="anything", category=None,
    )

    for leg in ("chat_history", "preferences", "long_term", "profile_facts", "episodes"):
        assert leg in manager.legs


# ── The filter must not lose information ─────────────────────────────────

async def test_a_skills_question_still_renders_its_skills(manager):
    """
    The point of the pairing: a cheaper turn that answers worse is a regression,
    not an optimisation.
    """
    context = await _retrieve(manager, QueryCategory.PROFILE_SKILLS.value)

    assert manager.sections_seen is not None
    assert "skills" in manager.sections_seen
    assert "long_term" in manager.legs
    assert context["profile_facts"] == [{"key": "name", "value": "Vansh"}]


async def test_the_cached_path_respects_the_same_filter(manager):
    """
    A cache hit refreshes chat history and profile facts. Those refreshes must
    obey the category too, or a TEMPORAL turn served from cache would fetch
    profile facts the first one correctly skipped.
    """
    await _retrieve(manager, QueryCategory.TEMPORAL_CURRENT.value)
    first = list(manager.legs)
    manager.legs.clear()

    await _retrieve(manager, QueryCategory.TEMPORAL_CURRENT.value)

    assert "profile_facts" not in manager.legs, (
        f"cache-hit refresh ignored the category filter: {manager.legs}"
    )
    assert manager.legs, "the cache hit refreshed nothing at all"
    assert "profile_facts" not in first
