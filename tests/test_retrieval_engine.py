"""
Hybrid retrieval and context assembly (Phase 2).

Two properties carry most of the weight here:

* **Graceful degradation.** Any channel can fail — Qdrant down, no embeddings
  computed yet, the full-text index missing. Retrieval must return what it can
  rather than failing the turn.
* **Tier-0 is guaranteed.** Identity, preferences and active goals are never
  dropped for budget. Forgetting who the user is to make room for a résumé
  fragment would be a worse failure than dropping the fragment.

See docs/MEMORY_ARCHITECTURE.md §3.6–3.7.
"""
from datetime import timedelta

import pytest

from app.memory.kinds import MemoryKind, Sensitivity, Visibility
from app.memory.record import MemoryRecord, utcnow
from app.memory.retrieval import ContextAssembler, RetrievalEngine
from app.memory.retrieval.assembler import TIER_0_KINDS, tier_for
from app.memory.stores import InMemoryLexicalIndex, InMemoryRecordStore


CORPUS = [
    (MemoryKind.IDENTITY, "The user's name is Vansh Pratap Singh."),
    (MemoryKind.PREFERENCE, "The user's preferred tone is concise."),
    (MemoryKind.GOAL, "The user's goal is an SDE internship by June 2027."),
    (MemoryKind.SEMANTIC, "The user knows Python, FastAPI and PostgreSQL."),
    (MemoryKind.EPISODIC, "The user asked about ML jobs. The job agent listed five roles."),
    (MemoryKind.DOCUMENT, "Built My_Agent, a personal assistant using Python and LangGraph."),
    (MemoryKind.PROCEDURAL, "When the job agent used job_search with ML filters it found roles."),
]


@pytest.fixture
def store():
    return InMemoryRecordStore()


@pytest.fixture
async def populated(store):
    for kind, content in CORPUS:
        await store.add(MemoryRecord(owner_id="vansh", kind=kind, content=content))
    return store


@pytest.fixture
def engine(populated):
    return RetrievalEngine(populated, lexical_index=InMemoryLexicalIndex(populated))


# ─────────────────────────────────────────────────────────────────────────
# Channels
# ─────────────────────────────────────────────────────────────────────────

async def test_all_three_channels_are_reported(engine):
    result = await engine.retrieve("vansh", "python projects")
    assert {c.name for c in result.trace.channels} == {"vector", "lexical", "structured"}


async def test_structured_channel_always_supplies_the_guaranteed_kinds(engine):
    """These belong in context whether or not the question mentions them."""
    result = await engine.retrieve("vansh", "something entirely unrelated")
    kinds = {item.record.kind for item in result.scored}
    assert TIER_0_KINDS <= kinds


async def test_lexical_matches_outrank_unrelated_records(engine):
    result = await engine.retrieve("vansh", "python fastapi postgresql")
    top = result.scored[0].record
    assert "Python" in top.content


async def test_similarity_is_zero_for_structured_only_candidates(engine):
    result = await engine.retrieve("vansh", "python")
    identity = next(i for i in result.scored if i.record.kind is MemoryKind.IDENTITY)
    assert identity.similarity == 0.0
    # It still appears — priors alone justify it.
    assert identity.score > 0


async def test_results_are_sorted_by_score(engine):
    result = await engine.retrieve("vansh", "python projects")
    scores = [item.score for item in result.scored]
    assert scores == sorted(scores, reverse=True)


async def test_limit_is_respected(engine):
    result = await engine.retrieve("vansh", "python", limit=2)
    assert len(result.scored) == 2


# ─────────────────────────────────────────────────────────────────────────
# Graceful degradation
# ─────────────────────────────────────────────────────────────────────────

async def test_missing_vector_channel_is_not_an_error(engine):
    """No embeddings computed yet is the normal state early in rollout."""
    result = await engine.retrieve("vansh", "python")
    vector = next(c for c in result.trace.channels if c.name == "vector")
    assert vector.ok is True
    assert vector.candidates == 0
    assert result.scored


async def test_a_failing_channel_degrades_rather_than_raises(populated):
    class BrokenLexical:
        async def search(self, *args, **kwargs):
            raise RuntimeError("index unavailable")

    engine = RetrievalEngine(populated, lexical_index=BrokenLexical())
    result = await engine.retrieve("vansh", "python")

    lexical = next(c for c in result.trace.channels if c.name == "lexical")
    assert lexical.ok is False
    assert "index unavailable" in lexical.error
    assert result.trace.degraded is True
    # Structured candidates still came through.
    assert result.scored


async def test_a_failing_vector_channel_degrades_rather_than_raises(populated):
    class BrokenVectors:
        async def search(self, *args, **kwargs):
            raise RuntimeError("qdrant unreachable")

    async def embedder(text):
        return [0.1] * 8

    engine = RetrievalEngine(populated, vector_store=BrokenVectors(), embedder=embedder)
    result = await engine.retrieve("vansh", "python")
    assert result.trace.degraded is True
    assert result.scored


async def test_every_channel_failing_still_returns_a_trace(store):
    class Broken:
        async def search(self, *a, **k):
            raise RuntimeError("down")
        async def list(self, *a, **k):
            raise RuntimeError("down")
        async def get_many(self, *a, **k):
            raise RuntimeError("down")

    engine = RetrievalEngine(Broken(), lexical_index=Broken())
    result = await engine.retrieve("vansh", "python")
    assert result.scored == []
    assert result.trace.degraded is True


# ─────────────────────────────────────────────────────────────────────────
# Safety and scoping
# ─────────────────────────────────────────────────────────────────────────

async def test_secret_records_never_reach_the_candidate_set(store):
    await store.add(MemoryRecord(
        owner_id="vansh", kind=MemoryKind.SEMANTIC,
        content="A secret the model must never see.",
        sensitivity=Sensitivity.SECRET,
    ))
    engine = RetrievalEngine(store, lexical_index=InMemoryLexicalIndex(store))
    result = await engine.retrieve("vansh", "secret")
    assert result.scored == []


async def test_retrieval_is_scoped_to_the_owner(store):
    await store.add(MemoryRecord(
        owner_id="guest-abc", kind=MemoryKind.SEMANTIC, content="Guest private note."
    ))
    await store.add(MemoryRecord(
        owner_id="vansh", kind=MemoryKind.SEMANTIC, content="Owner private note."
    ))
    engine = RetrievalEngine(store, lexical_index=InMemoryLexicalIndex(store))
    result = await engine.retrieve("vansh", "note")
    assert all(item.record.owner_id == "vansh" for item in result.scored)


async def test_visibility_filter_scopes_the_structured_channel(store):
    """The mechanism behind the recruiter fix in Phase 6."""
    await store.add(MemoryRecord(
        owner_id="vansh", kind=MemoryKind.IDENTITY,
        content="The user's name is Vansh.", visibility=Visibility.PUBLIC,
    ))
    await store.add(MemoryRecord(
        owner_id="vansh", kind=MemoryKind.IDENTITY,
        content="The user's private phone number is on file.",
        visibility=Visibility.PRIVATE,
    ))
    engine = RetrievalEngine(store)
    result = await engine.retrieve("vansh", "", visibilities=[Visibility.PUBLIC])
    assert [i.record.visibility for i in result.scored] == [Visibility.PUBLIC]


# ─────────────────────────────────────────────────────────────────────────
# Context assembly and budget
# ─────────────────────────────────────────────────────────────────────────

async def test_assembly_groups_records_under_section_titles(engine):
    result = await engine.retrieve("vansh", "python projects")
    context = ContextAssembler().assemble(result.scored, trace=result.trace)
    assert "About the user:" in context.text
    assert "User preferences:" in context.text
    assert "Active goals:" in context.text


async def test_working_memory_is_rendered_last(engine):
    result = await engine.retrieve("vansh", "python")
    context = ContextAssembler().assemble(
        result.scored, working_memory="User: hi\nAssistant: hello", trace=result.trace
    )
    assert "Recent conversation:" in context.text
    assert context.text.index("Recent conversation:") > context.text.index("About the user:")


async def test_tier_zero_survives_an_absurdly_small_budget(engine):
    """
    The guarantee. Under extreme pressure the assistant may lose a résumé
    fragment; it must not lose who it is talking to.
    """
    result = await engine.retrieve("vansh", "python projects")
    context = ContextAssembler(budget_tokens=10).assemble(result.scored, trace=result.trace)

    kinds = {item.record.kind for item in context.selected}
    assert TIER_0_KINDS <= kinds
    assert "About the user:" in context.text


async def test_lower_tiers_are_dropped_before_tier_zero(engine):
    result = await engine.retrieve("vansh", "python projects")
    context = ContextAssembler(budget_tokens=60).assemble(result.scored, trace=result.trace)

    dropped_kinds = {d.kind for d in result.trace.dropped}
    assert not (dropped_kinds & {k.value for k in TIER_0_KINDS})


async def test_dropped_records_are_recorded_in_the_trace(engine):
    result = await engine.retrieve("vansh", "python projects")
    ContextAssembler(budget_tokens=50).assemble(result.scored, trace=result.trace)
    assert result.trace.dropped
    assert all(d.reason.endswith("_budget") for d in result.trace.dropped)


async def test_a_generous_budget_drops_nothing(engine):
    result = await engine.retrieve("vansh", "python projects")
    ContextAssembler(budget_tokens=100_000).assemble(result.scored, trace=result.trace)
    assert result.trace.dropped == []


async def test_records_are_never_cut_mid_sentence(engine):
    """
    Allocation-before-render can only drop whole records. Truncation-after-
    render produced fragments the model reads as complete statements.
    """
    result = await engine.retrieve("vansh", "python projects")
    context = ContextAssembler(budget_tokens=80).assemble(result.scored, trace=result.trace)
    for item in context.selected:
        assert item.record.content in context.text


async def test_assembly_records_budget_utilisation(engine):
    result = await engine.retrieve("vansh", "python")
    ContextAssembler(budget_tokens=6000).assemble(result.scored, trace=result.trace)
    assert result.trace.budget_tokens == 6000
    assert result.trace.used_tokens > 0
    assert 0.0 < result.trace.budget_utilisation < 1.0


def test_empty_selection_renders_an_empty_context():
    context = ContextAssembler().assemble([])
    assert context.text == ""
    assert not context


def test_tier_assignment_matches_the_taxonomy():
    def rec(kind):
        return MemoryRecord(owner_id="vansh", kind=kind, content="x y z")

    assert tier_for(rec(MemoryKind.IDENTITY)) == 0
    assert tier_for(rec(MemoryKind.PREFERENCE)) == 0
    assert tier_for(rec(MemoryKind.GOAL)) == 0
    assert tier_for(rec(MemoryKind.SEMANTIC)) == 2
    assert tier_for(rec(MemoryKind.DOCUMENT)) == 2
    assert tier_for(rec(MemoryKind.PROCEDURAL)) == 3
    assert tier_for(rec(MemoryKind.RELATION)) == 3


# ─────────────────────────────────────────────────────────────────────────
# Trace
# ─────────────────────────────────────────────────────────────────────────

async def test_trace_summary_is_log_safe(engine):
    """Summaries go to logs; they must not carry full record content."""
    result = await engine.retrieve("vansh", "python")
    summary = result.trace.summary()
    assert summary["owner_id"] == "vansh"
    assert isinstance(summary["by_kind"], dict)
    assert "content" not in summary


async def test_trace_counts_selection_by_kind(engine):
    result = await engine.retrieve("vansh", "python")
    ContextAssembler().assemble(result.scored, trace=result.trace)
    by_kind = result.trace.selected_by_kind()
    assert by_kind.get("identity") == 1
    assert sum(by_kind.values()) == len(result.trace.selected)
