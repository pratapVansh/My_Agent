"""
Asynchronous extraction pipeline (Phase 3).

The property under test throughout is resilience. This code runs unattended in
a background loop, so the failure mode that matters is not "wrong answer" but
"stopped silently a month ago and nobody noticed". Every stage must isolate its
failures and keep the queue moving.

See docs/MEMORY_ARCHITECTURE.md §3.5.
"""
import asyncio
import json
from datetime import timedelta

import pytest

from app.config import settings
from app.memory.cognition.embedder import EmbeddingPass
from app.memory.cognition.extractor import (
    Candidate,
    MemoryExtractor,
    parse_extraction,
    render_transcript,
)
from app.memory.cognition.ingest import MemoryIngestor, is_credential_bearing
from app.memory.cognition.worker import MemoryWorker
from app.memory.events import (
    MAX_ATTEMPTS,
    EventStatus,
    EventType,
    GroupReadiness,
    MemoryEvent,
)
from app.memory.kinds import EmbeddingStatus, MemoryKind, RecordStatus
from app.memory.record import MemoryRecord, utcnow
from app.memory.stores import InMemoryEventQueue, InMemoryRecordStore
from app.memory.writer import MemoryWriter


# ─────────────────────────────────────────────────────────────────────────
# Batching readiness
# ─────────────────────────────────────────────────────────────────────────

def group(pending=1, age_seconds=0.0):
    return GroupReadiness(
        owner_id="vansh",
        group_key="s1",
        pending=pending,
        oldest=utcnow() - timedelta(seconds=age_seconds),
    )


def test_a_full_batch_is_ready():
    assert group(pending=5).is_ready(batch_size=5, idle_seconds=999) is True


def test_a_partial_recent_batch_is_not_ready():
    assert group(pending=2).is_ready(batch_size=5, idle_seconds=999) is False


def test_an_idle_partial_batch_flushes():
    """
    The idle flush is what stops a two-turn conversation being stranded
    unextracted forever because it never reached the batch size.
    """
    assert group(pending=2, age_seconds=300).is_ready(
        batch_size=5, idle_seconds=180
    ) is True


def test_readiness_tolerates_a_naive_timestamp():
    from datetime import datetime
    stale = GroupReadiness("vansh", "s1", pending=1, oldest=datetime(2020, 1, 1))
    assert stale.is_ready(batch_size=5, idle_seconds=60) is True


# ─────────────────────────────────────────────────────────────────────────
# Queue semantics
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def queue():
    return InMemoryEventQueue()


def turn_event(session="s1", user="hi", assistant="hello"):
    return MemoryEvent(
        owner_id="vansh", event_type=EventType.TURN, group_key=session,
        payload={"user": user, "assistant": assistant},
    )


async def test_claiming_removes_events_from_the_pending_pool(queue):
    for _ in range(3):
        await queue.enqueue(turn_event())
    assert await queue.pending_count() == 3

    claimed = await queue.claim_group("s1")
    assert len(claimed) == 3
    assert await queue.pending_count() == 0


async def test_a_second_claim_returns_nothing(queue):
    """Two workers polling at once must never process the same conversation."""
    await queue.enqueue(turn_event())
    assert len(await queue.claim_group("s1")) == 1
    assert await queue.claim_group("s1") == []


async def test_claiming_increments_the_attempt_counter(queue):
    await queue.enqueue(turn_event())
    claimed = await queue.claim_group("s1")
    assert claimed[0].attempts == 1


async def test_failure_returns_events_for_retry(queue):
    """A transient LLM outage must not discard what the user said."""
    await queue.enqueue(turn_event())
    claimed = await queue.claim_group("s1")
    await queue.mark_failed([e.id for e in claimed], "groq timeout")
    assert await queue.pending_count() == 1


async def test_events_are_parked_once_attempts_are_exhausted(queue):
    await queue.enqueue(turn_event())
    for _ in range(MAX_ATTEMPTS):
        claimed = await queue.claim_group("s1")
        await queue.mark_failed([e.id for e in claimed], "still broken")

    assert await queue.pending_count() == 0
    assert queue.all_events()[0].status is EventStatus.FAILED


async def test_groups_are_isolated_from_each_other(queue):
    await queue.enqueue(turn_event(session="s1"))
    await queue.enqueue(turn_event(session="s2"))
    assert len(await queue.claim_group("s1")) == 1
    assert await queue.pending_count() == 1


async def test_skipped_is_distinct_from_failed(queue):
    """Small talk yielding no fact is a normal outcome, never a retry."""
    await queue.enqueue(turn_event())
    claimed = await queue.claim_group("s1")
    await queue.mark_done([e.id for e in claimed], EventStatus.SKIPPED)
    assert await queue.pending_count() == 0
    assert queue.all_events()[0].status is EventStatus.SKIPPED


# ─────────────────────────────────────────────────────────────────────────
# Extraction parsing — must survive whatever the model emits
# ─────────────────────────────────────────────────────────────────────────

def valid_payload(**overrides):
    entry = {
        "kind": "semantic", "content": "The user knows Python and FastAPI.",
        "importance": 0.6, "confidence": 0.9, "dedup_key": None,
    }
    entry.update(overrides)
    return json.dumps({"memories": [entry]})


def test_parses_a_well_formed_response():
    candidates = parse_extraction(valid_payload())
    assert len(candidates) == 1
    assert candidates[0].kind is MemoryKind.SEMANTIC
    assert candidates[0].importance == 0.6


def test_parses_a_response_wrapped_in_prose_or_fences():
    wrapped = f"Here you go:\n```json\n{valid_payload()}\n```"
    assert len(parse_extraction(wrapped)) == 1


@pytest.mark.parametrize("raw", ["", "   ", "not json at all", "[]", "null", "{}"])
def test_unusable_responses_yield_nothing_rather_than_raising(raw):
    assert parse_extraction(raw) == []


def test_a_malformed_entry_does_not_discard_its_valid_siblings():
    payload = json.dumps({"memories": [
        {"kind": "nonsense", "content": "This kind does not exist at all."},
        {"kind": "semantic", "content": "The user knows Python and FastAPI."},
        "not even an object",
    ]})
    candidates = parse_extraction(payload)
    assert len(candidates) == 1
    assert candidates[0].kind is MemoryKind.SEMANTIC


def test_fragments_too_short_to_stand_alone_are_dropped():
    """"yes" stored alone and read months later is worse than nothing."""
    assert parse_extraction(valid_payload(content="yes")) == []


def test_non_extractable_kinds_are_refused():
    """
    document/procedural/relation come from other sources; letting the model
    emit them would be letting it invent provenance it does not have.
    """
    for kind in ("document", "procedural", "relation"):
        assert parse_extraction(valid_payload(kind=kind)) == []


def test_salience_values_are_clamped():
    assert parse_extraction(valid_payload(importance=99))[0].importance == 1.0
    assert parse_extraction(valid_payload(confidence=-5))[0].confidence == 0.0


def test_non_numeric_salience_falls_back_to_defaults():
    assert parse_extraction(valid_payload(importance="high"))[0].importance == 0.5


def test_unnamespaced_dedup_keys_are_discarded():
    """An unnamespaced key would collide across unrelated attributes."""
    assert parse_extraction(valid_payload(dedup_key="name"))[0].dedup_key is None
    assert parse_extraction(valid_payload(dedup_key="profile:name"))[0].dedup_key == "profile:name"


def test_runaway_responses_are_capped():
    payload = json.dumps({"memories": [
        {"kind": "semantic", "content": f"The user knows technology number {i}."}
        for i in range(50)
    ]})
    assert len(parse_extraction(payload)) <= 12


def test_transcript_rendering_labels_and_trims():
    text = render_transcript(
        [{"role": "user", "content": "x" * 900}, {"role": "assistant", "content": "ok"}],
        max_chars=100,
    )
    assert text.startswith("User: ")
    assert "Assistant: ok" in text
    assert "x" * 101 not in text


async def test_extractor_returns_nothing_for_an_empty_transcript():
    class NeverCalled:
        async def chat_completion(self, **kwargs):
            raise AssertionError("the LLM must not be called for an empty transcript")

    assert await MemoryExtractor(llm=NeverCalled()).extract("vansh", []) == []


# ─────────────────────────────────────────────────────────────────────────
# Ingestion: governance, dedup, reconciliation
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def store():
    return InMemoryRecordStore()


@pytest.fixture
def ingestor(store):
    return MemoryIngestor(record_store=store, writer=MemoryWriter(record_store=store))


@pytest.mark.parametrize("text", [
    "The user's password is hunter2.",
    "The user's API key was shared in chat.",
    "The user's credit card number is on file.",
])
def test_credential_bearing_text_is_detected(text):
    assert is_credential_bearing(text) is True


def test_ordinary_facts_are_not_flagged():
    assert is_credential_bearing("The user knows Python and FastAPI.") is False


async def test_credential_bearing_candidates_are_rejected(ingestor, store):
    """Defence in depth: the extractor is told not to, which is not enforcement."""
    result = await ingestor.ingest("vansh", [
        Candidate(kind=MemoryKind.SEMANTIC, content="The user's password is hunter2."),
        Candidate(kind=MemoryKind.SEMANTIC, content="The user knows Python well."),
    ])
    assert result.rejected == 1
    assert result.created == 1
    assert await store.count("vansh") == 1


async def test_exact_duplicates_are_counted_not_stored_twice(ingestor, store):
    candidate = Candidate(kind=MemoryKind.SEMANTIC, content="The user knows Python.")
    first = await ingestor.ingest("vansh", [candidate])
    second = await ingestor.ingest("vansh", [candidate])

    assert first.created == 1
    assert second.created == 0
    assert second.duplicates == 1
    assert await store.count("vansh") == 1


async def test_a_contradicting_fact_supersedes_rather_than_duplicating(ingestor, store):
    await ingestor.ingest("vansh", [Candidate(
        kind=MemoryKind.IDENTITY, content="The user's role is student.",
        dedup_key="profile:role",
    )])
    result = await ingestor.ingest("vansh", [Candidate(
        kind=MemoryKind.IDENTITY, content="The user's role is engineer.",
        dedup_key="profile:role",
    )])

    assert result.superseded == 1
    assert await store.count("vansh", statuses=[RecordStatus.ACTIVE]) == 1
    assert await store.count("vansh", statuses=[RecordStatus.SUPERSEDED]) == 1

    current = await store.find_by_dedup_key("vansh", "profile:role")
    assert "engineer" in current[0].content
    assert current[0].version == 2


async def test_facts_without_a_dedup_key_accumulate(ingestor, store):
    """Two goals do not contradict; two values for `profile:role` do."""
    await ingestor.ingest("vansh", [
        Candidate(kind=MemoryKind.GOAL, content="The user wants an SDE internship."),
        Candidate(kind=MemoryKind.GOAL, content="The user wants to learn Rust."),
    ])
    assert await store.count("vansh", statuses=[RecordStatus.ACTIVE]) == 2


async def test_a_store_failure_is_counted_not_raised(store):
    class BrokenWriter:
        async def upsert_with_outcome(self, record):
            raise RuntimeError("database gone")

    ingestor = MemoryIngestor(record_store=store, writer=BrokenWriter())
    result = await ingestor.ingest("vansh", [
        Candidate(kind=MemoryKind.SEMANTIC, content="The user knows Python.")
    ])
    assert result.failed == 1
    assert result.created == 0


# ─────────────────────────────────────────────────────────────────────────
# Embedding pass
# ─────────────────────────────────────────────────────────────────────────

class FakeVectorStore:
    def __init__(self):
        self.upserted = []

    async def upsert(self, entries):
        self.upserted.extend(entries)


async def seed(store, count=3):
    for i in range(count):
        await store.add(MemoryRecord(
            owner_id="vansh", kind=MemoryKind.SEMANTIC,
            content=f"The user knows technology number {i}.",
        ))


async def test_embedding_marks_records_ready(store):
    await seed(store)
    vectors = FakeVectorStore()

    async def embedder(texts):
        return [[0.1] * 8 for _ in texts]

    stats = await EmbeddingPass(store, vectors, embedder).run_once()
    assert stats.embedded == 3
    assert len(vectors.upserted) == 3
    assert await store.pending_embeddings() == []


async def test_a_provider_outage_leaves_records_pending(store):
    """
    Marking them failed would exclude them from semantic search permanently
    over what is usually a transient outage.
    """
    await seed(store)

    async def broken(texts):
        raise RuntimeError("cohere unavailable")

    stats = await EmbeddingPass(store, FakeVectorStore(), broken).run_once()
    assert stats.failed == 3
    assert stats.embedded == 0
    assert len(await store.pending_embeddings()) == 3


async def test_a_partial_embedding_batch_is_refused(store):
    """
    Pairing records with the wrong vectors is worse than not embedding: the
    resulting search results look plausible.
    """
    await seed(store)
    vectors = FakeVectorStore()

    async def short_batch(texts):
        return [[0.1] * 8 for _ in texts[:-1]]

    stats = await EmbeddingPass(store, vectors, short_batch).run_once()
    assert stats.embedded == 0
    assert vectors.upserted == []
    assert len(await store.pending_embeddings()) == 3


async def test_a_vector_store_failure_leaves_records_pending(store):
    await seed(store)

    class BrokenVectors:
        async def upsert(self, entries):
            raise RuntimeError("qdrant down")

    async def embedder(texts):
        return [[0.1] * 8 for _ in texts]

    stats = await EmbeddingPass(store, BrokenVectors(), embedder).run_once()
    assert stats.embedded == 0
    assert len(await store.pending_embeddings()) == 3


async def test_an_empty_queue_is_a_no_op(store):
    async def never(texts):
        raise AssertionError("must not embed when nothing is pending")

    stats = await EmbeddingPass(store, FakeVectorStore(), never).run_once()
    assert stats.claimed == 0


# ─────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────

class StubExtractor:
    def __init__(self, candidates=None, error=None):
        self.candidates = candidates or []
        self.error = error
        self.calls = []

    async def extract(self, owner_id, turns):
        self.calls.append((owner_id, turns))
        if self.error:
            raise self.error
        return self.candidates


class NoOpEmbedder:
    async def run_once(self, limit=32):
        from app.memory.cognition.embedder import EmbedStats
        return EmbedStats()


class NoOpSummarizer:
    """
    Stands in for the real summariser, which would otherwise reach Postgres.

    Injected explicitly rather than left to default: the worker swallows stage
    failures, so a real summariser here would silently open a database
    connection on every test and still pass.
    """

    async def run_once(self, limit=5):
        from app.memory.cognition.summarizer import SummaryStats
        return SummaryStats()


class NoOpMaintenance:
    """
    Stands in for the real maintenance sweeps, which reach Postgres.

    Same reasoning as NoOpSummarizer: the worker swallows stage failures, so a
    real sweep here would open a database connection on every test and the
    tests would still pass.
    """

    async def run_once(self):
        from app.memory.cognition.maintenance import MaintenanceStats
        return MaintenanceStats()


def build_worker(queue, store, extractor):
    return MemoryWorker(
        event_queue=queue,
        extractor=extractor,
        ingestor=MemoryIngestor(record_store=store, writer=MemoryWriter(record_store=store)),
        embedder=NoOpEmbedder(),
        summarizer=NoOpSummarizer(),
        maintenance=NoOpMaintenance(),
    )


async def test_worker_extracts_and_stores_a_ready_group(queue, store, monkeypatch):
    monkeypatch.setattr(settings, "memory_extraction_batch_size", 2)
    for i in range(2):
        await queue.enqueue(turn_event(user=f"question {i}", assistant=f"answer {i}"))

    extractor = StubExtractor([
        Candidate(kind=MemoryKind.SEMANTIC, content="The user is learning Rust.")
    ])
    stats = await build_worker(queue, store, extractor).run_once()

    assert stats.groups == 1
    assert stats.events == 2
    assert stats.created == 1
    assert await store.count("vansh") == 1
    assert await queue.pending_count() == 0


async def test_worker_batches_several_turns_into_one_extraction(queue, store, monkeypatch):
    """
    The point of batching: one LLM call per N turns, and a window wide enough
    that a fact spanning turns can actually be stated.
    """
    monkeypatch.setattr(settings, "memory_extraction_batch_size", 3)
    for i in range(3):
        await queue.enqueue(turn_event(user=f"question {i}", assistant=f"answer {i}"))

    extractor = StubExtractor()
    await build_worker(queue, store, extractor).run_once()

    assert len(extractor.calls) == 1
    _, turns = extractor.calls[0]
    assert len(turns) == 6  # three exchanges, two messages each


async def test_worker_leaves_an_unready_group_alone(queue, store, monkeypatch):
    monkeypatch.setattr(settings, "memory_extraction_batch_size", 5)
    monkeypatch.setattr(settings, "memory_extraction_idle_flush_seconds", 9999)
    await queue.enqueue(turn_event())

    extractor = StubExtractor()
    stats = await build_worker(queue, store, extractor).run_once()

    assert stats.groups == 0
    assert extractor.calls == []
    assert await queue.pending_count() == 1


async def test_extraction_yielding_nothing_is_skipped_not_retried(queue, store, monkeypatch):
    monkeypatch.setattr(settings, "memory_extraction_batch_size", 1)
    await queue.enqueue(turn_event())

    await build_worker(queue, store, StubExtractor([])).run_once()

    assert await queue.pending_count() == 0
    assert queue.all_events()[0].status is EventStatus.SKIPPED


async def test_a_failing_group_is_retried_not_lost(queue, store, monkeypatch):
    monkeypatch.setattr(settings, "memory_extraction_batch_size", 1)
    await queue.enqueue(turn_event())

    extractor = StubExtractor(error=RuntimeError("groq exploded"))
    stats = await build_worker(queue, store, extractor).run_once()

    assert stats.failed_groups == 1
    assert await queue.pending_count() == 1  # back for another attempt


async def test_one_bad_group_does_not_block_the_others(queue, store, monkeypatch):
    """A malformed conversation must not stop every other one being extracted."""
    monkeypatch.setattr(settings, "memory_extraction_batch_size", 1)
    await queue.enqueue(turn_event(session="a", user="broken", assistant="x"))
    await queue.enqueue(turn_event(session="b", user="fine", assistant="y"))

    async def extract(owner_id, turns):
        if turns[0]["content"] == "broken":
            raise RuntimeError("bad payload")
        return [Candidate(kind=MemoryKind.SEMANTIC, content="The user likes Rust.")]

    extractor = StubExtractor()
    extractor.extract = extract

    stats = await build_worker(queue, store, extractor).run_once()

    assert stats.failed_groups == 1
    assert stats.created == 1


async def test_worker_never_raises_into_its_caller(store, monkeypatch):
    """The loop must survive a queue that is entirely broken."""
    class BrokenQueue:
        async def ready_groups(self, **kwargs):
            raise RuntimeError("database unreachable")

    worker = MemoryWorker(
        event_queue=BrokenQueue(),
        extractor=StubExtractor(),
        ingestor=MemoryIngestor(record_store=store, writer=MemoryWriter(record_store=store)),
        embedder=NoOpEmbedder(),
        summarizer=NoOpSummarizer(),
        maintenance=NoOpMaintenance(),
    )
    stats = await worker.run_once()
    assert stats.groups == 0


async def test_a_failing_embedding_pass_does_not_fail_the_cycle(queue, store, monkeypatch):
    monkeypatch.setattr(settings, "memory_extraction_batch_size", 1)
    await queue.enqueue(turn_event())

    class BrokenEmbedder:
        async def run_once(self, limit=32):
            raise RuntimeError("embedder exploded")

    worker = MemoryWorker(
        event_queue=queue,
        extractor=StubExtractor([Candidate(kind=MemoryKind.SEMANTIC,
                                           content="The user is learning Rust.")]),
        ingestor=MemoryIngestor(record_store=store, writer=MemoryWriter(record_store=store)),
        embedder=BrokenEmbedder(),
        summarizer=NoOpSummarizer(),
        maintenance=NoOpMaintenance(),
    )
    stats = await worker.run_once()
    assert stats.created == 1  # extraction still succeeded


async def test_run_forever_stops_when_asked(queue, store, monkeypatch):
    monkeypatch.setattr(settings, "memory_worker_poll_seconds", 0.05)
    worker = build_worker(queue, store, StubExtractor())

    stop = asyncio.Event()
    task = asyncio.create_task(worker.run_forever(stop))
    await asyncio.sleep(0.15)
    stop.set()
    await asyncio.wait_for(task, timeout=2.0)
    assert task.done()
