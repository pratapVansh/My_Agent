"""
Memory lifecycle: decay, reconciliation, de-duplication, forgetting (Phase 5).

These jobs are the only ones in the system that *remove* things, so the tests
are weighted toward what must never be removed: pinned records, identity,
preferences, goals, and anything too young to have proven itself. A decay
engine that is slightly too lazy wastes rows; one that is slightly too eager
loses the user's name.

See docs/MEMORY_ARCHITECTURE.md §3.9.
"""
from datetime import timedelta
from uuid import uuid4

import pytest

from app.config import settings
from app.memory.cognition.maintenance import (
    DEDUPABLE_KINDS,
    MemoryMaintenance,
    effective_score,
    is_decay_exempt,
)
from app.memory.kinds import MemoryKind, RecordStatus
from app.memory.ports import VectorHit
from app.memory.record import MemoryRecord, utcnow
from app.memory.stores import InMemoryRecordStore


def make(store_age_days=0, **overrides):
    created = utcnow() - timedelta(days=store_age_days)
    base = dict(
        owner_id="vansh",
        kind=MemoryKind.SEMANTIC,
        content="The user knows Python and FastAPI.",
        created_at=created,
        occurred_at=created,
    )
    base.update(overrides)
    return MemoryRecord(**base)


@pytest.fixture
def store():
    return InMemoryRecordStore()


class FakeVectors:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.deleted = []

    async def search(self, owner_id, vector, *, kinds=None, limit=10, score_threshold=None):
        return self.hits

    async def delete(self, record_ids):
        self.deleted.extend(record_ids)


async def embedder(texts):
    return [[0.1] * 8 for _ in texts]


# ─────────────────────────────────────────────────────────────────────────
# Scoring and exemptions
# ─────────────────────────────────────────────────────────────────────────

def test_effective_score_is_importance_for_non_decaying_kinds():
    old_identity = make(store_age_days=3650, kind=MemoryKind.IDENTITY, importance=0.9)
    assert effective_score(old_identity) == pytest.approx(0.9)


def test_effective_score_falls_with_age():
    fresh = make(store_age_days=0, kind=MemoryKind.EPISODIC, importance=0.8)
    stale = make(store_age_days=365, kind=MemoryKind.EPISODIC, importance=0.8)
    assert effective_score(stale) < effective_score(fresh)


@pytest.mark.parametrize("kind", [
    MemoryKind.IDENTITY, MemoryKind.PREFERENCE, MemoryKind.GOAL
])
def test_always_injected_kinds_are_decay_exempt(kind):
    """Forgetting the user's name to save space is never the right trade."""
    assert is_decay_exempt(make(kind=kind)) is True


def test_pinned_records_are_decay_exempt():
    assert is_decay_exempt(make(pinned=True)) is True


def test_ordinary_records_are_not_exempt():
    assert is_decay_exempt(make(kind=MemoryKind.EPISODIC)) is False


# ─────────────────────────────────────────────────────────────────────────
# Decay
# ─────────────────────────────────────────────────────────────────────────

async def test_stale_low_value_records_are_archived(store):
    await store.add(make(store_age_days=900, kind=MemoryKind.EPISODIC, importance=0.3))
    stats = await run_decay(store)

    assert stats.archived == 1
    assert await store.count(statuses=[RecordStatus.ACTIVE]) == 0
    assert await store.count(statuses=[RecordStatus.ARCHIVED]) == 1


async def test_archived_records_are_never_deleted(store):
    """Archived leaves retrieval; it does not destroy anything."""
    await store.add(make(store_age_days=900, kind=MemoryKind.EPISODIC, importance=0.3))
    await run_decay(store)
    assert await store.count() == 1


async def test_young_records_are_spared_regardless_of_score(store):
    """A new low-importance memory has not had a chance to prove useful."""
    await store.add(make(store_age_days=1, kind=MemoryKind.EPISODIC, importance=0.01))
    stats = await run_decay(store)
    assert stats.archived == 0


async def test_identity_survives_decay_no_matter_how_old(store):
    await store.add(make(store_age_days=3650, kind=MemoryKind.IDENTITY, importance=0.9))
    stats = await run_decay(store)
    assert stats.archived == 0
    assert await store.count(statuses=[RecordStatus.ACTIVE]) == 1


async def test_a_pinned_record_survives_decay(store):
    await store.add(make(
        store_age_days=3650, kind=MemoryKind.EPISODIC, importance=0.01, pinned=True
    ))
    stats = await run_decay(store)
    assert stats.archived == 0


async def test_important_old_records_survive(store):
    """Decay is importance × recency, not age alone."""
    await store.add(make(store_age_days=200, kind=MemoryKind.SEMANTIC, importance=1.0))
    stats = await run_decay(store)
    assert stats.archived == 0


async def test_decay_is_idempotent(store):
    await store.add(make(store_age_days=900, kind=MemoryKind.EPISODIC, importance=0.3))
    first = await run_decay(store)
    second = await run_decay(store)
    assert first.archived == 1
    assert second.archived == 0  # nothing active left to archive


async def run_decay(store):
    maintenance = MemoryMaintenance(record_store=store, vector_store=FakeVectors())
    from app.memory.cognition.maintenance import MaintenanceStats
    stats = MaintenanceStats()
    await maintenance.decay_and_archive(stats)
    return stats


# ─────────────────────────────────────────────────────────────────────────
# Conflict reconciliation
# ─────────────────────────────────────────────────────────────────────────

async def test_duplicate_dedup_keys_are_reconciled(store):
    """
    Should never happen — the writer supersedes on conflict — but two
    concurrent writers can race past that, and this is what notices.
    """
    for value in ("student", "engineer"):
        await store.add(make(
            content=f"The user's role is {value}.",
            kind=MemoryKind.IDENTITY,
            dedup_key="profile:role",
        ))
    assert await store.count(statuses=[RecordStatus.ACTIVE]) == 2

    stats = await run_conflicts(store)

    assert stats.conflicts_resolved == 1
    assert await store.count(statuses=[RecordStatus.ACTIVE]) == 1


async def test_the_newest_conflicting_record_survives(store):
    import asyncio
    await store.add(make(content="The user's role is student.", dedup_key="profile:role"))
    await asyncio.sleep(0.01)
    await store.add(make(content="The user's role is engineer.", dedup_key="profile:role"))

    await run_conflicts(store)

    survivors = await store.find_by_dedup_key("vansh", "profile:role")
    assert len(survivors) == 1
    assert "engineer" in survivors[0].content


async def test_reconciliation_leaves_healthy_keys_alone(store):
    await store.add(make(content="The user's role is engineer.", dedup_key="profile:role"))
    await store.add(make(content="The user's name is Vansh.", dedup_key="profile:name"))

    stats = await run_conflicts(store)

    assert stats.conflicts_resolved == 0
    assert await store.count(statuses=[RecordStatus.ACTIVE]) == 2


async def run_conflicts(store):
    maintenance = MemoryMaintenance(record_store=store, vector_store=FakeVectors())
    from app.memory.cognition.maintenance import MaintenanceStats
    stats = MaintenanceStats()
    await maintenance.reconcile_conflicts(stats)
    return stats


# ─────────────────────────────────────────────────────────────────────────
# Semantic de-duplication
# ─────────────────────────────────────────────────────────────────────────

def test_episodes_are_not_dedupable():
    """
    Two similar-sounding events are usually two different events; merging them
    would fabricate a history that never happened.
    """
    assert MemoryKind.EPISODIC not in DEDUPABLE_KINDS
    assert MemoryKind.SEMANTIC in DEDUPABLE_KINDS


async def test_near_duplicates_are_merged(store):
    keeper = await store.add(make(
        content="The user knows Python and FastAPI.", confidence=0.9
    ))
    other = await store.add(make(
        content="The user is proficient with Python and FastAPI.", confidence=0.6
    ))

    vectors = FakeVectors(hits=[VectorHit(record_id=other.id, score=0.97, payload={})])
    stats = await run_dedup(store, vectors)

    assert stats.duplicates_merged == 1
    assert await store.count("vansh", statuses=[RecordStatus.ACTIVE]) == 1


async def test_a_merge_keeps_the_absorbed_record_in_provenance(store):
    """"Why do you know this?" must still resolve after a merge."""
    keeper = await store.add(make(content="The user knows Python.", confidence=0.9))
    other = await store.add(make(content="The user uses Python daily.", confidence=0.6))

    vectors = FakeVectors(hits=[VectorHit(record_id=other.id, score=0.97, payload={})])
    await run_dedup(store, vectors)

    survivors = await store.list("vansh", statuses=[RecordStatus.ACTIVE])
    assert other.id in survivors[0].derived_from


async def test_a_merge_keeps_the_stronger_evidence(store):
    await store.add(make(content="The user knows Python.", confidence=0.5, importance=0.4))
    other = await store.add(make(
        content="The user uses Python daily.", confidence=0.95, importance=0.8
    ))

    vectors = FakeVectors(hits=[VectorHit(record_id=other.id, score=0.97, payload={})])
    await run_dedup(store, vectors)

    survivor = (await store.list("vansh", statuses=[RecordStatus.ACTIVE]))[0]
    assert survivor.confidence == 0.95
    assert survivor.importance == 0.8


async def test_similarity_below_the_threshold_is_left_alone(store):
    await store.add(make(content="The user knows Python."))
    other = await store.add(make(content="The user enjoys hiking on weekends."))

    vectors = FakeVectors(hits=[VectorHit(record_id=other.id, score=0.55, payload={})])
    stats = await run_dedup(store, vectors)

    assert stats.duplicates_merged == 0
    assert await store.count("vansh", statuses=[RecordStatus.ACTIVE]) == 2


async def test_dedup_is_skipped_without_an_embedder(store):
    """Degrades to a no-op rather than failing when embedding is unavailable."""
    await store.add(make())
    maintenance = MemoryMaintenance(
        record_store=store, vector_store=FakeVectors(), embedder=None
    )
    from app.memory.cognition.maintenance import MaintenanceStats
    stats = MaintenanceStats()
    await maintenance.sweep_duplicates(stats)
    assert stats.duplicates_merged == 0


async def run_dedup(store, vectors):
    maintenance = MemoryMaintenance(
        record_store=store, vector_store=vectors, embedder=embedder
    )
    from app.memory.cognition.maintenance import MaintenanceStats
    stats = MaintenanceStats()
    await maintenance.sweep_duplicates(stats)
    return stats


# ─────────────────────────────────────────────────────────────────────────
# Forgetting
# ─────────────────────────────────────────────────────────────────────────

async def test_forgetting_cascades_to_derived_memories(store):
    """
    A distillation must not outlive the fact it came from — that would be a
    deletion that did not delete.
    """
    source = await store.add(make(content="The user mentioned working with Django."))
    derived = await store.add(make(
        content="The user is an experienced web developer.",
        derived_from=[source.id],
    ))

    vectors = FakeVectors()
    maintenance = MemoryMaintenance(record_store=store, vector_store=vectors)
    deleted = await maintenance.forget_record("vansh", source.id)

    assert deleted == 2
    assert await store.count() == 0
    assert set(vectors.deleted) == {source.id, derived.id}


async def test_forgetting_follows_multi_level_chains(store):
    a = await store.add(make(content="Fact A about the user."))
    b = await store.add(make(content="Summary of fact A.", derived_from=[a.id]))
    c = await store.add(make(content="Summary of the summary.", derived_from=[b.id]))

    maintenance = MemoryMaintenance(record_store=store, vector_store=FakeVectors())
    assert await maintenance.forget_record("vansh", a.id) == 3
    assert await store.count() == 0


async def test_forgetting_leaves_unrelated_records_untouched(store):
    source = await store.add(make(content="Fact to be deleted."))
    unrelated = await store.add(make(content="An entirely unrelated fact."))

    maintenance = MemoryMaintenance(record_store=store, vector_store=FakeVectors())
    await maintenance.forget_record("vansh", source.id)

    assert await store.count() == 1
    assert (await store.get("vansh", unrelated.id)) is not None


async def test_forgetting_an_owner_removes_every_status(store):
    await store.add(make(content="An active memory."))
    archived = await store.add(make(content="An archived memory."))
    await store.set_status([archived.id], RecordStatus.ARCHIVED)

    maintenance = MemoryMaintenance(record_store=store, vector_store=FakeVectors())
    assert await maintenance.forget_owner("vansh") == 2
    assert await store.count() == 0


async def test_a_vector_deletion_failure_still_removes_the_records(store):
    """A stale vector is bad; a record the user asked to erase surviving is worse."""
    class BrokenVectors:
        async def delete(self, record_ids):
            raise RuntimeError("qdrant down")

    record = await store.add(make())
    maintenance = MemoryMaintenance(record_store=store, vector_store=BrokenVectors())
    assert await maintenance.forget_record("vansh", record.id) == 1
    assert await store.count() == 0


# ─────────────────────────────────────────────────────────────────────────
# Guest collection
# ─────────────────────────────────────────────────────────────────────────

async def test_abandoned_guest_partitions_are_purged(store, monkeypatch):
    monkeypatch.setattr(settings, "memory_guest_retention_days", 30)
    await store.add(make(owner_id="guest-abc123", store_age_days=90))

    stats = await run_guest_gc(store)

    assert stats.guests_purged == 1
    assert await store.count("guest-abc123") == 0


async def test_recent_guests_are_kept(store, monkeypatch):
    monkeypatch.setattr(settings, "memory_guest_retention_days", 30)
    await store.add(make(owner_id="guest-abc123", store_age_days=2))

    stats = await run_guest_gc(store)

    assert stats.guests_purged == 0
    assert await store.count("guest-abc123") == 1


async def test_the_owner_is_never_collected(store, monkeypatch):
    """Retention applies to anonymous sessions, never to the account owner."""
    monkeypatch.setattr(settings, "memory_guest_retention_days", 30)
    await store.add(make(owner_id="vansh", store_age_days=3650))

    stats = await run_guest_gc(store)

    assert stats.guests_purged == 0
    assert await store.count("vansh") == 1


class FakeQdrant:
    """Records which legacy collections an erasure actually cleared."""

    def __init__(self):
        self.cleared: list = []

    async def delete_by_filter(self, collection_name, filter_conditions):
        self.cleared.append((collection_name, filter_conditions.get("user_id")))


class FakeSessionMaker:
    """Records the Postgres tables an erasure deleted from."""

    def __init__(self):
        self.statements: list = []

    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement):
        self.statements.append(statement)

        class _Result:
            rowcount = 1

        return _Result()

    async def commit(self):
        return None


def build_erasure(maintenance):
    """A cross-store eraser wired entirely to doubles — no network, no DB."""
    from app.memory.erasure import MemoryErasure
    from app.memory.memory_cache import MemoryCache

    return MemoryErasure(
        maintenance=maintenance,
        cache=MemoryCache(),
        session_maker=FakeSessionMaker(),
        qdrant=FakeQdrant(),
    )


async def run_guest_gc(store):
    """
    Guest collection over doubles for every store it now touches.

    Collection stopped being a `memory_records`-only operation once it was
    found to leave chat history, profile facts and four Qdrant collections
    behind. The doubles are injected rather than reached for so this stays a
    unit test — an eraser that quietly required a live Postgres would make the
    one path where silent partial failure is unacceptable the least tested.
    """
    maintenance = MemoryMaintenance(record_store=store, vector_store=FakeVectors())
    maintenance._erasure = build_erasure(maintenance)
    from app.memory.cognition.maintenance import MaintenanceStats
    stats = MaintenanceStats()
    await maintenance.collect_abandoned_guests(stats)
    return stats


async def test_guest_collection_clears_every_store_not_just_records(store, monkeypatch):
    """
    The purge used to cover one store out of nine.

    `memory_records` is not the store that answers questions — until the read
    cutover the legacy tables are authoritative — so purging it alone let
    abandoned guest data accumulate indefinitely while the sweep reported
    success. This pins that every registered store is actually visited.
    """
    from app.memory.erasure import _POSTGRES_TABLES, _QDRANT_COLLECTIONS

    monkeypatch.setattr(settings, "memory_guest_retention_days", 30)
    await store.add(make(owner_id="guest-abc123", store_age_days=90))

    maintenance = MemoryMaintenance(record_store=store, vector_store=FakeVectors())
    erasure = build_erasure(maintenance)
    maintenance._erasure = erasure

    from app.memory.cognition.maintenance import MaintenanceStats
    stats = MaintenanceStats()
    await maintenance.collect_abandoned_guests(stats)

    assert stats.guests_purged == 1
    assert await store.count("guest-abc123") == 0
    assert [name for name, _ in erasure._qdrant.cleared] == list(_QDRANT_COLLECTIONS)
    assert all(owner == "guest-abc123" for _, owner in erasure._qdrant.cleared)
    assert len(erasure._session_maker.statements) == len(_POSTGRES_TABLES)
    assert stats.failed_jobs == []


# ─────────────────────────────────────────────────────────────────────────
# Cycle isolation
# ─────────────────────────────────────────────────────────────────────────

async def test_one_failing_job_does_not_stop_the_others(store):
    await store.add(make(store_age_days=900, kind=MemoryKind.EPISODIC, importance=0.3))

    maintenance = MemoryMaintenance(record_store=store, vector_store=FakeVectors())

    async def broken(stats):
        raise RuntimeError("conflict scan exploded")

    maintenance.reconcile_conflicts = broken
    stats = await maintenance.run_once()

    assert "conflicts" in stats.failed_jobs
    assert stats.archived == 1  # decay still ran


async def test_the_cycle_never_raises(store):
    class BrokenStore:
        async def duplicate_dedup_keys(self, limit=100):
            raise RuntimeError("database gone")
        async def iter_active(self, **kwargs):
            raise RuntimeError("database gone")
        async def owner_activity(self):
            raise RuntimeError("database gone")

    maintenance = MemoryMaintenance(
        record_store=BrokenStore(), vector_store=FakeVectors()
    )
    stats = await maintenance.run_once()
    assert len(stats.failed_jobs) >= 3
    assert stats.archived == 0
