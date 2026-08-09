"""
Deletion that actually deletes, and a cache that cannot outlive it.

Two defects with one consequence: the assistant kept answering from data the
user had removed.

**Erasure covered one store out of nine.** `MemoryMaintenance.forget_owner`
deleted `memory_records` and its vectors. Until the Phase 6 read cutover the
*legacy* stores are the ones that answer questions — `user_profile`,
`chat_history`, `episodic_memory`, `tool_memory`, conversations and turns, plus
four Qdrant collections — and none of them were touched. A user who erased their
memory would have watched the assistant carry on exactly as before. The same
gap made guest collection a no-op: every recruiter visit's partition survived
the sweep that exists to collect it.

**The retrieval cache was never invalidated.** Nothing outside `memory_cache`
called `invalidate`. A deleted memory, a corrected name, a re-uploaded résumé —
all kept being served for the full five-minute TTL. On the deletion path that is
not staleness, it is the deletion silently not taking effect.

The tests below assert the properties that make erasure trustworthy: every
registered store is visited, partial failure is reported rather than swallowed,
and no write leaves a stale entry behind.
"""
import pytest

from app.memory.erasure import (
    _POSTGRES_TABLES,
    _QDRANT_COLLECTIONS,
    ErasureReport,
    MemoryErasure,
    StoreResult,
)
from app.memory.memory_cache import MemoryCache


# ─────────────────────────────────────────────────────────────────────────
# Doubles
# ─────────────────────────────────────────────────────────────────────────

class FakeMaintenance:
    def __init__(self, deleted=3, explode=False):
        self.deleted = deleted
        self.explode = explode
        self.calls: list = []

    async def forget_owner(self, owner_id):
        self.calls.append(owner_id)
        if self.explode:
            raise RuntimeError("postgres gone")
        return self.deleted


class FakeQdrant:
    def __init__(self, explode_on=()):
        self.cleared: list = []
        self.explode_on = set(explode_on)

    async def delete_by_filter(self, collection_name, filter_conditions):
        if collection_name in self.explode_on:
            raise RuntimeError(f"qdrant unreachable for {collection_name}")
        self.cleared.append((collection_name, filter_conditions.get("user_id")))


class FakeSession:
    def __init__(self, owner, explode=False):
        self.owner = owner
        self.explode = explode

    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement):
        if self.explode:
            raise RuntimeError("connection reset")
        self.owner.statements.append(str(statement))

        class _Result:
            rowcount = 2

        return _Result()

    async def commit(self):
        self.owner.commits += 1


class FakeSessionMaker:
    def __init__(self, explode=False):
        self.statements: list = []
        self.commits = 0
        self.explode = explode

    def __call__(self):
        return FakeSession(self, explode=self.explode)


def build(**overrides):
    cache = overrides.pop("cache", MemoryCache())
    return MemoryErasure(
        maintenance=overrides.pop("maintenance", FakeMaintenance()),
        cache=cache,
        session_maker=overrides.pop("session_maker", FakeSessionMaker()),
        qdrant=overrides.pop("qdrant", FakeQdrant()),
    )


# ─────────────────────────────────────────────────────────────────────────
# Every store is visited
# ─────────────────────────────────────────────────────────────────────────

async def test_erasure_clears_every_registered_store():
    """
    The core regression. `forget_owner` covered `memory_records` alone, and
    `memory_records` is not the store that answers questions.
    """
    qdrant = FakeQdrant()
    sessions = FakeSessionMaker()
    maintenance = FakeMaintenance()
    erasure = build(maintenance=maintenance, qdrant=qdrant, session_maker=sessions)

    report = await erasure.erase_owner("guest-abc")

    assert report.complete is True
    assert maintenance.calls == ["guest-abc"]
    assert [name for name, _ in qdrant.cleared] == list(_QDRANT_COLLECTIONS)
    assert all(owner == "guest-abc" for _, owner in qdrant.cleared)
    assert len(sessions.statements) == len(_POSTGRES_TABLES)
    assert sessions.commits == len(_POSTGRES_TABLES)


async def test_the_legacy_conversational_vector_store_is_erased():
    """
    `smart_memory_chunks` holds embedded conversation turns and had no erasure
    path at all — `reset_user_memories` existed and was called from nowhere.
    """
    qdrant = FakeQdrant()
    await build(qdrant=qdrant).erase_owner("vansh")
    assert "smart_memory_chunks" in [name for name, _ in qdrant.cleared]


async def test_resume_skills_and_project_chunks_are_erased():
    """The résumé is the most personal thing in the store; it survived erasure."""
    qdrant = FakeQdrant()
    await build(qdrant=qdrant).erase_owner("vansh")
    cleared = {name for name, _ in qdrant.cleared}
    assert {"resume_chunks", "skills_chunks", "projects_chunks"} <= cleared


def test_every_live_qdrant_collection_is_registered_for_erasure():
    """
    Drift guard. The erasure list is written out by name, so a collection added
    later would be silently missed — and a store missed by erasure is a store
    that survives a right-to-erasure request. This fails the moment the two
    diverge, which is the only reliable way to keep an enumerated list honest.
    """
    from app.memory.long_term_memory_qdrant import long_term_memory_qdrant
    from app.memory.smart_memory import smart_memory

    live = set(long_term_memory_qdrant.collections.values())
    live.add(smart_memory.collection_name)

    missing = live - set(_QDRANT_COLLECTIONS)
    assert not missing, (
        f"These Qdrant collections hold user data but are not erased: {missing}. "
        f"Add them to _QDRANT_COLLECTIONS in app/memory/erasure.py."
    )


def test_every_owner_keyed_memory_table_is_registered_for_erasure():
    """The same guard for Postgres, driven off the ORM rather than a hand list."""
    from app.memory import models as memory_models

    registered = {name for name, _ in _POSTGRES_TABLES}
    # `MemoryRecordORM` is erased through the cascading record deleter, not by
    # a bulk table delete, so it is expected to be absent from this list.
    expected_absent = {"MemoryRecordORM"}

    owner_keyed = set()
    for attribute in dir(memory_models):
        model = getattr(memory_models, attribute)
        columns = getattr(getattr(model, "__table__", None), "columns", None)
        if columns is None:
            continue
        names = {c.name for c in columns}
        if names & {"user_id", "owner_id"}:
            owner_keyed.add(attribute)

    missing = owner_keyed - registered - expected_absent
    assert not missing, (
        f"These tables are keyed by owner but are not erased: {missing}. "
        f"Add them to _POSTGRES_TABLES in app/memory/erasure.py."
    )


async def test_conversation_turns_are_deleted_before_their_conversations():
    """Turns reference conversations; deleting the parent first would fail."""
    tables = [name for name, _ in _POSTGRES_TABLES]
    assert tables.index("TurnORM") < tables.index("ConversationORM")


async def test_the_pending_extraction_outbox_is_erased():
    """
    Queued events carry verbatim conversation text. Leaving them would let the
    worker re-derive memories from an erased conversation minutes later.
    """
    assert "MemoryEventORM" in [name for name, _ in _POSTGRES_TABLES]


# ─────────────────────────────────────────────────────────────────────────
# Partial failure is reported, never swallowed
# ─────────────────────────────────────────────────────────────────────────

async def test_a_failing_store_makes_the_erasure_incomplete():
    """
    Reporting success while data survives is the same false statement as
    reporting NO_DATA after a failed lookup.
    """
    erasure = build(qdrant=FakeQdrant(explode_on={"resume_chunks"}))

    report = await erasure.erase_owner("vansh")

    assert report.complete is False
    assert "resume_chunks" in report.failed_stores


async def test_every_other_store_is_still_attempted_after_a_failure():
    """Stopping at the first error would leave *more* data behind, not less."""
    qdrant = FakeQdrant(explode_on={"resume_chunks"})
    sessions = FakeSessionMaker()
    erasure = build(qdrant=qdrant, session_maker=sessions)

    report = await erasure.erase_owner("vansh")

    assert not report.complete
    # The three surviving collections were still cleared.
    assert len(qdrant.cleared) == len(_QDRANT_COLLECTIONS) - 1
    assert len(sessions.statements) == len(_POSTGRES_TABLES)


async def test_a_record_store_failure_is_reported_not_hidden():
    erasure = build(maintenance=FakeMaintenance(explode=True))
    report = await erasure.erase_owner("vansh")
    assert report.complete is False
    assert "memory_records" in report.failed_stores


async def test_a_postgres_failure_names_every_affected_table():
    erasure = build(session_maker=FakeSessionMaker(explode=True))
    report = await erasure.erase_owner("vansh")
    assert report.complete is False
    assert len(report.failed_stores) == len(_POSTGRES_TABLES)


async def test_an_empty_owner_is_rejected_rather_than_erasing_everything():
    """A blank owner id must never be interpreted as "all owners"."""
    qdrant = FakeQdrant()
    maintenance = FakeMaintenance()
    report = await build(qdrant=qdrant, maintenance=maintenance).erase_owner("")

    assert report.complete is False
    assert qdrant.cleared == []
    assert maintenance.calls == []


async def test_the_report_never_leaks_memory_content():
    report = ErasureReport(owner_id="vansh", results=[StoreResult("chat_history", 12)])
    assert report.summary()["deleted"] == 12
    assert "chat_history" in str(report.summary())


async def test_a_store_that_cannot_count_reports_none_not_zero():
    """
    Qdrant clears by filter and returns no count. Printing `deleted=0` reads as
    "nothing was there" — a false statement in a report whose whole job is to be
    trusted about what survived.
    """
    report = await build().erase_owner("vansh")

    qdrant_results = [r for r in report.results if r.store in _QDRANT_COLLECTIONS]
    assert qdrant_results
    assert all(r.deleted is None for r in qdrant_results)
    assert all(r.ok for r in qdrant_results)

    # The total counts only what was actually counted, and does not crash on None.
    assert isinstance(report.deleted, int)


# ─────────────────────────────────────────────────────────────────────────
# The cache cannot outlive a deletion
# ─────────────────────────────────────────────────────────────────────────

async def test_erasure_clears_the_retrieval_cache():
    """
    A cached context assembled before the delete would keep being served for
    the full TTL — the deletion silently not taking effect for five minutes.
    """
    cache = MemoryCache()
    cache.set("vansh", {"long_term": {"resume": {"content": "secret"}}}, "cpi")
    assert cache.size() == 1

    report = await build(cache=cache).erase_owner("vansh")

    assert cache.get("vansh", "cpi") is None
    assert report.complete is True


async def test_erasure_does_not_clear_another_users_cache():
    cache = MemoryCache()
    cache.set("vansh", {"a": 1}, "q")
    cache.set("other", {"b": 2}, "q")

    await build(cache=cache).erase_owner("vansh")

    assert cache.get("vansh", "q") is None
    assert cache.get("other", "q") == {"b": 2}


# ─────────────────────────────────────────────────────────────────────────
# Cache correctness
# ─────────────────────────────────────────────────────────────────────────

def test_the_cache_hands_out_copies_not_its_own_storage():
    """
    `retrieve_context` overwrites chat history and profile facts on every hit.
    It was doing that to the cache's own dict, so each read mutated the entry
    for every subsequent reader.
    """
    cache = MemoryCache()
    cache.set("vansh", {"chat_history": ["original"]}, "q")

    first = cache.get("vansh", "q")
    first["chat_history"] = ["mutated by caller"]

    second = cache.get("vansh", "q")
    assert second["chat_history"] == ["original"]


def test_stored_context_is_snapshotted_on_the_way_in():
    """A caller mutating what it stored must not retroactively change the entry."""
    cache = MemoryCache()
    payload = {"long_term": {"skills": ["Python"]}}
    cache.set("vansh", payload, "q")

    payload["long_term"]["skills"].append("leaked")

    assert cache.get("vansh", "q")["long_term"]["skills"] == ["Python"]


def test_invalidate_reports_how_many_entries_it_dropped():
    cache = MemoryCache()
    cache.set("vansh", {"a": 1}, "one")
    cache.set("vansh", {"a": 2}, "two")
    cache.set("other", {"a": 3}, "one")

    assert cache.invalidate("vansh") == 2
    assert cache.invalidate("vansh") == 0
    assert cache.get("other", "one") == {"a": 3}


def test_entries_for_different_scopes_never_collide():
    """
    A guest reads the *owner's* public records under the owner's id. Keying on
    the user alone would let that share an entry with the owner's own
    full-visibility context — private memory served to a guest out of cache.
    """
    cache = MemoryCache()
    cache.set("vansh", {"visibility": "all"}, "who am i", scope="own")
    cache.set("vansh", {"visibility": "public"}, "who am i", scope="public")

    assert cache.get("vansh", "who am i", scope="own") == {"visibility": "all"}
    assert cache.get("vansh", "who am i", scope="public") == {"visibility": "public"}


def test_a_missing_scope_is_its_own_namespace():
    cache = MemoryCache()
    cache.set("vansh", {"scoped": False}, "q")
    assert cache.get("vansh", "q", scope="public") is None
    assert cache.get("vansh", "q") == {"scoped": False}


def test_expiry_uses_a_monotonic_clock(monkeypatch):
    """
    Wall-clock time moves backwards across NTP corrections and DST. An entry
    written just before a backward jump would never expire.
    """
    import app.memory.memory_cache as module

    clock = {"t": 1000.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["t"])

    cache = MemoryCache(ttl_seconds=300)
    cache.set("vansh", {"a": 1}, "q")
    assert cache.get("vansh", "q") == {"a": 1}

    clock["t"] += 301
    assert cache.get("vansh", "q") is None


def test_eviction_does_not_drop_the_entry_being_written():
    cache = MemoryCache(max_size=2)
    cache.set("a", {"n": 1}, "q")
    cache.set("b", {"n": 2}, "q")
    cache.set("c", {"n": 3}, "q")

    assert cache.size() == 2
    assert cache.get("c", "q") == {"n": 3}


def test_rewriting_a_key_does_not_evict_a_different_entry():
    cache = MemoryCache(max_size=2)
    cache.set("a", {"n": 1}, "q")
    cache.set("b", {"n": 2}, "q")
    cache.set("a", {"n": 9}, "q")

    assert cache.size() == 2
    assert cache.get("a", "q") == {"n": 9}
    assert cache.get("b", "q") == {"n": 2}


def test_a_crafted_user_id_cannot_collide_with_another_users_key():
    """The id is hashed into the key rather than concatenated raw."""
    cache = MemoryCache()
    cache.set("vansh", {"owner": "vansh"}, "q")
    assert cache.get("vansh:extra", "q") is None
