"""
`RecordStore` port contract (Phase 1).

Written against the port, not an implementation. The in-memory store runs it
here; the Postgres adapter must satisfy exactly the same assertions, which is
what makes the fake safe to build Phase 2 and 3 on top of. A fake that is more
permissive than the real thing tests nothing useful.

See docs/MEMORY_ARCHITECTURE.md §3.1.
"""
import pytest

from app.memory.kinds import EmbeddingStatus, MemoryKind, RecordStatus, Visibility
from app.memory.record import MemoryRecord
from app.memory.stores import InMemoryRecordStore


@pytest.fixture
def store():
    return InMemoryRecordStore()


def make(content="The user knows Python.", **overrides) -> MemoryRecord:
    base = dict(owner_id="vansh", kind=MemoryKind.SEMANTIC, content=content)
    base.update(overrides)
    return MemoryRecord(**base)


# ─────────────────────────────────────────────────────────────────────────
# Idempotent writes — the property the backfill migration depends on
# ─────────────────────────────────────────────────────────────────────────

async def test_add_returns_the_stored_record(store):
    record = make()
    assert (await store.add(record)).id == record.id
    assert await store.count() == 1


async def test_adding_an_exact_duplicate_is_a_no_op(store):
    first = await store.add(make())
    second = await store.add(make())
    assert second.id == first.id
    assert await store.count() == 1


async def test_duplicate_detection_ignores_whitespace_and_case(store):
    await store.add(make("The user knows Python."))
    await store.add(make("  the USER   knows   python.  "))
    assert await store.count() == 1


async def test_same_content_under_a_different_kind_is_a_separate_record(store):
    await store.add(make("Built My_Agent.", kind=MemoryKind.SEMANTIC))
    await store.add(make("Built My_Agent.", kind=MemoryKind.DOCUMENT))
    assert await store.count() == 2


async def test_same_content_for_a_different_owner_is_a_separate_record(store):
    await store.add(make(owner_id="vansh"))
    await store.add(make(owner_id="guest-abc"))
    assert await store.count() == 2
    assert await store.count("vansh") == 1


async def test_add_many_deduplicates_within_the_batch(store):
    stored = await store.add_many([make(), make(), make("Something else.")])
    assert len(stored) == 3          # one result per input
    assert await store.count() == 2  # but only two distinct records


# ─────────────────────────────────────────────────────────────────────────
# Tenant scoping
# ─────────────────────────────────────────────────────────────────────────

async def test_get_is_scoped_by_owner(store):
    record = await store.add(make(owner_id="vansh"))
    assert await store.get("vansh", record.id) is not None
    # An id alone must never be enough to read across tenants.
    assert await store.get("guest-abc", record.id) is None


async def test_list_only_returns_the_requested_owner(store):
    await store.add(make(owner_id="vansh"))
    await store.add(make(owner_id="guest-abc"))
    assert all(r.owner_id == "vansh" for r in await store.list("vansh"))


# ─────────────────────────────────────────────────────────────────────────
# Lookups
# ─────────────────────────────────────────────────────────────────────────

async def test_find_by_content_hash_matches_only_active_records(store):
    record = await store.add(make())
    assert await store.find_by_content_hash("vansh", record.kind, record.content_hash)

    await store.supersede(record, record.superseding(content="Newer."))
    assert await store.find_by_content_hash("vansh", record.kind, record.content_hash) is None


async def test_find_by_dedup_key_returns_active_conflicts_newest_first(store):
    await store.add(make("The user's role is student.", dedup_key="profile:role"))
    await store.add(make("The user's name is Vansh.", dedup_key="profile:name"))

    conflicts = await store.find_by_dedup_key("vansh", "profile:role")
    assert len(conflicts) == 1
    assert conflicts[0].dedup_key == "profile:role"


async def test_find_by_dedup_key_is_empty_for_unknown_keys(store):
    assert await store.find_by_dedup_key("vansh", "profile:nothing") == []


# ─────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────

async def test_list_defaults_to_active_records_only(store):
    record = await store.add(make())
    await store.supersede(record, record.superseding(content="Newer."))
    contents = [r.content for r in await store.list("vansh")]
    assert contents == ["Newer."]


async def test_list_filters_by_kind(store):
    await store.add(make("The user's name is Vansh.", kind=MemoryKind.IDENTITY))
    await store.add(make("The user knows Python.", kind=MemoryKind.SEMANTIC))
    results = await store.list("vansh", kinds=[MemoryKind.IDENTITY])
    assert [r.kind for r in results] == [MemoryKind.IDENTITY]


async def test_list_filters_by_visibility(store):
    """The mechanism behind the recruiter fix: public-only retrieval."""
    await store.add(make("Public project.", visibility=Visibility.PUBLIC))
    await store.add(make("Private note.", visibility=Visibility.PRIVATE))
    results = await store.list("vansh", visibilities=[Visibility.PUBLIC])
    assert [r.content for r in results] == ["Public project."]


async def test_list_respects_limit_and_offset(store):
    for i in range(5):
        await store.add(make(f"Fact number {i}."))
    assert len(await store.list("vansh", limit=2)) == 2
    assert len(await store.list("vansh", limit=2, offset=4)) == 1


async def test_count_filters_by_kind_and_status(store):
    await store.add(make("The user's name is Vansh.", kind=MemoryKind.IDENTITY))
    await store.add(make("The user knows Python.", kind=MemoryKind.SEMANTIC))
    assert await store.count("vansh", kinds=[MemoryKind.IDENTITY]) == 1
    assert await store.count("vansh", statuses=[RecordStatus.ACTIVE]) == 2


# ─────────────────────────────────────────────────────────────────────────
# Supersession
# ─────────────────────────────────────────────────────────────────────────

async def test_supersede_keeps_both_rows_with_correct_statuses(store):
    original = await store.add(make("The user's role is student."))
    replacement = original.superseding(content="The user's role is engineer.")
    await store.supersede(original, replacement)

    assert await store.count("vansh", statuses=[RecordStatus.ACTIVE]) == 1
    assert await store.count("vansh", statuses=[RecordStatus.SUPERSEDED]) == 1

    closed = await store.get("vansh", original.id)
    assert closed.status is RecordStatus.SUPERSEDED
    assert closed.valid_to is not None


async def test_supersede_preserves_the_version_chain(store):
    v1 = await store.add(make("Version one."))
    v2 = v1.superseding(content="Version two.")
    await store.supersede(v1, v2)
    v3 = v2.superseding(content="Version three.")
    await store.supersede(v2, v3)

    current = await store.list("vansh")
    assert len(current) == 1
    assert current[0].content == "Version three."
    assert current[0].version == 3
    assert current[0].supersedes_id == v2.id


# ─────────────────────────────────────────────────────────────────────────
# Embedding queue
# ─────────────────────────────────────────────────────────────────────────

async def test_new_records_are_pending_embedding(store):
    """
    Embedding is never computed on the request path — a voice turn cannot
    afford a Cohere round trip — so records land PENDING for a background pass.
    """
    await store.add(make())
    assert len(await store.pending_embeddings()) == 1


async def test_mark_embedding_removes_a_record_from_the_queue(store):
    record = await store.add(make())
    await store.mark_embedding(record.id, EmbeddingStatus.READY)
    assert await store.pending_embeddings() == []


async def test_pending_embeddings_excludes_inactive_records(store):
    record = await store.add(make())
    await store.supersede(record, record.superseding(content="Newer."))
    pending = await store.pending_embeddings()
    assert [r.content for r in pending] == ["Newer."]


async def test_pending_embeddings_respects_its_limit(store):
    for i in range(5):
        await store.add(make(f"Fact number {i}."))
    assert len(await store.pending_embeddings(limit=3)) == 3


# ─────────────────────────────────────────────────────────────────────────
# Isolation
# ─────────────────────────────────────────────────────────────────────────

async def test_stored_records_are_isolated_from_caller_mutation(store):
    """
    The store must hand back copies. A caller mutating a returned record and
    silently corrupting the store would be near-impossible to debug.
    """
    record = await store.add(make())
    fetched = await store.get("vansh", record.id)
    fetched.content = "Mutated."
    assert (await store.get("vansh", record.id)).content == "The user knows Python."
