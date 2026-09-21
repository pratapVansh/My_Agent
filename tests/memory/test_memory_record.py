"""
MemoryRecord invariants (Phase 1).

The record is the centre of the redesign — every kind of memory is one of
these — so its invariants are load-bearing for everything built on top:
dedup depends on content hashing, versioning depends on supersession, and
prompt safety depends on `is_injectable`.

See docs/MEMORY_ARCHITECTURE.md §3.3.
"""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.memory.kinds import (
    ALWAYS_INJECTED_KINDS,
    KIND_HALF_LIFE_DAYS,
    EmbeddingStatus,
    MemoryKind,
    RecordStatus,
    Sensitivity,
    SourceType,
    Visibility,
)
from app.memory.record import (
    MemoryRecord,
    compute_content_hash,
    normalize_content,
)


def make(**overrides) -> MemoryRecord:
    base = dict(owner_id="vansh", kind=MemoryKind.SEMANTIC, content="The user knows Python.")
    base.update(overrides)
    return MemoryRecord(**base)


# ─────────────────────────────────────────────────────────────────────────
# Construction and validation
# ─────────────────────────────────────────────────────────────────────────

def test_content_is_whitespace_normalised():
    assert make(content="  The   user\n\nknows  Python. ").content == "The user knows Python."


def test_empty_content_is_rejected():
    with pytest.raises(ValueError, match="content cannot be empty"):
        make(content="   \n  ")


def test_empty_owner_is_rejected():
    with pytest.raises(ValueError, match="owner_id is required"):
        make(owner_id="  ")


def test_string_values_are_coerced_to_enums():
    """Rows from the database and JSON payloads arrive as plain strings."""
    record = make(kind="episodic", source_type="chat", visibility="public",
                  sensitivity="sensitive", status="archived", embedding_status="ready")
    assert record.kind is MemoryKind.EPISODIC
    assert record.source_type is SourceType.CHAT
    assert record.visibility is Visibility.PUBLIC
    assert record.sensitivity is Sensitivity.SENSITIVE
    assert record.status is RecordStatus.ARCHIVED
    assert record.embedding_status is EmbeddingStatus.READY


@pytest.mark.parametrize("given,expected", [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (7.3, 1.0)])
def test_salience_is_clamped_to_unit_range(given, expected):
    assert make(importance=given).importance == expected
    assert make(confidence=given).confidence == expected


def test_defaults_are_conservative():
    """A record with no governance stated must not be publicly readable."""
    record = make()
    assert record.visibility is Visibility.PRIVATE
    assert record.sensitivity is Sensitivity.NORMAL
    assert record.status is RecordStatus.ACTIVE
    assert record.embedding_status is EmbeddingStatus.PENDING
    assert record.version == 1
    assert record.valid_to is None


def test_timestamps_are_timezone_aware():
    """Naive datetimes are a bug waiting for a deployment in another region."""
    record = make()
    assert record.created_at.tzinfo is not None
    assert record.valid_from.tzinfo is not None


# ─────────────────────────────────────────────────────────────────────────
# Content hashing / dedup
# ─────────────────────────────────────────────────────────────────────────

def test_hash_ignores_whitespace_and_case():
    assert (
        compute_content_hash(MemoryKind.SEMANTIC, "The user knows Python.")
        == compute_content_hash(MemoryKind.SEMANTIC, "  the USER   knows python.  ")
    )


def test_hash_is_scoped_by_kind():
    """
    The same sentence as a semantic fact and as a document chunk are different
    memories with different retrieval behaviour; they must not collide.
    """
    assert (
        compute_content_hash(MemoryKind.SEMANTIC, "Built My_Agent.")
        != compute_content_hash(MemoryKind.DOCUMENT, "Built My_Agent.")
    )


def test_hash_differs_for_different_content():
    assert (
        compute_content_hash(MemoryKind.SEMANTIC, "The user knows Python.")
        != compute_content_hash(MemoryKind.SEMANTIC, "The user knows Rust.")
    )


def test_hash_is_computed_when_not_supplied():
    record = make()
    assert record.content_hash == compute_content_hash(record.kind, record.content)


def test_supplied_hash_is_preserved():
    """Rows loaded from the database must round-trip their stored hash."""
    assert make(content_hash="deadbeef").content_hash == "deadbeef"


def test_normalize_content_handles_none():
    assert normalize_content(None) == ""


# ─────────────────────────────────────────────────────────────────────────
# Predicates
# ─────────────────────────────────────────────────────────────────────────

def test_active_record_with_open_validity_is_current():
    assert make().is_currently_valid is True


def test_closed_validity_is_not_current():
    assert make(valid_to=datetime.now(timezone.utc)).is_currently_valid is False


def test_archived_record_is_not_active():
    assert make(status=RecordStatus.ARCHIVED).is_active is False


def test_secret_records_are_never_injectable():
    """
    Checked on the record rather than at each call site, so no future retrieval
    path can forget to.
    """
    assert make(sensitivity=Sensitivity.SECRET).is_injectable is False
    assert make(sensitivity=Sensitivity.SENSITIVE).is_injectable is True
    assert make(sensitivity=Sensitivity.NORMAL).is_injectable is True


def test_superseded_records_are_not_injectable():
    assert make(status=RecordStatus.SUPERSEDED).is_injectable is False


# ─────────────────────────────────────────────────────────────────────────
# Versioning
# ─────────────────────────────────────────────────────────────────────────

def test_superseding_links_and_increments_version():
    original = make(content="The user's role is student.")
    nxt = original.superseding(content="The user's role is engineer.")

    assert nxt.supersedes_id == original.id
    assert nxt.id != original.id
    assert nxt.version == original.version + 1
    assert nxt.status is RecordStatus.ACTIVE
    assert nxt.valid_to is None


def test_superseding_recomputes_the_content_hash():
    """A stale hash would make the new version collide with the old one."""
    original = make(content="The user's role is student.")
    nxt = original.superseding(content="The user's role is engineer.")
    assert nxt.content_hash != original.content_hash
    assert nxt.content_hash == compute_content_hash(nxt.kind, nxt.content)


def test_superseding_resets_embedding_status():
    """New content needs a new vector; inheriting READY would leave a stale one."""
    original = make(embedding_status=EmbeddingStatus.READY)
    assert original.superseding(content="Something else.").embedding_status is EmbeddingStatus.PENDING


def test_superseded_by_closes_the_old_record_without_mutating_it():
    original = make()
    replacement = original.superseding(content="Newer.")
    closed = original.superseded_by(replacement)

    assert closed.status is RecordStatus.SUPERSEDED
    assert closed.valid_to is not None
    assert closed.id == original.id
    # The caller writes both rows in one transaction; mutating in place would
    # make a half-applied supersession unrecoverable.
    assert original.status is RecordStatus.ACTIVE
    assert original.valid_to is None


def test_superseding_preserves_owner_and_kind():
    original = make(owner_id="vansh", kind=MemoryKind.IDENTITY)
    nxt = original.superseding(content="Newer.")
    assert nxt.owner_id == "vansh"
    assert nxt.kind is MemoryKind.IDENTITY


def test_accessed_increments_the_frequency_signal():
    record = make()
    hit = record.accessed()
    assert hit.access_count == 1
    assert hit.last_accessed_at is not None
    assert record.access_count == 0  # original untouched


# ─────────────────────────────────────────────────────────────────────────
# Taxonomy
# ─────────────────────────────────────────────────────────────────────────

def test_every_kind_has_a_half_life_entry():
    """A kind with no decay policy would silently never be archived."""
    for kind in MemoryKind:
        assert kind in KIND_HALF_LIFE_DAYS


def test_identity_and_preferences_never_decay():
    """Forgetting the user's name for lack of recent mention is a bug."""
    assert KIND_HALF_LIFE_DAYS[MemoryKind.IDENTITY] is None
    assert KIND_HALF_LIFE_DAYS[MemoryKind.PREFERENCE] is None


def test_tasks_decay_faster_than_semantic_facts():
    assert KIND_HALF_LIFE_DAYS[MemoryKind.TASK] < KIND_HALF_LIFE_DAYS[MemoryKind.SEMANTIC]


def test_always_injected_kinds_are_the_guaranteed_tier():
    assert ALWAYS_INJECTED_KINDS == {
        MemoryKind.IDENTITY, MemoryKind.PREFERENCE, MemoryKind.GOAL
    }
