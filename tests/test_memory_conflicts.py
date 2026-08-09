"""
Contradiction handling on the write path, and explicit-vs-inferred standing.

The conflict resolver in `app.memory.identity` was written, tested, and never
called. Meanwhile the live write path — `MemoryWriter.upsert_with_outcome` —
resolved every contradiction the same way: any content difference superseded the
stored value. "Newest wins" was the entire policy.

That is the wrong policy in one specific, damaging direction. A profile key
holds one current value, so an inferred remark picked out of conversation
("sounds like you're at IIT now") contradicted the résumé-derived value and
replaced it. The best evidence in the store lost to the most recent noise, and
because supersession is not reversible from the outside, it lost permanently.

A second gap ran alongside it. `MemoryRecord.pinned` means "never decays, never
dropped for budget"; `is_decay_exempt` honours it; nothing ever set it. So a
value the user explicitly asked to keep aged out on exactly the same schedule as
one the extractor guessed at, and the explicit/inferred distinction existed in
the schema and nowhere in behaviour.

These tests pin both, plus the identity-classification regression that made the
user's own name decay after it moved to its own key.
"""
import pytest

from app.memory.identity import (
    CANONICAL_NAME_KEY,
    REMEMBERED_NAME_KEY,
    FactClaim,
    Resolution,
    resolve_conflict,
)
from app.memory.kinds import MemoryKind, RecordStatus
from app.memory.record import utcnow
from app.memory.sources import MemorySource
from app.memory.stores.in_memory_record_store import InMemoryRecordStore
from app.memory.writer import (
    MemoryWriter,
    WriteOutcome,
    classify_profile_key,
    is_pinned_key,
    render_profile_fact,
)


@pytest.fixture
def store():
    return InMemoryRecordStore()


@pytest.fixture
def writer(store):
    return MemoryWriter(record_store=store)


async def active_value(store, owner="vansh"):
    records = await store.list(owner, statuses=[RecordStatus.ACTIVE])
    assert len(records) == 1
    return records[0].structured["value"]


# ═════════════════════════════════════════════════════════════════════════
# The resolver is actually wired into the write path
# ═════════════════════════════════════════════════════════════════════════

async def test_an_inferred_remark_cannot_overwrite_an_explicit_fact(writer, store):
    """
    The core regression. Before the resolver was wired, this test's second
    write silently replaced the first and the original value was unrecoverable.
    """
    await writer.record_profile_fact("vansh", "college", "RGIPT", source="explicit")

    record, outcome = await writer.upsert_with_outcome(
        _fact_record("college", "Some Other College", source="inferred")
    )

    assert outcome is WriteOutcome.REJECTED
    assert await active_value(store) == "RGIPT"
    assert await store.count("vansh", statuses=[RecordStatus.SUPERSEDED]) == 0


async def test_an_explicit_correction_still_supersedes_an_inferred_value(writer, store):
    """The protection is directional — it must not freeze the store."""
    await writer.record_profile_fact("vansh", "college", "Guessed", source="inferred")

    _, outcome = await writer.upsert_with_outcome(
        _fact_record("college", "RGIPT", source="explicit")
    )

    assert outcome is WriteOutcome.SUPERSEDED
    assert await active_value(store) == "RGIPT"


async def test_a_user_correcting_their_own_explicit_fact_is_accepted(writer, store):
    """
    Equal standing: the claim being made now is the current one.

    A resolver that kept the stored value on a tie would freeze the first value
    ever written — and would do it invisibly, since both writes report success.
    """
    await writer.record_profile_fact("vansh", "role", "student")
    _, outcome = await writer.record_profile_fact_with_outcome("vansh", "role", "engineer")

    assert outcome is WriteOutcome.SUPERSEDED
    assert await active_value(store) == "engineer"


async def test_a_rejected_write_returns_the_value_that_survived(writer, store):
    """
    The caller is told what is stored, not what it submitted. Returning the
    rejected record would let a caller believe its value took effect.
    """
    await writer.record_profile_fact("vansh", "college", "RGIPT", source="explicit")
    record, outcome = await writer.upsert_with_outcome(
        _fact_record("college", "Elsewhere", source="inferred")
    )

    assert outcome is WriteOutcome.REJECTED
    assert record.structured["value"] == "RGIPT"


async def test_a_rejected_write_leaves_exactly_one_active_record(writer, store):
    """The one-slot invariant survives rejection as well as supersession."""
    await writer.record_profile_fact("vansh", "college", "RGIPT", source="explicit")
    await writer.upsert_with_outcome(_fact_record("college", "A", source="inferred"))
    await writer.upsert_with_outcome(_fact_record("college", "B", source="inferred"))

    assert await store.count("vansh", statuses=[RecordStatus.ACTIVE]) == 1


async def test_identical_values_from_a_weaker_source_are_not_rejected(writer, store):
    """Agreement is not a conflict — corroboration must not look like an attack."""
    await writer.record_profile_fact("vansh", "college", "RGIPT", source="explicit")
    _, outcome = await writer.upsert_with_outcome(
        _fact_record("college", "RGIPT", source="inferred")
    )
    assert outcome is WriteOutcome.DUPLICATE


# ═════════════════════════════════════════════════════════════════════════
# Resolver semantics
# ═════════════════════════════════════════════════════════════════════════

def _claim(value, source, *, explicit=False, key="college", at=None, confidence=1.0):
    return FactClaim(
        key=key, value=value, source=source, explicit=explicit,
        confidence=confidence, timestamp=at,
    )


def test_a_resume_outranks_a_conversational_mention():
    verdict = resolve_conflict(
        _claim("RGIPT", MemorySource.RESUME_DOCUMENT),
        _claim("Elsewhere", MemorySource.CONVERSATION_CURRENT),
    )
    assert verdict.resolution is Resolution.KEEP_EXISTING


def test_low_confidence_lowers_a_claims_standing():
    """Confidence scales trust — a hedged claim does not beat a certain one."""
    verdict = resolve_conflict(
        _claim("RGIPT", MemorySource.PROFILE_MEMORY, confidence=1.0),
        _claim("Elsewhere", MemorySource.PROFILE_MEMORY, confidence=0.2),
    )
    assert verdict.resolution is Resolution.KEEP_EXISTING


def test_an_older_claim_arriving_late_does_not_win():
    """Out-of-order writes must not let a stale value overwrite a fresh one."""
    from datetime import timedelta

    now = utcnow()
    verdict = resolve_conflict(
        _claim("current", MemorySource.PROFILE_MEMORY, at=now),
        _claim("stale", MemorySource.PROFILE_MEMORY, at=now - timedelta(days=5)),
    )
    assert verdict.resolution is Resolution.KEEP_EXISTING
    assert verdict.winner == "current"


def test_every_verdict_records_the_losing_value():
    """Nothing is destroyed without a record of what it was."""
    for existing, incoming in (
        (_claim("A", MemorySource.RESUME_DOCUMENT), _claim("B", MemorySource.CONVERSATION_CURRENT)),
        (_claim("A", MemorySource.CONVERSATION_CURRENT), _claim("B", MemorySource.RESUME_DOCUMENT)),
        (_claim("A", MemorySource.PROFILE_MEMORY), _claim("B", MemorySource.PROFILE_MEMORY)),
    ):
        assert resolve_conflict(existing, incoming).previous is not None


# ═════════════════════════════════════════════════════════════════════════
# Explicit memories are pinned; inferred ones are not
# ═════════════════════════════════════════════════════════════════════════

async def test_an_explicitly_remembered_value_is_pinned(writer):
    record = await writer.record_profile_fact(
        "vansh", REMEMBERED_NAME_KEY, "Devasi", source="explicit"
    )
    assert record.pinned is True

    from app.memory.cognition.maintenance import is_decay_exempt

    assert is_decay_exempt(record) is True


async def test_canonical_identity_is_pinned_and_never_decays(writer):
    record = await writer.record_profile_fact(
        "vansh", CANONICAL_NAME_KEY, "Vansh Pratap Singh"
    )
    assert record.pinned is True
    assert record.kind is MemoryKind.IDENTITY


async def test_an_inferred_semantic_fact_is_not_pinned(writer):
    record = await writer.record_profile_fact(
        "vansh", "favourite_editor", "vim", source="inferred", confidence=0.9
    )
    assert record.pinned is False


async def test_explicitness_is_recorded_on_the_record(writer):
    explicit = await writer.record_profile_fact("vansh", "hobby", "chess")
    inferred = await writer.record_profile_fact(
        "other", "hobby", "chess", source="inferred", confidence=0.9
    )
    assert explicit.structured["explicit"] is True
    assert inferred.structured["explicit"] is False


# ═════════════════════════════════════════════════════════════════════════
# Identity classification — the regression from renaming the key
# ═════════════════════════════════════════════════════════════════════════

def test_the_canonical_name_key_classifies_as_identity():
    """
    Moving the name to its own key silently demoted it to `semantic`, which
    costs always-injected status, the identity kind prior, and — worst — decay
    exemption. A name on a 365-day half-life is a name eventually forgotten.
    """
    assert classify_profile_key(CANONICAL_NAME_KEY) is MemoryKind.IDENTITY
    assert is_pinned_key(CANONICAL_NAME_KEY) is True


def test_a_remembered_name_is_never_classified_as_identity():
    """Identity classification would put it in the always-injected tier beside
    the real name — recreating, through the taxonomy, the confusion the separate
    keys exist to prevent."""
    assert classify_profile_key(REMEMBERED_NAME_KEY) is MemoryKind.SEMANTIC
    assert classify_profile_key("alternate_name") is MemoryKind.SEMANTIC


def test_a_remembered_name_reads_as_what_it_is_when_injected():
    """
    `content` is what gets embedded and injected. The generic template produced
    "The user's remembered name is Devasi", which a model reads as a statement
    about the user's name. Separate keys stop the wrong value being *retrieved*;
    only the wording stops the right value being *misread* in the prompt.
    """
    rendered = render_profile_fact(REMEMBERED_NAME_KEY, "Devasi")
    assert "Devasi" in rendered
    assert "not the user's own name" in rendered
    assert "The user's remembered name is" not in rendered


def test_the_canonical_name_renders_as_a_plain_identity_statement():
    assert render_profile_fact(CANONICAL_NAME_KEY, "Vansh") == "The user's name is Vansh."


def test_a_remembered_non_name_value_is_still_marked_as_remembered():
    rendered = render_profile_fact("remembered_gym_time", "6am")
    assert "asked the assistant to remember" in rendered
    assert "6am" in rendered


# ── helper ───────────────────────────────────────────────────────────────

def _fact_record(key, value, *, source="explicit", owner="vansh"):
    """Build the record `record_profile_fact` would build, without writing it."""
    from app.memory.kinds import SourceType, Visibility
    from app.memory.record import MemoryRecord
    from app.memory.writer import BASE_IMPORTANCE

    kind = classify_profile_key(key)
    return MemoryRecord(
        owner_id=owner,
        kind=kind,
        content=render_profile_fact(key, value),
        structured={"key": key, "value": value, "source": source,
                    "explicit": source == "explicit"},
        importance=BASE_IMPORTANCE[kind],
        source_type=SourceType.CHAT if source == "inferred" else SourceType.SYSTEM,
        source_ref=f"user_profile:{key}",
        dedup_key=f"profile:{key}",
        visibility=Visibility.PRIVATE,
    )
