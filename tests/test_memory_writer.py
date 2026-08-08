"""
Dual-write translation (Phase 1).

The writer decides two things that are expensive to get wrong:

* **Which kind a fact becomes**, which decides whether it is injected into
  *every* prompt (identity, preference, goal) or only retrieved on relevance.
  Misclassifying downward means a stated preference is silently ignored;
  misclassifying upward burns guaranteed context budget on every turn.

* **What the content sentence says**, which is simultaneously what gets
  embedded and what gets injected. "concise" is useless out of context.

See docs/MEMORY_ARCHITECTURE.md §3.5.
"""
import pytest

from app.memory.kinds import MemoryKind, RecordStatus, SourceType
from app.memory.stores import InMemoryRecordStore
from app.memory.writer import (
    BASE_IMPORTANCE,
    MemoryWriter,
    classify_profile_key,
    render_episode,
    render_profile_fact,
    render_tool_outcome,
)


@pytest.fixture
def store():
    return InMemoryRecordStore()


@pytest.fixture
def writer(store):
    return MemoryWriter(record_store=store)


# ─────────────────────────────────────────────────────────────────────────
# Profile-key classification
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", [
    "name", "role", "email", "location", "timezone",
    "university", "degree", "github", "linkedin", "leetcode",
])
def test_identity_keys_classify_as_identity(key):
    assert classify_profile_key(key) is MemoryKind.IDENTITY


@pytest.mark.parametrize("key", [
    "tone", "language", "verbosity", "communication_style",
    "preferred_tone", "prefers_bullet_points", "favourite_editor",
])
def test_preference_keys_and_prefixes_classify_as_preference(key):
    assert classify_profile_key(key) is MemoryKind.PREFERENCE


@pytest.mark.parametrize("key", ["goal", "job_target", "career_goal", "target_role"])
def test_goal_keys_classify_as_goal(key):
    assert classify_profile_key(key) is MemoryKind.GOAL


def test_unknown_keys_fall_back_to_semantic():
    """Semantic is the safe default: retrieved on relevance, never forced."""
    assert classify_profile_key("hobby") is MemoryKind.SEMANTIC
    assert classify_profile_key("random_thing") is MemoryKind.SEMANTIC


def test_classification_is_case_and_whitespace_insensitive():
    assert classify_profile_key("  NAME  ") is MemoryKind.IDENTITY


def test_classification_handles_empty_input():
    assert classify_profile_key("") is MemoryKind.SEMANTIC


# ─────────────────────────────────────────────────────────────────────────
# Content rendering — must be self-contained
# ─────────────────────────────────────────────────────────────────────────

def test_known_keys_use_a_natural_template():
    assert render_profile_fact("name", "Vansh") == "The user's name is Vansh."
    assert render_profile_fact("github", "gh/x") == "The user's GitHub profile is gh/x."


def test_unknown_keys_get_a_readable_generic_sentence():
    assert render_profile_fact("preferred_tone", "concise") == (
        "The user's preferred tone is concise."
    )
    assert render_profile_fact("favourite_editor", "vim") == (
        "The user's favourite editor is vim."
    )


def test_rendered_fact_carries_both_key_and_value():
    """Neither half is recoverable from the other once embedded."""
    rendered = render_profile_fact("timezone", "Asia/Kolkata")
    assert "timezone" in rendered.lower()
    assert "Asia/Kolkata" in rendered


def test_episode_rendering_names_the_agent_that_answered():
    rendered = render_episode("find ML jobs", "returned 5 listings", "job")
    assert "find ML jobs" in rendered
    assert "returned 5 listings" in rendered
    assert "job agent" in rendered


def test_episode_rendering_falls_back_when_no_agent_is_known():
    assert "the assistant" in render_episode("a question", "an answer", None)


def test_episode_rendering_tolerates_a_missing_half():
    assert render_episode("just the question", "", None).startswith("The user asked")
    assert render_episode("", "", None) == ""


def test_tool_outcome_rendering_keeps_inputs_and_insight():
    rendered = render_tool_outcome("job", "job_search", '{"q": "ML"}', "found 5 roles")
    assert "job_search" in rendered
    assert '{"q": "ML"}' in rendered
    assert "found 5 roles" in rendered


# ─────────────────────────────────────────────────────────────────────────
# Writing profile facts
# ─────────────────────────────────────────────────────────────────────────

async def test_profile_fact_is_written_with_kind_and_salience(writer, store):
    record = await writer.record_profile_fact("vansh", "name", "Vansh Pratap Singh")

    assert record.kind is MemoryKind.IDENTITY
    assert record.importance == BASE_IMPORTANCE[MemoryKind.IDENTITY]
    assert record.content == "The user's name is Vansh Pratap Singh."
    assert record.structured == {
        "key": "name", "value": "Vansh Pratap Singh", "source": "explicit"
    }
    assert record.dedup_key == "profile:name"
    assert await store.count("vansh") == 1


async def test_rewriting_the_same_value_does_not_duplicate(writer, store):
    await writer.record_profile_fact("vansh", "name", "Vansh")
    await writer.record_profile_fact("vansh", "name", "Vansh")
    assert await store.count("vansh") == 1


async def test_a_changed_value_supersedes_rather_than_duplicating(writer, store):
    """
    A profile key holds exactly one current value, so a new value contradicts
    the old one. Two active "the user's role is..." records would be a bug.
    """
    await writer.record_profile_fact("vansh", "role", "student")
    await writer.record_profile_fact("vansh", "role", "engineer")

    assert await store.count("vansh", statuses=[RecordStatus.ACTIVE]) == 1
    assert await store.count("vansh", statuses=[RecordStatus.SUPERSEDED]) == 1

    current = await store.find_by_dedup_key("vansh", "profile:role")
    assert len(current) == 1
    assert "engineer" in current[0].content
    assert current[0].version == 2


async def test_repeated_changes_build_an_unbroken_version_chain(writer, store):
    """
    Regression: the replacement was originally built as a fresh record, so
    every version landed as v1 with a null `supersedes_id`. The history looked
    present — one active row, N superseded rows — while the links between them
    were gone, which is the part that makes it reconstructable.
    """
    for value in ("student", "intern", "engineer"):
        await writer.record_profile_fact("vansh", "role", value)

    active = await store.list("vansh", statuses=[RecordStatus.ACTIVE])
    assert len(active) == 1
    assert active[0].version == 3
    assert active[0].supersedes_id is not None

    # Walk the chain backwards; it must reach v1 without a break.
    seen, cursor = [], active[0]
    while cursor is not None:
        seen.append(cursor.version)
        cursor = (
            await store.get("vansh", cursor.supersedes_id)
            if cursor.supersedes_id else None
        )
    assert seen == [3, 2, 1]


async def test_superseded_history_is_retained_not_deleted(writer, store):
    await writer.record_profile_fact("vansh", "role", "student")
    await writer.record_profile_fact("vansh", "role", "engineer")
    old = await store.list("vansh", statuses=[RecordStatus.SUPERSEDED])
    assert "student" in old[0].content
    assert old[0].valid_to is not None


async def test_different_keys_do_not_conflict(writer, store):
    await writer.record_profile_fact("vansh", "name", "Vansh")
    await writer.record_profile_fact("vansh", "role", "engineer")
    assert await store.count("vansh", statuses=[RecordStatus.ACTIVE]) == 2


async def test_inferred_facts_are_attributed_to_chat(writer):
    inferred = await writer.record_profile_fact("vansh", "hobby", "chess", source="inferred")
    explicit = await writer.record_profile_fact("vansh", "role", "engineer", source="explicit")
    assert inferred.source_type is SourceType.CHAT
    assert explicit.source_type is SourceType.SYSTEM


async def test_confidence_is_carried_through(writer):
    record = await writer.record_profile_fact("vansh", "hobby", "chess", confidence=0.85)
    assert record.confidence == 0.85


async def test_facts_are_private_by_default(writer):
    from app.memory.kinds import Visibility
    record = await writer.record_profile_fact("vansh", "name", "Vansh")
    assert record.visibility is Visibility.PRIVATE


# ─────────────────────────────────────────────────────────────────────────
# Writing episodes
# ─────────────────────────────────────────────────────────────────────────

async def test_episode_is_written_as_episodic_memory(writer, store):
    record = await writer.record_episode(
        "vansh", "s1", "asked for ML jobs", "returned 5 listings", agent_used="job"
    )
    assert record.kind is MemoryKind.EPISODIC
    assert record.source_ref == "session:s1"
    assert record.structured["agent_used"] == "job"
    assert await store.count("vansh") == 1


async def test_failed_turns_are_less_important_than_successful_ones(writer):
    ok = await writer.record_episode("vansh", "s1", "q one", "answered", outcome="success")
    bad = await writer.record_episode("vansh", "s2", "q two", "could not", outcome="failed")
    assert bad.importance < ok.importance


async def test_episodes_carry_no_dedup_key(writer):
    """Episodes are events, not claims — two similar ones do not contradict."""
    record = await writer.record_episode("vansh", "s1", "a question", "an answer")
    assert record.dedup_key is None


async def test_similar_episodes_in_different_sessions_both_persist(writer, store):
    await writer.record_episode("vansh", "s1", "asked about jobs", "listed roles A")
    await writer.record_episode("vansh", "s2", "asked about jobs", "listed roles B")
    assert await store.count("vansh") == 2


async def test_an_empty_episode_is_not_written(writer, store):
    assert await writer.record_episode("vansh", "s1", "", "") is None
    assert await store.count("vansh") == 0


# ─────────────────────────────────────────────────────────────────────────
# Writing tool outcomes and documents
# ─────────────────────────────────────────────────────────────────────────

async def test_tool_outcome_becomes_procedural_memory(writer):
    record = await writer.record_tool_outcome(
        "vansh", "job", "job_search", '{"q": "ML"}', "found 5 roles"
    )
    assert record.kind is MemoryKind.PROCEDURAL
    assert record.source_ref == "tool:job:job_search"
    assert record.structured["tool_name"] == "job_search"


async def test_document_chunk_records_its_provenance(writer):
    record = await writer.record_document_chunk(
        "vansh", "Built My_Agent, a personal assistant.",
        document_id="resume_vansh_ab12", semantic_type="projects",
        chunk_index=3, source_file="resume.pdf",
    )
    assert record.kind is MemoryKind.DOCUMENT
    assert record.source_type is SourceType.UPLOAD
    assert record.source_ref == "resume_vansh_ab12"
    assert record.structured["chunk_index"] == 3
    assert record.structured["source_file"] == "resume.pdf"


async def test_blank_document_chunks_are_skipped(writer, store):
    assert await writer.record_document_chunk("vansh", "   ", document_id="d1") is None
    assert await store.count("vansh") == 0


# ─────────────────────────────────────────────────────────────────────────
# Tenant isolation
# ─────────────────────────────────────────────────────────────────────────

async def test_owners_do_not_share_profile_state(writer, store):
    """A guest stating their role must not supersede the owner's."""
    await writer.record_profile_fact("vansh", "role", "engineer")
    await writer.record_profile_fact("guest-abc", "role", "recruiter")

    assert await store.count("vansh", statuses=[RecordStatus.ACTIVE]) == 1
    assert await store.count("guest-abc", statuses=[RecordStatus.ACTIVE]) == 1
    assert await store.count(statuses=[RecordStatus.SUPERSEDED]) == 0
