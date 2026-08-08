"""
Structural guards for the memory / application-data boundary (Phase 0).

`ShortTermMemory` reached 1,145 lines owning twelve unrelated entity types, and
`MemoryManager` grew thirty pass-through methods to reach them, because nothing
prevented it. These tests are that prevention: adding an attendance or email
method back onto the memory objects fails here rather than in review.

See docs/MEMORY_ARCHITECTURE.md §1.1–1.2.
"""
import pytest

from app.db import Base, async_session_maker, engine
from app.domain.academic import academic_repository
from app.domain.email import email_repository
from app.domain.jobs import jobs_repository
from app.memory.memory_manager import memory_manager
from app.memory.short_term_memory import short_term_memory


# Vocabulary that marks a method as belonging to an application record rather
# than to memory.
DOMAIN_TERMS = (
    "attendance", "timetable", "exam", "plan",
    "bookmark", "draft", "template", "playbook",
)


def public_methods(obj) -> list:
    return [
        name for name in dir(obj)
        if not name.startswith("_") and callable(getattr(obj, name))
    ]


# ─────────────────────────────────────────────────────────────────────────
# The boundary
# ─────────────────────────────────────────────────────────────────────────

def test_memory_manager_exposes_no_application_record_methods():
    leaked = [
        name for name in public_methods(memory_manager)
        if any(term in name.lower() for term in DOMAIN_TERMS)
    ]
    assert leaked == [], (
        f"MemoryManager grew application-record methods: {leaked}. "
        "These belong on a repository in app/domain/."
    )


def test_short_term_memory_exposes_no_application_record_methods():
    leaked = [
        name for name in public_methods(short_term_memory)
        if any(term in name.lower() for term in DOMAIN_TERMS)
    ]
    assert leaked == [], (
        f"ShortTermMemory grew application-record methods: {leaked}. "
        "These belong on a repository in app/domain/."
    )


def test_memory_manager_still_exposes_its_memory_operations():
    """The boundary must not have been enforced by deleting the wrong side."""
    for name in (
        "on_user_input", "on_agent_response", "retrieve_context",
        "format_context_for_prompt", "save_profile_fact", "get_profile_facts",
        "forget_profile_fact", "forget_all_profile", "store_episode",
        "get_recent_episodes", "save_tool_outcome", "get_tool_insights",
        "store_resume", "search_long_term",
    ):
        assert hasattr(memory_manager, name), f"MemoryManager lost {name}()"


# ─────────────────────────────────────────────────────────────────────────
# One engine, one metadata registry
# ─────────────────────────────────────────────────────────────────────────

def test_every_repository_shares_the_process_wide_engine():
    """
    A second engine means a second connection pool competing for the same
    Postgres connection limit — and the voice worker runs in this process.
    """
    for repo in (short_term_memory, academic_repository,
                 email_repository, jobs_repository):
        assert repo.async_session_maker is async_session_maker, (
            f"{type(repo).__name__} does not use the shared session factory"
        )
    assert short_term_memory.engine is engine


def test_split_models_register_on_a_single_metadata():
    """
    Memory and domain models live in separate modules but must share one Base,
    or create_all() builds only half the schema.
    """
    import app.memory.models  # noqa: F401
    import app.domain.models  # noqa: F401

    expected = {
        # memory
        "chat_history", "user_profile", "episodic_memory", "tool_memory",
        # application records
        "attendance", "timetable", "job_bookmarks",
        "email_drafts", "email_templates", "exams", "plans",
    }
    assert expected <= set(Base.metadata.tables), (
        f"missing tables: {expected - set(Base.metadata.tables)}"
    )


def test_deleted_playbook_table_is_no_longer_registered():
    assert "agent_playbooks" not in Base.metadata.tables


# ─────────────────────────────────────────────────────────────────────────
# Repositories carry the behaviour that moved
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("repo,method", [
    (academic_repository, "store_attendance"),
    (academic_repository, "upsert_attendance"),
    (academic_repository, "retrieve_attendance"),
    (academic_repository, "store_timetable_entry"),
    (academic_repository, "retrieve_timetable"),
    (academic_repository, "clear_timetable"),
    (academic_repository, "store_exam"),
    (academic_repository, "retrieve_exams"),
    (academic_repository, "store_plan"),
    (academic_repository, "retrieve_plans"),
    (academic_repository, "mark_plan_done"),
    (email_repository, "save_draft"),
    (email_repository, "get_drafts"),
    (email_repository, "mark_draft_sent"),
    (email_repository, "save_template"),
    (email_repository, "get_templates"),
    (jobs_repository, "save_bookmark"),
    (jobs_repository, "is_bookmarked"),
    (jobs_repository, "get_bookmarks"),
    (jobs_repository, "get_bookmarked_urls"),
])
def test_repository_provides_moved_method(repo, method):
    assert callable(getattr(repo, method, None)), (
        f"{type(repo).__name__}.{method}() went missing in the split"
    )


def test_sensitive_fact_filtering_stayed_with_memory():
    """
    _is_sensitive guards what reaches an LLM prompt. It must travel with the
    profile-fact code, not be stranded on the domain side.
    """
    assert short_term_memory._is_sensitive("password", "hunter2") is True
    assert short_term_memory._is_sensitive("preferred_tone", "concise") is False
