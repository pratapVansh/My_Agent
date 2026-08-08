"""
Memory tables — things the assistant remembers about the user.

Application records (attendance, timetable, exams, plans, job bookmarks, email
drafts and templates) live in `app/domain/models.py`. The two modules share the
single `Base` from `app.db.base` so one `create_all` builds the whole schema.

Phase 1 of the memory redesign replaces the tables below with a single typed
`MemoryRecord`; see docs/MEMORY_ARCHITECTURE.md §3.3.
"""
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func
import uuid

from app.db.base import Base

# Re-exported so existing imports of `Base` from this module keep working.
__all__ = [
    "Base",
    "ChatHistory",
    "UserProfile",
    "EpisodicMemory",
    "ToolMemory",
    "MemoryRecordORM",
    "MemoryEventORM",
    "ConversationORM",
    "TurnORM",
]


class MemoryRecordORM(Base):
    """
    The unified memory record (memory redesign, Phase 1).

    One table with a `kind` discriminator replaces the per-type tables below.
    Those remain the read path until Phase 2 cuts over; this table is populated
    by dual-write and by `scripts/migrate_memory_v2.py`.

    The Python-side model is `app.memory.record.MemoryRecord`; this class is
    only its persistence mapping. See docs/MEMORY_ARCHITECTURE.md §3.3.
    """
    __tablename__ = "memory_records"
    __table_args__ = (
        # Primary retrieval access pattern: one owner's active records of a
        # given kind, best-first.
        Index(
            "ix_memory_records_owner_kind_status",
            "owner_id", "kind", "status", "importance",
        ),
        # Exact-duplicate prevention as a database invariant rather than an
        # application check-then-insert race — the same reasoning as the
        # unique constraint on job bookmarks. Partial, because a superseded or
        # archived row legitimately shares its hash with the version that
        # replaced it.
        Index(
            "uq_memory_records_active_content",
            "owner_id", "kind", "content_hash",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
        # Conflict detection: two active records sharing a dedup_key
        # contradict each other by definition.
        Index(
            "ix_memory_records_owner_dedup_key",
            "owner_id", "dedup_key",
            postgresql_where=text("dedup_key IS NOT NULL AND status = 'active'"),
        ),
        # The background embedding pass claims work through this.
        Index("ix_memory_records_embedding_status", "embedding_status"),
        # Full-text search channel (Phase 2). Functional GIN index over the
        # same expression PostgresLexicalIndex queries; the explicit 'english'
        # regconfig is what makes to_tsvector IMMUTABLE and therefore indexable.
        #
        # NOTE for deployments created during Phase 1: create_all() adds
        # missing *tables*, not missing indexes on existing tables. Lexical
        # search still returns correct results without this index (sequential
        # scan), so its absence is a performance issue, never a correctness
        # one. `scripts/migrate_memory_v2.py --apply` creates it.
        Index(
            "ix_memory_records_content_fts",
            text("to_tsvector('english', content)"),
            postgresql_using="gin",
        ),
        # Recency ordering for episodic recall.
        Index("ix_memory_records_owner_occurred", "owner_id", "occurred_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(String(255), nullable=False, index=True)
    kind = Column(String(32), nullable=False)

    # Dual representation: `content` is the self-contained sentence that gets
    # embedded and injected; `structured` is the machine-readable form.
    content = Column(Text, nullable=False)
    structured = Column(JSONB, nullable=False, default=dict)

    # Salience
    importance = Column(Float, nullable=False, default=0.5)
    confidence = Column(Float, nullable=False, default=1.0)
    pinned = Column(Boolean, nullable=False, default=False)

    # Temporal (bitemporal: when it was true vs when we recorded it)
    occurred_at = Column(DateTime(timezone=True), nullable=True)
    valid_from = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    valid_to = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False,
                        server_default=func.now(), onupdate=func.now())
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    access_count = Column(Integer, nullable=False, default=0)

    # Lineage
    source_type = Column(String(32), nullable=False, default="system")
    source_ref = Column(String(512), nullable=True)
    derived_from = Column(JSONB, nullable=False, default=list)
    supersedes_id = Column(UUID(as_uuid=True), nullable=True)
    version = Column(Integer, nullable=False, default=1)

    # Governance
    visibility = Column(String(16), nullable=False, default="private")
    sensitivity = Column(String(16), nullable=False, default="normal")
    status = Column(String(16), nullable=False, default="active")

    # Dedup / vector linkage. The Qdrant point id is the record id itself, so
    # no separate reference column is needed — and the eventual move to
    # pgvector becomes a column addition rather than a remapping.
    content_hash = Column(String(64), nullable=False)
    dedup_key = Column(String(255), nullable=True)
    embedding_status = Column(String(16), nullable=False, default="pending")


class ConversationORM(Base):
    """
    A conversation thread (Phase 4).

    The primary key is the session/conversation *string* rather than a UUID, so
    the identifier the client already sends as `session_id` is the row key
    directly. That removes an entire mapping layer, and lets the LiveKit
    worker's `lk_<room>_<identity>` keys and the browser's `session_<uuid>`
    keys coexist without translation.

    A conversation is what survives a page refresh: the browser persists its id,
    so reconnecting resumes the thread instead of silently starting a new one.
    See docs/MEMORY_ARCHITECTURE.md §3.8.
    """
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_owner_active", "owner_id", "last_active_at"),
    )

    id = Column(String(255), primary_key=True)
    owner_id = Column(String(255), nullable=False, index=True)
    title = Column(String(512), nullable=True)
    # active | archived
    status = Column(String(16), nullable=False, default="active")
    # text | voice | mixed — how this thread has been conducted so far
    modality = Column(String(16), nullable=False, default="text")

    turn_count = Column(Integer, nullable=False, default=0)
    # Everything older than the last `summary_through_seq` turns is represented
    # by `running_summary`, which is what keeps working memory bounded no
    # matter how long a thread runs.
    running_summary = Column(Text, nullable=True)
    summary_through_seq = Column(Integer, nullable=False, default=0)

    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_active_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class TurnORM(Base):
    """
    One message within a conversation (Phase 4).

    Supersedes `chat_history`, which is keyed only by an ephemeral session id
    and carries no ordering guarantee beyond its timestamp. `sequence` is
    assigned atomically so ordering survives concurrent writes — which happen
    routinely, since a voice turn and a text turn can land together.
    """
    __tablename__ = "conversation_turns"
    __table_args__ = (
        UniqueConstraint("conversation_id", "sequence", name="uq_turn_conversation_seq"),
        Index("ix_turns_conversation_seq", "conversation_id", "sequence"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(String(255), nullable=False, index=True)
    owner_id = Column(String(255), nullable=False, index=True)
    sequence = Column(Integer, nullable=False)

    role = Column(String(50), nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    # text | voice — a thread may mix both, which is the point of one shared
    # conversation rather than one per transport.
    modality = Column(String(16), nullable=False, default="text")

    agent = Column(String(100), nullable=True)
    intent = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class MemoryEventORM(Base):
    """
    Outbox for asynchronous memory ingestion (Phase 3).

    Written on the request path, drained by the worker. Durable at-least-once
    delivery without operating a message broker.

    The Python-side model is `app.memory.events.MemoryEvent`.
    See docs/MEMORY_ARCHITECTURE.md §3.5.
    """
    __tablename__ = "memory_events"
    __table_args__ = (
        # The worker's poll: pending events, oldest first.
        Index("ix_memory_events_status_created", "status", "created_at"),
        # Grouping for batched extraction. A real indexed column rather than a
        # JSONB lookup, because this is read on every poll.
        Index("ix_memory_events_group", "group_key", "status"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(String(255), nullable=False, index=True)
    event_type = Column(String(32), nullable=False)
    group_key = Column(String(255), nullable=False, default="")
    payload = Column(JSONB, nullable=False, default=dict)

    status = Column(String(16), nullable=False, default="pending")
    attempts = Column(Integer, nullable=False, default=0)
    last_error = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    claimed_at = Column(DateTime(timezone=True), nullable=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)


class ChatHistory(Base):
    """Chat conversation history."""
    __tablename__ = "chat_history"
    # Every read filters on user_id + session_id and orders by created_at;
    # a composite index matches that access pattern directly.
    __table_args__ = (
        Index("ix_chat_history_user_session_time", "user_id", "session_id", "created_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    meta_data = Column(JSONB, nullable=True)  # agent_used, intent, etc. (renamed from metadata)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserProfile(Base):
    """
    Explicit profile memory — user facts and preferences.
    Stable, consent-gated facts, as distinct from inferred conversational memory.
    Each (user_id, key) pair is unique; saving an existing key upserts the value.
    """
    __tablename__ = "user_profile"
    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_user_profile_key"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    # Fact key: e.g. "preferred_tone", "name", "role", "job_target", "timezone"
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=False)
    # source: explicit (user said "remember this") | inferred (extracted from conversation)
    source = Column(String(50), nullable=False, default="explicit")
    # confidence: 0.0–1.0
    confidence = Column(Float, nullable=False, default=1.0)
    # consent_level: explicit | implicit | none
    consent_level = Column(String(50), nullable=False, default="explicit")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EpisodicMemory(Base):
    """
    Per-turn episodic memory — brief summary of what happened each conversation turn.
    Gives the agent cross-session context without loading full chat history.
    """
    __tablename__ = "episodic_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False)
    # Brief (<=150 chars) summary of the user's query
    user_summary = Column(String(300), nullable=False, default="")
    # Brief summary of what the agent did / the outcome
    agent_summary = Column(String(300), nullable=False, default="")
    agent_used = Column(String(100), nullable=True)
    intent = Column(String(100), nullable=True)
    # outcome: success | failed | clarification_needed
    outcome = Column(String(50), nullable=False, default="success")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class ToolMemory(Base):
    """
    Cross-session tool-use memory (Fix 3: Tool-Use Memory).
    Records what inputs/strategies produced good results for each tool,
    so agents can learn from past successes across sessions.
    Only 'good' outcome records are retrieved for future hint injection.
    """
    __tablename__ = "tool_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    # Which agent made this tool call: job, email, academic, profile
    agent_name = Column(String(100), nullable=False, index=True)
    # The specific tool called: job_search, email_draft, get_attendance_summary, etc.
    tool_name = Column(String(100), nullable=False, index=True)
    # Key input parameters (compressed, ≤500 chars) — e.g. {"query": "ML engineer NYC"}
    inputs_summary = Column(Text, nullable=False, default="")
    # "good" (produced useful output) | "poor" (ran but result was unhelpful) | "failed" (error)
    outcome_quality = Column(String(20), nullable=False, default="good")
    # What the tool returned / the insight extracted (≤500 chars)
    key_insight = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
