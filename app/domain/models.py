"""
Application-record tables.

These are things the application *stores*, not things the assistant
*remembers*: a timetable entry is a scheduling record, an email draft is a
document the user is composing. They were previously filed under `app/memory/`,
which is how `ShortTermMemory` grew to own twelve unrelated entity types and
why `MemoryManager` accumulated thirty pass-through methods.

The memory system may reference these records — an episodic memory can note
"attendance updated" — but it does not own them.
"""
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    Time,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class PendingActionRecord(Base):
    """
    An action waiting for a decision, durable enough to survive the process.

    Deliberately separate from `action_audit`, and the split is the design.
    The audit row is permanent and holds only digests — it must be safe to keep
    forever and safe to hand to anyone reviewing what happened. This row is the
    opposite: transient, and it necessarily holds the real arguments, because
    you cannot execute an approved action without them.

    Keeping them apart means the permanent record never accumulates message
    bodies and recipient addresses. This row is deleted the moment it is
    claimed or refused, so the executable payload exists only during the window
    where someone might say yes.

    What is *not* stored is the callable. A row that could become executable
    code is a remote-execution primitive; `tool` is a name, resolved through
    `app.agents.confirmable_tools` at confirmation time, so the code that runs
    always comes from the application rather than from the database.
    """

    __tablename__ = "pending_action"
    __table_args__ = (
        # Single-use is enforced here as well as by deletion: two rows could
        # never share a token, so a claim cannot match more than one action.
        UniqueConstraint("token_hash", name="uq_pending_action_token"),
        Index("ix_pending_action_owner_conversation", "owner_id", "conversation_id"),
        Index("ix_pending_action_expires_at", "expires_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    token_hash = Column(String(64), nullable=False)
    audit_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    owner_id = Column(String(255), nullable=False, index=True)
    conversation_id = Column(String(255), nullable=True)

    tool = Column(String(128), nullable=False)
    effect = Column(String(32), nullable=False)

    # The executable payload. Present only while a decision is outstanding.
    arguments = Column(JSONB, nullable=False, default=dict)
    preview = Column(Text, nullable=False, default="")

    content_hash = Column(String(64), nullable=False)
    idempotency_key = Column(String(64), nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)


class ActionAudit(Base):
    """
    One row per consequential action, from the moment it is proposed.

    Written for EXTERNAL_WRITE and DESTRUCTIVE effects only. Reads and local
    writes do not appear here: they are reversible by the user and auditing
    every résumé lookup would bury the handful of rows that matter.

    The row is created *before* anything can happen, so an action that was
    prepared and never approved is as visible as one that was sent. That is the
    point — "did the assistant try to email someone?" and "did an email go out?"
    are different questions, and only a record written at proposal time can
    answer the first.

    Nothing sensitive is stored. Arguments, preview text and the confirmation
    token are all reduced to SHA-256 digests: enough to prove that what
    executed is what was approved, not enough to reconstruct a recipient or a
    message body from a database dump.
    """

    __tablename__ = "action_audit"
    __table_args__ = (
        # The durable idempotency guarantee. At most one *executed* row may
        # exist per idempotency key, enforced by Postgres rather than by
        # process memory — which is what makes a retry after a restart, or on
        # another replica, unable to send the same email twice.
        Index(
            "uq_action_audit_executed_idempotency",
            "idempotency_key",
            unique=True,
            postgresql_where=text("status = 'executed'"),
        ),
        Index("ix_action_audit_owner_created", "owner_id", "created_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    owner_id = Column(String(255), nullable=False, index=True)
    conversation_id = Column(String(255), nullable=True, index=True)

    effect = Column(String(32), nullable=False)
    tool = Column(String(128), nullable=False)

    # Digests, never the values themselves.
    arguments_hash = Column(String(64), nullable=False)
    preview_hash = Column(String(64), nullable=False)
    token_hash = Column(String(64), nullable=False, index=True)

    idempotency_key = Column(String(64), nullable=False, index=True)

    # pending → confirmed → executed | failed
    #         → cancelled
    #         → expired
    status = Column(String(16), nullable=False, default="pending", index=True)

    outcome = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    confirmed_at = Column(DateTime(timezone=True), nullable=True)
    executed_at = Column(DateTime(timezone=True), nullable=True)


class Attendance(Base):
    """Attendance records."""
    __tablename__ = "attendance"
    # One record per (user, day, subject). Re-running a scrape must update the
    # existing row rather than append a duplicate — duplicates inflate the
    # denominator and corrupt the attendance-percentage risk calculation.
    __table_args__ = (
        UniqueConstraint("user_id", "date", "subject", name="uq_attendance_user_date_subject"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    subject = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)  # present, absent, late
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ScheduleUpload(Base):
    """
    One timetable document, and the rows it produced.

    Exists so a class can be traced back to the page it was read from, and so a
    new semester can replace an old one without either destroying it or leaving
    two live timetables that nothing can choose between. `Timetable` rows point
    here; the group is what gets activated or deactivated, not each row
    individually.

    `semester` is nullable and supplied by the uploader, because the document
    this was built for does not state one. Recording it as user-supplied keeps
    it distinguishable from anything actually read off the page.
    """
    __tablename__ = "schedule_upload"
    __table_args__ = (
        # Partial, not a plain UniqueConstraint. A user re-uploading a
        # document they used before — reverting to an earlier semester's
        # timetable, or simply re-running an upload — must not collide with
        # that same document's now-retired row; a plain (user_id,
        # content_hash) constraint would refuse it forever. Uniqueness holds
        # only among *active* uploads, which is the one that actually matters:
        # at most one live document per user at a time. Same pattern as
        # `ActionAudit.uq_action_audit_executed_idempotency` below.
        Index(
            "uq_schedule_upload_active_hash",
            "user_id", "content_hash",
            unique=True,
            postgresql_where=text("is_active = true"),
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    content_hash = Column(String(64), nullable=False)
    source_title = Column(Text, nullable=True)
    semester = Column(String(32), nullable=True)
    page_count = Column(Integer, nullable=True)
    entry_count = Column(Integer, nullable=False, default=0)
    parse_notes = Column(Text, nullable=True)
    ingest_method = Column(String(32), nullable=False, default="pdf_parser")
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Timetable(Base):
    """Class/event timetable."""
    __tablename__ = "timetable"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    day_of_week = Column(Integer, nullable=False)  # 0=Monday, 6=Sunday
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    subject = Column(String(255), nullable=False)
    location = Column(String(255), nullable=True)
    instructor = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    meta_data = Column(JSONB, nullable=True)  # custom fields (renamed from metadata)
    # Which upload produced this row. Nullable: rows added by voice
    # ("add Physics Monday 9am") have no document behind them, and pre-existing
    # rows predate this column entirely.
    schedule_upload_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class JobBookmark(Base):
    """Saved/bookmarked job listings per user."""
    __tablename__ = "job_bookmarks"
    # Makes "already bookmarked" a database invariant instead of a
    # check-then-insert race between concurrent tool calls.
    __table_args__ = (
        UniqueConstraint("user_id", "url", name="uq_job_bookmark_user_url"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    title = Column(String(512), nullable=False)
    url = Column(Text, nullable=False)
    company = Column(String(255), nullable=True)
    snippet = Column(Text, nullable=True)
    rank_score = Column(Float, nullable=True)
    search_query = Column(String(512), nullable=True)  # what query surfaced this job
    skills_matched = Column(JSONB, nullable=True)       # which user skills matched
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EmailDraft(Base):
    """Saved email drafts — never sent, always user-controlled."""
    __tablename__ = "email_drafts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    subject = Column(String(512), nullable=False, default="")
    recipient_name = Column(String(255), nullable=True)
    tone = Column(String(50), nullable=False, default="professional")
    greeting = Column(Text, nullable=True)
    body = Column(Text, nullable=False, default="")
    closing = Column(Text, nullable=True)
    signature = Column(Text, nullable=True)
    # status: draft | archived
    status = Column(String(50), nullable=False, default="draft")
    # free-form context bag: {"job_url": ..., "company": ..., "template_name": ...}
    context = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EmailTemplate(Base):
    """Reusable email templates (job application, follow-up, meeting request, etc.)."""
    __tablename__ = "email_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    # short slug: "job_application", "follow_up", "meeting_request", custom
    name = Column(String(255), nullable=False)
    tone = Column(String(50), nullable=False, default="professional")
    subject_template = Column(Text, nullable=False, default="")
    body_template = Column(Text, nullable=False, default="")
    # placeholders used: ["{{company}}", "{{position}}", ...]
    placeholders = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Exam(Base):
    """Upcoming exam / test schedule."""
    __tablename__ = "exams"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    subject = Column(String(255), nullable=False)
    exam_date = Column(Date, nullable=False, index=True)
    start_time = Column(Time, nullable=True)
    location = Column(String(255), nullable=True)
    exam_type = Column(String(100), nullable=True)  # midterm, final, quiz, assignment
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Plan(Base):
    """Daily plans, tasks, and reminders."""
    __tablename__ = "plans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    plan_date = Column(Date, nullable=False, index=True)   # the day this task belongs to
    title = Column(String(512), nullable=False)            # short label, e.g. "Study NLP chapter 3"
    description = Column(Text, nullable=True)              # optional detail
    priority = Column(String(20), nullable=False, default="medium")  # high | medium | low
    is_done = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
