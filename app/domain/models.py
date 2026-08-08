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
    Integer,
    String,
    Text,
    Time,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


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
