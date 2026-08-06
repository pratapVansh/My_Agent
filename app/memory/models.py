"""
PostgreSQL Database Models for Short-term Memory.
Stores chat history, attendance, timetable, email drafts/templates, and exams.
"""
from sqlalchemy import Column, String, DateTime, Integer, Text, Boolean, Date, Time, Float, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


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


class UserProfile(Base):
    """
    Explicit profile memory — user facts and preferences.
    Replaces fragile mem0 extraction for stable, consent-gated facts.
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


class AgentPlaybook(Base):
    """
    M5: Versioned system prompts for each specialist agent.
    Allows iterating on prompt versions without code deploys.
    Each (agent_name, version) pair is unique. One version per agent is active at a time.
    """
    __tablename__ = "agent_playbooks"
    __table_args__ = (UniqueConstraint("agent_name", "version", name="uq_playbook_agent_version"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Agent identifier: job | email | academic | profile | planner | response
    agent_name = Column(String(100), nullable=False, index=True)
    # Semantic version: "v1", "v2", "2024-04-01-hotfix"
    version = Column(String(50), nullable=False)
    # The full system prompt text for this agent at this version
    prompt = Column(Text, nullable=False)
    # Only one version per agent should be active at a time
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    # Free-text changelog note: what changed and why
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


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
