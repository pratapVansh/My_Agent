"""
PostgreSQL Short-term Memory Implementation.
Stores chat history, attendance, and timetable using async SQLAlchemy.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, and_, desc
from sqlalchemy.sql import func
from typing import List, Dict, Any, Optional
from datetime import date, time, datetime
import uuid

from app.config import settings
from app.memory.models import Base, ChatHistory, Attendance, Timetable, JobBookmark, EmailDraft, EmailTemplate, Exam, UserProfile, EpisodicMemory, AgentPlaybook, Plan, ToolMemory


class ShortTermMemory:
    """
    Short-term memory using PostgreSQL.
    Handles transient data: chat history, attendance, schedules.
    """

    def __init__(self):
        """Initialize async database engine and session."""
        self.engine = create_async_engine(
            settings.postgres_url,
            echo=settings.environment == "development",
            pool_pre_ping=True
        )
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def init_db(self):
        """Create database tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self):
        """Close database connections."""
        await self.engine.dispose()

    # Chat History Operations
    async def store_chat_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a chat message.

        Args:
            user_id: User identifier
            session_id: Conversation session ID
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Additional metadata

        Returns:
            Message ID
        """
        async with self.async_session_maker() as session:
            message = ChatHistory(
                user_id=user_id,
                session_id=session_id,
                role=role,
                content=content,
                meta_data=metadata  # Uses meta_data column
            )
            session.add(message)
            await session.commit()
            return str(message.id)

    async def retrieve_chat_history(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a user/session.

        Args:
            user_id: User identifier
            session_id: Optional session filter
            limit: Maximum number of messages

        Returns:
            List of chat messages
        """
        async with self.async_session_maker() as session:
            query = select(ChatHistory).where(ChatHistory.user_id == user_id)

            if session_id:
                query = query.where(ChatHistory.session_id == session_id)

            query = query.order_by(desc(ChatHistory.created_at)).limit(limit)

            result = await session.execute(query)
            messages = result.scalars().all()

            return [
                {
                    "id": str(msg.id),
                    "role": msg.role,
                    "content": msg.content,
                    "metadata": msg.meta_data,  # Read from meta_data column
                    "created_at": msg.created_at.isoformat()
                }
                for msg in reversed(messages)  # Return in chronological order
            ]

    async def get_recent_context(
        self,
        user_id: str,
        session_id: str,
        last_n: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation context for LLM injection.

        Args:
            user_id: User identifier
            session_id: Session identifier
            last_n: Number of recent messages

        Returns:
            List of messages in format [{"role": "...", "content": "..."}]
        """
        messages = await self.retrieve_chat_history(user_id, session_id, last_n)
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

    # Attendance Operations
    async def store_attendance(
        self,
        user_id: str,
        date: date,
        subject: str,
        status: str,
        notes: Optional[str] = None
    ) -> str:
        """
        Store attendance record.

        Args:
            user_id: User identifier
            date: Attendance date
            subject: Subject/class name
            status: Attendance status (present, absent, late)
            notes: Optional notes

        Returns:
            Attendance record ID
        """
        async with self.async_session_maker() as session:
            attendance = Attendance(
                user_id=user_id,
                date=date,
                subject=subject,
                status=status,
                notes=notes
            )
            session.add(attendance)
            await session.commit()
            return str(attendance.id)

    async def retrieve_attendance(
        self,
        user_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        subject: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve attendance records.

        Args:
            user_id: User identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            subject: Optional subject filter

        Returns:
            List of attendance records
        """
        async with self.async_session_maker() as session:
            query = select(Attendance).where(Attendance.user_id == user_id)

            if start_date:
                query = query.where(Attendance.date >= start_date)
            if end_date:
                query = query.where(Attendance.date <= end_date)
            if subject:
                query = query.where(Attendance.subject == subject)

            query = query.order_by(desc(Attendance.date))

            result = await session.execute(query)
            records = result.scalars().all()

            return [
                {
                    "id": str(record.id),
                    "date": record.date.isoformat(),
                    "subject": record.subject,
                    "status": record.status,
                    "notes": record.notes
                }
                for record in records
            ]

    # Timetable Operations
    async def store_timetable_entry(
        self,
        user_id: str,
        day_of_week: int,
        start_time: time,
        end_time: time,
        subject: str,
        location: Optional[str] = None,
        instructor: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store timetable entry.

        Args:
            user_id: User identifier
            day_of_week: Day (0=Monday, 6=Sunday)
            start_time: Class start time
            end_time: Class end time
            subject: Subject name
            location: Optional location
            instructor: Optional instructor name
            metadata: Optional additional data

        Returns:
            Timetable entry ID
        """
        async with self.async_session_maker() as session:
            entry = Timetable(
                user_id=user_id,
                day_of_week=day_of_week,
                start_time=start_time,
                end_time=end_time,
                subject=subject,
                location=location,
                instructor=instructor,
                meta_data=metadata  # Uses meta_data column
            )
            session.add(entry)
            await session.commit()
            return str(entry.id)

    async def retrieve_timetable(
        self,
        user_id: str,
        day_of_week: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve timetable entries.

        Args:
            user_id: User identifier
            day_of_week: Optional day filter

        Returns:
            List of timetable entries
        """
        async with self.async_session_maker() as session:
            query = select(Timetable).where(
                and_(Timetable.user_id == user_id, Timetable.is_active == True)
            )

            if day_of_week is not None:
                query = query.where(Timetable.day_of_week == day_of_week)

            query = query.order_by(Timetable.day_of_week, Timetable.start_time)

            result = await session.execute(query)
            entries = result.scalars().all()

            return [
                {
                    "id": str(entry.id),
                    "day_of_week": entry.day_of_week,
                    "start_time": entry.start_time.isoformat(),
                    "end_time": entry.end_time.isoformat(),
                    "subject": entry.subject,
                    "location": entry.location,
                    "instructor": entry.instructor,
                    "metadata": entry.meta_data  # Read from meta_data column
                }
                for entry in entries
            ]


    async def clear_timetable(self, user_id: str) -> int:
        """
        Soft-delete all active timetable entries for a user by setting is_active=False.
        Returns the number of entries deactivated.
        Used before uploading a new semester timetable so old entries don't pollute results.
        """
        from sqlalchemy import update as sa_update
        async with self.async_session_maker() as session:
            result = await session.execute(
                sa_update(Timetable)
                .where(and_(Timetable.user_id == user_id, Timetable.is_active == True))
                .values(is_active=False)
            )
            await session.commit()
            return result.rowcount

    # Job Bookmark Operations
    async def save_job_bookmark(
        self,
        user_id: str,
        title: str,
        url: str,
        company: Optional[str] = None,
        snippet: Optional[str] = None,
        rank_score: Optional[float] = None,
        search_query: Optional[str] = None,
        skills_matched: Optional[List[str]] = None,
    ) -> str:
        """Save a job bookmark. Silently skips if URL already bookmarked."""
        if await self.is_job_bookmarked(user_id, url):
            return "already_saved"
        async with self.async_session_maker() as session:
            bookmark = JobBookmark(
                user_id=user_id,
                title=title,
                url=url,
                company=company,
                snippet=snippet,
                rank_score=rank_score,
                search_query=search_query,
                skills_matched=skills_matched or [],
            )
            session.add(bookmark)
            await session.commit()
            return str(bookmark.id)

    async def is_job_bookmarked(self, user_id: str, url: str) -> bool:
        """Return True if this URL is already bookmarked by the user."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(JobBookmark).where(
                    and_(JobBookmark.user_id == user_id, JobBookmark.url == url)
                )
            )
            return result.scalars().first() is not None

    async def get_job_bookmarks(
        self, user_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve saved job bookmarks for a user, newest first."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(JobBookmark)
                .where(JobBookmark.user_id == user_id)
                .order_by(desc(JobBookmark.created_at))
                .limit(limit)
            )
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "title": r.title,
                    "url": r.url,
                    "company": r.company,
                    "snippet": r.snippet,
                    "rank_score": r.rank_score,
                    "search_query": r.search_query,
                    "skills_matched": r.skills_matched or [],
                    "saved_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    async def get_bookmarked_urls(self, user_id: str) -> set:
        """Return a set of all bookmarked URLs for fast deduplication."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(JobBookmark.url).where(JobBookmark.user_id == user_id)
            )
            return {row[0] for row in result.all()}

    # ── Email Draft Operations ──────────────────────────────────────────────

    async def save_email_draft(
        self,
        user_id: str,
        subject: str,
        body: str,
        recipient_name: Optional[str] = None,
        tone: str = "professional",
        greeting: Optional[str] = None,
        closing: Optional[str] = None,
        signature: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist an email draft. Returns draft ID."""
        async with self.async_session_maker() as session:
            draft = EmailDraft(
                user_id=user_id,
                subject=subject,
                body=body,
                recipient_name=recipient_name,
                tone=tone,
                greeting=greeting,
                closing=closing,
                signature=signature,
                status="draft",
                context=context or {},
            )
            session.add(draft)
            await session.commit()
            return str(draft.id)

    async def get_email_drafts(
        self, user_id: str, limit: int = 10, status: str = "draft"
    ) -> List[Dict[str, Any]]:
        """Retrieve saved email drafts, newest first."""
        async with self.async_session_maker() as session:
            query = (
                select(EmailDraft)
                .where(and_(EmailDraft.user_id == user_id, EmailDraft.status == status))
                .order_by(desc(EmailDraft.created_at))
                .limit(limit)
            )
            result = await session.execute(query)
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "subject": r.subject,
                    "recipient_name": r.recipient_name,
                    "tone": r.tone,
                    "greeting": r.greeting,
                    "body": r.body,
                    "closing": r.closing,
                    "signature": r.signature,
                    "status": r.status,
                    "context": r.context or {},
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    async def mark_draft_sent(
        self,
        draft_id: str,
        user_id: str,
        sent_to: str,
    ) -> bool:
        """Mark a draft as sent and record the recipient address."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(EmailDraft).where(
                    and_(
                        EmailDraft.id == uuid.UUID(draft_id),
                        EmailDraft.user_id == user_id,
                    )
                )
            )
            draft = result.scalar_one_or_none()
            if not draft:
                return False
            draft.status = "sent"
            draft.context = {**(draft.context or {}), "sent_to": sent_to}
            await session.commit()
            return True

    # ── Email Template Operations ───────────────────────────────────────────

    async def save_email_template(
        self,
        user_id: str,
        name: str,
        subject_template: str,
        body_template: str,
        tone: str = "professional",
        placeholders: Optional[List[str]] = None,
    ) -> str:
        """Save a reusable email template. Returns template ID."""
        async with self.async_session_maker() as session:
            tmpl = EmailTemplate(
                user_id=user_id,
                name=name,
                subject_template=subject_template,
                body_template=body_template,
                tone=tone,
                placeholders=placeholders or [],
            )
            session.add(tmpl)
            await session.commit()
            return str(tmpl.id)

    async def get_email_templates(
        self, user_id: str, name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve email templates for a user."""
        async with self.async_session_maker() as session:
            query = select(EmailTemplate).where(EmailTemplate.user_id == user_id)
            if name:
                query = query.where(EmailTemplate.name == name)
            query = query.order_by(desc(EmailTemplate.created_at))
            result = await session.execute(query)
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "name": r.name,
                    "tone": r.tone,
                    "subject_template": r.subject_template,
                    "body_template": r.body_template,
                    "placeholders": r.placeholders or [],
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    # ── Exam Operations ─────────────────────────────────────────────────────

    async def store_exam(
        self,
        user_id: str,
        subject: str,
        exam_date: date,
        start_time: Optional[time] = None,
        location: Optional[str] = None,
        exam_type: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """Store an upcoming exam. Returns exam ID."""
        async with self.async_session_maker() as session:
            exam = Exam(
                user_id=user_id,
                subject=subject,
                exam_date=exam_date,
                start_time=start_time,
                location=location,
                exam_type=exam_type or "exam",
                notes=notes,
            )
            session.add(exam)
            await session.commit()
            return str(exam.id)

    async def retrieve_exams(
        self,
        user_id: str,
        upcoming_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve exam schedule. upcoming_only=True returns only future exams."""
        from datetime import date as date_cls
        async with self.async_session_maker() as session:
            query = select(Exam).where(Exam.user_id == user_id)
            if upcoming_only:
                query = query.where(Exam.exam_date >= date_cls.today())
            query = query.order_by(Exam.exam_date, Exam.start_time)
            result = await session.execute(query)
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "subject": r.subject,
                    "exam_date": r.exam_date.isoformat(),
                    "start_time": r.start_time.isoformat() if r.start_time else None,
                    "location": r.location,
                    "exam_type": r.exam_type,
                    "notes": r.notes,
                }
                for r in rows
            ]


    # ── Plan Operations ─────────────────────────────────────────────────────

    async def store_plan(
        self,
        user_id: str,
        plan_date: date,
        title: str,
        description: Optional[str] = None,
        priority: str = "medium",
    ) -> str:
        """Store a daily plan/task. Returns plan ID."""
        async with self.async_session_maker() as session:
            plan = Plan(
                user_id=user_id,
                plan_date=plan_date,
                title=title,
                description=description,
                priority=priority,
            )
            session.add(plan)
            await session.commit()
            return str(plan.id)

    async def retrieve_plans(
        self,
        user_id: str,
        plan_date: Optional[date] = None,
        include_done: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve plans. If plan_date is None, returns all future plans."""
        from datetime import date as date_cls
        async with self.async_session_maker() as session:
            query = select(Plan).where(Plan.user_id == user_id)
            if plan_date:
                query = query.where(Plan.plan_date == plan_date)
            else:
                query = query.where(Plan.plan_date >= date_cls.today())
            if not include_done:
                query = query.where(Plan.is_done == False)
            query = query.order_by(Plan.plan_date, Plan.priority)
            result = await session.execute(query)
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "plan_date": r.plan_date.isoformat(),
                    "title": r.title,
                    "description": r.description,
                    "priority": r.priority,
                    "is_done": r.is_done,
                }
                for r in rows
            ]

    async def mark_plan_done(self, plan_id: str, user_id: str) -> bool:
        """Mark a plan as completed. Returns True if found and updated."""
        from sqlalchemy import update as sa_update
        import uuid as uuid_mod
        async with self.async_session_maker() as session:
            result = await session.execute(
                sa_update(Plan)
                .where(and_(Plan.id == uuid_mod.UUID(plan_id), Plan.user_id == user_id))
                .values(is_done=True)
            )
            await session.commit()
            return result.rowcount > 0

    # ── UserProfile Operations ───────────────────────────────────────────────

    # ── Sensitive data keywords — never store these keys or values ────────────
    _SENSITIVE_KEYS = frozenset({
        "password", "passwd", "pin", "secret", "token", "api_key", "apikey",
        "credit_card", "card_number", "cvv", "ssn", "social_security",
        "bank_account", "routing_number", "private_key",
    })

    def _is_sensitive(self, key: str, value: str) -> bool:
        """Return True if the key or value looks like sensitive credential data."""
        key_lower = key.lower()
        if any(s in key_lower for s in self._SENSITIVE_KEYS):
            return True
        # Crude value heuristic: reject values that look like raw passwords/tokens
        # (long alphanum strings with no spaces ≥ 20 chars that aren't URLs)
        if len(value) >= 20 and " " not in value and not value.startswith("http"):
            # Allow UUIDs, emails, dates — only block opaque credential-shaped strings
            import re
            if re.fullmatch(r"[A-Za-z0-9_\-\.]{20,}", value):
                return True
        return False

    async def save_profile_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        source: str = "explicit",
        confidence: float = 1.0,
        consent_level: str = "explicit",
    ) -> str:
        """
        Upsert a profile fact. If (user_id, key) already exists, update value + metadata.
        Returns the record ID. Blocks sensitive credential data silently.
        """
        if self._is_sensitive(key, value):
            return ""

        from sqlalchemy.dialects.postgresql import insert as pg_insert
        async with self.async_session_maker() as session:
            stmt = pg_insert(UserProfile).values(
                id=uuid.uuid4(),
                user_id=user_id,
                key=key,
                value=value,
                source=source,
                confidence=confidence,
                consent_level=consent_level,
            ).on_conflict_do_update(
                constraint="uq_user_profile_key",
                set_=dict(
                    value=value,
                    source=source,
                    confidence=confidence,
                    consent_level=consent_level,
                    updated_at=func.now(),
                ),
            ).returning(UserProfile.id)
            result = await session.execute(stmt)
            await session.commit()
            row = result.fetchone()
            return str(row[0]) if row else ""

    async def get_profile_facts(
        self,
        user_id: str,
        key: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Return all profile facts for a user, optionally filtered by key."""
        async with self.async_session_maker() as session:
            query = select(UserProfile).where(
                and_(
                    UserProfile.user_id == user_id,
                    UserProfile.confidence >= min_confidence,
                )
            )
            if key:
                query = query.where(UserProfile.key == key)
            query = query.order_by(UserProfile.updated_at.desc())
            result = await session.execute(query)
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "key": r.key,
                    "value": r.value,
                    "source": r.source,
                    "confidence": r.confidence,
                    "consent_level": r.consent_level,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in rows
            ]

    async def forget_profile_fact(self, user_id: str, key: str) -> bool:
        """Delete a single profile fact. Returns True if a row was deleted."""
        from sqlalchemy import delete
        async with self.async_session_maker() as session:
            result = await session.execute(
                delete(UserProfile).where(
                    and_(UserProfile.user_id == user_id, UserProfile.key == key)
                )
            )
            await session.commit()
            return result.rowcount > 0

    async def forget_all_profile(self, user_id: str) -> int:
        """Delete all profile facts for a user. Returns number of rows deleted."""
        from sqlalchemy import delete
        async with self.async_session_maker() as session:
            result = await session.execute(
                delete(UserProfile).where(UserProfile.user_id == user_id)
            )
            await session.commit()
            return result.rowcount

    # ── EpisodicMemory Operations ────────────────────────────────────────────

    async def store_episode(
        self,
        user_id: str,
        session_id: str,
        user_summary: str,
        agent_summary: str,
        agent_used: Optional[str] = None,
        intent: Optional[str] = None,
        outcome: str = "success",
    ) -> str:
        """Store a single conversation turn as an episodic memory."""
        async with self.async_session_maker() as session:
            ep = EpisodicMemory(
                user_id=user_id,
                session_id=session_id,
                user_summary=user_summary[:300],
                agent_summary=agent_summary[:300],
                agent_used=agent_used,
                intent=intent,
                outcome=outcome,
            )
            session.add(ep)
            await session.commit()
            return str(ep.id)

    async def get_recent_episodes(
        self,
        user_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the N most recent episodes for a user (newest first)."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(EpisodicMemory)
                .where(EpisodicMemory.user_id == user_id)
                .order_by(desc(EpisodicMemory.created_at))
                .limit(limit)
            )
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "session_id": r.session_id,
                    "user_summary": r.user_summary,
                    "agent_summary": r.agent_summary,
                    "agent_used": r.agent_used,
                    "intent": r.intent,
                    "outcome": r.outcome,
                    "created_at": r.created_at.isoformat(),
                }
                for r in reversed(rows)  # chronological order
            ]

    # ── Tool Memory Operations (Fix 3: Cross-Session Tool Learning) ─────────

    async def save_tool_memory(
        self,
        user_id: str,
        agent_name: str,
        tool_name: str,
        inputs_summary: str,
        outcome_quality: str,
        key_insight: str,
    ) -> str:
        """
        Persist a tool-use outcome so the agent can recall what worked.
        outcome_quality: "good" | "poor" | "failed"
        Only 'good' records are retrieved as future hints.
        """
        async with self.async_session_maker() as session:
            record = ToolMemory(
                user_id=user_id,
                agent_name=agent_name,
                tool_name=tool_name,
                inputs_summary=inputs_summary[:500],
                outcome_quality=outcome_quality,
                key_insight=key_insight[:500],
            )
            session.add(record)
            await session.commit()
            return str(record.id)

    async def get_tool_memories(
        self,
        user_id: str,
        agent_name: str,
        tool_names: List[str],
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve past successful tool-use records for the given tools.
        Returns newest-first, filtered to outcome_quality='good' only.
        """
        if not tool_names:
            return []
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(ToolMemory)
                .where(
                    and_(
                        ToolMemory.user_id == user_id,
                        ToolMemory.agent_name == agent_name,
                        ToolMemory.tool_name.in_(tool_names),
                        ToolMemory.outcome_quality == "good",
                    )
                )
                .order_by(desc(ToolMemory.created_at))
                .limit(limit)
            )
            rows = result.scalars().all()
            return [
                {
                    "tool_name": r.tool_name,
                    "inputs_summary": r.inputs_summary,
                    "key_insight": r.key_insight,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    # ── Turn counter ────────────────────────────────────────────────────────

    async def get_session_turn_count(self, user_id: str, session_id: str) -> int:
        """Count how many user messages exist in this session (= number of turns taken)."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(func.count(ChatHistory.id)).where(
                    and_(
                        ChatHistory.user_id == user_id,
                        ChatHistory.session_id == session_id,
                        ChatHistory.role == "user",
                    )
                )
            )
            return result.scalar() or 0

    # ── M5: Agent Playbook Operations ───────────────────────────────────────

    async def save_playbook(
        self,
        agent_name: str,
        version: str,
        prompt: str,
        notes: Optional[str] = None,
        set_active: bool = True,
    ) -> str:
        """
        Save a new versioned prompt for an agent.
        If set_active=True, deactivates all other versions for that agent first.
        Returns the playbook ID.
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy import update as sa_update
        async with self.async_session_maker() as session:
            if set_active:
                # Deactivate all existing versions for this agent
                await session.execute(
                    sa_update(AgentPlaybook)
                    .where(AgentPlaybook.agent_name == agent_name)
                    .values(is_active=False)
                )

            stmt = pg_insert(AgentPlaybook).values(
                id=uuid.uuid4(),
                agent_name=agent_name,
                version=version,
                prompt=prompt,
                is_active=set_active,
                notes=notes,
            ).on_conflict_do_update(
                constraint="uq_playbook_agent_version",
                set_=dict(prompt=prompt, is_active=set_active, notes=notes),
            ).returning(AgentPlaybook.id)

            result = await session.execute(stmt)
            await session.commit()
            row = result.fetchone()
            return str(row[0]) if row else ""

    async def get_active_playbook(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Return the currently active prompt for an agent.
        Returns None if no active playbook exists (agent falls back to hard-coded prompt).
        """
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(AgentPlaybook)
                .where(
                    and_(
                        AgentPlaybook.agent_name == agent_name,
                        AgentPlaybook.is_active == True,
                    )
                )
                .order_by(desc(AgentPlaybook.created_at))
                .limit(1)
            )
            row = result.scalars().first()
            if not row:
                return None
            return {
                "id": str(row.id),
                "agent_name": row.agent_name,
                "version": row.version,
                "prompt": row.prompt,
                "is_active": row.is_active,
                "notes": row.notes,
                "created_at": row.created_at.isoformat(),
            }

    async def list_playbooks(self, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all playbook versions, optionally filtered by agent name."""
        async with self.async_session_maker() as session:
            query = select(AgentPlaybook)
            if agent_name:
                query = query.where(AgentPlaybook.agent_name == agent_name)
            query = query.order_by(AgentPlaybook.agent_name, desc(AgentPlaybook.created_at))
            result = await session.execute(query)
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "agent_name": r.agent_name,
                    "version": r.version,
                    "prompt": r.prompt[:200] + "..." if len(r.prompt) > 200 else r.prompt,
                    "is_active": r.is_active,
                    "notes": r.notes,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]


# Singleton instance
short_term_memory = ShortTermMemory()
