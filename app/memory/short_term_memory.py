"""
PostgreSQL Short-term Memory Implementation.
Stores chat history, attendance, and timetable using async SQLAlchemy.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, and_, desc
from typing import List, Dict, Any, Optional
from datetime import date, time, datetime
import uuid

from app.config import settings
from app.memory.models import Base, ChatHistory, Attendance, Timetable


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


# Singleton instance
short_term_memory = ShortTermMemory()
