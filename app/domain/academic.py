"""
Academic records: attendance, timetable, exams, and plans.

Extracted verbatim from `ShortTermMemory`; behaviour is unchanged. These are
application records rather than memories — see `app/domain/models.py`.
"""
from datetime import date, time
from typing import Any, Dict, List, Optional
import logging
import uuid

from sqlalchemy import and_, delete, desc, select, update as sa_update

from app.db.session import async_session_maker
from app.domain.models import Attendance, Exam, Plan, Timetable

logger = logging.getLogger(__name__)

# Safety ceiling on unbounded history reads so a long-lived account cannot
# degrade a request into a full-table scan.
_MAX_ATTENDANCE_ROWS = 5000


class AcademicRepository:
    """Attendance, timetable, exam, and plan persistence."""

    def __init__(self):
        self.async_session_maker = async_session_maker

    # ── Attendance ──────────────────────────────────────────────────────

    async def store_attendance(
        self,
        user_id: str,
        date: date,
        subject: str,
        status: str,
        notes: Optional[str] = None,
    ) -> str:
        """
        Store an attendance record.

        Idempotent on (user_id, date, subject) — recording the same class twice
        updates the existing row rather than creating a duplicate.
        """
        return await self.upsert_attendance(
            user_id=user_id, date=date, subject=subject, status=status, notes=notes
        )

    async def upsert_attendance(
        self,
        user_id: str,
        date: date,
        subject: str,
        status: str,
        notes: Optional[str] = None,
    ) -> str:
        """
        Insert or update the attendance record for (user_id, date, subject).

        Attendance is a fact about a specific class on a specific day, so
        recording it twice must not create two rows: duplicates inflate the
        'total classes' denominator and silently corrupt the attendance
        percentages the academic agent bases its advice on.

        Returns the record ID.
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        async with self.async_session_maker() as session:
            stmt = pg_insert(Attendance).values(
                id=uuid.uuid4(),
                user_id=user_id,
                date=date,
                subject=subject,
                status=status,
                notes=notes,
            ).on_conflict_do_update(
                index_elements=["user_id", "date", "subject"],
                set_=dict(status=status, notes=notes),
            ).returning(Attendance.id)

            try:
                result = await session.execute(stmt)
                await session.commit()
                row = result.fetchone()
                return str(row[0]) if row else ""
            except Exception:
                # Falls back to a plain insert when the unique index is absent
                # (pre-migration database) so existing deployments keep working.
                await session.rollback()
                logger.debug(
                    "Attendance upsert unavailable (missing unique index?); "
                    "falling back to insert. Run the migration in AUDIT_REPORT.md."
                )
                attendance = Attendance(
                    user_id=user_id,
                    date=date,
                    subject=subject,
                    status=status,
                    notes=notes,
                )
                session.add(attendance)
                await session.commit()
                return str(attendance.id)

    async def retrieve_attendance(
        self,
        user_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        subject: Optional[str] = None,
        limit: int = _MAX_ATTENDANCE_ROWS,
    ) -> List[Dict[str, Any]]:
        """Retrieve attendance records, newest first (bounded by `limit`)."""
        async with self.async_session_maker() as session:
            query = select(Attendance).where(Attendance.user_id == user_id)

            if start_date:
                query = query.where(Attendance.date >= start_date)
            if end_date:
                query = query.where(Attendance.date <= end_date)
            if subject:
                query = query.where(Attendance.subject == subject)

            query = query.order_by(desc(Attendance.date)).limit(limit)

            result = await session.execute(query)
            records = result.scalars().all()

            return [
                {
                    "id": str(record.id),
                    "date": record.date.isoformat(),
                    "subject": record.subject,
                    "status": record.status,
                    "notes": record.notes,
                }
                for record in records
            ]

    # ── Timetable ───────────────────────────────────────────────────────

    async def store_timetable_entry(
        self,
        user_id: str,
        day_of_week: int,
        start_time: time,
        end_time: time,
        subject: str,
        location: Optional[str] = None,
        instructor: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a timetable entry. day_of_week is 0=Monday .. 6=Sunday."""
        async with self.async_session_maker() as session:
            entry = Timetable(
                user_id=user_id,
                day_of_week=day_of_week,
                start_time=start_time,
                end_time=end_time,
                subject=subject,
                location=location,
                instructor=instructor,
                meta_data=metadata,  # Uses meta_data column
            )
            session.add(entry)
            await session.commit()
            return str(entry.id)

    async def retrieve_timetable(
        self,
        user_id: str,
        day_of_week: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve active timetable entries, optionally for one day."""
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
                    "metadata": entry.meta_data,  # Read from meta_data column
                }
                for entry in entries
            ]

    async def clear_timetable(self, user_id: str) -> int:
        """
        Soft-delete all active timetable entries for a user by setting is_active=False.
        Returns the number of entries deactivated.
        Used before uploading a new semester timetable so old entries don't pollute results.
        """
        async with self.async_session_maker() as session:
            result = await session.execute(
                sa_update(Timetable)
                .where(and_(Timetable.user_id == user_id, Timetable.is_active == True))
                .values(is_active=False)
            )
            await session.commit()
            return result.rowcount

    # ── Exams ───────────────────────────────────────────────────────────

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

    # ── Plans ───────────────────────────────────────────────────────────

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
        async with self.async_session_maker() as session:
            result = await session.execute(
                sa_update(Plan)
                .where(and_(Plan.id == uuid.UUID(plan_id), Plan.user_id == user_id))
                .values(is_done=True)
            )
            await session.commit()
            return result.rowcount > 0


# Singleton instance
academic_repository = AcademicRepository()
