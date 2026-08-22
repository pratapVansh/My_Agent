"""
The uploaded timetable documents, and which one is live.

Separate from `AcademicRepository` because it answers a different question. That
one stores *classes*; this one stores the *document a set of classes came from*,
which is what makes "where did Thursday's lab come from" answerable and what
lets a new semester replace an old one as a unit.

Activation is a group operation on purpose. Deactivating rows one at a time
leaves a window in which half a timetable is live, and a question asked inside
that window gets an answer that is neither the old schedule nor the new one.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, select, update as sa_update

from app.db.session import async_session_maker
from app.domain.models import ScheduleUpload, Timetable

logger = logging.getLogger(__name__)


class ScheduleRepository:
    """Timetable document provenance and version selection."""

    def __init__(self):
        self.async_session_maker = async_session_maker

    async def record_upload(
        self,
        user_id: str,
        *,
        filename: str,
        content_hash: str,
        source_title: Optional[str] = None,
        semester: Optional[str] = None,
        page_count: Optional[int] = None,
        entry_count: int = 0,
        parse_notes: Optional[str] = None,
        ingest_method: str = "pdf_parser",
    ) -> str:
        """Record a document and return its id."""
        async with self.async_session_maker() as session:
            upload = ScheduleUpload(
                user_id=user_id,
                filename=filename,
                content_hash=content_hash,
                source_title=source_title,
                semester=semester,
                page_count=page_count,
                entry_count=entry_count,
                parse_notes=parse_notes,
                ingest_method=ingest_method,
                is_active=True,
            )
            session.add(upload)
            await session.commit()
            return str(upload.id)

    async def find_by_hash(
        self, user_id: str, content_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        An earlier upload of this exact file, if there is one.

        Uploading the same PDF twice should not produce a duplicate timetable —
        the second upload is almost always someone checking whether it worked.
        """
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(ScheduleUpload).where(and_(
                    ScheduleUpload.user_id == user_id,
                    ScheduleUpload.content_hash == content_hash,
                ))
            )
            row = result.scalars().first()
            return _as_dict(row) if row else None

    async def deactivate_all(self, user_id: str) -> int:
        """Retire every live document for this user. Returns how many."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                sa_update(ScheduleUpload)
                .where(and_(
                    ScheduleUpload.user_id == user_id,
                    ScheduleUpload.is_active == True,
                ))
                .values(is_active=False)
            )
            await session.commit()
            return result.rowcount or 0

    async def replace_active_timetable(
        self,
        user_id: str,
        *,
        filename: str,
        content_hash: str,
        valid_entries: List[Dict[str, Any]],
        skipped: Optional[List[Dict[str, str]]] = None,
        source_title: Optional[str] = None,
        semester: Optional[str] = None,
        page_count: Optional[int] = None,
        parse_notes: Optional[str] = None,
        ingest_method: str = "pdf_parser",
    ) -> Dict[str, Any]:
        """
        Retire the active timetable and activate a new one, as one transaction.

        This is the group operation the class docstring describes and the
        other methods here do not, on their own, deliver: `deactivate_all` and
        `record_upload` are each their own commit, so calling them in sequence
        still leaves a real window — a crash between them — where a user has
        no active timetable at all, or worse, a request lands between the old
        rows being deactivated and the new ones being inserted. Everything that
        writes here — deactivating the old `Timetable` rows, deactivating the
        old `ScheduleUpload`, recording the new one, inserting the new rows —
        happens in one session and one commit. Any error before that commit
        rolls the whole thing back and leaves the previous timetable exactly as
        it was; nothing here can leave old and new classes both active.

        `valid_entries` must already be validated (see
        `timetable_tool.validate_entries`) — this is the second gate, not the
        first: an empty list is refused before the transaction opens, so a
        bad upload can never activate a timetable with zero classes in it.
        """
        skipped = skipped or []
        if not valid_entries:
            return {
                "success": False,
                "reason": "no_valid_entries",
                "upload_id": None,
                "stored_count": 0,
                "deactivated_rows": 0,
                "deactivated_uploads": 0,
                "skipped_count": len(skipped),
                "skipped": skipped[:10],
            }

        async with self.async_session_maker() as session:
            deactivated_rows = await session.execute(
                sa_update(Timetable)
                .where(and_(
                    Timetable.user_id == user_id,
                    Timetable.is_active == True,
                ))
                .values(is_active=False)
            )
            deactivated_uploads = await session.execute(
                sa_update(ScheduleUpload)
                .where(and_(
                    ScheduleUpload.user_id == user_id,
                    ScheduleUpload.is_active == True,
                ))
                .values(is_active=False)
            )

            upload = ScheduleUpload(
                user_id=user_id,
                filename=filename,
                content_hash=content_hash,
                source_title=source_title,
                semester=semester,
                page_count=page_count,
                entry_count=len(valid_entries),
                parse_notes=parse_notes,
                ingest_method=ingest_method,
                is_active=True,
            )
            session.add(upload)
            # Assigns upload.id within this same transaction, without
            # committing it — the new rows below must link to an id that
            # still rolls back together with everything else on failure.
            await session.flush()

            rows = [
                Timetable(
                    user_id=user_id,
                    day_of_week=row["day_of_week"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    subject=row["subject"],
                    location=row["location"],
                    instructor=row["instructor"],
                    meta_data=row["metadata"],
                    schedule_upload_id=upload.id,
                )
                for row in valid_entries
            ]
            session.add_all(rows)
            await session.commit()

            return {
                "success": True,
                "reason": None,
                "upload_id": str(upload.id),
                "stored_count": len(rows),
                "deactivated_rows": deactivated_rows.rowcount or 0,
                "deactivated_uploads": deactivated_uploads.rowcount or 0,
                "skipped_count": len(skipped),
                "skipped": skipped[:10],
            }

    async def active_upload(self, user_id: str) -> Optional[Dict[str, Any]]:
        """The document the current timetable came from, if any."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(ScheduleUpload)
                .where(and_(
                    ScheduleUpload.user_id == user_id,
                    ScheduleUpload.is_active == True,
                ))
                .order_by(desc(ScheduleUpload.created_at))
                .limit(1)
            )
            row = result.scalars().first()
            return _as_dict(row) if row else None

    async def list_uploads(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Upload history, newest first — so a replaced semester stays visible."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(ScheduleUpload)
                .where(ScheduleUpload.user_id == user_id)
                .order_by(desc(ScheduleUpload.created_at))
                .limit(limit)
            )
            return [_as_dict(row) for row in result.scalars().all()]

    async def delete_upload_rows(self, user_id: str, upload_id: str) -> int:
        """
        Deactivate the classes produced by one document.

        Soft, like `clear_timetable`: a replaced timetable stays readable for
        anyone auditing what the assistant used to say.
        """
        async with self.async_session_maker() as session:
            result = await session.execute(
                sa_update(Timetable)
                .where(and_(
                    Timetable.user_id == user_id,
                    Timetable.schedule_upload_id == uuid.UUID(upload_id),
                ))
                .values(is_active=False)
            )
            await session.commit()
            return result.rowcount or 0


def _as_dict(row: ScheduleUpload) -> Dict[str, Any]:
    return {
        "id": str(row.id),
        "user_id": row.user_id,
        "filename": row.filename,
        "content_hash": row.content_hash,
        "source_title": row.source_title,
        "semester": row.semester,
        "page_count": row.page_count,
        "entry_count": row.entry_count,
        "parse_notes": row.parse_notes,
        "ingest_method": row.ingest_method,
        "is_active": bool(row.is_active),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


schedule_repository = ScheduleRepository()

__all__ = ["ScheduleRepository", "schedule_repository"]
