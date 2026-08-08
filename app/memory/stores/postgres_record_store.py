"""
PostgreSQL adapter for the `RecordStore` port.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID
import logging

from sqlalchemy import and_, desc, func, select, update as sa_update
from sqlalchemy.exc import IntegrityError

from app.db.session import async_session_maker
from app.memory.kinds import (
    EmbeddingStatus,
    MemoryKind,
    RecordStatus,
    Sensitivity,
    SourceType,
    Visibility,
)
from app.memory.models import MemoryRecordORM
from app.memory.record import MemoryRecord

logger = logging.getLogger(__name__)


def _to_record(row: MemoryRecordORM) -> MemoryRecord:
    """ORM row → domain object."""
    return MemoryRecord(
        id=row.id,
        owner_id=row.owner_id,
        kind=MemoryKind(row.kind),
        content=row.content,
        structured=row.structured or {},
        importance=row.importance,
        confidence=row.confidence,
        pinned=row.pinned,
        occurred_at=row.occurred_at,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        created_at=row.created_at,
        updated_at=row.updated_at,
        last_accessed_at=row.last_accessed_at,
        access_count=row.access_count,
        source_type=SourceType(row.source_type),
        source_ref=row.source_ref,
        derived_from=[UUID(x) for x in (row.derived_from or [])],
        supersedes_id=row.supersedes_id,
        version=row.version,
        visibility=Visibility(row.visibility),
        sensitivity=Sensitivity(row.sensitivity),
        status=RecordStatus(row.status),
        content_hash=row.content_hash,
        dedup_key=row.dedup_key,
        embedding_status=EmbeddingStatus(row.embedding_status),
    )


def _to_values(record: MemoryRecord) -> Dict[str, Any]:
    """Domain object → column values."""
    return {
        "id": record.id,
        "owner_id": record.owner_id,
        "kind": record.kind.value,
        "content": record.content,
        "structured": record.structured or {},
        "importance": record.importance,
        "confidence": record.confidence,
        "pinned": record.pinned,
        "occurred_at": record.occurred_at,
        "valid_from": record.valid_from,
        "valid_to": record.valid_to,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "last_accessed_at": record.last_accessed_at,
        "access_count": record.access_count,
        "source_type": record.source_type.value,
        "source_ref": record.source_ref,
        "derived_from": [str(x) for x in record.derived_from],
        "supersedes_id": record.supersedes_id,
        "version": record.version,
        "visibility": record.visibility.value,
        "sensitivity": record.sensitivity.value,
        "status": record.status.value,
        "content_hash": record.content_hash,
        "dedup_key": record.dedup_key,
        "embedding_status": record.embedding_status.value,
    }


class PostgresRecordStore:
    """`RecordStore` backed by the shared async engine."""

    def __init__(self, session_maker=None):
        self.async_session_maker = session_maker or async_session_maker

    # ── writes ──────────────────────────────────────────────────────────

    async def add(self, record: MemoryRecord) -> MemoryRecord:
        """
        Insert, or return the existing active duplicate.

        The pre-check is only a fast path; two concurrent writers can both pass
        it. The authoritative guard is the partial unique index on
        (owner_id, kind, content_hash) and the IntegrityError it raises, which
        turns a race into the same idempotent outcome rather than a duplicate.
        """
        existing = await self.find_by_content_hash(
            record.owner_id, record.kind, record.content_hash
        )
        if existing is not None:
            return existing

        async with self.async_session_maker() as session:
            row = MemoryRecordORM(**_to_values(record))
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                duplicate = await self.find_by_content_hash(
                    record.owner_id, record.kind, record.content_hash
                )
                if duplicate is not None:
                    logger.debug(
                        "Concurrent insert for owner=%s kind=%s resolved as duplicate",
                        record.owner_id, record.kind.value,
                    )
                    return duplicate
                raise
            return record

    async def add_many(self, records: Sequence[MemoryRecord]) -> List[MemoryRecord]:
        """
        Insert many, de-duplicating each.

        Deliberately one statement per record rather than a bulk insert: a bulk
        insert aborts the whole batch on a single duplicate, which would make
        the backfill non-resumable — the property that matters most about it.
        """
        return [await self.add(record) for record in records]

    async def supersede(self, old: MemoryRecord, new: MemoryRecord) -> MemoryRecord:
        """Close `old` and insert `new` in one transaction."""
        closed = old.superseded_by(new)
        async with self.async_session_maker() as session:
            await session.execute(
                sa_update(MemoryRecordORM)
                .where(MemoryRecordORM.id == old.id)
                .values(
                    status=closed.status.value,
                    valid_to=closed.valid_to,
                    updated_at=closed.updated_at,
                )
            )
            session.add(MemoryRecordORM(**_to_values(new)))
            await session.commit()
        return new

    async def mark_embedding(self, record_id: UUID, status: EmbeddingStatus) -> None:
        async with self.async_session_maker() as session:
            await session.execute(
                sa_update(MemoryRecordORM)
                .where(MemoryRecordORM.id == record_id)
                .values(embedding_status=status.value)
            )
            await session.commit()

    # ── reads ───────────────────────────────────────────────────────────

    async def get(self, owner_id: str, record_id: UUID) -> Optional[MemoryRecord]:
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(MemoryRecordORM).where(
                    and_(
                        MemoryRecordORM.id == record_id,
                        MemoryRecordORM.owner_id == owner_id,
                    )
                )
            )
            row = result.scalars().first()
            return _to_record(row) if row else None

    async def get_many(
        self, owner_id: str, record_ids: Sequence[UUID]
    ) -> List[MemoryRecord]:
        """
        Bulk hydration for retrieval.

        Missing ids are omitted rather than raising: a vector index can outlive
        the record it points at, and a stale hit must degrade the result set
        rather than fail the whole retrieval.
        """
        if not record_ids:
            return []
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(MemoryRecordORM).where(
                    and_(
                        MemoryRecordORM.owner_id == owner_id,
                        MemoryRecordORM.id.in_(list(record_ids)),
                    )
                )
            )
            return [_to_record(r) for r in result.scalars().all()]

    async def find_by_content_hash(
        self, owner_id: str, kind: MemoryKind, content_hash: str
    ) -> Optional[MemoryRecord]:
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(MemoryRecordORM).where(
                    and_(
                        MemoryRecordORM.owner_id == owner_id,
                        MemoryRecordORM.kind == kind.value,
                        MemoryRecordORM.content_hash == content_hash,
                        MemoryRecordORM.status == RecordStatus.ACTIVE.value,
                    )
                )
            )
            row = result.scalars().first()
            return _to_record(row) if row else None

    async def find_by_dedup_key(
        self, owner_id: str, dedup_key: str
    ) -> List[MemoryRecord]:
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(MemoryRecordORM)
                .where(
                    and_(
                        MemoryRecordORM.owner_id == owner_id,
                        MemoryRecordORM.dedup_key == dedup_key,
                        MemoryRecordORM.status == RecordStatus.ACTIVE.value,
                    )
                )
                .order_by(desc(MemoryRecordORM.created_at))
            )
            return [_to_record(r) for r in result.scalars().all()]

    async def list(
        self,
        owner_id: str,
        *,
        kinds: Optional[Sequence[MemoryKind]] = None,
        statuses: Optional[Sequence[RecordStatus]] = None,
        visibilities: Optional[Sequence[Visibility]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        statuses = statuses or [RecordStatus.ACTIVE]
        async with self.async_session_maker() as session:
            query = select(MemoryRecordORM).where(MemoryRecordORM.owner_id == owner_id)
            query = query.where(
                MemoryRecordORM.status.in_([s.value for s in statuses])
            )
            if kinds:
                query = query.where(MemoryRecordORM.kind.in_([k.value for k in kinds]))
            if visibilities:
                query = query.where(
                    MemoryRecordORM.visibility.in_([v.value for v in visibilities])
                )
            query = (
                query.order_by(desc(MemoryRecordORM.created_at))
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(query)
            return [_to_record(r) for r in result.scalars().all()]

    async def count(
        self,
        owner_id: Optional[str] = None,
        *,
        kinds: Optional[Sequence[MemoryKind]] = None,
        statuses: Optional[Sequence[RecordStatus]] = None,
    ) -> int:
        async with self.async_session_maker() as session:
            query = select(func.count(MemoryRecordORM.id))
            if owner_id:
                query = query.where(MemoryRecordORM.owner_id == owner_id)
            if kinds:
                query = query.where(MemoryRecordORM.kind.in_([k.value for k in kinds]))
            if statuses:
                query = query.where(
                    MemoryRecordORM.status.in_([s.value for s in statuses])
                )
            result = await session.execute(query)
            return result.scalar() or 0

    async def pending_embeddings(self, limit: int = 100) -> List[MemoryRecord]:
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(MemoryRecordORM)
                .where(
                    and_(
                        MemoryRecordORM.embedding_status == EmbeddingStatus.PENDING.value,
                        MemoryRecordORM.status == RecordStatus.ACTIVE.value,
                    )
                )
                .order_by(MemoryRecordORM.created_at)
                .limit(limit)
            )
            return [_to_record(r) for r in result.scalars().all()]


    # ── lifecycle (Phase 5) ─────────────────────────────────────────────

    async def set_status(
        self, record_ids: Sequence[UUID], status: RecordStatus
    ) -> int:
        if not record_ids:
            return 0
        async with self.async_session_maker() as session:
            result = await session.execute(
                sa_update(MemoryRecordORM)
                .where(MemoryRecordORM.id.in_(list(record_ids)))
                .values(status=status.value, updated_at=func.now())
            )
            await session.commit()
            return result.rowcount

    async def hard_delete(self, record_ids: Sequence[UUID]) -> int:
        if not record_ids:
            return 0
        from sqlalchemy import delete as sa_delete

        async with self.async_session_maker() as session:
            result = await session.execute(
                sa_delete(MemoryRecordORM).where(
                    MemoryRecordORM.id.in_(list(record_ids))
                )
            )
            await session.commit()
            return result.rowcount

    async def find_derived_from(
        self, owner_id: str, record_id: UUID
    ) -> List[MemoryRecord]:
        """Records listing `record_id` in their provenance."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(MemoryRecordORM).where(
                    and_(
                        MemoryRecordORM.owner_id == owner_id,
                        MemoryRecordORM.derived_from.contains([str(record_id)]),
                    )
                )
            )
            return [_to_record(r) for r in result.scalars().all()]

    async def iter_active(
        self, *, limit: int = 500, offset: int = 0
    ) -> List[MemoryRecord]:
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(MemoryRecordORM)
                .where(MemoryRecordORM.status == RecordStatus.ACTIVE.value)
                .order_by(MemoryRecordORM.created_at)
                .limit(limit)
                .offset(offset)
            )
            return [_to_record(r) for r in result.scalars().all()]

    async def duplicate_dedup_keys(self, limit: int = 100) -> List[tuple]:
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(MemoryRecordORM.owner_id, MemoryRecordORM.dedup_key)
                .where(
                    and_(
                        MemoryRecordORM.dedup_key.isnot(None),
                        MemoryRecordORM.status == RecordStatus.ACTIVE.value,
                    )
                )
                .group_by(MemoryRecordORM.owner_id, MemoryRecordORM.dedup_key)
                .having(func.count(MemoryRecordORM.id) > 1)
                .limit(limit)
            )
            return [(row[0], row[1]) for row in result.all()]

    async def owner_activity(self) -> List[tuple]:
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(
                    MemoryRecordORM.owner_id,
                    func.max(MemoryRecordORM.created_at),
                ).group_by(MemoryRecordORM.owner_id)
            )
            return [(row[0], row[1]) for row in result.all()]


# Singleton instance
postgres_record_store = PostgresRecordStore()
