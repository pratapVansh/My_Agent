"""
PostgreSQL adapter for the `EventQueue` port.

Claiming uses `SELECT … FOR UPDATE SKIP LOCKED` inside the same transaction as
the status update. That is what makes the queue safe under concurrent workers:
`SKIP LOCKED` lets a second worker step over rows the first has locked instead
of blocking on them or — far worse — reading them and processing the same
conversation twice.
"""
from __future__ import annotations

from typing import List, Optional, Sequence
from uuid import UUID
import logging

from sqlalchemy import and_, func, select, update as sa_update

from app.db.session import async_session_maker
from app.memory.events import (
    MAX_ATTEMPTS,
    EventStatus,
    EventType,
    GroupReadiness,
    MemoryEvent,
)
from app.memory.models import MemoryEventORM
from app.memory.record import utcnow

logger = logging.getLogger(__name__)


def _to_event(row: MemoryEventORM) -> MemoryEvent:
    return MemoryEvent(
        id=row.id,
        owner_id=row.owner_id,
        event_type=EventType(row.event_type),
        group_key=row.group_key,
        payload=row.payload or {},
        status=EventStatus(row.status),
        attempts=row.attempts,
        last_error=row.last_error,
        created_at=row.created_at,
        claimed_at=row.claimed_at,
        processed_at=row.processed_at,
    )


class PostgresEventQueue:
    """`EventQueue` backed by the shared async engine."""

    def __init__(self, session_maker=None):
        self.async_session_maker = session_maker or async_session_maker

    async def enqueue(self, event: MemoryEvent) -> MemoryEvent:
        async with self.async_session_maker() as session:
            session.add(MemoryEventORM(
                id=event.id,
                owner_id=event.owner_id,
                event_type=event.event_type.value,
                group_key=event.group_key,
                payload=event.payload,
                status=event.status.value,
                attempts=event.attempts,
                created_at=event.created_at,
            ))
            await session.commit()
        return event

    async def ready_groups(
        self, *, batch_size: int, idle_seconds: float, limit: int = 20
    ) -> List[GroupReadiness]:
        """
        Aggregate pending work per group, then filter for readiness.

        Deliberately an aggregate query *before* claiming rather than
        claim-then-release: releasing unready events back would churn their
        attempt counts and make the retry ceiling meaningless.
        """
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(
                    MemoryEventORM.owner_id,
                    MemoryEventORM.group_key,
                    func.count(MemoryEventORM.id),
                    func.min(MemoryEventORM.created_at),
                )
                .where(MemoryEventORM.status == EventStatus.PENDING.value)
                .group_by(MemoryEventORM.owner_id, MemoryEventORM.group_key)
                .order_by(func.min(MemoryEventORM.created_at))
                .limit(limit)
            )
            groups = [
                GroupReadiness(
                    owner_id=row[0], group_key=row[1], pending=row[2], oldest=row[3]
                )
                for row in result.all()
            ]

        now = utcnow()
        return [
            group for group in groups
            if group.is_ready(
                batch_size=batch_size, idle_seconds=idle_seconds, now=now
            )
        ]

    async def claim_group(self, group_key: str, limit: int = 50) -> List[MemoryEvent]:
        async with self.async_session_maker() as session:
            # One transaction: the lock taken by the SELECT is held until the
            # UPDATE commits, so no other worker can claim these rows.
            locked = await session.execute(
                select(MemoryEventORM.id)
                .where(
                    and_(
                        MemoryEventORM.group_key == group_key,
                        MemoryEventORM.status == EventStatus.PENDING.value,
                    )
                )
                .order_by(MemoryEventORM.created_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            event_ids = [row[0] for row in locked.all()]
            if not event_ids:
                await session.rollback()
                return []

            await session.execute(
                sa_update(MemoryEventORM)
                .where(MemoryEventORM.id.in_(event_ids))
                .values(
                    status=EventStatus.PROCESSING.value,
                    claimed_at=utcnow(),
                    attempts=MemoryEventORM.attempts + 1,
                )
            )
            rows = await session.execute(
                select(MemoryEventORM)
                .where(MemoryEventORM.id.in_(event_ids))
                .order_by(MemoryEventORM.created_at)
            )
            claimed = [_to_event(row) for row in rows.scalars().all()]
            await session.commit()
            return claimed

    async def mark_done(
        self, event_ids: Sequence[UUID], status: EventStatus = EventStatus.DONE
    ) -> None:
        if not event_ids:
            return
        async with self.async_session_maker() as session:
            await session.execute(
                sa_update(MemoryEventORM)
                .where(MemoryEventORM.id.in_(list(event_ids)))
                .values(status=status.value, processed_at=utcnow(), last_error=None)
            )
            await session.commit()

    async def mark_failed(self, event_ids: Sequence[UUID], error: str) -> None:
        """
        Return events to PENDING for retry, or park them once exhausted.

        Discarding on first failure would mean a transient Groq outage silently
        loses everything the user said while it was down.
        """
        if not event_ids:
            return
        async with self.async_session_maker() as session:
            await session.execute(
                sa_update(MemoryEventORM)
                .where(
                    and_(
                        MemoryEventORM.id.in_(list(event_ids)),
                        MemoryEventORM.attempts < MAX_ATTEMPTS,
                    )
                )
                .values(status=EventStatus.PENDING.value, last_error=error[:500])
            )
            await session.execute(
                sa_update(MemoryEventORM)
                .where(
                    and_(
                        MemoryEventORM.id.in_(list(event_ids)),
                        MemoryEventORM.attempts >= MAX_ATTEMPTS,
                    )
                )
                .values(
                    status=EventStatus.FAILED.value,
                    processed_at=utcnow(),
                    last_error=error[:500],
                )
            )
            await session.commit()

    async def pending_count(self, owner_id: Optional[str] = None) -> int:
        async with self.async_session_maker() as session:
            query = select(func.count(MemoryEventORM.id)).where(
                MemoryEventORM.status == EventStatus.PENDING.value
            )
            if owner_id:
                query = query.where(MemoryEventORM.owner_id == owner_id)
            result = await session.execute(query)
            return result.scalar() or 0


# Singleton instance
postgres_event_queue = PostgresEventQueue()
