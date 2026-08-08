"""
In-memory `EventQueue`, for tests and local development.

Mirrors the Postgres adapter's semantics — including retry-until-exhausted on
failure — because a fake that discards on first failure would hide exactly the
bug that matters.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from uuid import UUID
import copy

from app.memory.events import (
    MAX_ATTEMPTS,
    EventStatus,
    GroupReadiness,
    MemoryEvent,
)
from app.memory.record import utcnow


class InMemoryEventQueue:
    """`EventQueue` backed by a dict."""

    def __init__(self):
        self._events: Dict[UUID, MemoryEvent] = {}

    async def enqueue(self, event: MemoryEvent) -> MemoryEvent:
        self._events[event.id] = copy.deepcopy(event)
        return event

    async def ready_groups(
        self, *, batch_size: int, idle_seconds: float, limit: int = 20
    ) -> List[GroupReadiness]:
        buckets: Dict[tuple, List[MemoryEvent]] = {}
        for event in self._events.values():
            if event.status is not EventStatus.PENDING:
                continue
            buckets.setdefault((event.owner_id, event.group_key), []).append(event)

        groups = [
            GroupReadiness(
                owner_id=owner_id,
                group_key=group_key,
                pending=len(events),
                oldest=min(e.created_at for e in events),
            )
            for (owner_id, group_key), events in buckets.items()
        ]
        groups.sort(key=lambda g: g.oldest)

        now = utcnow()
        ready = [
            g for g in groups
            if g.is_ready(batch_size=batch_size, idle_seconds=idle_seconds, now=now)
        ]
        return ready[:limit]

    async def claim_group(self, group_key: str, limit: int = 50) -> List[MemoryEvent]:
        pending = [
            event for event in self._events.values()
            if event.group_key == group_key and event.status is EventStatus.PENDING
        ]
        pending.sort(key=lambda e: e.created_at)
        claimed: List[MemoryEvent] = []
        for event in pending[:limit]:
            event.status = EventStatus.PROCESSING
            event.attempts += 1
            event.claimed_at = utcnow()
            claimed.append(copy.deepcopy(event))
        return claimed

    async def mark_done(
        self, event_ids: Sequence[UUID], status: EventStatus = EventStatus.DONE
    ) -> None:
        for event_id in event_ids:
            event = self._events.get(event_id)
            if event is not None:
                event.status = status
                event.processed_at = utcnow()
                event.last_error = None

    async def mark_failed(self, event_ids: Sequence[UUID], error: str) -> None:
        for event_id in event_ids:
            event = self._events.get(event_id)
            if event is None:
                continue
            event.last_error = error[:500]
            if event.attempts >= MAX_ATTEMPTS:
                event.status = EventStatus.FAILED
                event.processed_at = utcnow()
            else:
                event.status = EventStatus.PENDING

    async def pending_count(self, owner_id: Optional[str] = None) -> int:
        return sum(
            1 for event in self._events.values()
            if event.status is EventStatus.PENDING
            and (owner_id is None or event.owner_id == owner_id)
        )

    # ── test helpers ────────────────────────────────────────────────────

    def all_events(self) -> List[MemoryEvent]:
        return list(self._events.values())

    def clear(self) -> None:
        self._events.clear()
