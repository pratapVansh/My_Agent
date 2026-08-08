"""
In-memory `RecordStore`, for tests and local development.

This exists so the layers above L0 can be tested without Postgres. It is held
to the same contract as the Postgres adapter — including the
(owner_id, kind, content_hash) uniqueness among active records — because a fake
that is more permissive than the real thing tests nothing useful.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from uuid import UUID
import copy

from app.memory.kinds import EmbeddingStatus, MemoryKind, RecordStatus, Visibility
from app.memory.record import MemoryRecord


class InMemoryRecordStore:
    """`RecordStore` backed by a dict. Not thread-safe; not intended to be."""

    def __init__(self):
        self._records: Dict[UUID, MemoryRecord] = {}

    # ── writes ──────────────────────────────────────────────────────────

    async def add(self, record: MemoryRecord) -> MemoryRecord:
        existing = await self.find_by_content_hash(
            record.owner_id, record.kind, record.content_hash
        )
        if existing is not None:
            return existing
        self._records[record.id] = copy.deepcopy(record)
        return record

    async def add_many(self, records: Sequence[MemoryRecord]) -> List[MemoryRecord]:
        return [await self.add(record) for record in records]

    async def supersede(self, old: MemoryRecord, new: MemoryRecord) -> MemoryRecord:
        closed = old.superseded_by(new)
        self._records[old.id] = copy.deepcopy(closed)
        self._records[new.id] = copy.deepcopy(new)
        return new

    async def mark_embedding(self, record_id: UUID, status: EmbeddingStatus) -> None:
        record = self._records.get(record_id)
        if record is not None:
            record.embedding_status = status

    # ── reads ───────────────────────────────────────────────────────────

    async def get(self, owner_id: str, record_id: UUID) -> Optional[MemoryRecord]:
        record = self._records.get(record_id)
        if record is None or record.owner_id != owner_id:
            return None
        return copy.deepcopy(record)

    async def get_many(
        self, owner_id: str, record_ids: Sequence[UUID]
    ) -> List[MemoryRecord]:
        wanted = set(record_ids)
        return [
            copy.deepcopy(record)
            for record in self._records.values()
            if record.id in wanted and record.owner_id == owner_id
        ]

    async def find_by_content_hash(
        self, owner_id: str, kind: MemoryKind, content_hash: str
    ) -> Optional[MemoryRecord]:
        for record in self._records.values():
            if (
                record.owner_id == owner_id
                and record.kind is kind
                and record.content_hash == content_hash
                and record.status is RecordStatus.ACTIVE
            ):
                return copy.deepcopy(record)
        return None

    async def find_by_dedup_key(
        self, owner_id: str, dedup_key: str
    ) -> List[MemoryRecord]:
        matches = [
            record for record in self._records.values()
            if record.owner_id == owner_id
            and record.dedup_key == dedup_key
            and record.status is RecordStatus.ACTIVE
        ]
        matches.sort(key=lambda r: r.created_at, reverse=True)
        return [copy.deepcopy(r) for r in matches]

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
        statuses = tuple(statuses or (RecordStatus.ACTIVE,))
        matches = [
            record for record in self._records.values()
            if record.owner_id == owner_id
            and record.status in statuses
            and (kinds is None or record.kind in tuple(kinds))
            and (visibilities is None or record.visibility in tuple(visibilities))
        ]
        matches.sort(key=lambda r: r.created_at, reverse=True)
        return [copy.deepcopy(r) for r in matches[offset:offset + limit]]

    async def count(
        self,
        owner_id: Optional[str] = None,
        *,
        kinds: Optional[Sequence[MemoryKind]] = None,
        statuses: Optional[Sequence[RecordStatus]] = None,
    ) -> int:
        return sum(
            1 for record in self._records.values()
            if (owner_id is None or record.owner_id == owner_id)
            and (kinds is None or record.kind in tuple(kinds))
            and (statuses is None or record.status in tuple(statuses))
        )

    async def pending_embeddings(self, limit: int = 100) -> List[MemoryRecord]:
        matches = [
            record for record in self._records.values()
            if record.embedding_status is EmbeddingStatus.PENDING
            and record.status is RecordStatus.ACTIVE
        ]
        matches.sort(key=lambda r: r.created_at)
        return [copy.deepcopy(r) for r in matches[:limit]]


    # ── lifecycle (Phase 5) ─────────────────────────────────────────────

    async def set_status(
        self, record_ids: Sequence[UUID], status: RecordStatus
    ) -> int:
        changed = 0
        for record_id in record_ids:
            record = self._records.get(record_id)
            if record is not None:
                record.status = status
                changed += 1
        return changed

    async def hard_delete(self, record_ids: Sequence[UUID]) -> int:
        removed = 0
        for record_id in record_ids:
            if self._records.pop(record_id, None) is not None:
                removed += 1
        return removed

    async def find_derived_from(
        self, owner_id: str, record_id: UUID
    ) -> List[MemoryRecord]:
        return [
            copy.deepcopy(record)
            for record in self._records.values()
            if record.owner_id == owner_id and record_id in record.derived_from
        ]

    async def iter_active(
        self, *, limit: int = 500, offset: int = 0
    ) -> List[MemoryRecord]:
        matches = [
            record for record in self._records.values()
            if record.status is RecordStatus.ACTIVE
        ]
        matches.sort(key=lambda r: r.created_at)
        return [copy.deepcopy(r) for r in matches[offset:offset + limit]]

    async def duplicate_dedup_keys(self, limit: int = 100) -> List[tuple]:
        counts: Dict[tuple, int] = {}
        for record in self._records.values():
            if record.dedup_key and record.status is RecordStatus.ACTIVE:
                key = (record.owner_id, record.dedup_key)
                counts[key] = counts.get(key, 0) + 1
        return [key for key, count in counts.items() if count > 1][:limit]

    async def owner_activity(self) -> List[tuple]:
        latest: Dict[str, object] = {}
        for record in self._records.values():
            current = latest.get(record.owner_id)
            if current is None or record.created_at > current:
                latest[record.owner_id] = record.created_at
        return list(latest.items())

    # ── test helpers ────────────────────────────────────────────────────

    def clear(self) -> None:
        self._records.clear()
