"""
Storage ports (L0).

Layers above this one never import a database client. That is what keeps the
vector-store decision reversible — pgvector versus Qdrant becomes a choice of
adapter rather than a rewrite of the retrieval engine — and what makes L2/L3
testable without a live Postgres, Qdrant, or Cohere.

Ports are added when they gain their first real consumer, never speculatively.
`RecordStore` and `VectorStore` arrived with Phase 1; `LexicalIndex` with the
Phase 2 retrieval engine. `EventQueue` follows in Phase 3.

See docs/MEMORY_ARCHITECTURE.md §3.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence
from uuid import UUID

from app.memory.events import EventStatus, GroupReadiness, MemoryEvent
from app.memory.kinds import EmbeddingStatus, MemoryKind, RecordStatus, Visibility
from app.memory.record import MemoryRecord


@dataclass(frozen=True)
class VectorHit:
    """One similarity-search result."""

    record_id: UUID
    score: float
    payload: Dict[str, Any]


@dataclass(frozen=True)
class LexicalHit:
    """One full-text search result."""

    record_id: UUID
    score: float


class RecordStore(Protocol):
    """
    Durable storage for `MemoryRecord`.

    Implementations must treat the (owner_id, kind, content_hash) triple as
    unique among *active* records: `add` returns the existing record instead of
    raising when an exact duplicate is written. Callers rely on that to make
    ingestion idempotent, which is what allows the backfill migration to be
    re-run safely.
    """

    async def add(self, record: MemoryRecord) -> MemoryRecord:
        """Insert, or return the existing active record with the same hash."""
        ...

    async def add_many(self, records: Sequence[MemoryRecord]) -> List[MemoryRecord]:
        """Insert many, de-duplicating each. Returns the stored records."""
        ...

    async def get(self, owner_id: str, record_id: UUID) -> Optional[MemoryRecord]:
        """Fetch one record, scoped by owner so an id alone cannot cross tenants."""
        ...

    async def get_many(
        self, owner_id: str, record_ids: Sequence[UUID]
    ) -> List[MemoryRecord]:
        """
        Bulk hydration for retrieval.

        Search channels return ids; issuing one query per id would make a
        fan-out of candidates cost a fan-out of round trips. Missing ids are
        omitted rather than raising — a vector index can outlive the record it
        points at.
        """
        ...

    async def find_by_content_hash(
        self, owner_id: str, kind: MemoryKind, content_hash: str
    ) -> Optional[MemoryRecord]:
        """Exact-duplicate lookup among active records."""
        ...

    async def find_by_dedup_key(
        self, owner_id: str, dedup_key: str
    ) -> List[MemoryRecord]:
        """Active records sharing a semantic key — i.e. candidate conflicts."""
        ...

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
        """Filtered listing, newest first. Defaults to active records only."""
        ...

    async def count(
        self,
        owner_id: Optional[str] = None,
        *,
        kinds: Optional[Sequence[MemoryKind]] = None,
        statuses: Optional[Sequence[RecordStatus]] = None,
    ) -> int:
        ...

    async def supersede(
        self, old: MemoryRecord, new: MemoryRecord
    ) -> MemoryRecord:
        """
        Atomically close `old` and insert `new` in its place.

        One transaction by contract: a half-applied supersession leaves either
        two active contradictory records or none at all.
        """
        ...

    async def mark_embedding(
        self, record_id: UUID, status: EmbeddingStatus
    ) -> None:
        ...

    async def pending_embeddings(self, limit: int = 100) -> List[MemoryRecord]:
        """Records awaiting a vector, oldest first."""
        ...

    # ── lifecycle (Phase 5) ─────────────────────────────────────────────

    async def set_status(
        self, record_ids: Sequence[UUID], status: RecordStatus
    ) -> int:
        """Bulk status change — how decay archives, and how erasure marks."""
        ...

    async def hard_delete(self, record_ids: Sequence[UUID]) -> int:
        """
        Irreversible removal.

        Only ever reached through an explicit erasure request. Everything else
        archives, because a memory the user might want back is worth far more
        than the row it occupies.
        """
        ...

    async def find_derived_from(
        self, owner_id: str, record_id: UUID
    ) -> List[MemoryRecord]:
        """
        Records distilled from this one.

        Erasure follows these: a consolidated memory that outlives the fact it
        was derived from is a deletion that did not actually delete.
        """
        ...

    async def iter_active(
        self, *, limit: int = 500, offset: int = 0
    ) -> List[MemoryRecord]:
        """Active records across all owners — the decay engine's input."""
        ...

    async def duplicate_dedup_keys(self, limit: int = 100) -> List[tuple]:
        """
        `(owner_id, dedup_key)` pairs with more than one active record.

        Should always be empty: the writer supersedes on conflict. Two
        concurrent writers can still race past that, so this is the sweep that
        notices when the invariant has actually broken.
        """
        ...

    async def owner_activity(self) -> List[tuple]:
        """`(owner_id, most_recent_created_at)` — drives guest retention."""
        ...


class VectorStore(Protocol):
    """
    Similarity search over memory records.

    The vector's identity is the record's own id, so there is no mapping table
    to keep consistent and the eventual move to pgvector is a column addition
    rather than a re-keying exercise.
    """

    async def upsert(
        self, entries: Sequence[tuple[MemoryRecord, Sequence[float]]]
    ) -> None:
        ...

    async def search(
        self,
        owner_id: str,
        query_vector: Sequence[float],
        *,
        kinds: Optional[Sequence[MemoryKind]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[VectorHit]:
        ...

    async def delete(self, record_ids: Sequence[UUID]) -> None:
        ...


class EventQueue(Protocol):
    """
    Durable work queue for asynchronous ingestion.

    Implementations must make `claim_group` safe under concurrent workers: two
    workers polling simultaneously must never receive the same event. The
    Postgres adapter achieves this with `FOR UPDATE SKIP LOCKED`; any
    replacement must provide an equivalent guarantee, because double-processing
    an event means double-extracting memories from one conversation.
    """

    async def enqueue(self, event: "MemoryEvent") -> "MemoryEvent":
        ...

    async def ready_groups(
        self, *, batch_size: int, idle_seconds: float, limit: int = 20
    ) -> List["GroupReadiness"]:
        """Groups whose batch is full, or which have waited long enough."""
        ...

    async def claim_group(self, group_key: str, limit: int = 50) -> List["MemoryEvent"]:
        """Atomically take a group's pending events for processing."""
        ...

    async def mark_done(
        self, event_ids: Sequence[UUID], status: "EventStatus"
    ) -> None:
        ...

    async def mark_failed(self, event_ids: Sequence[UUID], error: str) -> None:
        """
        Record a failure.

        Events below the attempt ceiling return to PENDING for retry; only an
        exhausted event is parked as FAILED. A transient LLM outage must not
        permanently discard what the user said.
        """
        ...

    async def pending_count(self, owner_id: Optional[str] = None) -> int:
        ...


class LexicalIndex(Protocol):
    """
    Keyword search over record content.

    Runs alongside vector search rather than instead of it. Embeddings are
    weak on exact tokens — a repository name, "LeetCode 1462", an acronym, a
    hostel block — which is exactly where a personal assistant most needs to be
    precise. Vector-only retrieval is the most common RAG mistake and this port
    exists to avoid it.
    """

    async def search(
        self,
        owner_id: str,
        query: str,
        *,
        kinds: Optional[Sequence[MemoryKind]] = None,
        limit: int = 20,
    ) -> List[LexicalHit]:
        ...
