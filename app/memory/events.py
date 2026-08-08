"""
Outbox events for asynchronous memory ingestion.

Extraction must never run on the request path. A turn that waits for an LLM to
decide what was worth remembering pays that cost in latency the user feels —
and on a spoken turn it is dead air. So the turn writes an *event* and returns;
a worker does the thinking afterwards.

The outbox pattern gives durable at-least-once delivery with no message broker
to operate. `EventQueue` is a port, so moving to Redis or a real queue later is
an adapter swap rather than a redesign.

See docs/MEMORY_ARCHITECTURE.md §3.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from app.memory.record import utcnow


class EventType(str, Enum):
    """What produced this event."""

    TURN = "turn"
    """One user/assistant exchange, awaiting extraction."""

    DOCUMENT = "document"
    """An ingested artifact, awaiting chunk extraction."""

    CONNECTOR = "connector"
    """Material from an external source (GitHub, Gmail, …). Phase 7."""


class EventStatus(str, Enum):
    """
    Lifecycle of one event.

    `FAILED` is terminal only after the attempt ceiling; before that a failed
    event returns to `PENDING` so a transient LLM outage does not permanently
    discard what the user said.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    """Processed, but produced nothing worth storing — a normal outcome for
    small talk, and distinct from failure."""


# Beyond this many attempts an event is parked rather than retried forever.
MAX_ATTEMPTS = 3


@dataclass
class MemoryEvent:
    """One unit of pending memory work."""

    owner_id: str
    event_type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    group_key: str = ""
    """
    Batching key — the session/conversation id for turn events.

    A real column rather than a JSONB lookup because the worker groups on it on
    every poll, and grouping on an unindexed JSON field is how a queue quietly
    becomes the slowest thing in the system.
    """

    id: UUID = field(default_factory=uuid4)
    status: EventStatus = EventStatus.PENDING
    attempts: int = 0
    last_error: Optional[str] = None
    created_at: datetime = field(default_factory=utcnow)
    claimed_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        self.owner_id = (self.owner_id or "").strip()
        if not self.owner_id:
            raise ValueError("MemoryEvent.owner_id is required")
        if not isinstance(self.event_type, EventType):
            self.event_type = EventType(self.event_type)
        if not isinstance(self.status, EventStatus):
            self.status = EventStatus(self.status)
        if not self.group_key:
            # Ungrouped events still need a key, or they would all batch
            # together across unrelated owners.
            self.group_key = f"{self.owner_id}:{self.event_type.value}"

    @property
    def exhausted(self) -> bool:
        return self.attempts >= MAX_ATTEMPTS


@dataclass(frozen=True)
class GroupReadiness:
    """A batching group and whether it is ready to process."""

    owner_id: str
    group_key: str
    pending: int
    oldest: datetime

    def is_ready(self, *, batch_size: int, idle_seconds: float,
                 now: Optional[datetime] = None) -> bool:
        """
        Ready when the batch is full, or when it has waited long enough.

        The idle flush is what stops a two-turn conversation being stranded
        unextracted forever because it never reached the batch size.
        """
        if self.pending >= batch_size:
            return True
        moment = now or utcnow()
        oldest = self.oldest
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=moment.tzinfo)
        return (moment - oldest).total_seconds() >= idle_seconds
