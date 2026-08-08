"""
The unified memory record.

Every remembered thing — an identity fact, a conversation summary, a resume
chunk, a goal — is one of these. See docs/MEMORY_ARCHITECTURE.md §3.3.

Two properties earn their complexity:

**Dual representation.** `content` is always a complete, self-contained natural
language sentence, because it is simultaneously what gets embedded and what
gets injected into the prompt. `structured` carries the machine-readable form
for filters, UI, and integrations. One table then serves both semantic search
and structured query without a second schema.

**Bitemporal validity.** "Vansh is a student" was true until it wasn't.
Superseding a fact closes `valid_to` and links `supersedes_id` rather than
deleting the row, which is what makes versioning, honest conflict handling, and
"what did the assistant believe last March" all possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
import hashlib
import re

from app.memory.kinds import (
    EmbeddingStatus,
    MemoryKind,
    RecordStatus,
    Sensitivity,
    SourceType,
    Visibility,
)


def utcnow() -> datetime:
    """Timezone-aware now. Naive datetimes are a bug waiting for a deployment."""
    return datetime.now(timezone.utc)


def normalize_content(text: str) -> str:
    """Collapse whitespace so trivially different spellings hash alike."""
    return re.sub(r"\s+", " ", text or "").strip()


def compute_content_hash(kind: MemoryKind, content: str) -> str:
    """
    Stable identity for exact-duplicate detection.

    Scoped by kind: the same sentence stored as a `semantic` fact and as a
    `document` chunk are genuinely different memories with different retrieval
    behaviour, so they must not collide.
    """
    normalized = normalize_content(content).lower()
    kind_value = kind.value if isinstance(kind, MemoryKind) else str(kind)
    return hashlib.sha256(f"{kind_value}:{normalized}".encode("utf-8")).hexdigest()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass
class MemoryRecord:
    """One remembered thing."""

    # ── identity ────────────────────────────────────────────────────────
    owner_id: str
    kind: MemoryKind
    content: str
    id: UUID = field(default_factory=uuid4)

    # ── content ─────────────────────────────────────────────────────────
    structured: Dict[str, Any] = field(default_factory=dict)

    # ── salience ────────────────────────────────────────────────────────
    importance: float = 0.5
    confidence: float = 1.0
    pinned: bool = False
    """User override: never decays, never dropped for budget."""

    # ── temporal ────────────────────────────────────────────────────────
    occurred_at: Optional[datetime] = None
    """When the fact was true or the event happened, if different from when it
    was recorded. A resume uploaded today can describe a 2023 internship."""
    valid_from: datetime = field(default_factory=utcnow)
    valid_to: Optional[datetime] = None
    """None means "still true". Set when superseded."""
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0

    # ── lineage ─────────────────────────────────────────────────────────
    source_type: SourceType = SourceType.SYSTEM
    source_ref: Optional[str] = None
    """Conversation id, document id, external id — whatever identifies the
    origin within its source."""
    derived_from: List[UUID] = field(default_factory=list)
    """Provenance for consolidated memories. Answers "why do you know this?"
    and drives cascading deletion."""
    supersedes_id: Optional[UUID] = None
    version: int = 1

    # ── governance ──────────────────────────────────────────────────────
    visibility: Visibility = Visibility.PRIVATE
    sensitivity: Sensitivity = Sensitivity.NORMAL
    status: RecordStatus = RecordStatus.ACTIVE

    # ── dedup / indexing ────────────────────────────────────────────────
    content_hash: str = ""
    """Computed from kind + normalised content when not supplied."""
    dedup_key: Optional[str] = None
    """Normalised semantic key for conflict detection — e.g. `profile:name`.
    Two active records sharing one contradict each other by definition."""
    embedding_status: EmbeddingStatus = EmbeddingStatus.PENDING

    def __post_init__(self) -> None:
        self.owner_id = (self.owner_id or "").strip()
        if not self.owner_id:
            raise ValueError("MemoryRecord.owner_id is required")

        self.content = normalize_content(self.content)
        if not self.content:
            raise ValueError("MemoryRecord.content cannot be empty")

        # Accept plain strings at the boundary (JSON, DB rows) but hold enums.
        if not isinstance(self.kind, MemoryKind):
            self.kind = MemoryKind(self.kind)
        if not isinstance(self.source_type, SourceType):
            self.source_type = SourceType(self.source_type)
        if not isinstance(self.visibility, Visibility):
            self.visibility = Visibility(self.visibility)
        if not isinstance(self.sensitivity, Sensitivity):
            self.sensitivity = Sensitivity(self.sensitivity)
        if not isinstance(self.status, RecordStatus):
            self.status = RecordStatus(self.status)
        if not isinstance(self.embedding_status, EmbeddingStatus):
            self.embedding_status = EmbeddingStatus(self.embedding_status)

        self.importance = _clamp(self.importance)
        self.confidence = _clamp(self.confidence)

        if not self.content_hash:
            self.content_hash = compute_content_hash(self.kind, self.content)

    # ── predicates ──────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.status is RecordStatus.ACTIVE

    @property
    def is_currently_valid(self) -> bool:
        """Active *and* not closed off by a newer version."""
        return self.is_active and self.valid_to is None

    @property
    def is_injectable(self) -> bool:
        """
        Safe to place in a prompt.

        SECRET never reaches a model — that is the entire point of the
        classification, and it is checked here rather than at each call site so
        no future retrieval path can forget to.
        """
        return self.is_currently_valid and self.sensitivity is not Sensitivity.SECRET

    # ── transitions ─────────────────────────────────────────────────────

    def superseded_by(self, replacement: "MemoryRecord", *, at: Optional[datetime] = None) -> "MemoryRecord":
        """
        Return this record closed off in favour of `replacement`.

        Returns a copy rather than mutating: the caller writes both rows in one
        transaction, and a half-applied supersession is worse than none.
        """
        moment = at or utcnow()
        return replace(
            self,
            status=RecordStatus.SUPERSEDED,
            valid_to=moment,
            updated_at=moment,
        )

    def superseding(self, **changes: Any) -> "MemoryRecord":
        """Build the next version of this record, linked back to it."""
        moment = utcnow()
        base: Dict[str, Any] = {
            "id": uuid4(),
            "supersedes_id": self.id,
            "version": self.version + 1,
            "valid_from": moment,
            "valid_to": None,
            "created_at": moment,
            "updated_at": moment,
            "status": RecordStatus.ACTIVE,
            "content_hash": "",  # recomputed in __post_init__
            "embedding_status": EmbeddingStatus.PENDING,
        }
        base.update(changes)
        return replace(self, **base)

    def accessed(self, *, at: Optional[datetime] = None) -> "MemoryRecord":
        """Record a retrieval hit. Feeds the frequency term in ranking."""
        return replace(
            self,
            last_accessed_at=at or utcnow(),
            access_count=self.access_count + 1,
        )
