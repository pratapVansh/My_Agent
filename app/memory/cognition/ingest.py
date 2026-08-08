"""
Candidate → stored memory: govern, de-duplicate, reconcile, write.

Everything the extractor proposes passes through here. The extractor is an LLM
and is therefore capable of proposing a credential, a duplicate, or a fact that
contradicts one already held; none of those may reach storage unexamined.

De-duplication is exact only at this stage. Semantic near-duplicate merging is
the Phase 5 nightly sweep, where it can reuse vectors that already exist rather
than paying for an extra embedding per candidate on the ingest path.

See docs/MEMORY_ARCHITECTURE.md §3.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence
import logging

from app.memory.cognition.extractor import Candidate
from app.memory.kinds import SourceType, Visibility
from app.memory.record import MemoryRecord
from app.memory.stores.postgres_record_store import postgres_record_store
from app.memory.writer import MemoryWriter, WriteOutcome, memory_writer

logger = logging.getLogger(__name__)

# Words that mark a candidate as credential-bearing regardless of how the model
# phrased it. Defence in depth: the extractor is already instructed never to
# emit these, but an instruction is not an enforcement mechanism.
_CREDENTIAL_TERMS = frozenset({
    "password", "passwd", "passphrase", "pin code", "secret key", "api key",
    "apikey", "access token", "auth token", "private key", "credit card",
    "card number", "cvv", "ssn", "social security", "bank account",
    "routing number", "otp", "one-time code", "seed phrase",
})


def is_credential_bearing(text: str) -> bool:
    """True when a candidate looks like it carries a secret."""
    lowered = (text or "").lower()
    return any(term in lowered for term in _CREDENTIAL_TERMS)


@dataclass
class IngestResult:
    """Outcome of ingesting one batch of candidates."""

    stored: List[MemoryRecord] = field(default_factory=list)
    duplicates: int = 0
    superseded: int = 0
    rejected: int = 0
    failed: int = 0

    @property
    def created(self) -> int:
        return len(self.stored)

    def summary(self) -> dict:
        return {
            "created": self.created,
            "duplicates": self.duplicates,
            "superseded": self.superseded,
            "rejected": self.rejected,
            "failed": self.failed,
        }


class MemoryIngestor:
    """Applies governance and conflict resolution, then persists."""

    def __init__(self, record_store=None, writer: Optional[MemoryWriter] = None):
        self.records = record_store or postgres_record_store
        self.writer = writer or memory_writer

    async def ingest(
        self,
        owner_id: str,
        candidates: Sequence[Candidate],
        *,
        source_type: SourceType = SourceType.CHAT,
        source_ref: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
    ) -> IngestResult:
        result = IngestResult()

        for candidate in candidates:
            if is_credential_bearing(candidate.content):
                # Log the fact of the rejection, never the content that caused
                # it — writing it to the log would defeat the point.
                logger.warning(
                    "Rejected extracted memory for owner=%s: looks credential-bearing",
                    owner_id,
                )
                result.rejected += 1
                continue

            try:
                record = MemoryRecord(
                    owner_id=owner_id,
                    kind=candidate.kind,
                    content=candidate.content,
                    structured=candidate.structured,
                    importance=candidate.importance,
                    confidence=candidate.confidence,
                    occurred_at=occurred_at,
                    source_type=source_type,
                    source_ref=source_ref,
                    dedup_key=candidate.dedup_key,
                    visibility=Visibility.PRIVATE,
                )
            except ValueError as exc:
                # Empty content after normalisation, or a missing owner.
                logger.debug("Discarded malformed candidate: %s", exc)
                result.rejected += 1
                continue

            try:
                stored, outcome = await self.writer.upsert_with_outcome(record)
            except Exception as exc:
                logger.warning(
                    "Failed to store extracted memory for owner=%s: %s", owner_id, exc
                )
                result.failed += 1
                continue

            # The outcome comes from the writer rather than being inferred
            # here: a duplicate and a supersession both return a record whose
            # id differs from the one submitted, so they cannot be told apart
            # from the outside.
            if outcome is WriteOutcome.DUPLICATE:
                result.duplicates += 1
            else:
                if outcome is WriteOutcome.SUPERSEDED:
                    result.superseded += 1
                result.stored.append(stored)

        return result


# Singleton instance
memory_ingestor = MemoryIngestor()
