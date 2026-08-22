"""
Background embedding of pending records.

Records are written `PENDING` and vectorised here rather than at write time. A
profile fact saved mid-conversation must not pay for a Cohere round trip, and a
voice turn cannot afford one at all — so the cost moves off the request path
entirely.

This is what activates the vector channel added in Phase 2, which until now has
been wired but inert because nothing had produced any vectors.

See docs/MEMORY_ARCHITECTURE.md §3.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import logging

from app.memory.kinds import EmbeddingStatus

logger = logging.getLogger(__name__)


@dataclass
class EmbedStats:
    """Outcome of one embedding pass."""

    claimed: int = 0
    embedded: int = 0
    failed: int = 0

    def summary(self) -> dict:
        return {
            "claimed": self.claimed,
            "embedded": self.embedded,
            "failed": self.failed,
        }


class EmbeddingPass:
    """Embeds pending records and upserts them into the vector store."""

    def __init__(self, record_store=None, vector_store=None, embedder=None):
        # All three injected so the pass is testable without Cohere or Qdrant.
        if record_store is None:
            from app.memory.stores.postgres_record_store import postgres_record_store
            record_store = postgres_record_store
        if vector_store is None:
            from app.memory.stores.qdrant_vector_store import qdrant_vector_store
            vector_store = qdrant_vector_store
        if embedder is None:
            embedder = _default_embedder

        self.records = record_store
        self.vectors = vector_store
        self.embed_batch = embedder

    async def run_once(self, limit: int = 32) -> EmbedStats:
        """Embed one batch. Never raises — the worker must survive a bad pass."""
        stats = EmbedStats()

        try:
            pending = await self.records.pending_embeddings(limit=limit)
        except Exception as exc:
            logger.warning("Could not claim pending embeddings: %s", exc)
            return stats

        if not pending:
            return stats
        stats.claimed = len(pending)

        try:
            vectors = await self.embed_batch([record.content for record in pending])
        except Exception as exc:
            # Leave the records PENDING: a transient embedding-provider outage
            # must not permanently mark them failed and exclude them from
            # semantic search forever.
            logger.warning(
                "Embedding provider failed for %d records (left pending): %s",
                len(pending), exc,
            )
            stats.failed = len(pending)
            return stats

        if len(vectors) != len(pending):
            # A partial batch would silently pair records with the wrong
            # vectors — worse than not embedding at all, because the resulting
            # search results look plausible.
            logger.error(
                "Embedding batch size mismatch: expected %d, got %d. Skipping batch.",
                len(pending), len(vectors),
            )
            stats.failed = len(pending)
            return stats

        try:
            await self.vectors.upsert(list(zip(pending, vectors)))
        except Exception as exc:
            logger.warning("Vector upsert failed for %d records: %s", len(pending), exc)
            stats.failed = len(pending)
            return stats

        for record in pending:
            try:
                await self.records.mark_embedding(record.id, EmbeddingStatus.READY)
                stats.embedded += 1
            except Exception as exc:
                # The vector is stored but the record still says PENDING, so the
                # next pass re-embeds it. Wasteful, never wrong: upsert is
                # keyed by record id, so the retry overwrites rather than
                # duplicates.
                logger.warning(
                    "Could not mark record %s embedded: %s", record.id, exc
                )
                stats.failed += 1

        return stats


async def _default_embedder(texts: List[str]) -> List[List[float]]:
    """Cohere batch embedding, using the document input type for storage."""
    from app.services.cohere_service import cohere_service
    return await cohere_service.embed_batch(
        texts=texts, input_type="search_document"
    )


# Singleton instance
embedding_pass = EmbeddingPass()
