"""
Hybrid retrieval: candidate generation → fusion → reranking.

Three channels run concurrently and are fused by reciprocal rank, then every
surviving candidate is rescored with the full signal set.

Channel independence is a design requirement, not an accident. Any one of them
can fail — Qdrant unreachable, no embeddings computed yet, Postgres full-text
index missing — and retrieval must degrade to whatever remains rather than
failing the turn. A user whose vector store is down should get slightly worse
memory, not a broken assistant.

See docs/MEMORY_ARCHITECTURE.md §3.6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from uuid import UUID
import asyncio
import logging
import time

from app.memory.kinds import ALWAYS_INJECTED_KINDS, MemoryKind, RecordStatus, Visibility
from app.memory.record import MemoryRecord
from app.memory.retrieval.scoring import (
    normalize_scores,
    rank_score,
    reciprocal_rank_fusion,
)
from app.memory.retrieval.trace import ChannelResult, RetrievalTrace

logger = logging.getLogger(__name__)

# How many candidates each channel may contribute. Generous relative to what
# survives the budget, because reranking can only reorder what it was given.
_VECTOR_LIMIT = 30
_LEXICAL_LIMIT = 30
_STRUCTURED_LIMIT = 40


@dataclass
class ScoredRecord:
    """A record with its final rank score."""

    record: MemoryRecord
    score: float
    similarity: float = 0.0


@dataclass
class RetrievalResultSet:
    """Ranked records plus the trace explaining how they were chosen."""

    scored: List[ScoredRecord]
    trace: RetrievalTrace

    @property
    def records(self) -> List[MemoryRecord]:
        return [item.record for item in self.scored]


class RetrievalEngine:
    """Hybrid retrieval over the unified record store."""

    def __init__(
        self,
        record_store,
        *,
        vector_store=None,
        lexical_index=None,
        embedder=None,
    ):
        self.records = record_store
        self.vectors = vector_store
        self.lexical = lexical_index
        # Injected rather than imported so retrieval can be tested without a
        # Cohere key and swapped wholesale when the embedding provider changes.
        self.embedder = embedder

    async def retrieve(
        self,
        owner_id: str,
        query: str = "",
        *,
        visibilities: Optional[Sequence[Visibility]] = None,
        boosted_kinds: Optional[Sequence[MemoryKind]] = None,
        limit: int = 40,
    ) -> RetrievalResultSet:
        """
        Gather, fuse, and rank candidates for one query.

        `visibilities` is how the recruiter view is served: the same engine,
        scoped to the owner's public records instead of a guest's own empty
        partition.
        """
        started = time.perf_counter()
        trace = RetrievalTrace(owner_id=owner_id, query=query)

        vector_ids, lexical_ids, structured = await asyncio.gather(
            self._vector_channel(owner_id, query, trace),
            self._lexical_channel(owner_id, query, trace),
            self._structured_channel(owner_id, visibilities, trace),
        )

        # Structured candidates are already hydrated; the search channels give
        # ids that still need fetching.
        structured_by_id = {record.id: record for record in structured}
        fused = reciprocal_rank_fusion([vector_ids, lexical_ids])
        trace.fused_candidates = len(set(fused) | set(structured_by_id))

        missing = [rid for rid in fused if rid not in structured_by_id]
        hydrated = await self._hydrate(owner_id, missing)

        candidates: Dict[UUID, MemoryRecord] = dict(structured_by_id)
        candidates.update({record.id: record for record in hydrated})

        similarity = normalize_scores(fused)

        scored = [
            ScoredRecord(
                record=record,
                similarity=similarity.get(record.id, 0.0),
                score=rank_score(
                    record,
                    similarity=similarity.get(record.id, 0.0),
                    boosted_kinds=boosted_kinds,
                ),
            )
            for record in candidates.values()
            # Enforced here as well as in the store: a SECRET record must not
            # reach a prompt even if some future channel surfaces it.
            if record.is_injectable
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        scored = scored[:limit]

        trace.ranked_candidates = len(scored)
        trace.total_latency_ms = (time.perf_counter() - started) * 1000
        return RetrievalResultSet(scored=scored, trace=trace)

    # ── channels ────────────────────────────────────────────────────────

    async def _vector_channel(
        self, owner_id: str, query: str, trace: RetrievalTrace
    ) -> List[UUID]:
        """Semantic similarity. Silent no-op until embeddings exist."""
        result = ChannelResult(name="vector")
        started = time.perf_counter()

        if not query or self.vectors is None or self.embedder is None:
            result.candidates = 0
            trace.add_channel(result)
            return []

        try:
            vector = await self.embedder(query)
            hits = await self.vectors.search(
                owner_id, vector, limit=_VECTOR_LIMIT
            )
            ids = [hit.record_id for hit in hits]
            result.candidates = len(ids)
            return ids
        except Exception as exc:
            result.ok = False
            result.error = str(exc)[:200]
            logger.warning("Vector channel failed for owner=%s: %s", owner_id, exc)
            return []
        finally:
            result.latency_ms = (time.perf_counter() - started) * 1000
            trace.add_channel(result)

    async def _lexical_channel(
        self, owner_id: str, query: str, trace: RetrievalTrace
    ) -> List[UUID]:
        """Keyword match — the channel that catches exact tokens."""
        result = ChannelResult(name="lexical")
        started = time.perf_counter()

        if not query or self.lexical is None:
            trace.add_channel(result)
            return []

        try:
            hits = await self.lexical.search(owner_id, query, limit=_LEXICAL_LIMIT)
            ids = [hit.record_id for hit in hits]
            result.candidates = len(ids)
            return ids
        except Exception as exc:
            result.ok = False
            result.error = str(exc)[:200]
            logger.warning("Lexical channel failed for owner=%s: %s", owner_id, exc)
            return []
        finally:
            result.latency_ms = (time.perf_counter() - started) * 1000
            trace.add_channel(result)

    async def _structured_channel(
        self,
        owner_id: str,
        visibilities: Optional[Sequence[Visibility]],
        trace: RetrievalTrace,
    ) -> List[MemoryRecord]:
        """
        Always-relevant records, fetched regardless of the query.

        Identity, preferences and active goals belong in context whether or not
        the current question mentions them — "what should I work on?" needs the
        user's goals even though it shares no vocabulary with them. Relying on
        search alone to surface these is how an assistant forgets who it is
        talking to.
        """
        result = ChannelResult(name="structured")
        started = time.perf_counter()

        try:
            records = await self.records.list(
                owner_id,
                kinds=list(ALWAYS_INJECTED_KINDS),
                statuses=[RecordStatus.ACTIVE],
                visibilities=list(visibilities) if visibilities else None,
                limit=_STRUCTURED_LIMIT,
            )
            result.candidates = len(records)
            return records
        except Exception as exc:
            result.ok = False
            result.error = str(exc)[:200]
            logger.warning("Structured channel failed for owner=%s: %s", owner_id, exc)
            return []
        finally:
            result.latency_ms = (time.perf_counter() - started) * 1000
            trace.add_channel(result)

    async def _hydrate(
        self, owner_id: str, record_ids: Sequence[UUID]
    ) -> List[MemoryRecord]:
        if not record_ids:
            return []
        try:
            return await self.records.get_many(owner_id, record_ids)
        except Exception as exc:
            logger.warning("Hydration failed for owner=%s: %s", owner_id, exc)
            return []
