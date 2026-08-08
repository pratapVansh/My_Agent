"""
Qdrant adapter for the `VectorStore` port.

All memory kinds share one collection, filtered by payload. That mirrors the
single-table record model: one ranked candidate pool, not nine.

The Qdrant point id *is* the record id, so there is no mapping to keep
consistent — and migrating to pgvector later becomes a column addition rather
than a re-keying exercise.
"""
from __future__ import annotations

from typing import List, Optional, Sequence
from uuid import UUID
import logging

from qdrant_client.models import PointStruct

from app.memory.kinds import MemoryKind
from app.memory.ports import VectorHit
from app.memory.record import MemoryRecord
from app.services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)

MEMORY_COLLECTION = "memory_records"


class QdrantVectorStore:
    """`VectorStore` over a single Qdrant collection."""

    def __init__(self, collection_name: str = MEMORY_COLLECTION):
        self.qdrant = qdrant_service
        self.collection_name = collection_name
        self._ready = False

    # Every payload field `search()` filters on. Qdrant returns a 400 for a
    # filter on an unindexed field, so an omission here is an outage, not a
    # slow query — which is exactly how it first surfaced: the collection was
    # created with the legacy index set (`user_id`, `type`) while this store
    # filters on `owner_id` and `kind`.
    INDEX_FIELDS = ["owner_id", "kind", "visibility", "status"]

    async def initialize(self) -> None:
        """Create the collection and its payload indexes. Idempotent."""
        await self.qdrant.ensure_collection(
            self.collection_name, index_fields=self.INDEX_FIELDS
        )
        self._ready = True

    async def _ensure_ready(self) -> None:
        """
        Guarantee the collection exists before the first write.

        Called lazily rather than relying solely on application startup,
        because the memory worker also runs standalone
        (`scripts/run_memory_worker.py`) and never goes through the FastAPI
        lifespan. Without this the first embedding pass in that mode fails with
        a 404 for a collection nobody ever created — which is exactly what
        happened the first time this ran for real.
        """
        if self._ready:
            return
        await self.initialize()

    async def upsert(
        self, entries: Sequence[tuple[MemoryRecord, Sequence[float]]]
    ) -> None:
        if not entries:
            return

        await self._ensure_ready()

        points = [
            PointStruct(
                id=str(record.id),
                vector=list(vector),
                # Payload carries only what filtering and debugging need. The
                # record itself stays authoritative in Postgres, so this is a
                # denormalised index, never a second source of truth.
                payload={
                    "owner_id": record.owner_id,
                    "kind": record.kind.value,
                    "visibility": record.visibility.value,
                    "status": record.status.value,
                    "importance": record.importance,
                    "text": record.content,
                },
            )
            for record, vector in entries
        ]
        await self.qdrant.upsert_points(
            collection_name=self.collection_name, points=points
        )

    async def search(
        self,
        owner_id: str,
        query_vector: Sequence[float],
        *,
        kinds: Optional[Sequence[MemoryKind]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[VectorHit]:
        filters = {"owner_id": owner_id, "status": "active"}
        # QdrantService builds equality conditions only, so a multi-kind query
        # filters in Python rather than silently matching the wrong thing.
        single_kind = kinds[0] if kinds and len(kinds) == 1 else None
        if single_kind is not None:
            filters["kind"] = single_kind.value

        overfetch = limit if single_kind is not None or not kinds else limit * 4
        results = await self.qdrant.query_points(
            collection_name=self.collection_name,
            query_vector=list(query_vector),
            limit=overfetch,
            score_threshold=score_threshold,
            filter_conditions=filters,
        )

        wanted = {k.value for k in kinds} if kinds else None
        hits: List[VectorHit] = []
        for result in results:
            payload = result.payload or {}
            if wanted is not None and payload.get("kind") not in wanted:
                continue
            hits.append(
                VectorHit(
                    record_id=UUID(str(result.id)),
                    score=result.score,
                    payload=payload,
                )
            )
            if len(hits) >= limit:
                break
        return hits

    async def delete(self, record_ids: Sequence[UUID]) -> None:
        if not record_ids:
            return
        await self.qdrant.delete_points(
            collection_name=self.collection_name,
            point_ids=[str(rid) for rid in record_ids],
        )

    async def delete_by_owner(self, owner_id: str) -> None:
        """Erasure path — filtered delete, so it has no page-size ceiling."""
        await self.qdrant.delete_by_filter(
            collection_name=self.collection_name,
            filter_conditions={"owner_id": owner_id},
        )


# Singleton instance
qdrant_vector_store = QdrantVectorStore()
