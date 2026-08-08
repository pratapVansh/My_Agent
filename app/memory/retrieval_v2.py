"""
Wiring for the Phase 2 retrieval engine.

Composition root: the only place the production adapters are bound to the
engine. Keeping it out of `memory_manager` means the engine itself stays
importable — and testable — without Qdrant, Cohere, or Postgres.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import logging

from app.config import settings
from app.memory.kinds import MemoryKind, Visibility
from app.memory.retrieval import (
    AssembledContext,
    ContextAssembler,
    RetrievalEngine,
    RetrievalTrace,
    working_memory_builder,
)
from app.memory.stores import (
    postgres_lexical_index,
    postgres_record_store,
    qdrant_vector_store,
)
from app.services.cohere_service import cohere_service

logger = logging.getLogger(__name__)


async def _embed_query(text: str) -> List[float]:
    """
    Embed a query for the vector channel.

    `search_query` input type, not `search_document`: Cohere embeds the two
    asymmetrically, and using the document type for a query measurably degrades
    retrieval. CohereService caches these for 60s, so repeated turns on the
    same question do not re-embed.
    """
    return await cohere_service.embed_text(text=text, input_type="search_query")


_engine: Optional[RetrievalEngine] = None
_assembler: Optional[ContextAssembler] = None


def get_engine() -> RetrievalEngine:
    """The production engine, built once."""
    global _engine
    if _engine is None:
        _engine = RetrievalEngine(
            record_store=postgres_record_store,
            vector_store=qdrant_vector_store,
            lexical_index=postgres_lexical_index,
            embedder=_embed_query,
        )
    return _engine


def get_assembler() -> ContextAssembler:
    global _assembler
    if _assembler is None:
        _assembler = ContextAssembler(budget_tokens=settings.memory_v2_budget_tokens)
    return _assembler


async def retrieve_and_assemble(
    user_id: str,
    query: str = "",
    *,
    conversation_id: str = "",
    conversation_owner_id: Optional[str] = None,
    working_memory: Optional[str] = None,
    visibilities: Optional[Sequence[Visibility]] = None,
    boosted_kinds: Optional[Sequence[MemoryKind]] = None,
) -> Tuple[AssembledContext, RetrievalTrace]:
    """
    Full v2 path: retrieve, rank, allocate, render.

    `visibilities` is how the recruiter view will be served in Phase 6 — the
    same engine scoped to the owner's public records rather than to a guest's
    own empty partition.
    """
    # Load the conversation window unless the caller supplied one. Tier 1 of
    # the budget is the live thread, and it is what a user notices losing.
    if working_memory is None and conversation_id:
        window = await working_memory_builder.build(
            conversation_owner_id or user_id, conversation_id
        )
        working_memory = window.render()

    result = await get_engine().retrieve(
        owner_id=user_id,
        query=query,
        visibilities=visibilities,
        boosted_kinds=boosted_kinds,
    )
    assembled = get_assembler().assemble(
        result.scored, working_memory=working_memory or "", trace=result.trace
    )
    return assembled, result.trace
