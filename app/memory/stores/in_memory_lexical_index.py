"""
In-memory `LexicalIndex`, for tests and local development.

Token-overlap scoring rather than a real BM25: the engine consumes only the
*ranking*, never the absolute score, because reciprocal rank fusion reads
position alone. That makes a simple ranker a faithful stand-in for Postgres
full-text as far as everything downstream is concerned.
"""
from __future__ import annotations

from typing import List, Optional, Sequence
import re

from app.memory.kinds import MemoryKind, RecordStatus
from app.memory.ports import LexicalHit

_WORD = re.compile(r"[a-z0-9]+")

# Words too common to discriminate between records. A real tsvector config
# carries a far longer list; this covers the ones that would otherwise let any
# query match any record in a small test corpus.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "my", "me", "i", "user", "what", "who", "how",
    "do", "does", "did", "it", "its", "this", "that", "with", "at", "by",
})


def _tokenize(text: str) -> set:
    return {
        word for word in _WORD.findall((text or "").lower())
        if word not in _STOPWORDS and len(word) > 1
    }


class InMemoryLexicalIndex:
    """`LexicalIndex` over an `InMemoryRecordStore`."""

    def __init__(self, record_store):
        self.records = record_store

    async def search(
        self,
        owner_id: str,
        query: str,
        *,
        kinds: Optional[Sequence[MemoryKind]] = None,
        limit: int = 20,
    ) -> List[LexicalHit]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        wanted = set(kinds) if kinds else None
        hits: List[LexicalHit] = []

        for record in self.records._records.values():
            if record.owner_id != owner_id or record.status is not RecordStatus.ACTIVE:
                continue
            if wanted is not None and record.kind not in wanted:
                continue

            overlap = query_tokens & _tokenize(record.content)
            if overlap:
                hits.append(
                    LexicalHit(
                        record_id=record.id,
                        score=len(overlap) / len(query_tokens),
                    )
                )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]
