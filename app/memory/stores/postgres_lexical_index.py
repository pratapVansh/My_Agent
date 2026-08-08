"""
PostgreSQL full-text adapter for the `LexicalIndex` port.

Uses `to_tsvector('english', content)` with `plainto_tsquery`, ranked by
`ts_rank_cd`. The matching GIN index is declared on the model; without it these
queries still return correct results via a sequential scan, which at personal-
assistant scale is entirely acceptable — the index is an optimisation, not a
correctness requirement. That matters because an existing deployment created
the table before the index was declared (`create_all` adds tables, not indexes
to existing tables), and lexical search must not silently break there.
"""
from __future__ import annotations

from typing import List, Optional, Sequence
import logging

from sqlalchemy import and_, func, select

from app.db.session import async_session_maker
from app.memory.kinds import MemoryKind, RecordStatus
from app.memory.models import MemoryRecordORM
from app.memory.ports import LexicalHit

logger = logging.getLogger(__name__)

_TS_CONFIG = "english"


class PostgresLexicalIndex:
    """`LexicalIndex` over `memory_records.content`."""

    def __init__(self, session_maker=None):
        self.async_session_maker = session_maker or async_session_maker

    async def search(
        self,
        owner_id: str,
        query: str,
        *,
        kinds: Optional[Sequence[MemoryKind]] = None,
        limit: int = 20,
    ) -> List[LexicalHit]:
        cleaned = (query or "").strip()
        if not cleaned:
            return []

        # plainto_tsquery treats the input as plain words and ANDs them, so a
        # user's raw utterance cannot become a tsquery syntax error. Accepting
        # arbitrary text safely is the whole reason to prefer it over
        # to_tsquery here.
        tsquery = func.plainto_tsquery(_TS_CONFIG, cleaned)
        tsvector = func.to_tsvector(_TS_CONFIG, MemoryRecordORM.content)
        rank = func.ts_rank_cd(tsvector, tsquery)

        conditions = [
            MemoryRecordORM.owner_id == owner_id,
            MemoryRecordORM.status == RecordStatus.ACTIVE.value,
            tsvector.op("@@")(tsquery),
        ]
        if kinds:
            conditions.append(
                MemoryRecordORM.kind.in_([k.value for k in kinds])
            )

        statement = (
            select(MemoryRecordORM.id, rank.label("rank"))
            .where(and_(*conditions))
            .order_by(rank.desc())
            .limit(limit)
        )

        try:
            async with self.async_session_maker() as session:
                result = await session.execute(statement)
                return [
                    LexicalHit(record_id=row[0], score=float(row[1]))
                    for row in result.all()
                ]
        except Exception as exc:
            # A failed channel degrades retrieval; it must not fail the turn.
            # The engine treats an empty channel as "contributed nothing".
            logger.warning("Lexical search failed for owner=%s: %s", owner_id, exc)
            return []


# Singleton instance
postgres_lexical_index = PostgresLexicalIndex()
