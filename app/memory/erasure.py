"""
Erasing everything the assistant remembers about one person.

`MemoryMaintenance.forget_owner` claimed to do this and did not. It deleted
`memory_records` and their vectors — one of nine stores holding per-user memory,
and not the one that answers questions. Until the Phase 6 read cutover the
*legacy* tables are authoritative for every read, so a user who erased their
memory would have watched the assistant keep answering from `user_profile`,
`chat_history`, `episodic_memory`, and four Qdrant collections that were never
touched. A deletion that does not delete is worse than no deletion feature,
because the user believes it worked.

The same gap made guest collection a no-op in practice. Every recruiter visit
mints a `guest-<uuid>` partition; the sweep purged its `memory_records` and left
the chat history, profile facts and vectors to accumulate forever — which was
the exact cost the sweep exists to prevent.

Three properties this module is built around:

**Enumeration, not recollection.** Stores are registered in one table. Adding a
store without adding it here is a visible omission in one place rather than an
invisible one spread across call sites.

**Partial failure is reported, never swallowed.** If Qdrant is unreachable, some
data survives. Returning "success" then would be a false statement about the
user's data — the same class of error as reporting NO_DATA after a lookup
failure. `ErasureReport.complete` is false and the failed stores are named.

**Cache invalidation is part of erasure.** A retrieval cache that outlives a
deletion serves deleted memories for its full TTL. That is not a staleness
nuisance; it is the deletion failing for five minutes.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StoreResult:
    """What happened to one store."""

    store: str
    deleted: Optional[int] = 0
    """Rows removed, or None when the store does not report a count.

    Qdrant's delete-by-filter returns no count, and printing `deleted=0` for it
    reads as "nothing was there" — a false statement in a report whose entire
    job is to be trusted about what survived. None says "cleared, count not
    available", which is what actually happened."""

    ok: bool = True
    error: str = ""

    def summary(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "store": self.store, "deleted": self.deleted, "ok": self.ok,
        }
        if self.error:
            payload["error"] = self.error[:200]
        return payload


@dataclass
class ErasureReport:
    """The outcome of erasing one owner, store by store."""

    owner_id: str
    results: List[StoreResult] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """
        Whether every store was cleared.

        False means data survives somewhere. The caller must not report success
        to the user on the strength of the stores that did work.
        """
        return all(result.ok for result in self.results)

    @property
    def deleted(self) -> int:
        """Rows removed from the stores that report a count."""
        return sum(
            result.deleted for result in self.results
            if isinstance(result.deleted, int)
        )

    @property
    def failed_stores(self) -> List[str]:
        return [result.store for result in self.results if not result.ok]

    def summary(self) -> Dict[str, Any]:
        """Structured form for logs and API responses — counts, never content."""
        return {
            "owner_id": self.owner_id,
            "complete": self.complete,
            "deleted": self.deleted,
            "failed_stores": self.failed_stores,
            "stores": [result.summary() for result in self.results],
        }


# One entry per place per-user memory lives. A store missing from this list is
# a store that survives erasure, so the list is deliberately exhaustive and
# ordered from most to least authoritative.
#
# Application data — timetable, attendance, job bookmarks, email drafts — is
# deliberately absent. Those are records the user created, not things the
# assistant inferred about them, and deleting them belongs to account closure
# rather than to "forget what you know about me". The boundary is stated here
# so it is a decision rather than an oversight.
_QDRANT_COLLECTIONS: Tuple[str, ...] = (
    "resume_chunks",
    "skills_chunks",
    "projects_chunks",
    "smart_memory_chunks",
)

_POSTGRES_TABLES: Tuple[Tuple[str, str], ...] = (
    # (ORM attribute on app.memory.models, owner column)
    ("ChatHistory", "user_id"),
    ("UserProfile", "user_id"),
    ("EpisodicMemory", "user_id"),
    ("ToolMemory", "user_id"),
    ("TurnORM", "owner_id"),
    ("ConversationORM", "owner_id"),
    ("MemoryEventORM", "owner_id"),
)


class MemoryErasure:
    """Erases every trace of one owner across every memory store."""

    def __init__(
        self,
        maintenance=None,
        cache=None,
        session_maker=None,
        qdrant=None,
    ):
        # Every external dependency is injectable. Not for purity: an eraser
        # that reaches for module singletons cannot be exercised without a live
        # Postgres and a live Qdrant, which means the one code path where a
        # silent partial failure is unacceptable would be the least tested.
        self._maintenance = maintenance
        self._cache = cache
        self._session_maker = session_maker
        self._qdrant = qdrant

    @property
    def maintenance(self):
        if self._maintenance is None:
            from app.memory.cognition.maintenance import memory_maintenance

            self._maintenance = memory_maintenance
        return self._maintenance

    @property
    def cache(self):
        if self._cache is None:
            from app.memory.memory_cache import memory_cache

            self._cache = memory_cache
        return self._cache

    async def erase_owner(self, owner_id: str) -> ErasureReport:
        """
        Delete everything remembered about `owner_id`.

        Every store is attempted even when an earlier one fails: stopping at the
        first error would leave *more* data behind than continuing, and the
        report names what survived so the caller can retry.
        """
        owner_id = (owner_id or "").strip()
        report = ErasureReport(owner_id=owner_id)
        if not owner_id:
            report.results.append(
                StoreResult("input", ok=False, error="owner_id is required")
            )
            return report

        # The cache goes first. It holds assembled context that would otherwise
        # keep being served for its full TTL after the underlying rows are gone,
        # and clearing it before the deletes means a concurrent request cannot
        # repopulate it from data that is about to disappear.
        report.results.append(self._clear_cache(owner_id))

        report.results.append(await self._erase_records(owner_id))

        for collection in _QDRANT_COLLECTIONS:
            report.results.append(await self._erase_qdrant(owner_id, collection))

        report.results.extend(await self._erase_postgres(owner_id))

        # And again afterwards: a request in flight during the deletes may have
        # written a fresh entry from rows that no longer exist.
        self._clear_cache(owner_id)

        level = logger.info if report.complete else logger.error
        level("Memory erasure for owner=%s: %s", owner_id, report.summary())
        return report

    # ── individual stores ───────────────────────────────────────────────

    def _clear_cache(self, owner_id: str) -> StoreResult:
        result = StoreResult("retrieval_cache")
        try:
            result.deleted = self.cache.invalidate(owner_id) or 0
        except Exception as exc:
            result.ok = False
            result.error = str(exc)
            logger.error("Could not clear retrieval cache for %s: %s", owner_id, exc)
        return result

    async def _erase_records(self, owner_id: str) -> StoreResult:
        """The v2 unified store, plus its vectors, via the cascading deleter."""
        result = StoreResult("memory_records")
        try:
            result.deleted = await self.maintenance.forget_owner(owner_id)
        except Exception as exc:
            result.ok = False
            result.error = str(exc)
            logger.error("Could not erase memory_records for %s: %s", owner_id, exc)
        return result

    async def _erase_qdrant(self, owner_id: str, collection: str) -> StoreResult:
        """
        One legacy Qdrant collection.

        Deleted server-side by filter rather than scroll-then-delete-by-id: a
        scroll is bounded by however many points one walk returns, which on a
        right-to-erasure request silently leaves data behind.
        """
        # None, not 0: this store clears by filter and reports no count.
        result = StoreResult(collection, deleted=None)
        try:
            qdrant = self._qdrant
            if qdrant is None:
                from app.services.qdrant_service import qdrant_service

                qdrant = qdrant_service

            await qdrant.delete_by_filter(
                collection_name=collection,
                filter_conditions={"user_id": owner_id},
            )
        except Exception as exc:
            result.ok = False
            result.error = str(exc)
            logger.error(
                "Could not erase Qdrant collection %s for %s: %s",
                collection, owner_id, exc,
            )
        return result

    async def _erase_postgres(self, owner_id: str) -> List[StoreResult]:
        """Every legacy Postgres table keyed by this owner."""
        results: List[StoreResult] = []
        try:
            from sqlalchemy import delete

            from app.memory import models as memory_models

            session_maker = self._session_maker
            if session_maker is None:
                from app.db.session import async_session_maker

                session_maker = async_session_maker
        except Exception as exc:
            return [
                StoreResult(table, ok=False, error=f"import failed: {exc}")
                for table, _ in _POSTGRES_TABLES
            ]

        for attribute, owner_column in _POSTGRES_TABLES:
            model = getattr(memory_models, attribute, None)
            table_name = getattr(model, "__tablename__", attribute)
            result = StoreResult(table_name)
            if model is None:
                result.ok = False
                result.error = f"unknown model {attribute}"
                results.append(result)
                continue

            try:
                column = getattr(model, owner_column)
                async with session_maker() as session:
                    outcome = await session.execute(
                        delete(model).where(column == owner_id)
                    )
                    await session.commit()
                    result.deleted = int(outcome.rowcount or 0)
            except Exception as exc:
                result.ok = False
                result.error = str(exc)
                logger.error(
                    "Could not erase %s for %s: %s", table_name, owner_id, exc
                )
            results.append(result)

        return results


# Singleton instance
memory_erasure = MemoryErasure()
