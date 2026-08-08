"""
Memory lifecycle: decay, reconciliation, de-duplication, and forgetting.

Without this layer the store only ever grows, and a memory system that only
grows gets worse: retrieval quality falls as noise accumulates, and every
irrelevant record competes for the same context budget.

Forgetting is a state machine, never a silent delete:

    active ──decay──> archived ──explicit request──> deleted (hard, cascading)
       │                  │
       └──superseded──────┘

Archived records leave default retrieval but stay searchable on demand, so the
reversible operation is the default and the irreversible one requires asking.

See docs/MEMORY_ARCHITECTURE.md §3.9.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Set
from uuid import UUID
import logging

from app.config import settings
from app.memory.kinds import ALWAYS_INJECTED_KINDS, MemoryKind, RecordStatus
from app.memory.record import MemoryRecord, utcnow
from app.memory.retrieval.scoring import recency_score

logger = logging.getLogger(__name__)

# Kinds eligible for semantic de-duplication. Episodes are excluded on purpose:
# two similar-sounding events are usually two genuinely different events, and
# merging them would fabricate a history that never happened.
DEDUPABLE_KINDS = (MemoryKind.SEMANTIC, MemoryKind.DOCUMENT)


@dataclass
class MaintenanceStats:
    """Outcome of one maintenance cycle."""

    scanned: int = 0
    archived: int = 0
    conflicts_resolved: int = 0
    duplicates_merged: int = 0
    guests_purged: int = 0
    records_deleted: int = 0
    failed_jobs: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        return {
            "scanned": self.scanned,
            "archived": self.archived,
            "conflicts_resolved": self.conflicts_resolved,
            "duplicates_merged": self.duplicates_merged,
            "guests_purged": self.guests_purged,
            "records_deleted": self.records_deleted,
            "failed_jobs": self.failed_jobs,
        }

    @property
    def did_work(self) -> bool:
        return bool(
            self.archived or self.conflicts_resolved or self.duplicates_merged
            or self.guests_purged or self.records_deleted
        )


def effective_score(record: MemoryRecord, *, now=None) -> float:
    """
    How relevant a record still is, independent of any query.

    Importance decayed by age on the record's own half-life. Identity,
    preferences and goals have no half-life, so their score is simply their
    importance and they never decay out.
    """
    return record.importance * recency_score(record, now=now)


def is_decay_exempt(record: MemoryRecord) -> bool:
    """
    Records decay never touches.

    Pinned records are an explicit user instruction. The always-injected kinds
    are exempt because forgetting the user's name, their stated preferences, or
    what they are working towards is never the right way to save space.
    """
    return record.pinned or record.kind in ALWAYS_INJECTED_KINDS


class MemoryMaintenance:
    """Periodic sweeps that keep the store useful as it ages."""

    def __init__(self, record_store=None, vector_store=None, embedder=None):
        if record_store is None:
            from app.memory.stores.postgres_record_store import postgres_record_store
            record_store = postgres_record_store
        if vector_store is None:
            from app.memory.stores.qdrant_vector_store import qdrant_vector_store
            vector_store = qdrant_vector_store

        self.records = record_store
        self.vectors = vector_store
        self.embedder = embedder

    async def run_once(self) -> MaintenanceStats:
        """
        Run every job, isolating each.

        One failing sweep must not prevent the others: they are independent,
        and a store that skips de-duplication because guest collection broke is
        strictly worse off for no reason.
        """
        stats = MaintenanceStats()

        for name, job in (
            ("conflicts", self.reconcile_conflicts),
            ("decay", self.decay_and_archive),
            ("dedup", self.sweep_duplicates),
            ("guest_gc", self.collect_abandoned_guests),
        ):
            try:
                await job(stats)
            except Exception as exc:
                stats.failed_jobs.append(name)
                logger.error("Maintenance job '%s' failed: %s", name, exc, exc_info=True)

        if stats.did_work or stats.failed_jobs:
            logger.info("Memory maintenance: %s", stats.summary())
        return stats

    # ── decay ───────────────────────────────────────────────────────────

    async def decay_and_archive(self, stats: MaintenanceStats) -> None:
        """Archive records whose relevance has fallen away."""
        now = utcnow()
        min_age = timedelta(days=settings.memory_decay_min_age_days)
        threshold = settings.memory_decay_threshold

        offset = 0
        to_archive: List[UUID] = []

        while True:
            batch = await self.records.iter_active(limit=500, offset=offset)
            if not batch:
                break
            offset += len(batch)
            stats.scanned += len(batch)

            for record in batch:
                if is_decay_exempt(record):
                    continue
                reference = record.occurred_at or record.created_at
                if reference is None:
                    continue
                if reference.tzinfo is None:
                    reference = reference.replace(tzinfo=now.tzinfo)
                # A young record has not had a chance to prove useful yet.
                if now - reference < min_age:
                    continue
                if effective_score(record, now=now) < threshold:
                    to_archive.append(record.id)

            if len(batch) < 500:
                break

        if to_archive:
            stats.archived = await self.records.set_status(
                to_archive, RecordStatus.ARCHIVED
            )
            logger.info(
                "Archived %d records below relevance %.2f", stats.archived, threshold
            )

    # ── conflict reconciliation ─────────────────────────────────────────

    async def reconcile_conflicts(self, stats: MaintenanceStats) -> None:
        """
        Close duplicate `dedup_key` slots.

        The writer supersedes on conflict, so this should find nothing. Two
        concurrent writers can still both pass the check, and this is what
        notices the invariant has actually broken rather than assuming it holds.
        """
        pairs = await self.records.duplicate_dedup_keys()
        for owner_id, dedup_key in pairs:
            records = await self.records.find_by_dedup_key(owner_id, dedup_key)
            if len(records) < 2:
                continue

            # Newest wins; find_by_dedup_key already returns newest first.
            stale = [r.id for r in records[1:]]
            closed = await self.records.set_status(stale, RecordStatus.SUPERSEDED)
            stats.conflicts_resolved += closed
            logger.warning(
                "Reconciled %d duplicate active records for owner=%s key=%s",
                closed, owner_id, dedup_key,
            )

    # ── semantic de-duplication ─────────────────────────────────────────

    async def sweep_duplicates(self, stats: MaintenanceStats) -> None:
        """
        Merge near-identical memories.

        Deferred to here rather than done at ingest because it needs a vector,
        and paying for an embedding per candidate on the write path would put
        provider latency back into a turn — the exact cost Phase 3 moved out.

        The survivor keeps the union of both provenance chains, so "why do you
        know this?" still resolves after a merge.
        """
        if self.embedder is None or self.vectors is None:
            return

        threshold = settings.memory_dedup_similarity_threshold
        candidates = [
            record for record in await self.records.iter_active(limit=200)
            if record.kind in DEDUPABLE_KINDS
        ]
        if len(candidates) < 2:
            return

        vectors = await self.embedder([r.content for r in candidates])
        if len(vectors) != len(candidates):
            logger.warning("Dedup sweep: embedding batch mismatch, skipping")
            return

        merged: Set[UUID] = set()

        for record, vector in zip(candidates, vectors):
            if record.id in merged:
                continue

            hits = await self.vectors.search(
                record.owner_id, vector, kinds=[record.kind], limit=5
            )
            for hit in hits:
                if hit.record_id == record.id or hit.record_id in merged:
                    continue
                if hit.score < threshold:
                    continue

                other = await self.records.get(record.owner_id, hit.record_id)
                if other is None or not other.is_active or other.kind is not record.kind:
                    continue

                keeper, absorbed = self._choose_survivor(record, other)
                await self._merge(keeper, absorbed)
                merged.add(absorbed.id)
                stats.duplicates_merged += 1

    def _choose_survivor(self, a: MemoryRecord, b: MemoryRecord):
        """Keep the better-evidenced record; ties break toward the newer one."""
        a_rank = (a.confidence, a.importance, a.created_at)
        b_rank = (b.confidence, b.importance, b.created_at)
        return (a, b) if a_rank >= b_rank else (b, a)

    async def _merge(self, keeper: MemoryRecord, absorbed: MemoryRecord) -> None:
        """Fold provenance into the survivor and close the absorbed record."""
        provenance = list(dict.fromkeys(
            [*keeper.derived_from, *absorbed.derived_from, absorbed.id]
        ))
        updated = keeper.superseding(
            derived_from=provenance,
            # A fact corroborated twice is better evidenced than either alone.
            confidence=max(keeper.confidence, absorbed.confidence),
            importance=max(keeper.importance, absorbed.importance),
        )
        await self.records.supersede(keeper, updated)
        await self.records.set_status([absorbed.id], RecordStatus.SUPERSEDED)

    # ── guest collection ────────────────────────────────────────────────

    async def collect_abandoned_guests(self, stats: MaintenanceStats) -> None:
        """
        Purge anonymous partitions nobody will return to.

        Every recruiter visit mints a `guest-<uuid>` identity and writes memory
        under it. None of those visitors ever come back, so without collection
        abandoned guest data becomes the largest thing in the store — pure cost,
        zero value.
        """
        cutoff = utcnow() - timedelta(days=settings.memory_guest_retention_days)

        for owner_id, last_active in await self.records.owner_activity():
            if not owner_id.startswith("guest-"):
                continue
            if last_active is None:
                continue
            if last_active.tzinfo is None:
                last_active = last_active.replace(tzinfo=cutoff.tzinfo)
            if last_active >= cutoff:
                continue

            deleted = await self.forget_owner(owner_id)
            stats.guests_purged += 1
            stats.records_deleted += deleted
            logger.info(
                "Purged abandoned guest partition %s (%d records)", owner_id, deleted
            )

    # ── forgetting ──────────────────────────────────────────────────────

    async def forget_record(self, owner_id: str, record_id: UUID) -> int:
        """
        Erase a record and everything distilled from it.

        The cascade is the point. A consolidated memory derived from a fact the
        user asked to delete would outlive that deletion — the fact would be
        gone from the store yet still reachable in the summary of it, which is
        a deletion that did not delete.
        """
        doomed: List[UUID] = []
        seen: Set[UUID] = set()
        frontier = [record_id]

        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            doomed.append(current)

            for child in await self.records.find_derived_from(owner_id, current):
                if child.id not in seen:
                    frontier.append(child.id)

        if self.vectors is not None:
            try:
                await self.vectors.delete(doomed)
            except Exception as exc:
                # A surviving vector would keep surfacing a record that no
                # longer exists, so this is worth shouting about.
                logger.error(
                    "Could not delete vectors for erased records: %s", exc
                )

        return await self.records.hard_delete(doomed)

    async def forget_owner(self, owner_id: str) -> int:
        """Erase everything belonging to one owner."""
        deleted = 0
        while True:
            batch = await self.records.list(
                owner_id,
                statuses=[
                    RecordStatus.ACTIVE, RecordStatus.SUPERSEDED,
                    RecordStatus.ARCHIVED, RecordStatus.DELETED,
                ],
                limit=500,
            )
            if not batch:
                break

            ids = [record.id for record in batch]
            if self.vectors is not None:
                try:
                    await self.vectors.delete(ids)
                except Exception as exc:
                    logger.error("Could not delete vectors for owner=%s: %s", owner_id, exc)

            deleted += await self.records.hard_delete(ids)
            if len(batch) < 500:
                break

        return deleted


# Singleton instance
memory_maintenance = MemoryMaintenance()
