"""
The memory worker: drains the outbox, then embeds what it produced.

Every stage is wrapped so that one bad event cannot stall the queue and one bad
cycle cannot kill the loop. An unattended background worker that dies on the
first malformed payload is worse than no worker, because nothing reports it —
memory simply stops improving and no one notices for a month.

See docs/MEMORY_ARCHITECTURE.md §3.5.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import settings
from app.memory.cognition.embedder import EmbeddingPass, embedding_pass
from app.memory.cognition.extractor import MemoryExtractor, memory_extractor
from app.memory.cognition.ingest import MemoryIngestor, memory_ingestor
from app.memory.cognition.maintenance import (
    MemoryMaintenance,
    memory_maintenance,
)
from app.memory.cognition.summarizer import (
    ConversationSummarizer,
    conversation_summarizer,
)
from app.memory.events import EventStatus, EventType, MemoryEvent
from app.memory.kinds import SourceType

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    """Outcome of one worker cycle."""

    groups: int = 0
    events: int = 0
    created: int = 0
    duplicates: int = 0
    superseded: int = 0
    rejected: int = 0
    failed_groups: int = 0
    embedded: int = 0
    summarised: int = 0
    maintained: bool = False

    def summary(self) -> Dict[str, Any]:
        return {
            "groups": self.groups,
            "events": self.events,
            "created": self.created,
            "duplicates": self.duplicates,
            "superseded": self.superseded,
            "rejected": self.rejected,
            "failed_groups": self.failed_groups,
            "embedded": self.embedded,
            "summarised": self.summarised,
            "maintained": self.maintained,
        }

    @property
    def did_work(self) -> bool:
        return bool(self.events or self.embedded or self.summarised)


class MemoryWorker:
    """Drains ready event groups and runs the embedding pass."""

    def __init__(
        self,
        event_queue=None,
        *,
        extractor: Optional[MemoryExtractor] = None,
        ingestor: Optional[MemoryIngestor] = None,
        embedder: Optional[EmbeddingPass] = None,
        summarizer: Optional[ConversationSummarizer] = None,
        maintenance: Optional[MemoryMaintenance] = None,
    ):
        if event_queue is None:
            from app.memory.stores.postgres_event_queue import postgres_event_queue
            event_queue = postgres_event_queue

        self.events = event_queue
        self.extractor = extractor or memory_extractor
        self.ingestor = ingestor or memory_ingestor
        self.embedder = embedder or embedding_pass
        self.summarizer = summarizer or conversation_summarizer
        self.maintenance = maintenance or memory_maintenance
        # Maintenance sweeps the whole store, so it runs on its own far
        # slower clock rather than once per extraction cycle.
        self._last_maintenance = 0.0

    async def run_once(self) -> WorkerStats:
        """One full cycle: extraction, then embedding. Never raises."""
        stats = WorkerStats()

        try:
            groups = await self.events.ready_groups(
                batch_size=settings.memory_extraction_batch_size,
                idle_seconds=settings.memory_extraction_idle_flush_seconds,
            )
        except Exception as exc:
            logger.warning("Could not poll memory event queue: %s", exc)
            groups = []

        for group in groups:
            try:
                await self._process_group(group.group_key, stats)
            except Exception as exc:
                # Isolated per group: a malformed conversation must not stop
                # every other user's memory from being extracted.
                stats.failed_groups += 1
                logger.error(
                    "Memory extraction failed for group=%s: %s",
                    group.group_key, exc, exc_info=True,
                )

        try:
            embed_stats = await self.embedder.run_once(
                limit=settings.memory_embedding_batch_size
            )
            stats.embedded = embed_stats.embedded
        except Exception as exc:
            logger.warning("Embedding pass failed: %s", exc)

        try:
            summary_stats = await self.summarizer.run_once()
            stats.summarised = summary_stats.summarised
        except Exception as exc:
            logger.warning("Conversation summarisation failed: %s", exc)

        if self._maintenance_due():
            self._last_maintenance = time.monotonic()
            try:
                await self.maintenance.run_once()
                stats.maintained = True
            except Exception as exc:
                logger.warning("Memory maintenance failed: %s", exc)

        if stats.did_work:
            logger.info("Memory worker cycle: %s", stats.summary())
        return stats

    def _maintenance_due(self) -> bool:
        """
        True once the maintenance interval has elapsed.

        Deliberately not run every cycle: these are full-store sweeps, and at a
        30-second poll they would dominate the worker's cost while changing
        almost nothing between runs.
        """
        if not settings.memory_maintenance_enabled:
            return False
        elapsed = time.monotonic() - self._last_maintenance
        return elapsed >= settings.memory_maintenance_interval_seconds

    async def _process_group(self, group_key: str, stats: WorkerStats) -> None:
        claimed = await self.events.claim_group(group_key)
        if not claimed:
            return

        stats.groups += 1
        stats.events += len(claimed)
        event_ids = [event.id for event in claimed]

        try:
            owner_id = claimed[0].owner_id
            turns = self._collect_turns(claimed)

            if not turns:
                # Nothing extractable — an empty exchange, or a document event
                # this phase does not yet handle. Not a failure.
                await self.events.mark_done(event_ids, EventStatus.SKIPPED)
                return

            candidates = await self.extractor.extract(owner_id, turns)
            if not candidates:
                # A conversation that yielded no durable fact is the normal
                # outcome for small talk, and is deliberately distinct from
                # failure so it is never retried.
                await self.events.mark_done(event_ids, EventStatus.SKIPPED)
                return

            result = await self.ingestor.ingest(
                owner_id,
                candidates,
                source_type=SourceType.CHAT,
                source_ref=f"session:{group_key}",
                occurred_at=claimed[-1].created_at,
            )

            stats.created += result.created
            stats.duplicates += result.duplicates
            stats.superseded += result.superseded
            stats.rejected += result.rejected

            await self.events.mark_done(event_ids, EventStatus.DONE)

        except Exception as exc:
            # Return the events to PENDING for retry (or park them once the
            # attempt ceiling is reached) rather than losing what was said.
            await self.events.mark_failed(event_ids, str(exc))
            raise

    def _collect_turns(self, events: List[MemoryEvent]) -> List[Dict[str, Any]]:
        """Flatten turn events into a single chronological transcript."""
        turns: List[Dict[str, Any]] = []
        for event in events:
            if event.event_type is not EventType.TURN:
                continue
            payload = event.payload or {}
            user_text = (payload.get("user") or "").strip()
            assistant_text = (payload.get("assistant") or "").strip()
            if user_text:
                turns.append({"role": "user", "content": user_text})
            if assistant_text:
                turns.append({"role": "assistant", "content": assistant_text})
        return turns

    async def run_forever(self, stop: Optional[asyncio.Event] = None) -> None:
        """
        Poll until stopped.

        Backs off after an idle cycle so a quiet system is not polling the
        database every few seconds for no reason.
        """
        stop = stop or asyncio.Event()
        interval = settings.memory_worker_poll_seconds
        logger.info("Memory worker started (poll interval %.0fs)", interval)

        while not stop.is_set():
            try:
                stats = await self.run_once()
                delay = interval if not stats.did_work else min(interval, 5.0)
            except Exception as exc:
                # run_once already swallows its own failures; this is the last
                # line of defence for anything unforeseen.
                logger.error("Memory worker cycle crashed: %s", exc, exc_info=True)
                delay = interval

            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                continue

        logger.info("Memory worker stopped")


# Singleton instance
memory_worker = MemoryWorker()
