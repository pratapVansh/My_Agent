"""
Backfill the unified `memory_records` table from the legacy memory tables.

    python scripts/migrate_memory_v2.py                  # dry run (default)
    python scripts/migrate_memory_v2.py --apply
    python scripts/migrate_memory_v2.py --apply --owner vansh
    python scripts/migrate_memory_v2.py --apply --skip-documents

It also creates indexes that `create_all` cannot add to an already-existing
table (see `ensure_indexes`), so running it is how a Phase 1 deployment gains
the Phase 2 full-text index.

Sources:
    user_profile     → identity | preference | goal | semantic
    episodic_memory  → episodic
    tool_memory      → procedural   (only outcome_quality='good')
    chat_history     → conversations + turns, grouped by session_id
    Qdrant chunks    → document     (resume_chunks, skills_chunks, projects_chunks)

Safety properties, all of which the redesign's migration rules require:

* **Dry-run by default.** Writing requires `--apply`. Running the script by
  accident reports and changes nothing.
* **Idempotent.** `RecordStore.add()` returns the existing record when the
  (owner_id, kind, content_hash) triple already exists, so a second run creates
  nothing. That property is enforced by a partial unique index, not by this
  script's diligence.
* **Resumable.** Records are inserted one statement at a time rather than in a
  bulk batch, so an interrupted run leaves everything written so far intact and
  a re-run continues from there.
* **Non-destructive.** Nothing is deleted or altered. The legacy tables remain
  authoritative and continue to serve every read until Phase 2 cuts over.

Original timestamps are preserved. Backfilling everything with today's date
would make the entire history look simultaneous and destroy recency ranking
before it is ever switched on.
"""
import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.db.session import async_session_maker  # noqa: E402
from app.memory.kinds import MemoryKind, SourceType, Visibility  # noqa: E402
from app.memory.models import (  # noqa: E402
    ChatHistory,
    EpisodicMemory,
    ToolMemory,
    UserProfile,
)
from app.memory.record import MemoryRecord, utcnow  # noqa: E402
from app.memory.conversations import conversation_repository, derive_title  # noqa: E402
from app.memory.stores.postgres_record_store import postgres_record_store  # noqa: E402
from app.memory.writer import (  # noqa: E402
    BASE_IMPORTANCE,
    classify_profile_key,
    render_episode,
    render_profile_fact,
    render_tool_outcome,
)


@dataclass
class Stats:
    """Outcome of backfilling one source."""

    source: str
    scanned: int = 0
    created: int = 0
    existing: int = 0
    skipped: int = 0
    failed: int = 0
    notes: List[str] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"  {self.source:<18} scanned={self.scanned:<6} created={self.created:<6} "
            f"already-present={self.existing:<6} skipped={self.skipped:<5} failed={self.failed}"
        )


async def ensure_indexes(dry_run: bool) -> Stats:
    """
    Create indexes that `create_all` will not add to an existing table.

    `Base.metadata.create_all` creates missing *tables*; it does not add newly
    declared indexes to a table that already exists. A deployment that ran
    Phase 1 therefore has `memory_records` without the Phase 2 full-text index.

    Lexical search still returns correct results without it — Postgres falls
    back to a sequential scan — so this is a performance repair, never a
    correctness one, which is why it is safe to run at any time.
    """
    stats = Stats("indexes")
    statements = {
        "ix_memory_records_content_fts": (
            "CREATE INDEX IF NOT EXISTS ix_memory_records_content_fts "
            "ON memory_records USING gin (to_tsvector('english', content))"
        ),
    }

    from sqlalchemy import text as sql_text

    for name, ddl in statements.items():
        stats.scanned += 1
        if dry_run:
            stats.notes.append(f"would ensure {name}")
            continue
        try:
            async with async_session_maker() as session:
                await session.execute(sql_text(ddl))
                await session.commit()
            stats.created += 1
        except Exception as exc:
            stats.failed += 1
            stats.notes.append(f"{name}: {exc}")
    return stats


def _when(*candidates) -> "object":
    """
    First non-null timestamp, falling back to now.

    Legacy rows written before a column gained `server_default` can carry NULL
    timestamps, and the record columns are NOT NULL. Preserving the original
    moment matters — backfilling everything as "now" would make the whole
    history look simultaneous and destroy recency ranking before it is ever
    switched on — so `now` is genuinely the last resort.
    """
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return utcnow()


async def _persist(record: MemoryRecord, stats: Stats, dry_run: bool) -> None:
    """Insert one record, counting the outcome. Never raises."""
    if dry_run:
        existing = await postgres_record_store.find_by_content_hash(
            record.owner_id, record.kind, record.content_hash
        )
        if existing is not None:
            stats.existing += 1
        else:
            stats.created += 1
        return

    try:
        stored = await postgres_record_store.add(record)
        # add() returns the pre-existing record when the hash already exists;
        # comparing ids is how we tell an insert from a no-op.
        if stored.id == record.id:
            stats.created += 1
        else:
            stats.existing += 1
    except Exception as exc:
        stats.failed += 1
        stats.notes.append(f"{record.kind.value}: {exc}")


# ── Sources ─────────────────────────────────────────────────────────────

async def backfill_profile_facts(owner: Optional[str], dry_run: bool) -> Stats:
    stats = Stats("profile facts")
    async with async_session_maker() as session:
        query = select(UserProfile)
        if owner:
            query = query.where(UserProfile.user_id == owner)
        rows = (await session.execute(query)).scalars().all()

    for row in rows:
        stats.scanned += 1
        kind = classify_profile_key(row.key)
        content = render_profile_fact(row.key, row.value)
        if not content.strip():
            stats.skipped += 1
            continue

        record = MemoryRecord(
            owner_id=row.user_id,
            kind=kind,
            content=content,
            structured={"key": row.key, "value": row.value, "source": row.source},
            importance=BASE_IMPORTANCE[kind],
            confidence=row.confidence if row.confidence is not None else 1.0,
            occurred_at=row.created_at,
            valid_from=_when(row.created_at),
            created_at=_when(row.created_at),
            updated_at=_when(row.updated_at, row.created_at),
            source_type=SourceType.CHAT if row.source == "inferred" else SourceType.SYSTEM,
            source_ref=f"user_profile:{row.key}",
            dedup_key=f"profile:{(row.key or '').strip().lower()}",
            visibility=Visibility.PRIVATE,
        )
        await _persist(record, stats, dry_run)
    return stats


async def backfill_episodes(owner: Optional[str], dry_run: bool) -> Stats:
    stats = Stats("episodes")
    async with async_session_maker() as session:
        query = select(EpisodicMemory)
        if owner:
            query = query.where(EpisodicMemory.user_id == owner)
        rows = (await session.execute(query)).scalars().all()

    for row in rows:
        stats.scanned += 1
        content = render_episode(row.user_summary, row.agent_summary, row.agent_used)
        if not content.strip():
            stats.skipped += 1
            continue

        importance = BASE_IMPORTANCE[MemoryKind.EPISODIC]
        if row.outcome == "failed":
            importance -= 0.15

        record = MemoryRecord(
            owner_id=row.user_id,
            kind=MemoryKind.EPISODIC,
            content=content,
            structured={
                "user_summary": row.user_summary,
                "agent_summary": row.agent_summary,
                "agent_used": row.agent_used,
                "intent": row.intent,
                "outcome": row.outcome,
                "session_id": row.session_id,
            },
            importance=importance,
            occurred_at=row.created_at,
            valid_from=_when(row.created_at),
            created_at=_when(row.created_at),
            updated_at=_when(row.created_at),
            source_type=SourceType.CHAT,
            source_ref=f"session:{row.session_id}",
        )
        await _persist(record, stats, dry_run)
    return stats


async def backfill_tool_memories(owner: Optional[str], dry_run: bool) -> Stats:
    stats = Stats("tool outcomes")
    async with async_session_maker() as session:
        # Only successful strategies are knowledge worth reusing — the live
        # write path applies the same filter.
        query = select(ToolMemory).where(ToolMemory.outcome_quality == "good")
        if owner:
            query = query.where(ToolMemory.user_id == owner)
        rows = (await session.execute(query)).scalars().all()

    for row in rows:
        stats.scanned += 1
        content = render_tool_outcome(
            row.agent_name, row.tool_name, row.inputs_summary, row.key_insight
        )
        record = MemoryRecord(
            owner_id=row.user_id,
            kind=MemoryKind.PROCEDURAL,
            content=content,
            structured={
                "agent_name": row.agent_name,
                "tool_name": row.tool_name,
                "inputs_summary": row.inputs_summary,
                "key_insight": row.key_insight,
            },
            importance=BASE_IMPORTANCE[MemoryKind.PROCEDURAL],
            occurred_at=row.created_at,
            valid_from=_when(row.created_at),
            created_at=_when(row.created_at),
            updated_at=_when(row.created_at),
            source_type=SourceType.SYSTEM,
            source_ref=f"tool:{row.agent_name}:{row.tool_name}",
        )
        await _persist(record, stats, dry_run)
    return stats


async def backfill_documents(owner: Optional[str], dry_run: bool) -> Stats:
    """
    Snapshot existing Qdrant chunks as `document` records.

    Point-in-time only: new uploads are not yet dual-written, so this needs
    re-running after further ingestion until Phase 2 wires document ingestion
    into the record store. Re-running is safe by construction.
    """
    stats = Stats("documents")
    from app.memory.long_term_memory_qdrant import long_term_memory_qdrant as lt
    from app.services.qdrant_service import qdrant_service

    for label, collection in lt.collections.items():
        try:
            filters = {"user_id": owner} if owner else None
            points = await qdrant_service.scroll_collection(
                collection_name=collection, filter_conditions=filters
            )
        except Exception as exc:
            stats.notes.append(f"{collection}: unreachable ({exc})")
            continue

        for point in points:
            payload = point.get("payload") or {}
            text = (payload.get("text") or "").strip()
            point_owner = payload.get("user_id")
            if not text or not point_owner:
                stats.skipped += 1
                continue

            stats.scanned += 1
            record = MemoryRecord(
                owner_id=point_owner,
                kind=MemoryKind.DOCUMENT,
                content=text,
                structured={
                    "document_id": payload.get("parent_id"),
                    "semantic_type": payload.get("semantic_type", label),
                    "chunk_index": payload.get("chunk_index", 0),
                    "source_file": payload.get("source_file"),
                    "collection": collection,
                },
                importance=BASE_IMPORTANCE[MemoryKind.DOCUMENT],
                source_type=SourceType.UPLOAD,
                source_ref=str(payload.get("parent_id") or collection),
            )
            await _persist(record, stats, dry_run)
    return stats



async def backfill_conversations(owner: Optional[str], dry_run: bool) -> Stats:
    """
    Group existing `chat_history` into conversations and turns (Phase 4).

    Rows are grouped by (user_id, session_id) — the session id becomes the
    conversation's key directly, so a browser still holding an old id in
    localStorage resumes the thread it already had.

    Ordering comes from `created_at`, the only ordering `chat_history` ever
    had. Sequence numbers are assigned densely on the way in, which is what
    gives the new table the ordering guarantee the old one lacked.
    """
    stats = Stats("conversations")

    async with async_session_maker() as session:
        query = select(ChatHistory).order_by(ChatHistory.created_at)
        if owner:
            query = query.where(ChatHistory.user_id == owner)
        rows = (await session.execute(query)).scalars().all()

    grouped: Dict[tuple, List[Any]] = {}
    for row in rows:
        grouped.setdefault((row.user_id, row.session_id), []).append(row)

    for (user_id, session_id), messages in grouped.items():
        stats.scanned += 1

        existing = await conversation_repository.get(session_id, user_id)
        if existing is not None and existing.turn_count:
            stats.existing += 1
            continue

        if dry_run:
            stats.created += 1
            continue

        try:
            first_user = next(
                (m.content for m in messages if m.role == "user"), ""
            )
            await conversation_repository.ensure(
                session_id, user_id, title=derive_title(first_user) if first_user else None
            )
            for message in messages:
                await conversation_repository.append_turn(
                    conversation_id=session_id,
                    owner_id=user_id,
                    role=message.role,
                    content=message.content,
                    modality="text",
                    agent=(message.meta_data or {}).get("agent"),
                    intent=(message.meta_data or {}).get("intent"),
                )
            stats.created += 1
        except Exception as exc:
            stats.failed += 1
            stats.notes.append(f"{session_id}: {exc}")

    return stats


# ── Entry point ─────────────────────────────────────────────────────────

async def run(owner: Optional[str], dry_run: bool, skip_documents: bool) -> int:
    mode = "DRY RUN — nothing will be written" if dry_run else "APPLYING CHANGES"
    print(f"\nmemory_records backfill · {mode}")
    print(f"owner filter: {owner or 'all owners'}\n")

    before = await postgres_record_store.count()
    print(f"memory_records rows before: {before}\n")

    results = [
        await ensure_indexes(dry_run),
        await backfill_profile_facts(owner, dry_run),
        await backfill_episodes(owner, dry_run),
        await backfill_tool_memories(owner, dry_run),
        await backfill_conversations(owner, dry_run),
    ]
    if skip_documents:
        print("  (documents skipped by request)")
    else:
        results.append(await backfill_documents(owner, dry_run))

    print("Results:")
    for stats in results:
        print(stats.line())
        for note in stats.notes[:5]:
            print(f"      ! {note}")

    after = await postgres_record_store.count()
    total_created = sum(s.created for s in results)
    total_failed = sum(s.failed for s in results)

    print(f"\nmemory_records rows after: {after}")
    if dry_run:
        print(f"would create: {total_created}")
        print("\nRe-run with --apply to write these records.")
    else:
        print(f"created: {total_created}   failed: {total_failed}")
        if total_failed:
            print("\nSome records failed. The migration is idempotent — fix the")
            print("cause and re-run; already-written records will be skipped.")

    return 1 if total_failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill memory_records from the legacy memory tables."
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write records. Without this the script only reports.",
    )
    parser.add_argument(
        "--owner", default=None,
        help="Restrict the backfill to a single owner_id.",
    )
    parser.add_argument(
        "--skip-documents", action="store_true",
        help="Skip the Qdrant document snapshot (useful when Qdrant is unreachable).",
    )
    args = parser.parse_args()

    try:
        return asyncio.run(
            run(owner=args.owner, dry_run=not args.apply, skip_documents=args.skip_documents)
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Already-written records are intact; re-run to continue.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
