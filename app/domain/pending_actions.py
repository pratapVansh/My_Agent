"""
Where an action waits, when waiting has to outlive the process.

The gateway held pending actions in a dictionary. That was correct while a
confirmation could only be answered by the process that asked — but it made two
ordinary deployments broken in a quiet way. A restart between "shall I send
this?" and "yes" lost the action. Two workers behind a load balancer lost it
roughly half the time, because the approval landed on the replica that had
never heard of it. Both failed closed, which is the right direction and still
the wrong behaviour.

Moving the wait into Postgres fixes both, and brings one property a dictionary
could not offer at all:

**Claiming is atomic.** `DELETE ... RETURNING` either returns the row to you or
returns nothing, and exactly one caller can win it — across threads, across
processes, across machines. Single-use stops being a flag someone might read
before another writer sets it, and becomes something the database decides. The
in-memory implementation reproduces this under a lock so tests exercise the
same semantics.

What is stored is a tool *name* and its arguments. Never a callable: a row that
can turn into executable code is a remote-execution primitive, and the fact
that it sits behind an audit trail does not make it less of one. Reconstruction
goes through `app.agents.confirmable_tools`, so the code that eventually runs
comes from the application and the row only chooses among what the application
already offers.

Rows are deleted on claim, on cancellation, and on expiry. The permanent record
of what happened lives in `action_audit`; this table holds only what is needed
to still act, for as long as acting is still possible.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Protocol

from sqlalchemy import delete, select

from app.db.session import async_session_maker
from app.domain.models import PendingActionRecord

logger = logging.getLogger(__name__)


class PendingStoreError(RuntimeError):
    """The store could not be read or written. Callers must not proceed."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class StoredAction:
    """One action awaiting a decision, as persisted."""

    id: str
    token_hash: str
    audit_id: str
    owner_id: str
    conversation_id: str
    tool: str
    effect: str
    arguments: Mapping[str, Any]
    preview: str
    content_hash: str
    idempotency_key: str
    created_at: datetime
    expires_at: datetime

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        return (now or _utcnow()) >= (_as_utc(self.expires_at) or _utcnow())

    def summary(self) -> Dict[str, Any]:
        """Log form — identifiers only, never arguments or preview text."""
        return {
            "tool": self.tool,
            "effect": self.effect,
            "audit_id": self.audit_id[:8],
            "content_hash": self.content_hash[:12],
            "expires_at": self.expires_at.isoformat(),
        }


class PendingActionStore(Protocol):
    """What the gateway needs from a durable waiting room."""

    async def put(self, action: StoredAction) -> None: ...
    async def peek(self, token_hash: str) -> Optional[StoredAction]: ...
    async def claim(self, token_hash: str) -> Optional[StoredAction]: ...
    async def drop(self, token_hash: str) -> Optional[StoredAction]: ...
    async def list_for(
        self, conversation_id: str, owner_id: str
    ) -> List[StoredAction]: ...
    async def take_expired(self, limit: int = 100) -> List[StoredAction]: ...


# ── Postgres ─────────────────────────────────────────────────────────────────

class PostgresPendingActionStore:
    """The production waiting room. Every failure is raised, never swallowed."""

    def __init__(self, session_maker=None) -> None:
        self._session_maker = session_maker or async_session_maker

    async def put(self, action: StoredAction) -> None:
        try:
            async with self._session_maker() as session:
                session.add(PendingActionRecord(
                    id=uuid.UUID(action.id),
                    token_hash=action.token_hash,
                    audit_id=uuid.UUID(action.audit_id),
                    owner_id=action.owner_id,
                    conversation_id=action.conversation_id or None,
                    tool=action.tool,
                    effect=action.effect,
                    arguments=dict(action.arguments),
                    preview=action.preview,
                    content_hash=action.content_hash,
                    idempotency_key=action.idempotency_key,
                    expires_at=action.expires_at,
                ))
                await session.commit()
        except Exception as exc:
            logger.error("Pending store write failed for %s: %s", action.summary(), exc)
            raise PendingStoreError(str(exc)) from exc

    async def peek(self, token_hash: str) -> Optional[StoredAction]:
        try:
            async with self._session_maker() as session:
                result = await session.execute(
                    select(PendingActionRecord)
                    .where(PendingActionRecord.token_hash == token_hash)
                    .limit(1)
                )
                row = result.scalars().first()
                return _from_row(row) if row else None
        except Exception as exc:
            logger.error("Pending store read failed: %s", exc)
            raise PendingStoreError(str(exc)) from exc

    async def claim(self, token_hash: str) -> Optional[StoredAction]:
        """
        Take the action, atomically.

        `DELETE ... RETURNING` is the whole single-use guarantee: exactly one
        caller can receive a given row, decided by Postgres rather than by
        ordering luck between two workers.
        """
        try:
            async with self._session_maker() as session:
                result = await session.execute(
                    delete(PendingActionRecord)
                    .where(PendingActionRecord.token_hash == token_hash)
                    .returning(PendingActionRecord)
                )
                row = result.scalars().first()
                await session.commit()
                return _from_row(row) if row else None
        except Exception as exc:
            logger.error("Pending store claim failed: %s", exc)
            raise PendingStoreError(str(exc)) from exc

    async def drop(self, token_hash: str) -> Optional[StoredAction]:
        return await self.claim(token_hash)

    async def list_for(
        self, conversation_id: str, owner_id: str
    ) -> List[StoredAction]:
        try:
            async with self._session_maker() as session:
                result = await session.execute(
                    select(PendingActionRecord).where(
                        PendingActionRecord.owner_id == owner_id,
                        PendingActionRecord.conversation_id == (conversation_id or None),
                        PendingActionRecord.expires_at > _utcnow(),
                    ).order_by(PendingActionRecord.created_at)
                )
                return [_from_row(row) for row in result.scalars().all()]
        except Exception as exc:
            logger.error("Pending store listing failed: %s", exc)
            raise PendingStoreError(str(exc)) from exc

    async def take_expired(self, limit: int = 100) -> List[StoredAction]:
        try:
            async with self._session_maker() as session:
                result = await session.execute(
                    delete(PendingActionRecord)
                    .where(PendingActionRecord.expires_at <= _utcnow())
                    .returning(PendingActionRecord)
                )
                rows = result.scalars().all()[:limit]
                await session.commit()
                return [_from_row(row) for row in rows]
        except Exception as exc:
            logger.error("Pending store expiry sweep failed: %s", exc)
            raise PendingStoreError(str(exc)) from exc


def _from_row(row: PendingActionRecord) -> StoredAction:
    return StoredAction(
        id=str(row.id),
        token_hash=row.token_hash,
        audit_id=str(row.audit_id),
        owner_id=row.owner_id,
        conversation_id=row.conversation_id or "",
        tool=row.tool,
        effect=row.effect,
        arguments=dict(row.arguments or {}),
        preview=row.preview or "",
        content_hash=row.content_hash,
        idempotency_key=row.idempotency_key,
        created_at=_as_utc(row.created_at) or _utcnow(),
        expires_at=_as_utc(row.expires_at) or _utcnow(),
    )


# ── In-memory (tests only) ───────────────────────────────────────────────────

class InMemoryPendingActionStore:
    """
    A test double, not a deployment option.

    `claim` holds a lock across the read-and-remove so it is atomic in the same
    way the SQL is. A test double that let two callers claim one row would let
    a concurrency bug pass here and fail in production.
    """

    def __init__(self) -> None:
        self._rows: Dict[str, StoredAction] = {}
        self._lock = threading.Lock()
        self.failing = False

    def _maybe_fail(self, operation: str) -> None:
        if self.failing:
            raise PendingStoreError(f"simulated pending-store outage during {operation}")

    async def put(self, action: StoredAction) -> None:
        self._maybe_fail("put")
        with self._lock:
            self._rows[action.token_hash] = action

    async def peek(self, token_hash: str) -> Optional[StoredAction]:
        self._maybe_fail("peek")
        return self._rows.get(token_hash)

    async def claim(self, token_hash: str) -> Optional[StoredAction]:
        self._maybe_fail("claim")
        with self._lock:
            return self._rows.pop(token_hash, None)

    async def drop(self, token_hash: str) -> Optional[StoredAction]:
        return await self.claim(token_hash)

    async def list_for(self, conversation_id: str, owner_id: str) -> List[StoredAction]:
        self._maybe_fail("list_for")
        now = _utcnow()
        rows = [
            a for a in list(self._rows.values())
            if a.owner_id == owner_id
            and a.conversation_id == conversation_id
            and not a.is_expired(now)
        ]
        rows.sort(key=lambda a: a.created_at)
        return rows

    async def take_expired(self, limit: int = 100) -> List[StoredAction]:
        self._maybe_fail("take_expired")
        now = _utcnow()
        with self._lock:
            expired = [a for a in self._rows.values() if a.is_expired(now)][:limit]
            for action in expired:
                self._rows.pop(action.token_hash, None)
        return expired

    # Test conveniences ------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._rows)

    def reset(self) -> None:
        with self._lock:
            self._rows.clear()
        self.failing = False


pending_action_store: PendingActionStore = PostgresPendingActionStore()
"""The production waiting room. Tests swap this on the gateway, never here."""


__all__ = [
    "InMemoryPendingActionStore",
    "PendingActionStore",
    "PendingStoreError",
    "PostgresPendingActionStore",
    "StoredAction",
    "pending_action_store",
]
