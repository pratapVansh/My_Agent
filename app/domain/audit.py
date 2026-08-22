"""
The permanent record of every consequential action.

The gateway can already stop an email from being sent. It cannot yet answer the
question that follows a surprise: *did one go out, who approved it, and was it
the thing they were shown?* Until now the only trace of a completed send was a
log line and an in-process dictionary that a restart erased — which meant the
system could enforce a rule it could not afterwards prove it had followed.

This is the record that closes that gap, and it does two jobs at once:

**Evidence.** A row exists from the moment an action is *proposed*, not from
the moment it succeeds. An action prepared and never approved is as visible as
one that was sent, because "the assistant tried to email my professor" and "an
email reached my professor" are different events and only the first is
invisible if you record outcomes alone.

**Durable idempotency.** The unique partial index on executed rows is the thing
that makes a retry safe across a restart or a second replica. The in-process
set the gateway used could not survive either, so "already sent" was a hope
rather than a fact. Now it is a constraint the database enforces.

Two deliberate constraints on what is stored:

*Digests, never values.* Arguments, preview text and the confirmation token are
each reduced to SHA-256. That is sufficient to prove that what executed is what
was approved, and insufficient to reconstruct a recipient or a message body
from a backup. An audit log that leaks the contents of every email is a worse
liability than no audit log.

*Failure is loud.* Every method raises. The gateway is built to fail closed on
an audit error, and a store that silently swallowed a write would turn that
guarantee into its opposite — an action that executed with no record of having
done so.

Two implementations behind one interface, matching `app/memory/stores/`:
Postgres for production, in-memory for tests. The in-memory one is a test
double, not a deployment option; `audit_log` is the Postgres one.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from sqlalchemy import select, update

from app.db.session import async_session_maker
from app.domain.models import ActionAudit

logger = logging.getLogger(__name__)


class AuditStatus(str, Enum):
    """Where an action got to."""

    PENDING = "pending"
    """Proposed and described. Nothing has happened."""

    CONFIRMED = "confirmed"
    """The user approved it. Execution is imminent."""

    EXECUTED = "executed"
    """The effect was performed — or, after a crash mid-call, may have been.

    Claimed *before* the call rather than after, so a process that dies while
    SMTP is open leaves the slot taken. For an irreversible effect, "possibly
    sent, refuses to retry" is the safe direction of a wrong guess."""

    FAILED = "failed"
    """The call returned an error. The idempotency claim is released, because
    a send that demonstrably did not happen should be retryable."""

    CANCELLED = "cancelled"
    """The user refused it."""

    EXPIRED = "expired"
    """Nobody decided in time."""


TERMINAL = frozenset({
    AuditStatus.EXECUTED, AuditStatus.FAILED,
    AuditStatus.CANCELLED, AuditStatus.EXPIRED,
})


class AuditWriteError(RuntimeError):
    """The record could not be written. Callers must not proceed."""


def digest(value: str) -> str:
    """SHA-256 of a string, for storing a token without storing the token."""
    return hashlib.sha256((value or "").encode("utf-8", "replace")).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AuditEntry:
    """One audit row, in the form the gateway works with."""

    audit_id: str
    owner_id: str
    conversation_id: str
    effect: str
    tool: str
    arguments_hash: str
    preview_hash: str
    token_hash: str
    idempotency_key: str
    status: AuditStatus = AuditStatus.PENDING
    outcome: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)
    confirmed_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None

    def summary(self) -> Dict[str, Any]:
        """Log form — identifiers and classification, never content."""
        return {
            "audit_id": self.audit_id[:8],
            "tool": self.tool,
            "effect": self.effect,
            "status": self.status.value,
            "idempotency_key": self.idempotency_key[:12],
        }


class AuditLog(Protocol):
    """What the gateway needs from a durable record."""

    async def record_pending(self, entry: AuditEntry) -> str: ...
    async def mark(
        self,
        audit_id: str,
        status: AuditStatus,
        *,
        outcome: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None: ...
    async def was_executed(self, idempotency_key: str) -> bool: ...
    async def get(self, audit_id: str) -> Optional[AuditEntry]: ...
    async def find_by_token_hash(self, token_hash: str) -> Optional[AuditEntry]: ...
    async def history(
        self, owner_id: str, *, limit: int = 50
    ) -> List[AuditEntry]: ...


# ── Postgres ─────────────────────────────────────────────────────────────────

class PostgresAuditLog:
    """The production record. Every failure is raised, never swallowed."""

    def __init__(self, session_maker=None) -> None:
        self._session_maker = session_maker or async_session_maker

    async def record_pending(self, entry: AuditEntry) -> str:
        try:
            async with self._session_maker() as session:
                row = ActionAudit(
                    id=uuid.UUID(entry.audit_id),
                    owner_id=entry.owner_id,
                    conversation_id=entry.conversation_id or None,
                    effect=entry.effect,
                    tool=entry.tool,
                    arguments_hash=entry.arguments_hash,
                    preview_hash=entry.preview_hash,
                    token_hash=entry.token_hash,
                    idempotency_key=entry.idempotency_key,
                    status=AuditStatus.PENDING.value,
                )
                session.add(row)
                await session.commit()
                return entry.audit_id
        except Exception as exc:
            logger.error("Audit write failed for %s: %s", entry.summary(), exc)
            raise AuditWriteError(str(exc)) from exc

    async def mark(
        self,
        audit_id: str,
        status: AuditStatus,
        *,
        outcome: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        now = _utcnow()
        values: Dict[str, Any] = {"status": status.value}
        if outcome is not None:
            values["outcome"] = outcome[:2000]
        if error is not None:
            values["error"] = error[:2000]
        if status is AuditStatus.CONFIRMED:
            values["confirmed_at"] = now
        if status in (AuditStatus.EXECUTED, AuditStatus.FAILED):
            values["executed_at"] = now

        try:
            async with self._session_maker() as session:
                await session.execute(
                    update(ActionAudit)
                    .where(ActionAudit.id == uuid.UUID(audit_id))
                    .values(**values)
                )
                await session.commit()
        except Exception as exc:
            logger.error("Audit update failed for %s → %s: %s", audit_id[:8], status, exc)
            raise AuditWriteError(str(exc)) from exc

    async def was_executed(self, idempotency_key: str) -> bool:
        if not idempotency_key:
            return False
        try:
            async with self._session_maker() as session:
                result = await session.execute(
                    select(ActionAudit.id).where(
                        ActionAudit.idempotency_key == idempotency_key,
                        ActionAudit.status == AuditStatus.EXECUTED.value,
                    ).limit(1)
                )
                return result.scalars().first() is not None
        except Exception as exc:
            # Read failure is treated as a write failure: without knowing
            # whether this already ran, executing would risk a duplicate
            # irreversible action. The caller fails closed.
            logger.error("Audit idempotency check failed for %s: %s",
                         idempotency_key[:12], exc)
            raise AuditWriteError(str(exc)) from exc

    async def get(self, audit_id: str) -> Optional[AuditEntry]:
        async with self._session_maker() as session:
            row = await session.get(ActionAudit, uuid.UUID(audit_id))
            return _from_row(row) if row else None

    async def find_by_token_hash(self, token_hash: str) -> Optional[AuditEntry]:
        async with self._session_maker() as session:
            result = await session.execute(
                select(ActionAudit)
                .where(ActionAudit.token_hash == token_hash)
                .limit(1)
            )
            row = result.scalars().first()
            return _from_row(row) if row else None

    async def history(self, owner_id: str, *, limit: int = 50) -> List[AuditEntry]:
        async with self._session_maker() as session:
            result = await session.execute(
                select(ActionAudit)
                .where(ActionAudit.owner_id == owner_id)
                .order_by(ActionAudit.created_at.desc())
                .limit(limit)
            )
            return [_from_row(row) for row in result.scalars().all()]


def _from_row(row: ActionAudit) -> AuditEntry:
    return AuditEntry(
        audit_id=str(row.id),
        owner_id=row.owner_id,
        conversation_id=row.conversation_id or "",
        effect=row.effect,
        tool=row.tool,
        arguments_hash=row.arguments_hash,
        preview_hash=row.preview_hash,
        token_hash=row.token_hash,
        idempotency_key=row.idempotency_key,
        status=AuditStatus(row.status),
        outcome=row.outcome,
        error=row.error,
        created_at=row.created_at,
        confirmed_at=row.confirmed_at,
        executed_at=row.executed_at,
    )


# ── In-memory (tests only) ───────────────────────────────────────────────────

class InMemoryAuditLog:
    """
    A test double, not a deployment option.

    Kept faithful to the Postgres semantics that matter: the unique-executed
    constraint is enforced here too, so a test that would violate it fails the
    same way production would rather than passing and shipping the bug.

    `fail_next` and `failing` are what make the fail-closed path testable —
    a real database outage is not something a unit test can arrange.
    """

    def __init__(self) -> None:
        self.entries: Dict[str, AuditEntry] = {}
        self.failing = False
        self._fail_next = 0

    def fail_next(self, count: int = 1) -> None:
        self._fail_next = count

    def _maybe_fail(self, operation: str) -> None:
        if self.failing:
            raise AuditWriteError(f"simulated audit outage during {operation}")
        if self._fail_next > 0:
            self._fail_next -= 1
            raise AuditWriteError(f"simulated audit failure during {operation}")

    async def record_pending(self, entry: AuditEntry) -> str:
        self._maybe_fail("record_pending")
        self.entries[entry.audit_id] = entry
        return entry.audit_id

    async def mark(
        self,
        audit_id: str,
        status: AuditStatus,
        *,
        outcome: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self._maybe_fail(f"mark:{status.value}")
        existing = self.entries.get(audit_id)
        if existing is None:
            raise AuditWriteError(f"no audit row {audit_id}")

        if status is AuditStatus.EXECUTED:
            # Mirror the partial unique index rather than trusting callers.
            clash = [
                e for e in self.entries.values()
                if e.idempotency_key == existing.idempotency_key
                and e.status is AuditStatus.EXECUTED
                and e.audit_id != audit_id
            ]
            if clash:
                raise AuditWriteError(
                    "an executed row already exists for this idempotency key"
                )

        now = _utcnow()
        self.entries[audit_id] = replace(
            existing,
            status=status,
            outcome=outcome if outcome is not None else existing.outcome,
            error=error if error is not None else existing.error,
            confirmed_at=now if status is AuditStatus.CONFIRMED else existing.confirmed_at,
            executed_at=(
                now if status in (AuditStatus.EXECUTED, AuditStatus.FAILED)
                else existing.executed_at
            ),
        )

    async def was_executed(self, idempotency_key: str) -> bool:
        self._maybe_fail("was_executed")
        if not idempotency_key:
            return False
        return any(
            e.idempotency_key == idempotency_key and e.status is AuditStatus.EXECUTED
            for e in self.entries.values()
        )

    async def get(self, audit_id: str) -> Optional[AuditEntry]:
        return self.entries.get(audit_id)

    async def find_by_token_hash(self, token_hash: str) -> Optional[AuditEntry]:
        return next(
            (e for e in self.entries.values() if e.token_hash == token_hash), None
        )

    async def history(self, owner_id: str, *, limit: int = 50) -> List[AuditEntry]:
        rows = [e for e in self.entries.values() if e.owner_id == owner_id]
        rows.sort(key=lambda e: e.created_at, reverse=True)
        return rows[:limit]

    # Test conveniences ------------------------------------------------------

    def by_status(self, status: AuditStatus) -> List[AuditEntry]:
        return [e for e in self.entries.values() if e.status is status]

    def only(self) -> AuditEntry:
        assert len(self.entries) == 1, f"expected one audit row, found {len(self.entries)}"
        return next(iter(self.entries.values()))

    def reset(self) -> None:
        self.entries.clear()
        self.failing = False
        self._fail_next = 0


audit_log: AuditLog = PostgresAuditLog()
"""The production record. Tests swap this on the gateway, never here."""


__all__ = [
    "AuditEntry",
    "AuditLog",
    "AuditStatus",
    "AuditWriteError",
    "InMemoryAuditLog",
    "PostgresAuditLog",
    "audit_log",
    "digest",
    "TERMINAL",
]
