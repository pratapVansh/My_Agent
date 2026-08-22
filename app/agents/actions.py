"""
The gate every consequential tool call passes through.

T1 gave each tool a declared effect. Nothing read it. This is the thing that
reads it, and the property it establishes is narrow and absolute:

    a tool whose declared effect requires confirmation is never called
    until a valid, unexpired, single-use, user-bound token is presented
    that matches a hash of the exact action that was previewed.

Everything else here exists to make that sentence true against an adversary,
and the adversary assumed is the language model itself. Not because it is
malicious, but because it is the component most likely to be wrong, most
easily steered by text it read somewhere, and — with `send_email` in its
registry — the component holding the loudest available foot-gun. So the
enforcement is structural. There is no prompt anywhere in this file. A model
that decides to send mail simply does not reach the sender; it reaches this,
and this returns a preview.

Four decisions worth stating outright:

**Interception happens before the call, not after.** A gateway that inspects
results has already let the email leave. The tool callable is never awaited on
the confirmation path — see `intercept`.

**The preview is built without running the tool.** Some tools need a lookup to
describe themselves honestly (`send_email` may be handed only a `draft_id`, and
a preview that cannot show the body is not a preview). Builders may therefore
read — and only read. Whatever they resolve becomes the *stored* arguments, so
the thing hashed, previewed, confirmed and executed are one object rather than
four hopefully-identical ones.

**Confirmation is bound to content, not to intent.** The token alone is not
enough: it carries a hash of the exact tool, arguments and preview text the
user saw. Approving a draft to Alice cannot send a different draft to Bob,
because the second action hashes differently and no token exists for it.

**Completed effects are remembered.** The reflect loop re-runs a specialist
after a failure, so "already sent" has to be a fact the system holds rather
than a hope. Idempotency keys of executed actions are retained and re-offered
instead of re-executed.

The store is per-process and keyed by token, matching `clarification_policy`
and `provenance`. That is a real limitation and it is the safe direction of
one: a restart loses pending approvals, which fails closed — the action simply
has to be requested again.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from app.domain.audit import (
    AuditEntry,
    AuditLog,
    AuditStatus,
    AuditWriteError,
    audit_log,
    digest,
)
from app.domain.pending_actions import (
    PendingActionStore,
    PendingStoreError,
    StoredAction,
    pending_action_store,
)
from app.tools.contract import (
    Effect,
    ErrorKind,
    ToolResult,
    ToolStatus,
    coerce,
    derive_idempotency_key,
    effect_for_spec,
)

logger = logging.getLogger(__name__)


DEFAULT_TTL_SECONDS = 300.0
"""How long an approval stays offerable.

Long enough to read a drafted email and answer; short enough that a token
forgotten in a transcript is useless by the time anyone finds it."""

_MAX_PENDING = 500
"""Ceiling on outstanding actions, so a model looping on `send_email` cannot
grow the store without bound. Oldest first."""

_MAX_EXECUTED = 2000
"""Ceiling on remembered completions."""


# ── Preview ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActionPreview:
    """
    What the user will be shown, and the arguments it describes.

    `arguments` is the resolution: a builder handed `{"draft_id": 7}` returns
    the recipient, subject and body it found, and those become the arguments
    that get hashed and later executed. Without that, the preview describes one
    thing and the execution performs another.
    """

    text: str
    arguments: Optional[Mapping[str, Any]] = None
    valid: bool = True
    reason: str = ""

    @classmethod
    def invalid(cls, reason: str) -> "ActionPreview":
        """
        The action cannot be described, so it must not be offered.

        Used when a tool's own precondition fails — a send with no recipient.
        Previously the tool reported that itself; now it never runs, so the
        check has to live where the preview is built. Returning a rejection
        keeps the model's feedback identical without anything being sent.
        """
        return cls(text="", arguments=None, valid=False, reason=reason)


PreviewBuilder = Callable[[Mapping[str, Any]], Any]
"""`spec["preview"]`: a string, or a callable taking arguments and returning an
`ActionPreview`/str. May be async. Must never cause an effect."""


def _default_preview(tool: str, effect: Effect, arguments: Mapping[str, Any]) -> str:
    """
    Fallback rendering when a tool declares no builder.

    Deliberately plain and complete rather than pretty: an unreviewed action is
    worse than an ugly one, so every argument is shown.
    """
    lines = [f"Action: {tool}", f"Effect: {effect.value}", "Arguments:"]
    for key in sorted(arguments or {}):
        value = str(arguments[key])
        lines.append(f"  {key}: {value[:500]}")
    return "\n".join(lines)


async def build_preview(
    spec: Mapping[str, Any],
    *,
    tool: str,
    effect: Effect,
    arguments: Mapping[str, Any],
) -> ActionPreview:
    """Run a tool's preview builder, tolerating every shape it may return."""
    builder = (spec or {}).get("preview")
    if builder is None:
        return ActionPreview(_default_preview(tool, effect, arguments), dict(arguments))

    if isinstance(builder, str):
        return ActionPreview(builder, dict(arguments))

    try:
        produced = builder(arguments)
        if asyncio.iscoroutine(produced) or isinstance(produced, Awaitable):
            produced = await produced
    except Exception as exc:
        # A builder that raises must not degrade into "no preview, send it
        # anyway". Refusing here costs the user a retry; the alternative costs
        # them an unreviewed action.
        logger.error("Preview builder for '%s' raised: %s", tool, exc, exc_info=True)
        return ActionPreview.invalid(f"could not prepare a preview: {exc}")

    if isinstance(produced, ActionPreview):
        if produced.valid and produced.arguments is None:
            return ActionPreview(produced.text, dict(arguments))
        return produced
    if isinstance(produced, str):
        return ActionPreview(produced, dict(arguments))

    logger.error(
        "Preview builder for '%s' returned %s, expected ActionPreview or str",
        tool, type(produced).__name__,
    )
    return ActionPreview.invalid("preview builder returned an unusable value")


# ── Hashing ──────────────────────────────────────────────────────────────────

def compute_content_hash(
    tool: str, arguments: Mapping[str, Any], preview: str
) -> str:
    """
    A fingerprint of the exact action the user is being shown.

    Covers the preview text as well as the arguments, so approval is bound to
    what was actually displayed. If the rendering changes, the hash changes,
    and an old token stops matching — which is the correct outcome: the user
    approved a specific thing they read.
    """
    canonical = json.dumps(
        {"tool": tool, "arguments": arguments, "preview": preview},
        sort_keys=True, default=str, ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest()


# ── Pending action ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PendingAction:
    """One action prepared, described, and waiting for a decision."""

    token: str
    """The raw confirmation secret. Present only on the turn the action is
    minted — it is never persisted, so an action reloaded from the database
    carries an empty token and is referenced by `handle` instead."""

    tool: str
    arguments: Mapping[str, Any]
    effect: Effect
    preview: str
    idempotency_key: str
    content_hash: str
    owner_id: str
    conversation_id: str
    created_at: datetime
    expires_at: datetime
    audit_id: str = ""
    """The durable row this action is recorded under. Written before the
    action is offered, so an approval always has something to update."""

    handle: str = ""
    """The token's digest, and the durable way to refer to this action.

    A caller holding the raw token (a UI posting it back) proves possession; a
    caller working from a reloaded action — the conversational "yes" — has only
    this. Both are accepted, and both are still subject to owner binding, so
    the handle is a reference rather than a credential: knowing it does not let
    anyone but the owner act on it."""

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        return (now or _utcnow()) >= self.expires_at

    def summary(self) -> Dict[str, Any]:
        """Log form — no arguments, no preview, no recipient."""
        return {
            "tool": self.tool,
            "effect": self.effect.value,
            "token": self.token[:8],
            "content_hash": self.content_hash[:12],
            "expires_at": self.expires_at.isoformat(),
        }

    def to_result(self) -> ToolResult:
        """The `ToolResult` the reasoning loop sees in place of an execution."""
        return ToolResult(
            status=ToolStatus.PENDING_CONFIRMATION,
            effect=self.effect,
            data={
                "confirmation_token": self.token,
                "content_hash": self.content_hash,
                "expires_at": self.expires_at.isoformat(),
                "tool": self.tool,
            },
            preview=self.preview,
            idempotency_key=self.idempotency_key,
            tool=self.tool,
        )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Confirmation outcomes ────────────────────────────────────────────────────

class Denial(str, Enum):
    """Why a confirmation was refused. Each is a distinct security event."""

    UNKNOWN_TOKEN = "unknown_token"
    EXPIRED = "expired"
    ALREADY_USED = "already_used"
    CONTENT_MISMATCH = "content_mismatch"
    WRONG_USER = "wrong_user"
    ALREADY_EXECUTED = "already_executed"


@dataclass(frozen=True)
class ConfirmationOutcome:
    """The verdict on a presented token."""

    approved: bool
    denial: Optional[Denial] = None
    action: Optional[PendingAction] = None
    message: str = ""

    def summary(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "denial": self.denial.value if self.denial else None,
        }


def _to_pending(stored: "StoredAction") -> PendingAction:
    """Rebuild the in-memory view of a stored action. No token, by design."""
    return PendingAction(
        token="",
        handle=stored.token_hash,
        tool=stored.tool,
        arguments=dict(stored.arguments),
        effect=Effect(stored.effect),
        preview=stored.preview,
        idempotency_key=stored.idempotency_key,
        content_hash=stored.content_hash,
        owner_id=stored.owner_id,
        conversation_id=stored.conversation_id,
        created_at=stored.created_at,
        expires_at=stored.expires_at,
        audit_id=stored.audit_id,
    )


# ── Gateway ──────────────────────────────────────────────────────────────────

class ActionGateway:
    """
    Holds pending actions and decides whether one may run.

    A single process-wide instance; the module exports `action_gateway`.
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        audit: Optional["AuditLog"] = None,
        pending: Optional["PendingActionStore"] = None,
    ) -> None:
        self._ttl = float(ttl_seconds)
        # Both stores are durable. Nothing about an outstanding action lives in
        # this object any more: a restart between "shall I send this?" and
        # "yes" used to lose the action, and two workers used to lose it
        # roughly half the time.
        self._audit: "AuditLog" = audit if audit is not None else audit_log
        self._pending: "PendingActionStore" = (
            pending if pending is not None else pending_action_store
        )

    @property
    def audit(self) -> "AuditLog":
        return self._audit

    @property
    def pending_store(self) -> "PendingActionStore":
        return self._pending

    def use_audit_log(self, audit: "AuditLog") -> None:
        """Swap the record. Tests only — production uses Postgres."""
        self._audit = audit

    def use_pending_store(self, pending: "PendingActionStore") -> None:
        """Swap the waiting room. Tests only — production uses Postgres."""
        self._pending = pending

    @staticmethod
    def _handle_for(token: Optional[str], handle: Optional[str]) -> str:
        """
        The lookup key for an action.

        A caller may present the raw token or the handle. The token is hashed
        to the same value, so both reach one row and neither is stored.
        """
        if handle:
            return handle
        return digest(token or "") if token else ""

    # ── Policy ───────────────────────────────────────────────────────────────

    @staticmethod
    def requires_confirmation(effect: Effect) -> bool:
        """
        Whether this effect may not run unattended.

        Delegated to the contract so the classification has exactly one
        definition. An effect that cannot be read at all resolves to
        `DEFAULT_UNDECLARED_EFFECT`, which is confirmable — an undeclared tool
        is gated, never waved through.
        """
        return effect.requires_confirmation

    async def was_executed(self, idempotency_key: str) -> bool:
        """
        Whether this exact action already completed — durably.

        Raises `AuditWriteError` when the record cannot be read, and callers
        treat that as a refusal: without knowing whether something already ran,
        running it is how an irreversible action happens twice.
        """
        return await self._audit.was_executed(idempotency_key)

    # ── Interception ─────────────────────────────────────────────────────────

    async def intercept(
        self,
        *,
        tool: str,
        spec: Mapping[str, Any],
        arguments: Mapping[str, Any],
        owner_id: str,
        conversation_id: str = "",
        effect: Optional[Effect] = None,
    ) -> ToolResult:
        """
        Stand in for a confirmable tool call. **Never awaits the tool.**

        Returns a `PENDING_CONFIRMATION` result carrying the preview and token,
        an `ERROR` when the action cannot be described, or an `OK` noting that
        this exact action already completed.
        """
        declared = effect if effect is not None else effect_for_spec(spec, tool)
        preview = await build_preview(
            spec, tool=tool, effect=declared, arguments=arguments
        )

        if not preview.valid:
            return ToolResult.failed(
                preview.reason or "this action could not be prepared for review",
                kind=ErrorKind.INVALID_RESULT,
                effect=declared,
                tool=tool,
            )

        resolved: Dict[str, Any] = dict(preview.arguments or arguments)
        idempotency_key = derive_idempotency_key(tool, resolved)

        # The reflect loop's guard, now durable. A specialist re-run after a
        # failure asks for the same send again; this is where that becomes
        # "already done" instead of a second email — and because the answer
        # comes from the database, it stays true across a restart and across
        # replicas.
        try:
            already_done = await self.was_executed(idempotency_key)
        except AuditWriteError as exc:
            return self._audit_unavailable(tool, declared, "idempotency check", exc)

        if already_done:
            logger.info(
                "Refusing to repeat an already-completed action: tool=%s key=%s",
                tool, idempotency_key[:12],
            )
            return ToolResult(
                status=ToolStatus.OK,
                effect=declared,
                data={
                    "already_executed": True,
                    "message": (
                        "This exact action was already completed earlier. "
                        "It was NOT performed again."
                    ),
                },
                preview=preview.text,
                idempotency_key=idempotency_key,
                tool=tool,
            )

        target = (spec or {}).get("callable")
        if not callable(target):
            return ToolResult.failed(
                f"tool '{tool}' has no callable to execute",
                kind=ErrorKind.INVALID_RESULT,
                effect=declared,
                tool=tool,
            )

        content_hash = compute_content_hash(tool, resolved, preview.text)
        now = _utcnow()
        token = secrets.token_urlsafe(32)
        audit_id = str(uuid.uuid4())

        # The record is written before the action is offered — before a token
        # exists anywhere a user could act on. An action that is proposed and
        # never approved is therefore as visible as one that was sent, and an
        # approval always has a row to update.
        entry = AuditEntry(
            audit_id=audit_id,
            owner_id=owner_id or "",
            conversation_id=conversation_id or "",
            effect=declared.value,
            tool=tool,
            arguments_hash=derive_idempotency_key(tool, resolved),
            preview_hash=digest(preview.text),
            token_hash=digest(token),
            idempotency_key=idempotency_key,
        )
        try:
            await self._audit.record_pending(entry)
        except AuditWriteError as exc:
            # Fail closed. No row means no token, which means nothing to
            # approve — the action simply cannot be reached.
            return self._audit_unavailable(tool, declared, "recording the action", exc)

        expires_at = now + timedelta(seconds=self._ttl)
        token_hash = digest(token)

        # Persisted, not held. `target` is deliberately discarded here: the
        # callable is rebuilt from the tool's *name* at confirmation time, so
        # nothing executable ever reaches storage.
        try:
            await self._pending.put(StoredAction(
                id=str(uuid.uuid4()),
                token_hash=token_hash,
                audit_id=audit_id,
                owner_id=owner_id or "",
                conversation_id=conversation_id or "",
                tool=tool,
                effect=declared.value,
                arguments=resolved,
                preview=preview.text,
                content_hash=content_hash,
                idempotency_key=idempotency_key,
                created_at=now,
                expires_at=expires_at,
            ))
        except PendingStoreError as exc:
            # The audit row exists but the action cannot be waited on. Close it
            # out rather than leaving a row stuck at `pending` forever.
            await self._mark_quietly(audit_id, AuditStatus.FAILED)
            return self._audit_unavailable(tool, declared, "storing the action", exc)

        action = PendingAction(
            token=token,
            handle=token_hash,
            tool=tool,
            arguments=resolved,
            effect=declared,
            preview=preview.text,
            idempotency_key=idempotency_key,
            content_hash=content_hash,
            owner_id=owner_id or "",
            conversation_id=conversation_id or "",
            created_at=now,
            expires_at=expires_at,
            audit_id=audit_id,
        )

        logger.info("Action held for confirmation: %s", action.summary())
        return action.to_result()

    def _audit_unavailable(
        self, tool: str, effect: Effect, during: str, exc: Exception
    ) -> ToolResult:
        """
        The refusal used whenever the record cannot be written or read.

        An unauditable consequential action is treated exactly like a forbidden
        one. That is the whole point of writing the row first: if the evidence
        cannot be created, the effect does not happen.
        """
        logger.error(
            "Refusing %s (%s): audit unavailable while %s — %s",
            tool, effect.value, during, exc,
        )
        return ToolResult.failed(
            "I couldn't record this action for audit, so I haven't done it. "
            "Please try again shortly.",
            kind=ErrorKind.INVALID_RESULT,
            effect=effect,
            retryable=True,
            tool=tool,
        )

    async def _mark_quietly(self, audit_id: str, status: AuditStatus) -> None:
        """Best-effort status update for bookkeeping that must not raise.

        Used only for expiry and cancellation — outcomes where nothing
        happened, so a lost update costs a row's accuracy rather than a
        duplicate side effect.
        """
        if not audit_id:
            return
        try:
            await self._audit.mark(audit_id, status)
        except Exception as exc:
            logger.warning("Could not mark audit %s as %s: %s",
                           audit_id[:8], status.value, exc)

    # ── Confirmation ─────────────────────────────────────────────────────────

    async def confirm(
        self,
        token: Optional[str] = None,
        *,
        owner_id: str,
        handle: Optional[str] = None,
        content_hash: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ConfirmationOutcome:
        """
        Validate an approval, claim the action, and record it.

        Accepts either the raw token (a UI posting one back) or the handle its
        digest produces (the conversational "yes", working from an action
        reloaded out of the database). Both reach the same row and both remain
        subject to owner binding.

        The audit write comes after the claim and before any execution. If it
        fails the approval is refused — and the action is already claimed, so
        it is dead rather than half-approved. Losing a legitimate approval to a
        database blip is the acceptable direction of that trade.
        """
        handle = self._handle_for(token, handle)
        moment = now or _utcnow()

        # ── Read, then validate, then claim ──────────────────────────────────
        # The order matters. Claiming first would let anyone holding a handle
        # destroy a legitimate pending action just by presenting the wrong
        # owner — a denial of service disguised as a failed confirmation. So
        # every check that can refuse runs against a *read* of the row, and the
        # row is only removed once the caller has proved they may have it.
        try:
            stored = await self._pending.peek(handle)
        except PendingStoreError as exc:
            logger.error("Confirmation refused: pending store unreadable — %s", exc)
            return ConfirmationOutcome(
                False, Denial.UNKNOWN_TOKEN,
                message=(
                    "I couldn't look up that confirmation, so I haven't done "
                    "it. Please try again shortly."
                ),
            )

        if stored is None:
            logger.warning("Confirmation refused: unknown handle %s", handle[:8])
            return ConfirmationOutcome(
                False, Denial.UNKNOWN_TOKEN,
                message="That confirmation is not recognised. Ask for the action again.",
            )

        action = _to_pending(stored)

        # ── Integrity ────────────────────────────────────────────────────────
        # Recomputed from the row's own contents, because the row is now on
        # disk and the arguments are no longer beyond reach. Comparing only the
        # *presented* hash against the *stored* hash would miss an attacker who
        # edited the arguments and left the hash alone — the two would still
        # agree with each other while describing different actions.
        #
        # This and the presented-hash check cover different attackers: this one
        # catches a row that was altered, that one catches an approval being
        # transplanted onto a different action.
        recomputed = compute_content_hash(
            stored.tool, dict(stored.arguments), stored.preview
        )
        if not hmac.compare_digest(recomputed, stored.content_hash):
            logger.error(
                "Confirmation refused: stored action %s fails its own integrity "
                "check — arguments, preview or tool were altered after storage",
                handle[:8],
            )
            await self._pending.drop(handle)
            await self._mark_quietly_with(
                stored.audit_id, AuditStatus.FAILED,
                error="stored action failed its integrity check",
            )
            return ConfirmationOutcome(
                False, Denial.CONTENT_MISMATCH,
                message="That action could not be verified. Nothing was done.",
            )

        if stored.is_expired(moment):
            await self._pending.drop(handle)
            await self._mark_quietly(stored.audit_id, AuditStatus.EXPIRED)
            logger.warning("Confirmation refused: expired %s", handle[:8])
            return ConfirmationOutcome(
                False, Denial.EXPIRED,
                message="That confirmation expired. Ask for the action again.",
            )

        # Compared with compare_digest so a wrong owner cannot be probed by
        # timing. The empty-string case is explicit: an unowned pending action
        # is never confirmable.
        if not stored.owner_id or not hmac.compare_digest(
            stored.owner_id, owner_id or ""
        ):
            logger.error(
                "Confirmation refused: owner mismatch on %s (tool=%s)",
                handle[:8], stored.tool,
            )
            return ConfirmationOutcome(
                False, Denial.WRONG_USER,
                message="That confirmation does not belong to you.",
            )

        if content_hash is not None and not hmac.compare_digest(
            stored.content_hash, content_hash
        ):
            logger.error(
                "Confirmation refused: content mismatch on %s (tool=%s)",
                handle[:8], stored.tool,
            )
            return ConfirmationOutcome(
                False, Denial.CONTENT_MISMATCH,
                message="The action changed since you reviewed it. Nothing was done.",
            )

        # Durable replay check. Asked of the database rather than of process
        # memory, so a second confirmation after a restart — or on another
        # replica — still sees that this action completed.
        try:
            if await self.was_executed(stored.idempotency_key):
                await self._pending.drop(handle)
                return ConfirmationOutcome(
                    False, Denial.ALREADY_EXECUTED, action=action,
                    message="That action was already completed.",
                )
        except AuditWriteError as exc:
            logger.error(
                "Refusing confirmation of %s: audit unreadable — %s",
                stored.summary(), exc,
            )
            return ConfirmationOutcome(
                False, Denial.UNKNOWN_TOKEN,
                message=(
                    "I couldn't verify whether this had already run, so I "
                    "haven't done it. Please try again shortly."
                ),
            )

        # ── Claim ────────────────────────────────────────────────────────────
        # Atomic: the delete either returns this row to us or returns nothing.
        # Exactly one caller can win it, across threads, processes and
        # machines — which is what makes single-use a property of the database
        # rather than of ordering luck between two workers.
        try:
            claimed = await self._pending.claim(handle)
        except PendingStoreError as exc:
            logger.error("Confirmation refused: could not claim — %s", exc)
            return ConfirmationOutcome(
                False, Denial.UNKNOWN_TOKEN,
                message=(
                    "I couldn't secure that confirmation, so I haven't done "
                    "it. Please try again shortly."
                ),
            )

        if claimed is None:
            # Another caller took it between the read and the claim.
            logger.warning("Confirmation lost the race for %s", handle[:8])
            return ConfirmationOutcome(
                False, Denial.ALREADY_USED,
                message="That confirmation was already used.",
            )

        action = _to_pending(claimed)

        try:
            await self._audit.mark(action.audit_id, AuditStatus.CONFIRMED)
        except Exception as exc:
            logger.error(
                "Refusing confirmed action %s: audit could not record the "
                "approval — %s", action.summary(), exc,
            )
            return ConfirmationOutcome(
                False, Denial.UNKNOWN_TOKEN,
                message=(
                    "I couldn't record your approval for audit, so I haven't "
                    "done it. Please ask again."
                ),
            )

        logger.info("Action confirmed: %s", action.summary())
        return ConfirmationOutcome(True, None, action=action)

    async def confirm_and_execute(
        self,
        token: Optional[str] = None,
        *,
        owner_id: str,
        handle: Optional[str] = None,
        content_hash: Optional[str] = None,
        timeout: float = 30.0,
    ) -> ToolResult:
        """
        Validate, then run the held action exactly once.

        The only path in the system that invokes a confirmable tool.
        """
        outcome = await self.confirm(
            token, owner_id=owner_id, handle=handle, content_hash=content_hash
        )
        if not outcome.approved or outcome.action is None:
            return ToolResult.failed(
                outcome.message or "confirmation refused",
                kind=ErrorKind.INVALID_RESULT,
                effect=(outcome.action.effect if outcome.action else Effect.EXTERNAL_WRITE),
                tool=(outcome.action.tool if outcome.action else ""),
            )

        action = outcome.action

        # ── Reconstruct, never deserialise ───────────────────────────────────
        # The callable comes from the application's own registry, keyed by the
        # stored *name*. A tampered row can therefore only claim to be one of
        # the tools this build already ships; it cannot introduce behaviour.
        # The effect is taken from the rebuilt spec for the same reason — a row
        # that downgraded itself to READ must not thereby escape the gate.
        #
        # Imported here rather than at module scope: `confirmable_tools` needs
        # `ActionPreview` from this module, and a top-level import would be a
        # cycle.
        from app.agents.confirmable_tools import resolve_confirmable_tool

        spec = resolve_confirmable_tool(action.tool, action.owner_id)
        if spec is None:
            logger.error(
                "Refusing to execute %s: '%s' is not a known confirmable tool",
                action.summary(), action.tool,
            )
            await self._mark_quietly_with(
                action.audit_id, AuditStatus.FAILED,
                error=f"tool '{action.tool}' could not be resolved",
            )
            return ToolResult.failed(
                "That action refers to a capability I no longer have, so I "
                "haven't done it.",
                kind=ErrorKind.UNKNOWN_TOOL,
                effect=action.effect,
                tool=action.tool,
            )

        rebuilt_effect = effect_for_spec(spec, action.tool)
        if rebuilt_effect is not action.effect:
            # The stored effect and the registry disagree. Either the row was
            # altered or the tool's classification changed under it; neither is
            # a state in which an irreversible action should proceed.
            logger.error(
                "Refusing to execute %s: stored effect %s but registry says %s",
                action.summary(), action.effect.value, rebuilt_effect.value,
            )
            await self._mark_quietly_with(
                action.audit_id, AuditStatus.FAILED,
                error="stored effect does not match the tool registry",
            )
            return ToolResult.failed(
                "That action's recorded permissions no longer match this "
                "capability, so I haven't done it.",
                kind=ErrorKind.INVALID_RESULT,
                effect=rebuilt_effect,
                tool=action.tool,
            )

        target = spec.get("callable")
        if not callable(target):
            await self._mark_quietly_with(
                action.audit_id, AuditStatus.FAILED,
                error="resolved tool has no callable",
            )
            return ToolResult.failed(
                "the held action is no longer executable",
                kind=ErrorKind.INVALID_RESULT,
                effect=action.effect,
                tool=action.tool,
            )

        # The idempotency slot is claimed *before* the call, by writing the
        # executed row. Two reasons, and the second is the one that matters:
        #
        #   a tool that succeeds and then times out on the response would
        #   otherwise look un-run and be repeatable;
        #
        #   and a process that dies with SMTP open leaves the claim standing,
        #   so nothing retries. For an irreversible effect, "possibly sent,
        #   refuses to repeat" is the safe direction of a wrong guess.
        #
        # The write is also the last gate: if the record cannot be made, the
        # effect does not happen.
        try:
            await self._audit.mark(
                action.audit_id, AuditStatus.EXECUTED,
                outcome="execution started",
            )
        except Exception as exc:
            logger.error(
                "Refusing to execute %s: audit could not claim the action — %s",
                action.summary(), exc,
            )
            return self._audit_unavailable(
                action.tool, action.effect, "claiming the action", exc
            )

        try:
            raw = await asyncio.wait_for(target(action.arguments), timeout=timeout)
        except asyncio.TimeoutError:
            # Deliberately left claimed. A timeout says nothing about whether
            # the message went out, and the claim is what stops a retry from
            # finding out the expensive way.
            logger.error("Confirmed action timed out: %s", action.summary())
            await self._mark_quietly_with(
                action.audit_id, AuditStatus.EXECUTED,
                outcome=f"timed out after {timeout:.0f}s; outcome unknown",
            )
            return ToolResult.failed(
                f"{action.tool} timed out after {timeout:.0f}s; it may or may not "
                "have completed",
                kind=ErrorKind.TIMEOUT,
                effect=action.effect,
                retryable=False,
                tool=action.tool,
            )
        except Exception as exc:
            # An explicit failure is the one case where the effect demonstrably
            # did not happen, so the claim is released and the user can try
            # again. A crash would not have reached here and stays claimed.
            logger.error("Confirmed action failed: %s — %s", action.summary(), exc)
            await self._mark_quietly_with(
                action.audit_id, AuditStatus.FAILED, error=str(exc)[:2000],
            )
            return ToolResult.failed(
                str(exc),
                kind=ErrorKind.EXCEPTION,
                effect=action.effect,
                tool=action.tool,
            )

        result = coerce(
            raw,
            tool=action.tool,
            declared_effect=action.effect,
            tool_input=action.arguments,
        )

        if result.is_error:
            await self._mark_quietly_with(
                action.audit_id, AuditStatus.FAILED,
                error=(result.error.message if result.error else "tool reported failure"),
            )
        else:
            await self._mark_quietly_with(
                action.audit_id, AuditStatus.EXECUTED, outcome=result.status.value,
            )

        logger.info("Confirmed action executed: %s", result.summary())
        return result

    async def _mark_quietly_with(
        self,
        audit_id: str,
        status: AuditStatus,
        *,
        outcome: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record a terminal outcome without letting the write mask it.

        The gate has already been passed — the effect has happened. Raising
        here would replace a truthful "sent" with an error the user cannot act
        on, so the failure is logged loudly instead.
        """
        if not audit_id:
            return
        try:
            await self._audit.mark(audit_id, status, outcome=outcome, error=error)
        except Exception as exc:
            logger.error(
                "AUDIT INCONSISTENCY: action %s reached %s but the record could "
                "not be updated — %s", audit_id[:8], status.value, exc,
            )

    async def cancel(
        self, token: Optional[str] = None, *, owner_id: str,
        handle: Optional[str] = None,
    ) -> bool:
        """
        Drop a pending action, recording the refusal. Only its owner may.

        Ownership is checked against a read before the row is removed, for the
        same reason confirmation is: otherwise anyone holding a handle could
        delete someone else's outstanding action.
        """
        key = self._handle_for(token, handle)
        try:
            stored = await self._pending.peek(key)
        except PendingStoreError as exc:
            logger.error("Cancellation failed: pending store unreadable — %s", exc)
            return False

        if stored is None:
            return False
        if not stored.owner_id or not hmac.compare_digest(
            stored.owner_id, owner_id or ""
        ):
            return False

        try:
            claimed = await self._pending.claim(key)
        except PendingStoreError as exc:
            logger.error("Cancellation failed: could not claim — %s", exc)
            return False
        if claimed is None:
            return False

        await self._mark_quietly(claimed.audit_id, AuditStatus.CANCELLED)
        return True

    async def get(
        self, token: Optional[str] = None, *, handle: Optional[str] = None
    ) -> Optional[PendingAction]:
        """The held action, for rendering. Read-only and does not consume."""
        try:
            stored = await self._pending.peek(self._handle_for(token, handle))
        except PendingStoreError:
            return None
        return _to_pending(stored) if stored else None

    async def pending_for(
        self, conversation_id: str, owner_id: str
    ) -> list[PendingAction]:
        """
        Live actions awaiting this user's decision in this conversation.

        Read from the store, so a worker that never saw the action proposed can
        still offer it — which is the entire reason confirmation now survives a
        restart or a second replica.

        An unreadable store yields an empty list: with no way to know what is
        outstanding, offering nothing is the only safe answer.
        """
        try:
            rows = await self._pending.list_for(conversation_id or "", owner_id or "")
        except PendingStoreError as exc:
            logger.error("Could not list pending actions: %s", exc)
            return []
        return [_to_pending(row) for row in rows]

    # ── Bookkeeping ──────────────────────────────────────────────────────────

    async def sweep_expired(self, limit: int = 100) -> int:
        """
        Remove and record every action that timed out without a decision.

        A row nobody decided on is a fact worth keeping, and one left at
        `pending` forever would be a lie about what happened. Expiry is
        otherwise only noticed when a confirmation happens to touch the row, so
        a quiet system needs this called periodically.
        """
        try:
            expired = await self._pending.take_expired(limit)
        except PendingStoreError as exc:
            logger.error("Expiry sweep failed: %s", exc)
            return 0
        for stored in expired:
            await self._mark_quietly(stored.audit_id, AuditStatus.EXPIRED)
        return len(expired)

    def reset(self) -> None:
        """
        Clear the waiting room — for tests.

        Only works against an in-memory store; the production store has no such
        method and nothing should be casually erasing outstanding actions.
        """
        resetter = getattr(self._pending, "reset", None)
        if callable(resetter):
            resetter()


action_gateway = ActionGateway()


__all__ = [
    "ActionGateway",
    "ActionPreview",
    "ConfirmationOutcome",
    "Denial",
    "PendingAction",
    "action_gateway",
    "build_preview",
    "compute_content_hash",
    "DEFAULT_TTL_SECONDS",
]
