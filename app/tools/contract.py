"""
The typed result every tool returns — and the reason the type has to exist.

Tools in this system return free-form dictionaries. The conventions are real
(`success`, `found`, `count`, `message`) but they are conventions, so the only
way to learn what a call produced is to *guess from its shape*. `base_agent`
already does exactly that, and it works because the guesses were written
alongside the tools they guess about.

That approach has a ceiling, and the ceiling is the action layer. Two facts
about a tool call cannot be recovered from its return value at all:

  * **What it did.** `{"success": true, "id": 42}` is the same shape whether the
    tool bookmarked a job or delivered an email to a stranger. Nothing
    distinguishes a read from an irreversible external write, so nothing can
    gate one and not the other.
  * **Whether it is safe to repeat.** The reflect loop re-runs a specialist
    after a failure. If the specialist already sent mail, that is a second
    email. Today the only thing preventing it is an in-process dictionary
    inside the SMTP wrapper.

So the effect class is declared on the tool, in the registry, next to the scope
it already carries — never inferred from the result and never from the user's
request text, because the request text is precisely what a prompt injection
controls.

**The fail-safe rule.** For anything above `LOCAL_WRITE`, a result that cannot
be positively recognised is an `ERROR`, not an `OK`. An unrecognised shape from
a read is harmless — the worst case is data we fail to classify. An
unrecognised shape from something that may have sent, paid, or deleted is the
one case where guessing optimistically is unrecoverable. Reads keep the lenient
legacy heuristics; writes do not.

This module deliberately contains no gateway, no confirmation flow, and no
persistence. It is the vocabulary those will be written in.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


# ── Status ───────────────────────────────────────────────────────────────────

class ToolStatus(str, Enum):
    """What the call produced."""

    OK = "ok"
    """The tool ran and returned something."""

    NO_DATA = "no_data"
    """The tool ran correctly and there was genuinely nothing to return.

    A statable fact, and emphatically not an error — the distinction
    `app.memory.answerability` exists to protect, carried at the tool boundary
    so it survives the trip up to the response."""

    ERROR = "error"
    """The call failed. Absence of data is NOT established."""

    PENDING_CONFIRMATION = "pending_confirmation"
    """The call was intercepted before running and awaits the user's approval.

    Distinct from all three above because it is none of them: nothing failed,
    nothing was found, and — critically — nothing happened. A model shown an
    `ok` here would report an email as sent that was never sent, which is the
    single worst thing this system can say."""


# ── Effect ───────────────────────────────────────────────────────────────────

class Effect(str, Enum):
    """
    What a tool does to the world, declared by whoever registers it.

    The ordering is what makes this useful: `severity` is total, so "at least as
    dangerous as" is a comparison rather than a lookup table, and a tool can
    never quietly claim to be safer than it was registered as.
    """

    READ = "READ"
    """No state changes anywhere. Safe to call, safe to repeat."""

    LOCAL_WRITE = "LOCAL_WRITE"
    """Changes the user's own data, reversible by the user. Drafting an email,
    bookmarking a job, saving a preference."""

    EXTERNAL_WRITE = "EXTERNAL_WRITE"
    """Visible outside this system and hard to retract. Sending mail, creating
    a calendar event, submitting an application."""

    DESTRUCTIVE = "DESTRUCTIVE"
    """Removes data or is otherwise irreversible. Deleting memory, erasing a
    profile, cancelling a booking."""

    @property
    def severity(self) -> int:
        return _SEVERITY[self]

    @property
    def requires_confirmation(self) -> bool:
        """
        Whether executing this will need explicit user approval.

        Nothing enforces it yet — the gateway that will is a later change. It
        is defined here so that the classification and the rule that reads it
        cannot drift apart in the meantime.
        """
        return self.severity >= Effect.EXTERNAL_WRITE.severity

    @property
    def is_repeatable(self) -> bool:
        """Whether calling twice is harmless. Drives idempotency-key defaults."""
        return self is Effect.READ


_SEVERITY: Dict[Effect, int] = {
    Effect.READ: 0,
    Effect.LOCAL_WRITE: 1,
    Effect.EXTERNAL_WRITE: 2,
    Effect.DESTRUCTIVE: 3,
}


DEFAULT_UNDECLARED_EFFECT = Effect.EXTERNAL_WRITE
"""
What an undeclared tool is assumed to do.

Conservative on purpose. Every tool in this repository declares its effect
explicitly, so nothing reaches this today; it exists so that a tool added later
without a declaration is treated as consequential rather than waved through.
Assuming READ would make the omission invisible exactly where it is most
expensive. A later change should make registration fail outright instead — at
that point this constant becomes unreachable rather than wrong.
"""


# ── Errors ───────────────────────────────────────────────────────────────────

class ErrorKind(str, Enum):
    """Why a call failed. Distinct kinds because they warrant distinct replies."""

    EXCEPTION = "exception"
    """The tool raised."""

    TIMEOUT = "timeout"
    """The tool did not return in time. It may still be running."""

    TOOL_REPORTED = "tool_reported"
    """The tool returned normally and reported its own failure."""

    INVALID_RESULT = "invalid_result"
    """The result could not be interpreted. For a consequential effect this is
    treated as a failure rather than a success — see the module docstring."""

    UNKNOWN_TOOL = "unknown_tool"
    """The model named a tool that does not exist or is out of scope."""


@dataclass(frozen=True)
class ToolError:
    """Why a call failed, in a form the response layer can act on."""

    kind: ErrorKind
    message: str
    retryable: bool = False
    """Whether retrying the identical call could plausibly succeed. A timeout
    might; a malformed result will not."""

    def summary(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "retryable": self.retryable}


# ── Result ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolResult:
    """
    One tool call's outcome.

    Frozen because this crosses into observation rendering, answerability
    assessment and (later) the action gateway, and none of them should be able
    to alter what a tool reported.
    """

    status: ToolStatus
    effect: Effect

    data: Mapping[str, Any] = field(default_factory=dict)
    """The payload. Always a mapping so consumers need no isinstance checks;
    non-mapping returns are wrapped under a `value` key by `coerce`."""

    preview: Optional[str] = None
    """Human-readable description of what this call did or would do.

    For confirmable effects this is what the user will be shown before
    approving — so it must describe the *actual* action (real recipient, real
    subject), never a paraphrase."""

    idempotency_key: Optional[str] = None
    """Stable identifier for this call's effect, so a retry can be recognised
    as a repeat rather than performed twice."""

    error: Optional[ToolError] = None
    tool: str = ""

    raw: Any = None
    """The original return value, retained so observation rendering is
    byte-identical to the pre-contract behaviour for legacy dict results."""

    adapted: bool = False
    """True when this was reconstructed from a legacy return value rather than
    constructed by the tool. Useful for tracking migration progress."""

    # ── Predicates ───────────────────────────────────────────────────────────

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.OK

    @property
    def is_error(self) -> bool:
        return self.status is ToolStatus.ERROR

    @property
    def is_empty(self) -> bool:
        return self.status is ToolStatus.NO_DATA

    @property
    def is_pending(self) -> bool:
        """Intercepted and awaiting confirmation. Nothing has run."""
        return self.status is ToolStatus.PENDING_CONFIRMATION

    @property
    def yielded_evidence(self) -> bool:
        """Whether this call produced something an answer can be built on."""
        return self.status is ToolStatus.OK

    @property
    def requires_confirmation(self) -> bool:
        return self.effect.requires_confirmation

    @property
    def preview_missing(self) -> bool:
        """
        A confirmable success with nothing to show the user.

        Not an error here — no gateway is reading previews yet. It is surfaced
        so the gap is visible now rather than discovered when one is.
        """
        return self.ok and self.requires_confirmation and not (self.preview or "").strip()

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def success(
        cls,
        data: Optional[Mapping[str, Any]] = None,
        *,
        effect: Effect = Effect.READ,
        preview: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        tool: str = "",
        raw: Any = None,
    ) -> "ToolResult":
        return cls(
            status=ToolStatus.OK,
            effect=effect,
            data=dict(data or {}),
            preview=preview,
            idempotency_key=idempotency_key,
            tool=tool,
            raw=raw,
        )

    @classmethod
    def no_data(
        cls,
        message: str = "",
        *,
        effect: Effect = Effect.READ,
        tool: str = "",
        raw: Any = None,
    ) -> "ToolResult":
        return cls(
            status=ToolStatus.NO_DATA,
            effect=effect,
            data={"message": message} if message else {},
            tool=tool,
            raw=raw,
        )

    @classmethod
    def failed(
        cls,
        message: str,
        *,
        kind: ErrorKind = ErrorKind.EXCEPTION,
        effect: Effect = Effect.READ,
        retryable: bool = False,
        tool: str = "",
        raw: Any = None,
    ) -> "ToolResult":
        return cls(
            status=ToolStatus.ERROR,
            effect=effect,
            error=ToolError(kind=kind, message=message, retryable=retryable),
            tool=tool,
            raw=raw,
        )

    def with_idempotency_key(self, key: str) -> "ToolResult":
        return replace(self, idempotency_key=key)

    # ── Rendering ────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Structured log form — classification only, never payload content."""
        out: Dict[str, Any] = {
            "tool": self.tool,
            "status": self.status.value,
            "effect": self.effect.value,
            "adapted": self.adapted,
        }
        if self.error is not None:
            out["error"] = self.error.summary()
        if self.idempotency_key:
            out["idempotency_key"] = self.idempotency_key[:12]
        return out

    def observation(self, max_chars: int = 1200) -> str:
        """
        The line handed back to the model after this call.

        Native results get an explicit status marker, because a model shown an
        empty payload with no explanation is in exactly the state where it
        invents one. Adapted (legacy) results are rendered by the caller from
        `raw`, so their observation text is unchanged by this module.
        """
        if self.status is ToolStatus.ERROR:
            detail = self.error.message if self.error else "unknown error"
            return (
                f"[{self.status.value}] {self.tool or 'tool'} failed: {detail[:400]} "
                "(a failed lookup does NOT mean the information is missing)"
            )
        if self.status is ToolStatus.NO_DATA:
            message = str(self.data.get("message") or "").strip()
            return (
                f"[{self.status.value}] {self.tool or 'tool'} completed and found nothing"
                + (f": {message[:300]}" if message else "")
            )
        if self.status is ToolStatus.PENDING_CONFIRMATION:
            # Worded to leave the model no room to describe this as done. It is
            # also told not to retry, because retrying only mints another
            # pending action and the loop would burn its iterations on it.
            return (
                f"[{self.status.value}] {self.tool or 'tool'} was NOT executed. "
                "It needs the user's explicit approval first. Show the user this "
                "exact preview and ask them to confirm. Do NOT call this tool "
                "again and do NOT say the action was completed.\n"
                f"--- PREVIEW ---\n{(self.preview or '').strip()[:900]}"
            )
        body = _safe_json(dict(self.data))
        if len(body) > max_chars:
            body = body[: max_chars - 3] + "..."
        return f"[{self.status.value}] {body}"


# ── Idempotency ──────────────────────────────────────────────────────────────

def derive_idempotency_key(tool: str, payload: Any) -> str:
    """
    A stable key for "this tool, called with these arguments".

    Deterministic so the same call produces the same key across processes and
    restarts — the in-process dedupe it is meant to replace survives neither.
    Sorted keys so argument ordering cannot produce two keys for one action.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    digest = hashlib.sha256(f"{tool}\x00{canonical}".encode("utf-8", "replace"))
    return digest.hexdigest()


# ── Registry helpers ─────────────────────────────────────────────────────────

def effect_for_spec(spec: Mapping[str, Any], tool_name: str = "") -> Effect:
    """
    The declared effect of a registry entry.

    Accepts an `Effect`, its name, or its value, so a registry can be migrated
    without every entry changing at once. An unreadable or absent declaration
    falls back to `DEFAULT_UNDECLARED_EFFECT` with a warning rather than to
    something permissive.
    """
    declared = (spec or {}).get("effect")
    if isinstance(declared, Effect):
        return declared
    if isinstance(declared, str):
        try:
            return Effect(declared.strip().upper())
        except ValueError:
            logger.warning(
                "Tool '%s' declares an unrecognised effect %r — treating it as %s",
                tool_name, declared, DEFAULT_UNDECLARED_EFFECT.value,
            )
            return DEFAULT_UNDECLARED_EFFECT
    if declared is None:
        logger.warning(
            "Tool '%s' declares no effect — assuming %s. Declare `effect` in its "
            "registry entry.",
            tool_name, DEFAULT_UNDECLARED_EFFECT.value,
        )
        return DEFAULT_UNDECLARED_EFFECT
    logger.warning(
        "Tool '%s' declares effect of unsupported type %s — treating it as %s",
        tool_name, type(declared).__name__, DEFAULT_UNDECLARED_EFFECT.value,
    )
    return DEFAULT_UNDECLARED_EFFECT


# ── Legacy adaptation ────────────────────────────────────────────────────────

# Dict keys that describe the call rather than its payload. Excluded when
# deciding whether a legacy result actually carried anything.
_ENVELOPE_KEYS = frozenset({
    "success", "message", "error", "source", "provenance", "status",
    "effect", "preview", "idempotency_key", "tool",
})

# Explicit emptiness flags, in the order they are trusted.
_EMPTY_FLAGS = ("found", "has_data")


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        return str(value)


def _coerce_mapping(
    raw: Mapping[str, Any], *, tool: str, effect: Effect
) -> ToolResult:
    """Interpret a legacy dictionary result under the declared effect."""
    # A tool part-way through migration may already emit a status.
    declared_status = raw.get("status")
    status: Optional[ToolStatus] = None
    if isinstance(declared_status, ToolStatus):
        status = declared_status
    elif isinstance(declared_status, str):
        try:
            status = ToolStatus(declared_status.strip().lower())
        except ValueError:
            status = None  # e.g. {"status": "saved"} — a payload, not a verdict

    if status is None:
        if raw.get("success") is False or raw.get("error"):
            status = ToolStatus.ERROR
        else:
            status = _legacy_emptiness(raw)

    if status is ToolStatus.ERROR:
        message = str(raw.get("error") or raw.get("message") or "tool reported failure")
        return ToolResult(
            status=ToolStatus.ERROR,
            effect=effect,
            error=ToolError(ErrorKind.TOOL_REPORTED, message, retryable=False),
            tool=tool,
            raw=raw,
            adapted=True,
        )

    payload = {k: v for k, v in raw.items() if k not in _ENVELOPE_KEYS}
    key = raw.get("idempotency_key")
    return ToolResult(
        status=status,
        effect=effect,
        data=payload if status is ToolStatus.OK else {
            "message": str(raw.get("message") or "")
        },
        preview=str(raw["preview"]) if raw.get("preview") else None,
        idempotency_key=str(key) if key else None,
        tool=tool,
        raw=raw,
        adapted=True,
    )


def _legacy_emptiness(raw: Mapping[str, Any]) -> ToolStatus:
    """OK or NO_DATA for a legacy dict that did not report failure."""
    for flag in _EMPTY_FLAGS:
        if flag in raw:
            return ToolStatus.OK if bool(raw[flag]) else ToolStatus.NO_DATA
    if "count" in raw:
        try:
            return ToolStatus.OK if int(raw["count"]) > 0 else ToolStatus.NO_DATA
        except (TypeError, ValueError):
            return ToolStatus.OK
    payload = {k: v for k, v in raw.items() if k not in _ENVELOPE_KEYS}
    if not payload:
        return ToolStatus.NO_DATA
    return ToolStatus.OK if any(bool(v) for v in payload.values()) else ToolStatus.NO_DATA


def coerce(
    raw: Any,
    *,
    tool: str = "",
    declared_effect: Effect = Effect.READ,
    tool_input: Any = None,
) -> ToolResult:
    """
    Turn whatever a tool returned into a `ToolResult`.

    The migration bridge. Native results pass through with their effect checked
    against the registry; legacy shapes are interpreted by the conventions the
    tools already follow; anything unrecognisable is resolved according to the
    fail-safe rule — lenient for reads, an error for anything consequential.

    `tool_input` is used only to derive an idempotency key for non-read effects
    that did not supply one.
    """
    result = _coerce_inner(raw, tool=tool, declared_effect=declared_effect)

    # Give every consequential success a key, so a retry is recognisable even
    # when the tool did not think to provide one.
    if (
        result.ok
        and not result.effect.is_repeatable
        and not result.idempotency_key
        and tool_input is not None
    ):
        result = result.with_idempotency_key(derive_idempotency_key(tool, tool_input))

    if result.preview_missing:
        logger.warning(
            "Tool '%s' has effect %s but returned no preview — a confirmation "
            "step will have nothing to show the user.",
            tool or "?", result.effect.value,
        )
    return result


def _coerce_inner(raw: Any, *, tool: str, declared_effect: Effect) -> ToolResult:
    # ── Already typed ────────────────────────────────────────────────────────
    if isinstance(raw, ToolResult):
        if raw.effect.severity > declared_effect.severity:
            # A tool claiming to do more than it was registered to do. Refused
            # rather than honoured: the registry is the authority on what a
            # tool is permitted to do, and an escalation here would route
            # around whatever gate the declaration was chosen for.
            logger.error(
                "Tool '%s' returned effect %s but is registered as %s — refusing "
                "the result.",
                tool or raw.tool, raw.effect.value, declared_effect.value,
            )
            return ToolResult.failed(
                f"tool returned effect {raw.effect.value} above its registered "
                f"{declared_effect.value}",
                kind=ErrorKind.INVALID_RESULT,
                effect=declared_effect,
                tool=tool or raw.tool,
                raw=raw,
            )
        # The registered effect is authoritative for gating; a tool may not
        # downgrade itself either.
        return replace(raw, effect=declared_effect, tool=raw.tool or tool)

    # ── Legacy mapping ───────────────────────────────────────────────────────
    if isinstance(raw, Mapping):
        return _coerce_mapping(raw, tool=tool, effect=declared_effect)

    # ── Everything below is an unrecognised shape ────────────────────────────
    # For a read, interpreting it generously costs nothing. For anything that
    # may have changed the world, an uninterpretable result is a failure.
    if declared_effect.severity > Effect.LOCAL_WRITE.severity:
        return ToolResult.failed(
            f"unrecognised result of type {type(raw).__name__} from a "
            f"{declared_effect.value} tool",
            kind=ErrorKind.INVALID_RESULT,
            effect=declared_effect,
            tool=tool,
            raw=raw,
        )

    if raw is None:
        return ToolResult.no_data(effect=declared_effect, tool=tool, raw=raw)

    if isinstance(raw, (list, tuple, set, frozenset, str, bytes)):
        status = ToolStatus.OK if len(raw) > 0 else ToolStatus.NO_DATA
        return ToolResult(
            status=status,
            effect=declared_effect,
            data={"value": list(raw) if isinstance(raw, (set, frozenset)) else raw}
            if status is ToolStatus.OK else {},
            tool=tool,
            raw=raw,
            adapted=True,
        )

    if isinstance(raw, (bool, int, float)):
        status = ToolStatus.OK if raw else ToolStatus.NO_DATA
        return ToolResult(
            status=status,
            effect=declared_effect,
            data={"value": raw} if status is ToolStatus.OK else {},
            tool=tool,
            raw=raw,
            adapted=True,
        )

    # An arbitrary object from a read-only tool. Preserved rather than refused —
    # this is the historical behaviour and it is harmless here.
    return ToolResult(
        status=ToolStatus.OK,
        effect=declared_effect,
        data={"value": raw},
        tool=tool,
        raw=raw,
        adapted=True,
    )


__all__ = [
    "ToolStatus",
    "Effect",
    "ErrorKind",
    "ToolError",
    "ToolResult",
    "coerce",
    "derive_idempotency_key",
    "effect_for_spec",
    "DEFAULT_UNDECLARED_EFFECT",
]
