"""
Tools that misbehave on demand, and count what actually happened to them.

The three states a tool can end in — returned data, returned nothing, failed —
are the distinction the whole contract exists to preserve, and they are almost
impossible to provoke against real services. A real Qdrant does not return
`NO_DATA` because you asked it to; a real SMTP server does not time out on cue.

So each outcome is a value here, and a `FakeTool` is a scripted sequence of
them. That sequencing is what makes retry behaviour testable: `[ERROR, OK]` is
a tool that fails once and then works, which is exactly the shape the reflect
loop was built for and had no test covering.

The counting matters as much as the failing. For a gated tool the interesting
assertion is almost never about its return value — it is `was_called is False`.
"Did this run?" has to be a fact the test can read, not something inferred from
an absence of side effects.
"""
from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.tools.contract import Effect, ToolResult


class Outcome(str, Enum):
    """What a fake tool does when called."""

    OK = "ok"
    """Returns a populated legacy-style payload."""

    NO_DATA = "no_data"
    """Succeeds and finds nothing — `{"success": True, "found": False}`."""

    ERROR = "error"
    """Reports its own failure — `{"success": False, "error": ...}`."""

    RAISE = "raise"
    """Raises. Distinct from ERROR: the loop's except-branch, not the tool's."""

    TIMEOUT = "timeout"
    """Never returns in time. Exercises the loop's wait_for."""

    MALFORMED = "malformed"
    """Returns something uninterpretable — the fail-safe rule's input."""

    TYPED_OK = "typed_ok"
    """Returns a native `ToolResult`, the shape tools migrate to."""


class FakeTool:
    """
    A recording stand-in for one tool.

    `outcomes` is consumed one per call; the last entry repeats once exhausted,
    so a single-outcome tool can be called any number of times.
    """

    def __init__(
        self,
        name: str = "fake_tool",
        *,
        outcome: Outcome | Sequence[Outcome] = Outcome.OK,
        data: Optional[Mapping[str, Any]] = None,
        effect: Effect = Effect.READ,
        error_message: str = "simulated tool failure",
        timeout_seconds: float = 30.0,
        preview: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.effect = effect
        self.data = dict(data or {"value": "something"})
        self.error_message = error_message
        self.timeout_seconds = timeout_seconds
        self.preview = preview
        self._outcomes: List[Outcome] = (
            [outcome] if isinstance(outcome, Outcome) else list(outcome)
        )
        if not self._outcomes:
            self._outcomes = [Outcome.OK]
        self.calls: List[Dict[str, Any]] = []

    # ── Observation ──────────────────────────────────────────────────────────

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def was_called(self) -> bool:
        """The assertion most gateway tests actually want."""
        return bool(self.calls)

    @property
    def last_call(self) -> Optional[Dict[str, Any]]:
        return self.calls[-1] if self.calls else None

    def called_with(self, **expected: Any) -> bool:
        """Whether any invocation carried all of these argument values."""
        return any(
            all(call.get(key) == value for key, value in expected.items())
            for call in self.calls
        )

    def reset(self) -> None:
        self.calls.clear()

    # ── Behaviour ────────────────────────────────────────────────────────────

    def _next_outcome(self) -> Outcome:
        if len(self._outcomes) > 1:
            return self._outcomes.pop(0)
        return self._outcomes[0]

    async def __call__(self, tool_input: Optional[Mapping[str, Any]] = None) -> Any:
        self.calls.append(dict(tool_input or {}))
        outcome = self._next_outcome()

        if outcome is Outcome.TIMEOUT:
            await asyncio.sleep(self.timeout_seconds)
            return {"success": True}
        if outcome is Outcome.RAISE:
            raise RuntimeError(self.error_message)
        if outcome is Outcome.ERROR:
            return {"success": False, "error": self.error_message}
        if outcome is Outcome.NO_DATA:
            return {"success": True, "found": False, "message": "nothing on file"}
        if outcome is Outcome.MALFORMED:
            return object()
        if outcome is Outcome.TYPED_OK:
            return ToolResult.success(
                self.data, effect=self.effect, tool=self.name,
                preview=self.preview if isinstance(self.preview, str) else None,
            )
        return {"success": True, "found": True, **self.data}

    # ── Registry ─────────────────────────────────────────────────────────────

    def spec(self, **extra: Any) -> Dict[str, Any]:
        """A registry entry for this tool, shaped like the real ones."""
        entry: Dict[str, Any] = {
            "callable": self,
            "effect": self.effect,
            "description": f"fake {self.name}",
        }
        if self.preview is not None:
            entry["preview"] = self.preview
        entry.update(extra)
        return entry


def registry(*tools: FakeTool) -> Dict[str, Dict[str, Any]]:
    """Build a tool registry from fakes, keyed by name."""
    return {tool.name: tool.spec() for tool in tools}


__all__ = ["FakeTool", "Outcome", "registry"]
