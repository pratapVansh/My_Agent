"""
What a turn produced, and how a run of them adds up.

The metrics here are deliberately narrow and each one is falsifiable from a
single turn's observable record. There is no composite "quality score": a
number blended from routing accuracy and latency and grounding cannot be acted
on, because no single change moves it in a knowable direction.

The one metric that is not an average is `grounding_violations`. Every other
figure is a rate to be improved; that one is a count that must be zero, because
each violation is a personal fact stated to the user with nothing behind it.
A run with 98% task success and one grounding violation is a worse system than
one with 90% and none, and a scoreboard that cannot express that is the wrong
scoreboard.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class FailureKind(str, Enum):
    """Why a turn did not succeed. One cause per turn, most specific first."""

    NONE = "none"

    GROUNDING_VIOLATION = "grounding_violation"
    """An ungrounded personal fact reached the user. The only category that is
    a safety failure rather than a quality one."""

    WRONG_AGENT = "wrong_agent"
    """Routed somewhere that could not answer."""

    MISSING_TOOL_CALL = "missing_tool_call"
    """A required tool was never called and the turn still ended."""

    WRONG_CONTENT = "wrong_content"
    """Expected substance absent, or forbidden substance present."""

    STYLE_VIOLATION = "style_violation"
    """The answer was correct but read like a chat product — a markdown
    heading, a bold label, or a stock closing offer. Separated from
    `WRONG_CONTENT` because the fix is a prompt change rather than a logic
    one, and because a run can be entirely correct and still fail this."""

    TOOL_ERROR = "tool_error"
    """A tool the turn depended on failed."""

    EXCEPTION = "exception"
    """The turn raised."""

    TIMEOUT = "timeout"


@dataclass
class TurnResult:
    """One scenario, run once."""

    scenario_id: str
    passed: bool
    failure: FailureKind = FailureKind.NONE
    detail: str = ""

    latency_ms: float = 0.0
    agent: str = ""
    grounding: str = ""
    answerability: str = ""
    tools_called: List[str] = field(default_factory=list)
    answer: str = ""

    # Whether the turn went round the reflect loop at least once.
    retried: bool = False

    # Whether this scenario actually demanded a tool call. Adversarial
    # scenarios — where the model is scripted to skip its tool and refusing is
    # the correct outcome — set this False, so they cannot drag down a rate
    # that is meant to describe compliance.
    required_tool: bool = False

    def summary(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario_id,
            "passed": self.passed,
            "failure": self.failure.value,
            "detail": self.detail[:200],
            "latency_ms": round(self.latency_ms, 1),
            "agent": self.agent,
            "grounding": self.grounding,
            "tools": self.tools_called,
            "retried": self.retried,
        }


@dataclass
class EvalReport:
    """A run of scenarios, aggregated."""

    mode: str
    results: List[TurnResult]

    # ── Headline ─────────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def task_success_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    # ── Safety ───────────────────────────────────────────────────────────────

    @property
    def grounding_violations(self) -> int:
        """Must be zero. See the module docstring."""
        return sum(
            1 for r in self.results
            if r.failure is FailureKind.GROUNDING_VIOLATION
        )

    # ── Tool behaviour ───────────────────────────────────────────────────────

    @property
    def tool_call_rate(self) -> Optional[float]:
        """
        Of the turns that genuinely required a lookup, how many performed one.

        Scoped to scenarios that demanded a tool call. The adversarial
        scenarios deliberately script the model into skipping its tool, and
        counting those here would report the suite's own design as if it were
        the agent's failure rate.

        None when no scenario in the run required a tool — an honest absence
        rather than a fabricated 100%.
        """
        required = [r for r in self.results if r.required_tool]
        if not required:
            return None
        called = sum(1 for r in required if r.grounding != "skipped")
        return called / len(required)

    @property
    def retry_recovery_count(self) -> int:
        """
        Turns that skipped their tool, retried, and then actually called it.

        Deliberately not "retried and passed": an adversarial turn that
        retries three times and then correctly refuses has *passed*, but it
        recovered nothing. Only a turn that ended with its tool actually run
        counts.
        """
        return sum(
            1 for r in self.results
            if r.retried and r.grounding not in ("skipped", "")
        )

    # ── Latency ──────────────────────────────────────────────────────────────

    def latency(self, percentile: int) -> float:
        values = sorted(r.latency_ms for r in self.results)
        if not values:
            return 0.0
        if percentile >= 100:
            return values[-1]
        index = min(int(len(values) * percentile / 100), len(values) - 1)
        return values[index]

    @property
    def latency_mean(self) -> float:
        values = [r.latency_ms for r in self.results]
        return statistics.fmean(values) if values else 0.0

    # ── Breakdown ────────────────────────────────────────────────────────────

    @property
    def failures_by_kind(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.results:
            if r.failure is not FailureKind.NONE:
                out[r.failure.value] = out.get(r.failure.value, 0) + 1
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "total": self.total,
            "passed": self.passed,
            "task_success_rate": round(self.task_success_rate, 4),
            "grounding_violations": self.grounding_violations,
            "tool_call_rate": (
                round(self.tool_call_rate, 4)
                if self.tool_call_rate is not None else None
            ),
            "retry_recoveries": self.retry_recovery_count,
            "latency_ms": {
                "mean": round(self.latency_mean, 1),
                "p50": round(self.latency(50), 1),
                "p95": round(self.latency(95), 1),
                "max": round(self.latency(100), 1),
            },
            "failures_by_kind": self.failures_by_kind,
            "results": [r.summary() for r in self.results],
        }

    def render(self) -> str:
        """A human-readable report. Safety first, then quality, then speed."""
        required = sum(1 for r in self.results if r.required_tool)
        tool_rate = (
            f"{self.tool_call_rate:.0%}  ({required} turn(s) required one)"
            if self.tool_call_rate is not None
            else "n/a  (no scenario in this run required a lookup)"
        )
        lines = [
            "=" * 68,
            f"  AGENT EVALUATION — mode: {self.mode}",
            "=" * 68,
            "",
            f"  Task success         {self.passed}/{self.total}"
            f"  ({self.task_success_rate:.0%})",
            f"  Grounding violations {self.grounding_violations}"
            + ("   <-- MUST BE ZERO" if self.grounding_violations else "   (none)"),
            f"  Tool-call rate       {tool_rate}",
            f"  Retry recoveries     {self.retry_recovery_count}"
            "   (skipped the tool, retried, then called it)",
            "",
            f"  Latency  mean {self.latency_mean:7.1f} ms"
            f"   p50 {self.latency(50):7.1f} ms"
            f"   p95 {self.latency(95):7.1f} ms"
            f"   max {self.latency(100):7.1f} ms",
            "",
        ]

        if self.failures_by_kind:
            lines.append("  Failures by kind:")
            for kind, count in sorted(
                self.failures_by_kind.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"    {kind:<22} {count}")
            lines.append("")

        failed = [r for r in self.results if not r.passed]
        if failed:
            lines.append("  Failing scenarios:")
            for r in failed:
                lines.append(f"    [{r.failure.value}] {r.scenario_id}")
                if r.detail:
                    lines.append(f"        {r.detail[:120]}")
            lines.append("")

        lines.append("=" * 68)
        return "\n".join(lines)


def summarize(mode: str, results: Sequence[TurnResult]) -> EvalReport:
    return EvalReport(mode=mode, results=list(results))
