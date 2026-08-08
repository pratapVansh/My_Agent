"""
Retrieval observability.

Every retrieval records what it considered, what it chose, and what it dropped.
Without this, "the assistant forgot something" is unfalsifiable: there is no way
to tell a retrieval miss from a ranking miss from a budget eviction. It is also
what makes shadow-mode comparison possible at all.

See docs/MEMORY_ARCHITECTURE.md §3.11.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass
class ChannelResult:
    """What one candidate-generation channel contributed."""

    name: str
    candidates: int = 0
    latency_ms: float = 0.0
    ok: bool = True
    error: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "channel": self.name,
            "candidates": self.candidates,
            "latency_ms": round(self.latency_ms, 1),
            "ok": self.ok,
            **({"error": self.error} if self.error else {}),
        }


@dataclass
class SelectedItem:
    """One record that made it into the prompt."""

    record_id: UUID
    kind: str
    score: float
    tier: int
    tokens: int
    preview: str

    def summary(self) -> Dict[str, Any]:
        return {
            "id": str(self.record_id)[:8],
            "kind": self.kind,
            "score": round(self.score, 4),
            "tier": self.tier,
            "tokens": self.tokens,
            "preview": self.preview[:80],
        }


@dataclass
class DroppedItem:
    """One record that ranked well enough but did not fit."""

    record_id: UUID
    kind: str
    score: float
    reason: str

    def summary(self) -> Dict[str, Any]:
        return {
            "id": str(self.record_id)[:8],
            "kind": self.kind,
            "score": round(self.score, 4),
            "reason": self.reason,
        }


@dataclass
class RetrievalTrace:
    """The full record of one retrieval."""

    owner_id: str
    query: str = ""
    channels: List[ChannelResult] = field(default_factory=list)
    fused_candidates: int = 0
    ranked_candidates: int = 0
    selected: List[SelectedItem] = field(default_factory=list)
    dropped: List[DroppedItem] = field(default_factory=list)
    budget_tokens: int = 0
    used_tokens: int = 0
    total_latency_ms: float = 0.0

    def add_channel(self, result: ChannelResult) -> None:
        self.channels.append(result)

    @property
    def budget_utilisation(self) -> float:
        if self.budget_tokens <= 0:
            return 0.0
        return round(self.used_tokens / self.budget_tokens, 3)

    @property
    def degraded(self) -> bool:
        """True when any channel failed — the result is usable but incomplete."""
        return any(not channel.ok for channel in self.channels)

    def summary(self) -> Dict[str, Any]:
        """Compact, log-friendly form. Never includes full record content."""
        return {
            "owner_id": self.owner_id,
            "query": self.query[:120],
            "channels": [c.summary() for c in self.channels],
            "fused_candidates": self.fused_candidates,
            "ranked_candidates": self.ranked_candidates,
            "selected": len(self.selected),
            "dropped": len(self.dropped),
            "by_kind": self.selected_by_kind(),
            "budget_tokens": self.budget_tokens,
            "used_tokens": self.used_tokens,
            "utilisation": self.budget_utilisation,
            "degraded": self.degraded,
            "latency_ms": round(self.total_latency_ms, 1),
        }

    def selected_by_kind(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.selected:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return counts
