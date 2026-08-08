"""
Context assembly: tier allocation → token budget → render.

The distinction from the previous design is the ordering. Before: render every
section, concatenate, then cut at character 20,000 and rely on section order to
decide what survived. Now: decide what fits *first*, then render only that.

Truncation-after-render fails in the way that matters most — it cuts mid-record,
producing a fragment the model reads as a complete statement. Allocation-before-
render can only ever drop whole records, and it records every drop in the trace.

Tier 0 is guaranteed. If identity, preferences and active goals cannot fit, the
budget is misconfigured; silently dropping them would make the assistant forget
who it is speaking to in order to make room for a résumé fragment.

See docs/MEMORY_ARCHITECTURE.md §3.7.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import logging

from app.memory.kinds import MemoryKind
from app.memory.record import MemoryRecord
from app.memory.retrieval.engine import ScoredRecord
from app.memory.retrieval.trace import DroppedItem, RetrievalTrace, SelectedItem
from app.services.chunking_service import chunking_service

logger = logging.getLogger(__name__)


# Default budget, in tokens, for the whole memory block.
DEFAULT_BUDGET_TOKENS = 6000

# Share of the budget each tier may claim. Tier 0 is a floor, not a cap: it is
# always rendered in full, and its allocation only bounds how much the *other*
# tiers may assume is left.
TIER_SHARES: Dict[int, float] = {
    0: 0.15,   # identity, preferences, goals — guaranteed
    1: 0.35,   # working memory (conversation window)
    2: 0.40,   # ranked retrieval
    3: 0.10,   # opportunistic: procedural hints, relations
}

# Which kinds belong to which tier.
TIER_0_KINDS = frozenset({
    MemoryKind.IDENTITY, MemoryKind.PREFERENCE, MemoryKind.GOAL
})
TIER_3_KINDS = frozenset({MemoryKind.PROCEDURAL, MemoryKind.RELATION})

_SECTION_TITLES: Dict[MemoryKind, str] = {
    MemoryKind.IDENTITY: "About the user",
    MemoryKind.PREFERENCE: "User preferences",
    MemoryKind.GOAL: "Active goals",
    MemoryKind.TASK: "Open tasks",
    MemoryKind.SEMANTIC: "What I know about the user",
    MemoryKind.EPISODIC: "Recent activity",
    MemoryKind.DOCUMENT: "From the user's documents",
    MemoryKind.PROCEDURAL: "Approaches that worked before",
    MemoryKind.RELATION: "Related people, projects and tools",
}

# Order sections are rendered in. Independent of budget — allocation already
# decided what is present — but a stable order keeps prompts cacheable and
# diffable between turns.
_SECTION_ORDER: List[MemoryKind] = [
    MemoryKind.IDENTITY,
    MemoryKind.PREFERENCE,
    MemoryKind.GOAL,
    MemoryKind.TASK,
    MemoryKind.SEMANTIC,
    MemoryKind.EPISODIC,
    MemoryKind.DOCUMENT,
    MemoryKind.PROCEDURAL,
    MemoryKind.RELATION,
]


def count_tokens(text: str) -> int:
    """Token count via the shared tokenizer, with a character-based fallback."""
    try:
        return chunking_service.count_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


def tier_for(record: MemoryRecord) -> int:
    """Which budget tier a record competes in."""
    if record.kind in TIER_0_KINDS:
        return 0
    if record.kind in TIER_3_KINDS:
        return 3
    return 2


@dataclass
class AssembledContext:
    """The rendered memory block plus what it cost and what it cost us."""

    text: str
    used_tokens: int = 0
    selected: List[ScoredRecord] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class ContextAssembler:
    """Allocates a token budget across tiers, then renders."""

    def __init__(self, budget_tokens: int = DEFAULT_BUDGET_TOKENS):
        self.budget_tokens = budget_tokens

    def assemble(
        self,
        scored: Sequence[ScoredRecord],
        *,
        working_memory: str = "",
        trace: Optional[RetrievalTrace] = None,
    ) -> AssembledContext:
        """
        Select what fits, then render it.

        `working_memory` is the conversation window, supplied by the caller
        rather than retrieved: it is not a stored memory kind, it is the live
        transcript. It occupies tier 1 and is compressed rather than dropped.
        """
        budget = self.budget_tokens
        if trace is not None:
            trace.budget_tokens = budget

        by_tier: Dict[int, List[ScoredRecord]] = {0: [], 2: [], 3: []}
        for item in scored:
            by_tier[tier_for(item.record)].append(item)
        for items in by_tier.values():
            items.sort(key=lambda i: i.score, reverse=True)

        selected: List[ScoredRecord] = []
        dropped: List[tuple[ScoredRecord, str]] = []
        used = 0

        # ── Tier 0: guaranteed, never dropped for budget ────────────────
        for item in by_tier[0]:
            cost = self._cost(item.record)
            selected.append(item)
            used += cost
        tier0_used = used

        if tier0_used > budget * TIER_SHARES[0] * 2:
            # Not fatal — tier 0 is still rendered — but it means the other
            # tiers are being squeezed by something that should be small.
            logger.warning(
                "Tier-0 memory is %d tokens against a %d-token budget; "
                "identity/preference/goal records may need consolidating.",
                tier0_used, budget,
            )

        # ── Tier 1: working memory, compressed rather than dropped ──────
        working_budget = int(budget * TIER_SHARES[1])
        working_text = ""
        if working_memory.strip():
            working_text = self._fit_text(working_memory, working_budget)
            used += count_tokens(working_text)

        # ── Tier 2, then 3: greedy fill by score ────────────────────────
        for tier in (2, 3):
            tier_ceiling = used + int(budget * TIER_SHARES[tier])
            for item in by_tier[tier]:
                cost = self._cost(item.record)
                if used + cost > min(tier_ceiling, budget):
                    dropped.append((item, f"tier_{tier}_budget"))
                    continue
                selected.append(item)
                used += cost

        text = self._render(selected, working_text)

        if trace is not None:
            trace.used_tokens = used
            trace.selected = [
                SelectedItem(
                    record_id=item.record.id,
                    kind=item.record.kind.value,
                    score=item.score,
                    tier=tier_for(item.record),
                    tokens=self._cost(item.record),
                    preview=item.record.content,
                )
                for item in selected
            ]
            trace.dropped = [
                DroppedItem(
                    record_id=item.record.id,
                    kind=item.record.kind.value,
                    score=item.score,
                    reason=reason,
                )
                for item, reason in dropped
            ]

        return AssembledContext(text=text, used_tokens=used, selected=selected)

    # ── internals ───────────────────────────────────────────────────────

    def _cost(self, record: MemoryRecord) -> int:
        """Token cost of one rendered bullet, including its marker."""
        return count_tokens(f"- {record.content}") + 1

    def _fit_text(self, text: str, budget_tokens: int) -> str:
        """
        Trim free text to a token budget, dropping whole leading lines.

        Oldest-first: the conversation window's most recent turns are the ones
        that carry the thread.
        """
        if budget_tokens <= 0:
            return ""
        lines = [line for line in text.splitlines() if line.strip()]
        while lines and count_tokens("\n".join(lines)) > budget_tokens:
            lines.pop(0)
        return "\n".join(lines)

    def _render(self, selected: Sequence[ScoredRecord], working_text: str) -> str:
        """Group by kind and render in a stable order."""
        grouped: Dict[MemoryKind, List[ScoredRecord]] = {}
        for item in selected:
            grouped.setdefault(item.record.kind, []).append(item)

        blocks: List[str] = []
        for kind in _SECTION_ORDER:
            items = grouped.get(kind)
            if not items:
                continue
            title = _SECTION_TITLES.get(kind, kind.value.title())
            lines = "\n".join(f"- {item.record.content}" for item in items)
            blocks.append(f"{title}:\n{lines}")

        if working_text.strip():
            blocks.append(f"Recent conversation:\n{working_text}")

        return "\n\n".join(blocks)


# Singleton instance
context_assembler = ContextAssembler()
