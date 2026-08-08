"""
Ranking signals and fusion.

Pure functions, deliberately: ranking quality is the single hardest thing to
verify in a memory system, and it is only cheap to test while it has no I/O.

The replaced design had no ranking at all — sources were concatenated in a
fixed order and the tail was truncated away. A stale résumé line and an active
goal competed purely by their position in a hard-coded list.

See docs/MEMORY_ARCHITECTURE.md §3.6.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence
from uuid import UUID
import math

from app.memory.kinds import KIND_HALF_LIFE_DAYS, MemoryKind
from app.memory.record import MemoryRecord, utcnow


# ── Weights ─────────────────────────────────────────────────────────────
#
# Sum to 1.0 so a score is directly interpretable as a [0, 1] quality estimate
# rather than an arbitrary magnitude. Similarity dominates because relevance to
# the actual question matters more than any prior; importance is second because
# a highly salient fact is worth surfacing even on a weak lexical match.
#
# Confidence enters as a positive term rather than a penalty. Algebraically
# equivalent, but it keeps every term in [0, 1] and the total bounded, which is
# what makes the monotonicity tests meaningful.
W_SIMILARITY = 0.40
W_IMPORTANCE = 0.25
W_RECENCY = 0.15
W_KIND_PRIOR = 0.10
W_FREQUENCY = 0.05
W_CONFIDENCE = 0.05

# Access count at which the frequency signal saturates. Without a ceiling, one
# heavily-read record would dominate ranking forever purely by having been read.
FREQUENCY_SATURATION = 20

# Records the user explicitly pinned bypass ranking entirely by being placed in
# the guaranteed tier; this boost only orders them among themselves.
PINNED_BOOST = 0.15

# Standard RRF constant. Large enough that the top few ranks are not wildly
# over-weighted relative to each other.
RRF_K = 60


# Baseline usefulness of each kind, before the query is considered. Identity
# and preferences are almost always worth injecting; an episodic fragment
# usually is not unless it actually matches.
_DEFAULT_KIND_PRIOR: Dict[MemoryKind, float] = {
    MemoryKind.IDENTITY: 1.0,
    MemoryKind.PREFERENCE: 1.0,
    MemoryKind.GOAL: 0.9,
    MemoryKind.SEMANTIC: 0.7,
    MemoryKind.TASK: 0.7,
    MemoryKind.DOCUMENT: 0.6,
    MemoryKind.PROCEDURAL: 0.5,
    MemoryKind.EPISODIC: 0.5,
    MemoryKind.RELATION: 0.5,
}


def recency_score(
    record: MemoryRecord, *, now: Optional[datetime] = None
) -> float:
    """
    Exponential decay on the record's own half-life.

    Returns 1.0 for kinds that never decay. Forgetting the user's name because
    it has not come up lately would be a bug, not a feature — so identity,
    preferences and goals are exempt by their half-life being `None`.
    """
    half_life = KIND_HALF_LIFE_DAYS.get(record.kind)
    if half_life is None:
        return 1.0

    moment = now or utcnow()
    reference = record.occurred_at or record.created_at
    if reference is None:
        return 1.0

    # Tolerate a naive timestamp from an older row rather than raising.
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=moment.tzinfo)

    age_days = max(0.0, (moment - reference).total_seconds() / 86400.0)
    return 0.5 ** (age_days / half_life)


def frequency_score(access_count: int) -> float:
    """Logarithmic, saturating. Frequently useful ≠ infinitely useful."""
    if access_count <= 0:
        return 0.0
    return min(1.0, math.log1p(access_count) / math.log1p(FREQUENCY_SATURATION))


def kind_prior(
    kind: MemoryKind, *, boosted: Optional[Iterable[MemoryKind]] = None
) -> float:
    """Baseline usefulness, raised to 1.0 when the query targets this kind."""
    if boosted and kind in set(boosted):
        return 1.0
    return _DEFAULT_KIND_PRIOR.get(kind, 0.5)


def rank_score(
    record: MemoryRecord,
    *,
    similarity: float = 0.0,
    boosted_kinds: Optional[Iterable[MemoryKind]] = None,
    now: Optional[datetime] = None,
) -> float:
    """
    Combine every signal into one comparable score in [0, 1].

    `similarity` is the fused relevance of this record to the query, already
    normalised to [0, 1] by the caller. Records reached only through the
    structured channel have no similarity and rely on their priors — which is
    correct: an active goal deserves surfacing whether or not the current
    question happens to mention it.
    """
    score = (
        W_SIMILARITY * max(0.0, min(1.0, similarity))
        + W_IMPORTANCE * record.importance
        + W_RECENCY * recency_score(record, now=now)
        + W_KIND_PRIOR * kind_prior(record.kind, boosted=boosted_kinds)
        + W_FREQUENCY * frequency_score(record.access_count)
        + W_CONFIDENCE * record.confidence
    )
    if record.pinned:
        score += PINNED_BOOST
    return round(min(1.0, score), 6)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[UUID]], *, k: int = RRF_K
) -> Dict[UUID, float]:
    """
    Fuse several ranked id lists into one score per id.

    RRF is used rather than score averaging because the channels produce
    incomparable numbers: cosine similarity and `ts_rank_cd` share neither a
    scale nor a distribution, so averaging them would silently let whichever
    channel happens to emit larger magnitudes dominate. RRF only reads
    *position*, which every channel agrees on.

    Returns raw RRF scores; `normalize_scores` maps them into [0, 1] for use as
    the similarity term.
    """
    fused: Dict[UUID, float] = {}
    for ranking in rankings:
        for position, record_id in enumerate(ranking):
            fused[record_id] = fused.get(record_id, 0.0) + 1.0 / (k + position + 1)
    return fused


def normalize_scores(scores: Dict[UUID, float]) -> Dict[UUID, float]:
    """
    Scale scores to [0, 1] by their maximum.

    Max-scaling rather than min-max: when every candidate scores similarly,
    min-max would stretch trivial differences into the full range and
    manufacture confidence that the underlying signal does not support.
    """
    if not scores:
        return {}
    highest = max(scores.values())
    if highest <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / highest for key, value in scores.items()}
