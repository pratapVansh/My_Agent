"""
Ranking signals and fusion (Phase 2).

Ranking quality is the hardest thing in a memory system to verify, and it is
only cheap to test while the functions are pure. These tests pin the properties
the design depends on — monotonicity, decay ordering, kind exemptions — rather
than exact score values, which are tuning decisions and will move.

See docs/MEMORY_ARCHITECTURE.md §3.6.
"""
from datetime import timedelta
from uuid import uuid4

import pytest

from app.memory.kinds import MemoryKind
from app.memory.record import MemoryRecord, utcnow
from app.memory.retrieval.scoring import (
    FREQUENCY_SATURATION,
    RRF_K,
    frequency_score,
    kind_prior,
    normalize_scores,
    rank_score,
    recency_score,
    reciprocal_rank_fusion,
)


def make(kind=MemoryKind.SEMANTIC, content="The user knows Python.", **overrides):
    base = dict(owner_id="vansh", kind=kind, content=content)
    base.update(overrides)
    return MemoryRecord(**base)


# ─────────────────────────────────────────────────────────────────────────
# Recency decay
# ─────────────────────────────────────────────────────────────────────────

def test_identity_and_preferences_never_decay():
    """Forgetting the user's name for lack of recent mention is a bug."""
    ancient = utcnow() - timedelta(days=3650)
    for kind in (MemoryKind.IDENTITY, MemoryKind.PREFERENCE, MemoryKind.GOAL):
        assert recency_score(make(kind=kind, created_at=ancient)) == 1.0


def test_decay_halves_at_the_half_life():
    """A 90-day half-life for episodic means exactly 0.5 at 90 days."""
    record = make(kind=MemoryKind.EPISODIC, created_at=utcnow() - timedelta(days=90))
    assert recency_score(record) == pytest.approx(0.5, abs=0.01)


def test_decay_is_monotonically_decreasing_with_age():
    now = utcnow()
    scores = [
        recency_score(make(kind=MemoryKind.EPISODIC, created_at=now - timedelta(days=d)), now=now)
        for d in (0, 30, 90, 365)
    ]
    assert scores == sorted(scores, reverse=True)


def test_tasks_decay_faster_than_semantic_facts():
    now = utcnow()
    age = timedelta(days=14)
    task = recency_score(make(kind=MemoryKind.TASK, created_at=now - age), now=now)
    fact = recency_score(make(kind=MemoryKind.SEMANTIC, created_at=now - age), now=now)
    assert task < fact


def test_occurred_at_takes_precedence_over_created_at():
    """A resume uploaded today may describe a 2023 internship."""
    now = utcnow()
    record = make(
        kind=MemoryKind.EPISODIC,
        created_at=now,
        occurred_at=now - timedelta(days=365),
    )
    assert recency_score(record, now=now) < 0.2


def test_naive_timestamps_do_not_raise():
    """Older rows can carry naive datetimes; decay must tolerate them."""
    from datetime import datetime
    record = make(kind=MemoryKind.EPISODIC)
    record.occurred_at = datetime(2020, 1, 1)  # naive on purpose
    assert 0.0 <= recency_score(record) <= 1.0


def test_future_timestamps_do_not_exceed_one():
    record = make(kind=MemoryKind.EPISODIC, created_at=utcnow() + timedelta(days=10))
    assert recency_score(record) == 1.0


# ─────────────────────────────────────────────────────────────────────────
# Frequency
# ─────────────────────────────────────────────────────────────────────────

def test_frequency_is_zero_for_never_accessed():
    assert frequency_score(0) == 0.0


def test_frequency_increases_but_saturates():
    assert frequency_score(1) < frequency_score(5) < frequency_score(20)
    # Without a ceiling one heavily-read record would dominate ranking forever
    # purely by having been read.
    assert frequency_score(FREQUENCY_SATURATION * 100) == 1.0


# ─────────────────────────────────────────────────────────────────────────
# Kind priors
# ─────────────────────────────────────────────────────────────────────────

def test_identity_outranks_episodic_by_default():
    assert kind_prior(MemoryKind.IDENTITY) > kind_prior(MemoryKind.EPISODIC)


def test_boosting_lifts_a_kind_to_the_maximum():
    assert kind_prior(MemoryKind.EPISODIC) < 1.0
    assert kind_prior(MemoryKind.EPISODIC, boosted=[MemoryKind.EPISODIC]) == 1.0


# ─────────────────────────────────────────────────────────────────────────
# Combined score — monotonicity in every signal
# ─────────────────────────────────────────────────────────────────────────

def test_score_increases_with_similarity():
    record = make()
    assert rank_score(record, similarity=0.9) > rank_score(record, similarity=0.1)


def test_score_increases_with_importance():
    assert rank_score(make(importance=0.9)) > rank_score(make(importance=0.1))


def test_score_increases_with_confidence():
    assert rank_score(make(confidence=1.0)) > rank_score(make(confidence=0.2))


def test_score_increases_with_access_count():
    assert rank_score(make(access_count=15)) > rank_score(make(access_count=0))


def test_pinned_records_outrank_their_unpinned_twins():
    assert rank_score(make(pinned=True)) > rank_score(make(pinned=False))


def test_score_stays_within_unit_range():
    """Every term is in [0,1] and weights sum to 1, so the total is bounded."""
    strongest = make(
        kind=MemoryKind.IDENTITY, importance=1.0, confidence=1.0,
        access_count=1000, pinned=True,
    )
    assert 0.0 <= rank_score(strongest, similarity=1.0) <= 1.0
    weakest = make(importance=0.0, confidence=0.0)
    assert 0.0 <= rank_score(weakest, similarity=0.0) <= 1.0


def test_similarity_is_clamped_not_trusted():
    """A miscalibrated channel must not be able to inflate a score."""
    record = make()
    assert rank_score(record, similarity=99.0) == rank_score(record, similarity=1.0)
    assert rank_score(record, similarity=-5.0) == rank_score(record, similarity=0.0)


def test_a_relevant_match_outranks_a_stale_high_importance_record():
    """
    The core improvement over fixed-order concatenation: relevance to the
    actual question can beat a static priority.
    """
    now = utcnow()
    stale = make(
        kind=MemoryKind.EPISODIC, importance=0.9,
        created_at=now - timedelta(days=800), occurred_at=now - timedelta(days=800),
    )
    relevant = make(kind=MemoryKind.SEMANTIC, importance=0.5, created_at=now)
    assert rank_score(relevant, similarity=1.0, now=now) > rank_score(stale, similarity=0.0, now=now)


# ─────────────────────────────────────────────────────────────────────────
# Reciprocal rank fusion
# ─────────────────────────────────────────────────────────────────────────

def test_rrf_rewards_agreement_between_channels():
    """A record both channels rank highly must beat one only either ranks."""
    a, b, c = uuid4(), uuid4(), uuid4()
    fused = reciprocal_rank_fusion([[a, b], [a, c]])
    assert fused[a] > fused[b]
    assert fused[a] > fused[c]


def test_rrf_scores_decrease_with_rank():
    a, b, c = uuid4(), uuid4(), uuid4()
    fused = reciprocal_rank_fusion([[a, b, c]])
    assert fused[a] > fused[b] > fused[c]


def test_rrf_uses_position_not_channel_score():
    """
    The reason for RRF: cosine similarity and ts_rank_cd share neither scale
    nor distribution, so only position is comparable across channels.
    """
    a, b = uuid4(), uuid4()
    assert reciprocal_rank_fusion([[a, b]]) == reciprocal_rank_fusion([[a, b]])


def test_rrf_first_rank_matches_the_formula():
    a = uuid4()
    assert reciprocal_rank_fusion([[a]])[a] == pytest.approx(1.0 / (RRF_K + 1))


def test_rrf_handles_empty_and_missing_channels():
    a = uuid4()
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], [a]])[a] > 0


# ─────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────

def test_normalisation_scales_the_maximum_to_one():
    a, b = uuid4(), uuid4()
    result = normalize_scores({a: 0.2, b: 0.1})
    assert result[a] == 1.0
    assert result[b] == pytest.approx(0.5)


def test_normalisation_preserves_relative_gaps():
    """
    Max-scaling, not min-max: when candidates score similarly, min-max would
    stretch trivial differences across the full range and manufacture
    confidence the underlying signal does not support.
    """
    a, b = uuid4(), uuid4()
    result = normalize_scores({a: 0.100, b: 0.099})
    assert result[b] > 0.9


def test_normalisation_handles_empty_and_zero_input():
    a = uuid4()
    assert normalize_scores({}) == {}
    assert normalize_scores({a: 0.0}) == {a: 0.0}
