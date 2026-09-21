"""
The evaluation suite, as a build gate.

`scripts/run_eval.py` is for looking at the numbers. This is for stopping a
regression: the properties asserted here are the ones whose loss would make the
agent unsafe or useless, and they are asserted as thresholds rather than as
exact figures so that ordinary drift does not produce a red build for no reason.

Only deterministic mode runs here. Live mode needs a real API key, spends
tokens, is rate-limited, and is not reproducible — everything a test must not
be. The live figures belong in the report, not in the build.
"""
from __future__ import annotations

import pytest

from evals.metrics import FailureKind
from evals.runner import run_suite
from evals.scenarios import SCENARIOS, for_mode


@pytest.fixture(scope="module")
async def report():
    return await run_suite(mode="deterministic")


# ── The gate that matters most ───────────────────────────────────────────────

async def test_no_grounding_violation_reaches_the_user(report):
    """
    Zero, not "few". Every violation here is a personal fact stated to the user
    with nothing behind it — the failure the whole grounding layer exists to
    prevent, measured end to end through the real graph.
    """
    offenders = [
        r.scenario_id for r in report.results
        if r.failure is FailureKind.GROUNDING_VIOLATION
    ]
    assert not offenders, f"invented content was delivered by: {offenders}"


async def test_every_scenario_passes(report):
    """
    The suite is small and every scenario is a property worth holding. If one
    starts failing, that is a real regression rather than noise.
    """
    failing = [(r.scenario_id, r.failure.value, r.detail) for r in report.results if not r.passed]
    assert not failing, f"failing scenarios: {failing}"


async def test_a_turn_that_requires_a_lookup_performs_one(report):
    """Compliance, measured only over turns that genuinely demanded a tool."""
    rate = report.tool_call_rate
    assert rate is not None, "the suite must contain at least one tool-requiring turn"
    assert rate == 1.0, f"tool-call rate fell to {rate:.0%}"


async def test_the_retry_path_is_exercised(report):
    """
    A recovery is the reflect loop doing its job. If this drops to zero the
    retry has silently stopped firing — which is exactly how the bug it was
    written for went unnoticed, since a skipped tool still produces a polite
    answer.
    """
    assert report.retry_recovery_count >= 1


# ── Suite integrity ──────────────────────────────────────────────────────────

def test_scenarios_have_distinct_ids():
    ids = [s.id for s in SCENARIOS]
    assert len(ids) == len(set(ids))


def test_every_scenario_states_what_it_tests():
    """A scenario whose purpose is not written down cannot be maintained."""
    for scenario in SCENARIOS:
        assert scenario.why.strip(), scenario.id


def test_both_modes_have_scenarios():
    assert for_mode("deterministic")
    assert for_mode("live")


def test_the_suite_measures_misbehaviour_not_only_compliance():
    """
    The property that makes this suite worth anything. A suite that only
    scripts the model cooperating measures the easy half of the problem.
    """
    adversarial = [s for s in SCENARIOS if s.forbid_substrings]
    assert len(adversarial) >= 5, (
        "too few scenarios script the model misbehaving; the system's real job "
        "is to be correct when the model is wrong"
    )
