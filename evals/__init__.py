"""
Measuring whether the agent actually works.

The test suite answers "is the code correct". This answers a different and
less comfortable question: *given a real user turn, does the assistant do the
right thing* — call the right tool, refuse to invent, recover when it stumbles,
and answer fast enough to be usable.

Two modes, because two different things can be wrong and averaging them tells
you about neither:

    deterministic   the model's decisions are scripted (`tests.support`), so
                    what is measured is the SYSTEM — routing, grounding
                    enforcement, tool wiring, the retry path. Reproducible,
                    free, no network, runs in CI, byte-identical every run.

    live            the real model against the real Groq endpoint. What is
                    measured is the MODEL — does it actually call the tool it
                    was told to, does it invent facts when it shouldn't.
                    Costs tokens, is rate-limited, and is *not* reproducible.

Reporting a single blended number over both would be dishonest: a deterministic
pass proves the system routes and enforces correctly and says nothing about
whether `gpt-oss-120b` follows instructions; a live pass says the opposite. So
they are run and reported separately, and the report always states which one
produced a figure.

Nothing here asserts. It measures and reports. `tests/test_evals.py` is what
turns a subset of these into pass/fail gates, so a regression in the metrics
that matter breaks the build rather than quietly drifting.
"""
from evals.metrics import (
    EvalReport,
    FailureKind,
    TurnResult,
    summarize,
)
from evals.scenarios import SCENARIOS, Scenario

__all__ = [
    "EvalReport",
    "FailureKind",
    "SCENARIOS",
    "Scenario",
    "TurnResult",
    "summarize",
]
