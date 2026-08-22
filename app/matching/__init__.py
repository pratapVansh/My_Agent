"""
Job requirements, matched against evidence, explained without invention.

Stages 3, 4 and 6 of the job pipeline in `docs/ARCHITECTURE_AUDIT.md §5`:
posting text → `Requirement[]` → a verdict per requirement → rendered prose.

Everything here is a pure function. No model, no network, no clock — the same
posting and the same profile always produce the same report and the same score,
which is what makes the number arguable and the tests worth writing.

The package inherits `app.candidate`'s invariant rather than restating it: a
positive verdict is unconstructible without evidence, and the evidence carries
the identifiers ingestion already assigned. It also inherits the answerability
distinction: `MISSING` may only be emitted when the profile was healthy enough
to establish absence, and `UNKNOWN` covers every case where it was not.

Read `models.py` first — the four match statuses and the two constructor
refusals are the design; the rest is bookkeeping around them.
"""
from app.matching.explain import render, render_summary
from app.matching.extract import extract_requirements
from app.matching.matcher import match_requirements
from app.matching.models import (
    Band,
    JobRequirements,
    MatchReport,
    MatchStatus,
    Necessity,
    Requirement,
    RequirementKind,
    RequirementMatch,
)
from app.matching.scoring import compute_band, compute_score

__all__ = [
    "Band",
    "JobRequirements",
    "MatchReport",
    "MatchStatus",
    "Necessity",
    "Requirement",
    "RequirementKind",
    "RequirementMatch",
    "compute_band",
    "compute_score",
    "extract_requirements",
    "match_requirements",
    "render",
    "render_summary",
]
