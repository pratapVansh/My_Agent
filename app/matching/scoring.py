"""
The number, and why a model is not allowed to produce it.

A language model asked "how well do I match this?" will answer with a
percentage. The percentage will be fluent, confident, and unconnected to
anything — ask twice and get 72% and 85% for identical inputs. That is not a
rounding problem: a score nobody can reproduce cannot be argued with, and a
candidate cannot act on "the model felt good about it".

So the score is arithmetic over the match table, computed here, before any
prompt is assembled. The model receives a finished number and its working. It
has no opportunity to disagree because it is never asked.

Three rules, and each exists because of a specific way the previous
`0.5·source + 0.25·overlap + 0.25·skill_ratio` blend was wrong:

**Required outranks preferred, by construction.** Required requirements carry
80% of the weight and preferred 20%, so clearing every wish while failing a bar
cannot produce a good score. The old blend had no notion of a bar at all.

**Coverage is measured against what the posting asked for, not against the
résumé.** The old `skill_match_ratio` divided by the candidate's *entire* skill
set, which mechanically punished having a broad résumé — the better your
background, the lower every score. Here the denominator is the requirement
list, which is the only denominator that means anything.

**UNKNOWN is excluded rather than scored as zero.** This is the answerability
rule in arithmetic form. A requirement nobody could check must not depress the
score, because a Qdrant outage would then be indistinguishable from an
unqualified candidate — and the resulting number would be a statement about
infrastructure wearing the costume of a statement about a person.

The band is a second, coarser verdict with one rule the score alone cannot
express: **a missing required requirement caps the band below STRONG**, however
well everything else went. Nine matches and one failed bar is not a strong
match, and no weighted average says so loudly enough.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from app.matching.models import (
    Band,
    MatchStatus,
    Necessity,
    RequirementMatch,
)

# Weights. Required dominates by design — see the module docstring.
REQUIRED_WEIGHT = 0.8
PREFERRED_WEIGHT = 0.2

# Band thresholds, inclusive lower bounds.
BAND_THRESHOLDS: Tuple[Tuple[float, Band], ...] = (
    (0.80, Band.STRONG),
    (0.60, Band.GOOD),
    (0.40, Band.MODERATE),
    (0.00, Band.WEAK),
)

# Above this share of unknown required requirements, no verdict is honest.
UNKNOWN_REQUIRED_CEILING = 0.5


def coverage(matches: Sequence[RequirementMatch]) -> Optional[float]:
    """
    Mean credit over the requirements that could actually be checked.

    None when nothing in this group was scorable — which is a different
    statement from 0.0 and must stay distinguishable from it all the way to the
    band.
    """
    scorable = [m for m in matches if m.scorable]
    if not scorable:
        return None
    total = sum(m.credit or 0.0 for m in scorable)
    return round(total / len(scorable), 4)


def _split(
    matches: Sequence[RequirementMatch],
) -> Tuple[Tuple[RequirementMatch, ...], Tuple[RequirementMatch, ...]]:
    required = tuple(
        m for m in matches if m.requirement.necessity is Necessity.REQUIRED
    )
    preferred = tuple(
        m for m in matches if m.requirement.necessity is Necessity.PREFERRED
    )
    return required, preferred


def compute_score(
    matches: Sequence[RequirementMatch],
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    The overall score, and the two coverages it is built from.

    Returns `(score, required_coverage, preferred_coverage)`. The weights are
    renormalised when one side is absent, so a posting with only required
    requirements scores on those alone rather than being held to 0.8.
    """
    required, preferred = _split(matches)
    required_coverage = coverage(required)
    preferred_coverage = coverage(preferred)

    if required_coverage is None and preferred_coverage is None:
        return 0.0, None, None
    if preferred_coverage is None:
        return round(required_coverage, 4), required_coverage, None
    if required_coverage is None:
        return round(preferred_coverage, 4), None, preferred_coverage

    score = (
        REQUIRED_WEIGHT * required_coverage
        + PREFERRED_WEIGHT * preferred_coverage
    )
    return round(score, 4), required_coverage, preferred_coverage


def compute_band(
    matches: Sequence[RequirementMatch],
    score: float,
    *,
    required_coverage: Optional[float],
    degraded: bool = False,
) -> Tuple[Band, str]:
    """
    The coarse verdict, and the reason when it is not simply the score's band.

    Three overrides, all of which exist to stop the arithmetic saying something
    the evidence does not support.
    """
    required, _ = _split(matches)

    if required_coverage is None and not matches:
        return Band.UNKNOWN, "nothing could be assessed"

    # ── A degraded profile cannot produce a flattering verdict ───────────────
    # Excluding UNKNOWN from the score is right — a requirement nobody could
    # check must not depress it. But excluding the unknowns *deletes the gaps*,
    # so the mean is taken over the survivors, which are disproportionately the
    # things that matched. A live run made this concrete: with the skills
    # lookup failing, a profile that scores 61% healthy came back 93% and
    # STRONG, because every gap had turned into an unknown and dropped out.
    #
    # The score that remains is an upper bound rather than an estimate, and an
    # upper bound presented as a verdict is the flattering direction of a wrong
    # answer — precisely the direction this package must never fail in. So when
    # degradation actually cost us a check, there is no band.
    #
    # Gated on `assessable` because a RESPONSIBILITY row is unknown by nature
    # rather than by outage, and it must not suppress an otherwise sound
    # verdict on a healthy profile.
    if degraded:
        blocked = [
            m for m in matches
            if m.status is MatchStatus.UNKNOWN and m.requirement.assessable
        ]
        if blocked:
            return (
                Band.UNKNOWN,
                f"{len(blocked)} requirement(s) could not be checked because "
                "part of your profile failed to load",
            )

    # Too much of the bar could not be checked. Not a low score — no score.
    if required:
        unknown_required = sum(
            1 for m in required if m.status is MatchStatus.UNKNOWN
        )
        if unknown_required / len(required) > UNKNOWN_REQUIRED_CEILING:
            return (
                Band.UNKNOWN,
                f"{unknown_required} of {len(required)} required items could "
                "not be checked",
            )
    elif required_coverage is None and not any(m.scorable for m in matches):
        return Band.UNKNOWN, "no requirement could be checked"

    band = next(
        (b for threshold, b in BAND_THRESHOLDS if score >= threshold), Band.WEAK
    )

    # A failed bar is a failed bar. Nine matches and one missing must-have is
    # not a strong match, whatever the weighted average works out to.
    missing_required = [m for m in required if m.status is MatchStatus.MISSING]
    if missing_required and band is Band.STRONG:
        names = ", ".join(m.requirement.label() for m in missing_required[:3])
        return Band.GOOD, f"capped: required {names} not evidenced"

    return band, ""


__all__ = [
    "BAND_THRESHOLDS",
    "PREFERRED_WEIGHT",
    "REQUIRED_WEIGHT",
    "UNKNOWN_REQUIRED_CEILING",
    "compute_band",
    "compute_score",
    "coverage",
]
