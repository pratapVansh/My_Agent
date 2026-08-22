"""
Checking a posting's requirements against what the candidate can prove.

Pure and deterministic: requirements in, profile in, report out. No model, no
network, no clock. The same two inputs always produce the same report, which is
what makes the score arguable and the tests meaningful.

Every positive verdict is built from evidence that already exists in the
profile, carrying the identifiers ingestion assigned — a résumé chunk's
`string_id`, a project's `entity_id`. Nothing here mints a new provenance
scheme, and nothing here can construct a positive verdict without one, because
`RequirementMatch` refuses to be built that way.

**The rule that governs everything else in this file.** Finding no evidence is
not the same as finding evidence of absence, and the difference is decided by
`CandidateProfile.may_assert_gaps` rather than by how the caller feels:

    profile healthy, résumé simply lacks it  → MISSING  ("your résumé shows no X")
    a source failed to load                  → UNKNOWN  ("I couldn't check")
    no profile stored at all                 → UNKNOWN  ("nothing on file to check")

That third case is easy to miss and matters as much as the second. A user who
has not uploaded a résumé produces an empty profile through a *healthy* lookup,
so `may_assert_gaps` is True — and reporting every requirement as MISSING would
tell them they lack skills the system simply never looked at. Emptiness is
checked before absence for that reason.

**Two kinds are deliberately capped below MATCHED.**

`EXPERIENCE_YEARS` never reaches MATCHED. `CandidateProfileBuilder` says
outright that dates in parsed PDFs are unreliable, so no duration is stored;
answering "yes, 5 years" would be inventing the one number the requirement is
about. Having roles on file is real evidence and it is evidence of *having
worked*, not of having worked long enough — so it resolves to PARTIAL with a
rationale that says exactly that.

`DOMAIN` never reaches MATCHED either. A project description containing the
word "production" is genuine, quotable evidence and it is weaker than the
résumé naming a skill outright, so it can support a partial claim and not a
full one.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from app.candidate.models import (
    CandidateProfile,
    Evidence,
    EvidenceKind,
)
from app.matching.extract import DOMAIN_TERMS, degree_level_of
from app.matching.models import (
    Band,
    DEGREE_LEVELS,
    JobRequirements,
    MatchReport,
    MatchStatus,
    Necessity,
    Requirement,
    RequirementKind,
    RequirementMatch,
)
from app.matching.scoring import compute_band, compute_score

logger = logging.getLogger(__name__)

_MAX_EVIDENCE_PER_MATCH = 3
"""Evidence kept per requirement. The strongest few justify a claim; the rest
only inflate the payload the model has to read."""

# Sort order for the report, so a truncated render always shows the load-bearing
# rows first and two runs never disagree about ordering.
_NECESSITY_RANK = {Necessity.REQUIRED: 0, Necessity.PREFERRED: 1}
_STATUS_RANK = {
    MatchStatus.MATCHED: 0,
    MatchStatus.PARTIAL: 1,
    MatchStatus.MISSING: 2,
    MatchStatus.UNKNOWN: 3,
}


def _top(evidence: Sequence[Evidence]) -> Tuple[Evidence, ...]:
    """The strongest few pieces of evidence, deterministically ordered."""
    ordered = sorted(
        evidence,
        key=lambda e: (-e.weight, e.kind.value, e.source_id),
    )
    return tuple(ordered[:_MAX_EVIDENCE_PER_MATCH])


def _gap(
    requirement: Requirement, profile: CandidateProfile, *, what: str
) -> RequirementMatch:
    """
    The verdict when nothing was found.

    The single place MISSING is allowed to be produced, so the entitlement
    check cannot be forgotten at one call site and honoured at the others.
    """
    if profile.is_empty:
        return RequirementMatch(
            requirement=requirement,
            status=MatchStatus.UNKNOWN,
            rationale=(
                "there is no profile on file to check this against - "
                "nothing has been ruled out"
            ),
        )
    if not profile.may_assert_gaps:
        failed = ", ".join(s.value for s in profile.sources_failed)
        return RequirementMatch(
            requirement=requirement,
            status=MatchStatus.UNKNOWN,
            rationale=(
                f"could not check: the {failed} lookup failed, so absence is "
                "not established"
            ),
        )
    return RequirementMatch(
        requirement=requirement,
        status=MatchStatus.MISSING,
        rationale=f"no {what} in your profile evidences this",
    )


# ── Per-kind checks ──────────────────────────────────────────────────────────

def _match_skill(
    requirement: Requirement, profile: CandidateProfile
) -> RequirementMatch:
    """
    A named skill, against the evidence-backed claim table.

    `CandidateProfile.claim` is keyed by canonical name and the extractor emits
    canonical names, so alias resolution has already happened on both sides —
    a posting's "ReactJS" and a résumé's "React.js" meet here as `React`.
    """
    claim = profile.claim(requirement.canonical)
    if claim is None:
        return _gap(requirement, profile, what="skill, project or role")

    if claim.demonstrated:
        return RequirementMatch(
            requirement=requirement,
            status=MatchStatus.MATCHED,
            evidence=_top(claim.evidence),
            rationale=f"used in {', '.join(claim.where(limit=2))}",
        )

    # Listed, or evidenced only by coursework or an award. Real, and weaker
    # than having shipped it — the distinction a reviewer actually makes.
    return RequirementMatch(
        requirement=requirement,
        status=MatchStatus.PARTIAL,
        evidence=_top(claim.evidence),
        rationale=(
            f"named in your profile ({', '.join(claim.where(limit=2))}) but "
            "not evidenced in a project or role"
        ),
    )


def _match_experience_years(
    requirement: Requirement, profile: CandidateProfile
) -> RequirementMatch:
    """
    A minimum duration. Capped at PARTIAL — see the module docstring.
    """
    if not profile.experience:
        return _gap(requirement, profile, what="role")

    evidence = _top(tuple(
        Evidence(
            kind=EvidenceKind.RESUME_EXPERIENCE,
            source_id=entry.source_id,
            excerpt=entry.summary,
            detail=(entry.title or entry.organisation or "a role"),
        )
        for entry in profile.experience
    ))
    count = len(profile.experience)
    return RequirementMatch(
        requirement=requirement,
        status=MatchStatus.PARTIAL,
        evidence=evidence,
        rationale=(
            f"{count} role{'s' if count != 1 else ''} on file, but your resume "
            f"does not record durations - I cannot confirm "
            f"{requirement.min_years}+ years either way"
        ),
    )


def _match_education(
    requirement: Requirement, profile: CandidateProfile
) -> RequirementMatch:
    """A degree level, compared by level rather than by string."""
    if not profile.education:
        return _gap(requirement, profile, what="education entry")

    wanted = requirement.degree_level or DEGREE_LEVELS["bachelor"]

    best_level: Optional[int] = None
    best_entry = None
    for entry in profile.education:
        level = degree_level_of(f"{entry.qualification} {entry.raw}")
        if level is not None and (best_level is None or level > best_level):
            best_level, best_entry = level, entry

    if best_level is None or best_entry is None:
        # Education exists but no recognisable degree was parsed out of it.
        # That is a parsing limit, not a candidate shortfall.
        return RequirementMatch(
            requirement=requirement,
            status=MatchStatus.UNKNOWN,
            rationale=(
                "you have education on file but I could not read a degree "
                "level from it"
            ),
        )

    evidence = (Evidence(
        kind=EvidenceKind.RESUME_EDUCATION,
        source_id=best_entry.source_id,
        excerpt=best_entry.raw,
        detail=(best_entry.qualification or best_entry.institution
                or "your education"),
    ),)

    field_note = ""
    if requirement.field_of_study:
        haystack = f"{best_entry.qualification} {best_entry.raw}".lower()
        if requirement.field_of_study.lower() in haystack:
            field_note = f", in {requirement.field_of_study}"
        else:
            field_note = (
                f" (the posting asks for {requirement.field_of_study}; "
                "your resume names a different field)"
            )

    if best_level >= wanted:
        return RequirementMatch(
            requirement=requirement,
            status=MatchStatus.MATCHED,
            evidence=evidence,
            rationale=(
                f"{best_entry.qualification or 'your degree'} meets the "
                f"required level{field_note}"
            ),
        )

    return RequirementMatch(
        requirement=requirement,
        status=MatchStatus.PARTIAL,
        evidence=evidence,
        rationale=(
            f"{best_entry.qualification or 'your degree'} is below the "
            f"level asked for{field_note}"
        ),
    )


def _match_domain(
    requirement: Requirement, profile: CandidateProfile
) -> RequirementMatch:
    """
    A domain phrase, by literal search over the candidate's own text.

    Capped at PARTIAL. The evidence is a real quote from a real project or
    role, and a word appearing in a description is a weaker thing than the
    résumé naming a skill — so it supports a partial claim and never a full one.
    """
    surfaces = DOMAIN_TERMS.get(requirement.canonical, ())
    if not surfaces:
        return RequirementMatch(
            requirement=requirement,
            status=MatchStatus.UNKNOWN,
            rationale="no deterministic check exists for this requirement",
        )

    hits: List[Evidence] = []

    for project in profile.projects:
        haystack = f"{project.title} {project.summary}".lower()
        if any(surface in haystack for surface in surfaces):
            hits.append(Evidence(
                kind=EvidenceKind.RESUME_PROJECT,
                source_id=project.entity_id,
                excerpt=project.summary,
                detail=project.title or "a project",
            ))

    for role in profile.experience:
        haystack = f"{role.title} {role.organisation} {role.summary}".lower()
        if any(surface in haystack for surface in surfaces):
            hits.append(Evidence(
                kind=EvidenceKind.RESUME_EXPERIENCE,
                source_id=role.source_id,
                excerpt=role.summary,
                detail=(role.title or role.organisation or "a role"),
            ))

    if not hits:
        return _gap(requirement, profile, what="project or role")

    return RequirementMatch(
        requirement=requirement,
        status=MatchStatus.PARTIAL,
        evidence=_top(tuple(hits)),
        rationale=(
            f"described in {hits[0].describe()} - supporting evidence, not a "
            "stated qualification"
        ),
    )


def _match_responsibility(requirement: Requirement) -> RequirementMatch:
    """
    A duty no deterministic check covers.

    Reported rather than dropped, and excluded from the score rather than
    guessed at. "I did not assess this" is a useful sentence; a fabricated
    verdict is not.
    """
    return RequirementMatch(
        requirement=requirement,
        status=MatchStatus.UNKNOWN,
        rationale="not something I can verify from your resume",
    )


_CHECKS = {
    RequirementKind.SKILL: _match_skill,
    RequirementKind.EXPERIENCE_YEARS: _match_experience_years,
    RequirementKind.EDUCATION: _match_education,
    RequirementKind.DOMAIN: _match_domain,
}


# ── Entry point ──────────────────────────────────────────────────────────────

def match_requirements(
    requirements: JobRequirements, profile: CandidateProfile
) -> MatchReport:
    """
    Check every requirement and return a scored, self-contained report.

    The score is computed here rather than by any consumer, so the number that
    reaches a prompt is already final — see `app.matching.scoring`.
    """
    matches: List[RequirementMatch] = []

    for requirement in requirements:
        check = _CHECKS.get(requirement.kind)
        if check is None:
            matches.append(_match_responsibility(requirement))
            continue
        try:
            matches.append(check(requirement, profile))
        except ValueError:
            # `RequirementMatch` refused the verdict — a positive claim without
            # evidence, or an absence carrying some. That is a bug in a check,
            # and the safe resolution is to say nothing about this requirement
            # rather than to let a malformed claim through.
            logger.error(
                "Refusing an incoherent verdict for %s; reporting it as "
                "unchecked", requirement.summary(), exc_info=True,
            )
            matches.append(RequirementMatch(
                requirement=requirement,
                status=MatchStatus.UNKNOWN,
                rationale="this requirement could not be assessed",
            ))

    matches.sort(key=lambda m: (
        _NECESSITY_RANK[m.requirement.necessity],
        _STATUS_RANK[m.status],
        m.requirement.canonical.lower(),
    ))

    score, required_coverage, preferred_coverage = compute_score(matches)
    band, reason = compute_band(
        matches, score,
        required_coverage=required_coverage,
        degraded=profile.degraded,
    )

    report = MatchReport(
        matches=tuple(matches),
        score=score,
        band=band,
        required_coverage=required_coverage,
        preferred_coverage=preferred_coverage,
        title=requirements.title,
        profile_degraded=profile.degraded,
        profile_empty=profile.is_empty,
        unscorable_reason=reason,
    )
    logger.info("Match report: %s", report.summary())
    return report


__all__ = ["match_requirements"]
