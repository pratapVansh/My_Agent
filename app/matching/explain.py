"""
Turning the match table into sentences, without letting a model near the claims.

The explanation is rendered here, from the report, deterministically. That is
the whole mechanism by which "never invent a qualification" survives contact
with a language model: the model is handed finished prose describing what the
matcher found, and its remaining job is to present it. It is never asked
whether the candidate knows Kubernetes, so it never gets to answer.

Two distinctions the wording is built around, because collapsing either is how
an explanation becomes a lie:

**"Not found" is not "not demonstrated".** A skill listed on the résumé but
never used in a project is `PARTIAL` and reads as *"named in your profile but
not evidenced in a project or role"*. A skill nowhere in the résumé is
`MISSING` and reads as *"no skill, project or role evidences this"*. Those are
different facts about the candidate and a recruiter would treat them
differently.

**"Missing" is not "unchecked".** Requirements that could not be assessed —
because a lookup failed, because nothing is on file, or because no
deterministic check exists — are printed in their own section under "not
checked", never folded into the gaps. When the profile is degraded the report
opens with a banner saying so, because in that state every gap in it is
provisional.

Every positive row carries the stored identifier its evidence came from, so any
sentence here can be traced back to a resume chunk or a project. That is not
decoration: it is the difference between an explanation and an assertion.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from app.matching.models import (
    Band,
    MatchReport,
    MatchStatus,
    Necessity,
    RequirementMatch,
)

_BAND_PHRASE: Dict[Band, str] = {
    Band.STRONG: "Strong match",
    Band.GOOD: "Good match",
    Band.MODERATE: "Moderate match",
    Band.WEAK: "Weak match",
    Band.UNKNOWN: "Not enough information to judge the match",
}

_DEGRADED_BANNER = (
    "! Part of your profile could not be loaded, so nothing below is a "
    "statement that you lack a skill - only that I could not find it."
)

_EMPTY_BANNER = (
    "! There is no resume or profile data on file for you, so I could not "
    "check any requirement. This is not a finding about your background."
)

_SECTION_TITLES: Dict[MatchStatus, str] = {
    MatchStatus.MATCHED: "Evidenced",
    MatchStatus.PARTIAL: "Partial evidence",
    MatchStatus.MISSING: "No evidence found",
    MatchStatus.UNKNOWN: "Not checked",
}

# Render order. Strengths first, then the honest caveats.
_SECTION_ORDER: Tuple[MatchStatus, ...] = (
    MatchStatus.MATCHED,
    MatchStatus.PARTIAL,
    MatchStatus.MISSING,
    MatchStatus.UNKNOWN,
)

DEFAULT_LIMIT = 2400
"""Character ceiling for the rendered block.

Truncation is row-granular, lowest-priority first, and reports how many rows it
dropped. Callers feeding `ToolResult.observation` pass a smaller budget so the
whole payload survives *that* truncation intact — a cut there would land in the
middle of a JSON string and could sever a source id, producing an identifier
that looks real and is not."""

_MAX_RATIONALE_CHARS = 160

_MAX_SOURCE_ID_CHARS = 96
"""Display cap for a stored id.

Deliberately far above anything ingestion produces. It was 32, which is shorter
than a real id — `skills_resume_vansh_1a36dcef_chunk_1` rendered as
`skills_resume_vansh_1a36dcef_chu`, an identifier that looks real, cannot be
looked up, and is indistinguishable from a genuine one to a reader. A live run
caught it. The cap survives only so a malformed row cannot paste an unbounded
string into a spoken reply; it must never be reached by a well-formed id."""


def _row(match: RequirementMatch) -> str:
    """One requirement, its verdict, and where the verdict came from."""
    label = match.requirement.label()
    necessity = "required" if (
        match.requirement.necessity is Necessity.REQUIRED
    ) else "preferred"

    parts = [f"  - {label} [{necessity}]"]
    if match.rationale:
        parts.append(f": {match.rationale[:_MAX_RATIONALE_CHARS]}")

    # The provenance. Present for every positive row by construction, since
    # `RequirementMatch` cannot be built positive without it.
    if match.evidence:
        ids = ", ".join(
            sorted({e.source_id[:_MAX_SOURCE_ID_CHARS] for e in match.evidence})
        )
        parts.append(f" (source: {ids})")
    return "".join(parts)


def _headline(report: MatchReport) -> str:
    phrase = _BAND_PHRASE.get(report.band, "Match")
    if report.band is Band.UNKNOWN:
        return phrase + (
            f" - {report.unscorable_reason}" if report.unscorable_reason else ""
        )
    title = f" for {report.title}" if report.title else ""
    return f"{phrase}{title}: {report.score:.0%}"


def _coverage_line(report: MatchReport) -> str:
    bits: List[str] = []
    required = report.of_necessity(Necessity.REQUIRED)
    preferred = report.of_necessity(Necessity.PREFERRED)

    if report.required_coverage is not None:
        scorable = sum(1 for m in required if m.scorable)
        bits.append(
            f"required {report.required_coverage:.0%} "
            f"({scorable} of {len(required)} checkable)"
        )
    if report.preferred_coverage is not None:
        scorable = sum(1 for m in preferred if m.scorable)
        bits.append(
            f"preferred {report.preferred_coverage:.0%} "
            f"({scorable} of {len(preferred)} checkable)"
        )
    return " | ".join(bits)


def render(report: MatchReport, *, limit: int = DEFAULT_LIMIT) -> str:
    """
    The full explanation, composed from the evidence table.

    Deterministic: the same report always renders the same text, which is what
    lets a test assert on it and a user re-run it.
    """
    header: List[str] = [_headline(report)]

    if report.profile_empty:
        header.append(_EMPTY_BANNER)
    elif report.profile_degraded:
        header.append(_DEGRADED_BANNER)

    coverage = _coverage_line(report)
    if coverage:
        header.append(coverage)

    if report.unscorable_reason and report.band is not Band.UNKNOWN:
        header.append(f"({report.unscorable_reason})")

    # Pinned conclusions. Never dropped by the fitter: a truncated report that
    # loses "required X is not evidenced" reads as a better match than it is,
    # which is the one direction the truncation must never fail in.
    for line in _pinned(report):
        header.append(line)

    if not report.matches:
        header.append(
            "No requirements could be read from this job description."
        )
        return "\n".join(header)

    # Rows grouped by verdict. Required is dropped after preferred within a
    # section, and the whole section header disappears with its last row.
    sections: List[Tuple[str, List[Tuple[int, str]]]] = []
    for priority, status in enumerate(_SECTION_ORDER):
        rows = report.of_status(status)
        if not rows:
            continue
        sections.append((
            f"{_SECTION_TITLES[status]}:",
            [
                (
                    priority * 2 + (
                        0 if m.requirement.necessity is Necessity.REQUIRED
                        else 1
                    ),
                    _row(m),
                )
                for m in rows
            ],
        ))

    return _fit(header, sections, limit)


def _pinned(report: MatchReport) -> List[str]:
    """Conclusions that survive any truncation."""
    lines: List[str] = []

    strongest = report.strongest_evidence
    if strongest is not None:
        lines.append(
            f"Strongest evidence: {strongest.describe()} "
            f"(source: {strongest.source_id[:_MAX_SOURCE_ID_CHARS]})"
        )

    if report.missing_required and report.may_assert_gaps:
        names = ", ".join(
            m.requirement.label() for m in report.missing_required[:4]
        )
        lines.append(
            f"Required but not evidenced: {names}. I have not claimed otherwise."
        )
    return lines


def _fit(
    header: Sequence[str],
    sections: Sequence[Tuple[str, List[Tuple[int, str]]]],
    limit: int,
) -> str:
    """
    Assemble within the budget, dropping the least important rows first.

    Row-granular rather than character-granular, because a cut in the middle of
    a source id produces an identifier that looks real and is not — precisely
    the kind of thing this package exists to avoid emitting. A section whose
    last row is dropped loses its heading too, so the output never advertises
    an empty category.
    """
    remaining = [(title, list(rows)) for title, rows in sections]
    dropped = 0

    while True:
        lines = list(header)
        for title, rows in remaining:
            if not rows:
                continue
            lines.append(title)
            lines.extend(row for _, row in rows)
        if dropped:
            lines.append(f"(+{dropped} further requirement(s) not shown)")

        text = "\n".join(lines)
        if len(text) <= limit or not any(rows for _, rows in remaining):
            return text

        worst = max(
            rank for _, rows in remaining for rank, _ in rows
        )
        for _, rows in reversed(remaining):
            for index in range(len(rows) - 1, -1, -1):
                if rows[index][0] == worst:
                    del rows[index]
                    dropped += 1
                    break
            else:
                continue
            break


def render_summary(report: MatchReport) -> str:
    """
    One sentence, for a spoken reply or a compact card.

    Deliberately never says "you are a good fit" — it reports coverage, which
    is a fact about the evidence, rather than a judgement the system has no
    standing to make.
    """
    if report.profile_empty:
        return (
            "I have no resume or profile data on file, so I couldn't check "
            "this posting against your background."
        )
    if report.band is Band.UNKNOWN:
        reason = report.unscorable_reason or "too little could be checked"
        return f"I can't judge this match - {reason}."

    matched = len(report.matched)
    partial = len(report.partial)
    total = sum(1 for m in report.matches if m.scorable)
    phrase = _BAND_PHRASE.get(report.band, "Match")

    sentence = (
        f"{phrase}: {report.score:.0%}. "
        f"{matched} of {total} checkable requirements are evidenced in your "
        f"profile"
    )
    if partial:
        sentence += f", {partial} partially"
    sentence += "."

    if report.missing_required and report.may_assert_gaps:
        names = ", ".join(
            m.requirement.label() for m in report.missing_required[:3]
        )
        sentence += f" Not evidenced, and required: {names}."
    elif report.profile_degraded:
        sentence += " Some of your profile could not be loaded, so gaps here are not confirmed."
    return sentence


__all__ = ["DEFAULT_LIMIT", "render", "render_summary"]
