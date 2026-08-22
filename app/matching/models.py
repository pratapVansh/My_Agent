"""
What a job asks for, what the candidate can prove, and the gap between them.

`app.candidate` established one invariant: a skill the system is willing to
assert must carry evidence, and `SkillClaim` cannot be constructed without it.
This package extends that invariant across the join — because the join is where
a fabricated qualification would actually be spoken aloud.

The failure mode is specific. A model handed a job description and a résumé will
produce a fluent paragraph explaining why the candidate is a great fit, and the
paragraph will contain skills from the *posting* rather than from the résumé.
That is not a ranking error. It is the assistant telling a recruiter something
untrue about a real person, on the strength of having read the requirement.

So the same structural refusal applies here:

    RequirementMatch(status=MATCHED | PARTIAL) requires evidence
    RequirementMatch(status=MISSING)           forbids it

There is no way to represent "matched, because I felt like it". A requirement
with nothing behind it is a `MISSING` or an `UNKNOWN`, and both render as such.

The four statuses are the point of the whole design, and the split between the
last two is the one that matters:

    MATCHED   evidence exists and is strong — the skill was used, not just listed
    PARTIAL   evidence exists and is weak — listed, or coursework, or a phrase hit
    MISSING   no evidence, and we are entitled to say so
    UNKNOWN   no evidence, and we are NOT entitled to say so

`MISSING` and `UNKNOWN` both mean "found nothing". They license opposite
sentences — "your résumé doesn't show Kubernetes" versus "I couldn't check your
skills just now" — and collapsing them is exactly the error
`app.memory.answerability` and `CandidateProfile.may_assert_gaps` exist to
prevent. The matcher may only emit `MISSING` when the profile source that would
have answered actually loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.candidate.models import Evidence


# ── What a posting asks for ──────────────────────────────────────────────────

class Necessity(str, Enum):
    """Whether a requirement is a bar or a wish."""

    REQUIRED = "required"
    """Stated as a must. Failing one materially changes the answer."""

    PREFERRED = "preferred"
    """Nice to have. Contributes, but never decides."""

    @property
    def weight(self) -> float:
        """Relative importance in the score. See `app.matching.scoring`."""
        return 0.8 if self is Necessity.REQUIRED else 0.2


class RequirementKind(str, Enum):
    """What sort of thing is being asked for, which decides how it is checked."""

    SKILL = "skill"
    """A named technology from the canonical vocabulary. Checked against
    `CandidateProfile.skills`, which is evidence-backed by construction."""

    EXPERIENCE_YEARS = "experience_years"
    """A minimum duration. Checked against roles — and deliberately never
    resolved to MATCHED, because résumé date parsing is unreliable enough that
    asserting "yes, 3 years" would be inventing a number."""

    EDUCATION = "education"
    """A degree level, optionally in a field. Checked against parsed education
    entries by level ordering."""

    DOMAIN = "domain"
    """A closed-vocabulary domain phrase ("production", "distributed systems").
    Checked by literal phrase search over project and role text, and capped at
    PARTIAL — a word appearing in a project description is real evidence and
    weaker than a skill the résumé names."""

    RESPONSIBILITY = "responsibility"
    """A duty the posting describes that no deterministic check covers. Kept so
    the report is honest about what it did *not* assess, and excluded from the
    score rather than guessed at."""


# Degree levels, ordered so "master's or above" is a comparison.
DEGREE_LEVELS: Dict[str, int] = {
    "diploma": 1,
    "bachelor": 2,
    "master": 3,
    "phd": 4,
}


@dataclass(frozen=True)
class Requirement:
    """
    One extracted thing a posting asks for.

    `canonical` is the vocabulary's name for a SKILL requirement, so a posting
    saying "ReactJS" and a résumé saying "React.js" meet at `React`. For other
    kinds it is a normalised key used for de-duplication.
    """

    kind: RequirementKind
    canonical: str
    necessity: Necessity
    surface: str = ""
    """The exact words the posting used, quoted rather than paraphrased."""

    source_line: str = ""
    """The line it was extracted from, so an explanation can show its working."""

    min_years: Optional[int] = None
    """For EXPERIENCE_YEARS."""

    degree_level: Optional[int] = None
    """For EDUCATION. See `DEGREE_LEVELS`."""

    field_of_study: str = ""
    """For EDUCATION, when the posting named one."""

    @property
    def key(self) -> str:
        """De-duplication key. Two spellings of one requirement collapse."""
        return f"{self.kind.value}:{self.canonical.strip().lower()}"

    @property
    def assessable(self) -> bool:
        """Whether any deterministic check exists for this kind."""
        return self.kind is not RequirementKind.RESPONSIBILITY

    def label(self) -> str:
        """How this reads in an explanation."""
        if self.kind is RequirementKind.EXPERIENCE_YEARS:
            return f"{self.min_years}+ years of experience"
        if self.kind is RequirementKind.EDUCATION:
            level = next(
                (name for name, value in DEGREE_LEVELS.items()
                 if value == self.degree_level),
                "degree",
            )
            field_text = f" in {self.field_of_study}" if self.field_of_study else ""
            return f"{level.capitalize()}'s degree{field_text}"
        return self.canonical

    def summary(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.canonical,
            "necessity": self.necessity.value,
        }


@dataclass(frozen=True)
class JobRequirements:
    """Everything one posting asks for, plus what could not be parsed."""

    requirements: Tuple[Requirement, ...] = ()
    title: str = ""
    source: str = ""
    """Where the posting text came from — a url, or 'pasted'."""

    text_chars: int = 0

    def __iter__(self):
        return iter(self.requirements)

    def __len__(self) -> int:
        return len(self.requirements)

    def __bool__(self) -> bool:
        return bool(self.requirements)

    def of(self, necessity: Necessity) -> Tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if r.necessity is necessity)

    @property
    def required(self) -> Tuple[Requirement, ...]:
        return self.of(Necessity.REQUIRED)

    @property
    def preferred(self) -> Tuple[Requirement, ...]:
        return self.of(Necessity.PREFERRED)

    def skills(self) -> Tuple[Requirement, ...]:
        return tuple(
            r for r in self.requirements if r.kind is RequirementKind.SKILL
        )

    def summary(self) -> Dict[str, Any]:
        """Log form — counts and names, never the posting body."""
        return {
            "title": self.title[:80],
            "total": len(self.requirements),
            "required": len(self.required),
            "preferred": len(self.preferred),
            "kinds": sorted({r.kind.value for r in self.requirements}),
        }


# ── What the candidate can prove ─────────────────────────────────────────────

class MatchStatus(str, Enum):
    """
    The verdict on one requirement.

    Four states, not three. See the module docstring for why MISSING and
    UNKNOWN must not be collapsed.
    """

    MATCHED = "matched"
    PARTIAL = "partial"
    MISSING = "missing"
    UNKNOWN = "unknown"

    @property
    def is_positive(self) -> bool:
        """Whether this asserts something in the candidate's favour."""
        return self in (MatchStatus.MATCHED, MatchStatus.PARTIAL)

    @property
    def asserts_absence(self) -> bool:
        """Whether this claims the candidate lacks something."""
        return self is MatchStatus.MISSING


# What each status is worth when scoring. UNKNOWN is absent on purpose: it is
# excluded from the calculation entirely rather than scored as a zero, because
# scoring it as zero would let a Qdrant outage look like an unqualified
# candidate.
STATUS_CREDIT: Dict[MatchStatus, float] = {
    MatchStatus.MATCHED: 1.0,
    MatchStatus.PARTIAL: 0.5,
    MatchStatus.MISSING: 0.0,
}


@dataclass(frozen=True)
class RequirementMatch:
    """
    One requirement, checked.

    The constructor is the anti-invention guarantee, and it is deliberately
    unforgiving: a positive verdict without evidence, or a claimed absence
    carrying evidence, are both incoherent states and neither can be built.
    """

    requirement: Requirement
    status: MatchStatus
    evidence: Tuple[Evidence, ...] = ()
    rationale: str = ""
    """Why this verdict, in one phrase, derived from the evidence rather than
    written by a model."""

    def __post_init__(self) -> None:
        if self.status.is_positive and not self.evidence:
            raise ValueError(
                f"RequirementMatch({self.requirement.canonical!r}) is "
                f"{self.status.value} with no evidence. A positive claim "
                "without a source is exactly what this package exists to make "
                "unrepresentable — see app/matching/models.py."
            )
        if self.status is MatchStatus.MISSING and self.evidence:
            raise ValueError(
                f"RequirementMatch({self.requirement.canonical!r}) claims the "
                "candidate lacks something while carrying evidence for it."
            )

    @property
    def credit(self) -> Optional[float]:
        """Score contribution, or None when this requirement is unscorable."""
        return STATUS_CREDIT.get(self.status)

    @property
    def scorable(self) -> bool:
        return self.status in STATUS_CREDIT

    @property
    def strongest(self) -> Optional[Evidence]:
        """The single best reason to believe this. Drives the explanation."""
        if not self.evidence:
            return None
        return max(self.evidence, key=lambda e: e.weight)

    def where(self, limit: int = 2) -> List[str]:
        """The places this requirement is evidenced, strongest first."""
        ordered = sorted(self.evidence, key=lambda e: e.weight, reverse=True)
        seen: set = set()
        out: List[str] = []
        for item in ordered:
            label = item.describe()
            if label not in seen:
                seen.add(label)
                out.append(label)
            if len(out) >= limit:
                break
        return out

    def summary(self) -> Dict[str, Any]:
        """Log form — verdict and provenance ids, never excerpts."""
        return {
            "requirement": self.requirement.canonical,
            "necessity": self.requirement.necessity.value,
            "status": self.status.value,
            "evidence": [e.source_id[:24] for e in self.evidence],
        }


class Band(str, Enum):
    """A coarse verdict, for a sentence a person can act on."""

    STRONG = "strong"
    GOOD = "good"
    MODERATE = "moderate"
    WEAK = "weak"
    UNKNOWN = "unknown"
    """Too little could be checked to say anything. Not a low score — no score."""


@dataclass(frozen=True)
class MatchReport:
    """
    The complete, self-contained answer to "how well do I match this?".

    Carries its own score. The number is computed in `app.matching.scoring`
    before the report is built, so no consumer — and in particular no language
    model — is in a position to produce a different one.
    """

    matches: Tuple[RequirementMatch, ...] = ()
    score: float = 0.0
    band: Band = Band.UNKNOWN
    required_coverage: Optional[float] = None
    preferred_coverage: Optional[float] = None

    title: str = ""
    profile_degraded: bool = False
    """The profile had a failed source. Absence may not be reported as a gap."""

    profile_empty: bool = False
    unscorable_reason: str = ""

    def of_status(self, status: MatchStatus) -> Tuple[RequirementMatch, ...]:
        return tuple(m for m in self.matches if m.status is status)

    def of_necessity(self, necessity: Necessity) -> Tuple[RequirementMatch, ...]:
        return tuple(
            m for m in self.matches if m.requirement.necessity is necessity
        )

    @property
    def matched(self) -> Tuple[RequirementMatch, ...]:
        return self.of_status(MatchStatus.MATCHED)

    @property
    def partial(self) -> Tuple[RequirementMatch, ...]:
        return self.of_status(MatchStatus.PARTIAL)

    @property
    def missing(self) -> Tuple[RequirementMatch, ...]:
        return self.of_status(MatchStatus.MISSING)

    @property
    def unknown(self) -> Tuple[RequirementMatch, ...]:
        return self.of_status(MatchStatus.UNKNOWN)

    @property
    def missing_required(self) -> Tuple[RequirementMatch, ...]:
        """The requirements that actually cost the candidate the job."""
        return tuple(
            m for m in self.missing
            if m.requirement.necessity is Necessity.REQUIRED
        )

    @property
    def may_assert_gaps(self) -> bool:
        """Whether this report is entitled to name anything as missing."""
        return not self.profile_degraded

    @property
    def strongest_evidence(self) -> Optional[Evidence]:
        """The single best thing the candidate has going for them."""
        positives = [m for m in self.matches if m.status.is_positive]
        if not positives:
            return None
        best = max(
            positives,
            key=lambda m: (
                m.requirement.necessity is Necessity.REQUIRED,
                m.strongest.weight if m.strongest else 0.0,
            ),
        )
        return best.strongest

    def evidence_ids(self) -> Tuple[str, ...]:
        """Every stored id this report leans on, de-duplicated, in order."""
        seen: set = set()
        out: List[str] = []
        for match in self.matches:
            for item in match.evidence:
                if item.source_id not in seen:
                    seen.add(item.source_id)
                    out.append(item.source_id)
        return tuple(out)

    def summary(self) -> Dict[str, Any]:
        """Log form — verdicts and counts, never résumé or posting content."""
        return {
            "title": self.title[:60],
            "score": self.score,
            "band": self.band.value,
            "matched": len(self.matched),
            "partial": len(self.partial),
            "missing": len(self.missing),
            "unknown": len(self.unknown),
            "missing_required": len(self.missing_required),
            "degraded": self.profile_degraded,
        }


__all__ = [
    "Band",
    "DEGREE_LEVELS",
    "JobRequirements",
    "MatchReport",
    "MatchStatus",
    "Necessity",
    "Requirement",
    "RequirementKind",
    "RequirementMatch",
    "STATUS_CREDIT",
]
