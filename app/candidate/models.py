"""
What the system may claim about the user professionally, and on what basis.

The job agent's failure mode is not a wrong ranking, it is a confident lie:
telling a recruiter — or telling the user, who then tells a recruiter — that
they know Kubernetes because the word appeared in a job description the model
was reading at the time. Ranking errors are recoverable. A fabricated
qualification on an application is not.

So a skill is not a string here. It is a claim with evidence attached, and the
type system makes an unsupported claim awkward to construct: `SkillClaim`
requires evidence, and `CandidateProfile.has_skill` is defined as "has at least
one piece of evidence" rather than "appears in a list". Nothing downstream gets
to assert a qualification the résumé does not support, because there is no
representation for one.

The second thing this carries is the distinction `app.memory.answerability`
protects, moved up to the profile level. "The résumé has no Kubernetes" and
"the skills lookup failed" both produce an empty skill set, and they license
opposite statements — the first is a gap worth telling the user about, the
second is a reason to say nothing at all. `CandidateProfile.degraded` is what
keeps them apart, and `may_assert_gaps` is the property every consumer must
check before reporting that something is missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple


class EvidenceKind(str, Enum):
    """Where a claim came from. Ordered by how much weight it carries."""

    RESUME_SKILLS = "resume_skills"
    """Named in the résumé's own skills section — the user asserting it directly."""

    RESUME_EXPERIENCE = "resume_experience"
    """Used in a role. The strongest kind: someone paid for it."""

    RESUME_PROJECT = "resume_project"
    """Used in a project the user built."""

    RESUME_EDUCATION = "resume_education"
    """Coursework or degree content. Real, and weaker than having shipped it."""

    RESUME_ACHIEVEMENT = "resume_achievement"
    """Named in an award or certification."""

    PROFILE_FACT = "profile_fact"
    """Stated by the user in conversation and stored deliberately."""

    GITHUB_REPO = "github_repo"
    """A public repository. Not yet wired — reserved so the shape is fixed
    before an integration is built against it."""


# How much a single piece of evidence is worth. Having built something with a
# tool outranks having listed it, which is the ordering a competent reviewer
# uses and the opposite of what a keyword scan produces.
_EVIDENCE_WEIGHT: Dict[EvidenceKind, float] = {
    EvidenceKind.RESUME_EXPERIENCE: 1.0,
    EvidenceKind.RESUME_PROJECT: 0.85,
    EvidenceKind.GITHUB_REPO: 0.85,
    EvidenceKind.RESUME_SKILLS: 0.6,
    EvidenceKind.RESUME_ACHIEVEMENT: 0.5,
    EvidenceKind.PROFILE_FACT: 0.5,
    EvidenceKind.RESUME_EDUCATION: 0.35,
}


@dataclass(frozen=True)
class Evidence:
    """
    One concrete reason to believe a claim.

    `source_id` is a real identifier from the memory store — a résumé chunk's
    `string_id`, a project's `entity_id` — so a claim can always be traced back
    to the text it came from. `excerpt` is the text itself, quoted rather than
    summarised: a paraphrase is where invention creeps in.
    """

    kind: EvidenceKind
    source_id: str
    excerpt: str
    detail: str = ""
    """Human-readable locator: a project title, a company name."""

    @property
    def weight(self) -> float:
        return _EVIDENCE_WEIGHT.get(self.kind, 0.4)

    def describe(self) -> str:
        """One short phrase naming this evidence, for an explanation."""
        if self.detail:
            return f"{self.detail}"
        return {
            EvidenceKind.RESUME_SKILLS: "listed in your skills section",
            EvidenceKind.RESUME_EDUCATION: "your education section",
            EvidenceKind.RESUME_ACHIEVEMENT: "an achievement on your résumé",
            EvidenceKind.PROFILE_FACT: "something you told me",
        }.get(self.kind, self.kind.value.replace("_", " "))

    def summary(self) -> Dict[str, Any]:
        """Log form — provenance only, never the excerpt."""
        return {"kind": self.kind.value, "source": self.source_id[:32]}


@dataclass(frozen=True)
class SkillClaim:
    """
    A skill the user demonstrably has, and why we think so.

    Construction requires evidence. There is deliberately no way to build an
    empty one: a skill with nothing behind it is not a weaker claim, it is a
    different thing entirely, and it belongs in a gap rather than in a profile.
    """

    name: str
    """Canonical name — 'PostgreSQL', not 'postgres' or 'Postgres DB'."""

    category: str = ""
    """The résumé's own grouping, when it had one ('Backend', 'Cloud & DevOps')."""

    aliases: Tuple[str, ...] = ()
    """Surface forms actually seen, kept so an explanation can quote the user."""

    evidence: Tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError(
                f"SkillClaim({self.name!r}) has no evidence. A skill with no "
                "source is a gap, not a claim — see app/candidate/models.py."
            )

    @property
    def strength(self) -> float:
        """
        How well supported this is, 0–1.

        Saturating rather than additive: three projects using Python is much
        stronger than one, and thirty mentions is not ten times stronger than
        three. Listing a skill and never using it stays visibly weak.
        """
        if not self.evidence:
            return 0.0
        best = max(e.weight for e in self.evidence)
        corroboration = min(0.15, 0.05 * (len(self.evidence) - 1))
        return round(min(1.0, best + corroboration), 3)

    @property
    def demonstrated(self) -> bool:
        """Whether it was actually *used*, not merely listed."""
        return any(
            e.kind in (
                EvidenceKind.RESUME_EXPERIENCE,
                EvidenceKind.RESUME_PROJECT,
                EvidenceKind.GITHUB_REPO,
            )
            for e in self.evidence
        )

    def where(self, limit: int = 3) -> List[str]:
        """The places this shows up, strongest first — for explanations."""
        ordered = sorted(self.evidence, key=lambda e: e.weight, reverse=True)
        seen, out = set(), []
        for item in ordered:
            label = item.describe()
            if label not in seen:
                seen.add(label)
                out.append(label)
            if len(out) >= limit:
                break
        return out


@dataclass(frozen=True)
class ProjectEntry:
    """One project, with the skills it evidences."""

    title: str
    entity_id: str
    summary: str
    skills: Tuple[str, ...] = ()
    """Canonical skill names detected in this project's text."""


@dataclass(frozen=True)
class ExperienceEntry:
    """One role. Deliberately thin — dates are unreliable in parsed PDFs."""

    title: str
    organisation: str
    source_id: str
    summary: str
    skills: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EducationEntry:
    institution: str
    qualification: str
    source_id: str
    raw: str
    cgpa: Optional[str] = None


class ProfileSource(str, Enum):
    """The stores a profile is assembled from."""

    NAME = "name"
    SKILLS = "skills"
    PROJECTS = "projects"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    ACHIEVEMENTS = "achievements"


@dataclass(frozen=True)
class CandidateProfile:
    """
    The structured view of one person's professional history.

    Everything in it is derived from stored text and carries a path back to
    that text. Nothing is inferred, generated, or defaulted — a résumé with no
    skills section produces a profile with no skills, and that is the correct
    answer rather than a failure.
    """

    owner_id: str
    name: Optional[str] = None

    skills: Mapping[str, SkillClaim] = field(default_factory=dict)
    """Canonical name → claim. Keyed lowercased for lookup."""

    projects: Tuple[ProjectEntry, ...] = ()
    experience: Tuple[ExperienceEntry, ...] = ()
    education: Tuple[EducationEntry, ...] = ()
    achievements: Tuple[str, ...] = ()

    sources_loaded: Tuple[ProfileSource, ...] = ()
    sources_failed: Tuple[ProfileSource, ...] = ()
    """Sources whose lookup *errored*. Not the same as sources that were empty,
    and the difference decides whether a gap may be reported."""

    built_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Queries ──────────────────────────────────────────────────────────────

    def has_skill(self, name: str) -> bool:
        """Whether there is at least one piece of evidence for this skill."""
        return self.claim(name) is not None

    def claim(self, name: str) -> Optional[SkillClaim]:
        return self.skills.get((name or "").strip().lower())

    def evidence_for(self, name: str) -> Tuple[Evidence, ...]:
        found = self.claim(name)
        return found.evidence if found else ()

    @property
    def degraded(self) -> bool:
        """Whether any source failed to load."""
        return bool(self.sources_failed)

    @property
    def may_assert_gaps(self) -> bool:
        """
        Whether absence can be reported as a gap.

        The property every consumer must consult before saying "you don't have
        X". With a failed source, the honest statement is that we could not
        check — reporting a gap would be asserting something about the user's
        own history on the basis of a network error.
        """
        return not self.degraded

    @property
    def is_empty(self) -> bool:
        return not (self.skills or self.projects or self.experience or self.education)

    def demonstrated_skills(self) -> List[SkillClaim]:
        """Skills actually used in work or projects, strongest first."""
        return sorted(
            (c for c in self.skills.values() if c.demonstrated),
            key=lambda c: c.strength, reverse=True,
        )

    def top_skills(self, limit: int = 12) -> List[SkillClaim]:
        return sorted(
            self.skills.values(), key=lambda c: c.strength, reverse=True
        )[:limit]

    def summary(self) -> Dict[str, Any]:
        """Structured form for logging — counts and provenance, never content."""
        return {
            "owner": self.owner_id,
            "skills": len(self.skills),
            "demonstrated": sum(1 for c in self.skills.values() if c.demonstrated),
            "projects": len(self.projects),
            "experience": len(self.experience),
            "education": len(self.education),
            "loaded": [s.value for s in self.sources_loaded],
            "failed": [s.value for s in self.sources_failed],
            "degraded": self.degraded,
        }


__all__ = [
    "CandidateProfile",
    "EducationEntry",
    "Evidence",
    "EvidenceKind",
    "ExperienceEntry",
    "ProfileSource",
    "ProjectEntry",
    "SkillClaim",
]
