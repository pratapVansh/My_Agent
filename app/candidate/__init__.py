"""
The user's professional profile, structured and evidenced.

Built from memory that already exists — résumé sections, skills, projects,
experience — and never from generation. The invariant the whole package is
arranged around is that a claim about the user must be traceable to text the
user supplied: `SkillClaim` cannot be constructed without evidence, and the
vocabulary is a closed set, so the system can only assert things it has names
for and only when it finds them in the résumé.

Consumers must also respect `CandidateProfile.may_assert_gaps` before reporting
that a skill is missing. A failed lookup and an empty résumé produce the same
empty skill set and license opposite statements.
"""
from app.candidate.builder import (
    CandidateProfileBuilder,
    build_candidate_profile,
    candidate_profile_builder,
)
from app.candidate.models import (
    CandidateProfile,
    EducationEntry,
    Evidence,
    EvidenceKind,
    ExperienceEntry,
    ProfileSource,
    ProjectEntry,
    SkillClaim,
)

__all__ = [
    "CandidateProfile",
    "CandidateProfileBuilder",
    "EducationEntry",
    "Evidence",
    "EvidenceKind",
    "ExperienceEntry",
    "ProfileSource",
    "ProjectEntry",
    "SkillClaim",
    "build_candidate_profile",
    "candidate_profile_builder",
]
