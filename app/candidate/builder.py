"""
Assembling a `CandidateProfile` out of memory that already exists.

Nothing here stores anything or re-parses a PDF. Résumé ingestion already
produces typed, document-ordered chunks carrying stable identifiers
(`string_id`, `entity_id`, `parent_id`), and that work is reused rather than
repeated — this module reads those chunks and turns them into claims with the
identifiers attached.

The interesting decisions are about what *not* to do.

**Skills come from two places and mean different things.** The skills section
is the user asserting a list. A project or a role that mentions a tool is the
user having *used* it. Both become evidence, with different weights, and a
skill supported only by the list stays visibly weaker than one that shipped —
which is the distinction a reviewer actually cares about and the one a keyword
scan destroys.

**A skill mentioned in prose is only claimed if it is already known.** The
vocabulary is a closed set, so an unrecognised token is dropped rather than
promoted to a qualification. That is the whole anti-invention property: the
system can only ever claim things it has names for, and it only claims those
when it finds them in the user's own text.

**A failed lookup is not an empty section.** Each source is loaded
independently and its `RetrievalResult` status is recorded. One broken
collection degrades the profile rather than silently shrinking it, and
`may_assert_gaps` then refuses to let anything report a missing skill — because
after a network error, "you don't know Kubernetes" is a statement about the
network, not about the user.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Mapping, Sequence

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
from app.candidate.vocabulary import (
    canonical_for,
    find_skills,
    normalise_surface,
    split_skill_line,
)
from app.memory.long_term_memory_qdrant import long_term_memory_qdrant
from app.memory.retrieval_result import RetrievalStatus

logger = logging.getLogger(__name__)


# ── Small parsers over résumé text ───────────────────────────────────────────

_CGPA_RE = re.compile(
    r"\b(?:CGPA|CPI|SPI|GPA|SGPA)\b\s*[:\-]?\s*([0-9]{1,2}(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)

# "B.Tech. in Information Technology", "M.Sc Physics", "Bachelor of Science"
_DEGREE_RE = re.compile(
    r"\b(B\.?\s?Tech|B\.?\s?E|B\.?\s?Sc|B\.?\s?C\.?A|M\.?\s?Tech|M\.?\s?Sc|"
    r"M\.?\s?C\.?A|MBA|Ph\.?\s?D|Bachelor|Master|Diploma)\b[^\n,]*",
    re.IGNORECASE,
)

_HEADING_LINE_RE = re.compile(
    r"^\s*(education|experience|projects?|skills?|achievements?|"
    r"technical skills|work experience)\s*:?\s*$",
    re.IGNORECASE,
)


def _first_meaningful_line(text: str) -> str:
    """First line that is content rather than a section heading."""
    for line in (text or "").split("\n"):
        stripped = line.strip()
        if stripped and not _HEADING_LINE_RE.match(stripped):
            return stripped
    return ""


def _strip_heading(text: str) -> str:
    """Drop a leading section heading carried on the first chunk of a block."""
    lines = (text or "").split("\n")
    if lines and _HEADING_LINE_RE.match(lines[0].strip()):
        return "\n".join(lines[1:]).strip()
    return (text or "").strip()


def _excerpt(text: str, limit: int = 240) -> str:
    """A quoted fragment. Never a paraphrase — that is where invention starts."""
    flat = " ".join((text or "").split())
    return flat[:limit]


# ── Evidence accumulation ────────────────────────────────────────────────────

class _SkillTable:
    """Collects evidence per canonical skill while sources are read."""

    def __init__(self) -> None:
        self._categories: Dict[str, str] = {}
        self._aliases: Dict[str, List[str]] = {}
        self._evidence: Dict[str, List[Evidence]] = {}
        self._names: Dict[str, str] = {}

    def add(
        self,
        canonical: str,
        evidence: Evidence,
        *,
        category: str = "",
        surface: str = "",
    ) -> None:
        key = canonical.lower()
        self._names.setdefault(key, canonical)
        self._evidence.setdefault(key, []).append(evidence)
        if category and not self._categories.get(key):
            self._categories[key] = category
        if surface:
            seen = self._aliases.setdefault(key, [])
            if surface not in seen:
                seen.append(surface)

    def build(self) -> Dict[str, SkillClaim]:
        claims: Dict[str, SkillClaim] = {}
        for key, evidence in self._evidence.items():
            if not evidence:
                # Unreachable by construction, and skipped rather than raising:
                # a claim with no evidence must never exist, and a bug here
                # should cost a skill rather than the whole profile.
                continue
            claims[key] = SkillClaim(
                name=self._names[key],
                category=self._categories.get(key, ""),
                aliases=tuple(self._aliases.get(key, [])),
                evidence=tuple(evidence),
            )
        return claims


# ── The builder ──────────────────────────────────────────────────────────────

class CandidateProfileBuilder:
    """Reads the existing stores and returns a typed profile."""

    def __init__(self, memory=None) -> None:
        self._memory = memory or long_term_memory_qdrant

    async def build(self, user_id: str) -> CandidateProfile:
        """
        Assemble the profile. Sources are loaded concurrently and independently:
        one failure degrades the result rather than aborting it, because a
        partial profile is useful and a missing one is not.
        """
        skills_res, projects_res, experience_res, education_res, achievements_res, name_res = (
            await asyncio.gather(
                self._safe(self._memory.retrieve_skills(user_id=user_id, limit=50)),
                self._safe(self._memory.retrieve_projects(user_id=user_id, limit=25)),
                self._safe(self._memory.retrieve_section(user_id=user_id, section="experience", limit=25)),
                self._safe(self._memory.retrieve_section(user_id=user_id, section="education", limit=10)),
                self._safe(self._memory.retrieve_section(user_id=user_id, section="achievements", limit=10)),
                self._safe(self._memory.retrieve_section(user_id=user_id, section="name", limit=2)),
            )
        )

        loaded: List[ProfileSource] = []
        failed: List[ProfileSource] = []
        table = _SkillTable()

        def record(source: ProfileSource, result) -> list:
            if result is None or result.status is RetrievalStatus.ERROR:
                failed.append(source)
                return []
            loaded.append(source)
            return list(result)

        skill_items = record(ProfileSource.SKILLS, skills_res)
        project_items = record(ProfileSource.PROJECTS, projects_res)
        experience_items = record(ProfileSource.EXPERIENCE, experience_res)
        education_items = record(ProfileSource.EDUCATION, education_res)
        achievement_items = record(ProfileSource.ACHIEVEMENTS, achievements_res)
        name_items = record(ProfileSource.NAME, name_res)

        self._read_skill_section(skill_items, table)
        projects = self._read_projects(project_items, table)
        experience = self._read_experience(experience_items, table)
        education = self._read_education(education_items, table)
        achievements = self._read_achievements(achievement_items, table)

        profile = CandidateProfile(
            owner_id=user_id,
            name=_first_meaningful_line(
                name_items[0].get("content", "")
            ) if name_items else None,
            skills=table.build(),
            projects=tuple(projects),
            experience=tuple(experience),
            education=tuple(education),
            achievements=tuple(achievements),
            sources_loaded=tuple(loaded),
            sources_failed=tuple(failed),
        )
        logger.info("Built candidate profile: %s", profile.summary())
        return profile

    @staticmethod
    async def _safe(coro):
        """Await a retrieval, converting a raised exception into None."""
        try:
            return await coro
        except Exception as exc:
            logger.error("Profile source failed: %s", exc)
            return None

    # ── Sources ──────────────────────────────────────────────────────────────

    def _read_skill_section(self, items: Sequence[Mapping], table: _SkillTable) -> None:
        """
        The skills section: the user listing what they know.

        Each line is `Category: a, b, c`. Only names the vocabulary recognises
        become claims — an unknown token is dropped rather than invented into a
        qualification.
        """
        for item in items:
            content = str(item.get("content") or "")
            source_id = self._source_id(item)
            for line in _strip_heading(content).split("\n"):
                category, entries = split_skill_line(line)
                for entry in entries:
                    surface = normalise_surface(entry)
                    skill = canonical_for(surface)
                    if skill is None:
                        continue
                    table.add(
                        skill.canonical,
                        Evidence(
                            kind=EvidenceKind.RESUME_SKILLS,
                            source_id=source_id,
                            excerpt=_excerpt(line),
                        ),
                        category=category or skill.category,
                        surface=surface,
                    )

    def _read_projects(
        self, items: Sequence[Mapping], table: _SkillTable
    ) -> List[ProjectEntry]:
        """Projects, and the skills each one demonstrates."""
        projects: List[ProjectEntry] = []
        for item in items:
            content = str(item.get("content") or "")
            metadata = item.get("metadata") or {}
            title = (
                str(item.get("title") or metadata.get("entity_title") or "").strip()
                or _first_meaningful_line(content)
            )
            entity_id = str(
                metadata.get("entity_id") or self._source_id(item)
            )
            detected = find_skills(content)
            for skill, surface in detected:
                table.add(
                    skill.canonical,
                    Evidence(
                        kind=EvidenceKind.RESUME_PROJECT,
                        source_id=entity_id,
                        excerpt=_excerpt(content),
                        detail=title or "a project",
                    ),
                    category=skill.category,
                    surface=surface,
                )
            projects.append(ProjectEntry(
                title=title,
                entity_id=entity_id,
                summary=_excerpt(_strip_heading(content), 400),
                skills=tuple(sorted(s.canonical for s, _ in detected)),
            ))
        return projects

    def _read_experience(
        self, items: Sequence[Mapping], table: _SkillTable
    ) -> List[ExperienceEntry]:
        """
        Roles, and the skills used in them — the strongest evidence there is.

        Title and organisation are read from the first two content lines, which
        is the shape the extractor produces ("DrUpsc (EdTech) May 2026 – June
        2026" then "Full Stack Developer Intern Remote"). Both are best-effort:
        a wrong label costs a nicer explanation, never a wrong claim, because
        the skills come from the text rather than from the parse.
        """
        entries: List[ExperienceEntry] = []
        for item in items:
            content = str(item.get("content") or "")
            source_id = self._source_id(item)
            body = _strip_heading(content)
            lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
            organisation = lines[0] if lines else ""
            title = lines[1] if len(lines) > 1 else ""

            detected = find_skills(content)
            label = title or organisation or "a role"
            for skill, surface in detected:
                table.add(
                    skill.canonical,
                    Evidence(
                        kind=EvidenceKind.RESUME_EXPERIENCE,
                        source_id=source_id,
                        excerpt=_excerpt(content),
                        detail=label,
                    ),
                    category=skill.category,
                    surface=surface,
                )
            entries.append(ExperienceEntry(
                title=title,
                organisation=organisation,
                source_id=source_id,
                summary=_excerpt(body, 400),
                skills=tuple(sorted(s.canonical for s, _ in detected)),
            ))
        return entries

    def _read_education(
        self, items: Sequence[Mapping], table: _SkillTable
    ) -> List[EducationEntry]:
        """
        Education, plus any coursework that names a known skill.

        Coursework is real evidence and the weakest kind — `RESUME_EDUCATION`
        is weighted accordingly, so "studied it" never outranks "shipped it".
        """
        entries: List[EducationEntry] = []
        for item in items:
            content = str(item.get("content") or "")
            source_id = self._source_id(item)
            body = _strip_heading(content)
            lines = [ln.strip() for ln in body.split("\n") if ln.strip()]

            cgpa_match = _CGPA_RE.search(content)
            degree_match = _DEGREE_RE.search(content)

            for skill, surface in find_skills(content):
                table.add(
                    skill.canonical,
                    Evidence(
                        kind=EvidenceKind.RESUME_EDUCATION,
                        source_id=source_id,
                        excerpt=_excerpt(content),
                        detail="your coursework",
                    ),
                    category=skill.category,
                    surface=surface,
                )

            entries.append(EducationEntry(
                institution=lines[0] if lines else "",
                qualification=(degree_match.group(0).strip() if degree_match else ""),
                source_id=source_id,
                raw=_excerpt(body, 400),
                cgpa=cgpa_match.group(1) if cgpa_match else None,
            ))
        return entries

    def _read_achievements(
        self, items: Sequence[Mapping], table: _SkillTable
    ) -> List[str]:
        achievements: List[str] = []
        for item in items:
            content = str(item.get("content") or "")
            source_id = self._source_id(item)
            body = _strip_heading(content)
            if body:
                achievements.append(_excerpt(body, 300))
            for skill, surface in find_skills(content):
                table.add(
                    skill.canonical,
                    Evidence(
                        kind=EvidenceKind.RESUME_ACHIEVEMENT,
                        source_id=source_id,
                        excerpt=_excerpt(content),
                    ),
                    category=skill.category,
                    surface=surface,
                )
        return achievements

    @staticmethod
    def _source_id(item: Mapping) -> str:
        """
        The stored identifier for this chunk.

        Ingestion already writes `string_id` on every point, so a claim can be
        traced to the exact chunk it came from without inventing a new id
        scheme. The fallbacks cover legacy points written before those fields
        existed.
        """
        metadata = item.get("metadata") or {}
        return str(
            metadata.get("string_id")
            or metadata.get("entity_id")
            or metadata.get("parent_id")
            or "unknown"
        )


candidate_profile_builder = CandidateProfileBuilder()


async def build_candidate_profile(user_id: str) -> CandidateProfile:
    """Convenience entry point used by the agent tools."""
    return await candidate_profile_builder.build(user_id)


__all__ = [
    "CandidateProfileBuilder",
    "build_candidate_profile",
    "candidate_profile_builder",
]
