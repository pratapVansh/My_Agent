"""
Turning a job description into requirements, without a model.

Extraction is deterministic for the same reason confirmation detection is: the
output decides what the assistant will later claim about a real person, and a
model asked to "list the requirements" will helpfully invent a few plausible
ones. A hallucinated *requirement* is subtler than a hallucinated skill and does
the same damage — it produces a gap the candidate does not actually have, or a
match against something the posting never asked for.

So this reads text and nothing else. Three signals, in falling priority:

**Section headings.** "Requirements:" opens a required block, "Nice to have:"
opens a preferred one, and "Benefits:" opens a block that yields nothing at all
— a perks list mentioning Slack is not asking for Slack.

**Line markers.** A line may override its section: "Kubernetes preferred" under
a Requirements heading is preferred, and preferred markers deliberately beat
required ones because "strong Kubernetes preferred" is a wish, not a bar.

**The canonical vocabulary.** Skills come from `app.candidate.vocabulary` — the
same closed set the résumé is read with, which is what makes `ReactJS` in a
posting meet `React.js` in a résumé. A token the vocabulary does not know is not
promoted into a requirement; it stays in the line, and if the line described a
duty it becomes a RESPONSIBILITY the report will openly say it could not check.

The default for an unmarked line is REQUIRED. That is the conservative reading:
treating an unlabelled bullet as optional would quietly delete bars the
candidate has to clear, and the whole point of the score is that missing a bar
shows up.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from app.candidate.vocabulary import find_skills
from app.matching.models import (
    DEGREE_LEVELS,
    JobRequirements,
    Necessity,
    Requirement,
    RequirementKind,
)

logger = logging.getLogger(__name__)


# ── Section headings ─────────────────────────────────────────────────────────

_REQUIRED_HEADING_RE = re.compile(
    r"^\s*[-*•·]?\s*(?:"
    r"requirements?|required(?:\s+qualifications?|\s+skills?)?"
    r"|must[\s-]?haves?|minimum\s+qualifications?|basic\s+qualifications?"
    r"|what\s+(?:you(?:'|’)?ll\s+need|we(?:'|’)?re\s+looking\s+for)"
    r"|who\s+you\s+are|qualifications?|skills?\s+(?:and|&)\s+experience"
    r"|essential(?:\s+skills?)?"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

_PREFERRED_HEADING_RE = re.compile(
    r"^\s*[-*•·]?\s*(?:"
    r"preferred(?:\s+qualifications?|\s+skills?)?|nice[\s-]?to[\s-]?haves?"
    r"|bonus(?:\s+points?)?|good\s+to\s+have|desirable|pluses?"
    r"|what\s+will\s+set\s+you\s+apart|extra\s+credit"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

# Blocks that describe the company or the package rather than the candidate.
# A benefits list naming Slack is not a Slack requirement.
_IGNORED_HEADING_RE = re.compile(
    r"^\s*[-*•·]?\s*(?:"
    r"benefits?|perks?(?:\s+(?:and|&)\s+benefits?)?|compensation|salary"
    r"|about\s+(?:us|the\s+company|the\s+team)|who\s+we\s+are|our\s+mission"
    r"|equal\s+opportunity|eeo\s+statement|how\s+to\s+apply|application\s+process"
    r"|location|why\s+join\s+us"
    r")\s*:?\s*$",
    re.IGNORECASE,
)


# ── Line-level markers ───────────────────────────────────────────────────────

# Checked first: "strong Kubernetes preferred" is a wish despite "strong".
_PREFERRED_MARKER_RE = re.compile(
    r"\b(?:preferred|preferably|nice\s+to\s+have|a\s+(?:big\s+)?plus"
    r"|bonus|ideally|desirable|would\s+be\s+(?:great|nice|a\s+plus)"
    r"|good\s+to\s+have|familiarity\s+with|exposure\s+to|appreciated"
    r"|not\s+required|optional)\b",
    re.IGNORECASE,
)

_REQUIRED_MARKER_RE = re.compile(
    r"\b(?:must\s+have|must\s+be|required|require[sd]?|at\s+least|minimum(?:\s+of)?"
    r"|strong(?:\s+(?:experience|background|knowledge|command))?"
    r"|proficien\w*|expert(?:ise)?|solid|deep\s+(?:experience|knowledge)"
    r"|essential|mandatory|you\s+(?:will\s+)?need)\b",
    re.IGNORECASE,
)


# ── Experience and education ─────────────────────────────────────────────────

# "3+ years", "at least 2 years", "2-4 years", "minimum 5 yrs"
_YEARS_RE = re.compile(
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*\d{1,2}\s*\+?\s*)?"
    r"(?:years?|yrs?)\b",
    re.IGNORECASE,
)

# Degree mentions. Deliberately tight: a bare "BS" or "MS" is far more often an
# abbreviation of something else than a degree, so the long forms stand alone
# but the initialisms need "degree" or an "in <field>" to count.
_DEGREE_PATTERNS: Tuple[Tuple[re.Pattern, int], ...] = (
    (re.compile(r"\b(?:ph\.?\s?d|doctorate|doctoral)\b", re.IGNORECASE), 4),
    (re.compile(r"\bmaster(?:'|’)?s?\b", re.IGNORECASE), 3),
    (re.compile(r"\bm\.?\s?tech\b", re.IGNORECASE), 3),
    (re.compile(r"\bm\.?\s?s\.?\s*(?=in\b|degree\b)", re.IGNORECASE), 3),
    (re.compile(r"\bbachelor(?:'|’)?s?\b", re.IGNORECASE), 2),
    (re.compile(r"\bb\.?\s?tech\b", re.IGNORECASE), 2),
    (re.compile(r"\bundergraduate\s+degree\b", re.IGNORECASE), 2),
    (re.compile(r"\bb\.?\s?s\.?\s*(?=in\b|degree\b)", re.IGNORECASE), 2),
    (re.compile(r"\bdiploma\b", re.IGNORECASE), 1),
)

_FIELD_RE = re.compile(
    r"\bin\s+([A-Za-z][A-Za-z /&-]{2,40}?)"
    r"(?=\s*(?:,|\.|;|\)|or\b|and\b|with\b|is\b|preferred\b|required\b|$))",
    re.IGNORECASE,
)


# ── Domain vocabulary ────────────────────────────────────────────────────────
#
# A deliberately small closed set. Each entry is a phrase a posting genuinely
# asks for and that can be checked by literal search over the candidate's own
# project and role text. Anything matched here is capped at PARTIAL by the
# matcher — a word appearing in a project description is real evidence and
# weaker than a skill the résumé names outright.
DOMAIN_TERMS: Dict[str, Tuple[str, ...]] = {
    "Production systems": ("production", "in production", "production-grade"),
    "Distributed systems": ("distributed system", "distributed systems"),
    "Real-time systems": ("real-time", "real time", "realtime", "low-latency",
                          "low latency", "streaming"),
    "Scalability": ("scalable", "scalability", "at scale", "high-throughput"),
    "Open source": ("open source", "open-source", "github"),
    "Startup experience": ("startup", "start-up", "early-stage"),
    "Research": ("research", "paper", "publication", "meta-analysis"),
    "Data pipelines": ("data pipeline", "etl", "ingestion pipeline"),
    "Observability": ("observability", "monitoring", "logging", "tracing"),
    "Code review": ("code review", "peer review", "pull request"),
    "Mentoring": ("mentor", "mentoring", "mentored"),
}

_DOMAIN_LOOKUP: Tuple[Tuple[str, str], ...] = tuple(
    sorted(
        ((surface, canonical)
         for canonical, surfaces in DOMAIN_TERMS.items()
         for surface in surfaces),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)


# ── Line splitting ───────────────────────────────────────────────────────────

# A snippet from job search arrives as one blob with no newlines. Sentence
# boundaries and bullet glyphs are the only structure available there.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+(?=[A-Z0-9])")
_BULLET_SPLIT_RE = re.compile(r"\s*[•·▪◦]\s*|\s+[-*]\s+")

_MIN_LINE_CHARS = 3
_MAX_LINES = 400
"""Ceiling on how much of a posting is read. A pasted 50-page careers page
should cost a bounded amount of work, not an unbounded one."""


def _split_lines(text: str) -> List[str]:
    """Break a posting into the smallest units that carry one requirement."""
    pieces: List[str] = []
    for raw_line in (text or "").splitlines():
        for bullet in _BULLET_SPLIT_RE.split(raw_line):
            stripped = bullet.strip(" \t-*•·|")
            if not stripped:
                continue
            # A run-on line (the job-search snippet case) still splits on
            # sentence boundaries, so "Requires Python. Docker a plus." yields
            # two requirements with different necessity rather than one.
            for sentence in _SENTENCE_SPLIT_RE.split(stripped):
                cleaned = sentence.strip(" \t-*•·|")
                if len(cleaned) >= _MIN_LINE_CHARS:
                    pieces.append(cleaned)
            if len(pieces) >= _MAX_LINES:
                return pieces[:_MAX_LINES]
    return pieces[:_MAX_LINES]


def _necessity_for(line: str, section: Optional[Necessity]) -> Necessity:
    """
    The necessity of one line.

    Preferred markers are tested first on purpose: "strong Kubernetes
    preferred" contains both a required marker and a preferred one, and reading
    it as a bar would invent a gap the posting did not create.
    """
    if _PREFERRED_MARKER_RE.search(line):
        return Necessity.PREFERRED
    if _REQUIRED_MARKER_RE.search(line):
        return Necessity.REQUIRED
    if section is not None:
        return section
    return Necessity.REQUIRED


def degree_level_of(text: str) -> Optional[int]:
    """
    The highest degree level named anywhere in this text.

    Shared by the extractor (reading a posting) and the matcher (reading the
    candidate's parsed education), so "Master's" on one side and "M.Tech" on
    the other are compared as levels rather than as strings. A second copy of
    this table would be free to drift, and a drift here silently invents or
    erases an education gap.
    """
    for pattern, level in _DEGREE_PATTERNS:
        if pattern.search(text or ""):
            return level
    return None


def _degree_in(line: str) -> Optional[Tuple[int, str]]:
    """The highest degree level named in this line, and its field if given."""
    level = degree_level_of(line)
    if level is None:
        return None
    field_match = _FIELD_RE.search(line)
    field = ""
    if field_match:
        field = " ".join(field_match.group(1).split()).strip(" .,;")
    return level, field


def _domains_in(line: str) -> List[Tuple[str, str]]:
    """Closed-vocabulary domain phrases named in this line."""
    lowered = line.lower()
    found: Dict[str, str] = {}
    for surface, canonical in _DOMAIN_LOOKUP:
        if canonical in found:
            continue
        if surface in lowered:
            found[canonical] = surface
    return list(found.items())


def _looks_like_duty(line: str) -> bool:
    """
    Whether an unmatched line describes work rather than boilerplate.

    Used only to decide what to keep as a RESPONSIBILITY, which is never
    scored — so the cost of being wrong here is a slightly longer "not
    assessed" list, never a wrong verdict.
    """
    words = line.split()
    if not (4 <= len(words) <= 40):
        return False
    return bool(re.search(
        r"\b(?:build|design|develop|deploy|maintain|own|lead|collaborat\w*"
        r"|ship|scale|optimi[sz]e|implement|architect|drive|partner"
        r"|experience|ability|able\s+to|comfortable|work\s+with)\b",
        line, re.IGNORECASE,
    ))


def extract_requirements(
    text: str,
    *,
    title: str = "",
    source: str = "",
    keep_responsibilities: bool = True,
) -> JobRequirements:
    """
    Read a posting and return what it asks for.

    De-duplicates by `Requirement.key`, keeping the strictest necessity seen:
    a posting that mentions Python in a required block and again in a preferred
    one is asking for Python, and reading the second mention as a downgrade
    would be wrong.
    """
    if not (text or "").strip():
        return JobRequirements(title=title, source=source, text_chars=0)

    found: Dict[str, Requirement] = {}
    section: Optional[Necessity] = None
    suppressed = False

    def keep(requirement: Requirement) -> None:
        existing = found.get(requirement.key)
        if existing is None:
            found[requirement.key] = requirement
            return
        # Strictest wins. REQUIRED outranks PREFERRED.
        if (existing.necessity is Necessity.PREFERRED
                and requirement.necessity is Necessity.REQUIRED):
            found[requirement.key] = requirement

    for line in _split_lines(text):
        if _IGNORED_HEADING_RE.match(line):
            suppressed = True
            section = None
            continue
        if _PREFERRED_HEADING_RE.match(line):
            section, suppressed = Necessity.PREFERRED, False
            continue
        if _REQUIRED_HEADING_RE.match(line):
            section, suppressed = Necessity.REQUIRED, False
            continue
        if suppressed:
            continue

        necessity = _necessity_for(line, section)
        matched_anything = False

        # ── Skills, from the same closed vocabulary the résumé is read with ──
        for skill, surface in find_skills(line):
            matched_anything = True
            keep(Requirement(
                kind=RequirementKind.SKILL,
                canonical=skill.canonical,
                necessity=necessity,
                surface=surface,
                source_line=line[:200],
            ))

        # ── A minimum duration ───────────────────────────────────────────────
        years_match = _YEARS_RE.search(line)
        if years_match:
            matched_anything = True
            years = int(years_match.group(1))
            keep(Requirement(
                kind=RequirementKind.EXPERIENCE_YEARS,
                canonical=f"{years}+ years",
                necessity=necessity,
                surface=years_match.group(0),
                source_line=line[:200],
                min_years=years,
            ))

        # ── A degree ─────────────────────────────────────────────────────────
        degree = _degree_in(line)
        if degree is not None:
            level, field_of_study = degree
            matched_anything = True
            name = next(
                (n for n, v in DEGREE_LEVELS.items() if v == level), "degree"
            )
            keep(Requirement(
                kind=RequirementKind.EDUCATION,
                canonical=f"{name} degree",
                necessity=necessity,
                surface=line[:80],
                source_line=line[:200],
                degree_level=level,
                field_of_study=field_of_study,
            ))

        # ── Domain phrases ───────────────────────────────────────────────────
        for canonical, surface in _domains_in(line):
            matched_anything = True
            keep(Requirement(
                kind=RequirementKind.DOMAIN,
                canonical=canonical,
                necessity=necessity,
                surface=surface,
                source_line=line[:200],
            ))

        # ── Anything left that still described work ──────────────────────────
        if not matched_anything and keep_responsibilities and _looks_like_duty(line):
            keep(Requirement(
                kind=RequirementKind.RESPONSIBILITY,
                canonical=" ".join(line.split())[:120],
                necessity=necessity,
                surface=line[:120],
                source_line=line[:200],
            ))

    result = JobRequirements(
        requirements=tuple(found.values()),
        title=title.strip(),
        source=source,
        text_chars=len(text),
    )
    logger.info("Extracted job requirements: %s", result.summary())
    return result


__all__ = ["DOMAIN_TERMS", "degree_level_of", "extract_requirements"]
