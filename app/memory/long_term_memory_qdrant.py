"""
Qdrant-based Long-term Memory Implementation.
Stores persistent information with Cohere embeddings and text chunking.
"""
from qdrant_client.models import PointStruct
from typing import AbstractSet, List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import asyncio
import uuid
import logging
import re
from app.services.qdrant_service import qdrant_service
from app.services.cohere_service import cohere_service
from app.services.chunking_service import chunking_service
from app.services.debug_logger import log_step
from app.memory.retrieval_result import RetrievalResult, RetrievalStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Résumé section parsing
#
# Heading recognition is *whole-line*, never substring. The previous
# implementation asked "does a known heading appear anywhere in this line?",
# which classified ordinary content as a delimiter and dropped it: the lines
# "Programming Languages: C/C++, Python, SQL" and "Tools: Git, Linux, Docker"
# both contain a known heading, so the two most valuable lines of a skills
# section were consumed as separators and never stored.
#
# PDF extractors also emit wildly inconsistent bullet glyphs — PyPDF2 renders a
# Wingdings bullet as U+FFFD or U+F0B7, usually with no trailing space — so
# matching only "-", "*" and "•" followed by whitespace meant real résumés never
# segmented into entries and every project collapsed into one blob.
# ─────────────────────────────────────────────────────────────────────────────

_BULLET_CHARS = (
    "-*•‣⁃▪▫▸▹●◦■"
    "∙·–—‐→»♦"
    "�"
)
_BULLET_RE = re.compile(f"^[{re.escape(_BULLET_CHARS)}]+[ \t]*")

# Words that carry no section meaning on their own, so they never block a match.
_HEADING_CONNECTORS = {"and", "of", "the", "in", "for", "my", "a", "to", "with"}

# A heading is short. These caps are what stop a content line that happens to
# mention a section word from being read as a boundary.
_MAX_HEADING_CHARS = 60
_MAX_HEADING_WORDS = 6

# Per section: full phrases (matched exactly), the complete set of tokens a
# heading of that section may contain, and the "strong" subset — at least one of
# which must be present, so a bare "Key" or "Core" never opens a section.
#
# A line matches a section only when *every* one of its tokens belongs to that
# section's token set. That whole-line rule is what makes the matcher safe to
# extend: adding a synonym cannot make it swallow content, because content
# lines always carry words outside the vocabulary.
_SECTION_VOCAB: Dict[str, Dict[str, set]] = {
    "skills": {
        "phrases": {
            "skills", "technical skills", "technology skills", "technologies",
            "tech stack", "technical stack", "tools", "tools & technologies",
            "programming languages", "programming languages & tools",
            "languages & tools", "technical expertise", "core competencies",
            "key skills", "skill set", "skillset", "technologies used",
            "frameworks", "software skills", "it skills", "computer skills",
            "technical proficiencies", "areas of expertise", "technical skills & tools",
        },
        "tokens": {
            "skill", "skills", "skillset", "technical", "technology",
            "technologies", "tech", "stack", "tool", "tools", "tooling",
            "toolset", "programming", "language", "languages", "competency",
            "competencies", "core", "key", "expertise", "proficiency",
            "proficiencies", "framework", "frameworks", "software", "it",
            "computer", "areas",
        },
        "strong": {
            "skill", "skills", "skillset", "technology", "technologies", "tech",
            "tool", "tools", "toolset", "tooling", "competency", "competencies",
            "expertise", "proficiency", "proficiencies", "framework",
            "frameworks", "programming", "languages",
        },
    },
    "projects": {
        "phrases": {
            "projects", "project experience", "key projects", "personal projects",
            "academic projects", "notable projects", "selected projects",
            "major projects", "side projects", "project work", "portfolio",
            "technical projects", "mini projects",
        },
        "tokens": {
            "project", "projects", "personal", "academic", "notable", "selected",
            "key", "major", "side", "portfolio", "work", "mini", "capstone",
            "technical",
        },
        "strong": {"project", "projects", "portfolio", "capstone"},
    },
    "experience": {
        "phrases": {
            "experience", "work experience", "employment", "internship",
            "internships", "professional experience", "work history",
            "career history", "job experience", "professional background",
            "industry experience", "employment history", "relevant experience",
        },
        "tokens": {
            "experience", "experiences", "work", "employment", "internship",
            "internships", "professional", "career", "history", "industry",
            "job", "jobs", "relevant", "background",
        },
        "strong": {
            "experience", "experiences", "employment", "internship",
            "internships", "career", "history", "professional",
        },
    },
    "education": {
        "phrases": {
            "education", "academic background", "academics", "qualification",
            "educational background", "academic qualifications", "degrees",
            "educational qualifications", "scholastic details", "academic details",
            "education & training", "academic record",
        },
        "tokens": {
            "education", "educational", "academic", "academics", "background",
            "qualification", "qualifications", "degree", "degrees", "scholastic",
            "details", "record", "training", "school", "schooling", "university",
            "college",
        },
        "strong": {
            "education", "educational", "academic", "academics", "qualification",
            "qualifications", "degree", "degrees", "scholastic", "school",
            "schooling", "university", "college",
        },
    },
    "achievements": {
        "phrases": {
            "achievements", "awards", "certifications", "honors", "honours",
            "accomplishments", "certificates", "recognitions", "honors & awards",
            "awards & achievements", "achievements & awards", "licenses",
        },
        "tokens": {
            "achievement", "achievements", "award", "awards", "certification",
            "certifications", "certificate", "certificates", "honor", "honors",
            "honour", "honours", "accomplishment", "accomplishments",
            "recognition", "recognitions", "license", "licenses", "scholarship",
            "scholarships",
        },
        "strong": {
            "achievement", "achievements", "award", "awards", "certification",
            "certifications", "certificate", "certificates", "honor", "honors",
            "honour", "honours", "accomplishment", "accomplishments",
            "recognition", "recognitions", "license", "licenses", "scholarship",
            "scholarships",
        },
    },
    # Recognised but untyped. These still have to be *detected*, otherwise their
    # content silently accretes onto whichever section came before — which is
    # how "Relevant Coursework" ended up inside this user's projects.
    "other": {
        "phrases": {
            "summary", "professional summary", "objective", "career objective",
            "profile", "about", "about me", "overview", "relevant coursework",
            "coursework", "courses", "relevant courses", "subjects", "curriculum",
            "interests", "hobbies", "activities", "extracurricular",
            "extracurricular activities", "co-curricular activities",
            "volunteer experience", "volunteering", "leadership",
            "positions of responsibility", "publications", "research",
            "references", "contact", "languages known", "personal details",
            "declaration", "strengths", "workshops", "seminars",
        },
        "tokens": {
            "summary", "professional", "objective", "career", "profile", "about",
            "me", "overview", "relevant", "coursework", "course", "courses",
            "subject", "subjects", "curriculum", "curricular", "co", "interest",
            "interests", "hobby", "hobbies", "activity", "activities",
            "extracurricular", "volunteer", "volunteering", "leadership",
            "position", "positions", "responsibility", "responsibilities",
            "publication", "publications", "research", "reference", "references",
            "contact", "known", "personal", "details", "declaration", "strength",
            "strengths", "workshop", "workshops", "seminar", "seminars",
            "languages",
        },
        "strong": {
            "summary", "objective", "profile", "overview", "coursework", "course",
            "courses", "curriculum", "interests", "hobbies", "activities",
            "extracurricular", "volunteer", "volunteering", "leadership",
            "publications", "research", "references", "declaration", "strengths",
            "workshops", "seminars", "responsibility", "responsibilities",
            "declaration",
        },
    },
}

# Ties are broken in this order — most specific section first.
_SECTION_PRIORITY = ["skills", "projects", "experience", "education", "achievements", "other"]

# "Label: values" — a self-describing content line such as "Backend: FastAPI".
_LABEL_RE = re.compile(r"^[^:]{1,40}:\s*\S")


def _normalise_heading_text(value: str) -> str:
    """Reduce a candidate heading to comparable form: no decoration, lowercase."""
    text = re.sub(r"\(.*?\)", " ", value)          # "Skills (Technical)" → "Skills"
    text = text.replace("’", "'")
    text = re.sub(r"[^\w&/|,+\-\s']+", " ", text)  # drop stray punctuation/glyphs
    return re.sub(r"\s+", " ", text).strip().lower()


def _heading_tokens(normalised: str) -> List[str]:
    # Hyphens and dashes join heading words as often as spaces do
    # ("Work-Experience", "Co-Curricular Activities").
    parts = re.split(r"[\s/|,&+\-–—]+", normalised)
    return [p.strip(".'") for p in parts if p.strip(".'")]


# "** Skills **", "— Skills —", "== EDUCATION ==" — styling carried through from
# the source document. Stripped before the bullet test, since several of these
# characters are also bullet glyphs and would otherwise make the heading look
# like a list item.
_DECORATION_RE = re.compile(r"^[*\-–—~=_#•●▪]+\s*(.+?)\s*[*\-–—~=_#•●▪]+$")


def classify_heading(line: str) -> Optional[Tuple[str, str]]:
    """
    Decide whether `line` opens a résumé section.

    Returns ``(section, inline_content)`` or None. ``inline_content`` is the
    remainder of a "SKILLS: Python, Java" style heading, which is a heading and
    a content line at once — both meanings have to survive, since dropping
    either loses a section boundary or loses the skills themselves.
    """
    raw = (line or "").strip()
    if not raw:
        return None

    decorated = _DECORATION_RE.match(raw)
    if decorated:
        raw = decorated.group(1).strip()

    # A bullet is always content. This single check removes a whole class of
    # false positives, e.g. "• Led the migration ... experience". Decoration is
    # stripped first, so "— Skills —" is still read as the heading it is.
    if _BULLET_RE.match(raw):
        return None

    candidate, inline = raw, ""
    if ":" in raw:
        prefix, remainder = raw.split(":", 1)
        candidate, inline = prefix.strip(), remainder.strip()

    if not candidate or len(candidate) > _MAX_HEADING_CHARS:
        return None

    # Headings do not carry dates, CGPAs or roll numbers. This is what keeps
    # "B.Tech. in Information Technology (CGPA: 8.80 / 10) Aug 2023-Present"
    # and "TRACE | GitHub June 2026" out of the heading path.
    if any(ch.isdigit() for ch in candidate):
        return None

    normalised = _normalise_heading_text(candidate)
    if not normalised:
        return None

    tokens = [t for t in _heading_tokens(normalised) if t not in _HEADING_CONNECTORS]
    if not tokens or len(tokens) > _MAX_HEADING_WORDS:
        return None

    matches: List[Tuple[int, int, str]] = []
    for section in _SECTION_PRIORITY:
        vocab = _SECTION_VOCAB[section]
        if normalised in vocab["phrases"]:
            matches.append((99, _SECTION_PRIORITY.index(section), section))
            continue
        if not all(t in vocab["tokens"] for t in tokens):
            continue
        strong = sum(1 for t in tokens if t in vocab["strong"])
        if strong:
            matches.append((strong, _SECTION_PRIORITY.index(section), section))

    if not matches:
        return None

    # Most strong-token evidence wins; ties fall back to section priority.
    matches.sort(key=lambda m: (-m[0], m[1]))
    return matches[0][2], inline


# A bullet that overflows its line continues on the next one. PDF extraction
# makes this common, and the continuation is not a new entry.
_CONTINUATION_TAIL = ("-", "–", "—", ",", ";", ":", "&", "/")

# A line ending on one of these is grammatically unfinished, so the next line
# continues it no matter how that next line is capitalised.
_DANGLING_WORDS = {
    "and", "or", "with", "for", "to", "of", "the", "a", "an", "in", "on", "at",
    "by", "using", "from", "as", "that", "which", "into", "per", "via",
    "including", "between", "across", "through", "under", "over",
}

# Entry titles carry a date, a date range, or a separator ("Project | GitHub",
# "Acme Corp — Engineer"). These are what let a bullet-free résumé segment.
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
_DATE_RE = re.compile(
    rf"\b(?:{_MONTHS}[a-z]*\.?\s*'?\d{{2,4}}"
    rf"|\d{{1,2}}[/-]\d{{4}}"
    rf"|\d{{4}}\s*[-–—]\s*(?:\d{{4}}|present|current|ongoing|now)"
    rf"|present|current|ongoing)\b",
    re.IGNORECASE,
)
_TITLE_SEP_RE = re.compile(r"\s[|·]\s|\s[-–—]\s|\s\|\s")
_SENTENCE_END = (".", "!", "?")
_MAX_TITLE_CHARS = 140


def _is_continuation(text: str, prev_text: str) -> bool:
    """True when `text` is the tail of a wrapped line rather than a new entry."""
    if not prev_text:
        return False
    # "based schizophrenia classification." — wrapped text keeps running in
    # lower case, whereas a job or project title starts capitalised.
    if text[:1].islower():
        return True
    stripped = prev_text.rstrip()
    # "...using Python, Pandas, and Scikit-learn for machine learning-"
    if stripped.endswith(_CONTINUATION_TAIL):
        return True
    last_word = stripped.split()[-1].strip(".,;:").lower() if stripped.split() else ""
    return last_word in _DANGLING_WORDS


def _looks_like_entry_title(text: str) -> bool:
    """
    Whether a line reads as the title of an entry.

    Titles carry a date or a separator and do not end like prose. The
    prose test matters: "Built with React and deployed on Vercel in 2024."
    contains a date but is a description, and treating it as a title would
    split one project into two.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > _MAX_TITLE_CHARS:
        return False
    if stripped.endswith(_SENTENCE_END):
        return False
    return bool(_DATE_RE.search(stripped) or _TITLE_SEP_RE.search(stripped))


def _segment_entries(lines: List[str]) -> List[str]:
    """
    Split a section into entries using structure rather than keywords.

    Four independent signals mark a boundary, so no single layout convention is
    load-bearing — a résumé that uses none of them still degrades to one entry
    rather than to nonsense:

    1. a non-bullet line after a bullet (bullets belong to the entry above);
    2. a blank line, when blank lines are used *selectively* — double-spaced
       extraction separates every line and so carries no information;
    3. a title-shaped line (date or separator, not prose) following a line that
       is not title-shaped;
    4. block-level fallbacks for layouts with no internal structure at all:
       every line title-shaped → one entry per line; every line a bullet →
       one entry per bullet.

    `lines` may contain empty strings; they are the blank-line signal and are
    deliberately not filtered out before this point.
    """
    records: List[Dict[str, Any]] = []
    blank_pending = False
    for raw in lines:
        is_bullet = bool(_BULLET_RE.match(raw))
        text = _BULLET_RE.sub("", raw).strip() if is_bullet else raw.strip()
        if not text:
            blank_pending = True
            continue
        records.append({"text": text, "bullet": is_bullet, "blank_before": blank_pending})
        blank_pending = False

    if not records:
        return []

    body = records[1:]
    blank_informative = (
        any(r["blank_before"] for r in body)
        and not all(r["blank_before"] for r in body)
    )

    entries: List[str] = []
    current: List[str] = []
    prev: Optional[Dict[str, Any]] = None

    for rec in records:
        starts_entry = False
        if current and not rec["bullet"] and prev is not None:
            if not _is_continuation(rec["text"], prev["text"]):
                if prev["bullet"]:
                    starts_entry = True
                elif blank_informative and rec["blank_before"]:
                    starts_entry = True
                elif _looks_like_entry_title(rec["text"]) and not _looks_like_entry_title(prev["text"]):
                    # A title after a description opens the next entry. Guarding
                    # on the previous line keeps a company/role pair — where both
                    # lines look like titles — as a single entry.
                    starts_entry = True

        if starts_entry:
            entries.append("\n".join(current))
            current = []

        current.append(rec["text"])
        prev = rec

    if current:
        entries.append("\n".join(current))

    # Fallbacks for sections that offered no internal boundary at all.
    if len(entries) == 1 and len(records) > 1:
        if all(r["bullet"] for r in records):
            # Nothing to attach bullets to, so each bullet is its own entry.
            return [r["text"] for r in records]
        if not any(r["bullet"] for r in records) and all(
            _looks_like_entry_title(r["text"]) for r in records
        ):
            # "Portfolio Website | GitHub" one per line, no bullets, no dates.
            return [r["text"] for r in records]

    return entries


def derive_entry_title(content: str, max_words: int = 14) -> str:
    """
    Best-effort title for an entry, used to identify a project independently of
    the résumé it arrived in.

    The first chunk of a section carries its heading, which is skipped: the
    title of the first project is the project's name, not "Projects".
    """
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = classify_heading(line)
        if match and not match[1]:
            continue
        title = _TITLE_SEP_RE.split(line)[0].strip()
        title = _DATE_RE.sub("", title).strip(" -–—|,·")
        title = " ".join(title.split()[:max_words]).strip()
        return title or line.strip()
    return ""


class LongTermMemoryQdrant:
    """
    Long-term memory using Qdrant + Cohere + Chunking.
    Maintains API compatibility with ChromaDB implementation.
    """

    def __init__(self):
        """Initialize services and collection names."""
        self.qdrant = qdrant_service
        self.cohere = cohere_service
        self.chunker = chunking_service

        # Collection names
        self.collections = {
            "resume": "resume_chunks",
            "skills": "skills_chunks",
            "projects": "projects_chunks"
        }

    def _classify_chunk_types(self, chunk_text: str) -> List[str]:
        """Classify resume chunk into one or more specialized collections."""
        text = chunk_text.lower()

        skill_markers = ["skills", "technology", "technologies", "programming", "tools", "framework"]
        project_markers = ["project", "projects", "built", "developed", "internship", "experience"]
        chunk_types: List[str] = []

        if any(marker in text for marker in skill_markers):
            chunk_types.append("skills")
        if any(marker in text for marker in project_markers):
            chunk_types.append("projects")
        return chunk_types

    def _normalize_resume_text(self, text: str) -> str:
        """Normalize OCR/PDF text artifacts for more reliable parsing."""
        cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        # Collapse the extractor's bullet glyph zoo to a single marker. Doing it
        # once here means every downstream rule sees one shape, and the stored
        # text reads as bullets rather than replacement characters.
        cleaned = re.sub(
            f"^[{re.escape(_BULLET_CHARS)}]+[ \t]*",
            "• ",
            cleaned,
            flags=re.MULTILINE,
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _detect_name(self, lines: List[str]) -> Optional[str]:
        """Detect a likely name from top lines without hallucinating values."""
        for line in lines[:5]:
            # Strip leading/trailing punctuation
            line = line.strip(" -|:")
            if not line:
                continue
            # Strip phone numbers, emails, roll numbers appended to the name line
            # e.g. "Vansh Pratap Singh+91-6392306428" → "Vansh Pratap Singh"
            line = re.split(r"[+\|]|\s*\d{7,}", line)[0].strip(" -|:")
            if not line or len(line.split()) > 6:
                continue
            lower = line.lower()
            if any(x in lower for x in ["@", "http", "linkedin", "github", "phone", "email"]):
                continue
            if ":" in line:
                continue
            # A name always precedes the first section heading, so the search
            # stops there rather than skipping past it. A résumé whose name line
            # extracted as an image — common, PDF headers often render as
            # graphics — otherwise yields "SKILLS", or the first plausible-looking
            # line underneath it ("B.Tech"), as the user's identity: stored at
            # high importance and injected into every prompt thereafter.
            if classify_heading(line):
                break
            if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,80}", line):
                return line
        return None

    def _infer_type_from_text(self, text: str) -> str:
        """Infer semantic type from text when explicit section headers are missing."""
        lower = text.lower()
        if any(k in lower for k in ["skills", "technologies", "tools", "framework", "languages"]):
            return "skills"
        if any(k in lower for k in ["project", "built", "developed", "implemented", "designed"]):
            return "projects"
        if any(k in lower for k in ["experience", "worked", "intern", "employment", "role", "company"]):
            return "experience"
        if any(k in lower for k in ["education", "bachelor", "master", "university", "college", "cgpa", "gpa"]):
            return "education"
        if any(k in lower for k in ["achievement", "award", "winner", "certification", "certified"]):
            return "other"
        return "other"

    def _importance_for_type(self, section_type: str) -> str:
        if section_type in {"name", "skills", "projects", "experience", "education"}:
            return "high"
        return "medium"

    def _extract_semantic_resume_chunks(self, resume_text: str) -> List[Dict[str, Any]]:
        """
        Extract semantic resume chunks with robust fallback.
        Guarantees non-empty output for non-empty input and at least 3 chunks when possible.
        """
        cleaned = self._normalize_resume_text(resume_text)
        if not cleaned:
            return []

        lines = [ln.strip() for ln in cleaned.split("\n")]
        non_empty_lines = [ln for ln in lines if ln]
        chunks: List[Dict[str, Any]] = []

        # 1) Name detection
        detected_name = self._detect_name(non_empty_lines)
        if detected_name:
            chunks.append(
                {
                    "type": "name",
                    "content": detected_name,
                    "tags": ["identity"],
                    "importance": "high",
                }
            )

        # 2) Split the document into blocks at recognised section headings.
        #
        # Blocks rather than one buffer per section: a résumé may open the same
        # section twice, and the untyped sections ("Relevant Coursework",
        # "Summary") each need their own boundary. Document order is preserved,
        # which is what lets retrieve_resume reassemble a faithful résumé.
        blocks: List[Dict[str, Any]] = []
        current_block: Dict[str, Any] = {"section": "other", "heading": None, "lines": []}

        # Blank lines are carried into the block, not filtered out: they are the
        # only boundary a bullet-free, date-free résumé offers, and _segment_entries
        # reads them.
        for line in lines:
            if not line:
                current_block["lines"].append("")
                continue
            match = classify_heading(line)
            if match:
                section, inline = match
                if any(ln for ln in current_block["lines"]) or current_block["heading"]:
                    blocks.append(current_block)
                current_block = {"section": section, "heading": line.strip(), "lines": []}
                if inline:
                    # "SKILLS: Python, Java" is both boundary and content. Keep
                    # the whole line so the label stays attached to its values.
                    current_block["lines"].append(line.strip())
                    current_block["heading"] = None
                continue
            current_block["lines"].append(line)

        if current_block["lines"] or current_block["heading"]:
            blocks.append(current_block)

        # 3) Build chunks per block, in document order.
        section_tags = {
            "skills": ["skills", "technologies"],
            "projects": ["project"],
            "experience": ["experience", "work"],
            "education": ["education", "academics"],
            "achievements": ["achievements"],
            "other": ["general"],
        }

        for block in blocks:
            section = block["section"]
            block_lines = block["lines"]                       # blanks preserved
            lines = [ln for ln in block_lines if ln.strip()]
            if not lines:
                continue

            # Achievements have no dedicated collection, so they are stored as
            # `other` — tagged, so the distinction is not lost.
            chunk_type = "other" if section == "achievements" else section
            tags = section_tags.get(section, ["general"])
            importance = self._importance_for_type(chunk_type)

            if section == "skills":
                # A labelled section ("Backend: FastAPI, Node.js") is far more
                # useful split per label: a query about databases then matches
                # the database line instead of the whole wall of skills.
                labelled = [ln for ln in lines if _LABEL_RE.match(ln)]
                if labelled and len(labelled) * 2 >= len(lines):
                    entries = [ln.strip() for ln in lines]
                else:
                    raw = " ".join(lines)
                    tokens = [s.strip(" -•") for s in re.split(r"[,|/]", raw) if s.strip()]
                    entries = [", ".join(tokens)] if tokens else []
            elif section in {"projects", "experience"}:
                entries = _segment_entries(block_lines)
            else:
                entries = ["\n".join(_BULLET_RE.sub("", ln).strip() for ln in lines)]

            entries = [e.strip() for e in entries if e.strip()]
            if not entries:
                continue

            # The heading is carried on the first chunk of its block. Without it
            # the section boundary exists only in the original PDF: store_resume
            # persists chunk text, and retrieve_resume rebuilds the résumé by
            # joining those chunks — so a heading that was consumed as a
            # delimiter is gone for good, and re-parsing the stored résumé finds
            # no sections at all.
            if block["heading"]:
                entries[0] = f"{block['heading']}\n{entries[0]}"

            for entry in entries:
                chunks.append(
                    {
                        "type": chunk_type,
                        "content": entry,
                        "tags": list(tags),
                        "importance": importance,
                    }
                )

        # 4) Guaranteed fallback: if extraction is sparse, build paragraph blocks.
        # Do not infer new skills/projects from unlabeled text.
        meaningful_chunks = [c for c in chunks if c.get("content", "").strip()]
        if len(meaningful_chunks) < 3:
            paragraphs = [p.strip() for p in re.split(r"\n\n+", cleaned) if p.strip()]
            for para in paragraphs:
                if len(" ".join(para.split()).split()) < 8:
                    continue
                meaningful_chunks.append(
                    {
                        "type": "other",
                        "content": " ".join(para.split()),
                        "tags": ["fallback"],
                        "importance": "medium",
                    }
                )
                if len(meaningful_chunks) >= 5:
                    break

        # 5) Final fallback to token windows (100-150 words) as other.
        if len(meaningful_chunks) < 3:
            words = cleaned.split()
            window = 120
            step = 110
            for i in range(0, len(words), step):
                block = words[i:i + window]
                if not block:
                    break
                meaningful_chunks.append(
                    {
                        "type": "other",
                        "content": " ".join(block),
                        "tags": ["fallback"],
                        "importance": "medium",
                    }
                )
                if len(meaningful_chunks) >= 5:
                    break

        # De-duplicate very similar chunk content while preserving order.
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for chunk in meaningful_chunks:
            # Collapse horizontal whitespace only. Newlines separate a heading
            # from its content and one bullet from the next; flattening them
            # would put the heading and the first entry on one line, and the
            # stored résumé would no longer re-parse into sections.
            content = "\n".join(
                re.sub(r"[ \t]+", " ", ln).strip()
                for ln in chunk.get("content", "").split("\n")
            )
            content = re.sub(r"\n{2,}", "\n", content).strip()
            if not content:
                continue
            key = re.sub(r"\s+", " ", content).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "type": chunk.get("type", "other"),
                    "content": content,
                    "tags": chunk.get("tags", ["general"]),
                    "importance": chunk.get("importance", "medium"),
                }
            )

        return deduped

    async def initialize(self):
        """Initialize collections in Qdrant."""
        try:
            for collection_name in self.collections.values():
                await self.qdrant.ensure_collection(collection_name)
            logger.info("Long-term memory (Qdrant) initialized")
        except Exception as e:
            logger.error(f"Failed to initialize long-term memory: {str(e)}")
            raise

    async def store_resume(
        self,
        user_id: str,
        resume_text: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store resume information with chunking and embeddings.

        Args:
            user_id: User identifier
            resume_text: Full resume text
            metadata: Additional metadata

        Returns:
            Document ID (parent ID for all chunks)
        """
        try:
            parent_id = f"resume_{user_id}_{uuid.uuid4().hex[:8]}"

            cleaned_resume = self._normalize_resume_text(resume_text)
            if not cleaned_resume:
                logger.warning("Empty or malformed resume text received; skipping storage")
                return parent_id

            # Prepare metadata.
            # uploaded_at makes "which resume is newest" an explicit, sortable
            # fact. Without it retrieval had to guess from Qdrant's scroll order,
            # which reflects internal storage layout rather than upload recency —
            # so a partially-failed replacement could serve the stale resume.
            meta = metadata or {}
            meta.update({
                "user_id": user_id,
                "type": "resume",
                "parent_id": parent_id,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            })

            # Semantic extraction with guaranteed fallback for robust resume ingestion.
            semantic_chunks = self._extract_semantic_resume_chunks(cleaned_resume)

            if not semantic_chunks:
                logger.warning("Semantic extraction returned no chunks; using generic tokenizer chunking")
                generic_chunks = self.chunker.chunk_text(text=cleaned_resume, metadata=meta)
                semantic_chunks = [
                    {
                        "type": "other",
                        "content": chunk.text,
                        "tags": ["fallback"],
                        "importance": "medium",
                        "chunk_index": chunk.metadata.get("chunk_index", idx),
                    }
                    for idx, chunk in enumerate(generic_chunks)
                    if chunk.text.strip()
                ]

            if not semantic_chunks:
                logger.warning("No chunks created from resume text after fallback")
                return parent_id

            # Generate embeddings for all chunks
            chunk_texts = [chunk["content"] for chunk in semantic_chunks]
            embeddings = await self.cohere.embed_batch(
                texts=chunk_texts,
                input_type="search_document"
            )

            # Guard: if Cohere returns a partial batch, zip() would silently
            # drop trailing chunks — better to abort and let the caller retry.
            if len(embeddings) != len(chunk_texts):
                raise ValueError(
                    f"Embedding batch size mismatch for resume: "
                    f"expected {len(chunk_texts)}, got {len(embeddings)}. "
                    f"Aborting to prevent partial write."
                )

            # Create Qdrant points
            resume_points = []
            skills_points = []
            projects_points = []
            # Each project is its own entity, identified independently of the
            # résumé upload it arrived in. Grouping by parent_id made every
            # project from one upload behave as a single document, so a query
            # about one project returned all of them concatenated and ranking
            # between projects was impossible.
            project_ordinal = 0
            for idx, (chunk, embedding) in enumerate(zip(semantic_chunks, embeddings)):
                # Generate UUID for Qdrant compatibility
                point_id = str(uuid.uuid4())
                section_type = chunk.get("type", "other")
                resume_payload = {
                    **meta,
                    "type": "resume",
                    "semantic_type": section_type,
                    "importance": chunk.get("importance", "medium"),
                    "tags": chunk.get("tags", []),
                    "text": chunk["content"],
                    "chunk_index": idx,
                    "total_chunks": len(semantic_chunks),
                    "string_id": f"{parent_id}_chunk_{idx}"
                }
                resume_points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=resume_payload
                ))

                # Also route relevant chunks to specialized collections for strict retrieval.
                specialized_types: List[str] = []
                if section_type == "skills":
                    specialized_types.append("skills")
                if section_type == "projects":
                    specialized_types.append("projects")
                if "skills" in specialized_types:
                    skills_points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                **meta,
                                "type": "skills",
                                "semantic_type": section_type,
                                "importance": chunk.get("importance", "medium"),
                                "tags": chunk.get("tags", []),
                                "text": chunk["content"],
                                "chunk_index": idx,
                                "total_chunks": len(semantic_chunks),
                                "string_id": f"skills_{parent_id}_chunk_{idx}",
                            },
                        )
                    )
                if "projects" in specialized_types:
                    project_ordinal += 1
                    # entity_id is scoped to the parent, so re-uploading the same
                    # résumé reproduces the same ids, and provenance back to the
                    # source document is preserved by parent_id alongside it.
                    # Nothing is duplicated: these are two payload fields on the
                    # points that already existed.
                    entity_id = f"{parent_id}_project_{project_ordinal}"
                    projects_points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                **meta,
                                "type": "projects",
                                "semantic_type": section_type,
                                "importance": chunk.get("importance", "medium"),
                                "tags": chunk.get("tags", []),
                                "text": chunk["content"],
                                "chunk_index": idx,
                                "total_chunks": len(semantic_chunks),
                                "string_id": f"projects_{parent_id}_chunk_{idx}",
                                "entity_id": entity_id,
                                "entity_title": derive_entry_title(chunk["content"]),
                            },
                        )
                    )

            # Collect stale point IDs BEFORE inserting new data.
            # Pattern: upsert new → delete old (readers always see at least one
            # version, never a zero-data gap).
            #
            # CRITICAL: we also scroll old resume chunks so they don't
            # accumulate across uploads — without this, retrieve_resume()
            # picks an arbitrary chunk group, not necessarily the latest one.
            old_resume_points, old_skills_points, old_projects_points = await asyncio.gather(
                self.qdrant.scroll_collection(
                    collection_name=self.collections["resume"],
                    filter_conditions={"user_id": user_id},
                ),
                self.qdrant.scroll_collection(
                    collection_name=self.collections["skills"],
                    filter_conditions={"user_id": user_id},
                ),
                self.qdrant.scroll_collection(
                    collection_name=self.collections["projects"],
                    filter_conditions={"user_id": user_id},
                ),
            )
            old_resume_ids = [p["id"] for p in old_resume_points]
            old_skills_ids = [p["id"] for p in old_skills_points]
            old_projects_ids = [p["id"] for p in old_projects_points]

            # Upsert new data first so readers are never left with nothing.
            await self.qdrant.upsert_points(
                collection_name=self.collections["resume"],
                points=resume_points
            )

            if skills_points:
                await self.qdrant.upsert_points(
                    collection_name=self.collections["skills"],
                    points=skills_points,
                )

            if projects_points:
                await self.qdrant.upsert_points(
                    collection_name=self.collections["projects"],
                    points=projects_points,
                )

            # Delete stale points AFTER the new data is live.
            #
            # CRITICAL FIX: only delete old skills/projects when we actually
            # upserted replacements. If the new resume has no skills section,
            # skills_points is empty — deleting old_skills_ids without a
            # replacement would permanently destroy the user's skills data.
            if old_resume_ids:
                await self.qdrant.delete_points(
                    collection_name=self.collections["resume"],
                    point_ids=old_resume_ids,
                )
            if skills_points and old_skills_ids:
                await self.qdrant.delete_points(
                    collection_name=self.collections["skills"],
                    point_ids=old_skills_ids,
                )
            if projects_points and old_projects_ids:
                await self.qdrant.delete_points(
                    collection_name=self.collections["projects"],
                    point_ids=old_projects_ids,
                )

            logger.info(
                f"Stored resume for user '{user_id}' as {len(semantic_chunks)} chunks; "
                f"skills_chunks={len(skills_points)}, projects_chunks={len(projects_points)}"
            )
            return parent_id

        except Exception as e:
            logger.error(f"Failed to store resume: {str(e)}")
            raise

    async def store_skill(
        self,
        user_id: str,
        skill_name: str,
        skill_level: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a skill.

        Args:
            user_id: User identifier
            skill_name: Name of the skill
            skill_level: Proficiency level
            metadata: Additional metadata

        Returns:
            Document ID
        """
        try:
            # Generate UUID for Qdrant compatibility
            point_id = str(uuid.uuid4())
            string_id = f"skill_{user_id}_{uuid.uuid4().hex[:8]}"

            # Format skill document
            document = f"{skill_name}: {skill_level}"
            if metadata and "description" in metadata:
                document += f" - {metadata['description']}"

            # Prepare metadata
            meta = metadata or {}
            meta.update({
                "user_id": user_id,
                "skill_name": skill_name,
                "skill_level": skill_level,
                "type": "skills",
                "parent_id": string_id,
                "string_id": string_id
            })

            # Generate embedding (skills are usually short, single chunk)
            embedding = await self.cohere.embed_text(
                text=document,
                input_type="search_document"
            )

            # Create Qdrant point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    **meta,
                    "text": document
                }
            )

            # Upsert to Qdrant
            await self.qdrant.upsert_points(
                collection_name=self.collections["skills"],
                points=[point]
            )

            logger.info(f"Stored skill '{skill_name}' for user '{user_id}'")
            return string_id

        except Exception as e:
            logger.error(f"Failed to store skill: {str(e)}")
            raise

    async def store_project(
        self,
        user_id: str,
        project_name: str,
        description: str,
        technologies: List[str],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store project information.

        Args:
            user_id: User identifier
            project_name: Project name
            description: Project description
            technologies: List of technologies used
            metadata: Additional metadata

        Returns:
            Document ID (parent ID for chunks)
        """
        try:
            parent_id = f"project_{user_id}_{uuid.uuid4().hex[:8]}"

            # Format project document
            document = (
                f"Project: {project_name}\n"
                f"Description: {description}\n"
                f"Technologies: {', '.join(technologies)}"
            )

            # Prepare metadata
            meta = metadata or {}
            meta.update({
                "user_id": user_id,
                "project_name": project_name,
                "technologies": ",".join(technologies),
                "type": "projects",
                "parent_id": parent_id
            })

            # Chunk the project description (if long)
            chunks = self.chunker.chunk_text(
                text=document,
                metadata=meta
            )

            if not chunks:
                logger.warning("No chunks created from project")
                return parent_id

            # Generate embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self.cohere.embed_batch(
                texts=chunk_texts,
                input_type="search_document"
            )

            if len(embeddings) != len(chunk_texts):
                raise ValueError(
                    f"Embedding batch size mismatch for project '{project_name}': "
                    f"expected {len(chunk_texts)}, got {len(embeddings)}. "
                    f"Aborting to prevent partial write."
                )

            # Create Qdrant points
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                # Generate UUID for Qdrant compatibility
                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        **chunk.metadata,
                        "text": chunk.text,
                        "string_id": f"{parent_id}_chunk_{chunk.metadata['chunk_index']}"
                    }
                ))

            # Upsert to Qdrant
            await self.qdrant.upsert_points(
                collection_name=self.collections["projects"],
                points=points
            )

            logger.info(
                f"Stored project '{project_name}' as {len(chunks)} chunks"
            )
            return parent_id

        except Exception as e:
            logger.error(f"Failed to store project: {str(e)}")
            raise

    async def retrieve_resume(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent resume for a user.

        Args:
            user_id: User identifier

        Returns:
            Resume content and metadata, or None
        """
        try:
            logger.debug("retrieve_resume called for user_id=%s", user_id)

            # Walk every resume chunk for this user (scroll cursor followed to
            # exhaustion) so a large or multi-version resume is never truncated.
            results = await self.qdrant.scroll_collection(
                collection_name=self.collections["resume"],
                filter_conditions={"user_id": user_id},
            )

            if not results:
                return None

            # Group by parent_id; each group is one uploaded resume version.
            parent_groups: Dict[Any, List[Dict[str, Any]]] = {}
            for point in results:
                parent_id = point["payload"].get("parent_id")
                parent_groups.setdefault(parent_id, []).append(point)

            # Pick the newest version by explicit upload timestamp. Chunks
            # written before uploaded_at existed sort as empty string, so any
            # timestamped version correctly wins over legacy data; if none are
            # timestamped this degrades to stable insertion order.
            def _group_uploaded_at(points: List[Dict[str, Any]]) -> str:
                return max(
                    (str(p["payload"].get("uploaded_at") or "") for p in points),
                    default="",
                )

            latest_parent = max(parent_groups, key=lambda pid: _group_uploaded_at(parent_groups[pid]))
            chunks = parent_groups[latest_parent]

            if len(parent_groups) > 1:
                logger.warning(
                    "Found %d resume versions for user=%s; serving newest (parent_id=%s). "
                    "Stale versions suggest a previous replacement did not finish cleanly.",
                    len(parent_groups), user_id, latest_parent,
                )

            # Sort chunks by index
            sorted_chunks = sorted(
                chunks,
                key=lambda x: x["payload"].get("chunk_index", 0)
            )

            # Reconstruct text
            full_text = "\n\n".join(
                chunk["payload"]["text"] for chunk in sorted_chunks
            )

            # The name, from the chunk ingestion already classified as one.
            #
            # Callers have always asked for `resume_data["name"]` and always
            # received None, because this method never produced the key — so
            # "what is my name?" had no source at all, and the answer came from
            # whatever else happened to be in context. The chunk exists
            # (semantic_type="name", tagged identity); it simply was not read.
            #
            # Re-parsing the reconstructed text is the fallback for résumés
            # ingested before that classification existed.
            name = next(
                (
                    chunk["payload"]["text"].strip()
                    for chunk in sorted_chunks
                    if chunk["payload"].get("semantic_type") == "name"
                    and (chunk["payload"].get("text") or "").strip()
                ),
                None,
            )
            if not name:
                name = self._detect_name(full_text.splitlines())

            return {
                "content": full_text,
                "name": name,
                "metadata": sorted_chunks[0]["payload"],
            }

        except Exception as e:
            logger.error(f"Failed to retrieve resume: {str(e)}")
            return None

    async def retrieve_skills(
        self,
        user_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> RetrievalResult:
        """
        Retrieve skills for a user.

        Args:
            user_id: User identifier
            query: Optional semantic search query
            limit: Maximum number of results

        Returns:
            RetrievalResult whose status distinguishes "nothing stored"
            (NO_DATA) from "lookup failed" (ERROR).
        """
        try:
            log_step("RETRIEVAL FILTER", {"user_id": user_id, "type": "skills"})

            if query:
                # Semantic search
                query_embedding = await self.cohere.embed_text(
                    text=query,
                    input_type="search_query"
                )

                log_step("EMBEDDING DONE", {"input_type": "search_query", "target": "skills"})

                results = await self.qdrant.query_points(
                    collection_name=self.collections["skills"],
                    query_vector=query_embedding,
                    limit=limit,
                    filter_conditions={"user_id": user_id, "type": "skills"}
                )

                if not results:
                    return RetrievalResult.no_data()

                return RetrievalResult.ok([
                    {
                        "content": result.payload["text"],
                        "metadata": result.payload,
                        "score": result.score
                    }
                    for result in results
                ])
            else:
                # Get all skills from skills_chunks
                results = await self.qdrant.scroll_collection(
                    collection_name=self.collections["skills"],
                    filter_conditions={"user_id": user_id, "type": "skills"},
                    limit=limit
                )

                if not results:
                    return RetrievalResult.no_data()

                return RetrievalResult.ok([
                    {
                        "content": point["payload"]["text"],
                        "metadata": point["payload"]
                    }
                    for point in results
                ])

        except Exception as e:
            # ERROR, not an empty list: an empty list would be reported upstream
            # as "this user has no skills", which is a claim we cannot support
            # after a failed lookup.
            logger.error(f"Failed to retrieve skills: {str(e)}")
            return RetrievalResult.error()

    async def retrieve_projects(
        self,
        user_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> RetrievalResult:
        """
        Retrieve projects for a user.

        Args:
            user_id: User identifier
            query: Optional semantic search query
            limit: Maximum number of results

        Returns:
            RetrievalResult whose status distinguishes "nothing stored"
            (NO_DATA) from "lookup failed" (ERROR).
        """
        try:
            log_step("RETRIEVAL FILTER", {"user_id": user_id, "type": "projects"})

            if query:
                # Semantic search
                query_embedding = await self.cohere.embed_text(
                    text=query,
                    input_type="search_query"
                )

                log_step("EMBEDDING DONE", {"input_type": "search_query", "target": "projects"})

                results = await self.qdrant.query_points(
                    collection_name=self.collections["projects"],
                    query_vector=query_embedding,
                    limit=limit * 3,  # Get more chunks, then group
                    # No score floor, matching retrieve_skills. The search is
                    # already constrained to this user's project chunks, so the
                    # worst hit is still one of their projects and ranking is
                    # what matters. A 0.3 floor made the specialised collection
                    # *stricter* than the resume_chunks fallback (0.25) it is
                    # meant to outrank: real project chunks scored 0.26-0.29
                    # against "projects built developed", so this path returned
                    # nothing and every projects query silently degraded to the
                    # fallback — or, for the profile agent, to "I don't have
                    # information about your projects".
                    filter_conditions={"user_id": user_id, "type": "projects"}
                )

                if not results:
                    return RetrievalResult.no_data()

                # Group by project, not by upload. `entity_id` identifies one
                # project; `parent_id` is the fallback for points written before
                # entity ids existed, which preserves the old behaviour for
                # legacy data rather than dropping it.
                entity_groups: Dict[Any, List[Any]] = {}
                for result in results:
                    key = result.payload.get("entity_id") or result.payload.get("parent_id")
                    entity_groups.setdefault(key, []).append(result)

                projects = []
                for key, group in entity_groups.items():
                    ordered = sorted(group, key=lambda x: x.payload.get("chunk_index", 0))
                    projects.append({
                        "content": "\n\n".join(c.payload["text"] for c in ordered),
                        "metadata": ordered[0].payload,
                        "title": ordered[0].payload.get("entity_title", ""),
                        "score": max(c.score for c in group),
                    })

                # Rank projects against each other before truncating. Taking
                # dict order and slicing — as this did — returned whichever
                # projects Qdrant happened to list first, so "tell me about
                # TRACE" was not answered with TRACE.
                projects.sort(key=lambda p: p["score"], reverse=True)
                return RetrievalResult.ok(projects[:limit])

            else:
                # Get all projects from projects_chunks
                results = await self.qdrant.scroll_collection(
                    collection_name=self.collections["projects"],
                    filter_conditions={"user_id": user_id, "type": "projects"},
                    limit=limit * 10  # Get more to account for chunks
                )

                if not results:
                    return RetrievalResult.no_data()

                # Unqueried listing: one item per project, in document order so
                # the résumé's own ordering survives.
                entity_groups: Dict[Any, List[Any]] = {}
                for point in results:
                    key = point["payload"].get("entity_id") or point["payload"].get("parent_id")
                    entity_groups.setdefault(key, []).append(point)

                projects = []
                for key, group in entity_groups.items():
                    ordered = sorted(group, key=lambda x: x["payload"].get("chunk_index", 0))
                    projects.append({
                        "content": "\n\n".join(c["payload"]["text"] for c in ordered),
                        "metadata": ordered[0]["payload"],
                        "title": ordered[0]["payload"].get("entity_title", ""),
                        "_order": ordered[0]["payload"].get("chunk_index", 0),
                    })

                projects.sort(key=lambda p: p.pop("_order"))
                return RetrievalResult.ok(projects[:limit])

        except Exception as e:
            # ERROR, not an empty list — see retrieve_skills().
            logger.error(f"Failed to retrieve projects: {str(e)}")
            return RetrievalResult.error()

    # Typed résumé sections that have no dedicated collection. Skills and
    # projects have one; experience, education and achievements are stored in
    # resume_chunks with a `semantic_type` payload field and were, until now,
    # reachable only through the first 1500 characters of the whole résumé.
    #
    # Achievements are stored as `other` (the schema predates the section) and
    # are identified by their tag, so the distinction survives.
    _SECTION_FILTERS: Dict[str, Dict[str, Any]] = {
        "experience": {"semantic_type": "experience"},
        "education": {"semantic_type": "education"},
        "achievements": {"semantic_type": "other", "tag": "achievements"},
        # The name is chunked and stored at ingestion with semantic_type="name",
        # and nothing read it back. `retrieve_resume` returned only content and
        # metadata, so every caller doing `resume_data.get("name")` — the
        # profile summary among them — got None from a store that held
        # "Vansh Pratap Singh" the whole time.
        "name": {"semantic_type": "name"},
    }

    async def retrieve_section(
        self,
        user_id: str,
        section: str,
        limit: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve one typed résumé section in document order.

        Scroll-and-filter rather than a filtered vector search: `semantic_type`
        carries no payload index, and Qdrant rejects a filter on an unindexed
        field with a 400 rather than falling back to a scan. `user_id` is
        indexed, a résumé is a couple of dozen chunks, and a question like
        "what is my CGPA" wants the whole education section rather than a fuzzy
        top-k — so this is both cheaper and more accurate than embedding the
        query would be.

        Returns a RetrievalResult so callers keep the NO_DATA/ERROR distinction:
        "nothing is stored" and "the lookup failed" must not look alike.
        """
        spec = self._SECTION_FILTERS.get(section)
        if spec is None:
            logger.warning("Unknown résumé section requested: %s", section)
            return RetrievalResult.no_data()

        try:
            log_step("RETRIEVAL FILTER", {"user_id": user_id, "type": section})

            points = await self.qdrant.scroll_collection(
                collection_name=self.collections["resume"],
                filter_conditions={"user_id": user_id},
            )
            if not points:
                return RetrievalResult.no_data()

            # Serve only the newest résumé; stale versions would otherwise be
            # mixed into the answer.
            latest_parent = self._latest_parent_id(points)

            matches = []
            for point in points:
                payload = point.get("payload", {})
                if latest_parent and payload.get("parent_id") != latest_parent:
                    continue
                if payload.get("semantic_type") != spec["semantic_type"]:
                    continue
                if spec.get("tag") and spec["tag"] not in (payload.get("tags") or []):
                    continue
                if not (payload.get("text") or "").strip():
                    continue
                matches.append(payload)

            if not matches:
                return RetrievalResult.no_data()

            matches.sort(key=lambda p: p.get("chunk_index", 0))
            return RetrievalResult.ok([
                {
                    "content": payload["text"],
                    "metadata": payload,
                    "section": section,
                }
                for payload in matches[:limit]
            ])
        except Exception as e:
            logger.error("Section retrieval failed for %s: %s", section, e)
            return RetrievalResult.error()

    @staticmethod
    def _latest_parent_id(points: List[Dict[str, Any]]) -> Optional[str]:
        """Parent id of the most recently uploaded résumé among these points."""
        newest_at, newest_parent = "", None
        for point in points:
            payload = point.get("payload", {})
            uploaded_at = str(payload.get("uploaded_at") or "")
            if uploaded_at >= newest_at:
                newest_at, newest_parent = uploaded_at, payload.get("parent_id")
        return newest_parent

    async def _fallback_resume_search(
        self,
        user_id: str,
        query: str,
        semantic_type: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback: semantic search on resume_chunks when dedicated collection is empty.
        Filters only by user_id (no semantic_type index required on existing collections).
        """
        try:
            query_embedding = await self.cohere.embed_text(
                text=query,
                input_type="search_query"
            )
            results = await self.qdrant.query_points(
                collection_name=self.collections["resume"],
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.25,
                filter_conditions={"user_id": user_id}
            )
            if not results:
                return []
            return [
                {
                    "content": r.payload["text"],
                    "metadata": r.payload,
                    "score": r.score,
                    "source": "resume_fallback"
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Fallback resume search failed for {semantic_type}: {str(e)}")
            return []

    async def search_all(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        sections: Optional[AbstractSet[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search across all collections for relevant information.
        Falls back to resume_chunks when dedicated collections are empty.

        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results per collection
            sections: Which prompt sections this question can actually use, as
                `MemoryManager.sections_for` computed them. A lookup whose
                section is not in the set is skipped rather than performed and
                discarded — `retrieve_skills` and `retrieve_projects` are a
                Cohere embedding and a Qdrant query each, and `retrieve_resume`
                walks every résumé chunk. None means "all", the historical
                behaviour.

        Returns:
            Dictionary with results from each collection. `skills` and
            `projects` are plain lists (the prompt formatter consumes them
            directly); the accompanying `*_status` keys carry the provenance
            of each lookup.
        """
        try:
            log_step("USER QUERY", {"user_id": user_id, "query": query})

            want_skills = sections is None or "skills" in sections
            want_projects = sections is None or "projects" in sections
            want_resume = sections is None or "resume" in sections

            async def _skip_result() -> RetrievalResult:
                # NO_DATA, not ERROR: nothing failed. The section simply cannot
                # appear in this question's prompt, and NO_DATA is the status
                # the formatter already renders as absent rather than as broken.
                return RetrievalResult.no_data()

            async def _skip_resume():
                return None

            # Run all three primary lookups in parallel (~200ms gain)
            skills, projects, resume = await asyncio.gather(
                self.retrieve_skills(user_id, query, limit)
                if want_skills else _skip_result(),
                self.retrieve_projects(user_id, query, limit)
                if want_projects else _skip_result(),
                self.retrieve_resume(user_id) if want_resume else _skip_resume(),
            )

            # A skipped lookup must not trigger the résumé fallback below —
            # that would reinstate the cost this parameter exists to avoid.
            if not want_skills:
                skills = RetrievalResult.ok([])
            if not want_projects:
                projects = RetrievalResult.ok([])

            # Fall back to resume chunks only when the dedicated collection is
            # genuinely empty. A failed lookup (ERROR) is deliberately *not*
            # retried here: reporting its result as FALLBACK would claim the
            # dedicated collection was empty, which we do not know.
            needs_skills_fallback = skills.status is RetrievalStatus.NO_DATA
            needs_projects_fallback = projects.status is RetrievalStatus.NO_DATA

            skills_result, projects_result = skills, projects

            if needs_skills_fallback or needs_projects_fallback:
                fallback_tasks = []
                if needs_skills_fallback:
                    fallback_tasks.append(
                        self._fallback_resume_search(user_id, query, "skills", limit)
                    )
                if needs_projects_fallback:
                    fallback_tasks.append(
                        self._fallback_resume_search(user_id, query, "projects", limit)
                    )
                fallback_results = await asyncio.gather(*fallback_tasks)

                idx = 0
                if needs_skills_fallback:
                    recovered = fallback_results[idx]; idx += 1
                    skills_result = (
                        RetrievalResult.fallback(recovered)
                        if recovered else RetrievalResult.no_data()
                    )
                if needs_projects_fallback:
                    recovered = fallback_results[idx]
                    projects_result = (
                        RetrievalResult.fallback(recovered)
                        if recovered else RetrievalResult.no_data()
                    )

            return {
                "resume": resume or {},
                "skills": list(skills_result),
                "projects": list(projects_result),
                "skills_status": skills_result.status.value,
                "projects_status": projects_result.status.value,
            }

        except Exception as e:
            # Report ERROR rather than omitting the status keys. Omitting them
            # made a total search failure indistinguishable from a user with no
            # stored data, so the prompt carried neither the "no data" hint nor
            # the refusal policy — the exact state in which a model invents an
            # answer.
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            return {
                "resume": {},
                "skills": [],
                "projects": [],
                "skills_status": RetrievalStatus.ERROR.value,
                "projects_status": RetrievalStatus.ERROR.value,
            }


# Singleton instance
long_term_memory_qdrant = LongTermMemoryQdrant()
