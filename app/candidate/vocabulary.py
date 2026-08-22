"""
The names of things, and how to find them in text a PDF mangled.

Two problems this solves, both discovered by reading real stored résumé text
rather than by imagining what one looks like.

**The same skill has many surface forms.** A résumé says `React.js`, a posting
says `ReactJS`, a recruiter says `React`. Matching them as strings produces a
gap that does not exist, and a candidate who looks unqualified for a job they
could do. So every skill has one canonical name and a set of aliases, and
everything downstream — evidence, gaps, explanations — speaks canonical.

**PDF extraction glues words together.** This is real text from the stored
résumé:

    Built a multi-agent AI voice assistant usingFastAPIandLangGraph.

There is no word boundary before `FastAPI`. Tokenising finds nothing;
`\\bFastAPI\\b` finds nothing. A naive substring search finds it — and also
finds `Go` inside `going`, `R` inside every word containing the letter, and
`AI` inside `said`.

So `find_skills` uses two rules, and the second is the interesting one:

    word-boundary, case-insensitive   — normal prose, "built with React"
    exact-case substring              — glued text, "usingFastAPIand"

Exact case is what makes the second rule safe. `FastAPI`, `LangGraph`,
`PostgreSQL` and `Next.js` are capitalised distinctively enough that finding
that exact byte sequence inside a word is overwhelmingly a real mention, while
`react` inside `reacted` is lowercase and does not match `React`. Skills whose
canonical form is short or is an ordinary English word are marked
`boundary_only` and never use the substring rule at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Skill:
    """One canonical skill and everything it might be written as."""

    canonical: str
    aliases: Tuple[str, ...] = ()
    category: str = ""
    boundary_only: bool = False
    """Never matched as a bare substring.

    Set for names that are short, or that are ordinary words: `Go`, `R`, `C`,
    `AI`. Without it, `Go` matches `going`, `algorithm` and `Django`."""

    def surfaces(self) -> Tuple[str, ...]:
        return (self.canonical, *self.aliases)


def _s(canonical, *aliases, category="", boundary_only=False) -> Skill:
    return Skill(canonical, tuple(aliases), category, boundary_only)


# The vocabulary. Deliberately curated rather than generated: an unknown token
# is better dropped than turned into a claim, and a name in this list is a name
# the system is willing to assert.
SKILLS: Tuple[Skill, ...] = (
    # ── Languages ────────────────────────────────────────────────────────────
    _s("Python", "py", "python3", category="Languages"),
    _s("JavaScript", "js", "ecmascript", "es6", "es6+", category="Languages"),
    _s("TypeScript", "ts", category="Languages"),
    _s("SQL", category="Languages", boundary_only=True),
    _s("C++", "cpp", category="Languages", boundary_only=True),
    _s("C", category="Languages", boundary_only=True),
    _s("Java", category="Languages", boundary_only=True),
    _s("Go", "golang", category="Languages", boundary_only=True),
    _s("Rust", category="Languages", boundary_only=True),
    _s("R", category="Languages", boundary_only=True),
    _s("Bash", "shell scripting", "shell", category="Languages"),
    # ── Frontend ─────────────────────────────────────────────────────────────
    _s("React", "react.js", "reactjs", category="Frontend"),
    _s("Next.js", "nextjs", "next js", category="Frontend"),
    _s("HTML", "html5", category="Frontend", boundary_only=True),
    _s("CSS", "css3", category="Frontend", boundary_only=True),
    _s("Tailwind CSS", "tailwind", "tailwindcss", category="Frontend"),
    _s("Redux", category="Frontend"),
    # ── Backend ──────────────────────────────────────────────────────────────
    _s("FastAPI", "fast api", category="Backend"),
    _s("Node.js", "nodejs", "node", category="Backend"),
    _s("Express.js", "express", "expressjs", category="Backend"),
    _s("Django", category="Backend"),
    _s("Flask", category="Backend"),
    _s("REST APIs", "rest", "rest api", "restful", category="Backend"),
    _s("GraphQL", category="Backend"),
    _s("WebSockets", "websocket", category="Backend"),
    _s("gRPC", category="Backend"),
    # ── Data ─────────────────────────────────────────────────────────────────
    _s("PostgreSQL", "postgres", "psql", category="Databases"),
    _s("MySQL", category="Databases"),
    _s("MongoDB", "mongo", category="Databases"),
    _s("Redis", category="Databases"),
    _s("Qdrant", category="Databases"),
    _s("Neo4j", category="Databases"),
    _s("Pinecone", category="Databases"),
    _s("Prisma ORM", "prisma", category="Databases"),
    _s("SQLAlchemy", category="Databases"),
    _s("Elasticsearch", "elastic search", category="Databases"),
    # ── Cloud / infra ────────────────────────────────────────────────────────
    _s("AWS", "amazon web services", category="Cloud", boundary_only=True),
    _s("GCP", "google cloud", category="Cloud", boundary_only=True),
    _s("Azure", category="Cloud"),
    _s("Docker", category="Cloud"),
    _s("Kubernetes", "k8s", category="Cloud"),
    _s("Terraform", category="Cloud"),
    _s("CI/CD", "cicd", "continuous integration", category="Cloud"),
    _s("GitHub Actions", category="Cloud"),
    _s("Linux", category="Cloud"),
    _s("Nginx", category="Cloud"),
    # ── AI / ML ──────────────────────────────────────────────────────────────
    _s("Machine Learning", "ml", category="AI"),
    _s("Deep Learning", category="AI"),
    _s("NLP", "natural language processing", category="AI", boundary_only=True),
    _s("LangChain", category="AI"),
    _s("LangGraph", category="AI"),
    _s("RAG", "retrieval augmented generation", "retrieval-augmented generation",
       category="AI", boundary_only=True),
    _s("LLM", "large language model", "llms", category="AI", boundary_only=True),
    _s("PyTorch", "torch", category="AI"),
    _s("TensorFlow", category="AI"),
    _s("Scikit-learn", "sklearn", "scikit learn", category="AI"),
    _s("Pandas", category="AI"),
    _s("NumPy", "numpy", category="AI"),
    _s("Hugging Face", "huggingface", category="AI"),
    _s("Vector Search", "vector database", "embeddings", category="AI"),
    _s("Groq API", "groq", category="AI"),
    _s("OpenAI API", "openai", category="AI"),
    # ── Security / auth ──────────────────────────────────────────────────────
    _s("JWT", "json web token", category="Security", boundary_only=True),
    _s("OAuth 2.0", "oauth2", "oauth", category="Security"),
    _s("RBAC", "role based access control", category="Security", boundary_only=True),
    _s("Rate Limiting", category="Security"),
    # ── Practice ─────────────────────────────────────────────────────────────
    _s("Git", category="Tools", boundary_only=True),
    _s("Postman", category="Tools"),
    _s("Vercel", category="Tools"),
    _s("Render", category="Tools", boundary_only=True),
    _s("Agile", "scrum", category="Practice"),
    _s("Testing", "unit testing", "pytest", "jest", category="Practice"),
    _s("System Design", category="Practice"),
    _s("Data Structures", "dsa", "data structures and algorithms", category="Practice"),
    _s("Algorithms", category="Practice"),
    _s("Object-Oriented Programming", "oop", category="Practice"),
    _s("Operating Systems", category="Practice"),
    _s("Computer Networks", "networking", category="Practice"),
    _s("DBMS", "database management systems", category="Practice", boundary_only=True),
    _s("Microservices", category="Practice"),
)


# alias (lowercased) → canonical
_BY_SURFACE: Dict[str, Skill] = {}
for _skill in SKILLS:
    for _surface in _skill.surfaces():
        _BY_SURFACE.setdefault(_surface.strip().lower(), _skill)

_BY_CANONICAL: Dict[str, Skill] = {s.canonical.lower(): s for s in SKILLS}

# Longest surfaces first, so "Next.js" is matched before "Node.js" cannot steal
# it and "Machine Learning" wins over a bare "ML" hidden inside it.
_SEARCH_ORDER: Tuple[Tuple[str, Skill], ...] = tuple(
    sorted(_BY_SURFACE.items(), key=lambda kv: len(kv[0]), reverse=True)
)


def canonical_for(text: str) -> Optional[Skill]:
    """The skill this exact surface form names, if any."""
    return _BY_SURFACE.get((text or "").strip().lower())


def is_known(text: str) -> bool:
    return canonical_for(text) is not None


def all_canonical_names() -> Tuple[str, ...]:
    return tuple(s.canonical for s in SKILLS)


# ── Detection ────────────────────────────────────────────────────────────────

_WORD_CACHE: Dict[str, re.Pattern] = {}


def _boundary_pattern(surface: str) -> re.Pattern:
    """
    Word-boundary matcher for one surface form.

    `\\b` is wrong at the edges of names like `C++` and `Next.js`, whose first
    or last character is not a word character — Python would demand a word
    character adjacent to it and never match. Lookarounds on *alphanumeric*
    neighbours give the intended meaning: not embedded in a longer token.
    """
    cached = _WORD_CACHE.get(surface)
    if cached is None:
        escaped = re.escape(surface)
        cached = re.compile(
            rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE
        )
        _WORD_CACHE[surface] = cached
    return cached


def find_skills(text: str) -> List[Tuple[Skill, str]]:
    """
    Every known skill named in this text, with the surface form that matched.

    Two rules, in order, and a skill is claimed if either fires:

    1. **Word-boundary, case-insensitive** — ordinary prose. "built with react"
       matches `React`.
    2. **Exact-case substring** — glued PDF text. `usingFastAPIandLangGraph`
       contains the literal bytes `FastAPI`, and matching case exactly is what
       keeps `react` inside `reacted` from counting.

    Rule 2 is skipped for `boundary_only` skills, where a substring hit is far
    more likely to be a coincidence than a mention.
    """
    if not text:
        return []

    found: Dict[str, Tuple[Skill, str]] = {}
    for surface, skill in _SEARCH_ORDER:
        if skill.canonical.lower() in found:
            continue

        if _boundary_pattern(surface).search(text):
            found[skill.canonical.lower()] = (skill, surface)
            continue

        if skill.boundary_only:
            continue

        # Exact-case substring: the glued-text case. Only the surface as it is
        # conventionally written, so a lowercase mention in running prose does
        # not slip through the boundary rule's rejection.
        for written in skill.surfaces():
            if len(written) >= 4 and written in text:
                found[skill.canonical.lower()] = (skill, written)
                break

    return list(found.values())


# ── Skill-line parsing ───────────────────────────────────────────────────────

# "Programming Languages:C/C++, Python" — the label and its values. The colon
# may have no space after it; the extractor emits exactly that.
_LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 &/+.#-]{1,40}?)\s*:\s*(.+)$")


def split_skill_line(line: str) -> Tuple[str, List[str]]:
    """
    Split a résumé skills line into its category and its individual skills.

    Parenthesis-aware, which the previous splitter was not: `AWS (EC2, S3)` is
    one skill with a parenthetical, and splitting it on every comma produced
    `AWS (EC2` and `S3)`. Slashes are left alone for the same reason — `C/C++`
    and `CI/CD` are single names, not two each.
    """
    raw = (line or "").strip()
    if not raw:
        return "", []

    category = ""
    match = _LABEL_RE.match(raw)
    if match:
        category, raw = match.group(1).strip(), match.group(2).strip()

    parts, buffer, depth = [], [], 0
    for char in raw:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(buffer))
            buffer = []
            continue
        buffer.append(char)
    parts.append("".join(buffer))

    cleaned = []
    for part in parts:
        item = part.strip(" \t-•·|")
        if item:
            cleaned.append(item)
    return category, cleaned


def normalise_surface(text: str) -> str:
    """
    Reduce a written skill to something the vocabulary can be asked about.

    Drops a trailing parenthetical — `AWS (EC2, S3)` is evidence of `AWS`, and
    the sub-services are detail rather than separate claims.
    """
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", (text or "").strip())
    return stripped.strip(" \t.,;:-•·|")


__all__ = [
    "SKILLS",
    "Skill",
    "all_canonical_names",
    "canonical_for",
    "find_skills",
    "is_known",
    "normalise_surface",
    "split_skill_line",
]
