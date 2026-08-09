"""
Deterministic detection of questions about the user.

The planner is an LLM scoring intent from the query text alone, with no view of
memory. Asked "What is my name?" it concluded the question was ambiguous and
asked which name was meant, while the name sat in profile memory. That failure
mode is structural: a model that cannot see the answer cannot know the question
is answerable, so it treats absence of context as ambiguity.

This module removes that judgement from the model for the one category where it
is reliably wrong — questions about the user themselves. Detection is by
grammatical shape (a self-reference plus a subject), not a list of phrasings, so
"what is my cgpa", "which college am I studying at" and "where did I intern" all
resolve without being enumerated anywhere.

What this module does NOT do is decide the answer. It decides that *retrieval
must be attempted*, which is the whole of the answer-first policy: retrieval may
still come back empty, and reporting that honestly is a correct outcome.
Ambiguity is a property of the question; emptiness is a property of the store,
and the two must never be confused.
"""
from __future__ import annotations

import re
from typing import Optional, Set

# ── Intent labels ────────────────────────────────────────────────────────────
PROFILE_NAME = "PROFILE_NAME"
PROFILE_EDUCATION = "PROFILE_EDUCATION"
PROFILE_CGPA = "PROFILE_CGPA"
PROFILE_COLLEGE = "PROFILE_COLLEGE"
PROFILE_SKILLS = "PROFILE_SKILLS"
PROFILE_PROJECTS = "PROFILE_PROJECTS"
PROFILE_EXPERIENCE = "PROFILE_EXPERIENCE"
PROFILE_INTERNSHIP = "PROFILE_INTERNSHIP"
PROFILE_ACHIEVEMENTS = "PROFILE_ACHIEVEMENTS"
PROFILE_GENERAL = "PROFILE_GENERAL"
CONVERSATION_FOLLOWUP = "CONVERSATION_FOLLOWUP"

# Which retrieval each intent should reach for. Consumed by the profile agent so
# the tool choice for these questions is not left to the model either.
INTENT_TOOLS = {
    # Identity reads canonical identity, not a profile summary. "What is my
    # name?" must not be answerable from a name the user once asked to have
    # remembered — see app/memory/identity.py.
    PROFILE_NAME: ["get_identity"],
    PROFILE_EDUCATION: ["get_education"],
    PROFILE_CGPA: ["get_education"],
    PROFILE_COLLEGE: ["get_education"],
    PROFILE_SKILLS: ["get_skills"],
    PROFILE_PROJECTS: ["get_projects"],
    PROFILE_EXPERIENCE: ["get_experience"],
    PROFILE_INTERNSHIP: ["get_experience"],
    PROFILE_ACHIEVEMENTS: ["get_achievements"],
    PROFILE_GENERAL: ["get_profile_summary"],
    CONVERSATION_FOLLOWUP: [],
}

# First- and second-person markers: the referent is the account being spoken to.
_SELF_REFERENCE_TOKENS: Set[str] = {
    "my", "mine", "myself", "i", "im", "ive", "id", "ill", "me",
    "we", "our", "ours", "us", "weve",
}

# Only these *claim* a query for retrieval. The object pronouns "me" and "us"
# are excluded on purpose: "find me jobs" is a job search, not a question about
# the user's employment history, and treating every dative "me" as ownership
# would hijack ordinary requests.
_OWNERSHIP_TOKENS: Set[str] = {
    "my", "mine", "myself", "our", "ours", "ourselves",
    "i", "im", "ive", "id", "ill", "we", "weve",
}

# "what do you know about me" owns its subject without a possessive.
_ABOUT_SELF_RE = re.compile(r"\babout\s+(me|myself|us|ourselves)\b")

# Subject keywords per intent, ordered most specific first. These are *topic*
# words, not question templates — the same set answers "where did I intern",
# "what company did I intern at" and "tell me about my internship".
_SUBJECTS = [
    (PROFILE_INTERNSHIP, {"intern", "interned", "interning", "internship", "internships"}),
    # CPI/SPI are what this user's institute calls the CGPA, and their absence
    # here is the whole of failure #1: "what is my current CPI" matched no
    # subject, fell through to the planner, and came back "I don't have
    # real-time access to your current CPI" — from a system holding an
    # education chunk that states it.
    (PROFILE_CGPA, {"cgpa", "gpa", "cpi", "spi", "sgpa", "cgpi", "grade", "grades",
                    "percentage", "marks", "score", "aggregate", "result", "results"}),
    (PROFILE_COLLEGE, {"college", "university", "institute", "school", "campus"}),
    (PROFILE_EDUCATION, {"education", "degree", "branch", "major", "btech", "b.tech",
                         "bachelor", "bachelors", "master", "masters", "course",
                         "stream", "academic", "academics", "studying", "study"}),
    (PROFILE_PROJECTS, {"project", "projects", "portfolio", "built", "build"}),
    # "know" is deliberately absent: "what do you know about me" is a general
    # profile question, not a skills question.
    (PROFILE_SKILLS, {"skill", "skills", "technology", "technologies", "tech",
                      "stack", "language", "languages", "framework", "frameworks",
                      "tool", "tools", "proficient"}),
    (PROFILE_ACHIEVEMENTS, {"achievement", "achievements", "award", "awards",
                            "certification", "certifications", "honor", "honors",
                            "scholarship", "accomplishment", "accomplishments"}),
    (PROFILE_EXPERIENCE, {"experience", "experiences", "work", "worked", "job",
                          "jobs", "company", "companies", "employer", "employers",
                          "role", "roles", "position", "career"}),
    (PROFILE_NAME, {"name", "called"}),
]

# Phrases that point at a store rather than asking a new question. "I think it
# was in my resume" is not an ambiguous request — it is an instruction to look
# again, and answering it with a clarifying question is the loop the user hit.
_SOURCE_HINTS = {
    "resume", "cv", "profile", "memory", "remember", "recall", "stored",
    "saved", "uploaded", "document", "file", "mentioned", "said", "told",
}

# Referential openers with no subject of their own — they inherit one from the
# previous turn ("tell me more about TRACE", "what about that one").
_FOLLOWUP_MARKERS = {
    "it", "that", "this", "those", "these", "there", "then", "one", "ones",
    "more", "about", "again", "also", "too", "else", "other", "another",
}

# Verbs that ask for an action on the world rather than for information. These
# carry referential words too — "send this to him" — but their missing pieces
# are recipients and dates, which no store holds. Retrieval cannot answer them
# and clarification is the correct response, so they are never claimed as
# follow-ups. Informational verbs (tell, show, list, describe) are absent by
# design: "tell me more about TRACE" is a retrieval.
_ACTION_VERBS = {
    "send", "email", "mail", "forward", "reply", "schedule", "book", "cancel",
    "apply", "submit", "call", "message", "text", "remind", "delete", "remove",
    "order", "buy", "pay", "share", "post", "upload", "download", "install",
}

_WORD_RE = re.compile(r"[a-z0-9.+#']+")


def _tokens(query: str) -> list:
    """
    Words, lowercased, with sentence punctuation stripped.

    Interior dots survive so "b.tech" stays one token, but a trailing one is
    removed — without that, "Tell me about my education." tokenised to
    "education." and matched no subject at all.

    Possessive "'s" is dropped before apostrophes are removed. Without that step
    "today's date" tokenised to "todays", which matches no temporal subject and
    is why "what is today's date?" reached memory instead of the clock.
    """
    raw = (query or "").lower().replace("’", "'")
    raw = re.sub(r"'s\b", "", raw).replace("'", "")
    return [token.strip(".") for token in _WORD_RE.findall(raw) if token.strip(".")]


def is_self_referential(query: str) -> bool:
    """Whether the query is about the user or the conversation with them."""
    return any(token in _SELF_REFERENCE_TOKENS for token in _tokens(query))


def mentions_a_source(query: str) -> bool:
    """Whether the query points at a store the assistant should consult."""
    return any(token in _SOURCE_HINTS for token in _tokens(query))


def classify(query: str, *, has_context: bool = False) -> Optional[str]:
    """
    Classify a query as a personal-information question.

    Returns an intent label, or None when the query is not about the user — in
    which case ordinary planning applies and clarification remains available.

    `has_context` reports whether earlier turns exist. It is what makes
    "I think it was mentioned in my resume" resolvable: the sentence has a
    source hint but no subject, so it only means "look again" when there is a
    previous question to look again *for*.
    """
    tokens = _tokens(query)
    if not tokens:
        return None

    token_set = set(tokens)
    self_referential = bool(token_set & _OWNERSHIP_TOKENS) or bool(
        _ABOUT_SELF_RE.search((query or "").lower())
    )
    source_hint = bool(token_set & _SOURCE_HINTS)

    # A subject decides the intent whenever the query is about the user, or
    # names a store to search. "What is in my resume about internships" and
    # "where did I intern" reach the same retrieval.
    if self_referential or source_hint:
        for intent, subjects in _SUBJECTS:
            if token_set & subjects:
                return intent

    # No subject of its own. Two ways that still resolves:
    #   "I think it was mentioned in my resume" — a source hint plus history.
    #   "tell me more about TRACE"              — a referential opener.
    #
    # An imperative action is excluded: "send this to him" has a referential
    # word and prior turns, but its missing recipient is not in any store.
    requests_an_action = bool(token_set & _ACTION_VERBS)
    if has_context and not requests_an_action and (source_hint or token_set & _FOLLOWUP_MARKERS):
        return CONVERSATION_FOLLOWUP

    # Self-referential with no subject at all: "what do you know about me".
    if self_referential:
        return PROFILE_GENERAL

    return None


def is_personal_information_query(query: str, *, has_context: bool = False) -> bool:
    """Whether retrieval must be attempted before any clarification."""
    return classify(query, has_context=has_context) is not None


def tools_for(intent: Optional[str]) -> list:
    """The retrieval an intent should reach for; empty when it needs context."""
    return list(INTENT_TOOLS.get(intent or "", []))


def tokens(query: str) -> list:
    """
    Public tokenizer, shared with the query-category layer.

    Exported so the two classifiers cannot drift apart on what a word is —
    a possessive handled in one and not the other is precisely the kind of
    divergence that sends "today's date" to the wrong store.
    """
    return _tokens(query)
