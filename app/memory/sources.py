"""
Which memory source answers which question.

The store was never the problem. Résumé sections, profile facts, conversation
turns, episodes and vectors were all present and all retrievable — and the
assistant still answered "I don't have real-time access to your current CPI"
to a question whose answer sat in an `education` chunk. The failure was that
*nothing decided where to look*: every source was flattened into one prompt
block, and the model was left to infer from an undifferentiated wall of text
both which fact was relevant and how much to trust it.

This module supplies the missing decision. A query category names a *kind of
question*; `SOURCE_PRECEDENCE` names the ordered sources that answer it. The
order is the point — "what is my name" and "what name did I ask you to
remember" are the same words about the same attribute, and they must reach
different stores. Precedence, not similarity, is what separates them.

`RetrievedMemory` carries the provenance that makes precedence enforceable
after retrieval too: a fact that came from a résumé and a fact the user typed
once in passing are not interchangeable, and a system that forgets which was
which cannot resolve a contradiction between them.

See docs/MEMORY_ARCHITECTURE.md §3.4 for the ownership/visibility model these
sources are read under.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class QueryCategory(str, Enum):
    """The kind of question being asked, which decides where to look."""

    # ── Questions about the user, answered from stored personal data ────────
    PROFILE_IDENTITY = "PROFILE_IDENTITY"
    PROFILE_EDUCATION = "PROFILE_EDUCATION"
    PROFILE_EXPERIENCE = "PROFILE_EXPERIENCE"
    PROFILE_SKILLS = "PROFILE_SKILLS"
    PROFILE_PROJECTS = "PROFILE_PROJECTS"
    PROFILE_ACHIEVEMENTS = "PROFILE_ACHIEVEMENTS"
    PROFILE_GENERAL = "PROFILE_GENERAL"

    # ── Memory the user deliberately created, kept apart from who they are ──
    EXPLICIT_MEMORY = "EXPLICIT_MEMORY"
    """Reading back something the user asked to be remembered."""

    EXPLICIT_MEMORY_WRITE = "EXPLICIT_MEMORY_WRITE"
    """"Remember X" — a write, not a question."""

    IDENTITY_UPDATE = "IDENTITY_UPDATE"
    """An explicit instruction to change canonical identity. Rare and narrow:
    a sentence merely *containing* a name is not one of these."""

    # ── Conversation ────────────────────────────────────────────────────────
    CONVERSATION_CURRENT = "CONVERSATION_CURRENT"
    """"What did I just tell you" — this thread, recent turns."""

    CONVERSATION_FOLLOWUP = "CONVERSATION_FOLLOWUP"
    """A question whose subject is inherited from the previous turn."""

    EPISODIC_MEMORY = "EPISODIC_MEMORY"
    """"What did I tell you yesterday" — past sessions."""

    DOCUMENT_RESUME = "DOCUMENT_RESUME"
    """Explicitly about the résumé as a document."""

    # ── Not memory at all ───────────────────────────────────────────────────
    TEMPORAL_CURRENT = "TEMPORAL_CURRENT"
    """The clock and the calendar. Never answerable from memory."""

    SCHEDULE_TEMPORAL = "SCHEDULE_TEMPORAL"
    """"What class do I have today" — the clock resolves "today", memory holds
    the timetable. Both are required; neither is sufficient."""

    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    """"How do I build an AI agent" — answer it, do not interview the user."""

    ACTION_REQUEST = "ACTION_REQUEST"
    """A complete instruction to do something in the world."""

    AMBIGUOUS_ACTION = "AMBIGUOUS_ACTION"
    """An action missing a parameter no store holds. The one honest reason to
    ask a clarifying question."""

    SMALL_TALK = "SMALL_TALK"
    """Greetings, acknowledgements, venting. Answered, never filed."""


class MemorySource(str, Enum):
    """A place an answer can come from."""

    CANONICAL_IDENTITY = "canonical_identity"
    """Who the user actually is. Changed only by an explicit identity update."""

    EXPLICIT_MEMORY = "explicit_memory"
    """Things the user asked to be remembered, under their own keys."""

    RESUME_DOCUMENT = "resume_document"
    """Typed résumé sections: education, experience, skills, projects,
    achievements."""

    PROFILE_MEMORY = "profile_memory"
    """Structured profile facts accumulated outside the résumé."""

    CONVERSATION_CURRENT = "conversation_current"
    """Recent turns of the live thread."""

    EPISODIC_MEMORY = "episodic_memory"
    """Summaries of earlier sessions."""

    SEMANTIC_MEMORY = "semantic_memory"
    """Vector recall over durable extracted facts."""

    TEMPORAL_TOOL = "temporal_tool"
    """The clock. Authoritative for "now"; memory never is."""

    GENERAL_KNOWLEDGE = "general_knowledge"
    """The model's own knowledge. Not a store."""

    EXTERNAL_TOOL = "external_tool"
    """Timetable, attendance, email, job search."""

    NONE = "none"


# How much a source's claim about a *fact* is worth when two sources disagree.
# Higher wins. This is deliberately not the same ordering as SOURCE_PRECEDENCE:
# precedence says where to look first, trust says whom to believe. A passing
# remark in conversation is looked at early (it is the freshest thing there is)
# and believed least.
SOURCE_TRUST: Dict[MemorySource, int] = {
    MemorySource.CANONICAL_IDENTITY: 100,
    MemorySource.EXPLICIT_MEMORY: 90,
    MemorySource.RESUME_DOCUMENT: 80,
    MemorySource.PROFILE_MEMORY: 70,
    MemorySource.TEMPORAL_TOOL: 60,
    MemorySource.EXTERNAL_TOOL: 55,
    MemorySource.SEMANTIC_MEMORY: 40,
    MemorySource.EPISODIC_MEMORY: 30,
    MemorySource.CONVERSATION_CURRENT: 20,
    MemorySource.GENERAL_KNOWLEDGE: 10,
    MemorySource.NONE: 0,
}


# Ordered sources per category. First entry is consulted first; later entries
# supplement rather than replace it.
#
# The two identity rows are the whole reason this table exists. "What is my
# name?" reads canonical identity; "what name did I ask you to remember?" reads
# explicit memory. Same attribute, same words, different store — and no amount
# of embedding similarity distinguishes them, because the difference is in what
# the user is asking *about*, not what they are asking *for*.
SOURCE_PRECEDENCE: Dict[QueryCategory, Tuple[MemorySource, ...]] = {
    QueryCategory.PROFILE_IDENTITY: (
        MemorySource.CANONICAL_IDENTITY,
        MemorySource.RESUME_DOCUMENT,
        MemorySource.PROFILE_MEMORY,
    ),
    QueryCategory.PROFILE_EDUCATION: (
        MemorySource.RESUME_DOCUMENT,
        MemorySource.PROFILE_MEMORY,
        MemorySource.SEMANTIC_MEMORY,
    ),
    QueryCategory.PROFILE_EXPERIENCE: (
        MemorySource.RESUME_DOCUMENT,
        MemorySource.PROFILE_MEMORY,
        MemorySource.SEMANTIC_MEMORY,
    ),
    QueryCategory.PROFILE_SKILLS: (
        MemorySource.RESUME_DOCUMENT,
        MemorySource.SEMANTIC_MEMORY,
        MemorySource.PROFILE_MEMORY,
    ),
    QueryCategory.PROFILE_PROJECTS: (
        MemorySource.RESUME_DOCUMENT,
        MemorySource.SEMANTIC_MEMORY,
        MemorySource.PROFILE_MEMORY,
    ),
    QueryCategory.PROFILE_ACHIEVEMENTS: (
        MemorySource.RESUME_DOCUMENT,
        MemorySource.PROFILE_MEMORY,
    ),
    QueryCategory.PROFILE_GENERAL: (
        MemorySource.PROFILE_MEMORY,
        MemorySource.CANONICAL_IDENTITY,
        MemorySource.RESUME_DOCUMENT,
        MemorySource.SEMANTIC_MEMORY,
    ),
    QueryCategory.EXPLICIT_MEMORY: (
        MemorySource.EXPLICIT_MEMORY,
        MemorySource.CONVERSATION_CURRENT,
        MemorySource.EPISODIC_MEMORY,
    ),
    QueryCategory.EXPLICIT_MEMORY_WRITE: (
        MemorySource.EXPLICIT_MEMORY,
    ),
    QueryCategory.IDENTITY_UPDATE: (
        MemorySource.CANONICAL_IDENTITY,
    ),
    QueryCategory.CONVERSATION_CURRENT: (
        MemorySource.CONVERSATION_CURRENT,
    ),
    QueryCategory.CONVERSATION_FOLLOWUP: (
        MemorySource.CONVERSATION_CURRENT,
        MemorySource.RESUME_DOCUMENT,
        MemorySource.SEMANTIC_MEMORY,
    ),
    QueryCategory.EPISODIC_MEMORY: (
        MemorySource.EPISODIC_MEMORY,
        MemorySource.CONVERSATION_CURRENT,
        MemorySource.SEMANTIC_MEMORY,
    ),
    QueryCategory.DOCUMENT_RESUME: (
        MemorySource.RESUME_DOCUMENT,
        MemorySource.PROFILE_MEMORY,
    ),
    QueryCategory.TEMPORAL_CURRENT: (
        MemorySource.TEMPORAL_TOOL,
    ),
    QueryCategory.SCHEDULE_TEMPORAL: (
        MemorySource.TEMPORAL_TOOL,
        MemorySource.EXTERNAL_TOOL,
        MemorySource.PROFILE_MEMORY,
    ),
    QueryCategory.GENERAL_KNOWLEDGE: (
        MemorySource.GENERAL_KNOWLEDGE,
    ),
    QueryCategory.ACTION_REQUEST: (
        MemorySource.EXTERNAL_TOOL,
        MemorySource.CONVERSATION_CURRENT,
    ),
    QueryCategory.AMBIGUOUS_ACTION: (
        MemorySource.CONVERSATION_CURRENT,
    ),
    QueryCategory.SMALL_TALK: (
        MemorySource.CONVERSATION_CURRENT,
    ),
}


# Categories whose answer must never be taken from memory. A stored sentence
# saying "today is the 9th" was true when written and is a lie now.
NON_MEMORY_CATEGORIES = frozenset({
    QueryCategory.TEMPORAL_CURRENT,
    QueryCategory.GENERAL_KNOWLEDGE,
})

# Categories where clarification can ever be the right response. Everything
# else must retrieve and then answer or report honestly — asking the user to
# disambiguate their own CGPA is not a clarification, it is a failure to look.
CLARIFIABLE_CATEGORIES = frozenset({
    QueryCategory.AMBIGUOUS_ACTION,
    QueryCategory.ACTION_REQUEST,
})


def sources_for(category: QueryCategory) -> Tuple[MemorySource, ...]:
    """Ordered sources that answer this category of question."""
    return SOURCE_PRECEDENCE.get(category, (MemorySource.PROFILE_MEMORY,))


def requires_retrieval(category: QueryCategory) -> bool:
    """Whether an answer must be looked up before anything is said."""
    return category not in NON_MEMORY_CATEGORIES and category not in (
        QueryCategory.SMALL_TALK,
        QueryCategory.AMBIGUOUS_ACTION,
    )


def may_clarify(category: QueryCategory) -> bool:
    """Whether asking the user a question is a permissible response."""
    return category in CLARIFIABLE_CATEGORIES


@dataclass(frozen=True)
class RetrievedMemory:
    """
    One retrieved item, with everything needed to rank and defend it.

    Kept flat and frozen on purpose: these cross the boundary into prompt
    assembly and into the conflict resolver, and both must be able to read
    provenance without reaching back into the store that produced it.
    """

    content: str
    source: MemorySource
    memory_type: str
    """The record kind — `identity`, `document`, `episodic`, … — as stored."""

    confidence: float = 1.0
    provenance: str = ""
    """Human-readable origin: "resume:education", "conversation:turn-42"."""

    timestamp: Optional[datetime] = None
    owner_id: str = ""
    visibility: str = "private"

    explicit: bool = False
    """The user stated this deliberately, rather than it being inferred."""

    canonical_identity: bool = False
    """This is *the* value of an identity attribute, not a mention of one."""

    derived: bool = False
    """Produced by extraction/summarisation rather than stated verbatim."""

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def trust(self) -> int:
        """Ranking weight when two items disagree."""
        base = SOURCE_TRUST.get(self.source, 0)
        if self.explicit:
            base += 5
        if self.derived:
            base -= 5
        return base

    def describe(self) -> str:
        """Compact provenance line for logs and traces — never the content."""
        return (
            f"{self.source.value}/{self.memory_type}"
            f"{'/explicit' if self.explicit else ''}"
            f"{'/canonical' if self.canonical_identity else ''}"
            f" conf={self.confidence:.2f}"
        )
