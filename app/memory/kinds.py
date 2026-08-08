"""
The memory taxonomy.

One `MemoryRecord` table with a `kind` discriminator, rather than a subsystem
per memory type. That choice is load-bearing: unified ranking across kinds is
only possible when every candidate lives in one place and can be scored by one
function. Nine separate stores would recreate exactly the fragmentation this
redesign exists to remove.

`kind` still carries real semantics — it drives decay half-life, context-budget
tier, and retrieval strategy — it simply does not drive *storage*.

See docs/MEMORY_ARCHITECTURE.md §3.2.
"""
from enum import Enum


class MemoryKind(str, Enum):
    """What sort of thing is remembered."""

    IDENTITY = "identity"
    """Who the user is: name, role, education, location, profile links.
    Very stable, tiny, and always injected into context."""

    PREFERENCE = "preference"
    """How the assistant should behave: tone, format, language, verbosity.
    Always injected."""

    SEMANTIC = "semantic"
    """Durable facts about the user: skills, habits, opinions, relationships."""

    EPISODIC = "episodic"
    """Timestamped events and conversation summaries."""

    GOAL = "goal"
    """An intention with state and often a target date."""

    TASK = "task"
    """An actionable item, usually derived from a goal or a conversation."""

    PROCEDURAL = "procedural"
    """How to do something — including which tool inputs previously worked."""

    DOCUMENT = "document"
    """A chunk of an ingested artifact, with provenance back to its source."""

    RELATION = "relation"
    """A typed edge between entities: (user)-[built]->(project)."""


# Retention half-life per kind, in days. Used by the decay engine (Phase 5) and
# by recency scoring during retrieval (Phase 2). `None` means "never decays".
#
# Identity and preferences do not fade: forgetting the user's name because it
# was not mentioned recently would be a bug, not a feature. Tasks decay fast
# because a stale task is worse than no task.
KIND_HALF_LIFE_DAYS: dict[MemoryKind, float | None] = {
    MemoryKind.IDENTITY: None,
    MemoryKind.PREFERENCE: None,
    MemoryKind.GOAL: None,
    MemoryKind.SEMANTIC: 365.0,
    MemoryKind.DOCUMENT: None,
    MemoryKind.PROCEDURAL: 180.0,
    MemoryKind.EPISODIC: 90.0,
    MemoryKind.RELATION: 365.0,
    MemoryKind.TASK: 7.0,
}

# Kinds that are injected into every prompt regardless of query relevance.
# These occupy the guaranteed tier of the context budget (§3.7).
ALWAYS_INJECTED_KINDS = frozenset({
    MemoryKind.IDENTITY,
    MemoryKind.PREFERENCE,
    MemoryKind.GOAL,
})


class SourceType(str, Enum):
    """Where a memory came from.

    Every future integration — MCP or otherwise — enters the system as one of
    these rather than as a new code path (§3.10).
    """

    CHAT = "chat"
    UPLOAD = "upload"
    SYSTEM = "system"
    """Written by the assistant itself, e.g. consolidation output."""

    # Connectors. Declared ahead of their implementations so that backfilled
    # and externally-sourced records are distinguishable from day one.
    GITHUB = "github"
    LEETCODE = "leetcode"
    LINKEDIN = "linkedin"
    GMAIL = "gmail"
    CALENDAR = "calendar"
    NOTION = "notion"
    DRIVE = "drive"


class Visibility(str, Enum):
    """
    Who may retrieve a record — orthogonal to who owns it.

    Ownership alone was the previous access model, which is why the recruiter
    demo retrieved an empty partition: a guest's own `user_id` has no data, and
    there was no way to express "the owner's résumé is publicly readable".
    """

    PRIVATE = "private"
    """Owner only. The default for everything."""

    SHARED = "shared"
    """Readable by explicitly granted principals (multi-user, future)."""

    PUBLIC = "public"
    """Readable by any principal holding PROFILE_READ — the recruiter view."""


class Sensitivity(str, Enum):
    """How carefully a record must be handled once retrieved."""

    NORMAL = "normal"
    SENSITIVE = "sensitive"
    """Personal but legitimately usable — health, finances, relationships."""
    SECRET = "secret"
    """Must never reach a prompt. Credential-shaped values are rejected at
    write time; this exists for anything that slips past."""


class RecordStatus(str, Enum):
    """
    Lifecycle state. Forgetting is a state machine, never a silent delete:

        active ──decay──> archived ──user action──> deleted
           │                  │
           └──superseded──────┘
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    """Replaced by a newer version. Retained for versioning and audit."""
    ARCHIVED = "archived"
    """Decayed below relevance. Excluded from default retrieval, still
    searchable on explicit deep recall."""
    DELETED = "deleted"
    """Soft-deleted, pending hard erasure."""


class EmbeddingStatus(str, Enum):
    """
    Whether this record's vector exists yet.

    Embedding is deliberately *not* computed on the request path. A profile
    fact saved mid-conversation must not pay for a Cohere round trip, and voice
    turns cannot afford one at all. Records are written `PENDING` and embedded
    by a background pass.
    """

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    SKIPPED = "skipped"
    """Deliberately not embedded — too short or too low-value to be worth a
    vector slot."""
