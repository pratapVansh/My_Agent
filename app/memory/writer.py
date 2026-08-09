"""
Translation from the legacy write paths into `MemoryRecord`.

Phase 1 mirrors writes: the existing tables stay authoritative and continue to
serve every read, while the same facts are also written here. That is what
makes the eventual cutover a flag flip rather than a migration event.

Only *structured* writes are mirrored — profile facts, episodes, tool
outcomes. Raw conversation turns are deliberately excluded: mirroring every
utterance is precisely the noise-accumulation behaviour Phase 3 exists to
replace, and Phase 4 gives turns a proper home in `Conversation`/`Turn`.

See docs/MEMORY_ARCHITECTURE.md §3.5.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import logging

from app.memory.kinds import (
    MemoryKind,
    SourceType,
    Visibility,
)
from app.memory.record import MemoryRecord
from app.memory.stores.postgres_record_store import postgres_record_store

logger = logging.getLogger(__name__)


class WriteOutcome(str, Enum):
    """What a write actually did — reported, never inferred."""

    CREATED = "created"
    DUPLICATE = "duplicate"
    """An identical active record already existed; nothing was written."""
    SUPERSEDED = "superseded"
    """A contradictory record was closed and replaced by this one."""
    REJECTED = "rejected"
    """A contradictory record existed and outranked this write. The stored
    value is unchanged; the incoming one is reported, not silently dropped."""


def _claim_for(record: MemoryRecord):
    """Describe a record as a claim the conflict resolver can rank."""
    from app.memory.identity import FactClaim
    from app.memory.sources import MemorySource

    source = {
        SourceType.UPLOAD: MemorySource.RESUME_DOCUMENT,
        SourceType.SYSTEM: MemorySource.PROFILE_MEMORY,
        SourceType.CHAT: MemorySource.CONVERSATION_CURRENT,
    }.get(record.source_type, MemorySource.SEMANTIC_MEMORY)

    structured = record.structured or {}
    explicit = bool(structured.get("explicit"))
    if "explicit" not in structured:
        # Records written before the flag existed: an inferred write is the only
        # one that arrives over SourceType.CHAT.
        explicit = record.source_type is not SourceType.CHAT

    return FactClaim(
        key=str(structured.get("key") or record.dedup_key or ""),
        value=str(structured.get("value") or record.content),
        source=source,
        confidence=record.confidence,
        explicit=explicit,
        timestamp=record.updated_at or record.created_at,
    )


# ── Profile-key classification ──────────────────────────────────────────
#
# Which kind a profile fact becomes decides whether it is *always* injected
# into context (identity, preference) or retrieved on relevance (semantic).
# Getting this wrong is expensive in both directions: a misclassified identity
# fact wastes guaranteed budget on every single turn, while a misclassified
# preference is silently ignored until someone happens to ask about it.

_IDENTITY_KEYS = frozenset({
    "name", "full_name", "first_name", "last_name", "nickname",
    # The canonical namespace, added when identity was given its own keys. Their
    # absence made this classifier return SEMANTIC for the user's own name,
    # which quietly cost it three things: always-injected status, the identity
    # kind prior, and — worst — decay exemption. A name on a 365-day half-life
    # is a name the assistant eventually forgets.
    "canonical_name", "canonical_email", "canonical_college",
    "canonical_degree", "canonical_location",
    "role", "title", "occupation", "profession",
    "email", "contact_email", "phone", "location", "city", "country",
    "timezone", "university", "college", "school", "degree", "major",
    "branch", "graduation_year", "student_id", "roll_number",
    "github", "linkedin", "leetcode", "portfolio", "website", "twitter",
})

_PREFERENCE_PREFIXES = ("preferred_", "prefers_", "favourite_", "favorite_")

_PREFERENCE_KEYS = frozenset({
    "tone", "language", "response_format", "verbosity", "style",
    "communication_style", "formality", "detail_level",
})

_GOAL_KEYS = frozenset({
    "goal", "goals", "job_target", "career_goal", "target_role",
    "objective", "aspiration",
})


def classify_profile_key(key: str) -> MemoryKind:
    """
    Map a legacy `user_profile.key` onto a memory kind.

    The `remembered_` namespace is checked *first* and never becomes IDENTITY.
    "remembered_name" holds a name the user asked to be kept beside their own,
    and classifying it as identity would put it in the always-injected tier
    alongside the real one — recreating, through the taxonomy, exactly the
    confusion the separate keys exist to prevent.
    """
    normalized = (key or "").strip().lower()

    if _is_remembered_key(normalized):
        return MemoryKind.SEMANTIC
    if normalized in _IDENTITY_KEYS:
        return MemoryKind.IDENTITY
    if normalized in _GOAL_KEYS:
        return MemoryKind.GOAL
    if normalized in _PREFERENCE_KEYS:
        return MemoryKind.PREFERENCE
    if normalized.startswith(_PREFERENCE_PREFIXES):
        return MemoryKind.PREFERENCE
    return MemoryKind.SEMANTIC


def _is_remembered_key(normalized: str) -> bool:
    """Whether this key holds something the user explicitly asked to keep."""
    from app.memory.identity import is_explicit_memory_key

    return is_explicit_memory_key(normalized)


def is_pinned_key(key: str) -> bool:
    """
    Whether a profile key should be exempt from decay.

    `pinned` already means "never decays, never dropped for budget" and
    `is_decay_exempt` honours it — but nothing ever set it, so a value the user
    explicitly asked to be remembered aged out on the same schedule as one the
    extractor guessed at. Two categories earn the exemption: canonical identity,
    and anything the user deliberately asked to keep.
    """
    from app.memory.identity import is_canonical_key, is_explicit_memory_key

    normalized = (key or "").strip().lower()
    return is_canonical_key(normalized) or is_explicit_memory_key(normalized)


# ── Content rendering ───────────────────────────────────────────────────
#
# `content` is what gets embedded and what gets injected, so it has to read as
# a complete statement on its own. "concise" is useless out of context;
# "The user's preferred tone is concise." survives retrieval into any prompt.

_KEY_TEMPLATES: Dict[str, str] = {
    "name": "The user's name is {value}.",
    "canonical_name": "The user's name is {value}.",
    "canonical_email": "The user's email address is {value}.",
    "canonical_college": "The user studies at {value}.",
    "canonical_degree": "The user is pursuing {value}.",
    "canonical_location": "The user is located in {value}.",
    "full_name": "The user's full name is {value}.",
    "github": "The user's GitHub profile is {value}.",
    "linkedin": "The user's LinkedIn profile is {value}.",
    "leetcode": "The user's LeetCode profile is {value}.",
    "portfolio": "The user's portfolio is at {value}.",
    "website": "The user's website is {value}.",
    "email": "The user's email address is {value}.",
    "contact_email": "The user's contact email address is {value}.",
    "timezone": "The user's timezone is {value}.",
    "goal": "The user's goal is {value}.",
    "job_target": "The user is targeting this kind of role: {value}.",
}


def render_profile_fact(key: str, value: str) -> str:
    """
    Turn a key/value pair into a self-contained sentence.

    Remembered values get a sentence that says what they are *and what they are
    not*. This matters more than it looks: `content` is what gets embedded and
    injected, so "The user's remembered name is Devasi" — the phrasing the
    generic template produced — reads to a model as a statement about the user's
    name. Separate keys stop the wrong value being *retrieved*; only the wording
    stops the right value being *misread* once it is in the prompt.
    """
    normalized = (key or "").strip().lower()

    if _is_remembered_key(normalized):
        subject = normalized
        for prefix in ("remembered_",):
            if subject.startswith(prefix):
                subject = subject[len(prefix):]
        subject = subject.replace("_", " ").strip() or "value"
        if subject in ("name", "alternate name", "nickname", "preferred name", "alias"):
            return (
                f"The user asked the assistant to remember the name {value}. "
                f"This is not the user's own name."
            )
        return (
            f"The user asked the assistant to remember this {subject}: {value}."
        )

    template = _KEY_TEMPLATES.get(normalized)
    if template:
        return template.format(value=value)
    readable = normalized.replace("_", " ").strip() or "attribute"
    return f"The user's {readable} is {value}."


def render_episode(
    user_summary: str, agent_summary: str, agent_used: Optional[str]
) -> str:
    """Turn a conversation turn summary into a self-contained statement."""
    parts = []
    if user_summary:
        parts.append(f"The user asked: {user_summary}")
    if agent_summary:
        actor = f"the {agent_used} agent" if agent_used else "the assistant"
        parts.append(f"In response, {actor}: {agent_summary}")
    return ". ".join(p.rstrip(".") for p in parts) + "." if parts else ""


def render_tool_outcome(
    agent_name: str, tool_name: str, inputs_summary: str, key_insight: str
) -> str:
    """Turn a successful tool call into reusable procedural knowledge."""
    return (
        f"When the {agent_name} agent used the {tool_name} tool with "
        f"{inputs_summary}, the useful result was: {key_insight}"
    )


# Baseline importance per kind. Extraction refines these in Phase 3; until then
# they are the only salience signal ranking will have.
#
# Public because the backfill migration builds records directly — it must
# preserve each legacy row's original timestamps, which the writer's
# convenience methods deliberately do not expose — and it has to assign exactly
# the salience the live write path would.
BASE_IMPORTANCE: Dict[MemoryKind, float] = {
    MemoryKind.IDENTITY: 0.9,
    MemoryKind.PREFERENCE: 0.85,
    MemoryKind.GOAL: 0.8,
    MemoryKind.SEMANTIC: 0.6,
    MemoryKind.DOCUMENT: 0.5,
    MemoryKind.PROCEDURAL: 0.5,
    MemoryKind.EPISODIC: 0.4,
    MemoryKind.TASK: 0.5,
    MemoryKind.RELATION: 0.5,
}


class MemoryWriter:
    """Writes `MemoryRecord`s derived from the legacy write paths."""

    def __init__(self, record_store=None):
        self.records = record_store or postgres_record_store

    async def record_profile_fact(
        self,
        owner_id: str,
        key: str,
        value: str,
        *,
        source: str = "explicit",
        confidence: float = 1.0,
        occurred_at: Optional[datetime] = None,
    ) -> Optional[MemoryRecord]:
        """
        Mirror a profile fact. Returns the record that is now current.

        Note the record returned is not necessarily the one submitted: a write
        that loses to a higher-trust stored value hands back the survivor. Use
        `record_profile_fact_with_outcome` when the caller needs to know which
        happened.
        """
        stored, _ = await self.record_profile_fact_with_outcome(
            owner_id, key, value,
            source=source, confidence=confidence, occurred_at=occurred_at,
        )
        return stored

    async def record_profile_fact_with_outcome(
        self,
        owner_id: str,
        key: str,
        value: str,
        *,
        source: str = "explicit",
        confidence: float = 1.0,
        occurred_at: Optional[datetime] = None,
    ) -> tuple[MemoryRecord, WriteOutcome]:
        """
        Mirror a profile fact and report what the write actually did.

        A profile key holds exactly one current value, so re-saving a key with
        a different value is a *contradiction*, not a second fact. Whether the
        new value wins is decided by `resolve_conflict` rather than by arrival
        order — an inferred remark must not displace something the user stated
        deliberately. The loser is reported, never silently dropped.

        The superseded record is retained rather than deleted, which preserves
        the history the redesign needs for versioning and for answering "what
        did you think my role was in March".
        """
        kind = classify_profile_key(key)
        content = render_profile_fact(key, value)
        dedup_key = f"profile:{(key or '').strip().lower()}"

        # Explicit beats inferred, structurally rather than by prompt wording.
        # A value the user asked to keep is pinned — exempt from decay and from
        # budget eviction — while an inferred one takes its chances with the
        # ranker. Before this, `pinned` was honoured everywhere and set nowhere,
        # so the distinction existed in the schema and nowhere in behaviour.
        pinned = is_pinned_key(key) or (
            source == "explicit" and kind is MemoryKind.PREFERENCE
        )

        record = MemoryRecord(
            owner_id=owner_id,
            kind=kind,
            content=content,
            structured={
                "key": key,
                "value": value,
                "source": source,
                "explicit": source == "explicit",
            },
            importance=BASE_IMPORTANCE[kind],
            confidence=confidence,
            pinned=pinned,
            occurred_at=occurred_at,
            source_type=SourceType.CHAT if source == "inferred" else SourceType.SYSTEM,
            source_ref=f"user_profile:{key}",
            dedup_key=dedup_key,
            visibility=Visibility.PRIVATE,
        )

        return await self.upsert_with_outcome(record)

    async def upsert(self, record: MemoryRecord) -> MemoryRecord:
        """Write a record, superseding any active contradiction."""
        stored, _ = await self.upsert_with_outcome(record)
        return stored

    async def upsert_with_outcome(
        self, record: MemoryRecord
    ) -> tuple[MemoryRecord, WriteOutcome]:
        """
        Write a record and report what actually happened.

        A record carrying a `dedup_key` claims a slot that holds exactly one
        current value, so an existing active record under that key with
        different content is a *contradiction*, not a second fact. Records
        without a dedup key — episodes, documents, tool outcomes — simply
        accumulate, de-duplicated only by exact content.

        The outcome is returned rather than inferred by the caller. Both a
        duplicate and a supersession hand back a record whose id differs from
        the one submitted, so there is no reliable way to tell them apart from
        the outside — a caller that tries will silently miscount one as the
        other.

        Shared by the Phase 1 dual-write path and the Phase 3 extraction
        pipeline so both resolve conflicts identically.
        """
        if not record.dedup_key:
            stored = await self.records.add(record)
            return stored, (
                WriteOutcome.CREATED if stored.id == record.id
                else WriteOutcome.DUPLICATE
            )

        existing = await self.records.find_by_dedup_key(
            record.owner_id, record.dedup_key
        )
        conflicting = [r for r in existing if r.content_hash != record.content_hash]
        if not conflicting:
            stored = await self.records.add(record)
            return stored, (
                WriteOutcome.CREATED if stored.id == record.id
                else WriteOutcome.DUPLICATE
            )

        previous = conflicting[0]

        # Whether the incoming claim is actually *allowed* to win.
        #
        # This used to be unconditional: any content difference superseded the
        # stored value, so "newest wins" was the entire conflict policy. That
        # let a low-confidence inferred remark overwrite a fact the user had
        # stated deliberately or that came from their résumé — a silent,
        # unrecoverable downgrade of the best evidence in the store.
        #
        # The resolver is deterministic on purpose. A model asked to reconcile
        # two contradictory facts invents a rule, and invents a different one
        # next time; the ordering here is fixed, inspectable, and testable.
        from app.memory.identity import Resolution, resolve_conflict

        verdict = resolve_conflict(_claim_for(previous), _claim_for(record))
        if verdict.resolution is not Resolution.SUPERSEDE:
            logger.info(
                "Rejected a contradictory write to %s for owner=%s: %s",
                record.dedup_key, record.owner_id, verdict.reason,
            )
            return previous, WriteOutcome.REJECTED

        # The replacement is derived from the previous record rather than
        # inserted fresh, so it inherits the version chain. Writing the
        # freshly-built record here would leave every version at v1 with a null
        # `supersedes_id`, silently breaking the history that makes "what did
        # you think my role was in March" answerable.
        replacement = previous.superseding(
            kind=record.kind,
            content=record.content,
            structured=record.structured,
            importance=record.importance,
            confidence=record.confidence,
            occurred_at=record.occurred_at,
            source_type=record.source_type,
            source_ref=record.source_ref,
            dedup_key=record.dedup_key,
            visibility=record.visibility,
        )
        stored = await self.records.supersede(previous, replacement)
        return stored, WriteOutcome.SUPERSEDED

    async def record_episode(
        self,
        owner_id: str,
        session_id: str,
        user_summary: str,
        agent_summary: str,
        *,
        agent_used: Optional[str] = None,
        intent: Optional[str] = None,
        outcome: str = "success",
        occurred_at: Optional[datetime] = None,
    ) -> Optional[MemoryRecord]:
        """Mirror an episodic turn summary."""
        content = render_episode(user_summary, agent_summary, agent_used)
        if not content:
            return None

        # A turn that failed is worth remembering, but not worth surfacing as
        # readily as one that worked.
        importance = BASE_IMPORTANCE[MemoryKind.EPISODIC]
        if outcome == "failed":
            importance -= 0.15

        record = MemoryRecord(
            owner_id=owner_id,
            kind=MemoryKind.EPISODIC,
            content=content,
            structured={
                "user_summary": user_summary,
                "agent_summary": agent_summary,
                "agent_used": agent_used,
                "intent": intent,
                "outcome": outcome,
                "session_id": session_id,
            },
            importance=importance,
            occurred_at=occurred_at,
            source_type=SourceType.CHAT,
            source_ref=f"session:{session_id}",
            # Episodes are events, not claims: two similar ones do not
            # contradict, so they carry no dedup key.
            dedup_key=None,
        )
        return await self.records.add(record)

    async def record_tool_outcome(
        self,
        owner_id: str,
        agent_name: str,
        tool_name: str,
        inputs_summary: str,
        key_insight: str,
        *,
        occurred_at: Optional[datetime] = None,
    ) -> Optional[MemoryRecord]:
        """Mirror a successful tool use as procedural memory."""
        content = render_tool_outcome(
            agent_name, tool_name, inputs_summary, key_insight
        )
        record = MemoryRecord(
            owner_id=owner_id,
            kind=MemoryKind.PROCEDURAL,
            content=content,
            structured={
                "agent_name": agent_name,
                "tool_name": tool_name,
                "inputs_summary": inputs_summary,
                "key_insight": key_insight,
            },
            importance=BASE_IMPORTANCE[MemoryKind.PROCEDURAL],
            occurred_at=occurred_at,
            source_type=SourceType.SYSTEM,
            source_ref=f"tool:{agent_name}:{tool_name}",
            dedup_key=None,
        )
        return await self.records.add(record)

    async def record_document_chunk(
        self,
        owner_id: str,
        text: str,
        *,
        document_id: str,
        semantic_type: str = "other",
        chunk_index: int = 0,
        source_file: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
    ) -> Optional[MemoryRecord]:
        """Mirror one chunk of an ingested document."""
        if not (text or "").strip():
            return None

        record = MemoryRecord(
            owner_id=owner_id,
            kind=MemoryKind.DOCUMENT,
            content=text,
            structured={
                "document_id": document_id,
                "semantic_type": semantic_type,
                "chunk_index": chunk_index,
                "source_file": source_file,
            },
            importance=BASE_IMPORTANCE[MemoryKind.DOCUMENT],
            occurred_at=occurred_at,
            source_type=SourceType.UPLOAD,
            source_ref=document_id,
            dedup_key=None,
        )
        return await self.records.add(record)


# Singleton instance
memory_writer = MemoryWriter()
