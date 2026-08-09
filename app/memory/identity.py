"""
Canonical identity, and why a remembered name is not one.

A user said "Remember the name Devasi." The assistant changed their name.

That is the worst class of bug this system can have, and it is not a prompting
mistake — it is a missing distinction. Both sentences below contain a name, a
first-person possessive, and an imperative:

    "Remember the name Devasi."       → store something, under its own key
    "Update my name to Vansh Pratap." → change who the user is

Nothing in an embedding separates them. What separates them is *intent to
redefine identity*, which is a property of the clause, not of the name. So
identity is given its own key, its own write path, and a rule the model cannot
route around: canonical identity changes only on an explicit identity-update
instruction. Every other name the user mentions is stored beside it, never over
it.

The same asymmetry governs reads. "What is my name?" and "What name did I ask
you to remember?" are answered from different keys, and the second must never
be able to answer the first.

Conflict resolution is deterministic here for the same reason. A model asked to
reconcile "résumé says RGIPT" against "user typed IIT once" will invent a rule,
and it will invent a different one next time. The resolver below always
supersedes rather than overwrites: the previous value and its provenance stay
recorded, so a wrong resolution is recoverable instead of destructive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from app.memory.sources import MemorySource, SOURCE_TRUST

# ── Keys ─────────────────────────────────────────────────────────────────────

CANONICAL_NAME_KEY = "canonical_name"
"""Who the user actually is. Written only by an identity update or by résumé
ingestion, and read for "what is my name"."""

REMEMBERED_NAME_KEY = "remembered_name"
"""A name the user asked to have remembered. Read for "what name did I ask you
to remember", and never for "what is my name"."""

EXPLICIT_MEMORY_PREFIX = "remembered_"
"""Namespace for everything the user explicitly asked to be kept. Keeping it
distinct from profile keys is what stops an explicit memory silently becoming a
profile attribute."""

# Identity attributes that may only be changed by an explicit update. Anything
# outside this set is an ordinary profile fact.
CANONICAL_KEYS = frozenset({
    CANONICAL_NAME_KEY, "canonical_email", "canonical_college",
    "canonical_degree", "canonical_location",
})


class DirectiveKind(str, Enum):
    """What a name-bearing sentence is actually asking for."""

    REMEMBER_VALUE = "remember_value"
    """Store this, under its own key. Identity untouched."""

    CANONICAL_UPDATE = "canonical_update"
    """Change who the user is. Requires an explicit identity-update clause."""


@dataclass(frozen=True)
class NameDirective:
    """A parsed instruction about a name."""

    kind: DirectiveKind
    value: str
    key: str
    reason: str


# "Remember the name X", "remember that my friend is called X", "save the name X".
_REMEMBER_NAME_RE = re.compile(
    r"\b(?:remember|save|store|note|keep)\s+(?:the\s+|this\s+|that\s+)?"
    r"name\s+(?:is\s+|as\s+)?([A-Za-z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3})",
    re.IGNORECASE,
)

# Explicit identity change. The identity words are mandatory — "my name is X"
# alone does NOT match, because people say it about friends, characters, and
# hypotheticals, and because the cost of a wrong identity write is unbounded.
_CANONICAL_UPDATE_RE = re.compile(
    r"\b(?:update|change|correct|fix|set|replace)\s+my\s+"
    r"(?:legal\s+|real\s+|full\s+|official\s+|canonical\s+)?name\s+"
    r"(?:to|as|into)\s+([A-Za-z][\w'’.-]*(?:\s+[\w'’.-]+){0,3})",
    re.IGNORECASE,
)

_CANONICAL_DECLARATION_RE = re.compile(
    r"\bmy\s+(?:legal|real|official|canonical|actual|correct|full)\s+name\s+is\s+"
    r"([A-Za-z][\w'’.-]*(?:\s+[\w'’.-]+){0,3})",
    re.IGNORECASE,
)

# Words that end a name when the sentence continues past it.
_NAME_STOPWORDS = frozenset({
    "and", "but", "please", "instead", "now", "then", "because", "so", "for",
    "in", "on", "at", "from", "with", "as", "the", "a", "an", "it", "that",
    "this", "also", "too", "okay", "ok", "thanks", "thank",
})


def _clean_name(raw: str) -> str:
    """Trim a captured name at the first non-name word."""
    words: list[str] = []
    for word in (raw or "").split():
        stripped = word.strip(".,;:!?\"'")
        if not stripped:
            break
        if stripped.lower() in _NAME_STOPWORDS:
            break
        words.append(stripped)
        if len(words) >= 4:
            break
    return " ".join(words).strip()


def detect_name_directive(text: str) -> Optional[NameDirective]:
    """
    Parse a sentence for an instruction about a name.

    Returns None for the overwhelmingly common case: a sentence that merely
    contains a name. That default is the safety property — silence here means
    identity is left alone.
    """
    if not text:
        return None

    for pattern, reason in (
        (_CANONICAL_UPDATE_RE, "explicit instruction to update the user's name"),
        (_CANONICAL_DECLARATION_RE, "user declared their legal/official name"),
    ):
        match = pattern.search(text)
        if match:
            value = _clean_name(match.group(1))
            if value:
                return NameDirective(
                    kind=DirectiveKind.CANONICAL_UPDATE,
                    value=value,
                    key=CANONICAL_NAME_KEY,
                    reason=reason,
                )

    match = _REMEMBER_NAME_RE.search(text)
    if match:
        value = _clean_name(match.group(1))
        if value:
            return NameDirective(
                kind=DirectiveKind.REMEMBER_VALUE,
                value=value,
                key=REMEMBERED_NAME_KEY,
                reason="user asked for a name to be remembered, not for their "
                       "identity to change",
            )

    return None


def explicit_key(label: str) -> str:
    """
    Namespaced key for something the user asked to be remembered.

    "Remember my gym is at 6am" becomes `remembered_gym`, not `gym` — so it can
    never be mistaken for a profile attribute the system derived itself.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", (label or "").strip().lower()).strip("_")
    slug = slug or "note"
    if slug.startswith(EXPLICIT_MEMORY_PREFIX):
        return slug
    return f"{EXPLICIT_MEMORY_PREFIX}{slug}"


def is_canonical_key(key: str) -> bool:
    """Whether writing this key changes who the user is."""
    return (key or "").strip().lower() in CANONICAL_KEYS


# Keys written before the `remembered_` namespace existed. A live store already
# holds `alternate_name: Devasis` — the residue of the bug this module fixes —
# and it is a legitimate remembered value that should still be readable rather
# than orphaned by the rename.
LEGACY_EXPLICIT_KEYS = frozenset({
    "alternate_name", "nickname", "preferred_name", "alias", "other_name",
})


def is_explicit_memory_key(key: str) -> bool:
    """Whether this key holds something the user asked to have remembered."""
    normalised = (key or "").strip().lower()
    return (
        normalised.startswith(EXPLICIT_MEMORY_PREFIX)
        or normalised in LEGACY_EXPLICIT_KEYS
    )


# ── Conflict resolution ──────────────────────────────────────────────────────

class Resolution(str, Enum):
    """What to do with an incoming value that disagrees with a stored one."""

    KEEP_EXISTING = "keep_existing"
    SUPERSEDE = "supersede"
    """Record the new value as current and retain the old one as superseded."""

    RECORD_ALONGSIDE = "record_alongside"
    """Both are legitimate and neither replaces the other — an alias next to a
    canonical name, for instance."""


@dataclass(frozen=True)
class ConflictVerdict:
    resolution: Resolution
    reason: str
    winner: str
    previous: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        """Log form. The values themselves are omitted — provenance only."""
        return {"resolution": self.resolution.value, "reason": self.reason}


@dataclass(frozen=True)
class FactClaim:
    """One source's assertion about an attribute."""

    key: str
    value: str
    source: MemorySource
    confidence: float = 1.0
    explicit: bool = False
    timestamp: Optional[datetime] = None

    @property
    def weight(self) -> float:
        """Trust in this claim: source rank, adjusted for explicitness."""
        base = float(SOURCE_TRUST.get(self.source, 0))
        if self.explicit:
            base += 15.0
        return base * max(0.1, min(1.0, self.confidence))


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def resolve_conflict(existing: FactClaim, incoming: FactClaim) -> ConflictVerdict:
    """
    Decide between two claims about the same attribute.

    The rules, in order:

    1. Same value — nothing to resolve.
    2. An explicit user correction supersedes an older source. People change
       colleges; a résumé from last year does not get to win forever.
    3. A non-explicit mention never overwrites a canonical or higher-trust
       source. This is the rule that keeps a passing "I study at X" from
       displacing a résumé, and it is deliberately asymmetric: the cost of
       wrongly keeping a stale fact is a correction, the cost of wrongly
       destroying one is unrecoverable.
    4. Equal trust — the newer claim wins.

    Nothing here deletes. `SUPERSEDE` means the previous value is retained with
    its provenance, which is what makes rule 3 safe to be strict about.
    """
    if existing.value.strip().lower() == incoming.value.strip().lower():
        return ConflictVerdict(
            Resolution.KEEP_EXISTING, "values agree", existing.value
        )

    if is_canonical_key(existing.key) and not incoming.explicit:
        return ConflictVerdict(
            Resolution.RECORD_ALONGSIDE,
            "canonical identity is not changed by an incidental mention",
            existing.value,
            previous=incoming.value,
        )

    if incoming.explicit and not existing.explicit:
        return ConflictVerdict(
            Resolution.SUPERSEDE,
            "explicit user statement supersedes a derived or inferred value",
            incoming.value,
            previous=existing.value,
        )

    if incoming.weight > existing.weight:
        return ConflictVerdict(
            Resolution.SUPERSEDE,
            f"higher-trust source ({incoming.source.value}) supersedes "
            f"{existing.source.value}",
            incoming.value,
            previous=existing.value,
        )

    if incoming.weight < existing.weight:
        return ConflictVerdict(
            Resolution.KEEP_EXISTING,
            f"{existing.source.value} outranks {incoming.source.value}",
            existing.value,
            previous=incoming.value,
        )

    # Equal trust: the claim being made *now* is the current one.
    #
    # The protection this whole function exists for is against a *weaker* source
    # overwriting a stronger one, and that is already settled above. Between two
    # claims of equal standing there is nothing left to protect — the user
    # correcting a value they themselves stated is the ordinary case, and
    # refusing it would freeze the first value ever written.
    #
    # Only a demonstrably *older* claim loses here. Equal timestamps supersede:
    # two writes in the same clock tick are not evidence that the stored one is
    # more current, and treating them as such made same-source corrections fail
    # whenever they happened faster than the clock's resolution.
    new_at = _as_utc(incoming.timestamp)
    old_at = _as_utc(existing.timestamp)
    if new_at and old_at and new_at < old_at:
        return ConflictVerdict(
            Resolution.KEEP_EXISTING,
            "equal trust; the stored statement is more recent",
            existing.value,
            previous=incoming.value,
        )

    return ConflictVerdict(
        Resolution.SUPERSEDE,
        "equal trust; the newer statement wins",
        incoming.value,
        previous=existing.value,
    )
