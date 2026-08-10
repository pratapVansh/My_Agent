"""
What produced the last answer, recorded so it can be explained.

"How did you know that?" is a question about the *previous* turn, and it is the
one question a language model cannot answer by thinking harder. Asked to
explain itself, a model reconstructs a plausible account of its own reasoning —
and a plausible account of a retrieval that did not happen is a fabrication
about the user's own data, which is the failure mode this whole system is built
to avoid.

There are two wrong ways to answer it and one right one:

    re-retrieve now      → describes where the answer *would* come from today,
                           which differs from where it came from whenever
                           routing, memory, or the résumé has changed since
    ask the model        → invents provenance it never had
    read what was logged → the only account that is actually true

So every answered turn writes down its category, its sources, its agent and the
tools it actually called, and a provenance question reads that record back. The
record is the claim; nothing is inferred from it.

When there is no record — the first turn of a conversation, or a process that
restarted — the honest answer is that it is not known, and `explain()` says so
rather than guessing. That is the same NO_DATA discipline `answerability`
applies to the user's facts, applied to the assistant's own.

The store is per-process and keyed by conversation, deliberately matching
`clarification_policy`: it must survive the turns of one conversation, not a
restart, and persisting it would put a write on the request path for a record
whose worst failure is one honest "I don't have that recorded".
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from app.memory.sources import MemorySource

# conversation_id → the provenance of the most recent answered turn.
_state: Dict[str, "AnswerProvenance"] = {}
_lock = threading.Lock()

_MAX_TRACKED = 2000
"""Ceiling on tracked conversations. Oldest insertions are dropped first."""


# How each source is described to a person. Named concretely — "your uploaded
# résumé" rather than "resume_document" — because this text is spoken aloud on
# the voice path, and because a user asking where a number came from is asking
# which of *their* documents it was in.
_SOURCE_PHRASES: Dict[str, str] = {
    MemorySource.CANONICAL_IDENTITY.value: "your saved profile, under the name you confirmed as canonical",
    MemorySource.EXPLICIT_MEMORY.value: "the notes you explicitly asked me to remember",
    MemorySource.RESUME_DOCUMENT.value: "your uploaded résumé",
    MemorySource.PROFILE_MEMORY.value: "your stored profile facts",
    MemorySource.CONVERSATION_CURRENT.value: "earlier in this conversation",
    MemorySource.EPISODIC_MEMORY.value: "summaries of your earlier sessions",
    MemorySource.SEMANTIC_MEMORY.value: "your long-term memory index",
    MemorySource.TEMPORAL_TOOL.value: "the system clock",
    MemorySource.EXTERNAL_TOOL.value: "your timetable and attendance records",
    MemorySource.GENERAL_KNOWLEDGE.value: "my own general knowledge, not anything stored about you",
    MemorySource.NONE.value: "no stored source",
}

NO_RECORD = (
    "I don't have a record of where that came from, so I can't tell you "
    "reliably rather than guess."
)
"""Said when nothing was recorded. Deliberately an admission, not a
reconstruction — see the module docstring."""


@dataclass(frozen=True)
class AnswerProvenance:
    """The recorded origin of one answer."""

    category: str = ""
    sources: Tuple[str, ...] = ()
    """Sources in the precedence order they were consulted."""

    agent: str = ""
    tools: Tuple[str, ...] = ()
    """Tools actually invoked, as reported by the agent — not the tools it was
    offered. The distinction is the whole value of the record."""

    answerability: str = ""
    question: str = ""
    """The question that was answered, so the explanation can name it."""

    metadata: Dict[str, Any] = field(default_factory=dict)

    def explain(self) -> str:
        """
        A sentence naming where the previous answer actually came from.

        Composed from the record only. Every clause is something that was
        observed and written down at the time.
        """
        if not self.sources and not self.tools:
            return NO_RECORD

        primary = self.sources[0] if self.sources else ""
        phrase = _SOURCE_PHRASES.get(primary, primary.replace("_", " ") or "an unrecorded source")

        lead = f"I got that from {phrase}"
        if self.tools:
            listed = ", ".join(self.tools)
            lead += f", via the {listed} tool" if len(self.tools) == 1 else f", via these tools: {listed}"

        if self.answerability == "NO_DATA":
            return (
                f"I checked {phrase} and it had nothing on file — that's why I "
                f"said I didn't have it, rather than estimating a value."
            )
        if self.answerability == "TOOL_ERROR":
            return (
                f"I tried {phrase} and the lookup failed, so I couldn't confirm "
                f"it either way."
            )

        return lead + "."

    def summary(self) -> Dict[str, Any]:
        """Structured log form — provenance only, never the answer's content."""
        return {
            "category": self.category,
            "sources": list(self.sources),
            "agent": self.agent,
            "tools": list(self.tools),
            "answerability": self.answerability,
        }


def record(
    conversation_id: str,
    *,
    category: str = "",
    sources: Optional[Sequence[str]] = None,
    agent: str = "",
    tools: Optional[Sequence[str]] = None,
    answerability: str = "",
    question: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> AnswerProvenance:
    """
    Write down where this turn's answer came from.

    Called on every answered turn. A provenance question is answered from the
    most recent record, so this must run for the turn *being* answered — not
    only for turns someone might later ask about.
    """
    entry = AnswerProvenance(
        category=category or "",
        sources=tuple(str(s) for s in (sources or []) if s),
        agent=agent or "",
        tools=tuple(str(t) for t in (tools or []) if t),
        answerability=answerability or "",
        question=question or "",
        metadata=dict(metadata or {}),
    )
    if not conversation_id:
        return entry
    with _lock:
        _state[conversation_id] = entry
        _evict_if_needed()
    return entry


def last(conversation_id: str) -> Optional[AnswerProvenance]:
    """The provenance of the most recent answered turn, if one was recorded."""
    if not conversation_id:
        return None
    return _state.get(conversation_id)


def explain_last(conversation_id: str) -> str:
    """Explain the previous answer's origin, or admit that it is not recorded."""
    entry = last(conversation_id)
    return entry.explain() if entry else NO_RECORD


def reset(conversation_id: Optional[str] = None) -> None:
    """Clear tracking — for a new conversation, or for tests."""
    with _lock:
        if conversation_id is None:
            _state.clear()
        else:
            _state.pop(conversation_id, None)


def _evict_if_needed() -> None:
    """Drop the oldest entries once tracking exceeds its ceiling."""
    overflow = len(_state) - _MAX_TRACKED
    if overflow <= 0:
        return
    for key in list(_state)[:overflow]:
        _state.pop(key, None)
