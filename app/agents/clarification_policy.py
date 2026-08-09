"""
When the assistant is allowed to ask a question back.

The observed failure was not one bad clarification, it was a loop. The user
asked how to build an AI agent; the assistant asked what kind; the user
answered; it asked again; the user said "don't ask too many questions"; it
asked again. Each question was individually defensible and the sequence was
useless — which is the signature of a rule applied per-turn with no memory of
the turn before.

So clarification is budgeted, per conversation, and the budget is one. A second
question about the same request means the first one failed to unblock anything,
and asking again will not either: at that point the right move is to state an
assumption and answer under it.

Three gates, all of which must pass:

    the category permits it   — a question about the user never does
    the budget is unspent     — one per conversation
    the user has not opted out — "just answer" ends it for the session

The budget lives in the process, keyed by conversation. That is deliberately
modest: it must survive the turns of one conversation, not a restart, and
persisting it would add a write to the request path for a counter whose worst
failure is one extra question after a redeploy.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from app.agents import query_intent
from app.memory.answerability import Answerability, Assessment
from app.memory.sources import QueryCategory, may_clarify

logger = logging.getLogger(__name__)

MAX_CLARIFICATION_ATTEMPTS = 1
"""One question per conversation. See the module docstring — a second is
evidence the first did not help."""

# conversation_id → (attempts, opted_out)
_state: Dict[str, Tuple[int, bool]] = {}
_lock = threading.Lock()

# Ceiling on tracked conversations, so a long-lived process cannot accumulate
# one entry per session forever. Oldest insertions are dropped first.
_MAX_TRACKED = 2000


@dataclass(frozen=True)
class ClarificationVerdict:
    """Whether to ask, and — when not — why not."""

    allowed: bool
    reason: str
    attempts: int = 0
    missing: str = ""

    def summary(self) -> Dict[str, object]:
        """Structured log form, matching the `clarification:` log shape."""
        return {
            "clarification": self.allowed,
            "reason": self.reason,
            "attempts": self.attempts,
            "missing": self.missing,
        }


def _get(conversation_id: str) -> Tuple[int, bool]:
    return _state.get(conversation_id, (0, False))


def _why_not(category: QueryCategory) -> str:
    """
    Why this category cannot ask a question.

    The reason is recorded, not just the refusal — a log line saying "answered
    by retrieval" against a date question would be misleading, and these reasons
    are what an operator reads when a suppression looks wrong.
    """
    if category is QueryCategory.TEMPORAL_CURRENT:
        return "the assistant has a clock; a date is never ambiguous"
    if category is QueryCategory.GENERAL_KNOWLEDGE:
        return "a request for an explanation is complete; answer with a stated assumption"
    if category is QueryCategory.SMALL_TALK:
        return "conversational turn; nothing to clarify"
    if category is QueryCategory.IDENTITY_UPDATE:
        return "an explicit identity instruction is unambiguous"
    return f"{category.value} is answered by retrieval, not by asking"


def note_user_message(conversation_id: str, text: str) -> bool:
    """
    Record an opt-out if the user asked for fewer questions.

    Returns whether the conversation is now opted out. Sticky by design: a
    person who has said "don't ask too many questions" has told you their
    preference for the conversation, not for one turn.
    """
    if not conversation_id:
        return False
    if not query_intent.asks_for_fewer_questions(text):
        return _get(conversation_id)[1]

    with _lock:
        attempts, _ = _state.get(conversation_id, (0, False))
        _state[conversation_id] = (attempts, True)
        _evict_if_needed()
    logger.info(
        "Clarification disabled for conversation=%s: user asked for fewer questions",
        conversation_id,
    )
    return True


def evaluate(
    conversation_id: str,
    category: QueryCategory,
    *,
    assessment: Optional[Assessment] = None,
) -> ClarificationVerdict:
    """
    Decide whether this turn may ask the user something.

    Does not consume the budget — call `record_attempt` when a question is
    actually asked, so a suppressed clarification costs nothing.
    """
    attempts, opted_out = _get(conversation_id or "")

    if not may_clarify(category):
        return ClarificationVerdict(False, _why_not(category), attempts)

    if assessment is not None and not assessment.may_clarify:
        # Evidence exists, or its absence is a fact. Either way there is
        # something honest to say, and saying it beats asking.
        return ClarificationVerdict(
            False,
            f"answerability={assessment.state.value} — answer rather than ask",
            attempts,
        )

    if opted_out:
        return ClarificationVerdict(
            False, "user asked not to be questioned; assume and answer", attempts
        )

    if attempts >= MAX_CLARIFICATION_ATTEMPTS:
        return ClarificationVerdict(
            False,
            f"clarification budget spent ({attempts}/{MAX_CLARIFICATION_ATTEMPTS})",
            attempts,
        )

    missing = ""
    if assessment is not None and assessment.missing:
        missing = ", ".join(assessment.missing)
    elif assessment is not None and assessment.state is Answerability.AMBIGUOUS:
        missing = "which item was meant"

    return ClarificationVerdict(
        True, "action is missing a parameter no store holds", attempts, missing
    )


def record_attempt(conversation_id: str) -> int:
    """Consume one unit of budget. Returns the new attempt count."""
    if not conversation_id:
        return 0
    with _lock:
        attempts, opted_out = _state.get(conversation_id, (0, False))
        attempts += 1
        _state[conversation_id] = (attempts, opted_out)
        _evict_if_needed()
    return attempts


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
