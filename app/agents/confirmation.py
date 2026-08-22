"""
Turning "yes" into an execution — without ever asking a model what "yes" meant.

The gateway made consequential actions unreachable from the reasoning loop.
That left a gap: nothing could approve one. This closes it, and the whole
design follows from a single refusal — the language model does not get a vote
on whether a message was consent.

That refusal is not stylistic. Consent is the one input where a plausible
misreading is indistinguishable from an attack: a model that decides "okay,
whatever" meant yes produces the same bytes as a model that was talked into
deciding it. So the classification here is a closed set of patterns matched
against the whole utterance, and everything outside that set is not consent.

Three states, and the middle one is the important one:

    AFFIRM      an unambiguous yes                 → execute, via the gateway
    REJECT      an unambiguous no                  → cancel, execute nothing
    AMBIGUOUS   "okay", "fine", "sure", "maybe"    → ask again, execute nothing

"Sure" and "okay" sit in AMBIGUOUS deliberately. Both read as agreement in
ordinary conversation and neither is worth an irreversible action: the cost of
asking twice is a sentence, the cost of being wrong is an email that cannot be
recalled. `NONE` — anything with content of its own — falls through to normal
routing, so a pending action never hijacks a real question.

Matching is anchored to the entire utterance for the same reason. "Yes" is
consent; "yes, but first tell me what my CGPA is" is a question that happens to
start with it, and treating the two alike would let any sentence containing an
affirmative word approve whatever happened to be pending.

This module is imported by the graph node and by the streaming path, and it is
the only implementation of any of this. Voice and text differ in how the words
arrive, not in what they mean.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from app.agents.actions import PendingAction, action_gateway
from app.tools.contract import ToolResult

logger = logging.getLogger(__name__)


class ConfirmationIntent(str, Enum):
    """What a reply to a confirmation request actually said."""

    AFFIRM = "affirm"
    REJECT = "reject"
    AMBIGUOUS = "ambiguous"
    """Recognisably a response, but not a decision. Never executes."""

    NONE = "none"
    """Not a response to the pending action at all — an ordinary message."""


# Filler that can precede a decision without changing it: "uh, yes",
# "ok send it", "alright go ahead". Note `ok`/`okay`/`sure`/`alright` appear
# here *and* in the ambiguous set: leading one of them onto a real decision is
# fine, but on their own they decide nothing.
_LEAD = r"(?:(?:uh|um|er|ah|oh|well|so|hmm|ok|okay|alright|sure|right|yes|yeah|yep|yup)\s+)*"

# Trailing politeness, which likewise changes nothing.
_TAIL = r"(?:\s+(?:please|now|then|thanks|thank\s+you|mate|buddy))*"

_AFFIRM_CORE = (
    r"yes|yeah|yep|yup|yes\s+please"
    r"|confirm(?:ed|\s+it|\s+that)?|approve[d]?|approved"
    r"|proceed|go\s+ahead|go\s+for\s+it|ship\s+it"
    r"|do\s+it|do\s+that|please\s+do"
    r"|send\s+it(?:\s+now)?|send\s+that|send\s+the\s+email|send\s+it\s+please"
    r"|thats\s+right\s+send\s+it|looks\s+good\s+send\s+it|sounds\s+good\s+send\s+it"
)

_REJECT_CORE = (
    r"no|nope|nah|no\s+thanks?|no\s+thank\s+you"
    r"|cancel(?:\s+it|\s+that)?|cancelled"
    r"|dont(?:\s+send)?(?:\s+it)?|do\s+not(?:\s+send)?(?:\s+it)?"
    r"|dont\s+do\s+it|do\s+not\s+do\s+it"
    r"|forget\s+it|forget\s+that|reject(?:\s+it)?|rejected"
    r"|abort|stop|never\s?mind|nevermind"
    r"|discard(?:\s+it)?|scrap\s+it|throw\s+it\s+away"
    r"|not\s+now|hold\s+off|wait\s+dont\s+send"
)

# Recognisably a reply, decidedly not a decision.
_AMBIGUOUS_CORE = (
    r"ok|okay|k|kk|fine|sure|alright|right|maybe|perhaps|possibly"
    r"|whatever|do\s+whatever|up\s+to\s+you|your\s+call|you\s+decide"
    r"|i\s+guess|i\s+suppose|probably|i\s+think\s+so|why\s+not"
    r"|hmm+|mhm|uh\s?huh|mm+|yeah\s+maybe|sure\s+i\s+guess"
)

_AFFIRM_RE = re.compile(rf"^{_LEAD}(?:{_AFFIRM_CORE}){_TAIL}$")
_REJECT_RE = re.compile(rf"^{_LEAD}(?:{_REJECT_CORE}){_TAIL}$")
_AMBIGUOUS_RE = re.compile(rf"^(?:{_AMBIGUOUS_CORE}){_TAIL}$")


def _normalise(text: str) -> str:
    """Lowercase, apostrophes dropped ("don't" → "dont"), punctuation stripped."""
    lowered = (text or "").lower().replace("’", "'").replace("'", "")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def detect(text: str) -> ConfirmationIntent:
    """
    Classify a reply. Never consults a model, never sees the pending action.

    Order matters only between reject and ambiguous: "no" must not be reachable
    by the ambiguous branch. Affirm is tested first because the lead-in group
    it shares with reject is greedy.
    """
    cleaned = _normalise(text)
    if not cleaned:
        return ConfirmationIntent.NONE

    # Long utterances carry their own content and are not bare decisions. The
    # ceiling is generous enough for "okay yes go ahead and send it please"
    # and far short of a real sentence.
    if len(cleaned.split()) > 8:
        return ConfirmationIntent.NONE

    if _REJECT_RE.match(cleaned):
        return ConfirmationIntent.REJECT
    if _AFFIRM_RE.match(cleaned):
        return ConfirmationIntent.AFFIRM
    if _AMBIGUOUS_RE.match(cleaned):
        return ConfirmationIntent.AMBIGUOUS
    return ConfirmationIntent.NONE


# ── Routing helper ───────────────────────────────────────────────────────────

async def pending_for_state(state: Dict[str, Any]) -> List[PendingAction]:
    """Live actions awaiting this caller's decision in this conversation."""
    return await action_gateway.pending_for(
        state.get("session_id") or "", state.get("user_id") or ""
    )


async def should_intercept(state: Dict[str, Any]) -> bool:
    """
    Whether this turn is a reply to a pending action rather than a new request.

    Both halves are required. Without a pending action, "yes" is an ordinary
    conversational message and must be routed as one — the route may not exist
    merely because someone said a word. Without a recognised reply, a real
    question is never swallowed by a confirmation that happens to be open.
    """
    # Cheapest check first: without a recognised reply there is nothing to
    # intercept, and asking the database on every turn would put a query on the
    # path of every ordinary message.
    if detect(state.get("user_input", "")) is ConfirmationIntent.NONE:
        return False
    return bool(await pending_for_state(state))


# ── Resolution ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConfirmationResult:
    """The outcome of one confirmation turn, in a form both paths can render."""

    text: str
    intent: ConfirmationIntent
    executed: bool = False
    cancelled: bool = False
    tool: str = ""
    result: Optional[ToolResult] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "executed": self.executed,
            "cancelled": self.cancelled,
            "tool": self.tool,
        }


_AMBIGUOUS_REPLY = (
    "I need a clear yes or no before I do that. Reply \"yes\" to go ahead, or "
    "\"no\" to cancel — nothing has happened yet."
)

_NOTHING_PENDING = (
    "There's nothing waiting for your approval right now."
)


def _describe(action: PendingAction) -> str:
    """One short line naming an action, for the disambiguation prompt."""
    first = (action.preview or "").strip().splitlines()
    headline = next((line.strip() for line in first if line.strip()), action.tool)
    return f"{action.tool}: {headline[:120]}"


async def resolve(state: Dict[str, Any]) -> ConfirmationResult:
    """
    Carry out a confirmation turn.

    The one place a held action is approved or cancelled, shared by the graph
    node and the streaming path. Execution goes through
    `ActionGateway.confirm_and_execute` and nowhere else — this function never
    holds a tool callable and could not invoke one if it tried.
    """
    intent = detect(state.get("user_input", ""))
    owner_id = state.get("user_id") or ""
    pending = await pending_for_state(state)

    if not pending:
        # Raced: the action expired or was confirmed elsewhere between routing
        # and here. Reporting it plainly beats executing something stale.
        return ConfirmationResult(_NOTHING_PENDING, intent)

    if intent is ConfirmationIntent.AMBIGUOUS:
        return ConfirmationResult(_AMBIGUOUS_REPLY, intent)

    if len(pending) > 1:
        # Never guess which one. Approving the wrong irreversible action is
        # exactly as bad as approving one nobody asked for.
        lines = [
            f"You have {len(pending)} actions waiting. Which one do you mean?",
            "",
        ]
        lines += [f"  {index}. {_describe(a)}" for index, a in enumerate(pending, 1)]
        lines.append("")
        lines.append("Nothing has been done. Tell me which one and I'll confirm it.")
        return ConfirmationResult("\n".join(lines), intent)

    action = pending[0]

    if intent is ConfirmationIntent.REJECT:
        # Referenced by handle, not by token: an action reloaded from the
        # database has no raw token, which is the whole point of the handle.
        await action_gateway.cancel(handle=action.handle, owner_id=owner_id)
        logger.info("Action rejected by user: %s", action.summary())
        return ConfirmationResult(
            f"Cancelled. I did not run {action.tool}, and nothing was sent.",
            intent, cancelled=True, tool=action.tool,
        )

    if intent is not ConfirmationIntent.AFFIRM:
        # Defensive: `should_intercept` filters NONE out before routing here.
        return ConfirmationResult(_AMBIGUOUS_REPLY, ConfirmationIntent.AMBIGUOUS)

    # The content hash is passed explicitly. The token alone would already be
    # bound to this action; re-presenting the hash means an action mutated
    # between preview and approval cannot be executed under the old consent.
    result = await action_gateway.confirm_and_execute(
        handle=action.handle, owner_id=owner_id, content_hash=action.content_hash
    )

    if result.ok:
        logger.info("Action executed after confirmation: %s", result.summary())
        return ConfirmationResult(
            _success_text(action, result), intent,
            executed=True, tool=action.tool, result=result,
        )

    detail = result.error.message if result.error else "it could not be completed"
    logger.warning("Confirmed action did not succeed: %s", result.summary())
    return ConfirmationResult(
        f"I couldn't complete that: {detail}", intent, tool=action.tool, result=result,
    )


def _success_text(action: PendingAction, result: ToolResult) -> str:
    """
    A short, concrete confirmation of what happened.

    Names the actual target rather than saying "done", so the user can catch a
    wrong recipient immediately rather than the next time they open their sent
    folder.
    """
    args = action.arguments or {}
    if action.tool == "send_email":
        recipient = args.get("to_email") or "the recipient"
        return f"Sent to {recipient}."
    if action.tool == "forget_preference":
        return f"Deleted \"{args.get('key', '')}\" from your saved memories."
    return f"Done — {action.tool} completed."


__all__ = [
    "ConfirmationIntent",
    "ConfirmationResult",
    "detect",
    "pending_for_state",
    "resolve",
    "should_intercept",
]
