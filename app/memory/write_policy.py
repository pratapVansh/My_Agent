"""
What is worth remembering.

`SmartMemory.extract_and_store` embedded every user utterance verbatim. That is
how "You are fool.", "Okay." and "Once in a million chat." became long-term
memories with the same standing as the user's degree. The damage is not storage
cost — it is retrieval: every filler fragment is a competing neighbour in
embedding space, so recall for real questions degrades monotonically as the
conversation history grows. A year of small talk buries a résumé.

The policy is a gate, not a classifier of meaning:

    store when explicitly requested, or when the utterance states a durable
    fact about the user — and not otherwise.

Deterministic, because it runs on the request path for every turn including
spoken ones, and because the failure mode of an LLM gatekeeper here is to be
agreeable and store everything, which is the behaviour being removed.

It is deliberately conservative in the direction of *not* storing. A durable
fact missed today is stated again tomorrow; a store filled with "okay" is not
recoverable by any later pass. The batched LLM extractor in
`cognition/extractor.py` remains the path for nuanced facts — this gate only
decides whether an utterance is worth showing it at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Explicit instructions to remember. These always store, regardless of content:
# the user has taken responsibility for the judgement.
_EXPLICIT_REQUEST_RE = re.compile(
    r"\b(remember|dont\s+forget|do\s+not\s+forget|keep\s+in\s+mind"
    r"|make\s+a\s+note|take\s+a\s+note|note\s+that|save\s+(this|that|it)"
    r"|store\s+(this|that|it)|memori[sz]e|bear\s+in\s+mind)\b",
    re.IGNORECASE,
)

# Statements that assert something durable about the speaker. Shape-based:
# a first-person subject plus a stative or biographical predicate.
_DURABLE_FACT_RES = (
    # "I am a final-year student", "I'm studying CS", "I work at X"
    re.compile(r"\bi\s*(am|m|was)\b(?!\s+(fine|good|ok|okay|great|sorry|here|back|tired|busy|confused|not\s+sure))",
               re.IGNORECASE),
    re.compile(r"\bi\s+(study|studied|work|worked|live|lived|graduated|joined"
               r"|interned|built|created|founded|led|won|use|know|speak)\b", re.IGNORECASE),
    # Preferences and goals
    re.compile(r"\bi\s+(prefer|like|love|hate|dislike|want|need|plan|intend"
               r"|aim|hope|always|usually|never)\b", re.IGNORECASE),
    re.compile(r"\bmy\s+(name|email|phone|college|university|degree|branch|major"
               r"|cgpa|cpi|gpa|birthday|goal|goals|target|deadline|project"
               r"|projects|skills?|internship|job|role|company|timezone|address"
               r"|preference|preferences)\b", re.IGNORECASE),
    # "call me X", "I go by X"
    re.compile(r"\b(call\s+me|i\s+go\s+by|refer\s+to\s+me\s+as)\b", re.IGNORECASE),
)

# Pure filler. Matched against the whole utterance, so "thanks — my CGPA is 8.9"
# is not discarded as an acknowledgement.
_FILLER_RE = re.compile(
    r"^\s*(h(i|ey|ello)|yo|good\s+(morning|afternoon|evening|night)"
    r"|thanks?|thank\s+you|ty|ok(ay)?|kk|k|cool|nice|great|awesome|perfect"
    r"|yes|yeah|yep|ya|no|nope|nah|sure|fine|alright|right|got\s+it"
    r"|never\s?mind|nvm|stop|wait|hmm+|umm+|lol|haha|hehe|bye|goodbye"
    r"|see\s+you|good\s+night|please|sorry|my\s+bad|understood|done"
    r")[\s.!?,]*$",
    re.IGNORECASE,
)

# Directed at the assistant rather than about the user. Venting, praise and
# insults are all conversation — none of them are facts about a person.
_ABOUT_ASSISTANT_RE = re.compile(
    r"^\s*(you|u|your|ur)\b|^\s*(that|this|it)\s+(was|is)\s+"
    r"(wrong|right|good|bad|great|useless|helpful|stupid|dumb|amazing)",
    re.IGNORECASE,
)

# Questions are requests, not statements. "What is my CGPA?" must not be stored
# as though the user had told the system their CGPA.
_QUESTION_RE = re.compile(
    r"^\s*(what|which|who|whom|whose|when|where|why|how|is|are|was|were|do|does"
    r"|did|can|could|would|should|will|shall|may|might|have|has|had|tell|show"
    r"|give|find|search|explain|list)\b",
    re.IGNORECASE,
)

# Questions *about* stored memory. These carry the same verbs as a write
# instruction and mean the opposite.
_MEMORY_READBACK_RE = re.compile(
    r"\b(what|which|who|do|did|have|has|can)\b[^?]{0,60}\b"
    r"(remember|remembered|saved|stored|noted|memori[sz]ed)\b",
    re.IGNORECASE,
)

# Below this an utterance cannot be a self-contained durable fact.
_MIN_MEANINGFUL_WORDS = 4


@dataclass(frozen=True)
class WriteDecision:
    """Whether to keep an utterance, and why."""

    store: bool
    reason: str
    importance: float = 0.0
    explicit: bool = False

    def summary(self) -> dict:
        """Log form — the decision and its reason, never the text."""
        return {
            "store": self.store,
            "reason": self.reason,
            "importance": round(self.importance, 2),
            "explicit": self.explicit,
        }


def should_store(text: str, *, role: str = "user") -> WriteDecision:
    """
    Decide whether an utterance earns a place in long-term memory.

    Assistant turns are never stored as facts about the user. They were, and
    that is why recall surfaced the assistant's own prose as though the user had
    said it — a self-reinforcing loop where the model's guesses become the
    evidence for its next guess.
    """
    content = (text or "").strip()
    if not content:
        return WriteDecision(False, "empty utterance")

    if (role or "user").strip().lower() != "user":
        return WriteDecision(
            False, "assistant output is not evidence about the user"
        )

    # Apostrophes are flattened before matching. Without this "Don't forget my
    # exam" missed the explicit-request pattern entirely — and an explicit
    # request is the one category that must never be dropped.
    normalised = content.replace("’", "'").replace("'", "")

    if _MEMORY_READBACK_RE.search(normalised):
        # "What name did I ask you to remember?" contains the write marker and
        # is a read. Storing it files the question as though it were its own
        # answer, which is how a store fills up with its own queries.
        return WriteDecision(False, "asks what was remembered; not a new memory")

    if _EXPLICIT_REQUEST_RE.search(normalised):
        return WriteDecision(
            True, "user explicitly asked for this to be remembered",
            importance=0.95, explicit=True,
        )

    if _FILLER_RE.match(normalised):
        return WriteDecision(False, "conversational filler")

    # Checked before the length gate so the recorded reason is the true one:
    # "You are fool." is rejected because it is about the assistant, and would
    # still be rejected at greater length.
    if _ABOUT_ASSISTANT_RE.match(normalised):
        return WriteDecision(False, "about the assistant, not about the user")

    if len(content.split()) < _MIN_MEANINGFUL_WORDS:
        return WriteDecision(False, "too short to be a self-contained fact")

    # Requests are filtered before the fact patterns, not after. "Tell me about
    # my projects" contains "my projects" and would otherwise be filed as though
    # the user had *stated* something about them — storing the question as its
    # own answer.
    if _QUESTION_RE.match(normalised):
        return WriteDecision(False, "a request or question, not a statement of fact")

    for pattern in _DURABLE_FACT_RES:
        if pattern.search(normalised):
            return WriteDecision(
                True, "states a durable fact about the user", importance=0.7
            )

    return WriteDecision(False, "no durable personal fact detected")


def should_store_turn(user_text: str, assistant_text: Optional[str] = None) -> WriteDecision:
    """
    Whether an exchange is worth queueing for batched extraction.

    Looser than `should_store` on purpose: the extractor is the component that
    decides what the *fact* is, and it needs the assistant's half for context
    ("yeah, that one" is meaningless alone). This gate only removes exchanges
    where the user contributed nothing to extract from.
    """
    decision = should_store(user_text, role="user")
    if decision.store:
        return decision

    content = (user_text or "").strip()
    if not content or _FILLER_RE.match(content.replace("’", "'").replace("'", "")):
        return WriteDecision(False, "nothing extractable in this exchange")

    # Long or substantive user turns go to the extractor even when the fast
    # gate found no fact pattern — it can read nuance this cannot.
    if len(content.split()) >= 8:
        return WriteDecision(True, "substantive turn; defer to batched extraction",
                             importance=0.4)

    return WriteDecision(False, "too slight to be worth extracting from")
