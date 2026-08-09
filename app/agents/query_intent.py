"""
Deterministic query categorisation — the layer that decides where to look.

`profile_intent` answers one question: is this about the user? That was enough
to stop the assistant interrogating people about their own CGPA, but it leaves
four categories mis-served, and all four showed up in real conversations:

* **The clock.** "What is today's date?" is not a memory question, and answering
  it from memory is guaranteed to be wrong eventually. It was routed to
  retrieval, found nothing, and reported having "no real-time access".
* **Explicit memory.** "Remember the name Devasi" is a *write*, and it is not a
  write to identity. Treated as a profile statement, it overwrote the user's
  name — the single most damaging thing this system can get wrong.
* **General knowledge.** "Tell me how to build an AI agent" needs an answer, not
  three rounds of scoping questions.
* **Follow-ups.** "Tell me more about that" was re-classified from zero every
  turn, discarding the subject established one message earlier.

Everything here is decided by grammatical shape and vocabulary, never by an LLM
call: this runs before the planner on every turn, including spoken ones, so it
must cost microseconds. The LLM keeps the genuinely ambiguous residue — which
is what `GENERAL_KNOWLEDGE` and `ACTION_REQUEST` hand back to it.

The categories and their source precedence live in `app.memory.sources`; this
module only decides which category a sentence is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.agents import profile_intent
from app.memory.sources import (
    MemorySource,
    QueryCategory,
    may_clarify,
    requires_retrieval,
    sources_for,
)

# ── Vocabulary ───────────────────────────────────────────────────────────────

# Nouns that make a question about the calendar or the clock itself.
_TEMPORAL_SUBJECTS: Set[str] = {
    "date", "time", "day", "month", "year", "weekday", "clock", "hour",
    "week", "datetime",
}

# Words that anchor a question to *now*. "current" is included, but a temporal
# subject is also required — "my current CPI" is a stored fact about the user,
# not a question about the present moment, and conflating the two is exactly
# how "what is my current CPI" became "I don't have real-time access".
_NOW_MARKERS: Set[str] = {
    "today", "tonight", "now", "currently", "current", "present",
    "tomorrow", "yesterday", "instant", "moment",
}

# Things that live in a timetable. These need the clock *and* memory: the clock
# resolves "today", memory holds the schedule, and neither alone can answer.
_SCHEDULE_SUBJECTS: Set[str] = {
    "class", "classes", "lecture", "lectures", "lab", "labs", "exam", "exams",
    "timetable", "schedule", "period", "periods", "test", "tests", "quiz",
    "attendance", "assignment", "assignments", "deadline", "deadlines",
}

# An explicit instruction to store something. "remember", on its own, is
# ambiguous between a write ("remember my birthday is May 2") and a read ("do
# you remember my birthday?"); the interrogative forms are excluded below.
_MEMORY_WRITE_MARKERS: Tuple[str, ...] = (
    "remember", "dont forget", "do not forget", "keep in mind", "note that",
    "make a note", "take a note", "save this", "store this", "memorize",
    "memorise", "keep this in mind", "bear in mind",
)

# Reading back what the user asked to have remembered. These are deliberately
# phrase-level: the difference between "what is my name" and "what name did I
# ask you to remember" is the clause, not any single word.
_MEMORY_READ_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\b(what|which)\b.{0,40}\b(did|have)\s+i\s+(ask|tell|told|want)"
               r"\w*\s+(you\s+)?to\s+(remember|save|store|note|keep)"),
    re.compile(r"\bwhat\b.{0,30}\b(did|have)\s+(you|i)\s+remember"),
    re.compile(r"\bwhat\s+(name|word|number|thing|value)\b.{0,30}\bremember"),
    re.compile(r"\bwhat\b.{0,30}\bi\s+(asked|told)\s+you\s+to\s+remember"),
    re.compile(r"\bwhat\s+(have\s+)?you\s+(saved|stored|remembered|noted)\b"),
    re.compile(r"\bwhat\s+did\s+i\s+ask\s+you\s+to\s+(remember|save|store)\b"),
)

# An explicit instruction to change who the user *is*. Narrow by design. A
# sentence that merely contains a name is never one of these — that mistake is
# what turned "remember the name Devasi" into an identity change.
_IDENTITY_UPDATE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\b(update|change|correct|fix|set|replace)\s+(my|the)\s+"
               r"(legal\s+|real\s+|full\s+|official\s+|canonical\s+)?name\b"),
    re.compile(r"\bmy\s+(legal|real|official|canonical|actual|correct|full)\s+name\s+is\b"),
    re.compile(r"\b(my\s+name\s+is\b.{0,60}\b(update|change|correct)\s+(it|my\s+name))"),
    re.compile(r"\b(i\s+(have\s+)?(legally\s+)?(changed|renamed))\b.{0,20}\bname\b"),
)

# Asking about this thread, right now.
_CURRENT_CONVERSATION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\bwhat\s+did\s+i\s+just\s+(say|tell|ask|mention)"),
    re.compile(r"\bwhat\s+(did|have)\s+(i|we)\s+just\s+(said|discussed|talked)"),
    re.compile(r"\b(repeat|say)\s+(that|it)\s+(again|back)"),
    re.compile(r"\bwhat\s+was\s+my\s+(last|previous)\s+(question|message)"),
)

# Asking about a past session. The marker is a *past* time reference attached to
# a speech verb — "what did I tell you yesterday", not "what did I tell you".
_PAST_TIME_MARKERS: Set[str] = {
    "yesterday", "earlier", "before", "previously", "last", "ago", "past",
    "other", "recently",
}
_SPEECH_VERBS: Set[str] = {
    "told", "tell", "said", "say", "mentioned", "mention", "discussed",
    "discuss", "talked", "talk", "asked", "ask", "shared", "spoke",
}

# Verbs that ask for something to happen in the world. Shared in spirit with
# profile_intent's list; kept separate because this one also drives the
# ACTION_REQUEST / AMBIGUOUS_ACTION split, which that module has no notion of.
_ACTION_VERBS: Set[str] = {
    "send", "email", "mail", "forward", "reply", "schedule", "book", "cancel",
    "apply", "submit", "call", "message", "text", "remind", "delete", "remove",
    "order", "buy", "pay", "share", "post", "upload", "download", "install",
    "draft", "compose", "register", "enroll", "sign",
}

# Pronouns with no antecedent inside the sentence. An action carrying one of
# these is missing a parameter no store holds.
_DANGLING_REFERENTS: Set[str] = {
    "him", "her", "them", "it", "this", "that", "those", "these", "one",
}

# Openers that inherit their subject from the previous turn.
_FOLLOWUP_MARKERS: Set[str] = {
    "it", "that", "this", "those", "these", "there", "then", "one", "ones",
    "more", "again", "also", "too", "else", "other", "another", "second",
    "third", "first", "last",
}

# Pure conversational filler. Matched at the start of the utterance so that
# "thanks, now what is my CGPA" is not mistaken for an acknowledgement.
_SMALL_TALK_RE = re.compile(
    r"^\s*(h(i|ey|ello)\b|yo\b|good\s+(morning|afternoon|evening|night)"
    r"|thanks?\b|thank\s+you|ok(ay)?\b|kk\b|cool\b|nice\b|great\b|awesome\b"
    r"|yes\b|yeah\b|yep\b|no\b|nope\b|nah\b|sure\b|got\s+it|never\s?mind"
    r"|how\s+are\s+you|whats?\s+up|hows?\s+it\s+going"
    r"|you\s+(are|re)\s+(a\s+)?(fool|stupid|dumb|useless|amazing|great|good|bad)"
    r"|shut\s+up|lol\b|haha|hmm+\b|bye\b|goodbye|see\s+you)",
    re.IGNORECASE,
)

# Requests for an explanation of something in the world. These must be answered,
# not scoped: "tell me how to build an AI agent" is a complete question.
_GENERAL_KNOWLEDGE_RE = re.compile(
    r"\b(how\s+(do|can|would|should|does|to)|what\s+is\s+a|what\s+are\s+the"
    r"|explain|describe\s+how|teach\s+me|tell\s+me\s+how|guide\s+me"
    r"|what.{0,12}\bdifference\s+between|why\s+(do|does|is|are)"
    r"|steps\s+to|best\s+way\s+to|help\s+me\s+(build|understand|learn))\b",
    re.IGNORECASE,
)

# The user asking to be interrupted less. Once seen, clarification is off for
# the rest of the conversation — repeating a question they just declined is the
# behaviour they were complaining about.
_NO_QUESTIONS_RE = re.compile(
    r"\b(dont|do\s+not|stop|no\s+more|quit)\s+(ask\w*|question\w*)"
    r"|\bask\w*\s+(me\s+)?(too\s+many|fewer|less|no)\s+question"
    r"|\bjust\s+(answer|tell\s+me|give\s+me)\b"
    r"|\bwithout\s+(asking|questions)\b",
    re.IGNORECASE,
)

# How profile_intent's labels map onto the source-aware categories.
_PROFILE_CATEGORY_MAP: Dict[str, QueryCategory] = {
    profile_intent.PROFILE_NAME: QueryCategory.PROFILE_IDENTITY,
    profile_intent.PROFILE_EDUCATION: QueryCategory.PROFILE_EDUCATION,
    profile_intent.PROFILE_CGPA: QueryCategory.PROFILE_EDUCATION,
    profile_intent.PROFILE_COLLEGE: QueryCategory.PROFILE_EDUCATION,
    profile_intent.PROFILE_SKILLS: QueryCategory.PROFILE_SKILLS,
    profile_intent.PROFILE_PROJECTS: QueryCategory.PROFILE_PROJECTS,
    profile_intent.PROFILE_EXPERIENCE: QueryCategory.PROFILE_EXPERIENCE,
    profile_intent.PROFILE_INTERNSHIP: QueryCategory.PROFILE_EXPERIENCE,
    profile_intent.PROFILE_ACHIEVEMENTS: QueryCategory.PROFILE_ACHIEVEMENTS,
    profile_intent.PROFILE_GENERAL: QueryCategory.PROFILE_GENERAL,
    profile_intent.CONVERSATION_FOLLOWUP: QueryCategory.CONVERSATION_FOLLOWUP,
}

# Which specialist owns each category. Academic keeps timetable and attendance;
# everything personal belongs to profile, which owns résumé retrieval.
_CATEGORY_AGENT: Dict[QueryCategory, str] = {
    QueryCategory.SCHEDULE_TEMPORAL: "academic",
    QueryCategory.TEMPORAL_CURRENT: "temporal",
}


@dataclass(frozen=True)
class QueryDecision:
    """What kind of question this is, and what follows from that."""

    category: QueryCategory
    sources: Tuple[MemorySource, ...]
    confidence: float
    deterministic: bool
    reason: str
    profile_intent: Optional[str] = None
    """The legacy label, kept so the profile agent's tool mapping still works."""

    subject: Optional[str] = None
    """For follow-ups: the entity inherited from the conversation."""

    @property
    def requires_retrieval(self) -> bool:
        return requires_retrieval(self.category)

    @property
    def may_clarify(self) -> bool:
        return may_clarify(self.category)

    @property
    def is_personal(self) -> bool:
        return self.category.value.startswith("PROFILE_") or self.category in (
            QueryCategory.DOCUMENT_RESUME,
            QueryCategory.EXPLICIT_MEMORY,
            QueryCategory.EPISODIC_MEMORY,
            QueryCategory.CONVERSATION_CURRENT,
            QueryCategory.CONVERSATION_FOLLOWUP,
        )

    def summary(self) -> Dict[str, Any]:
        """Structured form for logging — category and provenance, never content."""
        return {
            "intent": self.category.value,
            "sources": [s.value for s in self.sources],
            "confidence": round(self.confidence, 2),
            "deterministic": self.deterministic,
            "reason": self.reason,
        }


def _decide(
    category: QueryCategory,
    reason: str,
    confidence: float = 0.9,
    *,
    deterministic: bool = True,
    legacy: Optional[str] = None,
    subject: Optional[str] = None,
) -> QueryDecision:
    return QueryDecision(
        category=category,
        sources=sources_for(category),
        confidence=confidence,
        deterministic=deterministic,
        reason=reason,
        profile_intent=legacy,
        subject=subject,
    )


def _normalise(query: str) -> str:
    """Lowercased text with possessives and punctuation flattened for matching."""
    text = (query or "").lower().replace("’", "'")
    text = re.sub(r"'s\b", "", text).replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def asks_for_fewer_questions(query: str) -> bool:
    """Whether the user has told the assistant to stop asking things."""
    return bool(_NO_QUESTIONS_RE.search(_normalise(query)))


def classify(
    query: str,
    *,
    has_context: bool = False,
    history: Optional[Sequence[Dict[str, Any]]] = None,
) -> QueryDecision:
    """
    Categorise one utterance.

    `has_context` reports whether earlier turns exist; `history` supplies them
    when a follow-up's subject needs resolving. Both are optional — with neither,
    referential questions simply decline to claim a subject they cannot see.

    The order of the checks is the design. Temporal beats profile because "the
    date today" is never a stored fact; identity-update beats memory-write
    because "update my name to X" is both; memory-write beats profile because
    "remember the name Devasi" contains the word "name" and is emphatically not
    a question about the user's name.
    """
    text = _normalise(query)
    if not text:
        return _decide(QueryCategory.SMALL_TALK, "empty utterance", 0.5)

    tokens = profile_intent.tokens(query)
    token_set = set(tokens)
    has_context = has_context or bool(history)

    # ── 1. An explicit identity change, before anything else claims it ───────
    for pattern in _IDENTITY_UPDATE_PATTERNS:
        if pattern.search(text):
            return _decide(
                QueryCategory.IDENTITY_UPDATE,
                "explicit instruction to change canonical identity",
                0.95,
            )

    # ── 2. Reading back an explicitly stored memory ──────────────────────────
    # Before the write check: "what did I ask you to remember" contains
    # "remember" and is a question, not an instruction.
    for pattern in _MEMORY_READ_PATTERNS:
        if pattern.search(text):
            return _decide(
                QueryCategory.EXPLICIT_MEMORY,
                "asks what the user requested be remembered",
                0.95,
            )

    # ── 3. An instruction to store something ─────────────────────────────────
    if _is_memory_write(text):
        return _decide(
            QueryCategory.EXPLICIT_MEMORY_WRITE,
            "explicit request to remember something",
            0.95,
        )

    # ── 4. The timetable: clock plus memory, never one alone ─────────────────
    if token_set & _SCHEDULE_SUBJECTS and (
        token_set & _NOW_MARKERS or token_set & profile_intent._OWNERSHIP_TOKENS
    ):
        return _decide(
            QueryCategory.SCHEDULE_TEMPORAL,
            "schedule question anchored to a day",
            0.9,
        )

    # ── 5. The clock itself ──────────────────────────────────────────────────
    if _is_temporal(token_set):
        return _decide(
            QueryCategory.TEMPORAL_CURRENT,
            "asks for the current date or time",
            0.95,
        )

    # ── 6. This conversation ─────────────────────────────────────────────────
    for pattern in _CURRENT_CONVERSATION_PATTERNS:
        if pattern.search(text):
            return _decide(
                QueryCategory.CONVERSATION_CURRENT,
                "refers to the immediately preceding turns",
                0.9,
            )

    # ── 7. An earlier session ────────────────────────────────────────────────
    if token_set & _PAST_TIME_MARKERS and token_set & _SPEECH_VERBS:
        return _decide(
            QueryCategory.EPISODIC_MEMORY,
            "refers to something said in an earlier session",
            0.85,
        )

    # ── 7b. A standing request to be asked fewer questions ───────────────────
    # Handled before the profile check because "don't ask too many questions"
    # carries referential words ("too", "many") that would otherwise claim it as
    # a follow-up. The opt-out itself is recorded by the clarification policy;
    # this only stops the sentence being routed as though it were a question.
    if asks_for_fewer_questions(text):
        provisional = profile_intent.classify(query, has_context=has_context)
        if provisional in (None, profile_intent.CONVERSATION_FOLLOWUP):
            return _decide(
                QueryCategory.SMALL_TALK,
                "instruction about how to answer, not a question",
                0.9,
            )

    # ── 7c. How-to questions about the world ─────────────────────────────────
    # Before the profile check, and only without a possessive. "How do I build
    # an AI agent" contains "I" and "build", which the profile classifier reads
    # as a question about the user's projects — that misreading is why a request
    # for an explanation came back as a round of scoping questions. "How do I
    # improve my CGPA" keeps its possessive and stays personal.
    if _GENERAL_KNOWLEDGE_RE.search(text) and not _is_possessive(token_set):
        return _decide(
            QueryCategory.GENERAL_KNOWLEDGE,
            "how-to question about the world — answer with a stated assumption",
            0.9,
        )

    # ── 7d. A bare question about one named thing ────────────────────────────
    # "Tell me about TRACE." Nothing in the sentence says TRACE is the user's
    # project — but the user has projects, and looking first costs one cheap
    # lookup while not looking costs the answer entirely. If it turns out to be
    # NASA, retrieval returns nothing and the assistant answers from its own
    # knowledge: the answer-first policy behaving exactly as intended.
    #
    # Ahead of the profile check because "My_Agent" tokenises to "my" + "agent",
    # which reads as a possessive and would otherwise land in PROFILE_GENERAL
    # with no subject to retrieve by. The single-token requirement is what keeps
    # "tell me about my projects" out of this branch.
    named = _named_entity_request(query)
    if named:
        return _decide(
            QueryCategory.PROFILE_PROJECTS,
            "names a specific thing that may be the user's own — retrieve "
            "before answering from general knowledge",
            0.7,
            legacy=profile_intent.PROFILE_PROJECTS,
            subject=named,
        )

    # ── 8. Questions about the user ──────────────────────────────────────────
    legacy = profile_intent.classify(query, has_context=has_context)
    if legacy is not None:
        category = _PROFILE_CATEGORY_MAP.get(legacy, QueryCategory.PROFILE_GENERAL)

        # "What's in my resume?" with no section subject is a document question.
        if category is QueryCategory.PROFILE_GENERAL and _mentions_resume(token_set):
            category = QueryCategory.DOCUMENT_RESUME

        subject = None
        if category is QueryCategory.CONVERSATION_FOLLOWUP:
            subject = resolve_followup_subject(query, history)

        return _decide(
            category,
            f"personal-information query ({legacy})",
            0.9,
            legacy=legacy,
            subject=subject,
        )

    # ── 9. Actions on the world ──────────────────────────────────────────────
    if token_set & _ACTION_VERBS:
        if _is_underspecified_action(tokens, token_set):
            return _decide(
                QueryCategory.AMBIGUOUS_ACTION,
                "action request missing a parameter no store holds",
                0.85,
            )
        return _decide(QueryCategory.ACTION_REQUEST, "actionable instruction", 0.8)

    # ── 10. Filler ───────────────────────────────────────────────────────────
    if _SMALL_TALK_RE.match(text) and len(tokens) <= 8:
        return _decide(QueryCategory.SMALL_TALK, "conversational filler", 0.8)

    # ── 11. A referential question with no subject of its own ────────────────
    if has_context and token_set & _FOLLOWUP_MARKERS:
        return _decide(
            QueryCategory.CONVERSATION_FOLLOWUP,
            "referential question inheriting its subject",
            0.75,
            legacy=profile_intent.CONVERSATION_FOLLOWUP,
            subject=resolve_followup_subject(query, history),
        )

    # ── 12. Everything else is a question about the world ────────────────────
    if _GENERAL_KNOWLEDGE_RE.search(text):
        return _decide(
            QueryCategory.GENERAL_KNOWLEDGE,
            "request for an explanation — answer with a reasonable assumption",
            0.85,
        )

    return _decide(
        QueryCategory.GENERAL_KNOWLEDGE,
        "no personal, temporal or actionable signal",
        0.5,
        deterministic=False,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_memory_write(text: str) -> bool:
    """Whether the sentence instructs the assistant to store something."""
    if not any(marker in text for marker in _MEMORY_WRITE_MARKERS):
        return False
    # "do you remember ...", "can you remember ...", "what do you remember" are
    # questions about the store, not instructions to write to it.
    if re.search(r"\b(do|did|can|could|would|will)\s+you\s+(still\s+)?remember\b", text):
        return False
    if re.match(r"^\s*(what|which|who|when|where|why|how)\b", text):
        return False
    return True


def _is_temporal(token_set: Set[str]) -> bool:
    """
    Whether this asks for the present date or time.

    A temporal *subject* is mandatory. "current" alone is not enough, and this
    is not a detail: "what is my current CPI" carries a now-marker and is a
    question about a stored number.
    """
    if not token_set & _TEMPORAL_SUBJECTS:
        return False
    if token_set & _SCHEDULE_SUBJECTS:
        return False
    # A personal subject wins: "how much time have I spent on my projects" is
    # about the projects.
    for _, subjects in profile_intent._SUBJECTS:
        if token_set & subjects:
            return False
    return bool(token_set & _NOW_MARKERS) or bool(
        token_set & {"what", "whats", "which", "tell", "give"}
    )


_POSSESSIVES: Set[str] = {"my", "mine", "our", "ours", "myself", "ourselves"}


def _is_possessive(token_set: Set[str]) -> bool:
    """
    Whether the query claims ownership of its subject.

    The distinction between "how do I build an agent" and "how do I improve my
    CGPA": both have a first-person subject, only the second is about something
    the user owns. A bare "I" is not enough — it is the grammatical subject of
    almost every how-to question anyone asks.
    """
    return bool(token_set & _POSSESSIVES)


def _mentions_resume(token_set: Set[str]) -> bool:
    return bool(token_set & {"resume", "cv", "curriculum"})


# "Tell me about X", "what is X", "describe X" — an informational request whose
# whole subject is one named thing. Matched against a light normalisation that
# keeps underscores and hyphens, so "My_Agent" stays a single token.
_NAMED_REQUEST_RE = re.compile(
    r"^(tell\s+me\s+about|what\s+(is|was|are)|describe|explain|show\s+me)\s+"
    r"(the\s+)?[a-z0-9_\-]+$",
    re.IGNORECASE,
)

# Stricter than `_ENTITY_RE`: a compound name, or an acronym of four or more
# letters. Three-letter acronyms are overwhelmingly generic technical terms
# (RAG, API, LLM, CPU) and looking each one up in a résumé would turn every
# vocabulary question into a wasted round trip. Follow-up resolution keeps the
# looser rule because prior turns already constrain what "it" can mean.
_NAME_LIKE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+|[A-Z]{4,})\b"
)


def _named_entity_request(query: str) -> Optional[str]:
    """
    The entity in a bare informational request, if there is exactly one.

    Requires the entity to be the *entire* subject: "tell me about TRACE" yes,
    "tell me about the history of TRACE compilers in the 1980s" no. A sentence
    with more in it than a name is a general question that happens to mention
    one, and looking it up in a résumé would be a wasted round trip.
    """
    light = re.sub(r"\s+", " ", re.sub(r"[^\w\s_-]", " ", (query or "").lower())).strip()
    if not _NAMED_REQUEST_RE.match(light):
        return None
    for candidate in _NAME_LIKE_RE.findall(query or ""):
        if candidate.upper() not in _STOP_ENTITIES:
            return candidate
    return None


def _is_underspecified_action(tokens: List[str], token_set: Set[str]) -> bool:
    """
    Whether an action is missing something no store can supply.

    Two shapes: a dangling pronoun ("send this to him") and a bare verb phrase
    with no object at all ("schedule a meeting"). Both are genuine clarification
    cases — unlike every question about the user, where asking is a failure to
    look rather than a missing parameter.
    """
    if token_set & _DANGLING_REFERENTS:
        return True
    # No concrete object: short imperative with no proper noun, address, or
    # date-like token to act on.
    content = [t for t in tokens if t not in _ACTION_VERBS and len(t) > 2]
    if len(content) <= 3 and not any(
        char.isdigit() for token in tokens for char in token
    ):
        return True
    return False


# Entity-shaped tokens in prior turns: acronyms and CamelCase/underscore names
# such as TRACE or My_Agent, which is what follow-ups actually refer back to.
_ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+|[A-Z]{3,})\b")

_STOP_ENTITIES = frozenset({
    "USER", "AGENT", "ASSISTANT", "CGPA", "CPI", "GPA", "AND", "THE", "FOR",
    "YOU", "NOT", "BUT", "API", "PDF", "OKAY", "YES", "NO",
})


def resolve_followup_subject(
    query: str, history: Optional[Sequence[Dict[str, Any]]] = None
) -> Optional[str]:
    """
    The entity a follow-up is referring back to.

    "Tell me more about that" carries no subject; the subject was established a
    message earlier and is almost always a named thing — a project, a company, a
    course. The most recent such name wins, searched newest-first across both
    speakers, because the referent of "that" is whatever was most recently under
    discussion regardless of who said it.

    Returns None rather than guessing when history holds no named entity; the
    agent then falls back to broad retrieval, which is a worse answer but never
    a wrong one.
    """
    if not history:
        return None

    # A name in the query itself takes precedence over anything historical:
    # "tell me more about TRACE" is not actually referential.
    direct = _ENTITY_RE.findall(query or "")
    for candidate in direct:
        if candidate.upper() not in _STOP_ENTITIES:
            return candidate

    for turn in reversed(list(history)[-8:]):
        content = str((turn or {}).get("content") or "")
        for candidate in _ENTITY_RE.findall(content):
            if candidate.upper() not in _STOP_ENTITIES:
                return candidate
    return None


def agent_for(decision: QueryDecision, planner_choice: Optional[str]) -> str:
    """
    Which specialist should run.

    The planner's choice is honoured except where a category owns its agent
    outright: the clock is not a profile question, and a timetable question is
    not one either. Personal questions are pulled to `profile` whatever the
    planner concluded, because the planner scores intent without seeing the
    store it would otherwise ask the user to substitute for.
    """
    owned = _CATEGORY_AGENT.get(decision.category)
    if owned:
        return owned

    choice = planner_choice if planner_choice in ("job", "email", "academic", "profile") else "profile"

    if decision.is_personal or decision.category in (
        QueryCategory.EXPLICIT_MEMORY_WRITE,
        QueryCategory.IDENTITY_UPDATE,
    ):
        # Academic keeps anything it was legitimately chosen for; everything
        # else about the user belongs to profile.
        return choice if choice == "academic" else "profile"

    return choice
