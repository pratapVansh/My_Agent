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
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

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
#
# `professor`/`instructor`/`faculty`/`teacher` are included because they name a
# field the timetable actually stores (`Timetable.instructor`) — "who is my
# professor for X" is answerable only from the same row a "when is X" question
# reads, and routing it anywhere else invites a model to guess a name instead
# of reading the one that was transcribed from the document.
_SCHEDULE_SUBJECTS: Set[str] = {
    "class", "classes", "lecture", "lectures", "lab", "labs", "exam", "exams",
    "timetable", "schedule", "period", "periods", "test", "tests", "quiz",
    "attendance", "assignment", "assignments", "deadline", "deadlines",
    "professor", "professors", "instructor", "instructors", "faculty",
    "teacher", "teachers",
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

# Asking where the *previous answer* came from. These are questions about the
# assistant's own sourcing, not new questions about the user, and answering them
# by retrieving again reports where the answer *would* come from now — a
# different claim, and a wrong one whenever routing has changed since.
#
# Deliberately narrow and clause-level: "how did you know" is provenance, while
# "how do I know if I qualify" is an ordinary question that happens to share a
# verb.
_PROVENANCE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\bhow\s+(did|do|would)\s+you\s+know\b"),
    re.compile(r"\bhow\s+did\s+you\s+(find|figure)\s+(that|it|this)?\s*out\b"),
    re.compile(r"\bwhere\s+did\s+you\s+(get|find|read|hear|see)\b"),
    re.compile(r"\bwhat\s+(is\s+|was\s+)?your\s+source\b"),
    re.compile(r"\b(which|what)\s+(tool|source|memory|record|agent|file)\s+"
               r"(did|do)\s+you\s+(use|read|check)\b"),
    re.compile(r"\bsays\s+who\b"),
    re.compile(r"\b(what|which)\s+(made|makes)\s+you\s+(say|think)\b"),
    re.compile(r"\bhow\s+are\s+you\s+(so\s+)?(sure|certain)\b"),
    re.compile(r"\bwhere\s+(is|does)\s+that\s+(come\s+from|from)\b"),
)

# Day-scoped availability questions. "What do I have today" names no schedule
# noun at all, so the subject-based rule below never fired and it fell through
# to a generic personal-information route that has no access to the timetable.
# The schedule nouns catch "what classes do I have"; these catch the far more
# common phrasing that omits the noun entirely.
#
# Matched against the normalised text, where `_normalise` has already dropped
# the possessive: "what's on" arrives as "what on". Hence the optional-copula
# shape rather than a bare "what is".
_AVAILABILITY_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\bwhat\s+(do|have)\s+i\s+(have|got)\b"),
    re.compile(r"\bdo\s+i\s+have\s+(anything|something|much|any)\b"),
    re.compile(r"\bam\s+i\s+(free|busy|booked|occupied)\b"),
    re.compile(r"\bwhat(s)?(\s+(is|was))?\s+on\b"),
    re.compile(r"\bwhat(s)?(\s+(is|was))?\s+my\s+(routine|plan|day|agenda)\b"),
    re.compile(r"\bhow(s)?(\s+(is|does))?\s+my\s+day\s+look\b"),
    re.compile(r"\banything\s+(scheduled|planned|on)\b"),
)

# Named days. A day reference is what makes an availability question a
# *timetable* question rather than an open-ended one about the user's life.
_DAY_TOKENS: Set[str] = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "weekend", "weekday",
}

# Words that ask for the value as it stands *now*, used only for the academic
# split. Broader than `_NOW_MARKERS` — "my latest CPI" is as much a question
# about current standing as "my current CPI" — and kept separate precisely so
# that widening it cannot leak into the clock check, where "latest" would be a
# very different and much more damaging claim.
_CURRENCY_MARKERS: Set[str] = _NOW_MARKERS | {
    "latest", "newest", "updated", "recent", "standing", "uptodate",
}

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

# "How well do I match this role?" — a question whose words are about the user
# but whose answer is a computation over a posting.
#
# Three conjuncts, all required, because each one alone over-fires. The fit word
# alone catches "does this fit in my schedule"; the job noun alone catches every
# ordinary job search; and without a first-person marker "does this role match
# the description" is a question about two documents rather than about the
# person asking. Together they are narrow enough that the only sentences that
# reach here are the ones `app.matching` was built to answer.
_FIT_WORD_RE = re.compile(
    # `\w*qualif\w*` rather than `qualif\w*`: "underqualified" and
    # "overqualified" have no word boundary before the stem, so the tighter
    # form missed both — and both are asking precisely this question.
    r"\b(match(?:es|ed|ing)?|fit|suit(?:ed|able|ability)?|\w*qualif\w*|eligible"
    r"|stack\s+up|good\s+(?:for|enough)|right\s+for|cut\s+out\s+for"
    r"|chances?|shot\s+at|shortlist\w*|gaps?|missing"
    r"|meet\s+(?:the\s+)?(?:requirements?|criteria|bar)"
    r"|stand\s+a\s+chance|measure\s+up|compare\s+(?:to|against))\b",
    re.IGNORECASE,
)

_JOB_NOUN_RE = re.compile(
    r"\b(job|jobs|role|roles|position|positions|posting|postings|opening"
    r"|openings|vacancy|listing|listings|jd|job\s+description|this\s+one"
    r"|internship|internships|offer|offers|req|requisition)\b",
    re.IGNORECASE,
)

_FIRST_PERSON_RE = re.compile(r"\b(i|me|my|mine|im|ive)\b", re.IGNORECASE)

# "Find me AI engineer jobs" — a live lookup, not a question about the user.
#
# This existed nowhere. Every phrasing of it landed in GENERAL_KNOWLEDGE, whose
# route is whatever the planner said, so the only thing standing between a job
# search and a model inventing postings was an LLM agreeing to route it. The
# audit caught exactly that: planner says "profile", zero tools run, and the
# reply is "here are some jobs you might like" — listings that do not exist.
#
# A search verb plus a job noun, and no fit word: "do I match this job" is a
# JOB_MATCH and is checked first, so the two cannot collide.
_JOB_SEARCH_VERB_RE = re.compile(
    r"\b(find|search|look\s+for|show|list|get|any|browse|explore|hunt"
    r"|recommend|suggest|fetch)\b",
    re.IGNORECASE,
)

# "what jobs are available", "are there any openings" — a search phrased as a
# question about the board rather than as an instruction.
_JOB_AVAILABILITY_RE = re.compile(
    r"\b(?:what|which|any|are\s+there)\b[^.?]{0,40}?"
    r"\b(job|jobs|role|roles|position|positions|opening|openings|vacancy"
    r"|vacancies|listing|listings|internship|internships)\b",
    re.IGNORECASE,
)

# A bare demonstrative standing in for the posting: "am I qualified for this?".
# Only counts when the conversation was already about a job — see
# `_is_job_match`.
_DEMONSTRATIVE_RE = re.compile(
    r"\b(this|that|it|these|those|here|them)\b", re.IGNORECASE
)

# Whether earlier turns established a posting to refer back to.
_JOB_CONTEXT_RE = re.compile(
    r"\b(job|role|position|posting|opening|vacancy|listing|jd|internship"
    r"|hiring|recruiter|applicant|candidate|requirements?|responsibilities)\b",
    re.IGNORECASE,
)

# "Should I apply for this?" is a fit question wearing an action verb. Without
# this it classifies as ACTION_REQUEST and the assistant tries to apply.
_SHOULD_APPLY_RE = re.compile(
    r"\bshould\s+i\s+(?:apply|go\s+for|bother)\b", re.IGNORECASE
)

_JOB_CONTEXT_TURNS = 4


def _recent_job_context(history: Optional[Sequence[Dict[str, Any]]]) -> bool:
    """Whether a posting was under discussion in the last few turns."""
    for turn in list(history or [])[-_JOB_CONTEXT_TURNS:]:
        if not isinstance(turn, dict):
            continue
        if _JOB_CONTEXT_RE.search(str(turn.get("content") or "")):
            return True
    return False


def _is_job_match(
    text: str, history: Optional[Sequence[Dict[str, Any]]] = None
) -> bool:
    """
    Whether this asks how the user measures up against a posting.

    Three conjuncts, all required, because each alone over-fires. The fit word
    alone catches "does this fit in my schedule"; the job noun alone catches
    every ordinary job search; and without a first-person marker "does this role
    match the description" is a question about two documents rather than about
    the person asking.

    The object may be a bare demonstrative — "am I qualified for this?" is how
    people actually speak once a posting is on screen — but only when earlier
    turns established one. Resolving a referent from context is what the
    follow-up machinery already does for other categories; accepting "this" with
    no antecedent would let any sentence containing "fit" become a match query.
    """
    if _SHOULD_APPLY_RE.search(text):
        return True
    if not (_FIRST_PERSON_RE.search(text) and _FIT_WORD_RE.search(text)):
        return False
    if _JOB_NOUN_RE.search(text):
        return True
    return bool(
        _DEMONSTRATIVE_RE.search(text) and _recent_job_context(history)
    )


def _is_job_search(text: str) -> bool:
    """
    Whether this asks the board for postings.

    Requires a job noun *and* either a search verb or an availability phrasing,
    so "my job at Acme" (a fact about the user) and "should I apply" (a fit
    question) both stay out. Callers must run `_is_job_match` first: a fit
    question mentioning "job" is about the user, not about the board.
    """
    if not _JOB_NOUN_RE.search(text):
        return False
    if _JOB_AVAILABILITY_RE.search(text):
        return True
    return bool(_JOB_SEARCH_VERB_RE.search(text))


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
    # Owned outright rather than left to the planner: the words are personal
    # enough that both the planner and `is_personal` route it to profile, and
    # profile has no posting and no matcher. Only the job agent can answer it.
    QueryCategory.JOB_MATCH: "job",
    # Owned for the same reason, one step earlier in the funnel: a search of the
    # board is answerable only by `job_search`, which lives here and nowhere
    # else. Left to the planner it routed to profile roughly whenever the
    # planner felt like it, and profile answered from the model's own idea of
    # what jobs exist.
    QueryCategory.JOB_SEARCH: "job",
    # Answered from what was recorded about the previous turn, by a node that
    # reads that record rather than a model that would reconstruct it.
    QueryCategory.PROVENANCE_QUERY: "provenance",
}


# Categories whose answer is composed deterministically and is complete when the
# owning agent returns. The planner's multi-step plan does not apply to them.
#
# The planner scores intent from the query text and cannot see that routing
# overrode its agent choice, so it keeps proposing a follow-up step — and a
# follow-up step runs a *different* agent, whose envelope replaces the first
# one's. A live run made the cost concrete: a JOB_MATCH turn produced the
# evidence-backed match report, then advanced to a second step where the profile
# agent paraphrased it into free prose. The rendered answer, its source ids and
# its structured verdict were all discarded, which defeats the entire reason
# `app.matching` renders rather than generates.
#
# Deliberately not "every category that owns its agent": SCHEDULE_TEMPORAL owns
# `academic`, and "what class do I have today, then email my professor" is a
# legitimate two-step plan that must keep working.
SINGLE_STEP_CATEGORIES: FrozenSet[QueryCategory] = frozenset({
    QueryCategory.JOB_MATCH,
})


def is_single_step(category: Optional[str]) -> bool:
    """Whether this category's answer is complete without further planning."""
    if not category:
        return False
    try:
        return QueryCategory(category) in SINGLE_STEP_CATEGORIES
    except ValueError:
        return False


# Categories answered by a node that runs no model and reads nothing the
# planner produces. For these the planner call is pure waste: it is issued,
# waited for, billed, and then discarded by `agent_for`, which owns the agent
# for the category outright.
#
# The set is deliberately small. It is not "categories where the planner is
# usually wrong" — it is the two where its output provably cannot be read:
#
# * TEMPORAL_CURRENT is answered from the system clock by `temporal_node`.
# * PROVENANCE_QUERY is answered from the recorded provenance of the previous
#   turn by `provenance_node`.
#
# Both terminate at END without a specialist, so `selected_agent`,
# `execution_plan` and `detected_intent` are never consulted; and neither can
# reach the clarification branch, because a clock question and a "how did you
# know that" question have no missing parameter to ask about.
#
# Everything else keeps the planner, including the categories that own their
# agent (JOB_SEARCH, SCHEDULE_TEMPORAL): those still run a specialist, and its
# `execution_plan` can still carry a genuine second step.
_PLANNER_FREE_CATEGORIES: FrozenSet[QueryCategory] = frozenset({
    QueryCategory.TEMPORAL_CURRENT,
    QueryCategory.PROVENANCE_QUERY,
})


def planner_is_load_bearing(category: Optional[str]) -> bool:
    """
    Whether the planner's output can affect this turn at all.

    False means the turn is already fully determined by the deterministic
    classifier and the planner call can be skipped. Unknown or missing
    categories return True — the safe answer, since an unrecognised category
    falls through to planner-driven routing.
    """
    if not category:
        return True
    try:
        return QueryCategory(category) not in _PLANNER_FREE_CATEGORIES
    except ValueError:
        return True


# ── Where an answer is allowed to come from ──────────────────────────────────
#
# The streaming workflow has no tools. For most questions that is a latency
# trade: the reply is composed from the category-scoped memory prompt and is
# honest about what it does not know. For some categories it is not a trade at
# all, and the two sets below draw the line between three different situations.
#
# The test applied to every category was the same, and it is a question about
# *reachability*, not about how personal the words are:
#
#     Can this question be answered truthfully from the memory prompt that
#     `MemoryManager.format_context_for_prompt` renders for its category?
#
# That prompt is much narrower than the tool surface, and the gap is where
# fabrication lives. It carries at most 300 characters of résumé, the top five
# skills and the top three project snippets. It carries no typed `education`,
# `experience` or `achievements` section at all — those exist in the store and
# are reachable *only* through `get_education` / `get_experience` /
# `get_achievements`. So "what is my CGPA", spoken, was a model looking at a
# résumé header and a project list and being asked for a number that was not in
# front of it. That is not honest degradation; it is an invitation to guess.

# Categories that cannot be answered correctly without running a tool.
#
# The distinction this encodes is not "a tool would help" — it is "answering
# this from a prompt produces a false statement". A JOB_MATCH turn answered
# without `match_job` is a language model reading a résumé blob and deciding
# for itself how well the user matches a posting, which is exactly the
# unevidenced claim `app.matching` was built to make unrepresentable.
#
# It exists because the voice path has *two* routers and only one of them is
# deterministic. `hybrid_router` decides tool-vs-streaming from keywords and an
# LLM fallback, and `/agents/stream` skips it entirely; both funnel into
# `run_streaming_workflow`, which runs without tools. Naming the categories here
# lets that workflow re-derive the answer for itself and escalate, so the
# guarantee stops depending on which router ran or which words were spoken.
#
# Membership is justified one category at a time, and the justification is
# always a *specific* source the streaming prompt cannot reach:
#
#   JOB_MATCH            the verdict is a computation over a posting, rendered
#                        by `app.matching` from an evidence table. No prompt
#                        can produce it.
#   PROFILE_IDENTITY     canonical name has a strict precedence — canonical
#                        fact, then résumé header, never a remembered name —
#                        and `get_identity` is the only thing that enforces it.
#                        Getting this wrong is the single most damaging error
#                        this system can make.
#   PROFILE_EDUCATION    CGPA/college/branch live in the typed `education`
#   ACADEMIC_CURRENT     section. Neither is rendered into any prompt.
#   PROFILE_EXPERIENCE   internships and roles live in the typed `experience`
#                        section. Not rendered into any prompt.
#   PROFILE_ACHIEVEMENTS awards live in the typed `achievements` section, which
#                        has no prompt section whatsoever.
#   DOCUMENT_RESUME      the question names the document as the authority; the
#                        prompt holds 300 characters of it, `get_resume` 1500.
#   EXPLICIT_MEMORY      the remembered-value keys are deliberately *withheld*
#                        from prompts (see `_fact_is_visible`), so the only way
#                        to read them back is `recall_explicit_memory`.
#   EXPLICIT_MEMORY_WRITE a streamed reply says "I'll remember that" and writes
#   IDENTITY_UPDATE      nothing. The claim that the store changed is as false
#                        as any invented fact, and worse: it is acted on later.
#   SCHEDULE_TEMPORAL    the timetable and attendance live in Postgres behind
#                        the academic agent's tools. Nothing about them reaches
#                        a prompt, so a streamed answer is an invented day.
#
# Deliberately *not* here: PROFILE_SKILLS, PROFILE_PROJECTS, PROFILE_GENERAL,
# CONVERSATION_* and EPISODIC_MEMORY. Their sources genuinely are rendered into
# the prompt, so a streamed answer quotes the store rather than inventing —
# provided retrieval actually ran. That proviso is the second set.
TOOL_REQUIRED_CATEGORIES: FrozenSet[QueryCategory] = frozenset({
    QueryCategory.JOB_MATCH,
    #   JOB_SEARCH           the postings are on a board reached by `job_search`.
    #                        No prompt holds them, so a streamed answer is a list
    #                        of jobs that do not exist — and the user applies to
    #                        them.
    QueryCategory.JOB_SEARCH,
    QueryCategory.PROFILE_IDENTITY,
    QueryCategory.PROFILE_EDUCATION,
    QueryCategory.ACADEMIC_CURRENT,
    QueryCategory.PROFILE_EXPERIENCE,
    QueryCategory.PROFILE_ACHIEVEMENTS,
    QueryCategory.DOCUMENT_RESUME,
    QueryCategory.EXPLICIT_MEMORY,
    QueryCategory.EXPLICIT_MEMORY_WRITE,
    QueryCategory.IDENTITY_UPDATE,
    QueryCategory.SCHEDULE_TEMPORAL,
})


# Categories that may be streamed, but only once retrieval has actually
# delivered something to stream *from*.
#
# These are the user-specific questions whose sources the memory prompt does
# render: skills, project snippets, profile facts, the thread, episode
# summaries. When retrieval succeeds the streamed answer quotes the store, and
# when it comes back empty the prompt says so in as many words ("Retrieval
# Status: No skills data found", "treat skills as unknown, not absent"), so the
# reply degrades honestly and keeps its sub-second first token.
#
# The hole is what happens when retrieval does not come back at all. The
# streaming workflow caps memory + planning at a few seconds and, on timeout or
# exception, proceeds with `memory_prompt = ""`. There is then no data, no
# status hint and no policy line — just a personal question and a model. Every
# safeguard in the retrieval layer is expressed *inside* the prompt, so an
# absent prompt disarms all of them at once.
#
# So the guarantee is conditional and structural: stream when grounded,
# escalate to the tools when not. The tool path fails visibly where this one
# fails silently — `_section` reports "could not be read right now" and
# `answerability` records TOOL_ERROR, which is the difference between "I don't
# have that" and "I couldn't look that up".
#
# "Grounded" is not one condition, because these categories do not read one
# source. Splitting them by what `SOURCE_PRECEDENCE` says they actually consult
# is what keeps the rule from firing on turns it has no business touching.
MEMORY_GROUNDED_CATEGORIES: FrozenSet[QueryCategory] = frozenset({
    QueryCategory.PROFILE_SKILLS,
    QueryCategory.PROFILE_PROJECTS,
    QueryCategory.PROFILE_GENERAL,
    QueryCategory.EPISODIC_MEMORY,
})
"""Answered from the retrieved memory prompt: skills, project snippets,
profile facts, episode summaries. Ungrounded when that prompt is empty."""

CONVERSATION_GROUNDED_CATEGORIES: FrozenSet[QueryCategory] = frozenset({
    QueryCategory.CONVERSATION_CURRENT,
    QueryCategory.CONVERSATION_FOLLOWUP,
})
"""Answered from the transcript, which the streaming path passes inline and
which therefore survives a memory outage intact. Requiring a memory prompt for
these would escalate every "go ahead" and "tell me more" spoken while Qdrant is
slow — including, fatally, the ones that are replies to a pending action.

What they cannot survive is having no transcript: "what did I just tell you"
with nothing behind it is a question whose answer a model would have to
invent."""

GROUNDING_REQUIRED_CATEGORIES: FrozenSet[QueryCategory] = (
    MEMORY_GROUNDED_CATEGORIES | CONVERSATION_GROUNDED_CATEGORIES
)
"""Every category that may stream only once its own sources have arrived."""


# Categories that ask for something to happen in the world.
#
# The third rule, and the one with the sharpest edge. The two sets above are
# about *knowledge* — an answer that would be invented rather than retrieved.
# This one is about *effect*, and the failure it prevents is worse than a wrong
# fact: the tool-free path has no tools at all, so a model asked to send an
# email there cannot send one and will say it did. The user then believes a
# message went out. Nothing in the system disagrees with them, because nothing
# in the system knows.
#
# `AMBIGUOUS_ACTION` is deliberately *not* here, and the reason is that it is a
# different question rather than a milder version of this one. It means the
# request is missing a parameter no store holds — "send this to him" — and the
# only honest response is to ask which. Escalating it would replace a clarifying
# question with a graph invocation that has nothing more to go on, and it would
# also sweep up "send it", which is how a user approves an action that is
# already pending.
#
# The categories alone were never going to be sufficient, which is why they are
# not the whole rule: "forget my preference for tea" classifies as
# PROFILE_GENERAL because of the word "preference", while being a request to
# destroy a stored fact. `escalation_reason` therefore asks the capability
# vocabulary as well, and the two rules cover each other's misses.
#
# What escalation buys is not a confirmation — it is a real tool loop. Whether
# anything is then gated is decided per tool by its declared effect, in
# `ActionGateway`, exactly as before: `email_draft` still drafts freely, and only
# EXTERNAL_WRITE / DESTRUCTIVE reach the gate.
ACTION_CATEGORIES: FrozenSet[QueryCategory] = frozenset({
    QueryCategory.ACTION_REQUEST,
})


# Why a turn was taken off the streaming path. Carried into the emitted
# metadata so a trace says which of the three rules fired.
ESCALATION_TOOL_REQUIRED = "tool_required"
ESCALATION_UNGROUNDED = "ungrounded_memory"
ESCALATION_ACTION_REQUEST = "action_request"


# Which specialist owns an escalated turn. Only used to label the escalation
# before the graph has run; the agent that *actually* ran is read back off the
# returned envelope, never guessed from here.
_ESCALATION_AGENT: Dict[QueryCategory, str] = {
    QueryCategory.JOB_MATCH: "job",
    QueryCategory.SCHEDULE_TEMPORAL: "academic",
}

# Which specialist owns each consequential capability. Used only to label an
# escalated action turn, whose category (ACTION_REQUEST, AMBIGUOUS_ACTION, or a
# category the classifier got wrong) names no owner at all.
_CAPABILITY_AGENT: Dict[str, str] = {
    "send_email": "email",
    "forget_preference": "profile",
}


def _as_category(category: Optional[Any]) -> Optional[QueryCategory]:
    """A `QueryCategory` from either a category or its string value."""
    if not category:
        return None
    if isinstance(category, QueryCategory):
        return category
    try:
        return QueryCategory(str(category))
    except ValueError:
        return None


def requires_tools(category: Optional[Any]) -> bool:
    """
    Whether answering this category without tools would produce a false claim.

    Accepts a `QueryCategory` or its string value, so the callers — one holding
    a decision, one holding `state["query_category"]`, one holding a raw
    transcript's classification — can ask the same question without any of them
    re-deriving the category.
    """
    return _as_category(category) in TOOL_REQUIRED_CATEGORIES


def requires_grounded_memory(category: Optional[Any]) -> bool:
    """
    Whether this category may only be streamed when its sources arrived.

    True for the user-specific categories the memory prompt or the transcript
    does carry. `requires_tools` is the unconditional rule; this is the
    conditional one, and no category is in both.
    """
    return _as_category(category) in GROUNDING_REQUIRED_CATEGORIES


# ── Which academic question is this? ─────────────────────────────────────────
#
# `SCHEDULE_TEMPORAL` is one category over four tables: the timetable, the
# attendance log, the exam list and personal plans. Routing them all to the
# academic agent is right; requiring the same tools of all of them is not — the
# union let "what classes do I have tomorrow" be satisfied by a call to
# `get_plans`, which reads something else entirely.
#
# Deterministic and vocabulary-based, like every other classifier here: this
# runs on every spoken turn and must cost microseconds. Order matters — a
# sentence naming both an exam and a day is about the exam.
_EXAM_RE = re.compile(r"\b(exam|exams|test|tests|quiz|quizzes|midterm|final|finals)\b")
_ATTENDANCE_RE = re.compile(r"\b(attendance|attended|absent|present|bunk|bunked|missed)\b")
_PLAN_RE = re.compile(r"\b(plan|plans|task|tasks|todo|to\s?do|reminder|reminders)\b")
_TIMETABLE_RE = re.compile(
    r"\b(class|classes|lecture|lectures|lab|labs|timetable|time\s?table"
    r"|schedule|period|periods|subject|subjects|course|courses"
    r"|professor|professors|instructor|instructors|faculty|teacher|teachers)\b"
)

# "When is Generative AI?" — an opener plus a name, and nothing else. Anchored
# to the start of the utterance so "I wonder when is a good time" cannot reach
# it, and requiring at least one word after the opener so a bare "when is?" is
# left alone.
_WHEN_IS_SUBJECT_RE = re.compile(
    r"^(?:when(?:s| is| are)|what time (?:is|are))\s+(?!the\b|it\b|that\b|this\b)\w+"
)

SCHEDULE_TIMETABLE = "timetable"
SCHEDULE_ATTENDANCE = "attendance"
SCHEDULE_EXAMS = "exams"
SCHEDULE_PLANS = "plans"


def schedule_intent(query: str) -> Optional[str]:
    """
    Which academic capability a SCHEDULE_TEMPORAL turn is actually asking for.

    Returns None for anything unrecognised, and the caller then keeps the
    category's broader requirement — an unclassifiable academic question should
    be answered from *some* academic tool rather than from none.

    The default when a day is named but no noun is: the timetable. "What do I
    have on Monday" and "what's on tomorrow" carry no schedule noun at all, and
    they are the most common way this question is asked out loud.
    """
    text = _normalise(query)
    if not text:
        return None

    if _EXAM_RE.search(text):
        return SCHEDULE_EXAMS
    if _ATTENDANCE_RE.search(text):
        return SCHEDULE_ATTENDANCE
    if _PLAN_RE.search(text):
        return SCHEDULE_PLANS
    if _TIMETABLE_RE.search(text):
        return SCHEDULE_TIMETABLE

    # No noun, but a day reference and a first-person "have" — "what do I have
    # on Monday", "anything tomorrow".
    if _AVAILABILITY_PATTERNS and any(p.search(text) for p in _AVAILABILITY_PATTERNS):
        return SCHEDULE_TIMETABLE

    # "When is <subject>?" — see branch 4c in `classify`. The timetable is what
    # the assistant can answer this from.
    if _WHEN_IS_SUBJECT_RE.match(text):
        return SCHEDULE_TIMETABLE
    return None


def requests_action(category: Optional[Any]) -> bool:
    """
    Whether this category asks for an effect in the world.

    Kept separate from `requires_tools` rather than folded into it, because the
    two say different things and the difference is what a trace needs. A
    tool-required category would be *answered* wrongly without tools; an action
    category would be *performed* — or rather, claimed and not performed.
    """
    return _as_category(category) in ACTION_CATEGORIES


def names_consequential_capability(text: Optional[str]) -> Optional[str]:
    """
    The confirmable tool this utterance is reaching for, or None.

    The backstop for a classifier miss, and it exists because there is one:
    "forget my preference for tea" classifies as PROFILE_GENERAL — the word
    "preference" makes it look like a question about the user — while being a
    request to destroy a stored fact. Category alone would have streamed it.

    Asks the tool registry rather than holding a list of its own, so the
    vocabulary and the capability are declared together and cannot drift apart.
    Imported lazily: `confirmable_tools` reaches the email sender and the memory
    manager, and this module is imported by every router.
    """
    if not text:
        return None
    from app.agents import confirmable_tools

    return confirmable_tools.names_consequential_capability(text)


def escalation_reason(
    category: Optional[Any],
    *,
    text: Optional[str] = None,
    memory_grounded: bool = True,
    history_grounded: bool = True,
) -> Optional[str]:
    """
    Why this turn cannot be answered on the tool-free path, or None.

    The single question both routers and the streaming workflow ask. Callers
    that cannot know whether retrieval succeeded — `hybrid_router` runs before
    any of it — leave both flags at their defaults and get the unconditional
    verdict; `run_streaming_workflow` asks again once it knows.

    `text` is optional and additive. Given it, the utterance is also checked
    against the vocabulary of the consequential capabilities that actually
    exist, so a turn whose *category* was missed still escalates when it names
    one of them. Callers that hold only a category keep the previous behaviour
    exactly.

    The action rule is tested first because it is the one whose failure has an
    effect rather than only an answer, and because the reason reported should
    name the most specific thing known about the turn.

    The two grounding flags are separate because the categories they gate read
    different things. A memory outage does not take the transcript away, and a
    first turn with no transcript does not take the user's résumé away;
    collapsing them into one signal would escalate turns whose sources were
    sitting right there.

    One thing this deliberately does *not* do is exclude confirmation replies.
    "send it" is both an affirmation and an action-shaped sentence, and this
    function cannot tell which without knowing whether an action is pending —
    which it has no way to ask. So the ordering is the caller's job, and both
    callers do it: `run_streaming_workflow` resolves a pending action before it
    asks this at all, and `hybrid_router` sending a "yes" down the tool path is
    harmless because `decide_route` intercepts it for the gateway on arrival.
    """
    if requests_action(category) or names_consequential_capability(text):
        return ESCALATION_ACTION_REQUEST
    if requires_tools(category):
        return ESCALATION_TOOL_REQUIRED
    resolved = _as_category(category)
    if not memory_grounded and resolved in MEMORY_GROUNDED_CATEGORIES:
        return ESCALATION_UNGROUNDED
    if not history_grounded and resolved in CONVERSATION_GROUNDED_CATEGORIES:
        return ESCALATION_UNGROUNDED
    return None


def escalation_agent_for(
    category: Optional[Any], *, text: Optional[str] = None
) -> str:
    """
    The specialist an escalated turn is expected to reach.

    A label only — the agent that *actually* ran is read back off the returned
    envelope, never guessed from here. `text` refines the guess for action
    turns, whose category names no owner: a send belongs to email and a deletion
    to profile, and reporting "profile" for both made an escalation trace
    useless for the one case where it matters most.
    """
    capability = names_consequential_capability(text)
    if capability is not None:
        return _CAPABILITY_AGENT.get(capability, "profile")
    return _ESCALATION_AGENT.get(_as_category(category), "profile")


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
            QueryCategory.ACADEMIC_CURRENT,
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

    # ── 1b. A question about where the last answer came from ─────────────────
    # Ahead of everything except an identity change: these shapes are
    # unambiguous, and every one of them was previously scattered across
    # GENERAL_KNOWLEDGE and CONVERSATION_FOLLOWUP, neither of which can say
    # which source produced the answer.
    for pattern in _PROVENANCE_PATTERNS:
        if pattern.search(text):
            return _decide(
                QueryCategory.PROVENANCE_QUERY,
                "asks which source or tool produced the previous answer",
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

    # ── 4b. The same question with the schedule noun left out ────────────────
    # "What do I have today", "am I free tomorrow". A day reference is required:
    # without one these are open-ended questions about the user's life, and
    # sending those to the timetable would be as wrong as the generic routing
    # that sent the day-scoped ones to a store with no schedule in it.
    if (token_set & _NOW_MARKERS or token_set & _DAY_TOKENS) and any(
        pattern.search(text) for pattern in _AVAILABILITY_PATTERNS
    ):
        return _decide(
            QueryCategory.SCHEDULE_TEMPORAL,
            "day-scoped availability question — the timetable answers it",
            0.85,
        )

    # ── 4c. "When is <subject>?" ─────────────────────────────────────────────
    # A subject named without any day, schedule noun or possessive: "when is
    # Generative AI", "what time is Digital Image Processing". Nothing in the
    # sentence marks it as a class, and nothing can — distinguishing a course
    # title from any other proper noun would need the user's own subject list,
    # which this module deliberately cannot reach.
    #
    # So it is routed on what the assistant can actually answer "when is X" from:
    # the timetable. If X is not on it, `get_schedule` returns nothing and the
    # reply says the subject is not on the timetable, which is both true and
    # useful. That is a better failure than the previous one, where the question
    # fell through to GENERAL_KNOWLEDGE and a model answered from its own idea of
    # when a course meets.
    #
    # Kept after the clock-adjacent branches above so "when is the date" and
    # "when is my exam" are already claimed, and narrow enough to need an
    # explicit "when is"/"what time is" opener.
    if _WHEN_IS_SUBJECT_RE.match(text) and not (token_set & _TEMPORAL_SUBJECTS):
        return _decide(
            QueryCategory.SCHEDULE_TEMPORAL,
            "asks when a named subject meets — answerable only from the timetable",
            0.7,
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

    # ── 7b-bis. Measuring the user against a posting ──────────────────────────
    # Ahead of both the how-to check and every profile check, because both claim
    # it and neither can answer it.
    #
    # Against general knowledge: "how do I stack up against this job" opens with
    # "how do I", so `_GENERAL_KNOWLEDGE_RE` took it and the turn was answered
    # as an essay about job hunting. Against the profile checks: the sentence is
    # grammatically about the user's experience, so `profile_intent` labels it
    # PROFILE_EXPERIENCE and routing sends it to an agent holding résumé tools
    # and no posting — leaving the model to supply the comparison itself.
    #
    # The job agent owns this category outright (`_CATEGORY_AGENT`), and
    # `TOOL_REQUIRED_CATEGORIES` makes it unanswerable without `match_job` on
    # every path including the spoken one.
    if _is_job_match(text, history):
        return _decide(
            QueryCategory.JOB_MATCH,
            "asks how the user measures up against a specific role",
            0.9,
        )

    # ── 7b-ii. A search of the board ─────────────────────────────────────────
    # Immediately after the fit check, because the two share every noun and only
    # the fit check can tell them apart: "do I match this job" is about the
    # user, "find me AI engineer jobs" is about the board.
    #
    # Before the profile check for the same reason 7b is: "what jobs are
    # available for me" carries a possessive and would otherwise be read as a
    # question about the user's own employment.
    if _is_job_search(text):
        return _decide(
            QueryCategory.JOB_SEARCH,
            "asks the job board for postings",
            0.85,
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

    # ── 7e. "Who am I?" ──────────────────────────────────────────────────────
    # An identity question that contains neither "name" nor "called", so the
    # subject-token classifier never claimed it and it fell through to general
    # knowledge — where the answer is a stored fact the model cannot see.
    if _IDENTITY_QUESTION_RE.match(text):
        return _decide(
            QueryCategory.PROFILE_IDENTITY,
            "asks who the user is",
            0.9,
            legacy=profile_intent.PROFILE_NAME,
        )

    # ── 8. Questions about the user ──────────────────────────────────────────
    legacy = profile_intent.classify(query, has_context=has_context)
    if legacy is not None:
        category = _PROFILE_CATEGORY_MAP.get(legacy, QueryCategory.PROFILE_GENERAL)

        # Two questions that share every word except the one that decides where
        # to look. "The CPI on my résumé" is a question about a document and is
        # answered from it; "my current CPI" is a question about the present and
        # is answered from the most recently recorded value, with the document
        # as fallback. Collapsing them was harmless only while the two agreed.
        #
        # Résumé scope is checked first: "the current CPI on my résumé" names a
        # document, and the document wins.
        #
        # CONVERSATION_FOLLOWUP is in this set because of a bug the audit found
        # and measured: once *any* conversation history exists, `profile_intent`
        # returns CONVERSATION_FOLLOWUP for "what is on my resume", the override
        # did not apply, and the category fell out of TOOL_REQUIRED_CATEGORIES.
        # The same question was therefore tool-required on turn 1 and streamable
        # from 300 characters of résumé on turn 5. A sentence that names the
        # document is a question about the document whatever else is in the
        # thread — a follow-up marker does not make "summarise my resume" stop
        # being about the résumé.
        if _mentions_resume(token_set) and category in (
            QueryCategory.PROFILE_GENERAL,
            QueryCategory.PROFILE_EDUCATION,
            QueryCategory.CONVERSATION_FOLLOWUP,
        ):
            category = QueryCategory.DOCUMENT_RESUME
        elif category is QueryCategory.PROFILE_EDUCATION and token_set & _CURRENCY_MARKERS:
            category = QueryCategory.ACADEMIC_CURRENT

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


# "Who am I?" — an identity question carrying none of the identity vocabulary.
# Anchored to the start so "who am I to judge" and "remind me who I am talking
# to" do not claim it.
_IDENTITY_QUESTION_RE = re.compile(
    r"^(who\s+am\s+i|who\s+i\s+am|remind\s+me\s+who\s+i\s+am)\s*$"
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


def agent_for(
    decision: QueryDecision,
    planner_choice: Optional[str],
    *,
    plan_steps: int = 1,
) -> str:
    """
    Which specialist should run.

    Three rules, in this order, and the order is the whole design.

    **A category that owns its agent always wins.** The clock is not a profile
    question, a timetable question is not one either, and a job search is
    answerable only by the agent holding `job_search`. No planner opinion and
    no plan shape overrides these.

    **A genuine multi-step plan owns its own first step.** `plan_steps` is how
    many steps the planner produced, and when it is greater than one the
    planner has decomposed a compound request — "find a job, compare it with my
    profile, then draft an email". Pulling such a turn to `profile` because the
    sentence contains "my profile" replaced step 1 with step 2 and ran the plan
    one agent short; the audit measured exactly that, and `job_search` never
    ran. The personal-question pull below exists because the planner cannot see
    the store, and that reasoning is about a *single* question: when the plan
    already contains a profile step, the pull is not protecting anything.

    **Otherwise a personal question goes to profile.** Whatever the planner
    concluded, because the planner scores intent without seeing the store it
    would otherwise ask the user to substitute for.

    The academic carve-out that used to live in the last rule is gone. It read
    "keep academic if the planner chose it", and since every category academic
    legitimately owns is already handled by rule one, the only turns it could
    ever affect were personal questions the planner had misrouted — sending
    "what is my CGPA" to an agent with no education tools. It protected
    nothing and leaked the one thing the rule exists to prevent.
    """
    owned = _CATEGORY_AGENT.get(decision.category)
    if owned:
        return owned

    choice = planner_choice if planner_choice in ("job", "email", "academic", "profile") else "profile"

    if plan_steps > 1:
        return choice

    if decision.is_personal or decision.category in (
        QueryCategory.EXPLICIT_MEMORY_WRITE,
        QueryCategory.IDENTITY_UPDATE,
    ):
        return "profile"

    return choice
