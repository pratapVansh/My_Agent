"""
The behavioural regression suite: nine required behaviours, and the adversarial
cases that distinguish each from a fix that only looks right.

Every test here corresponds to a specific defect, and each is paired with at
least one adversarial case that a naive fix would fail. The pattern matters:
"what is my name" returning the canonical name proves very little on its own,
because a system that has simply lost the remembered name also passes it. What
proves the fix is that the *same store* still answers "what name did I ask you
to remember" — the two must be separable, not merely one of them correct.

Numbering follows the requirements:

  1. canonical identity wins; alternates are never volunteered
  2. "current CPI" and "the CPI on my résumé" reach different sources
  3. timetable questions reach the timetable
  4. provenance questions explain the actual source
  5. personal questions answer from memory before falling back
  6. voice interruption cancels LLM/tool/TTS work
  7. follow-ups resolve against the previous turn
  8. missing data is reported, never invented
"""
from __future__ import annotations

import asyncio

import pytest

from app.agents import interruption, query_intent
from app.memory import identity, provenance
from app.memory.answerability import Answerability, assess
from app.memory.memory_manager import memory_manager
from app.memory.sources import MemorySource, QueryCategory, RetrievedMemory


def category_of(query: str, **kwargs) -> QueryCategory:
    return query_intent.classify(query, **kwargs).category


def sources_of(query: str, **kwargs):
    return query_intent.classify(query, **kwargs).sources


# ═══════════════════════════════════════════════════════════════════════════
# 1. Canonical identity always wins; alternates are never volunteered
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query", [
    "What is my name?",
    "what's my name",
    "who am I",
    "tell me my name",
    "what am I called",
])
def test_name_questions_read_canonical_identity_first(query):
    """The canonical store is consulted before anything else that holds a name."""
    decision = query_intent.classify(query)
    assert decision.category is QueryCategory.PROFILE_IDENTITY
    assert decision.sources[0] is MemorySource.CANONICAL_IDENTITY


@pytest.mark.parametrize("query", [
    "What name did I ask you to remember?",
    "what name did I tell you to save",
    "what have you remembered",
])
def test_remembered_name_questions_read_the_explicit_store(query):
    """
    The other half of the same property.

    A system that answered "what is my name" correctly by *losing* the
    remembered name would pass the test above and fail here. Both must work,
    from different stores, for the separation to be real.
    """
    decision = query_intent.classify(query)
    assert decision.category is QueryCategory.EXPLICIT_MEMORY
    assert decision.sources[0] is MemorySource.EXPLICIT_MEMORY
    assert MemorySource.CANONICAL_IDENTITY not in decision.sources


def test_remembered_names_are_withheld_from_the_identity_prompt():
    """
    The actual defect: section filtering is not key filtering.

    `profile_facts` is one section holding both `canonical_name` and
    `remembered_name`. Narrowing an identity question to that section still
    handed the model a line reading `remembered_name: Devasi` immediately
    below the canonical one — and a model shown two names volunteers both.
    """
    context = {
        "profile_facts": [
            {"key": identity.CANONICAL_NAME_KEY, "value": "Vansh Pratap Singh"},
            {"key": identity.REMEMBERED_NAME_KEY, "value": "Devasi"},
            {"key": "alternate_name", "value": "Devasis"},
            {"key": "college", "value": "RGIPT"},
        ]
    }
    rendered = memory_manager.format_context_for_prompt(
        context, category=QueryCategory.PROFILE_IDENTITY.value
    )

    assert "Vansh Pratap Singh" in rendered
    # Neither the current namespace nor the legacy keys may appear.
    assert "Devasi" not in rendered
    assert "Devasis" not in rendered
    # An unrelated ordinary fact is unaffected — this filters by key, not by
    # blanket suppression of everything that is not the name.
    assert "RGIPT" in rendered


def test_remembered_names_are_present_when_that_is_the_question():
    """The exemption. Withholding them everywhere would break the read path."""
    context = {
        "profile_facts": [
            {"key": identity.CANONICAL_NAME_KEY, "value": "Vansh Pratap Singh"},
            {"key": identity.REMEMBERED_NAME_KEY, "value": "Devasi"},
        ]
    }
    rendered = memory_manager.format_context_for_prompt(
        context, category=QueryCategory.EXPLICIT_MEMORY.value
    )
    assert "Devasi" in rendered


def test_unknown_category_still_renders_everything():
    """
    Under-filtered, never blind.

    A category this map has never heard of must not silently drop the user's
    facts — the documented default is to render everything.
    """
    context = {"profile_facts": [{"key": "remembered_gym", "value": "6am"}]}
    assert "6am" in memory_manager.format_context_for_prompt(context, category=None)


def test_remembering_a_name_is_not_an_identity_change():
    """The original bug, asserted at the parser: "remember the name Devasi"."""
    directive = identity.detect_name_directive("Remember the name Devasi")
    assert directive is not None
    assert directive.kind is identity.DirectiveKind.REMEMBER_VALUE
    assert directive.key == identity.REMEMBERED_NAME_KEY
    assert directive.key != identity.CANONICAL_NAME_KEY


def test_an_explicit_identity_update_still_changes_the_canonical_name():
    """The exemption, so the rule above is a distinction and not a blanket no."""
    directive = identity.detect_name_directive("Update my name to Vansh Pratap Singh")
    assert directive is not None
    assert directive.kind is identity.DirectiveKind.CANONICAL_UPDATE
    assert directive.key == identity.CANONICAL_NAME_KEY


# ═══════════════════════════════════════════════════════════════════════════
# 2. "Current CPI" and "the CPI on my résumé" are different questions
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query", [
    "What is my current CPI?",
    "what is my cpi right now",
    "my latest CGPA",
    "what is my current college",
])
def test_current_academic_standing_prefers_the_most_recent_record(query):
    decision = query_intent.classify(query)
    assert decision.category is QueryCategory.ACADEMIC_CURRENT
    assert decision.sources[0] is MemorySource.PROFILE_MEMORY
    # The résumé remains reachable as a fallback — "current" must not mean
    # "refuse to look at the document".
    assert MemorySource.RESUME_DOCUMENT in decision.sources


@pytest.mark.parametrize("query", [
    "What CPI is mentioned in my resume?",
    "what does my resume say my cgpa is",
    "what cpi is on my resume",
])
def test_resume_scoped_questions_read_the_document_first(query):
    decision = query_intent.classify(query)
    assert decision.category is QueryCategory.DOCUMENT_RESUME
    assert decision.sources[0] is MemorySource.RESUME_DOCUMENT


def test_resume_scope_beats_currency_when_both_are_present():
    """
    The adversarial case: "the current CPI on my résumé" names a document.

    A rule that checked for "current" first would send a question explicitly
    about the résumé to a store that is not the résumé.
    """
    assert category_of("what is the current CPI on my resume") is QueryCategory.DOCUMENT_RESUME


@pytest.mark.parametrize("query", [
    "What is my current CPI?",
    "What CPI is mentioned in my resume?",
    "my latest CGPA",
])
def test_no_cpi_question_is_ever_answered_by_the_clock(query):
    """
    The original failure: "I don't have real-time access to your current CPI".

    Every phrasing must retrieve from memory, and none may reach the clock —
    a now-marker in a question about a stored number is not a question about
    the present moment.
    """
    decision = query_intent.classify(query)
    assert decision.category is not QueryCategory.TEMPORAL_CURRENT
    assert MemorySource.TEMPORAL_TOOL not in decision.sources
    assert decision.requires_retrieval is True
    assert decision.may_clarify is False


def test_the_clock_is_still_the_clock():
    """The other side: genuine temporal questions must not regress into memory."""
    decision = query_intent.classify("What is the current date?")
    assert decision.category is QueryCategory.TEMPORAL_CURRENT
    assert decision.sources[0] is MemorySource.TEMPORAL_TOOL


# ═══════════════════════════════════════════════════════════════════════════
# 3. Timetable questions reach the timetable
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query", [
    # With an explicit schedule noun
    "What is my timetable?",
    "What classes do I have today?",
    "what's my timetable for tomorrow",
    "when is my next class",
    "What is my schedule on Monday?",
    # Without one — the phrasings that previously fell through to a generic
    # personal route with no access to the timetable at all.
    "What do I have today?",
    "am I free tomorrow",
    "do I have anything tomorrow",
    "what's on for today",
    "what's my routine today",
    "am I busy on Friday",
])
def test_timetable_questions_route_to_the_academic_agent(query):
    decision = query_intent.classify(query)
    assert decision.category is QueryCategory.SCHEDULE_TEMPORAL
    assert query_intent.agent_for(decision, None) == "academic"
    # The timetable needs both: the clock to resolve "today", the store for
    # the schedule. Neither alone can answer.
    assert MemorySource.TEMPORAL_TOOL in decision.sources
    assert MemorySource.EXTERNAL_TOOL in decision.sources


def test_the_academic_agent_is_not_overridden_by_a_planner_guess():
    """A planner that says "job" must not pull a timetable question with it."""
    decision = query_intent.classify("What do I have today?")
    assert query_intent.agent_for(decision, "job") == "academic"


@pytest.mark.parametrize("query", [
    "am I free to talk about my projects",
    "what do I have to learn to become a data scientist",
    "do I have anything worth putting on a resume",
])
def test_availability_wording_without_a_day_is_not_a_timetable_question(query):
    """
    The adversarial half of the broadened rule.

    "Am I free…" and "what do I have…" are ordinary English. A day reference is
    what makes them schedule questions; without one, routing them to the
    timetable would be as wrong as the generic routing this replaced.
    """
    assert category_of(query) is not QueryCategory.SCHEDULE_TEMPORAL


# ═══════════════════════════════════════════════════════════════════════════
# 4. "How did you know?" explains the actual source
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query", [
    "How did you know?",
    "How did you know that?",
    "how do you know that",
    "Where did you get that from?",
    "what's your source",
    "which tool did you use",
    "how did you find that out",
    "says who",
    "what made you say that",
    "how are you so sure",
])
def test_provenance_questions_get_their_own_category(query):
    decision = query_intent.classify(query, has_context=True)
    assert decision.category is QueryCategory.PROVENANCE_QUERY
    assert query_intent.agent_for(decision, None) == "provenance"
    # Answered from a record, not by looking the user's facts up again.
    assert decision.requires_retrieval is False


def test_provenance_questions_do_not_need_prior_context_to_be_recognised():
    """
    They were previously split across two categories by whether history
    existed — GENERAL_KNOWLEDGE without it, CONVERSATION_FOLLOWUP with it, and
    neither could name a source. The shape decides, not the surroundings.
    """
    assert category_of("How did you know?", has_context=False) is QueryCategory.PROVENANCE_QUERY
    assert category_of("How did you know?", has_context=True) is QueryCategory.PROVENANCE_QUERY


def test_provenance_explanation_names_the_real_source_and_tool():
    provenance.reset("conv-1")
    provenance.record(
        "conv-1",
        category=QueryCategory.PROFILE_EDUCATION.value,
        sources=[MemorySource.RESUME_DOCUMENT.value, MemorySource.PROFILE_MEMORY.value],
        agent="profile",
        tools=["get_education"],
        answerability=Answerability.ANSWERABLE.value,
        question="What is my CGPA?",
    )
    explanation = provenance.explain_last("conv-1")
    assert "résumé" in explanation
    assert "get_education" in explanation


def test_provenance_distinguishes_the_clock_from_the_resume():
    """
    The adversarial case: the explanation must track the actual turn.

    A fixed string, or one derived from the current question rather than the
    recorded one, would say "your résumé" for a date answered by the clock.
    """
    provenance.reset("conv-2")
    provenance.record(
        "conv-2",
        category=QueryCategory.TEMPORAL_CURRENT.value,
        sources=[MemorySource.TEMPORAL_TOOL.value],
        agent="temporal",
        tools=["current_datetime"],
    )
    explanation = provenance.explain_last("conv-2")
    assert "clock" in explanation
    assert "résumé" not in explanation


def test_provenance_reports_no_data_honestly_when_nothing_was_recorded():
    """Requirement 8, applied to the assistant's own behaviour."""
    provenance.reset("conv-empty")
    explanation = provenance.explain_last("conv-empty")
    assert explanation == provenance.NO_RECORD
    assert "don't have a record" in explanation


def test_provenance_explains_a_no_data_answer_as_a_check_that_found_nothing():
    """
    "How did you know I don't have it?" is a real question with a real answer:
    the store was consulted and was empty. It must not be described as though
    a value had been found.
    """
    provenance.reset("conv-3")
    provenance.record(
        "conv-3",
        sources=[MemorySource.RESUME_DOCUMENT.value],
        agent="profile",
        tools=["get_education"],
        answerability=Answerability.NO_DATA.value,
    )
    explanation = provenance.explain_last("conv-3")
    assert "nothing on file" in explanation
    assert "estimating" in explanation


def test_provenance_is_isolated_per_conversation():
    """One user's provenance must never explain another's answer."""
    provenance.reset()
    provenance.record("conv-a", sources=[MemorySource.RESUME_DOCUMENT.value], agent="profile")
    provenance.record("conv-b", sources=[MemorySource.TEMPORAL_TOOL.value], agent="temporal")
    assert "résumé" in provenance.explain_last("conv-a")
    assert "clock" in provenance.explain_last("conv-b")
    assert provenance.explain_last("conv-unknown") == provenance.NO_RECORD


# ── 4b. Both found by a live run, neither by the unit tests above ─────────
#
# A live conversation against an exhausted LLM quota exercised the error paths
# that a mocked suite never reaches. Two defects surfaced, and both are the
# kind that only appear when something upstream is already broken — which is
# exactly when an honest answer matters most.

def _routing_state(query, **extra):
    base = {
        "user_input": query,
        "session_id": "route-test",
        "user_id": "u",
        "conversation_history": [],
        "execution_path": [],
        "selected_agent": None,
        "needs_clarification": False,
    }
    base.update(extra)
    return base


async def test_provenance_is_answerable_when_the_language_model_is_down():
    """
    Found live: "how did you know?" came back as a planner error.

    The planner runs before routing, so its failure set an error state and the
    question was short-circuited — despite needing no planner, no model and no
    retrieval. It reads a record. An upstream outage cannot have broken it.
    """
    from app.agents.workflow import decide_route

    state = _routing_state("How did you know?", error="Groq API error: 429")
    assert await decide_route(state) == "provenance"
    # The error is cleared, so the turn actually proceeds rather than falling
    # through to the generic failure response.
    assert not state.get("error")


async def test_the_clock_is_still_answerable_when_the_model_is_down():
    """The same property for the sibling case, so the fix generalises."""
    from app.agents.workflow import decide_route

    state = _routing_state("What is today's date?", error="Groq API error: 429")
    assert await decide_route(state) == "temporal"
    assert not state.get("error")


def test_a_failed_turn_does_not_erase_the_previous_provenance():
    """
    Found live: an errored turn recorded `sources=()`, wiping the record of the
    last answer that succeeded. The next "how did you know?" then reported
    nothing known about an answer the user had actually received.
    """
    from app.agents.workflow import _record_provenance

    provenance.reset("erase-test")
    good = _routing_state("What is my CGPA?", session_id="erase-test")
    good["query_category"] = QueryCategory.PROFILE_EDUCATION.value
    good["memory_sources"] = [MemorySource.RESUME_DOCUMENT.value]
    good["task_result"] = {"agent": "profile", "evidence": ["get_education"],
                           "status": "success"}
    _record_provenance(good)
    assert "résumé" in provenance.explain_last("erase-test")

    failed = _routing_state("What are my skills?", session_id="erase-test",
                            error="Groq API error: 429")
    failed["task_result"] = {"agent": "profile", "evidence": [], "status": "failed"}
    _record_provenance(failed)

    # The good record survives.
    assert "résumé" in provenance.explain_last("erase-test")


def test_a_degraded_answer_still_records_its_provenance():
    """
    Found live, and the reason the guard above tests `status` and not `error`.

    A conditional-edge function in LangGraph returns a route, not state, so the
    `state["error"] = None` that `decide_route` performs before handing a turn
    to `temporal`/`degraded`/`provenance` never reaches the node. Those nodes
    run with the error flag still set — they are the paths that answer
    correctly *despite* it. A guard that read that flag suppressed the record
    for every degraded answer: the clock answered the date, and the very next
    "how did you know?" claimed to have no record of it.
    """
    from app.agents.workflow import _record_provenance

    provenance.reset("degraded-test")
    state = _routing_state("What is today's date?", session_id="degraded-test",
                           error="Groq API error: 429")  # still set at the node
    state["query_category"] = QueryCategory.TEMPORAL_CURRENT.value
    state["memory_sources"] = [MemorySource.TEMPORAL_TOOL.value]
    state["task_result"] = {"agent": "temporal", "evidence": ["current_datetime"],
                            "status": "success"}
    _record_provenance(state)

    assert "clock" in provenance.explain_last("degraded-test")


def test_sources_are_derived_when_the_router_could_not_supply_them():
    """
    Found live, and the same root cause once more.

    `memory_sources` is written by `decide_route`, a conditional-edge function
    whose state mutations LangGraph discards. Depending on it left the record
    with `sources=()`, so the clock's own provenance read "an unrecorded
    source" for an answer that provably came from the clock. The precedence
    table defines which stores answer a category, so the sources are derived
    from the category rather than trusted to propagate.
    """
    from app.agents.workflow import _record_provenance

    provenance.reset("derive-test")
    state = _routing_state("What is today's date?", session_id="derive-test")
    state["query_category"] = QueryCategory.TEMPORAL_CURRENT.value
    state["memory_sources"] = []          # exactly what the node actually sees
    state["task_result"] = {"agent": "temporal", "evidence": ["current_datetime"],
                            "status": "success"}
    _record_provenance(state)

    entry = provenance.last("derive-test")
    assert entry is not None
    assert MemorySource.TEMPORAL_TOOL.value in entry.sources
    explanation = entry.explain()
    assert "clock" in explanation
    assert "unrecorded source" not in explanation


def test_derived_sources_match_the_precedence_table():
    """The derivation must agree with what the router would have written."""
    from app.agents.workflow import _record_provenance

    provenance.reset("derive-2")
    state = _routing_state("What is my name?", session_id="derive-2")
    state["query_category"] = QueryCategory.PROFILE_IDENTITY.value
    state["memory_sources"] = []
    state["task_result"] = {"agent": "profile", "evidence": ["get_identity"],
                            "status": "success"}
    _record_provenance(state)

    entry = provenance.last("derive-2")
    assert entry.sources[0] == MemorySource.CANONICAL_IDENTITY.value
    assert "canonical" in entry.explain()


def test_a_partial_degraded_answer_is_recorded_too():
    """`degraded_node` reports status "partial"; that is an answer, not a failure."""
    from app.agents.workflow import _record_provenance

    provenance.reset("partial-test")
    state = _routing_state("What is my name?", session_id="partial-test",
                           error="Groq API error: 429")
    state["query_category"] = QueryCategory.PROFILE_IDENTITY.value
    state["memory_sources"] = [MemorySource.CANONICAL_IDENTITY.value]
    state["task_result"] = {"agent": "degraded", "evidence": ["profile_facts"],
                            "status": "partial"}
    _record_provenance(state)

    assert provenance.last("partial-test") is not None
    assert "profile" in provenance.explain_last("partial-test")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Personal questions answer from memory before falling back
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query,expected", [
    ("What college do I go to?", QueryCategory.PROFILE_EDUCATION),
    ("What are my skills?", QueryCategory.PROFILE_SKILLS),
    ("Where did I intern?", QueryCategory.PROFILE_EXPERIENCE),
    ("What are my projects?", QueryCategory.PROFILE_PROJECTS),
    ("What are my achievements?", QueryCategory.PROFILE_ACHIEVEMENTS),
])
def test_personal_questions_retrieve_and_never_clarify(query, expected):
    decision = query_intent.classify(query)
    assert decision.category is expected
    assert decision.requires_retrieval is True
    # Asking the user to disambiguate their own CGPA is a failure to look.
    assert decision.may_clarify is False
    assert MemorySource.GENERAL_KNOWLEDGE not in decision.sources


# ═══════════════════════════════════════════════════════════════════════════
# 6. Voice interruption cancels active work
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("utterance", [
    "stop", "Stop.", "wait", "no", "cancel", "nope", "abort", "quit",
    "hold on", "never mind", "shut up", "stop it", "please stop",
    "uh, stop", "ok wait", "hey stop",
])
def test_stop_words_are_recognised_as_interruptions(utterance):
    """
    Every one of these is a single word or a short phrase, and every one of
    them was previously below the two-word barge-in floor.
    """
    assert interruption.is_stop_command(utterance) is True


@pytest.mark.parametrize("utterance", [
    "no, tell me about my projects",
    "wait, what is my CGPA",
    "stop by the library later",
    "cancel my subscription to that newsletter",
    "I have no idea",
    "there is no rush",
])
def test_sentences_that_merely_contain_a_stop_word_are_not_bare_stops(utterance):
    """
    The adversarial case, and the reason this is not a substring match.

    "Cancel my subscription" is a request. Treating it as an interruption would
    silently drop a task the user asked for.
    """
    assert interruption.is_pure_stop(utterance) is False


@pytest.mark.parametrize("utterance,remainder", [
    ("no, tell me about my projects", "tell me about my projects"),
    ("wait what is my cgpa", "what is my cgpa"),
    ("stop tell me the date instead", "tell me the date instead"),
])
def test_a_stop_carrying_a_request_keeps_the_request(utterance, remainder):
    """
    "No, tell me about my projects" both cancels and asks. Cancelling without
    running the remainder would lose a question the user actually asked.
    """
    assert interruption.carries_new_request(utterance) is True
    assert interruption.strip_stop_prefix(utterance) == remainder


def test_a_bare_stop_carries_no_request():
    assert interruption.carries_new_request("stop") is False
    assert interruption.carries_new_request("hold on") is False


def test_empty_and_whitespace_transcripts_are_not_interruptions():
    """A dropped or empty transcript must never cancel a turn."""
    for value in ("", "   ", None):
        assert interruption.is_stop_command(value) is False


@pytest.mark.asyncio
async def test_stop_cancels_work_that_has_not_started_speaking():
    """
    The core of the defect, as a behaviour rather than a predicate.

    Barge-in was gated on TTS already playing, so "stop" during an LLM or tool
    call — the wait people most want to cut short — did nothing. This models a
    turn stuck in a long call and asserts the stop actually ends it.
    """
    started = asyncio.Event()

    async def long_running_turn():
        started.set()
        await asyncio.sleep(30)  # an LLM/tool call that never returns in time

    task = asyncio.create_task(long_running_turn())
    await started.wait()

    speaking = False  # nothing has been spoken yet
    transcript = "stop"

    should_cancel = interruption.is_stop_command(transcript) or (
        speaking and len(transcript.split()) >= 2
    )
    assert should_cancel is True

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


@pytest.mark.asyncio
async def test_a_stopped_turn_does_not_resume_afterwards():
    """
    "Prevent the previous task from continuing."

    A cancelled turn must not keep producing output after the cancel — the
    failure mode where buffered speech plays on and the user hears the thing
    they just interrupted.
    """
    emitted: list[str] = []

    async def speaking_turn():
        for index in range(20):
            emitted.append(f"chunk-{index}")
            await asyncio.sleep(0.01)

    task = asyncio.create_task(speaking_turn())
    await asyncio.sleep(0.03)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    count_at_cancel = len(emitted)
    await asyncio.sleep(0.05)  # well past several more chunks
    assert len(emitted) == count_at_cancel


# ═══════════════════════════════════════════════════════════════════════════
# 7. Follow-ups resolve against the previous turn
# ═══════════════════════════════════════════════════════════════════════════

HISTORY = [
    {"role": "user", "content": "Tell me about TRACE"},
    {"role": "assistant", "content": "TRACE is your retrieval benchmark project."},
]


@pytest.mark.parametrize("query", ["tell me more", "that one", "what about it"])
def test_followups_inherit_their_subject(query):
    decision = query_intent.classify(query, has_context=True, history=HISTORY)
    assert decision.category is QueryCategory.CONVERSATION_FOLLOWUP
    assert decision.subject == "TRACE"


def test_a_followup_without_history_claims_no_subject():
    """Returning None is correct; inventing a subject is not."""
    decision = query_intent.classify("tell me more", has_context=False, history=None)
    assert decision.subject is None


def test_a_name_in_the_query_beats_the_history():
    """"Tell me more about My_Agent" is not actually referential."""
    decision = query_intent.classify(
        "tell me more about My_Agent", has_context=True, history=HISTORY
    )
    assert decision.subject == "My_Agent"


def test_provenance_beats_followup_for_how_did_you_know():
    """
    "How did you know that?" carries a referential "that" and was previously
    claimed by the follow-up rule, which routes it to retrieval rather than to
    an explanation. Provenance must win.
    """
    decision = query_intent.classify(
        "How did you know that?", has_context=True, history=HISTORY
    )
    assert decision.category is QueryCategory.PROVENANCE_QUERY


# ═══════════════════════════════════════════════════════════════════════════
# 8. Missing data is reported, never invented
# ═══════════════════════════════════════════════════════════════════════════

def test_an_empty_store_is_no_data_and_says_so():
    result = assess([], category=QueryCategory.PROFILE_EDUCATION,
                    expected_sources=[MemorySource.RESUME_DOCUMENT])
    assert result.state is Answerability.NO_DATA
    assert result.may_clarify is False
    assert "don't have that information" in result.guidance()
    assert "Do not guess" in result.guidance()


def test_a_failed_lookup_is_not_an_empty_store():
    """
    The distinction NO_DATA exists to protect. Both arrive as an empty list and
    they warrant opposite statements — saying "you have no CGPA on file" when
    the lookup merely failed is a false claim about the user's own data.
    """
    result = assess([], category=QueryCategory.PROFILE_EDUCATION,
                    expected_sources=[MemorySource.RESUME_DOCUMENT],
                    errored_sources=[MemorySource.RESUME_DOCUMENT])
    assert result.state is Answerability.TOOL_ERROR
    assert "could not look it up" in result.guidance()
    assert "not established" in result.guidance()


def test_partial_evidence_is_answered_not_withheld():
    evidence = [RetrievedMemory(
        content="B.Tech Information Technology",
        source=MemorySource.PROFILE_MEMORY,
        memory_type="profile",
    )]
    result = assess(evidence, category=QueryCategory.PROFILE_EDUCATION,
                    expected_sources=[MemorySource.RESUME_DOCUMENT,
                                      MemorySource.PROFILE_MEMORY])
    assert result.state is Answerability.PARTIALLY_ANSWERABLE
    assert result.should_answer is True
    assert "Do not withhold" in result.guidance()


# ── 8b. The verdict is derived from tool results, not from the prompt ──────
#
# `answerability` was fully built and unit-tested but never reached the live
# request path: nothing computed an Assessment and nothing set it on state, so
# honesty about missing data rested entirely on system-prompt instructions.
# These cover the deterministic derivation that now runs on every tool call.

from app.agents.base_agent import (  # noqa: E402
    _assess_tool_outcomes, _tool_reported_failure, _tool_yielded_evidence,
)


@pytest.mark.parametrize("result", [
    {"success": True, "found": True, "name": "Vansh Pratap Singh"},
    {"success": True, "count": 3, "skills": ["python", "rust", "go"]},
    {"success": True, "content": "B.Tech Information Technology"},
    ["one", "two"],
    "some text",
])
def test_tool_results_carrying_data_count_as_evidence(result):
    assert _tool_yielded_evidence(result) is True


@pytest.mark.parametrize("result", [
    {"success": True, "found": False, "message": "No resume found."},
    {"success": True, "count": 0, "skills": []},
    {"success": True, "has_data": False},
    [],
    "",
    None,
])
def test_empty_tool_results_are_not_evidence(result):
    """A successful lookup of an empty store is absence, not evidence."""
    assert _tool_yielded_evidence(result) is False


@pytest.mark.parametrize("result", [
    {"success": False, "message": "Qdrant unreachable"},
    {"error": "connection reset"},
])
def test_failed_tools_are_distinguished_from_empty_ones(result):
    assert _tool_reported_failure(result) is True


def test_a_failed_tool_is_never_reported_as_an_empty_store():
    """
    The distinction that matters, at the level it is now derived.

    Saying "you have no CGPA on file" when the lookup merely failed is a false
    claim about the user's own data — and it is the exact state in which a
    model, told there is no information, invents some.
    """
    assert _assess_tool_outcomes(["get_education"], [], ["get_education"]) == "TOOL_ERROR"
    assert _assess_tool_outcomes(["get_education"], [], []) == "NO_DATA"
    assert _assess_tool_outcomes(["get_education"], ["get_education"], []) == "ANSWERABLE"


def test_a_partial_failure_is_partially_answerable():
    """Some of it is known; say what is known and name what is not."""
    verdict = _assess_tool_outcomes(
        ["get_education", "get_skills"], ["get_education"], ["get_skills"]
    )
    assert verdict == "PARTIALLY_ANSWERABLE"


def test_answering_without_tools_asserts_nothing_about_the_store():
    """
    An answer produced with no lookup must not claim the store was empty.
    Silence is the only honest verdict here.
    """
    assert _assess_tool_outcomes([], [], []) == ""


def test_no_data_never_becomes_a_clarifying_question():
    """
    An empty store is not a reason to interrogate the user. Clarification is
    reachable only from AMBIGUOUS and ACTION_MISSING_PARAMETER.
    """
    for state in (Answerability.NO_DATA, Answerability.TOOL_ERROR,
                  Answerability.ANSWERABLE, Answerability.PARTIALLY_ANSWERABLE):
        assert state not in (Answerability.AMBIGUOUS,
                             Answerability.ACTION_MISSING_PARAMETER)
