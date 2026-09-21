"""
Memory intelligence: knowing which source answers which question.

The store was never the problem. Every fact these tests ask for was already
present and already retrievable when the failures below were reported — the
system simply had no layer that decided *where to look*, so a question about a
stored number was answered by declining to look anything up:

    "What is my current CPI?"
    "I don't have real-time access to your current CPI."

Five distinct defects produced that class of answer, and each has tests here:

1. **No source selection.** Every query received every memory source flattened
   into one prompt block, leaving the model to infer relevance from a wall of
   text. `CPI` was not even in the CGPA vocabulary, so the query matched no
   subject at all and fell through to the planner.
2. **No canonical identity.** "Remember the name Devasi" and "update my name to
   X" differ only in clause; both were treated as identity writes, and the
   user's name was overwritten by a name they asked to have stored beside it.
3. **No clock.** Current date and time were sought in memory, which cannot hold
   them, and the honest-sounding "no real-time access" followed.
4. **Clarification ahead of retrieval, without a budget.** A general request for
   an explanation was met with successive scoping questions, including after the
   user asked for fewer of them.
5. **Indiscriminate writes.** Every utterance was embedded verbatim, so "Okay."
   and "You are fool." became long-term memories competing with the résumé for
   retrieval.

Each test asserts the decision, not just the outcome: the selected category, the
selected source, whether retrieval was required, whether clarification occurred
and why, and provenance where it applies.
"""
import pytest

from app.agents import clarification_policy, query_intent
from app.agents.workflow import decide_route, route_after_init
from app.memory import identity, write_policy
from app.memory.answerability import Answerability, Assessment, assess, from_retrieval_result
from app.memory.memory_manager import memory_manager
from app.memory.retrieval_result import RetrievalResult
from app.memory.sources import (
    MemorySource,
    QueryCategory,
    RetrievedMemory,
    sources_for,
)


@pytest.fixture(autouse=True)
def _clean_clarification_budget():
    """Each test starts with a fresh, unspent clarification budget."""
    clarification_policy.reset()
    yield
    clarification_policy.reset()


def state(**overrides):
    base = {
        "user_input": "",
        "session_id": "",
        "memory_context": {},
        "memory_prompt": "",
        "selected_agent": "profile",
        "needs_clarification": False,
        "planner_confidence": 0.9,
    }
    base.update(overrides)
    return base


TRACE_HISTORY = [
    {"role": "user", "content": "Tell me about TRACE."},
    {"role": "assistant", "content": "TRACE is your resume-analysis project."},
]


def category_of(query, **kwargs):
    return query_intent.classify(query, **kwargs).category


# ═════════════════════════════════════════════════════════════════════════
# 1–8. Résumé-backed personal questions reach the right section
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "query,expected",
    [
        # 1 & 2 — the reported failure, in both phrasings. CPI is what this
        # user's institute calls the CGPA; its absence from the vocabulary is
        # the whole of the bug.
        ("What is my CPI?", QueryCategory.PROFILE_EDUCATION),
        # "current" and "on my résumé" are the same question about the same
        # number right up until the two disagree, which is when it matters.
        # Each now names the store that actually answers it.
        ("What is my current CPI?", QueryCategory.ACADEMIC_CURRENT),
        ("Can you tell what CPI is mentioned in my resume?", QueryCategory.DOCUMENT_RESUME),
        ("What is my SPI this semester?", QueryCategory.PROFILE_EDUCATION),
        # 3 & 4 — internships and experience
        ("Where did I do my internship?", QueryCategory.PROFILE_EXPERIENCE),
        ("What company did I intern at?", QueryCategory.PROFILE_EXPERIENCE),
        ("Which companies have I worked at?", QueryCategory.PROFILE_EXPERIENCE),
        ("Tell me about my experience.", QueryCategory.PROFILE_EXPERIENCE),
        # 5 — education
        ("Which college do I study at?", QueryCategory.PROFILE_EDUCATION),
        ("What degree am I pursuing?", QueryCategory.PROFILE_EDUCATION),
        # 6 — projects
        ("What are my projects?", QueryCategory.PROFILE_PROJECTS),
        ("Tell me about my projects.", QueryCategory.PROFILE_PROJECTS),
        # 7 — skills
        ("What skills do I have?", QueryCategory.PROFILE_SKILLS),
        # 8 — achievements
        ("What are my achievements?", QueryCategory.PROFILE_ACHIEVEMENTS),
    ],
)
def test_personal_questions_select_their_own_category(query, expected):
    assert category_of(query) is expected


@pytest.mark.parametrize(
    "query",
    [
        "What is my CPI?",
        "Can you tell what CPI is mentioned in my resume?",
        "Where did I do my internship?",
        "What are my projects?",
        "What skills do I have?",
        "What are my achievements?",
    ],
)
def test_resume_is_the_first_source_for_resume_backed_facts(query):
    """
    Résumé precedence, asserted as an ordering rather than a membership.

    A source list that merely *contains* the résumé is what the old undifferen-
    tiated context block already had. What was missing is that it comes first.
    """
    decision = query_intent.classify(query)
    assert decision.sources[0] is MemorySource.RESUME_DOCUMENT
    assert decision.requires_retrieval is True
    assert decision.may_clarify is False


@pytest.mark.parametrize(
    "query",
    [
        "What is my CPI?",
        "What is my current CPI?",
        "Where did I do my internship?",
        "What are my projects?",
        "What skills do I have?",
        "Which college do I study at?",
    ],
)
async def test_personal_questions_never_reach_clarification(query):
    """
    The acceptance list. Every one must retrieve even when the planner balked —
    the planner cannot see the store it is asking the user to substitute for.
    """
    s = state(
        user_input=query,
        needs_clarification=True,
        clarification_question="Could you be more specific?",
    )
    assert await route_after_init(s) == "profile"
    assert s["needs_clarification"] is False
    assert s["clarification_question"] == ""
    assert "resume_document" in s["memory_sources"]


# ═════════════════════════════════════════════════════════════════════════
# 9–11. Canonical identity is not a remembered name
# ═════════════════════════════════════════════════════════════════════════

def test_canonical_name_query_reads_canonical_identity():
    """9. "What is my name?" — identity first, résumé as fallback."""
    decision = query_intent.classify("What is my name?")
    assert decision.category is QueryCategory.PROFILE_IDENTITY
    assert decision.sources[0] is MemorySource.CANONICAL_IDENTITY
    assert MemorySource.RESUME_DOCUMENT in decision.sources
    # The remembered-name store must not be reachable from this question.
    assert MemorySource.EXPLICIT_MEMORY not in decision.sources


def test_remembering_a_name_is_not_an_identity_change():
    """10. The reported failure: "Remember the name Devasi" changed the name."""
    decision = query_intent.classify("Remember the name Devasi.")
    assert decision.category is QueryCategory.EXPLICIT_MEMORY_WRITE

    directive = identity.detect_name_directive("Remember the name Devasi.")
    assert directive is not None
    assert directive.kind is identity.DirectiveKind.REMEMBER_VALUE
    assert directive.value == "Devasi"
    assert directive.key == identity.REMEMBERED_NAME_KEY
    assert directive.key != identity.CANONICAL_NAME_KEY


def test_the_remembered_name_is_read_back_by_its_own_question():
    """10b. "What name did I ask you to remember?" — a different store."""
    decision = query_intent.classify("What name did I ask you to remember?")
    assert decision.category is QueryCategory.EXPLICIT_MEMORY
    assert decision.sources[0] is MemorySource.EXPLICIT_MEMORY
    assert MemorySource.CANONICAL_IDENTITY not in decision.sources


@pytest.mark.parametrize(
    "instruction,expected",
    [
        ("Update my name to Wanspatap Singh.", "Wanspatap Singh"),
        ("Change my name to Vansh.", "Vansh"),
        ("My legal name is Vansh Pratap Singh.", "Vansh Pratap Singh"),
        ("Please correct my official name to Vansh Singh", "Vansh Singh"),
    ],
)
def test_an_explicit_identity_update_is_permitted(instruction, expected):
    """11. Identity does change — but only when explicitly instructed to."""
    directive = identity.detect_name_directive(instruction)
    assert directive is not None
    assert directive.kind is identity.DirectiveKind.CANONICAL_UPDATE
    assert directive.value == expected
    assert directive.key == identity.CANONICAL_NAME_KEY
    assert query_intent.classify(instruction).category is QueryCategory.IDENTITY_UPDATE


@pytest.mark.parametrize(
    "utterance",
    [
        "My name is Devasi",
        "I met someone called Devasi today",
        "Devasi is my project partner",
        "The name Devasi comes from Sanskrit",
        "Use the name in my resume.",
    ],
)
def test_a_sentence_containing_a_name_is_not_an_identity_update(utterance):
    """
    The safety property, stated as a default.

    Silence here means identity is left alone. Every one of these sentences
    contains a name and a first-person context, and none of them asks for the
    user to be renamed.
    """
    directive = identity.detect_name_directive(utterance)
    assert directive is None or directive.kind is identity.DirectiveKind.REMEMBER_VALUE


def test_the_resume_name_is_reachable_as_a_typed_section():
    """
    The name was chunked at ingestion and never read back.

    `retrieve_resume` returned only content and metadata, so every caller doing
    `resume_data["name"]` — the profile summary among them — got None from a
    store that held the name the whole time. Without a source, "what is my
    name?" was answered from whatever else happened to be in context.
    """
    from app.memory.long_term_memory_qdrant import long_term_memory_qdrant

    assert "name" in long_term_memory_qdrant._SECTION_FILTERS
    assert long_term_memory_qdrant._SECTION_FILTERS["name"]["semantic_type"] == "name"


def test_legacy_remembered_keys_are_still_readable():
    """
    A live store already holds `alternate_name` from before the namespace
    existed. Renaming the convention must not orphan a value the user asked to
    have kept.
    """
    assert identity.is_explicit_memory_key("alternate_name") is True
    assert identity.is_explicit_memory_key("remembered_name") is True
    assert identity.is_explicit_memory_key("canonical_name") is False
    assert identity.is_explicit_memory_key("preferred_tone") is False


def test_a_generic_memory_write_cannot_claim_an_identity_key():
    """Even a well-formed write is redirected out of the canonical namespace."""
    assert identity.is_canonical_key(identity.CANONICAL_NAME_KEY) is True
    assert identity.is_canonical_key("preferred_tone") is False
    redirected = identity.explicit_key("name")
    assert redirected.startswith(identity.EXPLICIT_MEMORY_PREFIX)
    assert identity.is_canonical_key(redirected) is False


# ═════════════════════════════════════════════════════════════════════════
# 12–13. Follow-ups inherit their subject
# ═════════════════════════════════════════════════════════════════════════

def test_a_project_followup_resolves_its_subject_from_the_conversation():
    """12. "What technologies did it use?" — "it" is TRACE."""
    decision = query_intent.classify(
        "What technologies did it use?", has_context=True, history=TRACE_HISTORY
    )
    assert decision.category is QueryCategory.CONVERSATION_FOLLOWUP
    assert decision.subject == "TRACE"


async def test_tell_me_more_about_that_is_retrieval_not_a_new_question():
    """13. The referential follow-up must not restart intent classification."""
    s = state(
        user_input="Tell me more about that project.",
        conversation_history=TRACE_HISTORY,
        needs_clarification=True,
        clarification_question="Which project do you mean?",
    )
    assert await route_after_init(s) == "profile"
    assert s["query_category"] == QueryCategory.CONVERSATION_FOLLOWUP.value
    assert s["followup_subject"] == "TRACE"
    assert s["needs_clarification"] is False


@pytest.mark.parametrize(
    "query,subject",
    [
        ("Tell me about TRACE.", "TRACE"),
        ("Tell me about My_Agent.", "My_Agent"),
        ("What is TRACE?", "TRACE"),
    ],
)
def test_a_bare_question_about_a_named_thing_retrieves_first(query, subject):
    """
    "Tell me about TRACE." opens conversation C, with no history to inherit
    from. Nothing in the sentence says TRACE belongs to the user — but they have
    projects, and looking costs one lookup while not looking costs the answer.
    """
    decision = query_intent.classify(query)
    assert decision.category is QueryCategory.PROFILE_PROJECTS
    assert decision.subject == subject
    assert decision.requires_retrieval is True


@pytest.mark.parametrize(
    "query",
    [
        "What is RAG?",
        "What is an API?",
        "Explain RAG",
        "Tell me about the history of TRACE compilers",
        "tell me about yourself",
    ],
)
def test_vocabulary_questions_are_not_sent_to_the_resume(query):
    """
    The counterweight. Three-letter acronyms are overwhelmingly generic terms,
    and a sentence with more in it than a name is a general question that
    happens to mention one — neither is worth a résumé lookup.
    """
    assert query_intent.classify(query).category is QueryCategory.GENERAL_KNOWLEDGE


def test_a_followup_without_history_claims_no_subject():
    """With no prior turn there is nothing for "that" to refer to."""
    decision = query_intent.classify("what about that one", has_context=False)
    assert decision.subject is None


def test_conversation_and_episodic_recall_are_different_questions():
    """"Just told you" is this thread; "yesterday" is a past session."""
    assert category_of("What did I just tell you?") is QueryCategory.CONVERSATION_CURRENT
    assert category_of("What did I tell you yesterday?") is QueryCategory.EPISODIC_MEMORY
    assert sources_for(QueryCategory.EPISODIC_MEMORY)[0] is MemorySource.EPISODIC_MEMORY


# ═════════════════════════════════════════════════════════════════════════
# 14–17. The clock, not memory
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "query",
    ["What is today's date?", "What is the date today?", "what day is it"],
)
def test_date_questions_go_to_the_clock(query):
    """14. Memory cannot hold "now", and stored text about it is stale by design."""
    decision = query_intent.classify(query)
    assert decision.category is QueryCategory.TEMPORAL_CURRENT
    assert decision.sources == (MemorySource.TEMPORAL_TOOL,)
    assert decision.requires_retrieval is False


async def test_time_questions_go_to_the_clock():
    """15."""
    assert category_of("What time is it?") is QueryCategory.TEMPORAL_CURRENT
    assert await route_after_init(state(user_input="What time is it?")) == "temporal"


def test_the_clock_answers_without_a_model_call():
    from app.tools import time_tool

    answer = time_tool.answer_temporal_query("what is today's date?")
    assert answer
    assert "real-time" not in answer.lower()
    assert str(time_tool.today().year) in answer

    context = time_tool.current_context()
    assert context.date == time_tool.today()
    assert context.prompt_line().startswith("Current date and time:")


@pytest.mark.parametrize(
    "query", ["What class do I have today?", "What class do I have tomorrow?"]
)
async def test_schedule_questions_need_the_clock_and_memory_both(query):
    """
    16 & 17. The timetable is memory; "today" is not.

    Asserted as an ordered pair rather than a single source: answering from
    memory alone gets the wrong day, and answering from the clock alone has no
    schedule to report.
    """
    decision = query_intent.classify(query)
    assert decision.category is QueryCategory.SCHEDULE_TEMPORAL
    assert decision.sources[0] is MemorySource.TEMPORAL_TOOL
    assert MemorySource.EXTERNAL_TOOL in decision.sources
    assert decision.requires_retrieval is True
    # Timetable and attendance live in the academic agent, not profile.
    assert await route_after_init(state(user_input=query, selected_agent="profile")) == "academic"


async def test_the_clock_survives_an_upstream_failure():
    """
    A date needs neither the planner nor a model. When the LLM is rate-limited
    or down, withholding the one answer that requires nothing from it would be
    a self-inflicted outage.
    """
    s = state(user_input="What is today's date?", error="Planner error: 429")
    assert await route_after_init(s) == "temporal"
    assert s["error"] is None

    # Everything else genuinely needs what just broke.
    other = state(user_input="What is my CPI?", error="Planner error: 429")
    assert await route_after_init(other) == "response"


@pytest.mark.parametrize(
    "query,expected_route",
    [
        # A stored value read back verbatim needs no model.
        ("What is my name?", "degraded"),
        ("What name did I ask you to remember?", "degraded"),
        # Synthesis genuinely depends on what broke — a half-answer invented
        # from fragments is worse than an honest failure.
        ("Tell me about my projects.", "response"),
        ("Tell me how to build an AI agent.", "response"),
    ],
)
async def test_stored_answers_survive_an_llm_outage(query, expected_route):
    """
    Observed live, with Groq returning 429 on every call: every question became
    an error page, including ones whose answer is a database row. A memory
    system that cannot say the user's name because a language model is
    unavailable is not a memory system.
    """
    s = state(user_input=query, error="Planner error: 429 rate_limit_exceeded")
    assert await route_after_init(s) == expected_route


async def test_degraded_mode_clears_the_error_it_recovered_from():
    """The turn succeeded, so it must not also render as a failure."""
    s = state(user_input="What is my name?", error="Planner error: 429")
    await route_after_init(s)
    assert s["error"] is None
    assert "429" in s["degraded_reason"]


async def test_degraded_mode_reports_absence_and_unavailability_separately():
    """
    "I have no record of this" and "I couldn't think right now" must stay
    distinguishable — collapsing them is the failure this system is most
    careful about everywhere else.
    """
    from app.agents.workflow import degraded_node
    from app.memory.memory_manager import memory_manager as mm

    async def no_facts(user_id, key=None):
        return []

    async def no_resume(user_id):
        return None

    original_facts, original_resume = mm.get_profile_facts, mm.retrieve_resume
    mm.get_profile_facts, mm.retrieve_resume = no_facts, no_resume
    try:
        s = state(
            user_input="What is my name?",
            query_category=QueryCategory.PROFILE_IDENTITY.value,
            user_id="vansh", execution_path=[],
        )
        await degraded_node(s)
    finally:
        mm.get_profile_facts, mm.retrieve_resume = original_facts, original_resume

    answer = s["display_text"]
    assert "don't have your name on file" in answer
    assert "temporarily unavailable" in answer


async def test_degraded_mode_answers_from_the_stored_canonical_name():
    from app.agents.workflow import degraded_node
    from app.memory import identity
    from app.memory.memory_manager import memory_manager as mm

    async def facts(user_id, key=None):
        return [{"key": identity.CANONICAL_NAME_KEY, "value": "Vansh Pratap Singh"},
                {"key": identity.REMEMBERED_NAME_KEY, "value": "Devasi"}]

    original = mm.get_profile_facts
    mm.get_profile_facts = facts
    try:
        s = state(
            user_input="What is my name?",
            query_category=QueryCategory.PROFILE_IDENTITY.value,
            user_id="vansh", execution_path=[],
        )
        await degraded_node(s)
    finally:
        mm.get_profile_facts = original

    # The canonical name, and emphatically not the remembered one — the
    # identity separation must hold in degraded mode too, where none of the
    # prompt-level safeguards are running.
    assert "Vansh Pratap Singh" in s["display_text"]
    assert "Devasi" not in s["display_text"]
    assert s["task_result"]["status"] == "partial"


def test_the_configured_timezone_is_used_when_available():
    """
    Timezone comes from configuration, not the host: a server in UTC telling a
    user in Asia/Kolkata that it is still yesterday is the same class of error
    as answering the date from memory. Where tzdata is unavailable the fallback
    is the host clock, which must still produce a usable answer.
    """
    from app.config import settings
    from app.tools import time_tool

    context = time_tool.current_context()
    assert context.timezone_name in (settings.assistant_timezone, "")
    assert context.date == time_tool.today()
    assert context.time_sentence()


def test_relative_days_resolve_without_asking_the_user():
    from datetime import timedelta

    from app.tools import time_tool

    today = time_tool.today()
    assert time_tool.resolve_relative_day("today") == today
    assert time_tool.resolve_relative_day("tomorrow") == today + timedelta(days=1)
    assert time_tool.resolve_relative_day("yesterday") == today - timedelta(days=1)
    assert time_tool.resolve_relative_day("next Thursday") is None


def test_a_now_marker_does_not_make_a_stored_fact_temporal():
    """
    The precise misreading behind the reported failure.

    "current" appears in both "my current CPI" and "the current date". Only one
    of them is a question about the present moment, and treating the word as
    decisive is what routed a stored number to a clock that does not have it.

    The now-marker does route a stored fact to ACADEMIC_CURRENT rather than
    PROFILE_EDUCATION — that is the résumé/current split — but the property
    under test is unchanged and asserted directly: it must never become a
    question for the clock, and it must still be answered from memory.
    """
    assert category_of("What is my current CPI?") is QueryCategory.ACADEMIC_CURRENT
    assert category_of("What is my current college?") is QueryCategory.ACADEMIC_CURRENT
    assert category_of("What is the current date?") is QueryCategory.TEMPORAL_CURRENT

    for stored in ("What is my current CPI?", "What is my current college?"):
        decision = query_intent.classify(stored)
        assert decision.category is not QueryCategory.TEMPORAL_CURRENT
        assert MemorySource.TEMPORAL_TOOL not in decision.sources
        assert decision.requires_retrieval is True


# ═════════════════════════════════════════════════════════════════════════
# 18–20. Clarification is the last resort, and it is budgeted
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "query",
    [
        "Tell me how I can build an AI agent.",
        "Tell me how to build an AI agent.",
        "How do I build a virtual assistant?",
        "Explain how retrieval augmented generation works.",
    ],
)
async def test_a_general_question_is_answered_not_scoped(query):
    """
    18. The reported loop: "what type of AI agent?", then more questions.

    Note what makes this hard — "how do I build an agent" contains a first-person
    subject and a build verb, which the personal-information classifier reads as
    a question about the user's own projects. Ownership, not person, is the
    distinguishing signal.
    """
    decision = query_intent.classify(query)
    assert decision.category is QueryCategory.GENERAL_KNOWLEDGE
    assert decision.may_clarify is False

    s = state(user_input=query, needs_clarification=True,
              clarification_question="What type of agent do you want to build?")
    assert await route_after_init(s) != "clarification"
    assert s["needs_clarification"] is False
    assert "answered by retrieval" in s["clarification_reason"] or s["clarification_reason"]


def test_a_possessive_keeps_a_how_to_question_personal():
    """The counterpart: "how do I improve my CGPA" is about the user."""
    assert category_of("How do I improve my CGPA?") is QueryCategory.PROFILE_EDUCATION


@pytest.mark.parametrize(
    "query,question",
    [
        ("Send this to him.", "Who should I send it to?"),
        ("Schedule a meeting.", "What date and time?"),
        ("Apply for it.", "Which listing?"),
    ],
)
async def test_an_action_missing_a_parameter_still_clarifies(query, question):
    """19. A recipient is in no store. Asking is the only correct response."""
    decision = query_intent.classify(query, has_context=True)
    assert decision.category is QueryCategory.AMBIGUOUS_ACTION
    assert decision.may_clarify is True

    s = state(user_input=query, session_id="conv-action",
              needs_clarification=True, clarification_question=question)
    assert await route_after_init(s) == "clarification"
    assert s["clarification_reason"] == "action is missing a parameter no store holds"


async def test_only_one_clarification_per_conversation():
    """
    20. The budget. A second question about the same request means the first
    one did not unblock anything, and a third certainly will not.
    """
    first = state(user_input="Send this to him.", session_id="conv-1",
                  needs_clarification=True, clarification_question="Who?")
    assert await route_after_init(first) == "clarification"

    second = state(user_input="Schedule a meeting.", session_id="conv-1",
                   needs_clarification=True, clarification_question="When?")
    assert await route_after_init(second) == "profile"
    assert "budget spent" in second["clarification_reason"]


async def test_the_budget_is_per_conversation():
    """One conversation's spent budget must not silence another's."""
    spent = state(user_input="Send this to him.", session_id="conv-a",
                  needs_clarification=True, clarification_question="Who?")
    assert await route_after_init(spent) == "clarification"

    other = state(user_input="Send this to him.", session_id="conv-b",
                  needs_clarification=True, clarification_question="Who?")
    assert await route_after_init(other) == "clarification"


async def test_asking_for_fewer_questions_disables_clarification():
    """
    The user said "don't ask too many questions" and was asked again. Once
    stated, the preference holds for the conversation — it is a preference, not
    a comment on one turn.
    """
    assert query_intent.asks_for_fewer_questions("Don't ask too many questions.")
    assert query_intent.asks_for_fewer_questions("stop asking questions")
    assert query_intent.asks_for_fewer_questions("just answer me")
    assert not query_intent.asks_for_fewer_questions("what is my CGPA")

    opt_out = state(user_input="Don't ask too many questions.", session_id="conv-2")
    await route_after_init(opt_out)

    follow = state(user_input="Send this to him.", session_id="conv-2",
                   needs_clarification=True, clarification_question="Who?")
    assert await route_after_init(follow) == "profile"
    assert "asked not to be questioned" in follow["clarification_reason"]


async def test_clarification_is_not_disabled_wholesale():
    """The policy narrows clarification; it does not remove it."""
    s = state(user_input="Book it for Tuesday or Wednesday, whichever works",
              session_id="conv-3", needs_clarification=True,
              clarification_question="Which day?")
    assert await route_after_init(s) == "clarification"


# ═════════════════════════════════════════════════════════════════════════
# 21–22. Write policy — what earns a place in long-term memory
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "utterance",
    [
        "Okay.",
        "Thanks",
        "You are fool.",
        "Once in a million chat.",
        "How are you?",
        "hmm",
        "never mind",
        "What is my CGPA?",
        "Tell me about my projects.",
    ],
)
def test_casual_conversation_is_not_stored(utterance):
    """
    21. Every one of these became a long-term memory, ranked alongside the
    user's degree, and competed with it at retrieval time.
    """
    assert write_policy.should_store(utterance, role="user").store is False


@pytest.mark.parametrize(
    "utterance",
    [
        "Remember that I prefer concise answers",
        "Don't forget my exam is on the 12th",
        "Keep in mind that I work best in the mornings",
    ],
)
def test_an_explicit_request_is_always_stored(utterance):
    """22. The user has taken responsibility for the judgement."""
    decision = write_policy.should_store(utterance, role="user")
    assert decision.store is True
    assert decision.explicit is True
    assert decision.importance >= 0.9


@pytest.mark.parametrize(
    "utterance",
    [
        "My CGPA is 8.9",
        "I am a final year student at RGIPT",
        "I interned at a startup last summer",
        "I prefer short answers over long ones",
    ],
)
def test_durable_personal_facts_are_stored(utterance):
    decision = write_policy.should_store(utterance, role="user")
    assert decision.store is True
    assert decision.importance > 0


@pytest.mark.parametrize(
    "utterance",
    [
        "What name did I ask you to remember?",
        "Do you remember my birthday?",
        "What have you saved about me?",
    ],
)
def test_asking_what_was_remembered_does_not_create_a_memory(utterance):
    """
    These carry the write marker and mean the opposite of it. Storing them
    files the question as though it were its own answer — which is how a store
    fills up with its own queries.
    """
    decision = write_policy.should_store(utterance, role="user")
    assert decision.store is False
    assert "remembered" in decision.reason


def test_a_suppression_reason_names_the_actual_mechanism():
    """
    A log line saying "answered by retrieval" against a date question would be
    misleading, and these reasons are what an operator reads when a suppression
    looks wrong.
    """
    date_verdict = clarification_policy.evaluate("c", QueryCategory.TEMPORAL_CURRENT)
    assert "clock" in date_verdict.reason

    general_verdict = clarification_policy.evaluate("c", QueryCategory.GENERAL_KNOWLEDGE)
    assert "stated assumption" in general_verdict.reason

    profile_verdict = clarification_policy.evaluate("c", QueryCategory.PROFILE_EDUCATION)
    assert "retrieval" in profile_verdict.reason


def test_the_assistants_own_output_is_never_evidence_about_the_user():
    """
    Storing the reply closes a loop: the model's guess is recalled later as
    though the user had said it, and becomes the evidence for the next guess.
    """
    decision = write_policy.should_store(
        "You studied at RGIPT and your CGPA is 8.9.", role="assistant"
    )
    assert decision.store is False
    assert "assistant" in decision.reason


def test_filler_exchanges_are_not_queued_for_extraction():
    """A batch of "okay"/"thanks" costs an LLM call to be told there is nothing."""
    assert write_policy.should_store_turn("Okay.", "Sure thing!").store is False
    assert write_policy.should_store_turn("", "Hello!").store is False
    assert write_policy.should_store_turn(
        "I've been working on a voice assistant for the last three months",
        "That sounds interesting.",
    ).store is True


# ═════════════════════════════════════════════════════════════════════════
# 23. Conflict resolution is deterministic
# ═════════════════════════════════════════════════════════════════════════

def _claim(value, source, *, explicit=False, key="college", confidence=1.0, at=None):
    return identity.FactClaim(
        key=key, value=value, source=source,
        explicit=explicit, confidence=confidence, timestamp=at,
    )


def test_a_passing_remark_does_not_overwrite_the_resume():
    """
    23. The résumé says RGIPT; the user typed "I study at X" once in passing.

    The asymmetry is deliberate: wrongly keeping a stale fact costs a
    correction, wrongly destroying one is unrecoverable.
    """
    verdict = identity.resolve_conflict(
        _claim("RGIPT", MemorySource.RESUME_DOCUMENT),
        _claim("Some Other College", MemorySource.CONVERSATION_CURRENT),
    )
    assert verdict.resolution is identity.Resolution.KEEP_EXISTING
    assert verdict.winner == "RGIPT"
    assert verdict.previous == "Some Other College"


def test_an_explicit_correction_supersedes_an_older_source():
    """A newer explicit statement wins — and the old value is retained."""
    verdict = identity.resolve_conflict(
        _claim("RGIPT", MemorySource.RESUME_DOCUMENT),
        _claim("IIT Kanpur", MemorySource.CONVERSATION_CURRENT, explicit=True),
    )
    assert verdict.resolution is identity.Resolution.SUPERSEDE
    assert verdict.winner == "IIT Kanpur"
    assert verdict.previous == "RGIPT"


def test_canonical_identity_survives_an_incidental_mention():
    verdict = identity.resolve_conflict(
        _claim("Vansh Pratap Singh", MemorySource.CANONICAL_IDENTITY,
               key=identity.CANONICAL_NAME_KEY, explicit=True),
        _claim("Devasi", MemorySource.CONVERSATION_CURRENT,
               key=identity.CANONICAL_NAME_KEY),
    )
    assert verdict.resolution is identity.Resolution.RECORD_ALONGSIDE
    assert verdict.winner == "Vansh Pratap Singh"


def test_agreeing_sources_are_not_a_conflict():
    verdict = identity.resolve_conflict(
        _claim("RGIPT", MemorySource.RESUME_DOCUMENT),
        _claim("rgipt", MemorySource.CONVERSATION_CURRENT),
    )
    assert verdict.resolution is identity.Resolution.KEEP_EXISTING
    assert verdict.reason == "values agree"


def test_nothing_in_the_resolver_deletes():
    """Every outcome records the losing value, so a wrong call is recoverable."""
    for existing, incoming in (
        (_claim("A", MemorySource.RESUME_DOCUMENT),
         _claim("B", MemorySource.CONVERSATION_CURRENT)),
        (_claim("A", MemorySource.CONVERSATION_CURRENT),
         _claim("B", MemorySource.RESUME_DOCUMENT)),
    ):
        verdict = identity.resolve_conflict(existing, incoming)
        assert verdict.previous is not None


# ═════════════════════════════════════════════════════════════════════════
# 24–26. Answerability: an empty store is not a broken one
# ═════════════════════════════════════════════════════════════════════════

def _evidence(source=MemorySource.RESUME_DOCUMENT, content="B.Tech, CPI 8.9"):
    return RetrievedMemory(
        content=content, source=source, memory_type="document",
        provenance="resume:education", owner_id="vansh",
    )


def test_a_failed_lookup_is_not_an_absence_of_data():
    """
    24. Qdrant unavailable.

    Reporting "you have no CGPA on file" after a timeout asserts something
    false about the user's own data — and it is precisely the state in which a
    model, told there is nothing, invents something.
    """
    verdict = assess(
        [],
        category=QueryCategory.PROFILE_EDUCATION,
        expected_sources=[MemorySource.RESUME_DOCUMENT],
        errored_sources=[MemorySource.RESUME_DOCUMENT],
    )
    assert verdict.state is Answerability.TOOL_ERROR
    assert verdict.should_answer is False
    assert verdict.may_clarify is False
    guidance = verdict.guidance().lower()
    assert "do not say the information is missing" in guidance
    assert "could not look it up" in guidance


def test_an_empty_store_is_stated_plainly():
    """25. Retrieval worked; there is genuinely nothing. That is a fact."""
    verdict = assess(
        [],
        category=QueryCategory.PROFILE_ACHIEVEMENTS,
        expected_sources=[MemorySource.RESUME_DOCUMENT],
    )
    assert verdict.state is Answerability.NO_DATA
    assert verdict.may_clarify is False
    assert "don't have that information" in verdict.guidance()


def test_no_data_and_tool_error_are_distinguishable():
    """The distinction the whole module exists to preserve."""
    empty = assess([], expected_sources=[MemorySource.RESUME_DOCUMENT])
    broken = assess([], expected_sources=[MemorySource.RESUME_DOCUMENT],
                    errored_sources=[MemorySource.RESUME_DOCUMENT])
    assert empty.state is not broken.state


def test_partial_evidence_is_answered_not_withheld():
    """26. Say what is known; name only what is missing."""
    verdict = assess(
        [_evidence()],
        category=QueryCategory.PROFILE_EDUCATION,
        expected_sources=[MemorySource.RESUME_DOCUMENT, MemorySource.PROFILE_MEMORY],
        errored_sources=[MemorySource.PROFILE_MEMORY],
    )
    assert verdict.state is Answerability.PARTIALLY_ANSWERABLE
    assert verdict.should_answer is True
    assert verdict.may_clarify is False
    assert "do not withhold" in verdict.guidance().lower()


def test_evidence_from_the_primary_source_is_fully_answerable():
    verdict = assess(
        [_evidence()],
        category=QueryCategory.PROFILE_EDUCATION,
        expected_sources=[MemorySource.RESUME_DOCUMENT],
    )
    assert verdict.state is Answerability.ANSWERABLE
    assert verdict.sources_hit == [MemorySource.RESUME_DOCUMENT]


def test_a_secondary_source_answer_is_flagged_as_partial():
    verdict = assess(
        [_evidence(source=MemorySource.SEMANTIC_MEMORY)],
        category=QueryCategory.PROFILE_EDUCATION,
        expected_sources=[MemorySource.RESUME_DOCUMENT, MemorySource.SEMANTIC_MEMORY],
    )
    assert verdict.state is Answerability.PARTIALLY_ANSWERABLE
    assert verdict.missing == [MemorySource.RESUME_DOCUMENT.value]


def test_only_ambiguity_and_missing_parameters_permit_a_question():
    """Low confidence and an empty store are not on the list."""
    assert assess([], expected_sources=[]).may_clarify is False
    assert assess([_evidence()]).may_clarify is False
    assert assess([_evidence(), _evidence(content="other")],
                  ambiguous_candidates=2).may_clarify is True
    assert assess([], missing_parameters=["recipient"]).may_clarify is True


def test_retrieval_results_carry_their_status_into_the_assessment():
    """The store-level NO_DATA/ERROR distinction survives the adaptation."""
    ok_evidence, ok_errors = from_retrieval_result(
        RetrievalResult.ok([{"content": "B.Tech CSE, CPI 8.9", "section": "education"}]),
        MemorySource.RESUME_DOCUMENT,
    )
    assert len(ok_evidence) == 1
    assert ok_errors == []
    assert ok_evidence[0].provenance == "resume_document:education"
    assert assess(ok_evidence).state is Answerability.ANSWERABLE

    empty_evidence, empty_errors = from_retrieval_result(
        RetrievalResult.no_data(), MemorySource.RESUME_DOCUMENT
    )
    assert (empty_evidence, empty_errors) == ([], [])
    assert assess(empty_evidence).state is Answerability.NO_DATA

    err_evidence, err_errors = from_retrieval_result(
        RetrievalResult.error(), MemorySource.RESUME_DOCUMENT
    )
    assert err_errors == [MemorySource.RESUME_DOCUMENT]
    assert assess(err_evidence, errored_sources=err_errors).state is Answerability.TOOL_ERROR


def test_a_fallback_result_is_marked_derived():
    """Provenance survives the fallback path — it is not passed off as primary."""
    evidence, _ = from_retrieval_result(
        RetrievalResult.fallback([{"content": "Python, FastAPI"}]),
        MemorySource.RESUME_DOCUMENT,
    )
    assert evidence[0].derived is True
    assert evidence[0].trust < _evidence().trust


# ═════════════════════════════════════════════════════════════════════════
# 27–29. Routing and source precedence
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "query",
    [
        "What is my CPI?",
        "What college am I in?",
        "Where did I intern?",
        "What are my projects?",
    ],
)
async def test_personal_questions_are_not_routed_as_research(query):
    """
    27. A personal fact is a retrieval from the user's own memory, and it went
    to a research path that could not possibly hold it.
    """
    decision = query_intent.classify(query)
    assert decision.sources[0] is MemorySource.RESUME_DOCUMENT
    assert MemorySource.GENERAL_KNOWLEDGE not in decision.sources
    # Even when the planner picked the wrong specialist.
    assert await route_after_init(state(user_input=query, selected_agent="job")) == "profile"


def test_resume_precedence_for_document_questions():
    """28. "What's in my resume" is a document question, résumé first."""
    decision = query_intent.classify("What does my resume say?")
    assert decision.category in (
        QueryCategory.DOCUMENT_RESUME, QueryCategory.PROFILE_GENERAL
    )
    assert MemorySource.RESUME_DOCUMENT in decision.sources


def test_conversation_precedence_for_current_thread_questions():
    """29. "What did I just tell you" reads the thread, not the résumé."""
    decision = query_intent.classify("What did I just tell you?")
    assert decision.sources == (MemorySource.CONVERSATION_CURRENT,)
    assert MemorySource.RESUME_DOCUMENT not in decision.sources


def test_general_knowledge_consults_no_memory_source():
    decision = query_intent.classify("Explain how gradient descent works")
    assert decision.sources == (MemorySource.GENERAL_KNOWLEDGE,)
    assert decision.requires_retrieval is False


# ═════════════════════════════════════════════════════════════════════════
# 30. Voice and text share one decision path
# ═════════════════════════════════════════════════════════════════════════

async def test_voice_and_text_reach_the_same_routing_decision():
    """
    30. The spoken path read `needs_clarification` straight off the planner, so
    every policy the graph applied was silently absent from voice turns.
    """
    from app.agents import streaming_workflow

    assert streaming_workflow.decide_route is decide_route

    query = "What is my CPI?"
    text_state = state(user_input=query, session_id="shared-conv",
                       needs_clarification=True, clarification_question="Which?")
    voice_state = state(user_input=query, session_id="shared-conv",
                        needs_clarification=True, clarification_question="Which?")

    assert await decide_route(text_state) == "profile"
    assert await decide_route(voice_state) == "profile"
    assert text_state["query_category"] == voice_state["query_category"]
    assert text_state["memory_sources"] == voice_state["memory_sources"]


async def test_the_clarification_budget_is_shared_across_modalities():
    """One conversation, one budget — whichever transport spends it."""
    spoken = state(user_input="Send this to him.", session_id="shared-conv",
                   needs_clarification=True, clarification_question="Who?")
    assert await decide_route(spoken) == "clarification"

    typed = state(user_input="Apply for it.", session_id="shared-conv",
                  needs_clarification=True, clarification_question="Which?")
    assert await decide_route(typed) == "profile"


# ═════════════════════════════════════════════════════════════════════════
# Context assembly: each question sees only the sources that answer it
# ═════════════════════════════════════════════════════════════════════════

FULL_CONTEXT = {
    "profile_facts": [{"key": "preferred_tone", "value": "concise"}],
    "episodes": [{"agent_used": "profile", "user_summary": "asked about CPI",
                  "agent_summary": "reported 8.9"}],
    "chat_history": [{"role": "user", "content": "hello"}],
    "preferences": [{"memory": "likes backend work"}],
    "long_term": {
        "skills": [{"content": "Python"}],
        "projects": [{"content": "TRACE — resume analysis"}],
        "resume": {"content": "Vansh Pratap Singh, B.Tech"},
    },
}


def test_an_education_question_does_not_receive_the_project_list():
    prompt = memory_manager.format_context_for_prompt(
        FULL_CONTEXT, category=QueryCategory.PROFILE_EDUCATION.value
    )
    assert "User Resume" in prompt
    assert "User Profile Facts" in prompt
    assert "User Projects" not in prompt
    assert "User Skills" not in prompt


def test_a_date_question_receives_almost_nothing():
    prompt = memory_manager.format_context_for_prompt(
        FULL_CONTEXT, category=QueryCategory.TEMPORAL_CURRENT.value
    )
    assert "User Resume" not in prompt
    assert "User Skills" not in prompt
    assert "Recent Activity" not in prompt


def test_a_general_question_keeps_preferences_and_drops_the_resume():
    """Tone and format preferences shape the answer; the résumé does not."""
    prompt = memory_manager.format_context_for_prompt(
        FULL_CONTEXT, category=QueryCategory.GENERAL_KNOWLEDGE.value
    )
    assert "User Profile Facts" in prompt
    assert "User Preferences & Interests" in prompt
    assert "User Resume" not in prompt


def test_omitting_the_category_renders_everything_as_before():
    """The default is unchanged, so no existing caller is narrowed silently."""
    prompt = memory_manager.format_context_for_prompt(FULL_CONTEXT)
    for section in ("User Profile Facts", "Recent Activity", "User Skills",
                    "User Projects", "User Resume"):
        assert section in prompt


def test_an_unknown_category_is_under_filtered_never_blind():
    """A category added without a section map must degrade to showing more."""
    prompt = memory_manager.format_context_for_prompt(
        FULL_CONTEXT, category="SOME_FUTURE_CATEGORY"
    )
    assert "User Resume" in prompt
    assert memory_manager.sections_for("SOME_FUTURE_CATEGORY") == memory_manager._ALL_SECTIONS


# ═════════════════════════════════════════════════════════════════════════
# Observability: every memory decision is loggable without leaking content
# ═════════════════════════════════════════════════════════════════════════

def test_a_routing_decision_logs_its_category_and_sources_not_the_query():
    summary = query_intent.classify("What is my CPI?").summary()
    assert summary["intent"] == QueryCategory.PROFILE_EDUCATION.value
    assert summary["sources"][0] == MemorySource.RESUME_DOCUMENT.value
    assert summary["deterministic"] is True
    assert "CPI" not in str(summary)


def test_an_assessment_logs_counts_and_provenance_not_content():
    summary = assess([_evidence(content="CPI 8.9 at RGIPT")]).summary()
    assert summary["answerability"] == Answerability.ANSWERABLE.value
    assert summary["results"] == 1
    assert "8.9" not in str(summary)


def test_a_write_decision_logs_its_reason_not_the_text():
    summary = write_policy.should_store("My CGPA is 8.9").summary()
    assert summary["store"] is True
    assert "8.9" not in str(summary)


def test_a_retrieved_memory_carries_full_provenance():
    """Source, type, confidence, owner, visibility, explicitness, derivation."""
    item = RetrievedMemory(
        content="B.Tech CSE, CPI 8.9",
        source=MemorySource.RESUME_DOCUMENT,
        memory_type="document",
        confidence=0.9,
        provenance="resume:education",
        owner_id="vansh",
        visibility="private",
        explicit=False,
        canonical_identity=False,
        derived=False,
    )
    assert item.source is MemorySource.RESUME_DOCUMENT
    assert item.owner_id == "vansh"
    assert item.visibility == "private"
    assert item.provenance == "resume:education"
    assert "resume_document/document" in item.describe()
    assert "8.9" not in item.describe()


def test_explicit_memories_outrank_derived_ones_from_the_same_source():
    explicit = RetrievedMemory(content="x", source=MemorySource.PROFILE_MEMORY,
                               memory_type="identity", explicit=True)
    derived = RetrievedMemory(content="x", source=MemorySource.PROFILE_MEMORY,
                              memory_type="identity", derived=True)
    assert explicit.trust > derived.trust
