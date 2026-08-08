"""
Answer-first routing: retrieval is attempted before any clarification.

Three separate defects produced the same symptom — the assistant interrogating
the user about information it already held:

1. **A missing capability.** Internships live in an `experience` chunk and
   CGPA/college in an `education` chunk, and the profile agent had no tool for
   either. The only route to them was the first 1500 characters of get_resume.
   "I don't have information about the company where you interned" was an
   honest answer from an agent with no way to look.

2. **Clarification ahead of retrieval.** The planner scores intent from query
   text alone, with memory fetched concurrently, so it cannot know an answer
   exists. Asked "What is my name?" it asked which name was meant.

3. **Follow-ups read as new questions.** "I think it was mentioned in my resume"
   carries no subject, so it was treated as a fresh ambiguous query instead of
   an instruction to search again.

The rule these tests enforce: a retrieval failure is not user ambiguity.
"Where did I intern?" is perfectly clear; if the store is empty the answer is
"I couldn't find it", never "which internship do you mean?".
"""
import pytest

from app.agents import profile_intent
from app.agents.profile_intent import (
    CONVERSATION_FOLLOWUP,
    PROFILE_ACHIEVEMENTS,
    PROFILE_CGPA,
    PROFILE_COLLEGE,
    PROFILE_EDUCATION,
    PROFILE_EXPERIENCE,
    PROFILE_GENERAL,
    PROFILE_INTERNSHIP,
    PROFILE_NAME,
    PROFILE_PROJECTS,
    PROFILE_SKILLS,
    classify,
)
from app.agents.workflow import route_after_init


def state(**overrides):
    base = {
        "user_input": "",
        "memory_context": {},
        "memory_prompt": "",
        "selected_agent": "profile",
        "needs_clarification": False,
        "planner_confidence": 0.9,
    }
    base.update(overrides)
    return base


def with_history(**overrides):
    return state(
        memory_context={"chat_history": [
            {"role": "user", "content": "What company did I intern at?"},
            {"role": "assistant", "content": "I couldn't find the internship company."},
        ]},
        **overrides,
    )


# ─────────────────────────────────────────────────────────────────────────
# Intent classification — by shape, not by enumerated phrasing
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is my name?", PROFILE_NAME),
        ("what's my name", PROFILE_NAME),
        ("Which college am I currently studying at?", PROFILE_COLLEGE),
        ("What is my college?", PROFILE_COLLEGE),
        ("which university do I attend", PROFILE_COLLEGE),
        ("What is my CGPA?", PROFILE_CGPA),
        ("what are my grades", PROFILE_CGPA),
        ("What is my branch?", PROFILE_EDUCATION),
        ("What degree am I pursuing?", PROFILE_EDUCATION),
        ("Tell me about my education.", PROFILE_EDUCATION),
        ("What skills do I have?", PROFILE_SKILLS),
        ("What technologies do I know?", PROFILE_SKILLS),
        ("what frameworks am I proficient in", PROFILE_SKILLS),
        ("What projects have I built?", PROFILE_PROJECTS),
        ("Tell me about my projects.", PROFILE_PROJECTS),
        ("Where did I intern?", PROFILE_INTERNSHIP),
        ("What company did I intern at?", PROFILE_INTERNSHIP),
        ("What company did I do my internship at?", PROFILE_INTERNSHIP),
        ("Tell me about my internship.", PROFILE_INTERNSHIP),
        ("Tell me about my experience.", PROFILE_EXPERIENCE),
        ("What companies have I worked with?", PROFILE_EXPERIENCE),
        ("What are my achievements?", PROFILE_ACHIEVEMENTS),
        ("what awards have I won", PROFILE_ACHIEVEMENTS),
        ("What do you know about me?", PROFILE_GENERAL),
    ],
)
def test_personal_questions_classify_without_being_enumerated(query, expected):
    assert classify(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "Explain gradient descent.",
        "Search for backend internships in Bangalore.",
        "What is the capital of France?",
        "Schedule a meeting.",
        "Send this to him.",
    ],
)
def test_impersonal_queries_are_not_claimed_by_the_guard(query):
    assert classify(query) is None


def test_every_intent_maps_to_a_retrieval_tool():
    for intent in (PROFILE_NAME, PROFILE_EDUCATION, PROFILE_CGPA, PROFILE_COLLEGE,
                   PROFILE_SKILLS, PROFILE_PROJECTS, PROFILE_EXPERIENCE,
                   PROFILE_INTERNSHIP, PROFILE_ACHIEVEMENTS, PROFILE_GENERAL):
        assert profile_intent.tools_for(intent), f"{intent} has no tool"


def test_internship_and_education_questions_reach_the_new_tools():
    assert profile_intent.tools_for(PROFILE_INTERNSHIP) == ["get_experience"]
    assert profile_intent.tools_for(PROFILE_CGPA) == ["get_education"]
    assert profile_intent.tools_for(PROFILE_COLLEGE) == ["get_education"]


# ─────────────────────────────────────────────────────────────────────────
# Follow-ups inherit their subject from the conversation
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query",
    [
        "I think it was mentioned in my resume.",
        "it should be in my resume",
        "check my profile",
        "you should have that stored",
        "Tell me more about TRACE.",
        "what about that one",
    ],
)
def test_followups_resolve_against_context_rather_than_asking(query):
    assert classify(query, has_context=True) is not None


def test_a_resume_hint_with_a_subject_goes_straight_to_that_section():
    """"It was in my resume, under internships" — subject wins over the hint."""
    assert classify("I think my internship was mentioned in my resume",
                    has_context=True) == PROFILE_INTERNSHIP


def test_a_bare_followup_without_history_is_not_forced_into_retrieval():
    """With no prior turn there is nothing to search "again" for."""
    assert classify("what about that one", has_context=False) is None


# ─────────────────────────────────────────────────────────────────────────
# Routing — the answer-first guard
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query",
    [
        "What is my name?",
        "What is my college?",
        "What is my CGPA?",
        "What are my skills?",
        "What projects have I built?",
        "Where did I intern?",
        "What company did I intern at?",
        "Tell me about my experience.",
        "Tell me about my internship.",
    ],
)
def test_acceptance_queries_never_reach_clarification(query):
    """The acceptance list: every one must retrieve, even if the planner balked."""
    assert route_after_init(state(
        user_input=query, needs_clarification=True,
        clarification_question="Could you provide more context?",
    )) == "profile"


def test_retrieval_is_attempted_even_when_memory_is_empty():
    """
    An empty store is not a reason to interrogate the user about their own name.
    Retrieval runs; if it finds nothing the agent reports that honestly.
    """
    assert route_after_init(state(
        user_input="What is my name?",
        needs_clarification=True,
        memory_context={},
        memory_prompt="",
    )) == "profile"


def test_the_resume_followup_routes_to_retrieval_not_clarification():
    assert route_after_init(with_history(
        user_input="I think it was mentioned in my resume.",
        needs_clarification=True,
        clarification_question="Which resume detail do you mean?",
    )) == "profile"


def test_a_project_followup_routes_to_retrieval():
    assert route_after_init(with_history(
        user_input="Tell me more about TRACE.",
        needs_clarification=True,
    )) == "profile"


def test_the_suppressed_question_is_cleared_from_state():
    s = state(user_input="What is my CGPA?", needs_clarification=True,
              clarification_question="Which CGPA?")
    route_after_init(s)
    assert s["needs_clarification"] is False
    assert s["clarification_question"] == ""
    assert s["profile_intent"] == PROFILE_CGPA


def test_academic_questions_keep_their_agent():
    """Timetable and attendance belong to academic, not profile retrieval."""
    assert route_after_init(state(
        user_input="What classes do I have tomorrow?",
        selected_agent="academic", needs_clarification=True,
    )) == "academic"


def test_a_personal_question_overrides_a_mistaken_agent_choice():
    assert route_after_init(state(
        user_input="What company did I intern at?", selected_agent="job",
    )) == "profile"


# ─────────────────────────────────────────────────────────────────────────
# Clarification survives where it is genuinely required
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query,question",
    [
        ("Send this to him.", "Who should I send it to?"),
        ("Schedule a meeting.", "What date and time?"),
        ("Apply to that one.", "Which listing?"),
    ],
)
def test_operational_requests_missing_a_parameter_still_clarify(query, question):
    """A recipient or a date is not in any store — asking is the only option."""
    assert route_after_init(state(
        user_input=query, needs_clarification=True, clarification_question=question,
    )) == "clarification"


@pytest.mark.parametrize(
    "query,question",
    [
        ("Send this to him.", "Who should I send it to?"),
        ("Schedule a meeting for that.", "What date and time?"),
        ("Apply to that one.", "Which listing?"),
        ("Email it to her.", "Who is the recipient?"),
    ],
)
def test_an_action_with_a_referential_word_still_clarifies_mid_conversation(query, question):
    """
    The dangerous overlap: "send this to him" carries a referential word and has
    prior turns, so a naive follow-up rule claims it. But its missing recipient
    is not in any store — retrieval cannot help and asking is correct.
    """
    assert route_after_init(with_history(
        user_input=query, needs_clarification=True, clarification_question=question,
    )) == "clarification"


def test_an_informational_followup_is_still_retrieval_mid_conversation():
    """The counterpart: "tell me" is informational, so it must NOT clarify."""
    assert route_after_init(with_history(
        user_input="Tell me more about that project.", needs_clarification=True,
    )) == "profile"


def test_clarification_is_not_disabled_wholesale():
    assert route_after_init(state(
        user_input="Book it for Tuesday or Wednesday, whichever works",
        needs_clarification=True, clarification_question="Which day?",
    )) == "clarification"


def test_errors_still_short_circuit_to_the_response_agent():
    assert route_after_init(state(error="boom", needs_clarification=True)) == "response"


def test_ordinary_routing_is_untouched_when_nothing_needs_clarifying():
    assert route_after_init(state(user_input="find me jobs", selected_agent="job")) == "job"
