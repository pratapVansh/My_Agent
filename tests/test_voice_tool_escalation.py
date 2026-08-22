"""
The spoken path cannot answer a job-match question without the matcher.

Phase 2 made the *text* path safe: a JOB_MATCH turn routes to the job agent,
`match_job` runs, and the answer is rendered from the evidence table rather than
written by a model. The spoken path had a hole, and the hole was architectural
rather than lexical:

    livekit_worker -> hybrid_router.determine_route  (keywords, then an LLM
                                                      guess with a 1.2s timeout
                                                      defaulting to streaming)
        ROUTE_TOOL           -> run_workflow            tools
        ROUTE_CONVERSATIONAL -> run_streaming_workflow  NO tools

`run_streaming_workflow` then called `decide_route` — the same deterministic
classifier the graph uses — worked out that the turn was JOB_MATCH, and streamed
a tool-free reply anyway, because it had no way to escalate. So the guarantee
rested on a regular expression matching the exact words the user happened to say.

Two doors led there, which is why a fix in the router alone would not have been
enough: `/agents/stream` calls `run_streaming_workflow` directly and never
consults `hybrid_router` at all.

What is asserted below is the *architecture*, deliberately not a phrase list:

  * the categories that cannot be answered without tools are named once, in
    `query_intent`, and both routers ask that one module
  * `run_streaming_workflow` escalates on its own, so a turn arriving through
    any door — a wrong LLM guess, a router timeout, the raw WebSocket — still
    reaches the tools
  * the answer that comes back is the rendered `MatchReport`, and model prose
    cannot replace it

The phrasing tests exist as well, but they are recall checks on the classifier,
not the safety property. The safety property is that whatever `query_intent`
calls JOB_MATCH is unanswerable without `match_job` — and it is tested by
forcing the classifier's verdict rather than by hoping a sentence matches.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.agents import query_intent
from app.agents.hybrid_router import (
    ROUTE_CONVERSATIONAL,
    ROUTE_TOOL,
    classify_heuristically,
    determine_route,
)
from app.memory.sources import QueryCategory
from tests.support.fake_llm import final, tool_call
from tests.support.harness import OWNER, drive, state, stub_services
from tests.test_candidate_profile import FakeMemory

JOB_CONTEXT = [
    {"role": "user", "content": "find me AI engineer jobs"},
    {"role": "assistant", "content":
        "Here are 3 openings. 1. AI Engineer at Acme - requirements: Python, "
        "FastAPI, Kubernetes. 2. ML Engineer at Beta. 3. Backend Engineer."},
]

JD = """Senior AI Engineer

Requirements:
- Strong Python and FastAPI
- Experience with PostgreSQL
- Kubernetes in production

Nice to have:
- ReactJS
"""


async def _collect(agen) -> List[Dict[str, Any]]:
    return [event async for event in agen]


# ═══════════════════════════════════════════════════════════════════════════
# 1. The architecture: one place names the tool-requiring categories
# ═══════════════════════════════════════════════════════════════════════════

def test_tool_requiring_categories_are_declared_in_one_place():
    """
    The single source of truth both routers consult. A second list somewhere
    else would be free to drift, and the drift would be a spoken question
    answered without the tool its correctness depends on.

    Pinned exactly rather than by membership: this set is the whole security
    boundary, and a category silently added or dropped is a spoken answer that
    changes source without anyone deciding it should.
    """
    assert query_intent.TOOL_REQUIRED_CATEGORIES == {
        QueryCategory.JOB_MATCH,
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
    }
    assert query_intent.requires_tools(QueryCategory.JOB_MATCH) is True
    assert query_intent.requires_tools("JOB_MATCH") is True


def test_the_two_rules_are_disjoint():
    """
    A category is either unanswerable without tools or answerable when
    grounded. Being in both would make `escalation_reason` report whichever
    check ran first, and the reason is what a trace uses to tell a missing
    capability from a slow database.
    """
    assert not (
        query_intent.TOOL_REQUIRED_CATEGORIES
        & query_intent.GROUNDING_REQUIRED_CATEGORIES
    )


@pytest.mark.parametrize("category,why", [
    (QueryCategory.PROFILE_IDENTITY,
     "canonical name has a precedence only get_identity enforces"),
    (QueryCategory.PROFILE_EDUCATION,
     "CGPA lives in the typed education section, rendered into no prompt"),
    (QueryCategory.ACADEMIC_CURRENT,
     "current standing reads the same unrendered section"),
    (QueryCategory.PROFILE_EXPERIENCE,
     "internships live in the typed experience section"),
    (QueryCategory.PROFILE_ACHIEVEMENTS,
     "achievements have no prompt section at all"),
    (QueryCategory.DOCUMENT_RESUME,
     "the prompt holds 300 chars of a document the question names as authority"),
    (QueryCategory.EXPLICIT_MEMORY,
     "remembered-value keys are deliberately withheld from prompts"),
    (QueryCategory.EXPLICIT_MEMORY_WRITE,
     "a streamed reply claims a write that never happened"),
    (QueryCategory.IDENTITY_UPDATE,
     "same, for the one write that must never be claimed falsely"),
    (QueryCategory.SCHEDULE_TEMPORAL,
     "the timetable is in Postgres behind the academic agent's tools"),
    (QueryCategory.JOB_MATCH,
     "the verdict is a rendered computation, not prose"),
    (QueryCategory.JOB_SEARCH,
     "the postings live on a board that no prompt carries"),
])
def test_each_tool_required_category_names_a_source_no_prompt_reaches(category, why):
    """
    The membership test, stated as the reason rather than the list. Each of
    these has an answer that exists in the store and cannot arrive in the
    tool-free prompt — which is what separates "a tool would help" from
    "answering this without one produces a false statement".
    """
    assert query_intent.requires_tools(category) is True, why
    assert query_intent.escalation_reason(category) == (
        query_intent.ESCALATION_TOOL_REQUIRED
    ), why


@pytest.mark.parametrize("category", [
    QueryCategory.SMALL_TALK,
    QueryCategory.GENERAL_KNOWLEDGE,
    QueryCategory.PROFILE_SKILLS,
    QueryCategory.PROFILE_PROJECTS,
    QueryCategory.CONVERSATION_CURRENT,
    QueryCategory.TEMPORAL_CURRENT,
    QueryCategory.CONFIRM_ACTION,
    QueryCategory.PROVENANCE_QUERY,
])
def test_ordinary_categories_are_not_forced_through_the_tool_path(category):
    """
    The unconditional escalation stays narrow. Skills and projects *are*
    rendered into the memory prompt, so a grounded turn quotes the store;
    small talk and general knowledge need no store; the clock, provenance and
    confirmation are answered by deterministic routes that never reach a model
    at all. Forcing any of these through the graph would cost every spoken turn
    its latency for no correctness gain.
    """
    assert query_intent.requires_tools(category) is False


@pytest.mark.parametrize("category", sorted(
    query_intent.MEMORY_GROUNDED_CATEGORIES, key=lambda c: c.value
))
def test_the_streamable_personal_categories_still_require_grounding(category):
    """
    The conditional rule. These may stream, but only from something: with no
    memory prompt there is no data, no retrieval-status hint and no refusal
    policy — every safeguard in the retrieval layer is expressed *inside* that
    prompt, so an empty one disarms all of them at once.
    """
    assert query_intent.requires_grounded_memory(category) is True
    assert query_intent.escalation_reason(category, memory_grounded=True) is None
    assert query_intent.escalation_reason(category, memory_grounded=False) == (
        query_intent.ESCALATION_UNGROUNDED
    )


@pytest.mark.parametrize("category", sorted(
    query_intent.CONVERSATION_GROUNDED_CATEGORIES, key=lambda c: c.value
))
def test_the_conversation_categories_are_grounded_by_the_transcript(category):
    """
    These read the thread, which the streaming path passes inline and which a
    memory outage does not take away. Gating them on the memory prompt would
    escalate every "tell me more" and "go ahead" spoken while Qdrant is slow —
    including the ones that are replies to a pending action.

    What they cannot survive is having no thread at all.
    """
    assert query_intent.requires_grounded_memory(category) is True
    assert query_intent.escalation_reason(category, memory_grounded=False) is None
    assert query_intent.escalation_reason(category, history_grounded=False) == (
        query_intent.ESCALATION_UNGROUNDED
    )


@pytest.mark.parametrize("category", [
    QueryCategory.SMALL_TALK,
    QueryCategory.GENERAL_KNOWLEDGE,
])
def test_impersonal_categories_stream_even_with_no_memory_at_all(category):
    """
    Greetings and world knowledge make no claim about the user, so an empty
    memory prompt is not a hazard for them — it is the normal case. Escalating
    them would turn every "hey" into a graph invocation.
    """
    assert query_intent.escalation_reason(category, memory_grounded=False) is None


@pytest.mark.parametrize("junk", [None, "", "NOT_A_CATEGORY", 0, [], {}])
def test_an_unreadable_category_never_claims_to_need_tools(junk):
    """Fails to the ordinary path rather than raising mid-turn."""
    assert query_intent.requires_tools(junk) is False
    assert query_intent.requires_grounded_memory(junk) is False
    assert query_intent.escalation_reason(
        junk, memory_grounded=False, history_grounded=False
    ) is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. The router asks the classifier, not a keyword list
# ═══════════════════════════════════════════════════════════════════════════

def test_the_router_defers_to_the_classifier_not_to_its_own_patterns():
    """
    The mechanism, isolated. With the classifier stubbed to call anything
    JOB_MATCH, an utterance carrying none of the router's keywords still routes
    to the tool path — which is only possible if the category, not the wording,
    is what decides.
    """
    import app.agents.hybrid_router as router

    original = router.query_intent.classify
    try:
        router.query_intent.classify = lambda text, **kw: original(
            "how well do I match this job", **kw
        )
        assert classify_heuristically("purple monkey dishwasher") == ROUTE_TOOL
    finally:
        router.query_intent.classify = original


def test_the_keyword_list_no_longer_carries_the_job_match_guarantee():
    """
    The old patterns were removed rather than left as a second implementation.
    A phrase the regex misses must still route correctly, which proves the
    regex is not what is doing the work.
    """
    from app.agents.hybrid_router import _TOOL_PATTERNS

    spoken = "how do I stack up against this job"
    assert not _TOOL_PATTERNS.search(spoken), (
        "the keyword list should no longer be the mechanism"
    )
    assert classify_heuristically(spoken) == ROUTE_TOOL


async def test_the_llm_fallback_is_never_reached_for_a_job_match(monkeypatch):
    """
    `determine_route` falls back to an LLM with a 1.2s timeout that defaults to
    streaming. A JOB_MATCH turn must be decided before that coin is flipped.
    """
    async def _explode(*args, **kwargs):
        raise AssertionError("the LLM router was consulted for a JOB_MATCH turn")

    monkeypatch.setattr(
        "app.agents.hybrid_router.groq_service.chat_completion", _explode
    )
    assert await determine_route("am I a good fit for this role") == ROUTE_TOOL


async def test_ordinary_conversation_still_streams(monkeypatch):
    """Regression cover: the escalation must not swallow normal voice turns."""
    for spoken in ("hey", "thanks", "what can you do", "how are you"):
        assert await determine_route(spoken) == ROUTE_CONVERSATIONAL


# ═══════════════════════════════════════════════════════════════════════════
# 3. The backstop: the streaming workflow escalates on its own
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_streaming_workflow_escalates_without_being_told(monkeypatch):
    """
    THE regression test for this change.

    The streaming workflow is entered directly — exactly as `/agents/stream`
    enters it, with no router involved and no `ROUTE_TOOL` decision anywhere.
    It must work out for itself that this turn cannot be answered here, and
    hand it to the tool workflow.
    """
    from app.agents import streaming_workflow

    called: Dict[str, Any] = {}

    async def _fake_workflow(**kwargs):
        called.update(kwargs)
        return {
            "display_text": "Moderate match: 59%\nStrongest evidence: ...",
            "speech_text": "Moderate match: 59%",
            "task_result": {"agent": "job", "evidence": ["match_job"],
                            "result": {"content": "..."}},
            "execution_path": ["planner", "job", "reflect_1_done", "response"],
            "query_category": "JOB_MATCH",
            "error": None,
        }

    async def _never(*args, **kwargs):
        raise AssertionError("the tool-free path ran for a JOB_MATCH turn")

    monkeypatch.setattr(streaming_workflow, "run_workflow", _fake_workflow)
    monkeypatch.setattr(streaming_workflow, "parallel_init_node", _never)
    monkeypatch.setattr(
        streaming_workflow.groq_service, "stream_chat_completion", _never
    )

    events = await _collect(streaming_workflow.run_streaming_workflow(
        user_input="how well do I match this job",
        user_id=OWNER,
        session_id="conv-1",
        conversation_history=JOB_CONTEXT,
    ))

    kinds = [e["type"] for e in events]
    assert "token" not in kinds, "a JOB_MATCH turn must not stream model tokens"
    assert kinds == ["metadata", "complete"]

    metadata, complete = events
    assert metadata["route"] == "tool_required"
    assert metadata["selected_agent"] == "job"
    assert metadata["tools_used"] == ["match_job"]
    assert complete["success"] is True
    assert "59%" in complete["display_text"]

    # The turn was handed on intact — identity, conversation and scopes.
    assert called["user_id"] == OWNER
    assert called["session_id"] == "conv-1"
    assert called["conversation_history"] == JOB_CONTEXT


async def test_escalation_happens_before_any_memory_work(monkeypatch):
    """
    Placed ahead of `parallel_init_node` deliberately. Escalating afterwards
    would run memory retrieval twice and store the user's message twice with
    it; classification is pure, so doing it first costs nothing.
    """
    from app.agents import streaming_workflow

    async def _never_init(*args, **kwargs):
        raise AssertionError("memory retrieval ran before escalating")

    async def _workflow(**kwargs):
        return {"display_text": "ok", "speech_text": "ok",
                "task_result": {"agent": "job"}, "error": None}

    monkeypatch.setattr(streaming_workflow, "parallel_init_node", _never_init)
    monkeypatch.setattr(streaming_workflow, "run_workflow", _workflow)

    events = await _collect(streaming_workflow.run_streaming_workflow(
        user_input="am I a good fit for this role",
        user_id=OWNER, session_id="conv-2",
    ))
    assert events[-1]["type"] == "complete"


async def test_ordinary_streaming_turns_are_untouched(monkeypatch):
    """
    The other half of the guarantee. A conversational turn must still stream
    tokens from the tool-free path — the escalation is narrow, not a rewrite.
    """
    from app.agents import streaming_workflow

    async def _init(state):
        state["memory_prompt"] = ""
        state["selected_agent"] = "profile"
        state["detected_intent"] = "chat"
        return state

    async def _tokens(*args, **kwargs):
        for token in ("Hi ", "there."):
            yield token

    async def _no_workflow(**kwargs):
        raise AssertionError("an ordinary turn was escalated to the tool path")

    monkeypatch.setattr(streaming_workflow, "parallel_init_node", _init)
    monkeypatch.setattr(streaming_workflow, "run_workflow", _no_workflow)
    monkeypatch.setattr(
        streaming_workflow.groq_service, "stream_chat_completion", _tokens
    )

    events = await _collect(streaming_workflow.run_streaming_workflow(
        user_input="tell me something interesting about transformers",
        user_id=OWNER, session_id="conv-3",
    ))

    assert [e["type"] for e in events].count("token") == 2
    assert events[-1]["display_text"] == "Hi there."


async def test_a_job_match_without_a_caller_identity_refuses_rather_than_guesses(
    monkeypatch,
):
    """
    With nobody to match against, the honest answer is to say so. Falling
    through to the tool-free path would produce the ungrounded reply this
    whole mechanism exists to prevent.
    """
    from app.agents import streaming_workflow

    async def _never(*args, **kwargs):
        raise AssertionError("an anonymous JOB_MATCH turn reached a model")

    monkeypatch.setattr(streaming_workflow, "parallel_init_node", _never)
    monkeypatch.setattr(
        streaming_workflow.groq_service, "stream_chat_completion", _never
    )

    events = await _collect(streaming_workflow.run_streaming_workflow(
        user_input="do I match this job", user_id="", session_id="conv-4",
    ))

    complete = events[-1]
    assert complete["success"] is False
    assert "don't know who you are" in complete["display_text"]


async def test_a_failing_tool_workflow_does_not_fall_back_to_prose(monkeypatch):
    """
    An escalation that fails must report the failure. Retrying on the tool-free
    path would turn an outage into a fabricated verdict.
    """
    from app.agents import streaming_workflow

    async def _boom(**kwargs):
        raise RuntimeError("qdrant unreachable")

    async def _never(*args, **kwargs):
        raise AssertionError("fell back to the tool-free path after a failure")

    monkeypatch.setattr(streaming_workflow, "run_workflow", _boom)
    monkeypatch.setattr(streaming_workflow, "parallel_init_node", _never)
    monkeypatch.setattr(
        streaming_workflow.groq_service, "stream_chat_completion", _never
    )

    events = await _collect(streaming_workflow.run_streaming_workflow(
        user_input="am I a good fit for this role",
        user_id=OWNER, session_id="conv-5",
    ))

    complete = events[-1]
    assert complete["success"] is False
    assert "couldn't finish that comparison" in complete["display_text"].lower()


@pytest.mark.parametrize("reply", [
    "yes", "no", "yeah", "nope", "confirm", "cancel", "send it", "go ahead",
    "do it", "okay", "sure", "stop", "not now", "yes please", "dont send it",
])
def test_a_confirmation_reply_is_never_escalated(reply):
    """
    The escalation runs *before* `decide_route`, and `decide_route` is where a
    pending action intercepts the turn. So a confirmation reply that escalated
    would be routed away from the gateway — approving or refusing an
    irreversible action would silently stop working.

    Nothing in the confirmation vocabulary can reach JOB_MATCH: none of it
    carries a fit word and a job noun. Pinned because the escalation sits
    upstream of the one decision that must never be reordered around.
    """
    from app.agents import confirmation

    decision = query_intent.classify(reply, has_context=True)
    assert query_intent.requires_tools(decision.category) is False, reply
    # And it is still recognised as a decision by the confirmation detector.
    assert confirmation.detect(reply) is not confirmation.ConfirmationIntent.NONE


async def test_a_pending_confirmation_still_reaches_the_gateway(monkeypatch):
    """The interaction end to end: escalation must not shadow confirmation."""
    from app.agents import streaming_workflow

    async def _init(state):
        state["memory_prompt"] = ""
        return state

    async def _never(*args, **kwargs):
        raise AssertionError("a confirmation reply was escalated or streamed")

    resolved: Dict[str, Any] = {}

    class _Outcome:
        text = "Sent to alice@example.com."

    async def _resolve(state):
        resolved["called"] = True
        return _Outcome()

    async def _intercept(state):
        return True

    monkeypatch.setattr(streaming_workflow, "parallel_init_node", _init)
    monkeypatch.setattr(streaming_workflow, "run_workflow", _never)
    monkeypatch.setattr(
        streaming_workflow.groq_service, "stream_chat_completion", _never
    )
    monkeypatch.setattr(
        "app.agents.confirmation.should_intercept", _intercept
    )
    monkeypatch.setattr("app.agents.confirmation.resolve", _resolve)
    monkeypatch.setattr(
        "app.agents.confirmation.pending_for_state",
        lambda state: _noop_list(),
    )

    events = await _collect(streaming_workflow.run_streaming_workflow(
        user_input="yes", user_id=OWNER, session_id="conv-confirm",
    ))

    assert resolved.get("called") is True
    assert events[-1]["agent"] == "confirm_action"
    assert "Sent to alice@example.com." in events[-1]["display_text"]


async def _noop_list():
    return []


# ═══════════════════════════════════════════════════════════════════════════
# 4. Adversarial phrasing — recall, tested against the whole voice path
# ═══════════════════════════════════════════════════════════════════════════

SPOKEN_MATCH_QUESTIONS = [
    "Do I match this job?",
    "Am I a good fit for this role?",
    "How suitable am I for this position?",
    "What are my gaps for this AI Engineer role?",
    "Does my resume fit this posting?",
    "how do I stack up against this job",
    "am I underqualified for this position",
    "am I overqualified for this role",
    "do I meet the requirements for this job",
    "do I stand a chance for this role",
    "how does my resume compare to this job description",
    "am I missing anything for this role",
    "is this role right for me",
    "should I apply",
    "should I bother applying",
    "what are my chances for this opening",
    "am I eligible for this position",
    "would I be shortlisted for this job",
    "how well do I measure up against this posting",
]


@pytest.mark.parametrize("spoken", SPOKEN_MATCH_QUESTIONS)
def test_every_phrasing_is_classified_as_a_job_match(spoken):
    decision = query_intent.classify(spoken, has_context=True)
    assert decision.category is QueryCategory.JOB_MATCH, spoken


@pytest.mark.parametrize("spoken", SPOKEN_MATCH_QUESTIONS)
async def test_no_phrasing_can_reach_the_tool_free_path(spoken, monkeypatch):
    """
    The adversarial sweep, run through the real voice entry point rather than
    against a regex. Any phrasing that reached `stream_chat_completion` would
    be a spoken job-match answer written by a model.
    """
    from app.agents import streaming_workflow

    async def _never(*args, **kwargs):
        raise AssertionError(f"{spoken!r} reached the tool-free path")

    async def _workflow(**kwargs):
        return {"display_text": "Moderate match: 59%", "speech_text": "...",
                "task_result": {"agent": "job", "evidence": ["match_job"]},
                "error": None}

    monkeypatch.setattr(streaming_workflow, "run_workflow", _workflow)
    monkeypatch.setattr(streaming_workflow, "parallel_init_node", _never)
    monkeypatch.setattr(
        streaming_workflow.groq_service, "stream_chat_completion", _never
    )

    events = await _collect(streaming_workflow.run_streaming_workflow(
        user_input=spoken, user_id=OWNER, session_id="conv-x",
        conversation_history=JOB_CONTEXT,
    ))

    assert events[0]["route"] == "tool_required", spoken
    assert "token" not in [e["type"] for e in events]


@pytest.mark.parametrize("spoken", [
    "am I qualified for this",
    "what are my chances here",
    "do I fit it",
    "am I a match for it",
])
def test_a_bare_demonstrative_resolves_from_the_conversation(spoken):
    """
    How people actually speak once a posting is on screen. Accepting "this"
    with no antecedent would let any sentence containing "fit" become a match
    query, so it is gated on the conversation having been about a job.
    """
    with_context = query_intent.classify(spoken, has_context=True, history=JOB_CONTEXT)
    assert with_context.category is QueryCategory.JOB_MATCH, spoken

    without = query_intent.classify(
        spoken, has_context=True,
        history=[{"role": "assistant", "content": "Your CGPA is 8.80."}],
    )
    assert without.category is not QueryCategory.JOB_MATCH, spoken


@pytest.mark.parametrize("spoken,expected", [
    ("find me AI engineer jobs", "JOB_SEARCH"),
    ("what is my CGPA", "PROFILE_EDUCATION"),
    ("tell me about my projects", "PROFILE_PROJECTS"),
    ("what skills do I have", "PROFILE_SKILLS"),
    ("where did I intern", "PROFILE_EXPERIENCE"),
    ("does this fit in my schedule", "SCHEDULE_TEMPORAL"),
    ("what is my timetable today", "SCHEDULE_TEMPORAL"),
    ("what did I just tell you", "CONVERSATION_CURRENT"),
    ("how do I build an AI agent", "GENERAL_KNOWLEDGE"),
    ("explain how RAG works", "GENERAL_KNOWLEDGE"),
])
def test_the_broadened_classifier_does_not_steal_other_categories(spoken, expected):
    """
    Regression cover. The fit vocabulary was widened and the check moved ahead
    of the how-to branch; neither change may claim a question it cannot answer.
    """
    assert query_intent.classify(spoken, has_context=True).category.value == expected


# ═══════════════════════════════════════════════════════════════════════════
# 5. The escalated answer is still the rendered MatchReport
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def agent():
    from app.agents.job_agent import job_agent
    return job_agent


def _resume_memory(monkeypatch, **kwargs):
    from app.candidate import builder as builder_module

    monkeypatch.setattr(
        builder_module.candidate_profile_builder, "_memory",
        FakeMemory(**kwargs), raising=False,
    )


async def test_the_escalated_answer_cannot_be_replaced_by_model_prose(
    agent, monkeypatch
):
    """
    The end of the chain. Escalation is worth nothing if the agent it reaches
    lets a model write the verdict, so the adversarial case is re-asserted at
    the destination: the model insists the user is a perfect fit, and the
    answer that comes back says the opposite because the model is not its
    author.
    """
    stub_services(monkeypatch)
    _resume_memory(monkeypatch)

    result, _ = await drive(
        agent,
        [
            tool_call("match_job", job_description=JD, title="Senior AI Engineer"),
            final("You're a perfect fit — 100% match, expert in Kubernetes!"),
        ],
        state("how well do I match this job"),
    )

    content = result["task_result"]["result"]["content"]
    assert "perfect fit" not in content
    assert "100%" not in content
    assert "Strongest evidence:" in content
    assert "Required but not evidenced" in content

    match = result["task_result"]["result"]["match"]
    assert match["evidence_ids"], "the verdict must name the chunks it used"
    kubernetes = next(
        r for r in match["requirements"] if r["requirement"] == "Kubernetes"
    )
    assert kubernetes["status"] == "missing"


async def test_a_degraded_profile_still_refuses_to_assert_gaps(agent, monkeypatch):
    """The answerability rule survives the escalation, not just the text path."""
    stub_services(monkeypatch)
    _resume_memory(monkeypatch, fail={"skills", "projects", "experience"})

    result, _ = await drive(
        agent,
        [tool_call("match_job", job_description=JD), final("done")],
        state("do I match this job"),
    )

    content = result["task_result"]["result"]["content"]
    assert "could not be loaded" in content
    assert "Required but not evidenced" not in content
    assert result["task_result"]["result"]["match"]["degraded"] is True


# ═══════════════════════════════════════════════════════════════════════════
# 6. Metadata honesty
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_reported_agent_is_the_one_that_ran(monkeypatch):
    """
    `selected_agent` held the planner's choice, and the planner is routinely
    overruled — `decide_route` picks the specialist from the category, and a
    conditional-edge function's state writes never reach the nodes. So a
    JOB_MATCH turn ran the job agent while this field still said "profile",
    and that field is what the logs, the episode record and the client all
    reported.
    """
    from app.agents import workflow as workflow_module

    async def _graph(initial_state):
        return {
            **initial_state,
            "display_text": "Moderate match: 59%",
            "selected_agent": "profile",          # the planner's stale choice
            "task_result": {"agent": "job", "evidence": ["match_job"],
                            "result": {"content": "Moderate match: 59%"}},
            "execution_path": ["planner", "job", "reflect_1_done", "response"],
            "error": None,
        }

    class _Graph:
        ainvoke = staticmethod(_graph)

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(workflow_module, "multi_agent_workflow", _Graph())
    monkeypatch.setattr(workflow_module.memory_manager, "on_agent_response", _noop)
    monkeypatch.setattr(workflow_module.memory_manager, "store_episode", _noop)
    monkeypatch.setattr(
        workflow_module.memory_manager, "get_session_turn_count",
        lambda *a, **kw: _noop(),
    )

    final_state = await workflow_module.run_workflow(
        user_input="how well do I match this job", user_id=OWNER,
        session_id="conv-meta",
    )
    assert final_state["selected_agent"] == "job"
