"""
How much work one logical turn is allowed to generate.

The audit's finding was multiplicative: a planner call that was often discarded,
an initialization performed twice on escalation, and a reflect loop that
answered a rate limit by re-running the whole specialist. Each was defensible
alone; together they were the amplification.

Everything asserted here is a count or a call site — not an output shape —
because the outputs were always fine. That was the problem: the waste was
invisible from the answer.
"""
from __future__ import annotations

import asyncio
import json
import logging

import pytest

from app.agents import query_intent, workflow
from app.agents.planner_agent import planner_agent
from app.agents.workflow import parallel_init_node, reflect_node
from app.memory.sources import QueryCategory


# ═══════════════════════════════════════════════════════════════════════════
# I/J · The planner produces parseable JSON, or fails deterministically
# ═══════════════════════════════════════════════════════════════════════════

def _valid_plan() -> str:
    return json.dumps({
        "intent": "find software jobs",
        "agent": "job",
        "confidence": 0.95,
        "needs_clarification": False,
        "clarification_question": "",
        "execution_plan": [{"step": 1, "agent": "job", "goal": "search"}],
    })


def _planner_state(text: str) -> dict:
    return {
        "user_input": text,
        "user_id": "vansh",
        "session_id": "amp-1",
        "execution_path": [],
    }


async def test_the_planner_asks_for_constrained_json(monkeypatch):
    """
    `response_format` is what stops a preamble or a fenced block being emitted.

    It could not previously be passed at all: `call_groq` accepted no forwarding
    kwargs, so the planner had no way to ask for constrained decoding even
    though the voice router two modules over already did.
    """
    seen = {}

    async def capture(messages, temperature=0.7, max_tokens=1024, _retries=2, **kwargs):
        seen.update(kwargs)
        seen["max_tokens"] = max_tokens
        return _valid_plan()

    monkeypatch.setattr(planner_agent, "call_groq", capture)

    result = await planner_agent.execute(_planner_state("find me a job"))

    assert seen["response_format"] == {"type": "json_object"}
    assert result["selected_agent"] == "job"


async def test_the_planner_token_budget_leaves_room_for_reasoning(monkeypatch):
    """
    gpt-oss-120b bills its reasoning channel from this same budget.

    At 500 the object being requested — intent, agent, confidence, a
    clarification question and a multi-step plan — routinely ran out of budget
    mid-JSON. That truncation was the malformed output seen in the logs.
    """
    seen = {}

    async def capture(messages, temperature=0.7, max_tokens=1024, _retries=2, **kwargs):
        seen["max_tokens"] = max_tokens
        return _valid_plan()

    monkeypatch.setattr(planner_agent, "call_groq", capture)
    await planner_agent.execute(_planner_state("find me a job"))

    assert seen["max_tokens"] >= 1500


async def test_malformed_json_falls_back_without_a_second_call(monkeypatch, caplog):
    """
    The repair for bad JSON is prevention, never a second request.

    Asking the model again to fix its own output doubles the cost of the turn
    at exactly the moment the provider is most likely to be what broke it.
    """
    calls = {"n": 0}

    async def truncated(messages, temperature=0.7, max_tokens=1024, _retries=2, **kwargs):
        calls["n"] += 1
        return '{"intent": "find jobs", "agent": "job", "confid'

    monkeypatch.setattr(planner_agent, "call_groq", truncated)

    with caplog.at_level(logging.ERROR, logger="app.agents.planner_agent"):
        result = await planner_agent.execute(_planner_state("find me a job"))

    assert calls["n"] == 1, "malformed JSON must not trigger a repair call"
    assert result["selected_agent"] == "profile"
    assert result["needs_clarification"] is False
    assert any("unparseable" in r.message for r in caplog.records), (
        "a silent misroute must be visible at ERROR"
    )


async def test_a_fenced_response_is_still_parsed(monkeypatch):
    """Constrained decoding is the belt; the existing stripping is the braces."""
    async def fenced(messages, temperature=0.7, max_tokens=1024, _retries=2, **kwargs):
        return f"```json\n{_valid_plan()}\n```"

    monkeypatch.setattr(planner_agent, "call_groq", fenced)

    result = await planner_agent.execute(_planner_state("find me a job"))
    assert result["selected_agent"] == "job"


# ═══════════════════════════════════════════════════════════════════════════
# The planner is not called when its answer cannot be read
# ═══════════════════════════════════════════════════════════════════════════

def test_a_clock_question_does_not_need_the_planner():
    assert not query_intent.planner_is_load_bearing(
        QueryCategory.TEMPORAL_CURRENT.value
    )


def test_a_provenance_question_does_not_need_the_planner():
    assert not query_intent.planner_is_load_bearing(
        QueryCategory.PROVENANCE_QUERY.value
    )


@pytest.mark.parametrize(
    "category",
    [
        QueryCategory.JOB_SEARCH.value,
        QueryCategory.SCHEDULE_TEMPORAL.value,
        QueryCategory.ACTION_REQUEST.value,
        QueryCategory.PROFILE_SKILLS.value,
        None,
        "SOMETHING_UNRECOGNISED",
    ],
)
def test_a_category_alone_is_not_enough_to_skip_the_planner(category):
    """
    Called with a category and nothing else, only the narrow set may skip.

    The second rule needs the utterance — it turns on whether the sentence asks
    for one thing — so a caller that supplies no text must get the conservative
    answer. This is also the backwards-compatibility guarantee for every
    existing caller of the one-argument form.
    """
    assert query_intent.planner_is_load_bearing(category)


@pytest.mark.parametrize("text", [
    "what is my name",
    "what is my CGPA",
    "what are my skills",
    "find me machine learning jobs",
    "what classes do I have tomorrow",
])
def test_a_single_lookup_asked_for_once_does_not_need_the_planner(text):
    """
    The turn is fully determined before the model is asked anything.

    Three conditions hold together for each of these: `agent_for` returns the
    same specialist whatever the planner might say, `grounding.required_tools`
    names what the answer must come from, and the sentence contains one
    request. Nothing the planner could produce is read — so producing it costs
    ~3,500 tokens of an 8,000 TPM budget for output that is thrown away.
    """
    category = query_intent.classify(text).category.value
    assert not query_intent.planner_is_load_bearing(category, text=text), text


@pytest.mark.parametrize("text", [
    # Compound: the second clause is a second task, and a multi-step plan is
    # the one thing only the planner produces.
    "check my attendance and email my professor",
    "what are my projects then draft a cover letter",
    # No required tool: nothing names what this turn must call, so the
    # planner's routing is still the decision.
    "hello there",
    "explain how transformers work",
    # A conversation follow-up — answered from context, not from a lookup.
    "what did I just tell you",
])
def test_anything_less_determined_still_runs_the_planner(text):
    category = query_intent.classify(text).category.value
    assert query_intent.planner_is_load_bearing(category, text=text), text


def test_a_list_is_not_a_plan():
    """
    "and" alone must not be read as a second step.

    Requiring a conjunction *and* a task verb is what keeps "my skills and
    projects" — one lookup, two nouns — from paying for a planner call it has
    no use for.
    """
    assert not query_intent.looks_compound("what are my skills and projects")
    assert query_intent.looks_compound("what are my skills and email them to me")


async def test_the_planner_is_skipped_for_a_clock_question(monkeypatch):
    planner_calls = {"n": 0}

    async def counted(state):
        planner_calls["n"] += 1
        return state

    async def _memory(**kwargs):
        return {}, "some memory"

    monkeypatch.setattr(workflow.planner_agent, "execute", counted)
    monkeypatch.setattr(workflow.memory_manager, "build_memory_prompt", _memory)
    monkeypatch.setattr(
        workflow.memory_manager, "on_user_input", _noop_on_user_input
    )

    state = _workflow_state("what time is it")
    await parallel_init_node(state)

    assert planner_calls["n"] == 0
    assert state["route"] == "temporal"


async def test_the_planner_is_skipped_for_a_settled_personal_question(monkeypatch):
    """
    "What is my name" reached the planner, and every field it produced was
    discarded: `agent_for` sends a personal question to profile whatever the
    planner concluded, and `required_tools` already named `get_identity`.

    The skip has to leave the state downstream reads intact, which is why the
    route is asserted alongside the call count — a cheaper turn that routes
    somewhere else is not a cheaper turn, it is a broken one.
    """
    planner_calls = {"n": 0}

    async def counted(state):
        planner_calls["n"] += 1
        return state

    async def _memory(**kwargs):
        return {}, "some memory"

    monkeypatch.setattr(workflow.planner_agent, "execute", counted)
    monkeypatch.setattr(workflow.memory_manager, "build_memory_prompt", _memory)
    monkeypatch.setattr(
        workflow.memory_manager, "on_user_input", _noop_on_user_input
    )

    state = _workflow_state("what is my name")
    await parallel_init_node(state)

    assert planner_calls["n"] == 0
    assert state["route"] == "profile"
    assert state["required_tools"] == ["get_identity", "get_profile_summary"]
    # Nothing downstream may find these missing just because nobody was asked.
    assert state["needs_clarification"] is False
    assert state["detected_intent"] == "what is my name"


async def test_the_planner_still_runs_for_a_compound_request(monkeypatch):
    """
    The planner's one irreplaceable output is a multi-step plan, and a compound
    sentence is where one comes from. Skipping here would not save a call, it
    would silently drop half of what the user asked for.
    """
    planner_calls = {"n": 0}

    async def counted(state):
        planner_calls["n"] += 1
        state["selected_agent"] = "academic"
        state["execution_plan"] = []
        return state

    async def _memory(**kwargs):
        return {}, "some memory"

    monkeypatch.setattr(workflow.planner_agent, "execute", counted)
    monkeypatch.setattr(workflow.memory_manager, "build_memory_prompt", _memory)
    monkeypatch.setattr(
        workflow.memory_manager, "on_user_input", _noop_on_user_input
    )

    await parallel_init_node(
        _workflow_state("check my attendance and email my professor")
    )

    assert planner_calls["n"] == 1


async def _noop_on_user_input(**kwargs):
    return None


def _workflow_state(text: str) -> dict:
    return {
        "user_input": text,
        "user_id": "vansh",
        "session_id": "amp-2",
        "conversation_history": [],
        "execution_path": [],
        "execution_plan": [],
        "request_id": "req-amp",
    }


# ═══════════════════════════════════════════════════════════════════════════
# O · Escalation does not re-initialize
# ═══════════════════════════════════════════════════════════════════════════

async def test_prefetched_initialization_is_adopted(monkeypatch):
    """
    The streaming path retrieves and plans, then hands the turn to the tool
    workflow. Both were being done twice.
    """
    planner_calls = {"n": 0}
    memory_calls = {"n": 0}

    async def counted_planner(state):
        planner_calls["n"] += 1
        return state

    async def counted_memory(**kwargs):
        memory_calls["n"] += 1
        return {}, "retrieved"

    monkeypatch.setattr(workflow.planner_agent, "execute", counted_planner)
    monkeypatch.setattr(workflow.memory_manager, "build_memory_prompt", counted_memory)
    monkeypatch.setattr(workflow.memory_manager, "on_user_input", _noop_on_user_input)

    state = _workflow_state("find me a machine learning job")
    state["prefetched"] = {
        "memory_prompt": "already retrieved",
        "memory_context": {"chat_history": []},
        "query_category": QueryCategory.JOB_SEARCH.value,
        "selected_agent": "job",
        "detected_intent": "find jobs",
        "execution_plan": [],
    }

    await parallel_init_node(state)

    assert planner_calls["n"] == 0, "the planner must not run a second time"
    assert memory_calls["n"] == 0, "retrieval must not run a second time"
    assert state["memory_prompt"] == "already retrieved"


async def test_a_handover_without_memory_is_not_adopted(monkeypatch):
    """
    An absent memory prompt is indistinguishable from a failed retrieval.

    Adopting one would answer a personal question with no memory and no sign
    that anything had gone wrong — so a partial handover falls back to doing
    the work, which is slower and correct.
    """
    memory_calls = {"n": 0}

    async def counted_memory(**kwargs):
        memory_calls["n"] += 1
        return {}, "retrieved properly"

    async def _planner(state):
        return state

    monkeypatch.setattr(workflow.planner_agent, "execute", _planner)
    monkeypatch.setattr(workflow.memory_manager, "build_memory_prompt", counted_memory)
    monkeypatch.setattr(workflow.memory_manager, "on_user_input", _noop_on_user_input)

    state = _workflow_state("what are my skills")
    # Init timed out upstream, so there is no memory prompt to hand over.
    state["prefetched"] = {"query_category": "PROFILE_SKILLS", "selected_agent": "profile"}

    await parallel_init_node(state)

    assert memory_calls["n"] == 1
    assert state["memory_prompt"] == "retrieved properly"


def test_the_streaming_path_packages_what_it_has():
    from app.agents.streaming_workflow import _prefetched_from

    handover = _prefetched_from({
        "memory_prompt": "context",
        "memory_context": {"chat_history": []},
        "query_category": "PROFILE_SKILLS",
        "selected_agent": "profile",
        "detected_intent": "skills",
        "execution_plan": [],
    })

    assert handover["memory_prompt"] == "context"
    assert handover["query_category"] == "PROFILE_SKILLS"


def test_the_streaming_path_omits_memory_it_does_not_have():
    """Init timed out — the handover must not claim retrieval succeeded."""
    from app.agents.streaming_workflow import _prefetched_from

    handover = _prefetched_from({"memory_prompt": "", "query_category": "PROFILE_SKILLS"})

    assert "memory_prompt" not in handover


# ═══════════════════════════════════════════════════════════════════════════
# Reflect does not answer a rate limit by doing more work
# ═══════════════════════════════════════════════════════════════════════════

def _failed_state(**extra) -> dict:
    state = {
        "task_result": {"status": "failed", "agent": "job", "result": {"content": ""}},
        "iteration_count": 0,
        "execution_path": [],
        "execution_plan": [],
        "current_step_index": 0,
        "query_category": QueryCategory.JOB_SEARCH.value,
    }
    state.update(extra)
    return state


async def test_a_rate_limited_turn_is_not_retried():
    """
    Re-running a specialist is the most expensive thing this system can do.

    Doing it in response to being told we are making too many requests puts the
    retry into the same closed window, and every rejected attempt still counts
    against the account.
    """
    state = await reflect_node(_failed_state(rate_limited=True))

    assert state["reflect_outcome"] == "done"
    assert "rate_limited" in " ".join(state["execution_path"])


async def test_a_rate_limit_recorded_on_the_envelope_also_stops_the_retry():
    result = {
        "status": "failed", "agent": "job",
        "result": {"content": ""}, "rate_limited": True,
    }
    state = await reflect_node(_failed_state(task_result=result))

    assert state["reflect_outcome"] == "done"


async def test_a_rate_limit_reported_only_in_the_error_string_still_stops_it():
    """Some paths surface it as text; the classifier reads that too."""
    state = await reflect_node(
        _failed_state(error="Groq API error: 429 rate limited")
    )

    assert state["reflect_outcome"] == "done"


async def test_an_ordinary_failure_is_still_retried():
    """
    The gate is narrow on purpose.

    The failure this loop exists for — a required tool the first attempt did
    not call — must still be recovered.
    """
    state = await reflect_node(_failed_state())

    assert state["reflect_outcome"] == "retry"
    assert state["reflect_failure_context"]


async def test_the_retry_budget_is_two_passes():
    """Each pass re-runs a full reasoning loop; three was the amplification."""
    assert workflow.MAX_ITERATIONS == 2

    state = await reflect_node(_failed_state(iteration_count=1))

    assert state["reflect_outcome"] == "done", "the second pass must be the last"


async def test_no_data_is_still_not_a_failure():
    """
    An empty store is an answer.

    The NO_DATA / TOOL_ERROR distinction is load-bearing and untouched by the
    rate-limit gate above.
    """
    state = await reflect_node({
        "task_result": {"status": "success", "agent": "profile", "result": {"content": "none stored"}},
        "answerability": "NO_DATA",
        "iteration_count": 0,
        "execution_path": [],
        "execution_plan": [],
        "current_step_index": 0,
    })

    assert state["reflect_outcome"] == "done"


# ═══════════════════════════════════════════════════════════════════════════
# M · The user-facing path does not wait for memory writes
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_vector_write_does_not_block_user_input(monkeypatch):
    """
    The embedding and upsert used to be awaited *before* retrieval ran.

    Nothing in the turn reads them back — the utterance being stored is the one
    the caller just sent — so it was pure latency the user experienced as the
    assistant being slow.
    """
    from app.memory.memory_manager import MemoryManager

    manager = MemoryManager()
    released = asyncio.Event()
    finished = asyncio.Event()

    class _ShortTerm:
        async def store_chat_message(self, **kw):
            return None

    class _Smart:
        async def extract_and_store(self, **kw):
            await released.wait()
            finished.set()

    manager.short_term = _ShortTerm()
    manager.smart = _Smart()
    monkeypatch.setattr(manager, "_append_turn", _noop_append)

    # Returns while the vector write is still blocked.
    await asyncio.wait_for(
        manager.on_user_input(user_id="vansh", session_id="s", user_message="I study at NIT and my CGPA is 8.9"),
        timeout=1.0,
    )
    assert not finished.is_set(), "on_user_input waited for the vector write"

    released.set()
    await asyncio.wait_for(finished.wait(), timeout=1.0)


async def test_the_detached_write_still_runs_and_is_not_collected(monkeypatch):
    """
    Fire-and-forget without a strong reference is a task asyncio may collect
    mid-run, and whose exception nobody ever sees.
    """
    from app.memory.memory_manager import MemoryManager

    manager = MemoryManager()
    stored = asyncio.Event()

    class _ShortTerm:
        async def store_chat_message(self, **kw):
            return None

    class _Smart:
        async def extract_and_store(self, **kw):
            stored.set()

    manager.short_term = _ShortTerm()
    manager.smart = _Smart()
    monkeypatch.setattr(manager, "_append_turn", _noop_append)

    await manager.on_user_input(
        user_id="vansh", session_id="s", user_message="My CGPA is 8.9 and I study at NIT",
    )
    await asyncio.wait_for(stored.wait(), timeout=1.0)


async def test_a_failing_detached_write_does_not_surface_to_the_caller(monkeypatch):
    from app.memory.memory_manager import MemoryManager

    manager = MemoryManager()

    class _ShortTerm:
        async def store_chat_message(self, **kw):
            return None

    class _Smart:
        async def extract_and_store(self, **kw):
            raise RuntimeError("qdrant down")

    manager.short_term = _ShortTerm()
    manager.smart = _Smart()
    monkeypatch.setattr(manager, "_append_turn", _noop_append)

    await manager.on_user_input(
        user_id="vansh", session_id="s", user_message="My CGPA is 8.9 and I study at NIT",
    )
    await asyncio.sleep(0.05)  # let the detached task fail


async def _noop_append(*args, **kwargs):
    return None
