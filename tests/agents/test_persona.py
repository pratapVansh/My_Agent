"""
How the assistant speaks, and whether it can see what was already said.

Two properties, both of which were broken in ways that produced no error and
no failing test — the assistant simply read like a chat product and forgot the
previous turn.

Style is asserted at the *prompt* level rather than by inspecting generated
text. Whether a model obeys an instruction is a question about the model and
belongs in the live evaluation suite (`evals/`); whether the instruction is
present on every path is a question about this code, and that is what can be
pinned deterministically.
"""
from __future__ import annotations

import pytest

from app.agents import persona
from app.agents.academic_agent import AcademicAgent
from app.agents.base_agent import BaseAgent
from app.agents.email_agent import EmailAgent
from app.agents.job_agent import JobAgent
from app.agents.profile_agent import ProfileAgent
from tests.support import OWNER, state as make_state


class _Probe(BaseAgent):
    async def execute(self, state):
        return state


@pytest.fixture
def probe():
    return _Probe(name="probe", description="d")


# ═══════════════════════════════════════════════════════════════════════════
# 1. The style contract reaches every path
# ═══════════════════════════════════════════════════════════════════════════

def test_the_style_contract_is_applied_to_a_bare_prompt(probe):
    composed = probe.inject_memory_context("You do a thing.", {})
    assert "You do a thing." in composed
    assert persona.ASSISTANT_STYLE in composed


def test_the_style_contract_survives_memory_injection(probe):
    composed = probe.inject_memory_context(
        "You do a thing.", {"memory_prompt": "The user is Vansh."}
    )
    assert "The user is Vansh." in composed
    assert persona.ASSISTANT_STYLE in composed


def test_the_streaming_path_carries_the_same_contract():
    """
    Voice runs through a different prompt builder. It was previously given the
    capability text alone, so the tool-calling path and the spoken path could
    — and did — describe the same assistant differently.
    """
    from app.agents.streaming_workflow import _get_agent_system_prompt

    prompt = _get_agent_system_prompt(ProfileAgent(), {})
    assert persona.ASSISTANT_STYLE in prompt


@pytest.mark.parametrize("agent_cls", [ProfileAgent, AcademicAgent, EmailAgent, JobAgent])
def test_every_specialist_inherits_the_contract(agent_cls):
    """
    Applied in `inject_memory_context` rather than in each agent, so this holds
    for an agent added later without anyone remembering to opt in.
    """
    assert persona.ASSISTANT_STYLE in agent_cls().inject_memory_context("x", {})


def test_the_contract_is_appended_not_prepended(probe):
    """
    Ordering is load-bearing. Models weight the end of a system message
    heavily; style guidance placed first tends to be followed while the tool
    protocol drifts, which is the wrong half to lose.
    """
    composed = probe.inject_memory_context("TOOL PROTOCOL HERE", {})
    assert composed.index("TOOL PROTOCOL HERE") < composed.index(persona.ASSISTANT_STYLE)


# ═══════════════════════════════════════════════════════════════════════════
# 2. The contract says the things that were actually going wrong
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("rule", [
    "no markdown headings",       # literal asterisks on screen, noise aloud
    "contractions",               # the difference between speech and a report
    "don't restate the question", # doubles the length of every short answer
    "don't sign off",             # the clearest tell that a machine wrote it
    "never \"the user\"",         # talking about them instead of to them
])
def test_the_contract_names_the_observed_failures(rule):
    assert rule in persona.ASSISTANT_STYLE.lower()


def test_no_agent_prompt_asks_for_markdown_output():
    """
    The email agent used to specify a `**Subject:**` template, which produced
    literal asterisks on screen and the word "asterisk" when read aloud.
    """
    import inspect

    for agent_cls in (ProfileAgent, AcademicAgent, EmailAgent, JobAgent):
        source = inspect.getsource(agent_cls)
        assert "**Subject:**" not in source, agent_cls.__name__


def test_the_error_message_is_not_a_decorated_block():
    """An assistant reports trouble in a sentence, not a warning banner."""
    from app.agents.response_agent import response_agent

    rendered = response_agent._format_error_display("KeyError: 'x'")
    assert "⚠" not in rendered
    assert "**" not in rendered
    # The raw exception belongs in the logs, not on the user's screen.
    assert "KeyError" not in rendered


# ═══════════════════════════════════════════════════════════════════════════
# 3. Previous turns are actually visible to the agent
# ═══════════════════════════════════════════════════════════════════════════

async def _history_seen(agent, state) -> list:
    """The user/assistant turns an agent's loop would put in its messages."""
    seen = {}

    async def fake_call_groq(messages, **kwargs):
        seen["messages"] = messages
        return '{"type":"final","content":"ok","is_complete":true}'

    agent.call_groq = fake_call_groq  # type: ignore[assignment]
    await agent.execute_reasoning_loop(
        state=state, base_system_prompt="x", tools={}, max_iterations=1
    )
    return [
        m for m in seen.get("messages", [])
        if m.get("role") in ("user", "assistant")
        and "Iteration:" not in str(m.get("content", ""))
    ]


async def test_an_explicit_history_is_used(probe):
    """The voice path passes turns directly. Unchanged behaviour."""
    state = make_state("and the one after that?", user_id=OWNER)
    state["conversation_history"] = [
        {"role": "user", "content": "what is my next class"},
        {"role": "assistant", "content": "Digital Image Processing at 11:30."},
    ]
    turns = await _history_seen(probe, state)
    assert any("Digital Image Processing" in str(t["content"]) for t in turns)


async def test_stored_history_is_used_when_the_client_sends_none(probe):
    """
    The bug this fixes. The web chat never sends `conversation_history`, so on
    the typed path this list was always empty and every turn was answered as if
    it were the first — "send it to that address" could not resolve, because
    the draft it referred to was not in the messages.

    The conversation store had the turns the whole time; the loop was reading
    the wrong field.
    """
    state = make_state("send it to that address", user_id=OWNER)
    state["conversation_history"] = []
    state["memory_context"] = {
        "chat_history": [
            {"role": "user", "content": "draft an email to my professor"},
            {"role": "assistant", "content": "Subject: Extension request"},
        ]
    }
    turns = await _history_seen(probe, state)
    assert any("Extension request" in str(t["content"]) for t in turns)


async def test_an_explicit_history_wins_over_the_stored_one(probe):
    """
    Preferring the passed-in list keeps the voice path byte-identical: it
    already carries the turns it wants, including ones not yet persisted.
    """
    state = make_state("go on", user_id=OWNER)
    state["conversation_history"] = [{"role": "assistant", "content": "EXPLICIT"}]
    state["memory_context"] = {"chat_history": [{"role": "assistant", "content": "STORED"}]}
    turns = await _history_seen(probe, state)
    joined = " ".join(str(t["content"]) for t in turns)
    assert "EXPLICIT" in joined
    assert "STORED" not in joined


async def test_malformed_stored_history_does_not_break_a_turn(probe):
    """A bad row in the store costs context, never the answer."""
    state = make_state("hello", user_id=OWNER)
    state["conversation_history"] = []
    state["memory_context"] = {"chat_history": ["not a dict", None, {"role": "user"}]}
    assert await _history_seen(probe, state) == []


# ── The router sees the same conversation the agent will ─────────────────────

def test_the_planner_reads_stored_history_too():
    """
    Routing happens before the specialist runs, so a planner that cannot see
    the conversation classifies every follow-up in a vacuum — "email him about
    that" with no idea who "him" is. It read only `conversation_history`, which
    the web chat never sends.
    """
    from app.agents.planner_agent import planner_agent

    state = make_state("email him about that", user_id=OWNER)
    state["conversation_history"] = []
    state["memory_context"] = {
        "chat_history": [
            {"role": "user", "content": "who is my professor for Generative AI"},
            {"role": "assistant", "content": "Dr. Pallabi Saikia."},
        ]
    }
    messages = planner_agent._recent_history_messages(state)
    assert any("Pallabi Saikia" in str(m["content"]) for m in messages)


def test_the_planner_and_the_agents_read_history_the_same_way():
    """
    One helper, both readers. They previously disagreed, which is the failure
    mode where a turn is routed as a follow-up and then answered as if it were
    the first — or the reverse.
    """
    from app.agents.planner_agent import planner_agent

    state = make_state("go on", user_id=OWNER)
    state["conversation_history"] = []
    state["memory_context"] = {"chat_history": [{"role": "user", "content": "SHARED"}]}

    planner_view = planner_agent._recent_history_messages(state)
    assert planner_view
    assert persona.recent_turns(state) == state["memory_context"]["chat_history"]
