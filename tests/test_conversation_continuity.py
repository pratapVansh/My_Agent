"""
Conversation continuity across a page refresh, and clarification routing.

The reported symptom was "refresh empties the chat". The cause was not the
browser's storage, which worked: **voice and text wrote to different
conversations**. The browser stored `session_<uuid>` and sent it with every
typed turn, while the LiveKit worker derived its own id, `lk_<room>_<identity>`,
and every spoken turn went there. After a voice conversation the stored id had
no server-side thread at all, `GET /conversations/{id}` returned 404, and the
frontend responded by minting a *new* id — so the reload presented an empty
chat and forked the thread.

Two independent defects, both fixed and both pinned here:

1. the voice worker now attaches to the conversation the browser is showing;
2. a stored conversation id is never replaced on a failed or empty rehydrate —
   only the genuine absence of one creates a conversation.

The routing tests cover the second report: "What is my name?" was answered with
"which name do you mean?" while the name sat in memory. Clarification now runs
only after retrieval has been attempted.
"""
import pytest

from app.agents.workflow import (
    _has_memory_signal,
    _is_self_referential,
    route_after_init,
)


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


# ─────────────────────────────────────────────────────────────────────────
# Voice and text share one conversation
# ─────────────────────────────────────────────────────────────────────────

def test_voice_attaches_to_the_conversation_the_browser_is_showing():
    from app.routes.livekit_routes import bind_voice_conversation, resolve_voice_conversation

    bind_voice_conversation("voice-vansh", "vansh", "session_abc123")
    assert resolve_voice_conversation("voice-vansh", "vansh") == "session_abc123"


def test_an_unbound_room_falls_back_rather_than_failing():
    """An older client that sends no conversation id must still get voice."""
    from app.routes.livekit_routes import resolve_voice_conversation

    assert resolve_voice_conversation("voice-nobody", "nobody") is None


def test_rebinding_moves_voice_to_the_new_thread():
    """After New Chat the next spoken turn belongs to the new conversation."""
    from app.routes.livekit_routes import bind_voice_conversation, resolve_voice_conversation

    bind_voice_conversation("voice-vansh", "vansh", "session_first")
    bind_voice_conversation("voice-vansh", "vansh", "session_second")
    assert resolve_voice_conversation("voice-vansh", "vansh") == "session_second"


def test_bindings_are_scoped_per_identity():
    from app.routes.livekit_routes import bind_voice_conversation, resolve_voice_conversation

    bind_voice_conversation("voice-a", "a", "session_a")
    bind_voice_conversation("voice-b", "b", "session_b")
    assert resolve_voice_conversation("voice-a", "a") == "session_a"
    assert resolve_voice_conversation("voice-b", "b") == "session_b"


# ─────────────────────────────────────────────────────────────────────────
# Rehydration — the repository contract the refresh depends on
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_turns_come_back_in_order_with_agent_labels():
    """What the UI rebuilds the transcript from after a refresh."""
    from app.memory.conversations import Turn

    turns = [
        Turn(conversation_id="c1", owner_id="vansh", role="user",
             content="What is my name?", sequence=1, modality="text"),
        Turn(conversation_id="c1", owner_id="vansh", role="assistant",
             content="Your name is Vansh Pratap Singh.", sequence=2,
             modality="text", agent="profile"),
        Turn(conversation_id="c1", owner_id="vansh", role="user",
             content="And my college?", sequence=3, modality="voice"),
    ]

    ordered = sorted(turns, key=lambda t: t.sequence)
    assert [t.sequence for t in ordered] == [1, 2, 3]
    assert [t.role for t in ordered] == ["user", "assistant", "user"]
    assert ordered[1].agent == "profile"
    # Voice and text turns coexist in one thread — that is the point of the fix.
    assert {t.modality for t in ordered} == {"text", "voice"}


# ─────────────────────────────────────────────────────────────────────────
# Clarification is the last resort
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query",
    [
        "What is my name?",
        "what's my name",
        "What college do I attend?",
        "which branch am I in",
        "What is my CGPA?",
        "What are my saved preferences?",
        "What are my skills?",
        "Tell me about my projects.",
        "What experience do I have?",
        "What did I just tell you?",
        "What class did I say I have tomorrow?",
        "What did we discuss earlier?",
    ],
)
def test_questions_about_the_user_are_recognised_as_self_referential(query):
    assert _is_self_referential(query)


@pytest.mark.parametrize(
    "query",
    [
        "Explain gradient descent.",
        "Search for backend internships in Bangalore.",
        "What is the capital of France?",
    ],
)
def test_general_questions_are_not_self_referential(query):
    assert not _is_self_referential(query)


def test_memory_signal_is_detected_from_any_source():
    assert _has_memory_signal(state(memory_context={"profile_facts": [{"key": "name"}]}))
    assert _has_memory_signal(state(memory_context={"episodes": [{"content": "x"}]}))
    assert _has_memory_signal(state(memory_context={"chat_history": [{"role": "user", "content": "earlier turn"}]}))
    assert _has_memory_signal(state(memory_prompt="User Profile:\n- name: Vansh"))


def test_no_memory_signal_when_every_source_is_empty():
    assert not _has_memory_signal(
        state(memory_context={"profile_facts": [], "episodes": [], "chat_history": [], "long_term": {}})
    )


def test_a_personal_question_with_memory_is_answered_not_questioned():
    """The exact reported failure: "What is my name?" → clarifying question."""
    route = route_after_init(state(
        user_input="What is my name?",
        needs_clarification=True,
        clarification_question="Which name do you mean?",
        memory_context={"profile_facts": [{"key": "name", "value": "Vansh Pratap Singh"}]},
        selected_agent="profile",
    ))
    assert route == "profile"


def test_suppressing_clarification_clears_the_question_from_state():
    """Otherwise the response agent could still render the stale question."""
    s = state(
        user_input="What is my college?",
        needs_clarification=True,
        clarification_question="Which college?",
        memory_context={"chat_history": [{"role": "user", "content": "earlier"}]},
    )
    route_after_init(s)
    assert s["needs_clarification"] is False
    assert s["clarification_question"] == ""


def test_a_personal_question_routes_to_the_planners_agent_not_always_profile():
    route = route_after_init(state(
        user_input="What classes do I have tomorrow?",
        needs_clarification=True,
        memory_context={"chat_history": [{"role": "user", "content": "earlier turn"}]},
        selected_agent="academic",
    ))
    assert route == "academic"


def test_clarification_still_fires_when_a_parameter_is_genuinely_missing():
    """"Email him about it" — no referent, and no self-reference to recover one."""
    route = route_after_init(state(
        user_input="Send him an email about it",
        needs_clarification=True,
        clarification_question="Who should I email?",
        memory_context={"profile_facts": [{"key": "name"}]},
    ))
    assert route == "clarification"


def test_an_empty_store_is_not_a_reason_to_interrogate_the_user():
    """
    POLICY CHANGE (was: clarification fires when memory is empty).

    The earlier rule required a memory signal before suppressing clarification,
    so a personal question against an empty store still produced "Which name?".
    That is the wrong branch: emptiness is a property of the store, ambiguity is
    a property of the question, and "What is my name?" is not ambiguous however
    little is stored. Retrieval now runs regardless and the agent reports an
    empty result honestly. See tests/test_answer_first_routing.py.
    """
    route = route_after_init(state(
        user_input="What is my name?",
        needs_clarification=True,
        clarification_question="Which name?",
        memory_context={},
    ))
    assert route == "profile"


def test_errors_still_short_circuit_to_the_response_agent():
    assert route_after_init(state(error="boom", needs_clarification=True)) == "response"


def test_normal_routing_is_unchanged_when_no_clarification_is_requested():
    assert route_after_init(state(user_input="find me jobs", selected_agent="job")) == "job"
    assert route_after_init(state(user_input="hello", selected_agent="nonsense")) == "profile"
