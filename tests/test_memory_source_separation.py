"""
Profile vs resume vs conversation memory, and honest reporting of gaps.

Three distinct sources answer three distinct kinds of question, and the routing
work must not collapse them into one:

  profile memory      name, college, branch, CGPA, saved preferences
  resume/document     skills, projects, experience, education, technologies
  conversation        what was said earlier — "I have an Economics class tomorrow"

The Economics flow already worked before the routing change and is pinned here
so it cannot regress: it is answered from the conversation window, not from a
tool call, and it must survive both the clarification-suppression change and a
page refresh.

Also pinned: when information genuinely does not exist the system says so,
rather than inventing a value or asking a clarifying question in its place.
"""
import pytest

from app.agents.workflow import _is_self_referential, route_after_init
from app.memory.conversations import Conversation, Turn
from app.memory.memory_manager import memory_manager
from app.memory.retrieval.working import WorkingMemory, WorkingMemoryBuilder
from app.memory.retrieval_result import RetrievalResult


ECONOMICS_STATEMENT = "I have a class tomorrow of Principles of Economics."
ECONOMICS_QUESTION = "What class do I have tomorrow? Do you remember?"


class StubConversations:
    """A conversation repository holding one thread, without Postgres."""

    def __init__(self, turns, summary=None):
        self._turns = turns
        self._summary = summary

    async def get(self, conversation_id, owner_id):
        return Conversation(
            id=conversation_id, owner_id=owner_id,
            running_summary=self._summary,
        )

    async def recent_turns(self, conversation_id, owner_id, limit=20):
        return self._turns[-limit:]


def econ_turns():
    return [
        Turn(conversation_id="c1", owner_id="vansh", role="user",
             content=ECONOMICS_STATEMENT, sequence=1, modality="text"),
        Turn(conversation_id="c1", owner_id="vansh", role="assistant",
             content="Noted — Principles of Economics tomorrow.", sequence=2,
             modality="text", agent="profile"),
    ]


# ─────────────────────────────────────────────────────────────────────────
# P3 — the Economics conversation memory must keep working
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_something_said_earlier_is_available_to_the_next_turn():
    builder = WorkingMemoryBuilder(repository=StubConversations(econ_turns()))
    working = await builder.build("vansh", "c1")

    rendered = working.render()
    assert "Principles of Economics" in rendered
    assert working


@pytest.mark.asyncio
async def test_the_economics_answer_survives_a_refresh():
    """
    After a reload the same conversation id is reused, so the same window is
    rebuilt and the class is still recallable. This is the join between the
    persistence fix and the memory it protects.
    """
    repository = StubConversations(econ_turns())
    before = await WorkingMemoryBuilder(repository=repository).build("vansh", "c1")
    after = await WorkingMemoryBuilder(repository=repository).build("vansh", "c1")

    assert "Principles of Economics" in before.render()
    assert after.render() == before.render()


@pytest.mark.asyncio
async def test_the_economics_question_is_never_sent_to_clarification():
    builder = WorkingMemoryBuilder(repository=StubConversations(econ_turns()))
    working = await builder.build("vansh", "c1")

    assert _is_self_referential(ECONOMICS_QUESTION)
    route = await route_after_init({
        "user_input": ECONOMICS_QUESTION,
        "needs_clarification": True,
        "clarification_question": "Which class do you mean?",
        "memory_context": {"chat_context": working.render()},
        "memory_prompt": working.render(),
        "selected_agent": "academic",
    })
    assert route == "academic"


@pytest.mark.asyncio
async def test_a_turn_still_answers_when_the_conversation_window_is_unavailable():
    """Losing history degrades the answer; it must not fail the turn."""

    class Broken:
        async def get(self, *a, **k):
            raise RuntimeError("database down")

        async def recent_turns(self, *a, **k):
            raise RuntimeError("database down")

    working = await WorkingMemoryBuilder(repository=Broken()).build("vansh", "c1")
    assert isinstance(working, WorkingMemory)
    assert not working


# ─────────────────────────────────────────────────────────────────────────
# P4 — the three sources stay distinct
# ─────────────────────────────────────────────────────────────────────────

def test_profile_facts_and_resume_data_render_as_separate_sections():
    prompt = memory_manager.format_context_for_prompt({
        "profile_facts": [{"key": "college", "value": "RGIPT"}],
        "long_term": {
            "skills": [{"content": "Python, FastAPI"}],
            "skills_status": "OK",
            "projects": [{"content": "My_Agent"}],
            "projects_status": "OK",
        },
        "chat_history": [{"role": "user", "content": "I have Economics tomorrow."}],
    })

    assert "college: RGIPT" in prompt
    assert "Python, FastAPI" in prompt
    assert "Economics" in prompt
    # Distinct sections, not one merged blob.
    assert prompt.count("\n\n") >= 2


def test_an_empty_context_produces_no_fabricated_sections():
    assert memory_manager.format_context_for_prompt({}).strip() == ""


# ─────────────────────────────────────────────────────────────────────────
# P5 — missing information is reported, never invented
# ─────────────────────────────────────────────────────────────────────────

def test_nothing_stored_is_stated_plainly_and_arms_the_refusal_policy():
    prompt = memory_manager.format_context_for_prompt({
        "long_term": {
            "skills": [], "skills_status": "NO_DATA",
            "projects": [], "projects_status": "NO_DATA",
        },
    })
    assert "No skills data found" in prompt
    assert "I don't have information about that" in prompt


def test_a_failed_lookup_is_not_reported_as_an_absence():
    """"We could not find out" and "there is nothing" must stay distinguishable."""
    prompt = memory_manager.format_context_for_prompt({
        "long_term": {
            "skills": [], "skills_status": "ERROR",
            "projects": [], "projects_status": "OK",
        },
    })
    assert "unknown, not absent" in prompt
    assert "No skills data found" not in prompt


def test_no_data_and_error_are_different_retrieval_states():
    assert RetrievalResult.no_data().status.value == "NO_DATA"
    assert RetrievalResult.error().status.value == "ERROR"
    assert not RetrievalResult.no_data().items


def test_profile_tools_cannot_be_pointed_at_another_persons_data():
    """
    "What is the name in Mary's resume?" must not be answered from the owner's
    resume. The retrieval tools take no identity argument at all — the user id
    is bound when the agent is constructed — so there is no path by which a
    query names whose data is read.
    """
    import inspect

    from app.agents.profile_agent import ProfileAgent

    source = inspect.getsource(ProfileAgent)
    # Every retrieval tool closes over `user_id`; none reads an id from input.
    assert 'tool_input.get("user_id")' not in source
    assert 'tool_input.get("owner")' not in source
    assert "user_id=user_id" in source


def test_the_profile_agent_is_told_to_disclaim_other_peoples_data():
    import inspect

    from app.agents.profile_agent import ProfileAgent

    prompt = inspect.getsource(ProfileAgent).lower()
    assert "signed-in user's own memory" in prompt
    assert "named person's data" in prompt
    assert "don't have access" in prompt
    assert "never present the user's data as though it were someone else's" in prompt
