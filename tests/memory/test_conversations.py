"""
Conversations, working memory, and summarisation (Phase 4).

The bug this phase closes: the browser minted a fresh `session_id` on every
page load and never persisted it, while chat history was retrieved filtered by
exactly that id. A refresh therefore retrieved *zero* prior turns even though
the whole conversation sat in Postgres under the previous id.

These tests cover the pieces that make a thread resumable and bounded — the
in-memory ones exercise pure logic; repository behaviour that needs Postgres is
covered by the integration path rather than mocked into meaninglessness.

See docs/MEMORY_ARCHITECTURE.md §3.8.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.memory.cognition.summarizer import (
    SUMMARY_THRESHOLD_TURNS,
    SUMMARY_WINDOW,
    ConversationSummarizer,
    render_turns,
)
from app.memory.conversations import Conversation, Turn, derive_title
from app.memory.retrieval.working import (
    MAX_TURN_CHARS,
    WorkingMemory,
    WorkingMemoryBuilder,
)


def turn(role="user", content="hello", sequence=1, modality="text"):
    return Turn(
        conversation_id="c1", owner_id="vansh", role=role,
        content=content, sequence=sequence, modality=modality,
    )


# ─────────────────────────────────────────────────────────────────────────
# Titles
# ─────────────────────────────────────────────────────────────────────────

def test_title_comes_from_the_opening_message():
    assert derive_title("What ML internships are open?") == "What ML internships are open?"


def test_long_titles_are_truncated_with_an_ellipsis():
    title = derive_title("x" * 200, max_length=30)
    assert len(title) == 30
    assert title.endswith("…")


def test_title_collapses_whitespace():
    assert derive_title("  find   me\n\njobs  ") == "find me jobs"


def test_empty_input_gets_a_placeholder_title():
    assert derive_title("") == "New conversation"
    assert derive_title("   \n ") == "New conversation"


# ─────────────────────────────────────────────────────────────────────────
# Working memory rendering
# ─────────────────────────────────────────────────────────────────────────

def test_empty_working_memory_is_falsy_and_renders_nothing():
    assert not WorkingMemory()
    assert WorkingMemory().render() == ""


def test_turns_render_with_role_labels():
    window = WorkingMemory(turns=[
        turn("user", "What ML jobs are open?"),
        turn("assistant", "I found five roles.", sequence=2),
    ])
    rendered = window.render()
    assert "User: What ML jobs are open?" in rendered
    assert "Assistant: I found five roles." in rendered


def test_summary_precedes_the_verbatim_turns():
    """Older context first, matching the order events actually happened."""
    window = WorkingMemory(
        turns=[turn("user", "And the second one?")],
        running_summary="The user asked about internships.",
    )
    rendered = window.render()
    assert rendered.index("Earlier in this conversation") < rendered.index("User:")


def test_a_summary_alone_is_still_usable_context():
    window = WorkingMemory(running_summary="The user is preparing for interviews.")
    assert bool(window) is True
    assert "preparing for interviews" in window.render()


def test_long_turns_are_capped():
    """A long answer must not crowd out the turns that carry the thread."""
    window = WorkingMemory(turns=[turn("assistant", "z" * 2000)])
    rendered = window.render(max_turn_chars=100)
    assert "…" in rendered
    assert "z" * 101 not in rendered


def test_blank_turns_are_skipped():
    window = WorkingMemory(turns=[turn("user", "   "), turn("user", "real question", 2)])
    assert window.render().count("User:") == 1


def test_as_messages_matches_the_agent_message_shape():
    window = WorkingMemory(turns=[turn("user", "hi"), turn("assistant", "hello", 2)])
    assert window.as_messages() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_default_turn_cap_is_applied_when_unspecified():
    window = WorkingMemory(turns=[turn("assistant", "z" * (MAX_TURN_CHARS + 500))])
    assert len(window.render()) < MAX_TURN_CHARS + 100


# ─────────────────────────────────────────────────────────────────────────
# Working memory loading — degradation
# ─────────────────────────────────────────────────────────────────────────

class FakeRepo:
    def __init__(self, conversation=None, turns=None, error=None):
        self.conversation = conversation
        self.turns = turns or []
        self.error = error

    async def get(self, conversation_id, owner_id):
        if self.error:
            raise self.error
        return self.conversation

    async def recent_turns(self, conversation_id, owner_id, limit=12):
        if self.error:
            raise self.error
        return self.turns


async def test_missing_identifiers_yield_empty_working_memory():
    builder = WorkingMemoryBuilder(FakeRepo())
    assert not await builder.build("", "c1")
    assert not await builder.build("vansh", "")


async def test_an_unknown_conversation_yields_empty_working_memory():
    builder = WorkingMemoryBuilder(FakeRepo(conversation=None))
    assert not await builder.build("vansh", "missing")


async def test_a_repository_failure_degrades_rather_than_raising():
    """A turn must remain answerable without its history, just with less context."""
    builder = WorkingMemoryBuilder(FakeRepo(error=RuntimeError("database down")))
    assert not await builder.build("vansh", "c1")


async def test_a_known_conversation_loads_its_window_and_summary():
    conversation = Conversation(
        id="c1", owner_id="vansh", running_summary="Earlier: job hunting."
    )
    builder = WorkingMemoryBuilder(FakeRepo(conversation, [turn("user", "next?")]))
    window = await builder.build("vansh", "c1")

    assert window.running_summary == "Earlier: job hunting."
    assert len(window.turns) == 1
    assert window.conversation is conversation


# ─────────────────────────────────────────────────────────────────────────
# Summarisation
# ─────────────────────────────────────────────────────────────────────────

def test_render_turns_labels_and_trims():
    text = render_turns([turn("user", "y" * 900), turn("assistant", "ok", 2)], max_chars=50)
    assert text.startswith("User: ")
    assert "Assistant: ok" in text
    assert "y" * 51 not in text


def test_the_summary_threshold_exceeds_the_verbatim_window():
    """
    Otherwise a summary would be written for turns still shown in full, and the
    model would read the same exchange twice — once condensed, once verbatim.
    """
    assert SUMMARY_THRESHOLD_TURNS > SUMMARY_WINDOW


class SummaryRepo:
    def __init__(self, turns=None):
        self.turns = turns or []
        self.saved = None

    async def turns_between(self, conversation_id, owner_id, start, end):
        return self.turns

    async def set_summary(self, conversation_id, owner_id, summary, through_seq):
        self.saved = (summary, through_seq)
        return True


class FakeLLM:
    def __init__(self, content="The user has been job hunting."):
        self.content = content
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": self.content}


async def test_summarisation_folds_only_turns_that_aged_out():
    conversation = Conversation(
        id="c1", owner_id="vansh",
        turn_count=SUMMARY_WINDOW + 20, summary_through_seq=0,
    )
    repo = SummaryRepo([turn("user", "a question"), turn("assistant", "an answer", 2)])
    llm = FakeLLM()

    assert await ConversationSummarizer(repo, llm).summarise(conversation) is True

    summary, through = repo.saved
    assert summary == "The user has been job hunting."
    # Everything except the verbatim window.
    assert through == conversation.turn_count - SUMMARY_WINDOW


async def test_a_conversation_inside_the_window_is_not_summarised():
    conversation = Conversation(
        id="c1", owner_id="vansh", turn_count=5, summary_through_seq=0
    )
    llm = FakeLLM()
    assert await ConversationSummarizer(SummaryRepo(), llm).summarise(conversation) is False
    assert llm.calls == []


async def test_an_already_summarised_conversation_is_skipped():
    conversation = Conversation(
        id="c1", owner_id="vansh",
        turn_count=SUMMARY_WINDOW + 10, summary_through_seq=SUMMARY_WINDOW + 10,
    )
    llm = FakeLLM()
    assert await ConversationSummarizer(SummaryRepo(), llm).summarise(conversation) is False
    assert llm.calls == []


async def test_the_previous_summary_is_given_to_the_model():
    """It is a *running* summary — each pass folds new material into the old."""
    conversation = Conversation(
        id="c1", owner_id="vansh",
        turn_count=SUMMARY_WINDOW + 20, summary_through_seq=0,
        running_summary="Previously: the user asked about resumes.",
    )
    llm = FakeLLM()
    await ConversationSummarizer(SummaryRepo([turn()]), llm).summarise(conversation)

    prompt = llm.calls[0]["messages"][1]["content"]
    assert "Previously: the user asked about resumes." in prompt


async def test_an_empty_stretch_advances_the_marker_without_calling_the_model():
    """Otherwise the conversation is reconsidered on every single pass."""
    conversation = Conversation(
        id="c1", owner_id="vansh",
        turn_count=SUMMARY_WINDOW + 20, summary_through_seq=0,
    )
    repo = SummaryRepo([])
    llm = FakeLLM()

    assert await ConversationSummarizer(repo, llm).summarise(conversation) is False
    assert llm.calls == []
    assert repo.saved is not None
    assert repo.saved[1] == conversation.turn_count - SUMMARY_WINDOW


async def test_an_empty_model_response_leaves_the_summary_unchanged():
    conversation = Conversation(
        id="c1", owner_id="vansh",
        turn_count=SUMMARY_WINDOW + 20, summary_through_seq=0,
    )
    repo = SummaryRepo([turn()])
    assert await ConversationSummarizer(repo, FakeLLM(content="  ")).summarise(conversation) is False
    assert repo.saved is None


class BrokenSummaryRepo:
    async def needing_summary(self, *, threshold, limit=10):
        raise RuntimeError("database unreachable")


async def test_the_summary_pass_never_raises():
    stats = await ConversationSummarizer(BrokenSummaryRepo(), FakeLLM()).run_once()
    assert stats.summarised == 0


class OneFailingRepo(BrokenSummaryRepo):
    def __init__(self):
        self.conversations = [
            Conversation(id="a", owner_id="vansh", turn_count=100),
            Conversation(id="b", owner_id="vansh", turn_count=100),
        ]

    async def needing_summary(self, *, threshold, limit=10):
        return self.conversations

    async def turns_between(self, conversation_id, owner_id, start, end):
        if conversation_id == "a":
            raise RuntimeError("bad thread")
        return [turn()]

    async def set_summary(self, *args, **kwargs):
        return True


async def test_one_failing_conversation_does_not_block_the_others():
    stats = await ConversationSummarizer(OneFailingRepo(), FakeLLM()).run_once()
    assert stats.considered == 2
    assert stats.failed == 1
    assert stats.summarised == 1


# ─────────────────────────────────────────────────────────────────────────
# Conversation shape
# ─────────────────────────────────────────────────────────────────────────

def test_summary_dict_is_json_safe_and_titled():
    now = datetime.now(timezone.utc)
    conversation = Conversation(
        id="c1", owner_id="vansh", title="Job hunting",
        turn_count=4, started_at=now, last_active_at=now,
    )
    payload = conversation.summary_dict()
    assert payload["id"] == "c1"
    assert payload["title"] == "Job hunting"
    assert payload["turn_count"] == 4
    assert isinstance(payload["started_at"], str)
    # The owner is deliberately absent — this is returned to that owner.
    assert "owner_id" not in payload


def test_an_untitled_conversation_gets_a_readable_placeholder():
    assert Conversation(id="c1", owner_id="vansh").summary_dict()["title"] == (
        "Untitled conversation"
    )


def test_turn_converts_to_the_agent_message_shape():
    assert turn("user", "hi").as_message() == {"role": "user", "content": "hi"}
