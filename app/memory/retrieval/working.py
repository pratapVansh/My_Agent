"""
Working memory: the live conversation window.

Not a stored memory kind, and deliberately so. Working memory is assembled at
request time from the conversation's recent turns plus a running summary of
everything older. Storing it as a memory would duplicate `Turn` and create a
third thing to keep in sync — which is exactly the bug the redesign started
from, where three uncoordinated representations of "recent turns" existed at
once.

Occupies tier 1 of the context budget: compressed under pressure, never
dropped. Losing the thread of the current conversation is the one failure a
user notices immediately.

See docs/MEMORY_ARCHITECTURE.md §3.8.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from app.memory.conversations import (
    DEFAULT_WINDOW_TURNS,
    Conversation,
    Turn,
    conversation_repository,
)

# Per-message cap inside the window. Long assistant answers would otherwise
# crowd out the turns around them, which is what carries the thread.
MAX_TURN_CHARS = 400


@dataclass
class WorkingMemory:
    """The current conversation, as the model should see it."""

    turns: List[Turn] = field(default_factory=list)
    running_summary: Optional[str] = None
    conversation: Optional[Conversation] = None

    def __bool__(self) -> bool:
        return bool(self.turns or (self.running_summary or "").strip())

    def render(self, *, max_turn_chars: int = MAX_TURN_CHARS) -> str:
        """
        Render for prompt injection.

        The summary goes first: it is older context, and reading it before the
        verbatim turns matches the order events actually happened.
        """
        blocks: List[str] = []

        summary = (self.running_summary or "").strip()
        if summary:
            blocks.append(f"Earlier in this conversation: {summary}")

        lines: List[str] = []
        for turn in self.turns:
            content = (turn.content or "").strip()
            if not content:
                continue
            if len(content) > max_turn_chars:
                content = content[: max_turn_chars - 1].rstrip() + "…"
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {content}")

        if lines:
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    def as_messages(self) -> List[dict]:
        """Turns in the shape agents and the extractor consume."""
        return [turn.as_message() for turn in self.turns]


class WorkingMemoryBuilder:
    """Loads the conversation window for a request."""

    def __init__(self, repository=None):
        self.conversations = repository or conversation_repository

    async def build(
        self,
        owner_id: str,
        conversation_id: str,
        *,
        window: int = DEFAULT_WINDOW_TURNS,
    ) -> WorkingMemory:
        """
        Assemble the window.

        Returns an empty result rather than raising when the conversation does
        not exist or the lookup fails: a turn must still be answerable without
        its history, just with less context.
        """
        if not conversation_id or not owner_id:
            return WorkingMemory()

        try:
            conversation = await self.conversations.get(conversation_id, owner_id)
            if conversation is None:
                return WorkingMemory()

            turns = await self.conversations.recent_turns(
                conversation_id, owner_id, limit=window
            )
            return WorkingMemory(
                turns=turns,
                running_summary=conversation.running_summary,
                conversation=conversation,
            )
        except Exception:
            # Logged by the caller; an unavailable thread must degrade the
            # answer, never prevent one.
            return WorkingMemory()


# Singleton instance
working_memory_builder = WorkingMemoryBuilder()
