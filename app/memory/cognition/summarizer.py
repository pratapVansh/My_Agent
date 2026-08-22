"""
Rolling conversation summarisation.

Working memory carries the last N turns verbatim. Without a summary of what
came before, everything older simply vanishes from context once it falls out of
that window — so a long thread loses its own beginning. With one, the window
stays bounded no matter how long the conversation runs, which is the property
that lets a thread span months.

Runs in the worker, never on the request path: this is an LLM call, and a turn
must not wait for it.

See docs/MEMORY_ARCHITECTURE.md §3.8.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import logging

from app.config import settings
from app.memory.conversations import (
    DEFAULT_WINDOW_TURNS,
    Conversation,
    Turn,
    conversation_repository,
)

logger = logging.getLogger(__name__)

# Summarise once this many turns have accumulated beyond the last summarised
# point. Comfortably above the working-memory window, so a summary is only
# written for turns that have actually aged out of it.
SUMMARY_THRESHOLD_TURNS = 20

# Turns kept verbatim in the window; anything at or below
# `turn_count - WINDOW` is fair game to fold into the summary.
SUMMARY_WINDOW = DEFAULT_WINDOW_TURNS

_SYSTEM_PROMPT = """You maintain a running summary of a conversation between a user and their personal AI assistant.

You are given the summary so far (which may be empty) and the next stretch of the conversation.
Return an updated summary that folds the new material into the old.

Rules:
- 2 to 4 sentences. This is a running summary, not a transcript.
- Third person about the user: "The user asked about...", "The assistant helped them...".
- Keep decisions, stated preferences, open questions, and anything unresolved.
- Drop pleasantries, repetition, and anything already superseded by a later turn.
- Return ONLY the summary text. No preamble, no bullet points, no quotes."""


@dataclass
class SummaryStats:
    """Outcome of one summarisation pass."""

    considered: int = 0
    summarised: int = 0
    failed: int = 0

    def summary(self) -> dict:
        return {
            "considered": self.considered,
            "summarised": self.summarised,
            "failed": self.failed,
        }


def render_turns(turns: Sequence[Turn], *, max_chars: int = 300) -> str:
    lines: List[str] = []
    for turn in turns:
        content = (turn.content or "").strip()
        if not content:
            continue
        label = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{label}: {content[:max_chars]}")
    return "\n".join(lines)


class ConversationSummarizer:
    """Folds aged-out turns into each conversation's running summary."""

    def __init__(self, repository=None, llm=None):
        self.conversations = repository or conversation_repository
        if llm is None:
            from app.services.groq_service import groq_service
            llm = groq_service
        self.llm = llm

    async def run_once(self, limit: int = 5) -> SummaryStats:
        """Summarise a few eligible conversations. Never raises."""
        stats = SummaryStats()

        try:
            candidates = await self.conversations.needing_summary(
                threshold=SUMMARY_THRESHOLD_TURNS, limit=limit
            )
        except Exception as exc:
            logger.warning("Could not find conversations needing summary: %s", exc)
            return stats

        stats.considered = len(candidates)
        for conversation in candidates:
            try:
                if await self.summarise(conversation):
                    stats.summarised += 1
            except Exception as exc:
                stats.failed += 1
                logger.warning(
                    "Summarisation failed for conversation=%s: %s",
                    conversation.id, exc,
                )
        return stats

    async def summarise(self, conversation: Conversation) -> bool:
        """
        Fold everything outside the verbatim window into the running summary.

        Only turns that have *aged out* of the window are summarised. Folding
        in turns still shown verbatim would duplicate them in the prompt — the
        model would read the same exchange twice, once condensed and once in
        full.
        """
        target_seq = conversation.turn_count - SUMMARY_WINDOW
        if target_seq <= conversation.summary_through_seq:
            return False

        turns = await self.conversations.turns_between(
            conversation.id,
            conversation.owner_id,
            conversation.summary_through_seq,
            target_seq,
        )
        transcript = render_turns(turns)
        if not transcript.strip():
            # Nothing usable, but advance the marker so this conversation is
            # not reconsidered on every single pass.
            await self.conversations.set_summary(
                conversation.id, conversation.owner_id,
                conversation.running_summary or "", target_seq,
            )
            return False

        previous = (conversation.running_summary or "").strip() or "(none yet)"
        response = await self.llm.chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Summary so far:\n{previous}\n\n"
                        f"Next part of the conversation:\n{transcript}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=250,
            # None keeps the user-facing default; see `memory_worker_model`.
            model=settings.memory_worker_model,
        )

        updated = (response.get("content") or "").strip()
        if not updated:
            return False

        await self.conversations.set_summary(
            conversation.id, conversation.owner_id, updated, target_seq
        )
        logger.info(
            "Summarised conversation=%s through turn %d (%d turns folded in)",
            conversation.id, target_seq, len(turns),
        )
        return True


# Singleton instance
conversation_summarizer = ConversationSummarizer()
