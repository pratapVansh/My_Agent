"""
Conversation threads and their turns.

The problem this fixes: the browser minted a fresh `session_id` on every page
load and never persisted it, while chat history was retrieved filtered by
exactly that id. A refresh therefore retrieved *zero* prior turns even though
the whole conversation sat in Postgres under the previous id. Continuity
depended entirely on session-independent memory; the actual thread was lost.

A conversation is now a first-class, addressable entity. The browser persists
its id, voice and text write to the same one, and resuming is a lookup rather
than an accident.

See docs/MEMORY_ARCHITECTURE.md §3.8.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from sqlalchemy import and_, desc, func, select, update as sa_update

from app.db.session import async_session_maker
from app.memory.models import ConversationORM, TurnORM
from app.memory.record import utcnow

logger = logging.getLogger(__name__)

# How many turns are carried verbatim in working memory. Everything older is
# represented by the running summary.
DEFAULT_WINDOW_TURNS = 12


@dataclass
class Turn:
    """One message in a thread."""

    conversation_id: str
    owner_id: str
    role: str
    content: str
    sequence: int = 0
    modality: str = "text"
    agent: Optional[str] = None
    intent: Optional[str] = None
    created_at: Optional[datetime] = None

    def as_message(self) -> Dict[str, str]:
        """The shape agents and the extractor consume."""
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """A thread of turns, with a summary of everything older than the window."""

    id: str
    owner_id: str
    title: Optional[str] = None
    status: str = "active"
    modality: str = "text"
    turn_count: int = 0
    running_summary: Optional[str] = None
    summary_through_seq: int = 0
    started_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title or "Untitled conversation",
            "status": self.status,
            "modality": self.modality,
            "turn_count": self.turn_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_active_at": (
                self.last_active_at.isoformat() if self.last_active_at else None
            ),
        }


def _to_conversation(row: ConversationORM) -> Conversation:
    return Conversation(
        id=row.id,
        owner_id=row.owner_id,
        title=row.title,
        status=row.status,
        modality=row.modality,
        turn_count=row.turn_count,
        running_summary=row.running_summary,
        summary_through_seq=row.summary_through_seq,
        started_at=row.started_at,
        last_active_at=row.last_active_at,
    )


def _to_turn(row: TurnORM) -> Turn:
    return Turn(
        conversation_id=row.conversation_id,
        owner_id=row.owner_id,
        role=row.role,
        content=row.content,
        sequence=row.sequence,
        modality=row.modality,
        agent=row.agent,
        intent=row.intent,
        created_at=row.created_at,
    )


def derive_title(text: str, *, max_length: int = 60) -> str:
    """A provisional title from the opening message, until one is set."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "New conversation"
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 1].rstrip() + "…"


class ConversationRepository:
    """Persistence for conversations and their turns."""

    def __init__(self, session_maker=None):
        self.async_session_maker = session_maker or async_session_maker

    async def ensure(
        self,
        conversation_id: str,
        owner_id: str,
        *,
        modality: str = "text",
        title: Optional[str] = None,
    ) -> Conversation:
        """Fetch the conversation, creating it if this is its first turn."""
        existing = await self.get(conversation_id, owner_id)
        if existing is not None:
            return existing

        async with self.async_session_maker() as session:
            session.add(ConversationORM(
                id=conversation_id,
                owner_id=owner_id,
                title=title,
                modality=modality,
            ))
            try:
                await session.commit()
            except Exception:
                # A concurrent first turn created it; re-read rather than fail.
                await session.rollback()

        found = await self.get(conversation_id, owner_id)
        if found is None:
            raise RuntimeError(f"Could not create conversation {conversation_id}")
        return found

    async def get(
        self, conversation_id: str, owner_id: str
    ) -> Optional[Conversation]:
        """Owner-scoped: an id alone must never read another user's thread."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(ConversationORM).where(
                    and_(
                        ConversationORM.id == conversation_id,
                        ConversationORM.owner_id == owner_id,
                    )
                )
            )
            row = result.scalars().first()
            return _to_conversation(row) if row else None

    async def append_turn(
        self,
        conversation_id: str,
        owner_id: str,
        role: str,
        content: str,
        *,
        modality: str = "text",
        agent: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> Optional[Turn]:
        """
        Append a turn, assigning its sequence atomically.

        The sequence comes from an UPDATE … RETURNING on the conversation row,
        which takes a row lock for the duration. Computing it with a separate
        SELECT max(sequence) would race: a voice turn and a text turn landing
        together would both read the same maximum and collide on the unique
        constraint.
        """
        if not (content or "").strip():
            return None

        await self.ensure(conversation_id, owner_id, modality=modality)

        async with self.async_session_maker() as session:
            result = await session.execute(
                sa_update(ConversationORM)
                .where(
                    and_(
                        ConversationORM.id == conversation_id,
                        ConversationORM.owner_id == owner_id,
                    )
                )
                .values(
                    turn_count=ConversationORM.turn_count + 1,
                    last_active_at=utcnow(),
                )
                .returning(ConversationORM.turn_count, ConversationORM.title,
                           ConversationORM.modality)
            )
            row = result.fetchone()
            if row is None:
                await session.rollback()
                return None

            sequence, title, current_modality = row[0], row[1], row[2]

            turn = TurnORM(
                conversation_id=conversation_id,
                owner_id=owner_id,
                sequence=sequence,
                role=role,
                content=content,
                modality=modality,
                agent=agent,
                intent=intent,
            )
            session.add(turn)

            updates: Dict[str, Any] = {}
            # Title the thread from its first user message.
            if not title and role == "user":
                updates["title"] = derive_title(content)
            # A thread used both ways is "mixed" — that is the whole point of
            # one conversation rather than one per transport.
            if current_modality != modality and current_modality != "mixed":
                updates["modality"] = "mixed"
            if updates:
                await session.execute(
                    sa_update(ConversationORM)
                    .where(ConversationORM.id == conversation_id)
                    .values(**updates)
                )

            await session.commit()
            return _to_turn(turn)

    async def recent_turns(
        self, conversation_id: str, owner_id: str, limit: int = DEFAULT_WINDOW_TURNS
    ) -> List[Turn]:
        """The last N turns, chronological."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(TurnORM)
                .where(
                    and_(
                        TurnORM.conversation_id == conversation_id,
                        TurnORM.owner_id == owner_id,
                    )
                )
                .order_by(desc(TurnORM.sequence))
                .limit(limit)
            )
            rows = list(result.scalars().all())
        return [_to_turn(row) for row in reversed(rows)]

    async def turns_between(
        self, conversation_id: str, owner_id: str, start_seq: int, end_seq: int
    ) -> List[Turn]:
        """Turns in a sequence range — used by the summariser."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(TurnORM)
                .where(
                    and_(
                        TurnORM.conversation_id == conversation_id,
                        TurnORM.owner_id == owner_id,
                        TurnORM.sequence > start_seq,
                        TurnORM.sequence <= end_seq,
                    )
                )
                .order_by(TurnORM.sequence)
            )
            return [_to_turn(row) for row in result.scalars().all()]

    async def list_for_owner(
        self, owner_id: str, *, limit: int = 30, offset: int = 0,
        status: str = "active",
    ) -> List[Conversation]:
        """Threads, most recently active first."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(ConversationORM)
                .where(
                    and_(
                        ConversationORM.owner_id == owner_id,
                        ConversationORM.status == status,
                    )
                )
                .order_by(desc(ConversationORM.last_active_at))
                .limit(limit)
                .offset(offset)
            )
            return [_to_conversation(row) for row in result.scalars().all()]

    async def set_summary(
        self, conversation_id: str, owner_id: str, summary: str, through_seq: int
    ) -> bool:
        async with self.async_session_maker() as session:
            result = await session.execute(
                sa_update(ConversationORM)
                .where(
                    and_(
                        ConversationORM.id == conversation_id,
                        ConversationORM.owner_id == owner_id,
                    )
                )
                .values(running_summary=summary, summary_through_seq=through_seq)
            )
            await session.commit()
            return result.rowcount > 0

    async def archive(self, conversation_id: str, owner_id: str) -> bool:
        """Hide a thread without destroying it — deletion is Phase 6."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                sa_update(ConversationORM)
                .where(
                    and_(
                        ConversationORM.id == conversation_id,
                        ConversationORM.owner_id == owner_id,
                    )
                )
                .values(status="archived")
            )
            await session.commit()
            return result.rowcount > 0

    async def needing_summary(
        self, *, threshold: int, limit: int = 10
    ) -> List[Conversation]:
        """Conversations with enough unsummarised turns to be worth condensing."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(ConversationORM)
                .where(
                    ConversationORM.turn_count - ConversationORM.summary_through_seq
                    >= threshold
                )
                .order_by(desc(ConversationORM.last_active_at))
                .limit(limit)
            )
            return [_to_conversation(row) for row in result.scalars().all()]


# Singleton instance
conversation_repository = ConversationRepository()
