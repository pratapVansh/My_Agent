"""
Relational memory store (PostgreSQL).

Holds what the assistant remembers about the user: conversation history,
explicit profile facts, episodic turn summaries, and tool-use outcomes.

Application records — attendance, timetable, exams, plans, job bookmarks, email
drafts and templates — moved to `app/domain/`. They are data the app stores,
not things the assistant remembers, and keeping them here is what grew this
class to 1,145 lines across twelve unrelated entity types.

The class name is retained for now because Phase 1 of the memory redesign
replaces this module wholesale with typed `MemoryRecord` repositories; renaming
it in the interim would be churn on code that is about to be deleted. Note that
nothing here is actually short-term — Postgres never expires any of it.
See docs/MEMORY_ARCHITECTURE.md.
"""
from sqlalchemy import select, and_, desc
from sqlalchemy.sql import func
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import re
import uuid

from app.db.session import async_session_maker, dispose_engine, engine, init_db
from app.memory.models import ChatHistory, UserProfile, EpisodicMemory, ToolMemory

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """Conversation history, profile facts, episodes, and tool memory."""

    def __init__(self):
        # The engine and session factory are process-wide (app/db/session.py):
        # the voice worker shares this process, so a second pool would compete
        # for the same Postgres connection limit.
        self.engine = engine
        self.async_session_maker = async_session_maker

    async def init_db(self):
        """Create database tables if they don't exist."""
        await init_db()

    async def close(self):
        """Dispose the shared engine. Owned by the application lifespan."""
        await dispose_engine()

    # Chat History Operations
    async def store_chat_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a chat message.

        Args:
            user_id: User identifier
            session_id: Conversation session ID
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Additional metadata

        Returns:
            Message ID
        """
        async with self.async_session_maker() as session:
            message = ChatHistory(
                user_id=user_id,
                session_id=session_id,
                role=role,
                content=content,
                meta_data=metadata  # Uses meta_data column
            )
            session.add(message)
            await session.commit()
            return str(message.id)

    async def retrieve_chat_history(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a user/session.

        Args:
            user_id: User identifier
            session_id: Optional session filter
            limit: Maximum number of messages

        Returns:
            List of chat messages
        """
        async with self.async_session_maker() as session:
            query = select(ChatHistory).where(ChatHistory.user_id == user_id)

            if session_id:
                query = query.where(ChatHistory.session_id == session_id)

            query = query.order_by(desc(ChatHistory.created_at)).limit(limit)

            result = await session.execute(query)
            messages = result.scalars().all()

            return [
                {
                    "id": str(msg.id),
                    "role": msg.role,
                    "content": msg.content,
                    "metadata": msg.meta_data,  # Read from meta_data column
                    "created_at": msg.created_at.isoformat()
                }
                for msg in reversed(messages)  # Return in chronological order
            ]

    async def get_recent_context(
        self,
        user_id: str,
        session_id: str,
        last_n: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation context for LLM injection.

        Args:
            user_id: User identifier
            session_id: Session identifier
            last_n: Number of recent messages

        Returns:
            List of messages in format [{"role": "...", "content": "..."}]
        """
        messages = await self.retrieve_chat_history(user_id, session_id, last_n)
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

    # ── UserProfile Operations ───────────────────────────────────────────────

    # ── Sensitive data detection ──────────────────────────────────────────────
    # Profile facts are injected verbatim into every agent's system prompt, so
    # anything stored here reaches an LLM (and any transcript or trace of it).
    # These checks are defence in depth, not a guarantee — the durable fix is an
    # allowlist of permitted fact keys (see AUDIT_REPORT.md, M5).
    _SENSITIVE_KEYS = frozenset({
        "password", "passwd", "pwd", "pass", "passphrase", "pin", "secret",
        "token", "auth", "credential", "credentials", "api_key", "apikey",
        "access_key", "secret_key", "private_key", "session_key", "otp", "mfa",
        "credit_card", "card_number", "cardno", "cvv", "cvc", "ssn",
        "social_security", "bank_account", "account_number", "routing_number",
        "iban", "aadhaar", "passport", "license_key",
    })

    # Credential formats matched against the value exactly as given. These all
    # describe single opaque tokens, so a value containing whitespace can never
    # match one — that is what keeps ordinary prose facts out of scope.
    _SENSITIVE_TOKEN_PATTERNS = (
        re.compile(r"^[A-Za-z0-9_\-\.]{20,}$"),                           # opaque token
        re.compile(r"^(sk|pk|rk|api|key)[-_][A-Za-z0-9_\-]{10,}$", re.I),  # prefixed API key
        re.compile(r"^[A-Za-z0-9+/]{24,}={0,2}$"),                        # base64 blob
        re.compile(r"^eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\."),            # JWT
    )

    # Numeric identifiers, checked after stripping spaces and hyphens so that
    # "4111 1111 1111 1111" and "123-45-6789" are still recognised.
    _SENSITIVE_NUMERIC_PATTERNS = (
        re.compile(r"^\d{13,19}$"),      # card-like PAN
        re.compile(r"^\d{9}$"),          # SSN without separators
    )

    # Substrings that mark a value as a key blob wherever they appear.
    _SENSITIVE_SUBSTRINGS = (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    )

    def _is_sensitive(self, key: str, value: str) -> bool:
        """Return True if the key or value looks like sensitive credential data."""
        # Normalise separators so "user-password" and "My Password" are both
        # caught by the same substring check as "user_password".
        key_normalized = re.sub(r"[^a-z0-9]+", "_", (key or "").lower())
        if any(s in key_normalized for s in self._SENSITIVE_KEYS):
            return True

        candidate = (value or "").strip()
        if not candidate:
            return False

        for pattern in self._SENSITIVE_SUBSTRINGS:
            if pattern.search(candidate):
                return True

        # URLs and email addresses are legitimate profile values.
        if candidate.lower().startswith(("http://", "https://")) or "@" in candidate:
            return False

        # Only single-token values can be credentials; prose never is.
        if not re.search(r"\s", candidate):
            for pattern in self._SENSITIVE_TOKEN_PATTERNS:
                if pattern.match(candidate):
                    return True

        digits_only = re.sub(r"[\s\-]", "", candidate)
        if digits_only.isdigit():
            for pattern in self._SENSITIVE_NUMERIC_PATTERNS:
                if pattern.match(digits_only):
                    return True

        return False

    async def save_profile_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        source: str = "explicit",
        confidence: float = 1.0,
        consent_level: str = "explicit",
    ) -> str:
        """
        Upsert a profile fact. If (user_id, key) already exists, update value + metadata.
        Returns the record ID. Blocks sensitive credential data silently.
        """
        if self._is_sensitive(key, value):
            # Log the key only — never the value that triggered the block.
            logger.warning(
                "Rejected profile fact for user=%s key=%s: value looks like credential data",
                user_id, key,
            )
            return ""

        from sqlalchemy.dialects.postgresql import insert as pg_insert
        async with self.async_session_maker() as session:
            stmt = pg_insert(UserProfile).values(
                id=uuid.uuid4(),
                user_id=user_id,
                key=key,
                value=value,
                source=source,
                confidence=confidence,
                consent_level=consent_level,
            ).on_conflict_do_update(
                constraint="uq_user_profile_key",
                set_=dict(
                    value=value,
                    source=source,
                    confidence=confidence,
                    consent_level=consent_level,
                    updated_at=func.now(),
                ),
            ).returning(UserProfile.id)
            result = await session.execute(stmt)
            await session.commit()
            row = result.fetchone()
            return str(row[0]) if row else ""

    async def get_profile_facts(
        self,
        user_id: str,
        key: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Return all profile facts for a user, optionally filtered by key."""
        async with self.async_session_maker() as session:
            query = select(UserProfile).where(
                and_(
                    UserProfile.user_id == user_id,
                    UserProfile.confidence >= min_confidence,
                )
            )
            if key:
                query = query.where(UserProfile.key == key)
            query = query.order_by(UserProfile.updated_at.desc())
            result = await session.execute(query)
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "key": r.key,
                    "value": r.value,
                    "source": r.source,
                    "confidence": r.confidence,
                    "consent_level": r.consent_level,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in rows
            ]

    async def forget_profile_fact(self, user_id: str, key: str) -> bool:
        """Delete a single profile fact. Returns True if a row was deleted."""
        from sqlalchemy import delete
        async with self.async_session_maker() as session:
            result = await session.execute(
                delete(UserProfile).where(
                    and_(UserProfile.user_id == user_id, UserProfile.key == key)
                )
            )
            await session.commit()
            return result.rowcount > 0

    async def forget_all_profile(self, user_id: str) -> int:
        """Delete all profile facts for a user. Returns number of rows deleted."""
        from sqlalchemy import delete
        async with self.async_session_maker() as session:
            result = await session.execute(
                delete(UserProfile).where(UserProfile.user_id == user_id)
            )
            await session.commit()
            return result.rowcount

    # ── EpisodicMemory Operations ────────────────────────────────────────────

    async def store_episode(
        self,
        user_id: str,
        session_id: str,
        user_summary: str,
        agent_summary: str,
        agent_used: Optional[str] = None,
        intent: Optional[str] = None,
        outcome: str = "success",
    ) -> str:
        """Store a single conversation turn as an episodic memory."""
        async with self.async_session_maker() as session:
            ep = EpisodicMemory(
                user_id=user_id,
                session_id=session_id,
                user_summary=user_summary[:300],
                agent_summary=agent_summary[:300],
                agent_used=agent_used,
                intent=intent,
                outcome=outcome,
            )
            session.add(ep)
            await session.commit()
            return str(ep.id)

    async def get_recent_episodes(
        self,
        user_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the N most recent episodes for a user (newest first)."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(EpisodicMemory)
                .where(EpisodicMemory.user_id == user_id)
                .order_by(desc(EpisodicMemory.created_at))
                .limit(limit)
            )
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "session_id": r.session_id,
                    "user_summary": r.user_summary,
                    "agent_summary": r.agent_summary,
                    "agent_used": r.agent_used,
                    "intent": r.intent,
                    "outcome": r.outcome,
                    "created_at": r.created_at.isoformat(),
                }
                for r in reversed(rows)  # chronological order
            ]

    # ── Tool Memory Operations (Fix 3: Cross-Session Tool Learning) ─────────

    async def save_tool_memory(
        self,
        user_id: str,
        agent_name: str,
        tool_name: str,
        inputs_summary: str,
        outcome_quality: str,
        key_insight: str,
    ) -> str:
        """
        Persist a tool-use outcome so the agent can recall what worked.
        outcome_quality: "good" | "poor" | "failed"
        Only 'good' records are retrieved as future hints.
        """
        async with self.async_session_maker() as session:
            record = ToolMemory(
                user_id=user_id,
                agent_name=agent_name,
                tool_name=tool_name,
                inputs_summary=inputs_summary[:500],
                outcome_quality=outcome_quality,
                key_insight=key_insight[:500],
            )
            session.add(record)
            await session.commit()
            return str(record.id)

    async def get_tool_memories(
        self,
        user_id: str,
        agent_name: str,
        tool_names: List[str],
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve past successful tool-use records for the given tools.
        Returns newest-first, filtered to outcome_quality='good' only.
        """
        if not tool_names:
            return []
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(ToolMemory)
                .where(
                    and_(
                        ToolMemory.user_id == user_id,
                        ToolMemory.agent_name == agent_name,
                        ToolMemory.tool_name.in_(tool_names),
                        ToolMemory.outcome_quality == "good",
                    )
                )
                .order_by(desc(ToolMemory.created_at))
                .limit(limit)
            )
            rows = result.scalars().all()
            return [
                {
                    "tool_name": r.tool_name,
                    "inputs_summary": r.inputs_summary,
                    "key_insight": r.key_insight,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    # ── Turn counter ────────────────────────────────────────────────────────

    async def get_session_turn_count(self, user_id: str, session_id: str) -> int:
        """Count how many user messages exist in this session (= number of turns taken)."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(func.count(ChatHistory.id)).where(
                    and_(
                        ChatHistory.user_id == user_id,
                        ChatHistory.session_id == session_id,
                        ChatHistory.role == "user",
                    )
                )
            )
            return result.scalar() or 0

# Singleton instance
short_term_memory = ShortTermMemory()
