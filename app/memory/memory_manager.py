"""
Unified Memory Manager.
Orchestrates long-term, short-term, and smart memory systems.
Provides clean interface for storage and retrieval.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import date, time
import asyncio

from app.memory.long_term_memory_qdrant import long_term_memory_qdrant

logger = logging.getLogger(__name__)
from app.memory.short_term_memory import short_term_memory
from app.memory.smart_memory import smart_memory
from app.memory.memory_cache import memory_cache
from app.memory import identity, write_policy
from app.memory.retrieval_result import RetrievalResult
from app.memory.sources import QueryCategory
from app.memory.writer import memory_writer
from app.config import settings
from app.services.debug_logger import log_step


# Strong references to detached shadow tasks. asyncio holds only a weak
# reference to a running task, so without this a fire-and-forget task can be
# garbage-collected mid-execution and its exception never observed.
_shadow_tasks: set = set()


def _spawn_shadow(coro, description: str) -> None:
    """Run a non-critical coroutine detached, keeping it alive and logged."""
    task = asyncio.create_task(coro, name=description)
    _shadow_tasks.add(task)

    def _done(finished: asyncio.Task) -> None:
        _shadow_tasks.discard(finished)
        try:
            finished.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("Shadow task '%s' failed: %s", description, exc)

    task.add_done_callback(_done)


def _count_tokens(text: str) -> int:
    """Token count for comparison logging; never worth failing a turn over."""
    try:
        from app.services.chunking_service import chunking_service
        return chunking_service.count_tokens(text)
    except Exception:
        return max(0, len(text) // 4)


class MemoryManager:
    """
    Unified memory manager for hybrid memory system.

    Flow:
    1. On user input → save to short-term + extract to smart memory
    2. Before agent → retrieve relevant memories from all systems
    3. Inject into agent prompt
    """

    def __init__(self):
        """Initialize memory systems."""
        self.long_term = long_term_memory_qdrant
        self.short_term = short_term_memory
        self.smart = smart_memory

    async def initialize(self):
        """Initialize database connections and vector collections."""
        await self.short_term.init_db()
        await self.long_term.initialize()
        await self.smart.initialize()

        # The unified memory collection. Created here so the API process never
        # has to create it lazily mid-request; the worker also ensures it
        # itself, since it can run without ever touching this code path.
        try:
            from app.memory.stores.qdrant_vector_store import qdrant_vector_store
            await qdrant_vector_store.initialize()
        except Exception as exc:
            logger.warning("Could not ensure the memory_records collection: %s", exc)

    async def on_user_input(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Process user input - save to short-term and extract to smart memory.

        Args:
            user_id: User identifier
            session_id: Session identifier
            user_message: User's message
            metadata: Additional metadata
        """
        # Store in short-term (chat history)
        await self.short_term.store_chat_message(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=user_message,
            metadata=metadata
        )

        # Phase 4: mirror into the conversation thread. `session_id` is the
        # conversation's key, so voice and text land in the same thread.
        await self._append_turn(
            user_id, session_id, "user", user_message, metadata
        )

        # Extract preferences/interests to smart memory (async, non-blocking).
        #
        # Gated. Every user utterance used to be embedded verbatim, so "Okay.",
        # "You are fool." and "Once in a million chat." became long-term
        # memories ranked alongside the user's degree. The cost is not storage,
        # it is recall: each fragment is a competing neighbour in embedding
        # space, so retrieval for real questions degrades as history grows.
        #
        # The gate is deterministic and errs towards *not* storing: a durable
        # fact missed today is usually restated, while a store full of "okay"
        # cannot be cleaned up by any later pass.
        decision = write_policy.should_store(user_message, role="user")
        log_step("memory_write", {
            "user_id": user_id,
            "chars": len(user_message or ""),
            **decision.summary(),
        })
        if not decision.store:
            return

        try:
            await self.smart.extract_and_store(
                user_id=user_id,
                messages=[{"role": "user", "content": user_message}],
                metadata={
                    **(metadata or {}),
                    "importance": decision.importance,
                    "explicit": decision.explicit,
                },
            )
        except Exception as e:
            logger.warning("Smart memory extraction failed: %s", e)

    async def on_agent_response(
        self,
        user_id: str,
        session_id: str,
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Process agent response - save to short-term.

        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_response: Agent's response
            metadata: Additional metadata (agent name, intent, etc.)
        """
        await self.short_term.store_chat_message(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=agent_response,
            metadata=metadata
        )

        # Phase 4: mirror into the conversation thread.
        await self._append_turn(
            user_id, session_id, "assistant", agent_response, metadata
        )

        # Phase 3: enqueue the exchange for asynchronous extraction. This is
        # the write that matters going forward — a worker decides what was
        # worth remembering, rather than the whole utterance being embedded
        # verbatim.
        user_text = await self._last_user_message(user_id, session_id)
        await self._enqueue_turn(
            user_id, session_id, agent_response, metadata, user_text=user_text
        )

        # Legacy per-utterance vector write.
        #
        # Still on by default because `retrieve_context` reads this data for
        # the "User Preferences & Interests" section of the *served* prompt.
        # Disabling it before the Phase 6 read cutover would remove a live
        # signal while its replacement is still only running in shadow.
        if settings.memory_v2_replace_raw_embedding:
            return

        # Gated by what the *user* contributed, not by the reply's length.
        # Storing the assistant's own prose after "okay" or "you are fool"
        # closes a loop where the model's guesses become the evidence for its
        # next guess — the reply is recalled later as though the user had said
        # it. Substantive exchanges still reach the legacy path unchanged.
        turn_decision = write_policy.should_store_turn(user_text, agent_response)
        if not turn_decision.store:
            log_step("memory_write", {
                "user_id": user_id, "target": "legacy_vector",
                **turn_decision.summary(),
            })
            return

        # Two independent stores with no shared transaction. The Postgres write
        # above is authoritative for conversation history; if the vector write
        # fails, that turn is missing from semantic recall while chat history
        # still has it. store_memory() returns None on failure (and logs), so
        # surface the divergence here too rather than letting it pass silently.
        point_id = await self.smart.store_memory(
            user_id=user_id,
            text=agent_response,
            metadata=metadata,
        )
        if point_id is None:
            logger.error(
                "Vector memory write failed for user=%s session=%s — chat history was "
                "saved but this turn is absent from semantic memory.",
                user_id, session_id,
            )
        else:
            log_step(
                "MEMORY UPSERT",
                {"user_id": user_id, "type": "memory", "chars": len(agent_response)},
            )

    async def _append_turn(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mirror a message into its conversation thread.

        Best-effort, like every other Phase 1–4 mirror: `chat_history` remains
        the authoritative read path until the Phase 6 cutover, so a failure
        here costs thread continuity, never the turn itself.

        Modality is derived from metadata rather than the transport: the voice
        worker and the text API both reach this method, and a thread that mixes
        them is a feature — one conversation across devices, not one per
        transport.
        """
        if not settings.memory_v2_dual_write:
            return

        meta = metadata or {}
        modality = "voice" if meta.get("streaming") or meta.get("voice") else "text"

        try:
            from app.memory.conversations import conversation_repository

            await conversation_repository.append_turn(
                conversation_id=session_id,
                owner_id=user_id,
                role=role,
                content=content,
                modality=modality,
                agent=meta.get("agent"),
                intent=meta.get("intent"),
            )
        except Exception as exc:
            logger.warning(
                "Could not append turn to conversation=%s for user=%s: %s",
                session_id, user_id, exc,
            )

    async def _last_user_message(self, user_id: str, session_id: str) -> str:
        """
        The user's half of the turn just answered.

        Read back from chat history rather than threaded through, because the
        two halves are written by different code paths — streaming, voice and
        the tool-calling workflow all differ here — and only the store has both.
        """
        try:
            recent = await self.short_term.get_recent_context(
                user_id=user_id, session_id=session_id, last_n=2
            )
            return next(
                (m["content"] for m in reversed(recent) if m.get("role") == "user"),
                "",
            )
        except Exception as exc:
            logger.warning(
                "Could not read back the user turn for user=%s session=%s: %s",
                user_id, session_id, exc,
            )
            return ""

    async def _enqueue_turn(
        self,
        user_id: str,
        session_id: str,
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_text: Optional[str] = None,
    ) -> None:
        """
        Record one exchange in the extraction outbox.

        Filler exchanges are dropped here rather than queued: the extractor is
        an LLM call per batch, and a batch of "okay"/"thanks" costs a request to
        be told, correctly, that there is nothing to extract. The gate is looser
        than the one on the request path — the extractor can read nuance the
        pattern matcher cannot, so anything substantive still reaches it.

        Enqueueing is best-effort by design: an outbox failure must not fail a
        turn the user already received an answer to.
        """
        if not settings.memory_v2_dual_write:
            return

        try:
            if user_text is None:
                user_text = await self._last_user_message(user_id, session_id)

            if not write_policy.should_store_turn(user_text, agent_response).store:
                return

            from app.memory.events import EventType, MemoryEvent
            from app.memory.stores.postgres_event_queue import postgres_event_queue

            await postgres_event_queue.enqueue(MemoryEvent(
                owner_id=user_id,
                event_type=EventType.TURN,
                group_key=session_id,
                payload={
                    "user": user_text,
                    "assistant": agent_response,
                    "agent": (metadata or {}).get("agent"),
                    "intent": (metadata or {}).get("intent"),
                },
            ))
        except Exception as exc:
            logger.warning(
                "Could not enqueue memory event for user=%s session=%s: %s",
                user_id, session_id, exc,
            )

    async def retrieve_context(
        self,
        user_id: str,
        session_id: str,
        query: Optional[str] = None,
        include_episodes: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memory context for agent injection.
        Uses caching to reduce latency.

        Args:
            user_id: User identifier
            session_id: Session identifier
            query: Optional query for semantic search

        Returns:
            Dictionary with all relevant memories
        """
        # Try cache first. `get` returns a copy, so overwriting the volatile
        # sections below cannot corrupt the entry for the next reader — it used
        # to hand back its own storage, and every hit mutated it.
        cached = memory_cache.get(user_id, query)
        if cached:
            # Always fetch fresh chat history and profile facts — never stale
            cached["chat_history"] = await self.short_term.get_recent_context(
                user_id=user_id,
                session_id=session_id,
                last_n=10
            )
            cached["profile_facts"] = await self.short_term.get_profile_facts(user_id=user_id)
            return cached

        # Retrieve all memory sources in parallel
        async def _get_chat():
            return await self.short_term.get_recent_context(
                user_id=user_id, session_id=session_id, last_n=10
            )

        async def _get_prefs():
            return await self.smart.retrieve_preferences(
                user_id=user_id, query=query, limit=5
            )

        async def _get_long_term():
            if query:
                log_step("USER QUERY", {"user_id": user_id, "query": query})
                return await self.long_term.search_all(
                    user_id=user_id, query=query, limit=3
                )
            return {}

        async def _get_profile():
            return await self.short_term.get_profile_facts(user_id=user_id)

        async def _get_episodes():
            if include_episodes:
                return await self.short_term.get_recent_episodes(user_id=user_id, limit=4)
            return []

        chat_context, preferences, long_term_context, profile_facts, episodes = await asyncio.gather(
            _get_chat(), _get_prefs(), _get_long_term(), _get_profile(), _get_episodes()
        )

        context = {
            "chat_history": chat_context,
            "preferences": preferences,
            "long_term": long_term_context,
            "profile_facts": profile_facts,
            "episodes": episodes,
        }

        # Cache the result (profile_facts and episodes refreshed every call, chat too)
        cache_data = {
            "chat_history": [],
            "preferences": preferences,
            "long_term": long_term_context,
            "profile_facts": profile_facts,
            "episodes": episodes,
        }
        memory_cache.set(user_id, cache_data, query)

        return context

    async def build_memory_prompt(
        self,
        user_id: str,
        session_id: str,
        query: Optional[str] = None,
        memory_owner_id: Optional[str] = None,
        visibilities=None,
        category: Optional[str] = None,
    ) -> tuple[Dict[str, Any], str]:
        """
        Retrieve memory and render it into a prompt block.

        The single entry point for building memory context. Both workflow paths
        previously called `retrieve_context` and `format_context_for_prompt`
        separately, which meant two places to keep in step and no single point
        at which the v2 engine could be compared against v1.

        `category` is a `QueryCategory` value; when supplied, only the sources
        that answer that kind of question are rendered. Passing None keeps the
        historical behaviour of rendering everything.

        Returns (context, prompt) so callers keep access to the raw context.
        """
        context = await self.retrieve_context(
            user_id=user_id, session_id=session_id, query=query
        )
        prompt = self.format_context_for_prompt(context, category=category)

        if settings.memory_v2_shadow_read:
            # Detached deliberately. Shadow retrieval exists to gather
            # comparison data, and it must never add latency to a turn or be
            # able to fail one — especially a spoken turn, where the cost of an
            # extra round trip is audible.
            _spawn_shadow(
                self._shadow_compare(
                    user_id, query or "", prompt, session_id,
                    visibilities, memory_owner_id,
                ),
                f"memory-shadow-{user_id}",
            )

        return context, prompt

    async def _shadow_compare(
        self,
        user_id: str,
        query: str,
        legacy_prompt: str,
        conversation_id: str = "",
        visibilities=None,
        memory_owner_id: Optional[str] = None,
    ) -> None:
        """Run v2 retrieval and log how it differs from what was served."""
        from app.memory.retrieval_v2 import retrieve_and_assemble

        # Records come from the scoped owner; the conversation window stays
        # the caller's own thread.
        assembled, trace = await retrieve_and_assemble(
            user_id=memory_owner_id or user_id,
            query=query,
            conversation_id=conversation_id,
            conversation_owner_id=user_id,
            visibilities=visibilities,
        )
        legacy_tokens = _count_tokens(legacy_prompt)

        log_step("MEMORY SHADOW", {
            "user_id": user_id,
            "legacy_chars": len(legacy_prompt),
            "legacy_tokens": legacy_tokens,
            "v2_chars": len(assembled.text),
            "v2_tokens": assembled.used_tokens,
            **trace.summary(),
        })
        logger.info(
            "Memory shadow: legacy=%d tok, v2=%d tok (%d records, %s), "
            "channels=%s, degraded=%s",
            legacy_tokens, assembled.used_tokens, len(assembled.selected),
            trace.selected_by_kind(),
            {c.name: c.candidates for c in trace.channels},
            trace.degraded,
        )

    # Which memory sections answer which kind of question.
    #
    # Dumping every source into every prompt is how a question about the user's
    # CGPA arrived alongside their project list, their skills, four episode
    # summaries and a résumé excerpt — and how the model, given a wall of
    # loosely related text, produced a fluent answer drawn from the wrong part
    # of it. Narrower context is both cheaper and more accurate.
    #
    # A category absent from this map renders everything, which is also what
    # `category=None` does: a new category is under-filtered, never blind.
    _SECTIONS_BY_CATEGORY: Dict[str, frozenset] = {
        QueryCategory.PROFILE_IDENTITY.value: frozenset(
            {"profile_facts", "chat_history", "resume"}),
        QueryCategory.PROFILE_EDUCATION.value: frozenset(
            {"profile_facts", "chat_history", "resume", "status"}),
        QueryCategory.PROFILE_EXPERIENCE.value: frozenset(
            {"profile_facts", "chat_history", "resume", "status"}),
        QueryCategory.PROFILE_ACHIEVEMENTS.value: frozenset(
            {"profile_facts", "chat_history", "resume", "status"}),
        QueryCategory.PROFILE_SKILLS.value: frozenset(
            {"profile_facts", "chat_history", "skills", "resume", "status"}),
        QueryCategory.PROFILE_PROJECTS.value: frozenset(
            {"profile_facts", "chat_history", "projects", "resume", "status"}),
        QueryCategory.DOCUMENT_RESUME.value: frozenset(
            {"profile_facts", "chat_history", "resume", "skills", "projects", "status"}),
        # Current standing reads the same sections as education; what differs is
        # the precedence among them, which `sources.py` decides.
        QueryCategory.ACADEMIC_CURRENT.value: frozenset(
            {"profile_facts", "chat_history", "resume", "status"}),
        # Answered from the recorded provenance of the previous turn. It needs
        # the thread for continuity and nothing else — retrieving the user's
        # facts here would invite an answer about the facts rather than about
        # where the last one came from.
        QueryCategory.PROVENANCE_QUERY.value: frozenset({"chat_history"}),
        QueryCategory.EXPLICIT_MEMORY.value: frozenset(
            {"profile_facts", "chat_history", "episodes"}),
        QueryCategory.CONVERSATION_CURRENT.value: frozenset(
            {"chat_history", "profile_facts"}),
        QueryCategory.CONVERSATION_FOLLOWUP.value: frozenset(
            {"chat_history", "profile_facts", "projects", "skills", "resume"}),
        QueryCategory.EPISODIC_MEMORY.value: frozenset(
            {"episodes", "chat_history", "profile_facts"}),
        # The clock answers these; memory is present only so the assistant does
        # not lose the thread of the conversation it is in.
        QueryCategory.TEMPORAL_CURRENT.value: frozenset({"chat_history"}),
        QueryCategory.SCHEDULE_TEMPORAL.value: frozenset(
            {"chat_history", "profile_facts", "episodes"}),
        # A general question needs the user's stated preferences (tone, length)
        # and the thread — not their résumé.
        QueryCategory.GENERAL_KNOWLEDGE.value: frozenset(
            {"profile_facts", "chat_history", "preferences"}),
        QueryCategory.SMALL_TALK.value: frozenset({"chat_history", "profile_facts"}),
        QueryCategory.ACTION_REQUEST.value: frozenset(
            {"profile_facts", "chat_history", "preferences", "episodes"}),
        QueryCategory.AMBIGUOUS_ACTION.value: frozenset(
            {"chat_history", "profile_facts"}),
    }

    _ALL_SECTIONS = frozenset({
        "profile_facts", "episodes", "chat_history", "preferences",
        "skills", "projects", "status", "resume",
    })

    def sections_for(self, category: Optional[str]) -> frozenset:
        """Which prompt sections this category of question should see."""
        if not category:
            return self._ALL_SECTIONS
        return self._SECTIONS_BY_CATEGORY.get(category, self._ALL_SECTIONS)

    # Categories that are actually *about* the things the user asked to have
    # remembered. Everywhere else those keys are withheld from the prompt.
    _EXPLICIT_MEMORY_CATEGORIES = frozenset({
        QueryCategory.EXPLICIT_MEMORY.value,
        QueryCategory.EXPLICIT_MEMORY_WRITE.value,
        QueryCategory.PROVENANCE_QUERY.value,
    })

    @staticmethod
    def _fact_is_visible(key: str, category: Optional[str]) -> bool:
        """
        Whether one profile fact belongs in the prompt for this question.

        Section filtering is not enough, and the gap between the two is a real
        bug rather than a tidiness concern. `profile_facts` is a single section
        holding both `canonical_name` and `remembered_name` — so narrowing an
        identity question to "profile_facts + résumé" still hands the model a
        line reading `remembered_name: Devasi` directly beneath the canonical
        one. A model given two names and asked for the user's name volunteers
        both, and no amount of prompt instruction reliably stops it, because
        the fact is right there and looks relevant.

        The keys are therefore withheld at assembly. "What is my name?" cannot
        mention a remembered name because the remembered name is not in the
        prompt; "what name did I ask you to remember?" is a category that reads
        them, so it still sees them. Determinism at the source rather than an
        instruction the model may or may not follow.

        An unknown category (None) keeps the historical behaviour of rendering
        everything: new categories are under-filtered, never blind.
        """
        if not category:
            return True
        if category in MemoryManager._EXPLICIT_MEMORY_CATEGORIES:
            return True
        return not identity.is_explicit_memory_key(key)

    def format_context_for_prompt(
        self, context: Dict[str, Any], category: Optional[str] = None
    ) -> str:
        """
        Format memory context into a clean prompt injection.

        `category` narrows the rendered sections to the sources that answer that
        kind of question (see `_SECTIONS_BY_CATEGORY`). None renders everything,
        which is the historical behaviour and remains the default.

        Fix 5 — Priority-ordered sections to prevent silent data loss.
        Sections are written highest-priority first so that if the combined
        string is later truncated (at _MAX_MEMORY_CHARS in base_agent), only
        the lowest-priority tail (resume body) is dropped — never the user's
        explicit preferences, episodes, or recent conversation.

        Priority order (written first → survives truncation):
          1. Profile Facts   — explicit user preferences; tiny, critical
          2. Recent Episodes — cross-session context; small, critical
          3. Recent Chat     — in-session continuity; budgeted at 1200 chars
          4. Smart Prefs     — recalled conversational turns; medium
          5. Skills          — semantic retrieval; medium
          6. Projects        — semantic retrieval; medium
          7. Resume          — large blob; useful but survives partial loss
        """
        parts = []
        allowed = self.sections_for(category)

        # ── 1. Profile Facts (highest priority — explicit user preferences) ──
        profile_facts = context.get("profile_facts", []) if "profile_facts" in allowed else []
        if profile_facts:
            lines = [
                f"- {f['key']}: {f['value']}"
                for f in profile_facts
                if f.get("value") and self._fact_is_visible(str(f.get("key", "")), category)
            ]
            if lines:
                parts.append("User Profile Facts:\n" + "\n".join(lines))

        # ── 2. Recent Episode Summaries (cross-session context) ──────────────
        episodes = context.get("episodes", []) if "episodes" in allowed else []
        if episodes:
            lines = [
                f"- [{e.get('agent_used', '?')}] {e.get('user_summary', '')} → {e.get('agent_summary', '')}"
                for e in episodes
                if e.get("user_summary") or e.get("agent_summary")
            ]
            if lines:
                parts.append("Recent Activity:\n" + "\n".join(lines))

        # ── 3. Recent Conversation (in-session continuity) ───────────────────
        chat_history = context.get("chat_history", []) if "chat_history" in allowed else []
        history_block = self._format_chat_history_for_prompt(chat_history)
        if history_block:
            parts.append(history_block)

        # ── 4. Smart Memory Preferences (inferred from conversations) ────────
        preferences = context.get("preferences", []) if "preferences" in allowed else []
        if preferences:
            pref_text = []
            for pref in preferences:
                if isinstance(pref, dict):
                    memory = pref.get("memory", pref.get("text", ""))
                    if memory:
                        pref_text.append(f"- {memory}")
            if pref_text:
                parts.append("User Preferences & Interests:\n" + "\n".join(pref_text))

        # ── 5–7. Long-term Vector Memory (skills, projects, resume) ──────────
        long_term = context.get("long_term", {})

        # Skills. The isinstance guard is retained because a cached context
        # dictionary written before RetrievalResult existed may still carry the
        # bare "NO_DATA" string in this slot.
        skills = long_term.get("skills", []) if "skills" in allowed else []
        if not isinstance(skills, list):
            skills = []
        if skills:
            skill_list = [s.get("content", "") for s in skills[:5] if isinstance(s, dict)]
            if skill_list:
                parts.append("User Skills:\n" + "\n".join([f"- {s}" for s in skill_list if s]))

        # Projects — same legacy-cache guard as skills above.
        projects = long_term.get("projects", []) if "projects" in allowed else []
        if not isinstance(projects, list):
            projects = []
        if projects:
            project_list = [p.get("content", "") for p in projects[:3] if isinstance(p, dict)]
            if project_list:
                parts.append("User Projects:\n" + "\n".join([f"- {p[:200]}" for p in project_list if p]))

        # Retrieval status hints (must stay with skills/projects for coherence).
        #
        # NO_DATA and ERROR are reported differently on purpose. NO_DATA is a
        # fact — nothing is stored — and can be stated plainly. ERROR means the
        # lookup failed, so asserting "no skills data found" would be false;
        # the model is told the state is unknown instead. Both cases arm the
        # refusal policy, because both are states in which a model left to its
        # own devices invents an answer.
        skills_status = long_term.get("skills_status") if "status" in allowed else None
        projects_status = long_term.get("projects_status") if "status" in allowed else None
        if skills_status == "NO_DATA":
            parts.append("Retrieval Status: No skills data found in vector memory.")
        if projects_status == "NO_DATA":
            parts.append("Retrieval Status: No projects data found in vector memory.")
        if skills_status == "ERROR":
            parts.append("Retrieval Status: Skills lookup failed — treat skills as unknown, not absent.")
        if projects_status == "ERROR":
            parts.append("Retrieval Status: Projects lookup failed — treat projects as unknown, not absent.")

        both_absent = skills_status == "NO_DATA" and projects_status == "NO_DATA"
        any_failed = "ERROR" in (skills_status, projects_status)
        if both_absent or any_failed:
            parts.append(
                "Policy: If the user asks about unavailable personal attributes "
                "(for example hobbies or unknown profile facts), reply with: "
                "I don't have information about that."
            )
        if skills_status == "FALLBACK":
            parts.append("Note: Skills retrieved from resume text (dedicated skills collection is empty).")
        if projects_status == "FALLBACK":
            parts.append("Note: Projects retrieved from resume text (dedicated projects collection is empty).")

        # Resume — lowest priority: large, useful but partial loss is acceptable.
        # Capped at 300 chars (was 500) to leave room for other sections.
        resume = long_term.get("resume", {}) if "resume" in allowed else {}
        if resume and resume.get("content"):
            parts.append(f"User Resume:\n{resume['content'][:300]}")

        if parts:
            return "\n\n".join(parts)
        return ""

    def _format_chat_history_for_prompt(
        self,
        chat_history: List[Dict[str, str]],
        max_messages: int = 10,
        max_chars: int = 1200,
        max_message_chars: int = 220,
    ) -> str:
        """
        Format recent chat history for prompt injection.

        Keeps only recent messages and trims content to avoid token overflow.
        Output format:
        User: ...
        Assistant: ...
        """
        if not chat_history:
            return ""

        recent = chat_history[-max_messages:]
        lines: List[str] = []

        for msg in recent:
            role = (msg.get("role") or "user").strip().lower()
            content = (msg.get("content") or "").strip()
            if not content:
                continue

            if len(content) > max_message_chars:
                content = content[: max_message_chars - 3].rstrip() + "..."

            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")

        if not lines:
            return ""

        # Enforce a global character budget by dropping oldest lines first.
        while lines and len("\n".join(lines)) > max_chars:
            lines.pop(0)

        if not lines:
            return ""

        return "Recent Conversation:\n" + "\n".join(lines)

    # ── Cache coherence ─────────────────────────────────────────────────
    #
    # Every write that changes what retrieval would return must drop the cached
    # context, or the assistant keeps answering from the superseded version for
    # the cache's full TTL. Nothing called `invalidate` before: a résumé re-upload
    # served the old résumé for five minutes, and a deleted memory kept being
    # retrieved after the user had erased it — a deletion that silently did not
    # take effect.
    def _invalidate(self, user_id: str, reason: str) -> None:
        try:
            dropped = memory_cache.invalidate(user_id)
            if dropped:
                log_step("memory_cache_invalidated", {
                    "user_id": user_id, "entries": dropped, "reason": reason,
                })
        except Exception as exc:
            # A cache that fails to clear is a correctness problem, but failing
            # the write that already succeeded would be worse.
            logger.error(
                "Could not invalidate memory cache for user=%s after %s: %s",
                user_id, reason, exc,
            )

    # Long-term operations (pass-through)
    async def store_resume(self, user_id: str, resume_text: str, metadata: Optional[Dict] = None) -> str:
        """Store resume in long-term memory."""
        record_id = await self.long_term.store_resume(user_id, resume_text, metadata)
        self._invalidate(user_id, "store_resume")
        return record_id

    async def search_long_term(
        self, user_id: str, query: str, limit: int = 5
    ) -> Dict[str, Any]:
        """Semantic search across resume, skills, and projects collections."""
        return await self.long_term.search_all(user_id=user_id, query=query, limit=limit)

    async def retrieve_skills(
        self, user_id: str, query: Optional[str] = None, limit: int = 10
    ) -> RetrievalResult:
        """Retrieve stored skills from long-term vector memory."""
        return await self.long_term.retrieve_skills(user_id=user_id, query=query, limit=limit)

    async def retrieve_projects(
        self, user_id: str, query: Optional[str] = None, limit: int = 10
    ) -> RetrievalResult:
        """Retrieve stored projects from long-term vector memory."""
        return await self.long_term.retrieve_projects(user_id=user_id, query=query, limit=limit)

    async def retrieve_resume(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the most recently stored resume."""
        return await self.long_term.retrieve_resume(user_id=user_id)

    async def store_skill(self, user_id: str, skill_name: str, skill_level: str, metadata: Optional[Dict] = None) -> str:
        """Store skill in long-term memory."""
        record_id = await self.long_term.store_skill(user_id, skill_name, skill_level, metadata)
        self._invalidate(user_id, "store_skill")
        return record_id

    async def store_project(
        self,
        user_id: str,
        project_name: str,
        description: str,
        technologies: List[str],
        metadata: Optional[Dict] = None
    ) -> str:
        """Store project in long-term memory."""
        record_id = await self.long_term.store_project(
            user_id, project_name, description, technologies, metadata
        )
        self._invalidate(user_id, "store_project")
        return record_id

    # ── Phase 1 dual-write ──────────────────────────────────────────────
    #
    # Structured writes are mirrored into the unified `memory_records` table
    # while the legacy tables remain authoritative for every read. Mirroring is
    # strictly additive, so a failure here must never surface to the caller:
    # the primary write has already succeeded and the turn must not fail
    # because a shadow copy did. The backfill migration is idempotent and can
    # repair anything lost this way.
    async def _mirror(self, description: str, coro) -> None:
        if not settings.memory_v2_dual_write:
            return
        try:
            await coro
        except Exception as exc:
            logger.warning(
                "Memory v2 dual-write failed (%s): %s — legacy write is unaffected; "
                "re-run scripts/migrate_memory_v2.py to reconcile.",
                description, exc,
            )

    # UserProfile pass-throughs
    async def save_profile_fact(
        self, user_id: str, key: str, value: str,
        source: str = "explicit", confidence: float = 1.0, consent_level: str = "explicit"
    ) -> str:
        # M3 write policy: only persist inferred facts when confidence >= 0.8
        if source == "inferred" and confidence < 0.8:
            return ""
        record_id = await self.short_term.save_profile_fact(
            user_id, key, value, source, confidence, consent_level
        )
        # An empty id means the value was rejected as credential-shaped. Never
        # mirror what the sensitivity filter just refused.
        if record_id:
            self._invalidate(user_id, f"save_profile_fact:{key}")
            await self._mirror(
                f"profile_fact:{key}",
                memory_writer.record_profile_fact(
                    owner_id=user_id, key=key, value=value,
                    source=source, confidence=confidence,
                ),
            )
        return record_id

    async def get_profile_facts(self, user_id: str, key: Optional[str] = None) -> List[Dict[str, Any]]:
        return await self.short_term.get_profile_facts(user_id, key)

    async def forget_profile_fact(self, user_id: str, key: str) -> bool:
        deleted = await self.short_term.forget_profile_fact(user_id, key)
        if deleted:
            self._invalidate(user_id, f"forget_profile_fact:{key}")
        return deleted

    async def forget_all_profile(self, user_id: str) -> int:
        count = await self.short_term.forget_all_profile(user_id)
        self._invalidate(user_id, "forget_all_profile")
        return count

    # Turn counter
    async def get_session_turn_count(self, user_id: str, session_id: str) -> int:
        return await self.short_term.get_session_turn_count(user_id, session_id)

    # Tool Memory pass-throughs (Fix 3: Cross-Session Tool Learning)
    async def save_tool_outcome(
        self,
        user_id: str,
        agent_name: str,
        tool_name: str,
        inputs_summary: str,
        outcome_quality: str,
        key_insight: str,
    ) -> str:
        """Persist a tool-use outcome for future hint retrieval."""
        record_id = await self.short_term.save_tool_memory(
            user_id, agent_name, tool_name, inputs_summary, outcome_quality, key_insight
        )
        # Only successful strategies become procedural memory — a failed
        # approach is not knowledge worth reusing.
        if outcome_quality == "good":
            await self._mirror(
                f"tool_outcome:{agent_name}:{tool_name}",
                memory_writer.record_tool_outcome(
                    owner_id=user_id, agent_name=agent_name, tool_name=tool_name,
                    inputs_summary=inputs_summary, key_insight=key_insight,
                ),
            )
        return record_id

    async def get_tool_insights(
        self,
        user_id: str,
        agent_name: str,
        tool_names: list,
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        """Retrieve past successful tool-use records to inject as hints."""
        return await self.short_term.get_tool_memories(user_id, agent_name, tool_names, limit)

    # EpisodicMemory pass-throughs
    async def store_episode(
        self, user_id: str, session_id: str,
        user_summary: str, agent_summary: str,
        agent_used: Optional[str] = None, intent: Optional[str] = None,
        outcome: str = "success",
    ) -> str:
        record_id = await self.short_term.store_episode(
            user_id, session_id, user_summary, agent_summary, agent_used, intent, outcome
        )
        await self._mirror(
            f"episode:{session_id}",
            memory_writer.record_episode(
                owner_id=user_id, session_id=session_id,
                user_summary=user_summary, agent_summary=agent_summary,
                agent_used=agent_used, intent=intent, outcome=outcome,
            ),
        )
        return record_id

    async def get_recent_episodes(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        return await self.short_term.get_recent_episodes(user_id, limit)

    async def cleanup(self):
        """Cleanup resources."""
        await self.short_term.close()


# Singleton instance
memory_manager = MemoryManager()
