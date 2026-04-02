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
from app.services.debug_logger import log_step


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
        """Initialize database connections."""
        await self.short_term.init_db()
        await self.long_term.initialize()
        await self.smart.initialize()

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

        # Extract preferences/interests to smart memory (async, non-blocking)
        try:
            await self.smart.extract_and_store(
                user_id=user_id,
                messages=[{"role": "user", "content": user_message}],
                metadata=metadata
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

        # Persist assistant response as vector memory after every response.
        await self.smart.store_memory(
            user_id=user_id,
            text=agent_response,
            metadata=metadata,
        )
        log_step(
            "MEMORY UPSERT",
            {"user_id": user_id, "type": "memory", "chars": len(agent_response)},
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
        # Try cache first
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

    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format memory context into a clean prompt injection.

        Fix 5 — Priority-ordered sections to prevent silent data loss.
        Sections are written highest-priority first so that if the combined
        string is later truncated (at _MAX_MEMORY_CHARS in base_agent), only
        the lowest-priority tail (resume body) is dropped — never the user's
        explicit preferences, episodes, or recent conversation.

        Priority order (written first → survives truncation):
          1. Profile Facts   — explicit user preferences; tiny, critical
          2. Recent Episodes — cross-session context; small, critical
          3. Recent Chat     — in-session continuity; budgeted at 1200 chars
          4. Smart Prefs     — inferred interests from mem0; medium
          5. Skills          — semantic retrieval; medium
          6. Projects        — semantic retrieval; medium
          7. Resume          — large blob; useful but survives partial loss
        """
        parts = []

        # ── 1. Profile Facts (highest priority — explicit user preferences) ──
        profile_facts = context.get("profile_facts", [])
        if profile_facts:
            lines = [f"- {f['key']}: {f['value']}" for f in profile_facts if f.get("value")]
            if lines:
                parts.append("User Profile Facts:\n" + "\n".join(lines))

        # ── 2. Recent Episode Summaries (cross-session context) ──────────────
        episodes = context.get("episodes", [])
        if episodes:
            lines = [
                f"- [{e.get('agent_used', '?')}] {e.get('user_summary', '')} → {e.get('agent_summary', '')}"
                for e in episodes
                if e.get("user_summary") or e.get("agent_summary")
            ]
            if lines:
                parts.append("Recent Activity:\n" + "\n".join(lines))

        # ── 3. Recent Conversation (in-session continuity) ───────────────────
        chat_history = context.get("chat_history", [])
        history_block = self._format_chat_history_for_prompt(chat_history)
        if history_block:
            parts.append(history_block)

        # ── 4. Smart Memory Preferences (inferred from conversations) ────────
        preferences = context.get("preferences", [])
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

        # Skills — guard against "NO_DATA" sentinel string
        skills = long_term.get("skills", [])
        if not isinstance(skills, list):
            skills = []
        if skills:
            skill_list = [s.get("content", "") for s in skills[:5] if isinstance(s, dict)]
            if skill_list:
                parts.append("User Skills:\n" + "\n".join([f"- {s}" for s in skill_list if s]))

        # Projects — guard against "NO_DATA" sentinel string
        projects = long_term.get("projects", [])
        if not isinstance(projects, list):
            projects = []
        if projects:
            project_list = [p.get("content", "") for p in projects[:3] if isinstance(p, dict)]
            if project_list:
                parts.append("User Projects:\n" + "\n".join([f"- {p[:200]}" for p in project_list if p]))

        # Retrieval status hints (must stay with skills/projects for coherence)
        skills_status = long_term.get("skills_status")
        projects_status = long_term.get("projects_status")
        if skills_status == "NO_DATA":
            parts.append("Retrieval Status: No skills data found in vector memory.")
        if projects_status == "NO_DATA":
            parts.append("Retrieval Status: No projects data found in vector memory.")
        if skills_status == "NO_DATA" and projects_status == "NO_DATA":
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
        resume = long_term.get("resume", {})
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

    # Long-term operations (pass-through)
    async def store_resume(self, user_id: str, resume_text: str, metadata: Optional[Dict] = None) -> str:
        """Store resume in long-term memory."""
        return await self.long_term.store_resume(user_id, resume_text, metadata)

    async def store_skill(self, user_id: str, skill_name: str, skill_level: str, metadata: Optional[Dict] = None) -> str:
        """Store skill in long-term memory."""
        return await self.long_term.store_skill(user_id, skill_name, skill_level, metadata)

    async def store_project(
        self,
        user_id: str,
        project_name: str,
        description: str,
        technologies: List[str],
        metadata: Optional[Dict] = None
    ) -> str:
        """Store project in long-term memory."""
        return await self.long_term.store_project(user_id, project_name, description, technologies, metadata)

    # Short-term operations (pass-through)
    async def store_attendance(
        self,
        user_id: str,
        date: date,
        subject: str,
        status: str,
        notes: Optional[str] = None
    ) -> str:
        """Store attendance record."""
        return await self.short_term.store_attendance(user_id, date, subject, status, notes)

    async def retrieve_attendance(
        self,
        user_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        subject: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve attendance records."""
        return await self.short_term.retrieve_attendance(user_id, start_date, end_date, subject)

    async def store_timetable_entry(
        self,
        user_id: str,
        day_of_week: int,
        start_time: time,
        end_time: time,
        subject: str,
        location: Optional[str] = None,
        instructor: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store timetable entry."""
        return await self.short_term.store_timetable_entry(
            user_id, day_of_week, start_time, end_time, subject, location, instructor, metadata
        )

    async def retrieve_timetable(
        self,
        user_id: str,
        day_of_week: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve timetable entries."""
        return await self.short_term.retrieve_timetable(user_id, day_of_week)

    async def clear_timetable(self, user_id: str) -> int:
        """Soft-delete all active timetable entries. Returns count cleared."""
        return await self.short_term.clear_timetable(user_id)

    # Email draft pass-throughs
    async def save_email_draft(self, user_id: str, subject: str, body: str, **kwargs) -> str:
        return await self.short_term.save_email_draft(user_id, subject, body, **kwargs)

    async def get_email_drafts(self, user_id: str, limit: int = 10, status: str = "draft") -> List[Dict[str, Any]]:
        return await self.short_term.get_email_drafts(user_id, limit, status)

    async def mark_email_sent(self, draft_id: str, user_id: str, sent_to: str) -> bool:
        return await self.short_term.mark_draft_sent(draft_id, user_id, sent_to)

    # Email template pass-throughs
    async def save_email_template(self, user_id: str, name: str, subject_template: str, body_template: str, **kwargs) -> str:
        return await self.short_term.save_email_template(user_id, name, subject_template, body_template, **kwargs)

    async def get_email_templates(self, user_id: str, name: Optional[str] = None) -> List[Dict[str, Any]]:
        return await self.short_term.get_email_templates(user_id, name)

    # Exam pass-throughs
    async def store_exam(self, user_id: str, subject: str, exam_date, **kwargs) -> str:
        return await self.short_term.store_exam(user_id, subject, exam_date, **kwargs)

    async def retrieve_exams(self, user_id: str, upcoming_only: bool = True) -> List[Dict[str, Any]]:
        return await self.short_term.retrieve_exams(user_id, upcoming_only)

    # Plan pass-throughs
    async def store_plan(self, user_id: str, plan_date, title: str, **kwargs) -> str:
        return await self.short_term.store_plan(user_id, plan_date, title, **kwargs)

    async def retrieve_plans(self, user_id: str, plan_date=None, include_done: bool = False) -> List[Dict[str, Any]]:
        return await self.short_term.retrieve_plans(user_id, plan_date, include_done)

    async def mark_plan_done(self, plan_id: str, user_id: str) -> bool:
        return await self.short_term.mark_plan_done(plan_id, user_id)

    # UserProfile pass-throughs
    async def save_profile_fact(
        self, user_id: str, key: str, value: str,
        source: str = "explicit", confidence: float = 1.0, consent_level: str = "explicit"
    ) -> str:
        # M3 write policy: only persist inferred facts when confidence >= 0.8
        if source == "inferred" and confidence < 0.8:
            return ""
        return await self.short_term.save_profile_fact(user_id, key, value, source, confidence, consent_level)

    async def get_profile_facts(self, user_id: str, key: Optional[str] = None) -> List[Dict[str, Any]]:
        return await self.short_term.get_profile_facts(user_id, key)

    async def forget_profile_fact(self, user_id: str, key: str) -> bool:
        return await self.short_term.forget_profile_fact(user_id, key)

    async def forget_all_profile(self, user_id: str) -> int:
        return await self.short_term.forget_all_profile(user_id)

    # Turn counter
    async def get_session_turn_count(self, user_id: str, session_id: str) -> int:
        return await self.short_term.get_session_turn_count(user_id, session_id)

    # M5: Agent Playbook pass-throughs
    async def save_playbook(
        self, agent_name: str, version: str, prompt: str,
        notes: Optional[str] = None, set_active: bool = True
    ) -> str:
        return await self.short_term.save_playbook(agent_name, version, prompt, notes, set_active)

    async def get_active_playbook(self, agent_name: str) -> Optional[Dict]:
        return await self.short_term.get_active_playbook(agent_name)

    async def list_playbooks(self, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return await self.short_term.list_playbooks(agent_name)

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
        return await self.short_term.save_tool_memory(
            user_id, agent_name, tool_name, inputs_summary, outcome_quality, key_insight
        )

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
        return await self.short_term.store_episode(
            user_id, session_id, user_summary, agent_summary, agent_used, intent, outcome
        )

    async def get_recent_episodes(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        return await self.short_term.get_recent_episodes(user_id, limit)

    async def cleanup(self):
        """Cleanup resources."""
        await self.short_term.close()


# Singleton instance
memory_manager = MemoryManager()
