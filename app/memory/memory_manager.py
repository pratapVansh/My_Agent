"""
Unified Memory Manager.
Orchestrates long-term, short-term, and smart memory systems.
Provides clean interface for storage and retrieval.
"""
from typing import Dict, Any, List, Optional
from datetime import date, time

from app.memory.long_term_memory_qdrant import long_term_memory_qdrant
from app.memory.short_term_memory import short_term_memory
from app.memory.smart_memory import smart_memory
from app.memory.memory_cache import memory_cache


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
            # Silent fail - don't block on smart memory errors
            print(f"Smart memory extraction failed: {e}")

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

    async def retrieve_context(
        self,
        user_id: str,
        session_id: str,
        query: Optional[str] = None
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
            # Merge with fresh chat history (always needs to be current)
            cached["chat_history"] = await self.short_term.get_recent_context(
                user_id=user_id,
                session_id=session_id,
                last_n=10
            )
            return cached

        # Get recent chat history (short-term)
        chat_context = await self.short_term.get_recent_context(
            user_id=user_id,
            session_id=session_id,
            last_n=10
        )

        # Get user preferences/interests (smart memory)
        preferences = await self.smart.retrieve_preferences(
            user_id=user_id,
            query=query,
            limit=5
        )

        # Get relevant long-term info if query provided
        long_term_context = {}
        if query:
            long_term_context = await self.long_term.search_all(
                user_id=user_id,
                query=query,
                limit=3
            )

        context = {
            "chat_history": chat_context,
            "preferences": preferences,
            "long_term": long_term_context
        }

        # Cache the result (except chat_history which changes frequently)
        cache_data = {
            "chat_history": [],  # Will be merged when retrieved from cache
            "preferences": preferences,
            "long_term": long_term_context
        }
        memory_cache.set(user_id, cache_data, query)

        return context

    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format memory context into a clean prompt injection.

        Args:
            context: Context from retrieve_context()

        Returns:
            Formatted context string for LLM prompt
        """
        parts = []

        # Add user preferences/interests
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

        # Add relevant long-term info
        long_term = context.get("long_term", {})

        # Resume
        resume = long_term.get("resume", {})
        if resume and resume.get("content"):
            parts.append(f"User Resume:\n{resume['content'][:500]}")  # Limit length

        # Skills
        skills = long_term.get("skills", [])
        if skills:
            skill_list = [s.get("content", "") for s in skills[:5]]
            if skill_list:
                parts.append(f"User Skills:\n" + "\n".join([f"- {s}" for s in skill_list]))

        # Projects
        projects = long_term.get("projects", [])
        if projects:
            project_list = [p.get("content", "") for p in projects[:3]]
            if project_list:
                parts.append("User Projects:\n" + "\n".join([f"- {p[:200]}" for p in project_list]))

        # Combine all parts
        if parts:
            return "\n\n".join(parts)
        return ""

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

    async def cleanup(self):
        """Cleanup resources."""
        await self.short_term.close()


# Singleton instance
memory_manager = MemoryManager()
