"""
Conversational vector memory.

Stores user messages and assistant responses as embedded points in Qdrant so
past turns can be recalled semantically.

Historical note: this module was originally built around mem0, which is why it
is still named "smart memory". The mem0 object was constructed but never called
— every read and write went directly to Qdrant + Cohere, and mem0's own
configured embedder (MiniLM, 384-dim) never matched the vectors actually stored
(Cohere, 1024-dim). The dead integration was removed rather than left to mislead
readers about where embeddings come from.

This class is superseded by the extraction pipeline described in
docs/MEMORY_ARCHITECTURE.md §3.5: writing every raw utterance is a noise
amplifier that degrades retrieval as history grows. It remains in place until
Phase 3 replaces it.
"""
from typing import List, Dict, Any, Optional
import logging
import uuid
from qdrant_client.models import PointStruct
from app.services.qdrant_service import qdrant_service
from app.services.cohere_service import cohere_service
from app.services.debug_logger import log_step

logger = logging.getLogger(__name__)


class SmartMemory:
    """Semantic store for conversational turns, backed by Qdrant + Cohere."""

    def __init__(self):
        self.qdrant = qdrant_service
        self.cohere = cohere_service
        self.collection_name = "smart_memory_chunks"

    async def initialize(self):
        """Initialize smart memory collection in Qdrant."""
        await self.qdrant.ensure_collection(self.collection_name)

    async def store_memory(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Store assistant response as retrievable long-term memory."""
        if not text.strip():
            return None

        try:
            log_step("EMBEDDING DONE", {"input_type": "search_document", "target": "memory"})
            embedding = await self.cohere.embed_text(
                text=text,
                input_type="search_document",
            )

            point_id = str(uuid.uuid4())
            payload = {
                "user_id": user_id,
                "type": "memory",
                "text": text,
            }
            if metadata:
                payload.update(metadata)

            await self.qdrant.upsert_points(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )

            log_step(
                "MEMORY UPSERT",
                {
                    "collection": self.collection_name,
                    "point_id": point_id,
                    "user_id": user_id,
                    "type": "memory",
                },
            )
            return point_id
        except Exception as e:
            # Surfaced at error level (not swallowed silently): a failure here
            # means this turn is permanently absent from semantic memory while
            # the Postgres copy succeeded, so operators need to see the drift.
            logger.error(
                "Smart memory upsert failed for user=%s (%d chars): %s",
                user_id, len(text), e, exc_info=True,
            )
            return None

    async def extract_and_store(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Extract preferences/interests from conversation and store.

        Args:
            user_id: User identifier
            messages: Conversation messages [{"role": "...", "content": "..."}]
            metadata: Additional metadata

        Returns:
            List of extracted memory IDs
        """
        try:
            memory_ids: List[str] = []
            for message in messages:
                content = (message or {}).get("content", "").strip()
                role = (message or {}).get("role", "user")
                if not content:
                    continue
                memory_id = await self.store_memory(
                    user_id=user_id,
                    text=content,
                    metadata={**(metadata or {}), "role": role},
                )
                if memory_id:
                    memory_ids.append(memory_id)
            return memory_ids

        except Exception as e:
            # Non-fatal for the turn, but must stay visible in logs.
            logger.error(
                "Smart memory extraction failed for user=%s: %s", user_id, e, exc_info=True
            )
            return []

    async def add_preference(
        self,
        user_id: str,
        preference: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Manually add a user preference.

        Args:
            user_id: User identifier
            preference: Preference statement
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        try:
            return await self.store_memory(
                user_id=user_id,
                text=preference,
                metadata={**(metadata or {}), "kind": "preference"},
            )

        except Exception as e:
            logger.error("Smart memory add failed for user=%s: %s", user_id, e, exc_info=True)
            return None

    async def retrieve_preferences(
        self,
        user_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve user preferences and interests.

        Args:
            user_id: User identifier
            query: Optional search query for semantic matching
            limit: Maximum number of results

        Returns:
            List of preferences/interests
        """
        try:
            if not query:
                # Return latest stored memory points when no semantic query is provided.
                points = await self.qdrant.scroll_collection(
                    collection_name=self.collection_name,
                    filter_conditions={"user_id": user_id, "type": "memory"},
                    limit=limit,
                )

                formatted = [
                    {
                        "memory": p.get("payload", {}).get("text", ""),
                        "metadata": p.get("payload", {}),
                    }
                    for p in points
                    if p.get("payload", {}).get("text")
                ]
                if not formatted:
                    return []
                return formatted

            query_embedding = await self.cohere.embed_text(
                text=query,
                input_type="search_query",
            )
            log_step("EMBEDDING DONE", {"input_type": "search_query", "target": "memory"})

            results = await self.qdrant.query_points(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                filter_conditions={"user_id": user_id, "type": "memory"},
            )

            return [
                {
                    "memory": result.payload.get("text", ""),
                    "score": result.score,
                    "metadata": result.payload,
                }
                for result in results
                if result.payload.get("text")
            ]

        except Exception as e:
            logger.error(
                "Smart memory retrieval failed for user=%s: %s", user_id, e, exc_info=True
            )
            return []

    async def get_summary(self, user_id: str) -> str:
        """
        Get a summary of user preferences and interests.

        Args:
            user_id: User identifier

        Returns:
            Text summary of user profile
        """
        try:
            preferences = await self.retrieve_preferences(user_id, limit=20)

            if not preferences:
                return ""

            # Build summary from memories
            summary_parts = []
            for pref in preferences:
                if isinstance(pref, dict):
                    memory_text = pref.get("memory", pref.get("text", ""))
                    if memory_text:
                        summary_parts.append(f"- {memory_text}")

            return "\n".join(summary_parts) if summary_parts else ""

        except Exception as e:
            logger.error("Smart memory summary failed for user=%s: %s", user_id, e, exc_info=True)
            return ""

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """
        Delete a specific memory point by ID, verifying it belongs to this user.

        Args:
            user_id: User identifier (ownership check)
            memory_id: Qdrant point ID to delete

        Returns:
            True if deleted, False if not found or error
        """
        try:
            # Scroll all points for this user and check ownership before deleting.
            # This prevents one user from deleting another user's memories.
            # Ownership check walks every point for this user (the scroll cursor
            # is followed to exhaustion), so a memory beyond the first page is
            # still recognised as owned rather than reported "not found".
            points = await self.qdrant.scroll_collection(
                collection_name=self.collection_name,
                filter_conditions={"user_id": user_id},
            )
            owned_ids = {p["id"] for p in points}
            if memory_id not in owned_ids:
                logger.warning(
                    "delete_memory: memory_id=%s not found for user=%s", memory_id, user_id
                )
                return False

            await self.qdrant.delete_points(
                collection_name=self.collection_name,
                point_ids=[memory_id],
            )
            logger.info("delete_memory: deleted memory_id=%s for user=%s", memory_id, user_id)
            return True
        except Exception as e:
            logger.error("Smart memory delete error for user=%s: %s", user_id, e)
            return False

    async def reset_user_memories(self, user_id: str) -> bool:
        """
        Delete ALL memory points for a user (e.g. GDPR right-to-erasure).

        Args:
            user_id: User identifier

        Returns:
            True if all points deleted (or none existed), False on error
        """
        try:
            # Delete server-side by filter rather than scroll-then-delete-by-id.
            # A scroll-based erasure is bounded by however many points one walk
            # returns, which can silently leave data behind on a right-to-erasure
            # request; a filtered delete has no such ceiling.
            await self.qdrant.delete_by_filter(
                collection_name=self.collection_name,
                filter_conditions={"user_id": user_id},
            )
            logger.info("reset_user_memories: erased all memory points for user=%s", user_id)
            return True
        except Exception as e:
            logger.error("Smart memory reset error for user=%s: %s", user_id, e, exc_info=True)
            return False


# Singleton instance
smart_memory = SmartMemory()
