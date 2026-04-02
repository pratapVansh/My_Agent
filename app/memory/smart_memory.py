"""
mem0 Smart Memory Implementation.
Extracts and stores user preferences, interests, and behavioral patterns.

Note: mem0 embedding provider support varies by version. If Cohere is not supported,
the system will gracefully disable smart memory (non-blocking).
"""
from mem0 import Memory
from typing import List, Dict, Any, Optional
import logging
import uuid
from qdrant_client.models import PointStruct
from app.config import settings
from app.services.groq_service import groq_service
from app.services.qdrant_service import qdrant_service
from app.services.cohere_service import cohere_service
from app.services.debug_logger import log_step

logger = logging.getLogger(__name__)


class SmartMemory:
    """
    Smart memory using mem0.
    Automatically extracts and maintains user preferences and interests.
    """

    def __init__(self):
        """Initialize mem0 with Groq LLM and Cohere embeddings."""
        self.memory = None
        self.qdrant = qdrant_service
        self.cohere = cohere_service
        self.collection_name = "smart_memory_chunks"

        # Configure mem0 to use Groq LLM and HuggingFace embeddings (free, no API key)
        config = {
            "llm": {
                "provider": "groq",
                "config": {
                    "model": settings.groq_model,
                    "temperature": 0.7,
                    "api_key": settings.groq_api_key
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            },
            "version": "v1.1"
        }

        # Initialize mem0
        try:
            self.memory = Memory.from_config(config)
            print("✓ Smart memory initialized with Cohere embeddings")
        except Exception as e:
            print(f"WARNING: Smart memory initialization warning: {str(e)}")
            # Disable smart memory if mem0 setup fails. Avoid default mem0 fallback,
            # which can instantiate OpenAI embeddings and require OPENAI_API_KEY.
            self.memory = None

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
            print(f"Smart memory upsert error: {str(e)}")
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
            # Silent fail for memory extraction errors
            print(f"Smart memory extraction error: {str(e)}")
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
            print(f"Smart memory add error: {str(e)}")
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
            print(f"Smart memory retrieval error: {str(e)}")
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
            print(f"Smart memory summary error: {str(e)}")
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
            points = await self.qdrant.scroll_collection(
                collection_name=self.collection_name,
                filter_conditions={"user_id": user_id},
                limit=1000,
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
            points = await self.qdrant.scroll_collection(
                collection_name=self.collection_name,
                filter_conditions={"user_id": user_id},
                limit=1000,
            )
            if not points:
                logger.info("reset_user_memories: no memories found for user=%s", user_id)
                return True

            point_ids = [p["id"] for p in points]
            await self.qdrant.delete_points(
                collection_name=self.collection_name,
                point_ids=point_ids,
            )
            logger.info(
                "reset_user_memories: deleted %d memory points for user=%s",
                len(point_ids), user_id,
            )
            return True
        except Exception as e:
            logger.error("Smart memory reset error for user=%s: %s", user_id, e)
            return False


# Singleton instance
smart_memory = SmartMemory()
