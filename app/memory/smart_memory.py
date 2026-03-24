"""
mem0 Smart Memory Implementation.
Extracts and stores user preferences, interests, and behavioral patterns.

Note: mem0 embedding provider support varies by version. If Cohere is not supported,
the system will gracefully disable smart memory (non-blocking).
"""
from mem0 import Memory
from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.groq_service import groq_service


class SmartMemory:
    """
    Smart memory using mem0.
    Automatically extracts and maintains user preferences and interests.
    """

    def __init__(self):
        """Initialize mem0 with Groq LLM and Cohere embeddings."""
        self.memory = None

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
        if self.memory is None:
            return []

        try:
            # mem0 automatically extracts insights from messages
            result = self.memory.add(
                messages=messages,
                user_id=user_id,
                metadata=metadata
            )

            # Extract memory IDs from result
            if isinstance(result, dict) and "results" in result:
                return [item.get("id", "") for item in result.get("results", [])]
            elif isinstance(result, list):
                return [item.get("id", "") for item in result]
            else:
                return []

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
        if self.memory is None:
            return None

        try:
            result = self.memory.add(
                messages=[{"role": "user", "content": preference}],
                user_id=user_id,
                metadata=metadata
            )

            if isinstance(result, dict) and "results" in result:
                return result["results"][0].get("id") if result["results"] else None
            elif isinstance(result, list) and result:
                return result[0].get("id")
            return None

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
        if self.memory is None:
            return []

        try:
            if query:
                # Semantic search
                results = self.memory.search(
                    query=query,
                    user_id=user_id,
                    limit=limit
                )
            else:
                # Get all memories
                results = self.memory.get_all(
                    user_id=user_id,
                    limit=limit
                )

            # Format results
            if isinstance(results, dict) and "results" in results:
                return results["results"]
            elif isinstance(results, list):
                return results
            else:
                return []

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
        Delete a specific memory.

        Args:
            user_id: User identifier
            memory_id: Memory ID to delete

        Returns:
            Success status
        """
        if self.memory is None:
            return False

        try:
            self.memory.delete(memory_id=memory_id, user_id=user_id)
            return True
        except Exception as e:
            print(f"Smart memory delete error: {str(e)}")
            return False

    async def reset_user_memories(self, user_id: str) -> bool:
        """
        Reset all memories for a user.

        Args:
            user_id: User identifier

        Returns:
            Success status
        """
        if self.memory is None:
            return False

        try:
            self.memory.reset(user_id=user_id)
            return True
        except Exception as e:
            print(f"Smart memory reset error: {str(e)}")
            return False


# Singleton instance
smart_memory = SmartMemory()
