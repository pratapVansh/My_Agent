"""
Cohere API service wrapper for embeddings.
Provides async embedding generation with batching and retry logic.
"""
import cohere
from collections import OrderedDict
from typing import List, Optional, Tuple
import asyncio
import hashlib
import logging
import random
import time
from app.config import settings

logger = logging.getLogger(__name__)

_QUERY_CACHE_TTL_SECONDS = 60.0
_QUERY_CACHE_MAX_ENTRIES = 512


class CohereService:
    """
    Service wrapper for Cohere Embeddings API.
    Implements batching, retry logic, and error handling.
    """

    def __init__(self):
        """Prepare defaults. The API client itself is built lazily."""
        self._client: Optional[cohere.AsyncClient] = None
        self.model = settings.cohere_model
        self.embedding_dimension = settings.cohere_embedding_dimension
        self.max_batch_size = 96  # Cohere's max batch size

        # LRU+TTL cache for search_query embeddings only.
        # search_document embeddings are for ingestion only — not cached.
        # OrderedDict gives O(1) eviction; a plain dict required a full sort.
        self._query_cache: "OrderedDict[str, Tuple[List[float], float]]" = OrderedDict()

    @property
    def client(self) -> cohere.AsyncClient:
        """Build the Cohere client on first use (see GroqService.client)."""
        if self._client is None:
            if not (settings.cohere_api_key or "").strip():
                raise RuntimeError(
                    "COHERE_API_KEY is not configured. Set it in your .env file."
                )
            self._client = cohere.AsyncClient(
                api_key=settings.cohere_api_key,
                timeout=60
            )
        return self._client

    async def _retry_with_backoff(
        self,
        func,
        max_attempts: int = 3,
        backoff_base: float = 2.0
    ):
        """
        Retry a function with jittered exponential backoff.

        Jitter matters here because many callers embed concurrently: without it
        every caller that hits a rate limit backs off for an identical interval
        and retries in lockstep, amplifying the burst that caused the limit.

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(max_attempts):
            try:
                return await func()
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.error(f"All {max_attempts} attempts failed: {str(e)}")
                    raise

                wait_time = (backoff_base ** attempt) * (1.0 + random.random() * 0.5)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                    f"Retrying in {wait_time:.2f}s..."
                )
                await asyncio.sleep(wait_time)

    async def embed_text(
        self,
        text: str,
        input_type: str = "search_document"
    ) -> List[float]:
        """
        Generate embedding for a single text.
        search_query embeddings are cached for 60s to avoid redundant API calls.

        Args:
            text: Text to embed
            input_type: "search_document" for storage, "search_query" for retrieval

        Returns:
            List of floats representing the embedding (1024 dimensions)
        """
        if input_type == "search_query":
            cache_key = hashlib.md5(text.encode()).hexdigest()
            now = time.monotonic()

            cached = self._query_cache.get(cache_key)
            if cached is not None:
                embedding, cached_at = cached
                if now - cached_at < _QUERY_CACHE_TTL_SECONDS:
                    self._query_cache.move_to_end(cache_key)
                    return embedding
                del self._query_cache[cache_key]

            result = await self._embed_text_uncached(text, input_type)
            self._query_cache[cache_key] = (result, now)
            self._query_cache.move_to_end(cache_key)

            # O(1) LRU eviction — pop from the least-recently-used end.
            while len(self._query_cache) > _QUERY_CACHE_MAX_ENTRIES:
                self._query_cache.popitem(last=False)
            return result

        return await self._embed_text_uncached(text, input_type)

    async def _embed_text_uncached(
        self,
        text: str,
        input_type: str,
    ) -> List[float]:
        """Internal embed call without cache."""
        async def _embed():
            try:
                response = await self.client.embed(
                    texts=[text],
                    model=self.model,
                    input_type=input_type
                )
                embedding = [float(x) for x in response.embeddings[0]]
                logger.info(
                    f"Generated embedding: {len(embedding)} dimensions, "
                    f"input_type={input_type}"
                )
                return embedding
            except Exception as e:
                logger.error(f"Cohere embedding error: {str(e)}")
                raise

        return await self._retry_with_backoff(_embed)

    async def embed_batch(
        self,
        texts: List[str],
        input_type: str = "search_document"
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed
            input_type: Type of input - "search_document" or "search_query"

        Returns:
            List of embeddings (each embedding is a list of floats)

        Raises:
            Exception: If batch embedding fails
        """
        if not texts:
            return []

        # Process in batches if needed
        all_embeddings = []

        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]

            async def _embed_batch():
                try:
                    response = await self.client.embed(
                        texts=batch,
                        model=self.model,
                        input_type=input_type
                    )

                    # Convert to float32
                    embeddings = [
                        [float(x) for x in emb]
                        for emb in response.embeddings
                    ]

                    logger.info(
                        f"Generated {len(embeddings)} embeddings "
                        f"(batch {i // self.max_batch_size + 1})"
                    )

                    return embeddings

                except Exception as e:
                    logger.error(f"Cohere batch embedding error: {str(e)}")
                    raise

            batch_embeddings = await self._retry_with_backoff(_embed_batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def health_check(self) -> bool:
        """
        Verify Cohere API connectivity.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            test_embedding = await self.embed_text(
                "test",
                input_type="search_document"
            )

            # Verify embedding dimension
            if len(test_embedding) == self.embedding_dimension:
                logger.info("Cohere API health check passed")
                return True
            else:
                logger.error(
                    f"Expected {self.embedding_dimension} dimensions, "
                    f"got {len(test_embedding)}"
                )
                return False

        except Exception as e:
            logger.error(f"Cohere health check failed: {str(e)}")
            return False


# Global service instance (singleton pattern)
cohere_service = CohereService()
