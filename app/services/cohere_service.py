"""
Cohere API service wrapper for embeddings.
Provides async embedding generation with batching and retry logic.
"""
import cohere
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import asyncio
import hashlib
import logging
import random
import time
from app.config import settings
from app.services.call_metrics import (
    record_embed_cache_hit,
    record_embed_call,
    record_embed_coalesced,
)

logger = logging.getLogger(__name__)

_QUERY_CACHE_TTL_SECONDS = 60.0
_QUERY_CACHE_MAX_ENTRIES = 512

# How long to stop calling Cohere after a quota/auth rejection. The background
# memory worker retries the same pending records every 30s and leaves them
# PENDING on failure, so without a cooldown a single exhausted quota turns into
# a permanent retry loop that burns the *next* key the moment it is installed.
_CIRCUIT_COOLDOWN_SECONDS = 900.0


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

        # Query embeddings currently being computed, keyed exactly as the cache
        # above. This is what stops a concurrent fan-out from issuing the same
        # request five times — see `embed_text`. Entries live only for the
        # duration of one call and are removed in a `finally`, so a failure
        # cannot leave a permanently poisoned key behind.
        self._in_flight: Dict[str, "asyncio.Future[List[float]]"] = {}

        # Circuit breaker state — set when the account is out of quota or the
        # key is rejected. Both are conditions no amount of retrying fixes.
        self._circuit_open_until: float = 0.0
        self._circuit_reason: str = ""

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
                timeout=60,
                # The SDK retries twice on its own. Combined with
                # _retry_with_backoff that made 9 HTTP calls per logical embed
                # and burned the monthly trial quota ~9x faster than the logs
                # suggested. Retrying is this wrapper's job, not the SDK's.
                max_retries=0,
            )
        return self._client

    @staticmethod
    def _classify_error(exc: Exception) -> Optional[str]:
        """Return a reason string if `exc` is permanent, else None (retryable).

        Cohere answers both "you are going too fast" and "your account is out of
        calls for the month" with a 429; only the message distinguishes them,
        and retrying the second one is pure waste.
        """
        status = getattr(exc, "status_code", None)
        message = str(getattr(exc, "body", "") or exc).lower()

        if status in (401, 403):
            return "api key rejected"
        if status == 402:
            return "billing/credits exhausted"
        if status == 429 and ("month" in message or "trial key" in message):
            return "account quota exhausted (trial limit)"
        return None

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
        now = time.monotonic()
        if now < self._circuit_open_until:
            remaining = int(self._circuit_open_until - now)
            raise RuntimeError(
                f"Cohere circuit open ({self._circuit_reason}); "
                f"not retrying for another {remaining}s"
            )

        for attempt in range(max_attempts):
            try:
                result = await func()
                self._circuit_open_until = 0.0
                self._circuit_reason = ""
                return result
            except Exception as e:
                permanent = self._classify_error(e)
                if permanent:
                    self._circuit_open_until = (
                        time.monotonic() + _CIRCUIT_COOLDOWN_SECONDS
                    )
                    self._circuit_reason = permanent
                    logger.error(
                        "Cohere %s — pausing all embedding calls for %.0fs. "
                        "Retrying cannot fix this; install a key on an account "
                        "with remaining quota.",
                        permanent, _CIRCUIT_COOLDOWN_SECONDS,
                    )
                    raise

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
                    record_embed_cache_hit()
                    return embedding
                del self._query_cache[cache_key]

            # ── Single flight ────────────────────────────────────────────────
            #
            # The cache above is necessary and was not sufficient. One turn asks
            # for this exact embedding from five places at once —
            # `retrieve_preferences`, `retrieve_skills`, `retrieve_projects`,
            # two résumé fallbacks, and the v2 shadow engine — and they run
            # concurrently under `asyncio.gather`. Every one of them reached the
            # cache lookup before any of them had written to it, so all five
            # missed and all five called Cohere with identical arguments.
            #
            # The cache answers "has this been computed?"; this answers "is it
            # being computed right now?", which is the question a concurrent
            # fan-out actually asks. The future is registered *before* the first
            # await, so there is no window in which a second caller can look and
            # find nothing.
            in_flight = self._in_flight.get(cache_key)
            if in_flight is not None:
                record_embed_coalesced()
                # `shield` is not needed and would be wrong: awaiting the shared
                # future means a caller whose own task is cancelled simply stops
                # waiting. The future belongs to whoever created it.
                return await asyncio.shield(in_flight)

            future: "asyncio.Future[List[float]]" = asyncio.get_running_loop().create_future()
            self._in_flight[cache_key] = future
            try:
                result = await self._embed_text_uncached(text, input_type)
            except BaseException as exc:
                # Failures are never cached — only the in-flight entry is
                # cleared, so the next caller retries rather than inheriting
                # this one's error for the next 60 seconds. Waiters are woken
                # with the same exception so they fail together rather than
                # hanging on a future nobody will ever resolve.
                if not future.done():
                    future.set_exception(exc)
                # Retrieve it so a future nobody awaited does not log
                # "exception was never retrieved" on garbage collection.
                future.exception()
                raise
            finally:
                self._in_flight.pop(cache_key, None)

            if not future.done():
                future.set_result(result)

            self._query_cache[cache_key] = (result, time.monotonic())
            self._query_cache.move_to_end(cache_key)

            # O(1) LRU eviction — pop from the least-recently-used end.
            while len(self._query_cache) > _QUERY_CACHE_MAX_ENTRIES:
                self._query_cache.popitem(last=False)
            return result

        # Document embeddings are deliberately untouched. They are writes, each
        # one carries different text, and coalescing identical ones would be
        # solving a problem that does not exist on that path.
        return await self._embed_text_uncached(text, input_type)

    async def _embed_text_uncached(
        self,
        text: str,
        input_type: str,
    ) -> List[float]:
        """Internal embed call without cache."""
        async def _embed():
            try:
                record_embed_call()
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
                    record_embed_call()
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
