"""
Cohere API service wrapper for embeddings.
Provides async embedding generation with batching and retry logic.
"""
import cohere
from typing import List, Optional
import asyncio
import logging
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CohereService:
    """
    Service wrapper for Cohere Embeddings API.
    Implements batching, retry logic, and error handling.
    """

    def __init__(self):
        """Initialize Cohere client with API key from settings."""
        self.client = cohere.AsyncClient(
            api_key=settings.cohere_api_key,
            timeout=60
        )
        self.model = settings.cohere_model
        self.embedding_dimension = settings.cohere_embedding_dimension
        self.max_batch_size = 96  # Cohere's max batch size

    async def _retry_with_backoff(
        self,
        func,
        max_attempts: int = 3,
        backoff_base: float = 2.0
    ):
        """
        Retry a function with exponential backoff.

        Args:
            func: Async function to retry
            max_attempts: Maximum number of retry attempts
            backoff_base: Base for exponential backoff

        Returns:
            Function result

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

                wait_time = backoff_base ** attempt
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

    async def embed_text(
        self,
        text: str,
        input_type: str = "search_document"
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            input_type: Type of input - "search_document" for storage,
                       "search_query" for retrieval

        Returns:
            List of floats representing the embedding (1024 dimensions)

        Raises:
            Exception: If embedding generation fails
        """
        async def _embed():
            try:
                response = await self.client.embed(
                    texts=[text],
                    model=self.model,
                    input_type=input_type
                )

                # Extract embedding and convert to float32 for Qdrant
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
