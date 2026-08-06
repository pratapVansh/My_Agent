"""
Groq API service wrapper with async support and low-latency optimizations.
Provides a clean interface for interacting with Groq's LLM API.
"""
from groq import AsyncGroq
from typing import Optional, Dict, Any, List
import asyncio
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class GroqService:
    """
    Service wrapper for Groq API with optimized async calls.
    Implements connection pooling and low-latency patterns.
    """

    def __init__(self):
        """Prepare defaults. The API client itself is built lazily."""
        self._client: Optional[AsyncGroq] = None
        self.model = settings.groq_model
        self.temperature = settings.groq_temperature
        self.max_tokens = settings.groq_max_tokens

    @property
    def client(self) -> AsyncGroq:
        """
        Build the Groq client on first use.

        Lazy construction keeps module import free of credential requirements,
        so unrelated modules stay importable (and unit-testable) without a
        populated .env, while a missing key still fails with a clear message.
        """
        if self._client is None:
            if not (settings.groq_api_key or "").strip():
                raise RuntimeError(
                    "GROQ_API_KEY is not configured. Set it in your .env file."
                )
            self._client = AsyncGroq(
                api_key=settings.groq_api_key,
                max_retries=1,  # Reduced for lower latency (fail fast)
                timeout=20.0  # Reduced from 30s for faster failures
            )
        return self._client

    def _build_params(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        model: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Assemble request parameters shared by streaming and non-streaming calls.

        `is not None` rather than `or` is load-bearing: temperature=0.0 is
        falsy, so `temperature or self.temperature` silently discarded every
        request for deterministic output (e.g. the intent router).
        """
        return {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs,
        }

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to Groq API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (default from config)
            max_tokens: Maximum tokens to generate (default from config)
            model: Model to use (default from config)
            stream: Enable streaming responses
            **kwargs: Additional parameters for the API call

        Returns:
            API response as dictionary

        Raises:
            Exception: If API call fails
        """
        try:
            params = self._build_params(messages, temperature, max_tokens, model, **kwargs)
            response = await self.client.chat.completions.create(stream=stream, **params)

            if stream:
                return response

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
        except Exception as e:
            logger.error("Groq chat_completion failed (model=%s): %s", model or self.model, e)
            raise Exception(f"Groq API error: {str(e)}")

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Stream chat completion responses from Groq API.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model to use
            **kwargs: Additional parameters

        Yields:
            Chunks of the response as they arrive
        """
        try:
            params = self._build_params(messages, temperature, max_tokens, model, **kwargs)
            stream = await self.client.chat.completions.create(stream=True, **params)

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error("Groq stream_chat_completion failed (model=%s): %s", model or self.model, e)
            raise Exception(f"Groq streaming error: {str(e)}")

    async def health_check(self) -> bool:
        """
        Verify Groq API connectivity with minimal latency test.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            test_messages = [{"role": "user", "content": "test"}]
            await self.chat_completion(
                messages=test_messages,
                max_tokens=5
            )
            return True
        except Exception:
            return False


# Global service instance (singleton pattern)
groq_service = GroqService()
