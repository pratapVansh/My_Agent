"""
Groq API service wrapper with async support and low-latency optimizations.
Provides a clean interface for interacting with Groq's LLM API.
"""
from groq import AsyncGroq
from typing import Optional, Dict, Any, List
import logging
from app.config import settings
from app.services.call_metrics import record_groq_call
from app.services.groq_limiter import groq_limiter

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
                # Retry is owned by exactly one layer, and this is not it.
                #
                # With max_retries=1 the SDK silently doubled every attempt
                # underneath `BaseAgent.call_groq`, which was itself retrying
                # three times, inside a reasoning loop running three
                # iterations, inside a reflect loop re-running the specialist
                # three times. One logical call could become dozens of HTTP
                # requests, and on a 429 each of those requests made the rate
                # limit worse rather than better.
                #
                # `call_groq` retries, reads Retry-After, and distinguishes a
                # rate limit from a transient fault. The SDK cannot do the
                # first or the third, so it does none of them.
                #
                # This is the same conclusion `CohereService` reached for the
                # same reason — see the max_retries=0 note there.
                max_retries=0,
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

        No `reasoning_effort` override here, deliberately. `openai/gpt-oss-120b`
        is a reasoning model with a separate internal reasoning channel, and
        the instinct to force it to "low" to save latency was tried and
        measured: across repeated trials it made `content` come back empty on
        some calls and, worse, made the model answer confidently from a
        fabricated schedule without ever calling the tool on others. The
        default (no override) came back correct and non-empty on every trial.
        See the migration verification notes for the full comparison.
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
            # The gate, not a retry. Bounds how many requests are open at once
            # and how many tokens per minute are admitted, so a fan-out queues
            # instead of arriving as the burst that produces a 429.
            async with groq_limiter.reserve(messages, params.get("max_tokens")):
                response = await self.client.chat.completions.create(stream=stream, **params)

            if stream:
                return response

            record_groq_call(params.get("model"), stream=False)
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
            # Re-raised as-is rather than repackaged into a bare Exception.
            # The retry policy in `BaseAgent.call_groq` has to tell a 429 from
            # a 500 and has to read the Retry-After header to know how long the
            # window stays closed — and `Exception(str(e))` throws away the
            # status code, the response and the headers, leaving the caller to
            # guess from a message. Guessing is what made the retry loop fire
            # straight back into a closed rate-limit window.
            raise

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
            # The reservation is held for the whole iteration, not just the
            # initial response. A stream keeps the connection open until the
            # last token, so releasing the slot after `create` returns would
            # let N streams run concurrently under a limit of one.
            async with groq_limiter.reserve(messages, params.get("max_tokens")):
                record_groq_call(params.get("model"), stream=True)
                stream = await self.client.chat.completions.create(stream=True, **params)

                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error("Groq stream_chat_completion failed (model=%s): %s", model or self.model, e)
            raise  # see chat_completion — the exception type carries the retry signal

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
