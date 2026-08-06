"""
Text chunker — turns an LLM token stream into TTS-ready text chunks.

Chunk boundaries are chosen to minimise time-to-first-audio without making
speech sound clipped:

  * The FIRST chunk is released after only a few words. That single boundary
    dominates perceived response time, because nothing is audible until the
    first chunk reaches the TTS provider.
  * Later chunks prefer sentence ends, then clause ends, so prosody stays
    natural and the synthesiser has enough context to intone correctly.
  * An idle timeout releases whatever is buffered if the LLM stalls
    mid-sentence, so a slow token never stalls audio outright.

Boundaries are always word boundaries: handing TTS a partial word ("interes")
makes it audibly mispronounce the fragment.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncGenerator, AsyncIterable, Optional

logger = logging.getLogger(__name__)

_SENTENCE_END = ".?!…"
_CLAUSE_END = ",;:—"

# "Mr.", "e.g.", "3.14" — the period is not a sentence end. Splitting there
# sends a fragment to TTS which is then spoken with a falling, end-of-thought
# intonation in the middle of a sentence.
_ABBREVIATION_TAIL = re.compile(
    r"(?:^|\s)(?:[A-Za-z]|[Mm]r|[Mm]rs|[Mm]s|[Dd]r|[Pp]rof|[Ss]t|vs|etc|e\.g|i\.e|approx|No)\.$"
)
_DECIMAL_TAIL = re.compile(r"\d\.$")


def _is_sentence_boundary(text: str) -> bool:
    """True when `text` ends at a real sentence end (not an abbreviation)."""
    if not text or text[-1] not in _SENTENCE_END:
        return False
    if text[-1] == "." and (_DECIMAL_TAIL.search(text) or _ABBREVIATION_TAIL.search(text)):
        return False
    return True


def _word_count(text: str) -> int:
    return len(text.split())


class TextChunker:
    """Groups streamed tokens into speakable chunks.

    Args:
        first_chunk_words: words required before the very first chunk is
            released. Small on purpose — this is the latency the user feels.
        min_chunk_words: minimum size for a clause-boundary split, so
            "Yes, ..." does not become its own one-word utterance.
        idle_timeout: seconds of token silence after which the buffer is
            flushed at a word boundary.
    """

    def __init__(
        self,
        first_chunk_words: int = 4,
        min_chunk_words: int = 6,
        idle_timeout: float = 0.35,
    ) -> None:
        self._first_chunk_words = first_chunk_words
        self._min_chunk_words = min_chunk_words
        self._idle_timeout = idle_timeout

    async def chunk_tokens(
        self,
        token_stream: AsyncIterable[str],
        max_words: int = 30,
        timeout_seconds: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield speakable chunks from an async token stream.

        Every ``yield`` happens inside the normal body of the generator — never
        from a ``finally`` block. Yielding during cleanup raises
        ``RuntimeError: async generator ignored GeneratorExit`` the moment a
        consumer stops early, which is exactly what a barge-in does.
        """
        idle_timeout = self._idle_timeout if timeout_seconds is None else timeout_seconds

        buffer = ""
        chunk_index = 0
        pending: set[asyncio.Task] = set()

        def take(split_at: int) -> str:
            """Detach buffer[:split_at] as a chunk, keeping the remainder."""
            nonlocal buffer, chunk_index
            chunk = buffer[:split_at].strip()
            buffer = buffer[split_at:].lstrip()
            if chunk:
                chunk_index += 1
            return chunk

        def ready() -> Optional[str]:
            """Return the next chunk to emit, or None to keep buffering."""
            stripped = buffer.rstrip()
            if not stripped:
                return None
            words = _word_count(stripped)

            # The first chunk goes out as early as possible: any word boundary
            # will do once there are a few words. Waiting for a sentence here
            # adds most of a second to the assistant's apparent response time.
            if chunk_index == 0 and words >= self._first_chunk_words:
                if _is_sentence_boundary(stripped) or stripped[-1] in _CLAUSE_END:
                    return take(len(stripped))
                last_space = stripped.rfind(" ")
                if last_space > 0:
                    return take(last_space)

            if _is_sentence_boundary(stripped):
                return take(len(stripped))

            if words >= self._min_chunk_words and stripped[-1] in _CLAUSE_END:
                return take(len(stripped))

            if words >= max_words:
                last_space = stripped.rfind(" ")
                return take(last_space if last_space > 0 else len(stripped))

            return None

        try:
            iterator = token_stream.__aiter__()
            pending = {asyncio.create_task(iterator.__anext__())}
            exhausted = False

            while not exhausted:
                done, pending = await asyncio.wait(
                    pending, timeout=idle_timeout, return_when=asyncio.FIRST_COMPLETED
                )

                if not done:
                    # The model stalled. Speak what we have rather than letting
                    # audio go silent, but only up to a word boundary.
                    stripped = buffer.rstrip()
                    if _word_count(stripped) >= 1:
                        split_at = (
                            len(stripped)
                            if _is_sentence_boundary(stripped) or buffer != stripped
                            else stripped.rfind(" ")
                        )
                        if split_at > 0:
                            chunk = take(split_at)
                            if chunk:
                                yield chunk
                    continue

                try:
                    token = done.pop().result()
                except StopAsyncIteration:
                    exhausted = True
                    break

                pending.add(asyncio.create_task(iterator.__anext__()))
                if not token:
                    continue

                buffer += token
                while True:
                    chunk = ready()
                    if not chunk:
                        break
                    yield chunk

            # Final flush, in the normal body so cancellation cannot reach it.
            tail = buffer.strip()
            buffer = ""
            if tail:
                yield tail

        except asyncio.CancelledError:
            logger.debug("[TextChunker] Cancelled")
            raise
        finally:
            # Cancel and await the in-flight token fetch so its cancellation
            # completes and any teardown error is observed rather than left as
            # an unretrieved exception. No yield here, deliberately.
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)


text_chunker = TextChunker()
