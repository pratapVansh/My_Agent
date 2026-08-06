"""TTS text-chunker tests (audit finding L8)."""
import asyncio
from typing import AsyncIterator, List

import pytest

from app.services.text_chunker import text_chunker


async def _collect(stream) -> List[str]:
    return [chunk async for chunk in stream]


async def _tokens(items: List[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_flushes_on_sentence_punctuation():
    chunks = await _collect(
        text_chunker.chunk_tokens(_tokens(["Hello", " there", "."]), max_words=50)
    )
    assert "".join(chunks).strip() == "Hello there."


@pytest.mark.asyncio
async def test_emits_trailing_text_without_punctuation():
    chunks = await _collect(
        text_chunker.chunk_tokens(_tokens(["No", " final", " period"]), max_words=50)
    )
    assert "no final period" in " ".join(chunks).lower()


@pytest.mark.asyncio
async def test_bad_iterable_raises_original_error_not_unbound_local():
    """A failure in __aiter__ must surface as itself, not UnboundLocalError."""
    class Exploding:
        def __aiter__(self):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await _collect(text_chunker.chunk_tokens(Exploding()))


@pytest.mark.asyncio
async def test_empty_stream_produces_no_chunks():
    chunks = await _collect(text_chunker.chunk_tokens(_tokens([])))
    assert chunks == []
