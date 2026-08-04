"""
Text Chunker Service.
Accumulates LLM text tokens and emits TTS-friendly chunks based on:
- Sentence punctuation (. ? !)
- Word count limits (~15-20 words)
- Timeouts (if tokens pause)
"""
import asyncio
import logging
from typing import AsyncIterable, AsyncGenerator

logger = logging.getLogger(__name__)

class TextChunker:
    @staticmethod
    async def chunk_tokens(
        token_stream: AsyncIterable[str],
        max_words: int = 15,
        timeout_seconds: float = 0.5
    ) -> AsyncGenerator[str, None]:
        """
        Consumes an async stream of tokens.
        Yields complete sentences or chunks when limits are reached.
        Uses asyncio.wait to implement timeout without cancelling the underlying generator.
        """
        buffer = ""
        word_count = 0

        def _flush() -> str:
            nonlocal buffer, word_count
            chunk = buffer.strip()
            buffer = ""
            word_count = 0
            return chunk

        try:
            iterator = token_stream.__aiter__()
            # Create a task for the first token
            pending = {asyncio.create_task(iterator.__anext__())}
            
            while True:
                done, pending = await asyncio.wait(
                    pending, 
                    timeout=timeout_seconds, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                if not done:
                    # Timeout occurred, flush if there is anything
                    chunk = _flush()
                    if chunk:
                        yield chunk
                    continue
                    
                task = done.pop()
                try:
                    token = task.result()
                    # Enqueue the next token fetch
                    pending.add(asyncio.create_task(iterator.__anext__()))
                except StopAsyncIteration:
                    break
                
                if not token:
                    continue
                    
                buffer += token
                word_count += token.count(" ")
                
                # Condition 1: Punctuation
                stripped = buffer.rstrip()
                if stripped and stripped[-1] in ".?!":
                    chunk = _flush()
                    if chunk:
                        yield chunk
                    continue
                    
                # Condition 2: Word Count Limit
                if word_count >= max_words:
                    last_space = buffer.rfind(" ")
                    if last_space > 0:
                        chunk = buffer[:last_space].strip()
                        buffer = buffer[last_space:]
                        word_count = buffer.count(" ")
                        if chunk:
                            yield chunk
                    else:
                        chunk = _flush()
                        if chunk:
                            yield chunk
                            
        except asyncio.CancelledError:
            logger.info("[TextChunker] Cancelled")
            raise
        except Exception as e:
            logger.error("[TextChunker] Error: %s", e)
            raise
        finally:
            # Clean up any pending tasks from wait
            for p in pending:
                p.cancel()
                
            # Final flush
            chunk = _flush()
            if chunk:
                yield chunk

text_chunker = TextChunker()
