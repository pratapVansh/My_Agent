"""
Text chunking service for RAG.
Provides token-aware text splitting with overlap and metadata preservation.
"""
import tiktoken
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import re
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Represents a text chunk with metadata.
    """
    text: str
    metadata: Dict[str, Any]
    char_range: Tuple[int, int]


class ChunkingService:
    """
    Service for token-based text chunking.
    Uses tiktoken for accurate token counting.
    """

    def __init__(self):
        """Initialize tiktoken encoder."""
        try:
            # Use cl100k_base encoding (GPT-3.5/4 tokenizer)
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoder: {e}. Using fallback.")
            self.encoder = None

        self.default_chunk_size = settings.chunk_size
        self.default_overlap = settings.chunk_overlap

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens

        Returns:
            Number of tokens
        """
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Fallback: estimate 1 token ≈ 4 characters
            return len(text) // 4

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[Chunk]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            chunk_size: Target chunk size in tokens (default from config)
            overlap: Overlap size in tokens (default from config)

        Returns:
            List of Chunk objects
        """
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap

        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        # Split by paragraphs first (preserve structure)
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""
        current_start = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph exceeds chunk size
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            token_count = self.count_tokens(test_chunk)

            if token_count <= chunk_size:
                # Add paragraph to current chunk
                current_chunk = test_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": len(chunks),
                        "char_range_start": current_start,
                        "char_range_end": current_start + len(current_chunk)
                    }

                    chunks.append(Chunk(
                        text=current_chunk,
                        metadata=chunk_metadata,
                        char_range=(current_start, current_start + len(current_chunk))
                    ))

                    # Calculate overlap
                    if overlap > 0:
                        overlap_text = self._get_overlap_text(current_chunk, overlap)
                        current_chunk = overlap_text + "\n\n" + paragraph
                        current_start = current_start + len(current_chunk) - len(overlap_text)
                    else:
                        current_chunk = paragraph
                        current_start = current_start + len(chunks[-1].text)
                else:
                    # The paragraph alone exceeds the budget, so it must be
                    # split rather than emitted whole — text extracted from PDFs
                    # frequently has no blank-line breaks at all, which would
                    # otherwise produce one oversized chunk for a whole document.
                    for piece in self._split_oversized(paragraph, chunk_size):
                        chunk_metadata = {
                            **metadata,
                            "chunk_index": len(chunks),
                            "char_range_start": current_start,
                            "char_range_end": current_start + len(piece),
                        }
                        chunks.append(Chunk(
                            text=piece,
                            metadata=chunk_metadata,
                            char_range=(current_start, current_start + len(piece)),
                        ))
                        current_start += len(piece)
                    current_chunk = ""

        # Add remaining chunk
        if current_chunk:
            chunk_metadata = {
                **metadata,
                "chunk_index": len(chunks),
                "char_range_start": current_start,
                "char_range_end": current_start + len(current_chunk)
            }

            chunks.append(Chunk(
                text=current_chunk,
                metadata=chunk_metadata,
                char_range=(current_start, current_start + len(current_chunk))
            ))

        # Add total_chunks to all metadata
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        logger.debug(
            "Split text into %d chunks (chunk_size=%d, overlap=%d)",
            len(chunks), chunk_size, overlap,
        )

        return chunks

    def _split_oversized(self, text: str, chunk_size: int) -> List[str]:
        """
        Break a single over-budget paragraph into chunk_size-token pieces.

        Splits on sentence boundaries where possible so chunks stay semantically
        coherent, and falls back to hard token windows for text with no sentence
        punctuation (bullet lists, tabular resume sections).
        """
        if self.count_tokens(text) <= chunk_size:
            return [text]

        # Prefer sentence boundaries.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        pieces: List[str] = []
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if self.count_tokens(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                pieces.append(current)
            # A single sentence can still exceed the budget — window it.
            if self.count_tokens(sentence) > chunk_size:
                pieces.extend(self._split_by_tokens(sentence, chunk_size))
                current = ""
            else:
                current = sentence

        if current:
            pieces.append(current)

        return [p for p in pieces if p.strip()]

    def _split_by_tokens(self, text: str, chunk_size: int) -> List[str]:
        """Hard-split text into fixed token windows (last-resort splitter)."""
        if not self.encoder:
            # Without a tokenizer, approximate using the 4-chars-per-token rule.
            width = max(1, chunk_size * 4)
            return [text[i:i + width] for i in range(0, len(text), width)]

        tokens = self.encoder.encode(text)
        return [
            self.encoder.decode(tokens[i:i + chunk_size])
            for i in range(0, len(tokens), chunk_size)
        ]

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """
        Get the last N tokens of text for overlap.

        Args:
            text: Source text
            overlap_tokens: Number of tokens to overlap

        Returns:
            Overlap text
        """
        if not self.encoder or overlap_tokens <= 0:
            return ""

        tokens = self.encoder.encode(text)

        if len(tokens) <= overlap_tokens:
            return text

        # Get last overlap_tokens tokens
        overlap_token_ids = tokens[-overlap_tokens:]
        overlap_text = self.encoder.decode(overlap_token_ids)

        return overlap_text

    def reconstruct_text(self, chunks: List[Chunk]) -> str:
        """
        Reconstruct text from chunks (removes overlap).

        Args:
            chunks: List of Chunk objects

        Returns:
            Reconstructed text
        """
        if not chunks:
            return ""

        # Sort by chunk_index
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.metadata.get("chunk_index", 0)
        )

        # Simple reconstruction (concatenate)
        # Note: This doesn't perfectly remove overlap,
        # but works well enough for retrieval scenarios
        return "\n\n".join(chunk.text for chunk in sorted_chunks)

    def get_chunk_context(
        self,
        chunks: List[Chunk],
        target_chunk: int,
        context_window: int = 1
    ) -> str:
        """
        Get a chunk with surrounding context.

        Args:
            chunks: List of all chunks
            target_chunk: Index of the target chunk
            context_window: Number of chunks before and after

        Returns:
            Combined text with context
        """
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.metadata.get("chunk_index", 0)
        )

        start = max(0, target_chunk - context_window)
        end = min(len(sorted_chunks), target_chunk + context_window + 1)

        context_chunks = sorted_chunks[start:end]

        return "\n\n".join(chunk.text for chunk in context_chunks)


# Global service instance (singleton pattern)
chunking_service = ChunkingService()
