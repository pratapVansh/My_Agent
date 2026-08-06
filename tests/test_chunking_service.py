"""
Chunking tests (audit finding M9).

Text extracted from PDFs frequently has no blank-line paragraph breaks, so the
oversized-paragraph path is the common case for resumes, not an edge case.
"""
from app.services.chunking_service import chunking_service


def test_oversized_paragraph_is_split_not_passed_through():
    # One paragraph, no blank lines, far above the chunk budget.
    long_paragraph = " ".join(f"word{i}" for i in range(4000))
    chunk_size = 100

    chunks = chunking_service.chunk_text(
        text=long_paragraph, metadata={"user_id": "test"}, chunk_size=chunk_size, overlap=0
    )

    assert len(chunks) > 1, "an oversized paragraph must be split into several chunks"
    for chunk in chunks:
        # Allow modest slack for boundary handling, but nothing near the old
        # behaviour of emitting the entire document as one chunk.
        assert chunking_service.count_tokens(chunk.text) <= chunk_size * 2


def test_text_without_sentence_punctuation_still_splits():
    bullets = "\n".join(f"- item number {i} in a long skills list" for i in range(500))

    chunks = chunking_service.chunk_text(
        text=bullets, metadata={}, chunk_size=80, overlap=0
    )

    assert len(chunks) > 1
    assert all(chunk.text.strip() for chunk in chunks)


def test_short_text_stays_in_one_chunk():
    chunks = chunking_service.chunk_text(
        text="A short resume summary.", metadata={}, chunk_size=400, overlap=0
    )
    assert len(chunks) == 1


def test_empty_text_yields_no_chunks():
    assert chunking_service.chunk_text(text="   ", metadata={}) == []


def test_all_chunks_carry_metadata_and_total_count():
    chunks = chunking_service.chunk_text(
        text="Para one.\n\nPara two.\n\nPara three.",
        metadata={"user_id": "vansh", "type": "resume"},
        chunk_size=10,
        overlap=0,
    )

    assert chunks
    for chunk in chunks:
        assert chunk.metadata["user_id"] == "vansh"
        assert chunk.metadata["total_chunks"] == len(chunks)
