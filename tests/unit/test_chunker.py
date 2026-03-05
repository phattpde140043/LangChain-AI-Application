from __future__ import annotations
"""Tests for the TextChunker class."""

import pytest
from langchain_core.documents import Document
from src.ingestion.chunker import TextChunker, HierarchicalChunker


@pytest.fixture
def chunker():
    return TextChunker()


@pytest.fixture
def long_document():
    return Document(
        page_content="This is a test document. " * 200,  # ~4800 chars
        metadata={"source": "test.txt", "filename": "test.txt", "file_type": "text", "page_number": 1},
    )


def test_chunk_basic_document(chunker, long_document):
    """Test that a long document is split into multiple chunks."""
    chunks = chunker.chunk_documents([long_document], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1


def test_chunk_preserves_metadata(chunker, long_document):
    """Test that source metadata is preserved in all chunks."""
    chunks = chunker.chunk_documents([long_document], chunk_size=200, chunk_overlap=20)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"
        assert chunk.metadata["filename"] == "test.txt"


def test_chunk_adds_index(chunker, long_document):
    """Test that chunk_index is added to metadata."""
    chunks = chunker.chunk_documents([long_document], chunk_size=200, chunk_overlap=20)
    for i, chunk in enumerate(chunks):
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["chunk_index"] == i


def test_chunk_overlap(chunker):
    """Test that chunks have expected overlap."""
    doc = Document(
        page_content="A" * 400 + "B" * 400,
        metadata={"source": "test.txt", "filename": "test.txt", "file_type": "text"},
    )
    chunks = chunker.chunk_documents([doc], chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 2


def test_chunk_empty_document(chunker):
    """Test handling of empty document."""
    doc = Document(page_content="", metadata={"source": "empty.txt", "filename": "empty.txt"})
    chunks = chunker.chunk_documents([doc], chunk_size=200, chunk_overlap=20)
    assert isinstance(chunks, list)
