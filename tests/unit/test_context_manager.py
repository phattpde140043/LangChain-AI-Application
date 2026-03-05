from __future__ import annotations
"""Tests for the ContextManager class."""

import pytest
from langchain_core.documents import Document
from src.rag.context_manager import ContextManager


@pytest.fixture
def ctx_mgr():
    return ContextManager()


@pytest.fixture
def sample_chunks():
    return [
        Document(
            page_content="Microservices architecture enables independent deployment of services.",
            metadata={"source": "doc1.md"},
        ),
        Document(
            page_content="API Gateway serves as the entry point for all client requests.",
            metadata={"source": "doc2.md"},
        ),
        Document(
            page_content="Docker containers package applications with their dependencies.",
            metadata={"source": "doc3.md"},
        ),
        Document(
            page_content="Kubernetes orchestrates containerized applications at scale.",
            metadata={"source": "doc4.md"},
        ),
    ]


def test_deduplicate_removes_duplicates(ctx_mgr):
    """Test that near-duplicate chunks are removed."""
    duplicate_text = "Microservices architecture enables independent deployment of services."
    chunks = [
        Document(page_content=duplicate_text, metadata={"source": "doc1.md"}),
        Document(page_content=duplicate_text, metadata={"source": "doc1_copy.md"}),
        Document(page_content="Kubernetes orchestrates containers.", metadata={"source": "doc2.md"}),
    ]
    result = ctx_mgr.deduplicate_chunks(chunks, threshold=0.85)
    assert len(result) < len(chunks)


def test_select_context_respects_token_limit(ctx_mgr, sample_chunks):
    """Test that context selection respects the token limit."""
    # With very small token limit, should return fewer chunks
    selected = ctx_mgr.select_context(sample_chunks, max_tokens=20)
    assert len(selected) <= len(sample_chunks)
    # With large token limit, should return all chunks
    selected_all = ctx_mgr.select_context(sample_chunks, max_tokens=10000)
    assert len(selected_all) == len(sample_chunks)


def test_rank_chunks_returns_sorted(ctx_mgr, sample_chunks):
    """Test that chunks are ranked by relevance to the query."""
    query = "microservices deployment"
    ranked = ctx_mgr.rank_chunks(sample_chunks, query)
    assert len(ranked) == len(sample_chunks)
    # The first result should be more relevant to the query
    assert isinstance(ranked[0], Document)


def test_deduplicate_keeps_unique(ctx_mgr, sample_chunks):
    """Test that unique chunks are preserved."""
    result = ctx_mgr.deduplicate_chunks(sample_chunks, threshold=0.85)
    # All chunks are unique, so all should be kept
    assert len(result) == len(sample_chunks)
