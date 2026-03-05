from __future__ import annotations
"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# Set test environment variables BEFORE importing settings
os.environ.setdefault("ACTIVE_PROFILE", "testing")
os.environ.setdefault("ENABLE_SAFETY_CHECKS", "false")
os.environ.setdefault("ENABLE_COST_TRACKING", "false")
os.environ.setdefault("VECTOR_DB_TYPE", "faiss")


@pytest.fixture
def settings():
    """Return test settings."""
    from src.config.settings import Settings
    return Settings(
        ACTIVE_PROFILE="testing",
        ENABLE_SAFETY_CHECKS=False,
        ENABLE_COST_TRACKING=False,
        VECTOR_DB_TYPE="faiss",
        LOG_LEVEL="WARNING",
    )


@pytest.fixture
def sample_documents() -> List[Document]:
    """Return a list of sample Documents for testing."""
    return [
        Document(
            page_content="Microservices architecture decomposes applications into small, independent services.",
            metadata={"source": "test_doc1.md", "filename": "test_doc1.md", "file_type": "markdown", "page_number": 1},
        ),
        Document(
            page_content="API Gateway serves as the single entry point for all client requests.",
            metadata={"source": "test_doc2.md", "filename": "test_doc2.md", "file_type": "markdown", "page_number": 1},
        ),
        Document(
            page_content="Vector databases store high-dimensional embeddings for semantic search.",
            metadata={"source": "test_doc3.md", "filename": "test_doc3.md", "file_type": "markdown", "page_number": 1},
        ),
        Document(
            page_content="RAG combines retrieval with generation to ground LLM responses in facts.",
            metadata={"source": "test_doc4.md", "filename": "test_doc4.md", "file_type": "markdown", "page_number": 1},
        ),
    ]


@pytest.fixture
def mock_llm():
    """Return a mock LLM."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Mock LLM response for testing.")
    llm.predict.return_value = "Mock LLM response for testing."
    return llm


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """Return a temporary directory."""
    return tmp_path


@pytest.fixture
def vector_store(sample_documents, temp_dir):
    """Return an in-memory vector store for testing."""
    from unittest.mock import MagicMock
    store = MagicMock()
    store.similarity_search.return_value = sample_documents[:2]
    store.get_collection_stats.return_value = {
        "type": "faiss",
        "documents": ["test_doc1.md", "test_doc2.md"],
        "total_chunks": 4,
    }
    store.get_indexed_documents.return_value = ["test_doc1.md", "test_doc2.md"]
    return store
