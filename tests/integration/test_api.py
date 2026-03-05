from __future__ import annotations
"""Integration tests for the FastAPI application."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    with patch("src.config.settings.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            API_KEY="",
            ENABLE_SAFETY_CHECKS=False,
            ENABLE_COST_TRACKING=False,
            ENABLE_STREAMING=False,
            ACTIVE_PROFILE="testing",
            LOG_LEVEL="WARNING",
            MODEL_NAME="gpt-4o-mini",
            ACTIVE_RAG_PROMPT="rag_prompt_v1",
            ENABLE_HYBRID_RETRIEVAL=False,
            ENABLE_QUERY_EXPANSION=False,
            ENABLE_CONTEXT_COMPRESSION=False,
            MAX_CONTEXT_TOKENS=4000,
            RETRIEVAL_TOP_K=5,
            FEEDBACK_LOG_PATH="/tmp/test_feedback.jsonl",
        )
        from src.api.main import app
        return TestClient(app)


def test_health_endpoint(client):
    """Test that the health endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "profile" in data


def test_ask_requires_question(client):
    """Test that ask endpoint requires a question field."""
    response = client.post("/ask", json={})
    assert response.status_code == 422  # Validation error


def test_ask_with_valid_question(client):
    """Test ask endpoint with a valid question."""
    with patch("src.agents.coordinator.CoordinatorAgent") as MockCoordinator, \
         patch("src.rag.pipeline.RAGPipeline") as MockPipeline:

        mock_agent_response = MagicMock()
        mock_agent_response.answer = "Test answer"
        mock_agent_response.agent_used = "research_agent"
        mock_agent_response.confidence = "medium"
        MockCoordinator.return_value.route.return_value = mock_agent_response

        mock_rag_response = MagicMock()
        mock_rag_response.answer = "Test answer from RAG"
        mock_rag_response.sources = []
        mock_rag_response.confidence = "medium"
        MockPipeline.return_value.run.return_value = mock_rag_response

        response = client.post("/ask", json={"question": "What is microservices?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data


def test_ingest_endpoint(client):
    """Test ingest endpoint."""
    with patch("src.ingestion.pipeline.IngestionPipeline") as MockPipeline:
        MockPipeline.return_value.ingest.return_value = {
            "status": "success",
            "documents_processed": 3,
            "chunks_created": 15,
        }
        response = client.post("/ingest", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


def test_feedback_endpoint(client):
    """Test feedback endpoint."""
    response = client.post("/feedback", json={
        "query": "What is microservices?",
        "response": "Microservices are independent services.",
        "rating": "helpful",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "recorded"
    assert "feedback_id" in data


def test_knowledge_base_endpoint(client):
    """Test knowledge base endpoint."""
    with patch("src.ingestion.pipeline.IngestionPipeline") as MockPipeline:
        MockPipeline.return_value.get_indexed_documents.return_value = ["doc1.md", "doc2.md"]
        response = client.get("/knowledge-base")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total_documents" in data


def test_summarize_endpoint(client):
    """Test summarize endpoint."""
    with patch("src.agents.summarization_agent.SummarizationAgent") as MockAgent:
        MockAgent.return_value.run.return_value = "This is a test summary."
        response = client.post("/summarize", json={"document_name": "test_doc.md"})
        assert response.status_code == 200


def test_compare_endpoint(client):
    """Test compare endpoint."""
    with patch("src.agents.comparison_agent.ComparisonAgent") as MockAgent:
        MockAgent.return_value.run.return_value = "### Similarities\n- Both use APIs\n### Differences\n- Different scales"
        response = client.post("/compare", json={
            "document1": "doc1.md",
            "document2": "doc2.md",
        })
        assert response.status_code == 200
