from __future__ import annotations
"""Pydantic request/response schemas for the API."""

from typing import List, Optional
from pydantic import BaseModel, Field

class SourceRef(BaseModel):
    document_name: str
    page_number: Optional[int] = None
    section: Optional[str] = None

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    session_id: str = Field(default="default", description="Session ID for conversation history")
    stream: bool = Field(default=False, description="Whether to stream the response")

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceRef] = []
    confidence: str = "medium"
    agent_used: str = "research_agent"
    tokens_used: int = 0
    latency_ms: float = 0.0

class SummarizeRequest(BaseModel):
    document_name: str = Field(..., description="Name of the document to summarize")
    max_length: int = Field(default=500, ge=50, le=2000)

class SummarizeResponse(BaseModel):
    summary: str
    document_name: str
    latency_ms: float = 0.0

class CompareRequest(BaseModel):
    document1: str = Field(..., description="First document name")
    document2: str = Field(..., description="Second document name")

class CompareResponse(BaseModel):
    comparison: str
    similarities: List[str] = []
    differences: List[str] = []
    latency_ms: float = 0.0

class IngestRequest(BaseModel):
    directory_path: Optional[str] = Field(default=None, description="Path to documents directory. Uses default if not provided.")

class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    latency_ms: float = 0.0

class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: str = Field(..., pattern="^(helpful|not_helpful)$")

class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    profile: str

class KnowledgeBaseStats(BaseModel):
    documents: List[str] = []
    total_documents: int = 0
