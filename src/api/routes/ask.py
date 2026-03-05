from __future__ import annotations
"""POST /ask endpoint."""

import time
import json
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..schemas import AskRequest, AskResponse, SourceRef
from ...monitoring.logger import get_logger
from ...monitoring.metrics import MetricsCollector
from ...config.settings import get_settings

router = APIRouter()
logger = get_logger(__name__)
metrics = MetricsCollector()

@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    """Answer a question using the AI research assistant."""
    start_time = time.time()
    settings = get_settings()
    
    # Safety check
    if settings.ENABLE_SAFETY_CHECKS:
        from ...security.guardrails import GuardrailsManager
        guardrails = GuardrailsManager()
        result = guardrails.check_input(body.question)
        if not result.is_safe:
            raise HTTPException(status_code=400, detail=f"Input blocked: {result.reason}")
    
    # Handle streaming
    if body.stream and settings.ENABLE_STREAMING:
        return StreamingResponse(
            _stream_response(body.question, body.session_id),
            media_type="text/event-stream",
        )
    
    # Standard response
    try:
        from ...agents.coordinator import CoordinatorAgent
        coordinator = CoordinatorAgent()
        agent_response = coordinator.route(body.question, session_id=body.session_id)
        
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_request("/ask", latency_ms, success=True)
        
        sources = []
        # Try to get sources from RAG pipeline if research intent
        try:
            from ...rag.pipeline import RAGPipeline
            rag = RAGPipeline()
            rag_response = rag.run(body.question, session_id=body.session_id)
            sources = [
                SourceRef(
                    document_name=s.document_name,
                    page_number=s.page_number,
                    section=s.section,
                )
                for s in rag_response.sources
            ]
            answer = rag_response.answer
            confidence = rag_response.confidence
        except Exception:
            answer = agent_response.answer
            confidence = agent_response.confidence
        
        return AskResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            agent_used=agent_response.agent_used,
            tokens_used=0,  # Would be populated by cost tracker in production
            latency_ms=round(latency_ms, 2),
        )
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_request("/ask", latency_ms, success=False)
        logger.error(f"Ask endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(question: str, session_id: str) -> AsyncGenerator[str, None]:
    """Generate SSE stream for the response."""
    try:
        from ...rag.pipeline import RAGPipeline
        rag = RAGPipeline()
        response = rag.run(question, session_id=session_id)
        
        # Stream word by word
        words = response.answer.split()
        for word in words:
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
        
        # Send sources
        sources_data = [{"document_name": s.document_name} for s in response.sources]
        yield f"data: {json.dumps({'sources': sources_data, 'done': True})}\n\n"
    
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
