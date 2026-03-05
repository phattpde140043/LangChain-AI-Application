from __future__ import annotations
"""POST /summarize endpoint."""

import time
from fastapi import APIRouter, HTTPException, Request

from ..schemas import SummarizeRequest, SummarizeResponse
from ...monitoring.logger import get_logger
from ...monitoring.metrics import MetricsCollector

router = APIRouter()
logger = get_logger(__name__)
metrics = MetricsCollector()

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: Request, body: SummarizeRequest) -> SummarizeResponse:
    """Summarize a document from the knowledge base."""
    start_time = time.time()
    
    try:
        from ...agents.summarization_agent import SummarizationAgent
        agent = SummarizationAgent()
        summary = agent.run(
            body.document_name,
            context={"document_name": body.document_name, "max_length": body.max_length},
        )
        
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_request("/summarize", latency_ms, success=True)
        
        return SummarizeResponse(
            summary=summary,
            document_name=body.document_name,
            latency_ms=round(latency_ms, 2),
        )
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_request("/summarize", latency_ms, success=False)
        logger.error(f"Summarize endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
