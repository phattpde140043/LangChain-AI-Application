from __future__ import annotations
"""POST /ingest endpoint."""

import time
from fastapi import APIRouter, HTTPException, Request

from ..schemas import IngestRequest, IngestResponse
from ...monitoring.logger import get_logger
from ...monitoring.metrics import MetricsCollector

router = APIRouter()
logger = get_logger(__name__)
metrics = MetricsCollector()

@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, body: IngestRequest) -> IngestResponse:
    """Ingest documents into the knowledge base."""
    start_time = time.time()
    
    try:
        from ...ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        
        result = pipeline.ingest(documents_path=body.directory_path)
        
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_request("/ingest", latency_ms, success=True)
        
        return IngestResponse(
            status=result.get("status", "completed"),
            documents_processed=result.get("documents_processed", 0),
            chunks_created=result.get("chunks_created", 0),
            latency_ms=round(latency_ms, 2),
        )
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_request("/ingest", latency_ms, success=False)
        logger.error(f"Ingest endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
