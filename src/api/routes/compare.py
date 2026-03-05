from __future__ import annotations
"""POST /compare endpoint."""

import time
from typing import List
from fastapi import APIRouter, HTTPException, Request

from ..schemas import CompareRequest, CompareResponse
from ...monitoring.logger import get_logger
from ...monitoring.metrics import MetricsCollector

router = APIRouter()
logger = get_logger(__name__)
metrics = MetricsCollector()

@router.post("/compare", response_model=CompareResponse)
async def compare(request: Request, body: CompareRequest) -> CompareResponse:
    """Compare two documents from the knowledge base."""
    start_time = time.time()
    
    try:
        from ...agents.comparison_agent import ComparisonAgent
        agent = ComparisonAgent()
        comparison_text = agent.run(
            f"Compare {body.document1} and {body.document2}",
            context={"document1": body.document1, "document2": body.document2},
        )
        
        # Parse similarities and differences from the response
        similarities = _extract_list_section(comparison_text, "Similarities")
        differences = _extract_list_section(comparison_text, "Differences")
        
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_request("/compare", latency_ms, success=True)
        
        return CompareResponse(
            comparison=comparison_text,
            similarities=similarities,
            differences=differences,
            latency_ms=round(latency_ms, 2),
        )
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_request("/compare", latency_ms, success=False)
        logger.error(f"Compare endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _extract_list_section(text: str, section_name: str) -> List[str]:
    """Extract bullet points from a markdown section."""
    items = []
    in_section = False
    for line in text.splitlines():
        if section_name.lower() in line.lower() and "#" in line:
            in_section = True
            continue
        if in_section and line.startswith("#"):
            break
        if in_section and line.strip().startswith("-"):
            item = line.strip().lstrip("-").strip()
            if item:
                items.append(item)
    return items
