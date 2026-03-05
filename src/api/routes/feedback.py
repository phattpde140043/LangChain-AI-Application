from __future__ import annotations
"""POST /feedback endpoint."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request

from ..schemas import FeedbackRequest, FeedbackResponse
from ...config.settings import get_settings
from ...monitoring.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: Request, body: FeedbackRequest) -> FeedbackResponse:
    """Submit feedback about a response."""
    settings = get_settings()
    feedback_id = str(uuid.uuid4())[:8]
    
    try:
        log_path = Path(settings.FEEDBACK_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        record = {
            "feedback_id": feedback_id,
            "query": body.query,
            "response": body.response[:500],  # Truncate for storage
            "rating": body.rating,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        
        return FeedbackResponse(status="recorded", feedback_id=feedback_id)
    
    except Exception as e:
        logger.error(f"Feedback endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
