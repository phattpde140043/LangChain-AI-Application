from __future__ import annotations
"""Performance metrics collection for API requests, retrievals, and embeddings."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel


class RequestMetric(BaseModel):
    endpoint: str
    latency_ms: float
    success: bool
    timestamp: str


class RetrievalMetric(BaseModel):
    query: str
    num_chunks: int
    latency_ms: float
    timestamp: str


class MetricsCollector:
    """Collects and persists performance metrics to a JSONL file."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        from ..config.settings import get_settings

        settings = get_settings()
        self.log_path = Path(log_path or settings.METRICS_LOG_PATH)
        self._requests: List[RequestMetric] = []
        self._retrievals: List[RetrievalMetric] = []

    def record_request(self, endpoint: str, latency_ms: float, success: bool) -> None:
        """Record a completed API request."""
        metric = RequestMetric(
            endpoint=endpoint,
            latency_ms=latency_ms,
            success=success,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._requests.append(metric)
        self._append_to_file({"type": "request", **metric.model_dump()})

    def record_retrieval(self, query: str, num_chunks: int, latency_ms: float) -> None:
        """Record a vector/hybrid retrieval operation."""
        metric = RetrievalMetric(
            query=query[:100],
            num_chunks=num_chunks,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._retrievals.append(metric)
        self._append_to_file({"type": "retrieval", **metric.model_dump()})

    def record_embedding(self, num_docs: int, latency_ms: float) -> None:
        """Record a batch embedding operation."""
        self._append_to_file(
            {
                "type": "embedding",
                "num_docs": num_docs,
                "latency_ms": latency_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_summary(self) -> Dict:
        """Return aggregated metrics across all recorded requests."""
        if not self._requests:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "total_retrievals": 0,
            }

        total = len(self._requests)
        successful = sum(1 for r in self._requests if r.success)
        avg_latency = sum(r.latency_ms for r in self._requests) / total

        return {
            "total_requests": total,
            "success_rate": round(successful / total, 3),
            "avg_latency_ms": round(avg_latency, 2),
            "total_retrievals": len(self._retrievals),
        }

    def _append_to_file(self, data: Dict) -> None:
        """Append a metric entry to the JSONL log file (best-effort)."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except Exception:
            pass
