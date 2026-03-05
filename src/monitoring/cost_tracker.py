from __future__ import annotations
"""Token usage and cost tracking for LLM API calls."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel


class UsageRecord(BaseModel):
    """Record of a single LLM API call."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    query: str
    timestamp: str


class CostTracker:
    """Tracks token usage and estimated costs for every LLM call.

    Records are persisted to a JSONL file so they survive process restarts.
    """

    # Cost per 1 000 tokens (input / output) in USD — as of early 2025.
    COST_PER_1K_TOKENS: Dict[str, Dict[str, float]] = {
        "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
        "gpt-4o": {"input": 0.002500, "output": 0.010000},
        "gpt-3.5-turbo": {"input": 0.000500, "output": 0.001500},
        "text-embedding-3-small": {"input": 0.000020, "output": 0.0},
        "text-embedding-3-large": {"input": 0.000130, "output": 0.0},
    }

    def __init__(self, log_path: Optional[str] = None) -> None:
        from ..config.settings import get_settings

        settings = get_settings()
        self.log_path = Path(log_path or settings.COST_LOG_PATH)
        self._records: List[UsageRecord] = []

    def track_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        query: str = "",
    ) -> UsageRecord:
        """Record token usage and calculate the estimated USD cost."""
        costs = self.COST_PER_1K_TOKENS.get(model, {"input": 0.0, "output": 0.0})
        cost = (prompt_tokens / 1000 * costs["input"]) + (
            completion_tokens / 1000 * costs["output"]
        )

        record = UsageRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(cost, 8),
            latency_ms=latency_ms,
            query=query[:100] if query else "",
            timestamp=datetime.utcnow().isoformat(),
        )
        self._records.append(record)
        self._save_record(record)
        return record

    def get_stats(self) -> Dict:
        """Return aggregate statistics across all tracked calls."""
        if not self._records:
            return {
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_cost_usd": 0.0,
                "total_calls": 0,
            }

        total_tokens = sum(r.total_tokens for r in self._records)
        total_cost = sum(r.cost_usd for r in self._records)
        return {
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_usd": round(total_cost / len(self._records), 8),
            "total_calls": len(self._records),
        }

    def _save_record(self, record: UsageRecord) -> None:
        """Append a single record to the JSONL log file (best-effort)."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        except Exception:
            pass  # Never let logging failures propagate to callers
