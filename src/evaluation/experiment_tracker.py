from __future__ import annotations
"""Experiment tracking for RAG pipeline experiments."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class ExperimentTracker:
    """Tracks experiments and their metrics."""

    def __init__(self, storage_path: str = "./data/experiments.json"):
        self.storage_path = Path(storage_path)
        self._experiments: Dict[str, Dict] = {}
        self._load_experiments()

    def start_experiment(self, name: str, config: Dict) -> str:
        """Start a new experiment and return its ID."""
        experiment_id = str(uuid.uuid4())[:8]
        self._experiments[experiment_id] = {
            "id": experiment_id,
            "name": name,
            "config": config,
            "metrics": {},
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "ended_at": None,
        }
        self._save_experiments()
        return experiment_id

    def log_metric(self, experiment_id: str, metric: str, value: float) -> None:
        """Log a metric value for an experiment."""
        if experiment_id in self._experiments:
            self._experiments[experiment_id]["metrics"][metric] = value
            self._save_experiments()

    def end_experiment(self, experiment_id: str, results: Dict) -> None:
        """Mark an experiment as complete with final results."""
        if experiment_id in self._experiments:
            self._experiments[experiment_id].update({
                "status": "completed",
                "results": results,
                "ended_at": datetime.utcnow().isoformat(),
            })
            self._save_experiments()

    def get_experiments(self) -> List[Dict]:
        """Return all experiments."""
        return list(self._experiments.values())

    def _load_experiments(self) -> None:
        try:
            if self.storage_path.exists():
                with open(self.storage_path) as f:
                    self._experiments = json.load(f)
        except Exception:
            self._experiments = {}

    def _save_experiments(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(self._experiments, f, indent=2)
        except Exception:
            pass
