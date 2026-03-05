from __future__ import annotations
"""Base agent class."""

from abc import ABC, abstractmethod
from typing import Dict


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    name: str = ""
    description: str = ""

    def __init__(self):
        self._llm = None
        self._tools = []

    def _get_llm(self, task: str = "default"):
        if self._llm is None:
            from ..models.model_manager import ModelManager
            self._llm = ModelManager().get_llm(task)
        return self._llm

    @abstractmethod
    def run(self, query: str, context: Dict = None) -> str:
        """Execute the agent with the given query."""
        ...
