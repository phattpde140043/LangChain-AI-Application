from __future__ import annotations
"""Comparison agent."""

from typing import Dict
from .base_agent import BaseAgent


class ComparisonAgent(BaseAgent):
    name = "comparison_agent"
    description = "Compares two documents"

    def __init__(self):
        super().__init__()
        self._tool = None

    def _get_tool(self):
        if self._tool is None:
            from ..tools.comparison import ComparisonTool
            self._tool = ComparisonTool()
        return self._tool

    def run(self, query: str, context: Dict = None) -> str:
        doc1 = (context or {}).get("document1", "")
        doc2 = (context or {}).get("document2", "")
        if not doc1 or not doc2:
            return "Please provide two document names to compare."
        return self._get_tool().run(document1=doc1, document2=doc2)
