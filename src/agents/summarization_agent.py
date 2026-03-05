from __future__ import annotations
"""Summarization agent."""

from typing import Dict
from .base_agent import BaseAgent


class SummarizationAgent(BaseAgent):
    name = "summarization_agent"
    description = "Summarizes documents from the knowledge base"

    def __init__(self):
        super().__init__()
        self._tool = None

    def _get_tool(self):
        if self._tool is None:
            from ..tools.summarization import SummarizationTool
            self._tool = SummarizationTool()
        return self._tool

    def run(self, query: str, context: Dict = None) -> str:
        """Extract document name from query and summarize it."""
        doc_name = (context or {}).get("document_name", query)
        max_length = (context or {}).get("max_length", 500)
        return self._get_tool().run(document_name=doc_name, max_length=max_length)
