from __future__ import annotations
"""Analysis agent for concept extraction."""

from typing import Dict
from .base_agent import BaseAgent


class AnalysisAgent(BaseAgent):
    name = "analysis_agent"
    description = "Extracts insights and key concepts from documents"

    def __init__(self):
        super().__init__()
        self._tool = None

    def _get_tool(self):
        if self._tool is None:
            from ..tools.concept_extraction import ConceptExtractionTool
            self._tool = ConceptExtractionTool()
        return self._tool

    def run(self, query: str, context: Dict = None) -> str:
        doc_name = (context or {}).get("document_name", query)
        max_concepts = (context or {}).get("max_concepts", 10)
        return self._get_tool().run(document_name=doc_name, max_concepts=max_concepts)
