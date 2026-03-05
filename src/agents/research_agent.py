from __future__ import annotations
"""Research agent using RAG for complex question answering."""

from typing import Dict
from .base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    """Answers complex research questions using the RAG pipeline."""

    name = "research_agent"
    description = "Answers complex questions using document retrieval and reasoning"

    def __init__(self):
        super().__init__()
        self._rag_pipeline = None

    def _get_rag_pipeline(self):
        if self._rag_pipeline is None:
            from ..rag.pipeline import RAGPipeline
            self._rag_pipeline = RAGPipeline()
        return self._rag_pipeline

    def run(self, query: str, context: Dict = None) -> str:
        """Run RAG pipeline and return formatted answer."""
        session_id = (context or {}).get("session_id", "default")
        response = self._get_rag_pipeline().run(query, session_id=session_id)
        return response.answer
