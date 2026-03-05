from __future__ import annotations
"""Citation finding tool."""

from typing import Optional
from pydantic import BaseModel
from .base_tool import BaseTool, register_tool


class CitationInput(BaseModel):
    concept: str
    document_name: Optional[str] = None


@register_tool
class CitationFinderTool(BaseTool):
    name = "find_citations"
    description = "Find relevant citations and references for a concept in the knowledge base"
    input_schema = CitationInput

    def __init__(self):
        self._vector_store = None

    def _get_vector_store(self):
        if self._vector_store is None:
            from ..vector_store.store import VectorStoreManager
            self._vector_store = VectorStoreManager()
        return self._vector_store

    def _run(self, input_data: CitationInput) -> str:
        query = input_data.concept
        results = self._get_vector_store().similarity_search(query, k=5)

        if not results:
            return f"No citations found for: {input_data.concept}"

        citations = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page_number", "")
            page_str = f", p.{page}" if page else ""
            preview = doc.page_content[:200].strip()
            citations.append(f'[{i}] {source}{page_str}: "{preview}..."')

        return "\n\n".join(citations)
