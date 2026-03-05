from __future__ import annotations
"""Document search tool using the vector store."""

from typing import Optional
from pydantic import BaseModel
from .base_tool import BaseTool, register_tool


class DocumentSearchInput(BaseModel):
    query: str
    k: int = 5
    document_filter: Optional[str] = None


@register_tool
class DocumentSearchTool(BaseTool):
    """Search documents in the knowledge base."""

    name = "document_search"
    description = "Search for relevant information in the indexed documents"
    input_schema = DocumentSearchInput

    def __init__(self):
        self._vector_store = None

    def _get_vector_store(self):
        if self._vector_store is None:
            from ..vector_store.store import VectorStoreManager
            self._vector_store = VectorStoreManager()
        return self._vector_store

    def _run(self, input_data: DocumentSearchInput) -> str:
        """Search documents and return formatted results."""
        results = self._get_vector_store().similarity_search(input_data.query, k=input_data.k)

        if not results:
            return "No relevant documents found."

        output = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            filename = doc.metadata.get("filename", source)
            page = doc.metadata.get("page_number", "")
            page_str = f", Page {page}" if page else ""
            preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            output.append(f"[Result {i}] Source: {filename}{page_str}\n{preview}")

        return "\n\n".join(output)
