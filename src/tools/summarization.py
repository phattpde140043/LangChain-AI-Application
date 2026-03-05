from __future__ import annotations
"""Multi-stage document summarization tool."""

from pydantic import BaseModel
from .base_tool import BaseTool, register_tool


class SummarizationInput(BaseModel):
    document_name: str
    max_length: int = 500


@register_tool
class SummarizationTool(BaseTool):
    """Summarize a document from the knowledge base."""

    name = "summarize_document"
    description = "Create a concise summary of a document in the knowledge base"
    input_schema = SummarizationInput

    def __init__(self):
        self._vector_store = None
        self._llm = None

    def _get_vector_store(self):
        if self._vector_store is None:
            from ..vector_store.store import VectorStoreManager
            self._vector_store = VectorStoreManager()
        return self._vector_store

    def _get_llm(self):
        if self._llm is None:
            from ..models.model_manager import ModelManager
            self._llm = ModelManager().get_llm("summarize")
        return self._llm

    def _run(self, input_data: SummarizationInput) -> str:
        """Retrieve document chunks and generate a summary."""
        chunks = self._get_vector_store().similarity_search(
            f"summary of {input_data.document_name}", k=10
        )

        if not chunks:
            return f"No content found for document: {input_data.document_name}"

        content = "\n\n".join(doc.page_content for doc in chunks[:8])

        from pathlib import Path
        from ..config.settings import get_settings
        settings = get_settings()
        prompt_path = Path(settings.PROMPTS_DIR) / "summarization_prompt.txt"

        if prompt_path.exists():
            prompt_template = prompt_path.read_text()
            prompt = prompt_template.replace("{content}", content).replace(
                "{max_length}", str(input_data.max_length)
            )
        else:
            prompt = f"Summarize the following document in {input_data.max_length} words:\n\n{content}"

        try:
            result = self._get_llm().invoke(prompt)
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            return f"Summarization failed: {e}\n\nDocument preview:\n{content[:500]}"
