from __future__ import annotations
"""Concept extraction tool."""

from pydantic import BaseModel
from .base_tool import BaseTool, register_tool


class ConceptExtractionInput(BaseModel):
    document_name: str
    max_concepts: int = 10


@register_tool
class ConceptExtractionTool(BaseTool):
    name = "extract_concepts"
    description = "Extract key technical concepts from a document"
    input_schema = ConceptExtractionInput

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
            self._llm = ModelManager().get_llm("analyze")
        return self._llm

    def _run(self, input_data: ConceptExtractionInput) -> str:
        chunks = self._get_vector_store().similarity_search(
            f"key concepts {input_data.document_name}", k=8
        )
        if not chunks:
            return f"No content found for: {input_data.document_name}"

        content = "\n\n".join(doc.page_content for doc in chunks[:6])

        from pathlib import Path
        from ..config.settings import get_settings
        settings = get_settings()
        prompt_path = Path(settings.PROMPTS_DIR) / "concept_extraction_prompt.txt"

        if prompt_path.exists():
            prompt = prompt_path.read_text().replace("{content}", content).replace(
                "{max_concepts}", str(input_data.max_concepts)
            )
        else:
            prompt = f"Extract {input_data.max_concepts} key technical concepts from:\n\n{content}"

        try:
            result = self._get_llm().invoke(prompt)
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            return f"Concept extraction failed: {e}"
