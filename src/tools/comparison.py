from __future__ import annotations
"""Document comparison tool."""

from pydantic import BaseModel
from .base_tool import BaseTool, register_tool


class ComparisonInput(BaseModel):
    document1: str
    document2: str


@register_tool
class ComparisonTool(BaseTool):
    name = "compare_documents"
    description = "Compare two documents and identify similarities and differences"
    input_schema = ComparisonInput

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
            self._llm = ModelManager().get_llm("compare")
        return self._llm

    def _run(self, input_data: ComparisonInput) -> str:
        store = self._get_vector_store()

        chunks1 = store.similarity_search(f"overview of {input_data.document1}", k=5)
        chunks2 = store.similarity_search(f"overview of {input_data.document2}", k=5)

        content1 = "\n".join(doc.page_content for doc in chunks1[:4]) if chunks1 else f"No content for {input_data.document1}"
        content2 = "\n".join(doc.page_content for doc in chunks2[:4]) if chunks2 else f"No content for {input_data.document2}"

        from pathlib import Path
        from ..config.settings import get_settings
        settings = get_settings()
        prompt_path = Path(settings.PROMPTS_DIR) / "comparison_prompt.txt"

        if prompt_path.exists():
            prompt = prompt_path.read_text().replace("{content1}", content1).replace("{content2}", content2)
        else:
            prompt = f"Compare these two documents:\n\nDocument 1:\n{content1}\n\nDocument 2:\n{content2}"

        try:
            result = self._get_llm().invoke(prompt)
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            return f"Comparison failed: {e}"
