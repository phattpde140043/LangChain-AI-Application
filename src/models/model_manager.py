from __future__ import annotations
"""Multi-model LLM manager with routing based on task type."""

from typing import Optional
from langchain_core.language_models import BaseLLM, BaseLanguageModel
from langchain_core.embeddings import Embeddings

MODEL_ROUTING = {
    "summarize": "gpt-4o-mini",
    "analyze": "gpt-4o-mini",
    "compare": "gpt-4o-mini",
    "research": "gpt-4o-mini",
    "default": "gpt-4o-mini",
}


class ModelManager:
    """Manages LLM and embedding model instantiation with task-based routing."""

    def __init__(self):
        from ..config.settings import get_settings
        self.settings = get_settings()
        self._llm_cache: dict = {}
        self._embeddings: Optional[Embeddings] = None

    def get_llm(self, task: str = "default") -> BaseLanguageModel:
        """Get LLM for a specific task. Routes to appropriate model."""
        model_name = MODEL_ROUTING.get(task, self.settings.MODEL_NAME)

        if model_name in self._llm_cache:
            return self._llm_cache[model_name]

        llm = self._create_llm(model_name)
        self._llm_cache[model_name] = llm
        return llm

    def _create_llm(self, model_name: str) -> BaseLanguageModel:
        """Create an LLM instance. Falls back gracefully if API key not set."""
        provider = self.settings.MODEL_PROVIDER

        if provider == "openai" and self.settings.OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model_name,
                    openai_api_key=self.settings.OPENAI_API_KEY,
                    temperature=0.1,
                    max_tokens=2000,
                )
            except Exception as e:
                from ..monitoring.logger import get_logger
                get_logger(__name__).warning(f"Failed to create OpenAI LLM: {e}. Using mock.")

        # Fallback: return a mock LLM for testing/development
        return _MockLLM()

    def get_embeddings(self) -> Embeddings:
        """Get the embeddings model."""
        if self._embeddings is None:
            from ..embeddings.embedding_manager import EmbeddingManager
            self._embeddings = EmbeddingManager().get_embeddings_model()
        return self._embeddings


class _MockLLM(BaseLanguageModel):
    """Mock LLM for use when no API key is configured."""

    def predict(self, text: str, **kwargs) -> str:
        return "This is a mock response. Please configure OPENAI_API_KEY to use real LLM capabilities."

    def predict_messages(self, messages, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.predict(""))

    def invoke(self, input, config=None, **kwargs):
        from langchain_core.messages import AIMessage
        if isinstance(input, str):
            return AIMessage(content=self.predict(input))
        return AIMessage(content=self.predict(""))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage
        msg = AIMessage(content=self.predict(""))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "mock"
