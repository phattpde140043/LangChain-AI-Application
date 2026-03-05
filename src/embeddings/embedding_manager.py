from __future__ import annotations
"""Embedding manager with caching support for OpenAI and local models."""

import hashlib
import logging
from typing import Dict, List, Optional

from langchain_core.embeddings import Embeddings

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embeddings with an in-memory cache.

    Supports OpenAI embeddings when ``OPENAI_API_KEY`` is configured and
    falls back gracefully to a local HuggingFace model otherwise.
    """

    def __init__(self, model_name: str | None = None, provider: str | None = None) -> None:
        self.settings = get_settings()
        self.model_name: str = model_name or self.settings.EMBEDDING_MODEL
        self.provider: str = provider or self.settings.MODEL_PROVIDER
        self._cache: Dict[str, List[float]] = {}
        self._embeddings_model: Optional[Embeddings] = None

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def get_embeddings_model(self) -> Embeddings:
        """Return the LangChain Embeddings object, initialising it on first call.

        Uses OpenAI embeddings when the provider is ``"openai"`` and
        ``OPENAI_API_KEY`` is non-empty; otherwise falls back to a local
        HuggingFace model (``all-MiniLM-L6-v2``).
        """
        if self._embeddings_model is not None:
            return self._embeddings_model

        use_openai = (
            self.provider == "openai" and bool(self.settings.OPENAI_API_KEY)
        )

        if use_openai:
            try:
                from langchain_openai import OpenAIEmbeddings

                self._embeddings_model = OpenAIEmbeddings(
                    model=self.model_name,
                    openai_api_key=self.settings.OPENAI_API_KEY,
                )
                logger.info("Using OpenAI embeddings model: %s", self.model_name)
                return self._embeddings_model
            except Exception as exc:
                logger.warning("OpenAIEmbeddings init failed (%s); falling back to local model.", exc)

        # Local fallback — prefer the non-deprecated package when available
        try:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]

            local_model = "all-MiniLM-L6-v2"
            self._embeddings_model = HuggingFaceEmbeddings(model_name=local_model)
            logger.info("Using local HuggingFace embeddings model: %s", local_model)
        except Exception as exc:
            logger.error("HuggingFaceEmbeddings init failed (%s); using mock embeddings.", exc)
            self._embeddings_model = _MockEmbeddings()

        return self._embeddings_model

    # ------------------------------------------------------------------
    # Embedding methods
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string with caching.

        Args:
            text: The query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        key = self._get_cache_key(text)
        if key in self._cache:
            return self._cache[key]

        try:
            vector = self.get_embeddings_model().embed_query(text)
            self._cache[key] = vector
            return vector
        except Exception as exc:
            logger.error("embed_query failed: %s", exc)
            return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with per-item caching.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        keys = [self._get_cache_key(t) for t in texts]
        results: List[List[float] | None] = [self._cache.get(k) for k in keys]

        missing_indices = [i for i, v in enumerate(results) if v is None]
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            try:
                new_vectors = self.get_embeddings_model().embed_documents(missing_texts)
                for idx, vector in zip(missing_indices, new_vectors):
                    self._cache[keys[idx]] = vector
                    results[idx] = vector
            except Exception as exc:
                logger.error("embed_documents failed: %s", exc)
                for idx in missing_indices:
                    results[idx] = []

        return [v or [] for v in results]

    # ------------------------------------------------------------------
    # Cache utilities
    # ------------------------------------------------------------------

    def _get_cache_key(self, text: str) -> str:
        """Generate a deterministic cache key for a text string.

        MD5 is used here purely for speed and compactness as a cache lookup
        key — it has no security implications in this context.
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def clear_cache(self) -> None:
        """Clear the in-memory embedding cache."""
        self._cache.clear()
        logger.debug("Embedding cache cleared.")

    def get_cache_stats(self) -> Dict:
        """Return cache statistics.

        Returns:
            Dict with ``size`` (number of cached entries) and ``model`` name.
        """
        return {"size": len(self._cache), "model": self.model_name}


# ---------------------------------------------------------------------------
# Internal fallback
# ---------------------------------------------------------------------------

class _MockEmbeddings(Embeddings):
    """Zero-vector mock embeddings used when no real provider is available."""

    _DIM = 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * self._DIM for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.0] * self._DIM
