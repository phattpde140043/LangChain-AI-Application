from __future__ import annotations
"""Hybrid retrieval combining vector similarity search and BM25 keyword search."""

from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from ..monitoring.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Combines vector similarity search with BM25 keyword retrieval.

    Results from both searches are score-normalised and then merged using a
    configurable alpha weight before deduplication.
    """

    def __init__(self, vector_store=None) -> None:
        from ..config.settings import get_settings

        self.settings = get_settings()
        self._vector_store = vector_store
        self._bm25_index = None
        self._indexed_docs: List[Document] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_vector_store(self):
        """Lazy-load the vector store on first use."""
        if self._vector_store is None:
            from ..vector_store.store import VectorStoreManager

            self._vector_store = VectorStoreManager()
        return self._vector_store

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------

    def vector_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform a vector similarity search and return the top-k documents."""
        try:
            return self._get_vector_store().similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform BM25 keyword search over the in-memory indexed documents.

        Returns an empty list when no BM25 index has been built yet.
        """
        if self._bm25_index is None or not self._indexed_docs:
            logger.debug("BM25 index not available; skipping keyword search.")
            return []

        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            query_tokens = query.lower().split()
            scores = self._bm25_index.get_scores(query_tokens)

            # Pair each document with its BM25 score and sort descending.
            ranked = sorted(
                zip(scores, self._indexed_docs), key=lambda x: x[0], reverse=True
            )
            results: List[Document] = []
            for score, doc in ranked[:k]:
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "bm25_score": float(score)},
                )
                results.append(doc_copy)
            return results
        except Exception as e:
            logger.error(f"BM25 keyword search failed: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.7,
    ) -> List[Document]:
        """Return the top-k documents by combining vector and keyword scores.

        Args:
            query: User search query.
            k: Number of results to return.
            alpha: Weight for the vector score; ``1 - alpha`` is given to BM25.
        """
        vector_docs = self.vector_search(query, k=k * 2)
        keyword_docs = self.keyword_search(query, k=k * 2)

        def _normalize(docs: List[Document], score_key: str) -> Dict[str, float]:
            """Return a content→normalised-score mapping (0–1)."""
            scores = [doc.metadata.get(score_key, 0.0) for doc in docs]
            max_s = max(scores) if scores else 1.0
            min_s = min(scores) if scores else 0.0
            rng = max_s - min_s or 1.0
            return {
                doc.page_content: (doc.metadata.get(score_key, 0.0) - min_s) / rng
                for doc in docs
            }

        vector_scores = _normalize(vector_docs, "score")
        keyword_scores = _normalize(keyword_docs, "bm25_score")

        # Union of all seen content strings.
        all_content: Dict[str, Document] = {}
        for doc in vector_docs + keyword_docs:
            all_content.setdefault(doc.page_content, doc)

        # Compute hybrid score for every unique chunk.
        scored: List[Tuple[float, Document]] = []
        for content, doc in all_content.items():
            v_score = vector_scores.get(content, 0.0)
            k_score = keyword_scores.get(content, 0.0)
            hybrid = alpha * v_score + (1.0 - alpha) * k_score
            scored.append((hybrid, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Document] = []
        for score, doc in scored[:k]:
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "hybrid_score": round(score, 4)},
            )
            results.append(doc_copy)
        return results

    def apply_context_filter(
        self,
        docs: List[Document],
        threshold: float = 0.5,
    ) -> List[Document]:
        """Remove documents whose relevance score falls below *threshold*.

        Documents without an explicit ``relevance_score`` metadata field are
        always kept.
        """
        filtered: List[Document] = []
        for doc in docs:
            score = doc.metadata.get("relevance_score")
            if score is None or float(score) >= threshold:
                filtered.append(doc)
        return filtered

    def update_bm25_index(self, docs: List[Document]) -> None:
        """Rebuild the BM25 index from *docs*.

        Call this after new documents are ingested so keyword search stays
        up-to-date.
        """
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self._indexed_docs = list(docs)
            tokenized = [doc.page_content.lower().split() for doc in docs]
            self._bm25_index = BM25Okapi(tokenized)
            logger.info(f"BM25 index updated with {len(docs)} documents.")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
