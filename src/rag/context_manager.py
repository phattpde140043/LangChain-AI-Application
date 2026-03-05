from __future__ import annotations
"""Context window management: ranking, deduplication, compression, and selection."""

import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from ..monitoring.logger import get_logger

logger = get_logger(__name__)

_COMPRESSION_PROMPT = (
    "Extract only the sentences from the following text that are directly relevant "
    "to the query: '{query}'\n\nText:\n{text}\n\nRelevant sentences only:"
)


class ContextManager:
    """Prepares retrieved chunks for optimal LLM input.

    Responsibilities:
    * Rank chunks by keyword relevance to the query.
    * Remove near-duplicate chunks via Jaccard similarity.
    * Optionally compress each chunk with an LLM to strip irrelevant sentences.
    * Greedily select chunks that fit within a token budget.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank_chunks(self, chunks: List[Document], query: str) -> List[Document]:
        """Return *chunks* sorted by keyword-overlap relevance to *query*."""
        query_tokens = set(query.lower().split())
        if not query_tokens:
            return chunks

        scored: List[Tuple[float, Document]] = []
        for doc in chunks:
            doc_tokens = set(doc.page_content.lower().split())
            overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored]

    def deduplicate_chunks(
        self,
        chunks: List[Document],
        threshold: float = 0.85,
    ) -> List[Document]:
        """Remove chunks that are near-duplicates of an already-kept chunk.

        Two chunks are considered near-duplicates when their Jaccard token
        similarity exceeds *threshold*.
        """
        unique: List[Document] = []
        for candidate in chunks:
            is_duplicate = any(
                self._jaccard_similarity(candidate.page_content, kept.page_content)
                >= threshold
                for kept in unique
            )
            if not is_duplicate:
                unique.append(candidate)
        return unique

    def compress_context(
        self,
        chunks: List[Document],
        llm,
        query: str = "",
        max_tokens: int = 2000,
    ) -> List[Document]:
        """Use *llm* to strip sentences irrelevant to *query* from each chunk.

        Falls back to the original chunk on any per-chunk error.  When *query*
        is empty the original chunks are returned unchanged since meaningful
        compression requires a target query.
        """
        if not query.strip():
            return chunks

        compressed: List[Document] = []
        for doc in chunks:
            try:
                prompt = _COMPRESSION_PROMPT.format(
                    query=query, text=doc.page_content
                )
                result = llm.invoke(prompt)
                content = result.content if hasattr(result, "content") else str(result)
                compressed.append(
                    Document(
                        page_content=content.strip() or doc.page_content,
                        metadata=doc.metadata,
                    )
                )
            except Exception as e:
                logger.warning(f"Chunk compression failed, keeping original: {e}")
                compressed.append(doc)
        return compressed

    def select_context(
        self,
        chunks: List[Document],
        max_tokens: int = 4000,
    ) -> List[Document]:
        """Greedily select chunks until the token budget is exhausted."""
        selected: List[Document] = []
        used_tokens = 0
        for doc in chunks:
            chunk_tokens = self._count_tokens(doc.page_content)
            if used_tokens + chunk_tokens > max_tokens:
                break
            selected.append(doc)
            used_tokens += chunk_tokens
        return selected

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Return the token count for *text* using tiktoken when available."""
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Rough approximation: 1 token ≈ 4 characters.
            return len(text) // 4

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Return the Jaccard similarity of two texts based on word tokens."""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union else 0.0
