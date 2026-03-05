from __future__ import annotations
"""Full RAG pipeline: query expansion → retrieval → context management → generation."""

import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from ..config.settings import get_settings
from ..monitoring.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_PROMPT_TEMPLATE = (
    "You are an expert AI research assistant. Use the following retrieved context to "
    "answer the user's question accurately and concisely.\n\n"
    "If the context does not contain enough information to answer the question, say so "
    "clearly — do not fabricate an answer.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class Source(BaseModel):
    """Reference to a retrieved source document chunk."""

    document_name: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_text_preview: str = ""


class RAGResponse(BaseModel):
    """Structured response produced by :class:`RAGPipeline`."""

    answer: str
    sources: List[Source] = []
    confidence: str = "medium"
    relevant_concepts: List[str] = []


class RAGPipeline:
    """Orchestrates the end-to-end RAG workflow.

    Steps performed by :meth:`run`:

    1. Query expansion (optional, controlled by settings flag).
    2. Hybrid or vector-only retrieval for each expanded query.
    3. Deduplication and relevance ranking of retrieved chunks.
    4. Token-budget-aware context selection.
    5. LLM generation using the configured prompt template.
    6. Structured response assembly with source attribution.
    """

    def __init__(self, llm=None, vector_store=None) -> None:
        self.settings = get_settings()
        self._llm = llm
        self._vector_store = vector_store
        self._retriever = None
        self._query_expander = None
        self._context_manager = None

    # ------------------------------------------------------------------
    # Lazy-loaded sub-components
    # ------------------------------------------------------------------

    def _get_llm(self):
        """Lazy-load the LLM from ModelManager if not injected."""
        if self._llm is None:
            from ..models.model_manager import ModelManager

            self._llm = ModelManager().get_llm()
        return self._llm

    def _get_retriever(self):
        if self._retriever is None:
            from .retriever import HybridRetriever

            self._retriever = HybridRetriever(vector_store=self._vector_store)
        return self._retriever

    def _get_query_expander(self):
        if self._query_expander is None:
            from .query_expansion import QueryExpander

            self._query_expander = QueryExpander()
        return self._query_expander

    def _get_context_manager(self):
        if self._context_manager is None:
            from .context_manager import ContextManager

            self._context_manager = ContextManager()
        return self._context_manager

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    def _load_prompt_template(self) -> PromptTemplate:
        """Load the prompt template named by ``settings.ACTIVE_RAG_PROMPT``.

        Searches the configured ``PROMPTS_DIR`` for a ``.txt`` file whose stem
        matches the active prompt name.  Falls back to the built-in default
        template if the file cannot be found or read.
        """
        try:
            prompts_dir = Path(self.settings.PROMPTS_DIR)
            prompt_file = prompts_dir / f"{self.settings.ACTIVE_RAG_PROMPT}.txt"
            if prompt_file.is_file():
                template_text = prompt_file.read_text(encoding="utf-8")
                return PromptTemplate(
                    input_variables=["context", "question"],
                    template=template_text,
                )
            logger.warning(
                f"Prompt file '{prompt_file}' not found; using default template."
            )
        except Exception as e:
            logger.warning(f"Could not load prompt template: {e}")
        return PromptTemplate(
            input_variables=["context", "question"],
            template=_DEFAULT_PROMPT_TEMPLATE,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, query: str, session_id: str = "default") -> RAGResponse:
        """Execute the full RAG pipeline and return a structured response.

        Args:
            query:      The user's natural-language question.
            session_id: Identifier for the user session (reserved for future
                        memory integration).
        """
        start_time = time.time()

        try:
            # 1. Query expansion ------------------------------------------------
            queries: List[str] = [query]
            if self.settings.ENABLE_QUERY_EXPANSION:
                try:
                    queries = self._get_query_expander().expand_query(
                        query, self._get_llm()
                    )
                except Exception as e:
                    logger.warning(f"Query expansion skipped: {e}")
                    queries = [query]

            # 2. Retrieval -------------------------------------------------------
            all_chunks: List[Document] = []
            retriever = self._get_retriever()
            for q in queries[:2]:  # cap at 2 to limit API calls
                try:
                    if self.settings.ENABLE_HYBRID_RETRIEVAL:
                        chunks = retriever.hybrid_search(
                            q, k=self.settings.RETRIEVAL_TOP_K
                        )
                    else:
                        chunks = retriever.vector_search(
                            q, k=self.settings.RETRIEVAL_TOP_K
                        )
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"Retrieval failed for query '{q}': {e}")

            # 3. Context management ---------------------------------------------
            ctx_mgr = self._get_context_manager()
            chunks = ctx_mgr.deduplicate_chunks(all_chunks)
            chunks = ctx_mgr.rank_chunks(chunks, query)
            chunks = ctx_mgr.select_context(
                chunks, max_tokens=self.settings.MAX_CONTEXT_TOKENS
            )

            # 4. Guard against empty context ------------------------------------
            if not chunks:
                return RAGResponse(
                    answer=(
                        "No relevant documents found. "
                        "Please ingest documents before querying."
                    ),
                    sources=[],
                    confidence="low",
                )

            # 5. Generation -----------------------------------------------------
            context_text = self._format_context(chunks)
            prompt = self._load_prompt_template()
            llm = self._get_llm()
            chain = prompt | llm
            result = chain.invoke({"context": context_text, "question": query})
            answer_text = (
                result.content if hasattr(result, "content") else str(result)
            )

            # 6. Assemble response ----------------------------------------------
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"RAG pipeline completed in {elapsed_ms:.0f} ms "
                f"({len(chunks)} chunks used)."
            )

            return RAGResponse(
                answer=answer_text,
                sources=self._extract_sources(chunks),
                confidence=self._estimate_confidence(chunks, answer_text),
                relevant_concepts=self._extract_concepts(answer_text),
            )

        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}", exc_info=True)
            return RAGResponse(
                answer=f"An error occurred while processing your query: {e}",
                sources=[],
                confidence="low",
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_context(self, chunks: List[Document]) -> str:
        """Render retrieved chunks as a numbered context string."""
        parts: List[str] = []
        for i, doc in enumerate(chunks, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            header = f"[{i}] Source: {source}"
            if page:
                header += f", Page: {page}"
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(parts)

    def _extract_sources(self, chunks: List[Document]) -> List[Source]:
        """Build a deduplicated list of :class:`Source` objects from *chunks*."""
        seen: set = set()
        sources: List[Source] = []
        for doc in chunks:
            meta = doc.metadata
            doc_name = str(meta.get("source", "Unknown"))
            page = meta.get("page")
            section = meta.get("section")
            key = (doc_name, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                Source(
                    document_name=doc_name,
                    page_number=int(page) if page is not None else None,
                    section=str(section) if section else None,
                    chunk_text_preview=doc.page_content[:120],
                )
            )
        return sources

    def _estimate_confidence(self, chunks: List[Document], answer: str) -> str:
        """Estimate answer confidence from the number of supporting chunks."""
        n = len(chunks)
        if n >= 3:
            return "high"
        if n >= 1:
            return "medium"
        return "low"

    def _extract_concepts(self, answer: str) -> List[str]:
        """Extract bolded terms or capitalised multi-word phrases from *answer*."""
        concepts: List[str] = []

        # Markdown bold: **term**
        bold_terms = re.findall(r"\*\*(.+?)\*\*", answer)
        concepts.extend(bold_terms)

        # Capitalised phrases (2+ words starting with capital letters)
        cap_phrases = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b", answer)
        concepts.extend(cap_phrases)

        # Deduplicate while preserving order, limit to 10.
        seen: set = set()
        unique: List[str] = []
        for c in concepts:
            c_stripped = c.strip()
            if c_stripped and c_stripped not in seen:
                seen.add(c_stripped)
                unique.append(c_stripped)
        return unique[:10]
