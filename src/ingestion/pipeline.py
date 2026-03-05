from __future__ import annotations
"""Full ingestion pipeline: load -> chunk -> embed -> store."""

import os
import time
from pathlib import Path
from typing import Dict, List

from ..config.settings import get_settings
from ..monitoring.logger import get_logger
from .document_loader import DocumentLoader
from .chunker import TextChunker

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates document ingestion: load, chunk, embed, and store."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self._vector_store = None  # lazy init to avoid circular imports

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_vector_store(self):
        """Lazy-load vector store to avoid circular imports."""
        if self._vector_store is None:
            from ..vector_store.store import VectorStoreManager

            self._vector_store = VectorStoreManager()
        return self._vector_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, documents_path: str | None = None) -> Dict:
        """Run the full pipeline on a directory of documents.

        Args:
            documents_path: Path to the directory. Defaults to ``DOCUMENTS_DIR``
                from settings.

        Returns:
            Summary dict with keys ``status``, ``documents_processed``,
            ``chunks_created``, and ``latency_ms``.
        """
        path = documents_path or self.settings.DOCUMENTS_DIR
        start = time.monotonic()
        try:
            docs = self.loader.load_directory(path)
            if not docs:
                logger.warning("No documents found in %s", path)
                return {
                    "status": "success",
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "latency_ms": 0,
                }

            chunks = self.chunker.chunk_documents(
                docs,
                chunk_size=self.settings.CHUNK_SIZE,
                chunk_overlap=self.settings.CHUNK_OVERLAP,
            )
            self._get_vector_store().add_documents(chunks)

            latency_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "Ingestion complete: %d docs, %d chunks, %d ms",
                len(docs),
                len(chunks),
                latency_ms,
            )
            return {
                "status": "success",
                "documents_processed": len(docs),
                "chunks_created": len(chunks),
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def ingest_file(self, file_path: str) -> Dict:
        """Ingest a single file into the vector store.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            Summary dict with ingestion results.
        """
        start = time.monotonic()
        try:
            ext = Path(file_path).suffix.lower()
            docs = self.loader.load_file(file_path)
            if not docs:
                logger.warning("No content loaded from %s", file_path)
                return {"status": "success", "documents_processed": 0, "chunks_created": 0, "latency_ms": 0}

            chunks = self.chunker.chunk_documents(
                docs,
                chunk_size=self.settings.CHUNK_SIZE,
                chunk_overlap=self.settings.CHUNK_OVERLAP,
            )
            self._get_vector_store().add_documents(chunks)

            latency_ms = int((time.monotonic() - start) * 1000)
            logger.info("Ingested file %s: %d chunks in %d ms", file_path, len(chunks), latency_ms)
            return {
                "status": "success",
                "documents_processed": len(docs),
                "chunks_created": len(chunks),
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            logger.error("Failed to ingest file %s: %s", file_path, exc)
            return {"status": "error", "error": str(exc)}

    def re_index(self, document_name: str) -> Dict:
        """Re-index a specific document by name.

        Deletes existing vectors for the document and re-ingests it from
        ``DOCUMENTS_DIR``.

        Args:
            document_name: Filename (with or without extension) to re-index.

        Returns:
            Summary dict from :meth:`ingest_file`.
        """
        docs_dir = Path(self.settings.DOCUMENTS_DIR)
        candidates = list(docs_dir.rglob(document_name))
        if not candidates:
            logger.error("Document not found for re-indexing: %s", document_name)
            return {"status": "error", "error": f"Document '{document_name}' not found"}

        file_path = str(candidates[0])
        try:
            # delete_collection removes all stored vectors tagged with this document name
            self._get_vector_store().delete_collection(document_name)
        except Exception as exc:
            logger.warning("Could not delete old vectors for %s: %s", document_name, exc)

        return self.ingest_file(file_path)

    def get_indexed_documents(self) -> List[str]:
        """Return list of indexed document names from the vector store."""
        try:
            return self._get_vector_store().get_indexed_documents()
        except Exception as exc:
            logger.error("Failed to retrieve indexed documents: %s", exc)
            return []

    def delete_document(self, document_name: str) -> bool:
        """Delete a document from the vector store.

        Args:
            document_name: Name of the document to remove.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        try:
            self._get_vector_store().delete_collection(document_name)
            logger.info("Deleted document: %s", document_name)
            return True
        except Exception as exc:
            logger.error("Failed to delete document %s: %s", document_name, exc)
            return False
