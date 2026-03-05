from __future__ import annotations
"""Document loading module supporting PDF, text, and markdown files."""

import logging
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents from various file formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".text"}

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_pdf(self, path: str) -> List[Document]:
        """Load a PDF file and return a list of Documents with metadata."""
        from langchain_community.document_loaders import PyPDFLoader

        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            return self._add_metadata(docs, path, "pdf")
        except Exception as exc:
            logger.error("Failed to load PDF %s: %s", path, exc)
            return []

    def load_text(self, path: str) -> List[Document]:
        """Load a plain text file."""
        from langchain_community.document_loaders import TextLoader

        try:
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            return self._add_metadata(docs, path, "text")
        except Exception as exc:
            logger.error("Failed to load text file %s: %s", path, exc)
            return []

    def load_markdown(self, path: str) -> List[Document]:
        """Load a markdown file, falling back to TextLoader if needed."""
        try:
            from langchain_community.document_loaders import UnstructuredMarkdownLoader

            loader = UnstructuredMarkdownLoader(path)
            docs = loader.load()
            return self._add_metadata(docs, path, "markdown")
        except Exception:
            logger.debug("UnstructuredMarkdownLoader unavailable; falling back to TextLoader for %s", path)
            try:
                from langchain_community.document_loaders import TextLoader

                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
                return self._add_metadata(docs, path, "markdown")
            except Exception as exc:
                logger.error("Failed to load markdown file %s: %s", path, exc)
                return []

    def load_directory(self, path: str) -> List[Document]:
        """Auto-detect file type and load all supported files from a directory."""
        directory = Path(path)
        if not directory.is_dir():
            logger.error("Path is not a directory: %s", path)
            return []

        all_docs: List[Document] = []
        for file_path in sorted(directory.rglob("*")):
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                logger.debug("Skipping unsupported file: %s", file_path)
                continue

            logger.info("Loading %s (%s)", file_path.name, ext)
            docs = self._load_by_extension(str(file_path), ext)
            all_docs.extend(docs)

        logger.info("Loaded %d document(s) from %s", len(all_docs), path)
        return all_docs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def load_file(self, path: str) -> List[Document]:
        """Load a single file, auto-detecting its type by extension.

        Args:
            path: Absolute or relative path to the file.

        Returns:
            List of :class:`Document` objects, or an empty list on failure.
        """
        ext = Path(path).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning("Unsupported file extension '%s' for %s", ext, path)
            return []
        return self._load_by_extension(path, ext)

    def _load_by_extension(self, path: str, ext: str) -> List[Document]:
        """Dispatch to the correct loader based on file extension."""
        if ext == ".pdf":
            return self.load_pdf(path)
        if ext in {".md", ".markdown"}:
            return self.load_markdown(path)
        return self.load_text(path)

    def _add_metadata(self, docs: List[Document], path: str, file_type: str) -> List[Document]:
        """Add standard metadata fields to each document."""
        filename = Path(path).name
        for idx, doc in enumerate(docs):
            doc.metadata.setdefault("source", path)
            doc.metadata["filename"] = filename
            doc.metadata["file_type"] = file_type
            # PDFs set page via PyPDFLoader; default to 1 for single-page loaders
            doc.metadata.setdefault("page_number", doc.metadata.pop("page", idx) + 1 if file_type == "pdf" else 1)
        return docs
