from __future__ import annotations
"""Text chunking strategies for document processing."""

import logging
from typing import List

from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # langchain <0.2 bundled it in langchain.text_splitter
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class TextChunker:
    """Chunks documents using RecursiveCharacterTextSplitter."""

    def chunk_documents(
        self,
        docs: List[Document],
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> List[Document]:
        """Split documents into chunks, preserving metadata and adding chunk_index.

        Args:
            docs: Source documents to chunk.
            chunk_size: Maximum character length of each chunk.
            chunk_overlap: Number of overlapping characters between adjacent chunks.

        Returns:
            List of chunked Documents with copied metadata and ``chunk_index``.
        """
        if not docs:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        chunked: List[Document] = []
        for doc in docs:
            splits = splitter.split_documents([doc])
            for idx, chunk in enumerate(splits):
                # Ensure all source metadata is carried forward
                chunk.metadata = {**doc.metadata, **chunk.metadata, "chunk_index": idx}
                chunked.append(chunk)

        logger.debug("Chunked %d document(s) into %d chunks", len(docs), len(chunked))
        return chunked


class HierarchicalChunker:
    """Two-level chunking: large sections first, then smaller pieces."""

    def chunk_documents(
        self,
        docs: List[Document],
        section_size: int = 2000,
        section_overlap: int = 200,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> List[Document]:
        """Chunk documents into sections, then into smaller pieces.

        Args:
            docs: Source documents to chunk.
            section_size: Character length of first-pass (section) chunks.
            section_overlap: Overlap between sections.
            chunk_size: Character length of second-pass (final) chunks.
            chunk_overlap: Overlap between final chunks.

        Returns:
            Fully chunked Documents with metadata preserved through both passes.
        """
        if not docs:
            return []

        # First pass: split into sections
        section_splitter = RecursiveCharacterTextSplitter(
            chunk_size=section_size,
            chunk_overlap=section_overlap,
            length_function=len,
        )
        sections: List[Document] = []
        for doc in docs:
            splits = section_splitter.split_documents([doc])
            for s_idx, section in enumerate(splits):
                section.metadata = {**doc.metadata, **section.metadata, "section_index": s_idx}
                sections.append(section)

        # Second pass: split sections into final chunks
        chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks: List[Document] = []
        for section in sections:
            splits = chunk_splitter.split_documents([section])
            for c_idx, chunk in enumerate(splits):
                chunk.metadata = {**section.metadata, **chunk.metadata, "chunk_index": c_idx}
                chunks.append(chunk)

        logger.debug(
            "Hierarchically chunked %d doc(s) -> %d section(s) -> %d chunk(s)",
            len(docs),
            len(sections),
            len(chunks),
        )
        return chunks
