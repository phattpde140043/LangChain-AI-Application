#!/usr/bin/env python3
"""CLI script to ingest documents into the knowledge base."""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.monitoring.logger import setup_logging, get_logger

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the knowledge base")
    parser.add_argument("--dir", type=str, help="Directory containing documents to ingest")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = get_logger(__name__)

    documents_dir = args.dir or settings.DOCUMENTS_DIR

    print(f"\n{'='*60}")
    print(f"LangChain AI Research Assistant - Document Ingestion")
    print(f"{'='*60}")
    print(f"Source directory: {documents_dir}")
    print(f"Vector DB: {settings.VECTOR_DB_TYPE} at {settings.VECTOR_DB_PATH}")
    print(f"Chunk size: {settings.CHUNK_SIZE} | Overlap: {settings.CHUNK_OVERLAP}")
    print(f"{'='*60}\n")

    start_time = time.time()

    pipeline = IngestionPipeline()
    result = pipeline.ingest(documents_path=documents_dir)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Ingestion Complete!")
    print(f"{'='*60}")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Documents processed: {result.get('documents_processed', 0)}")
    print(f"Chunks created: {result.get('chunks_created', 0)}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
