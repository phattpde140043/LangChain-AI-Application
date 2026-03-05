#!/usr/bin/env python3
"""Benchmark script for measuring RAG pipeline performance."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import get_settings
from src.monitoring.logger import setup_logging, get_logger

BENCHMARK_QUERIES = [
    "What are the key components of microservices architecture?",
    "How does an API Gateway work in a distributed system?",
    "What is the difference between REST and gRPC?",
    "Explain the RAG pipeline and its components.",
    "What are the best practices for containerization with Docker?",
]

def main():
    settings = get_settings()
    setup_logging("WARNING")
    logger = get_logger(__name__)

    print(f"\n{'='*60}")
    print(f"LangChain AI Research Assistant - Benchmark")
    print(f"{'='*60}\n")

    eval_path = Path(settings.EVAL_DATASET_PATH)
    queries = BENCHMARK_QUERIES

    if eval_path.exists():
        try:
            with open(eval_path) as f:
                dataset = json.load(f)
            queries = [item["question"] for item in dataset[:10] if "question" in item]
            print(f"Loaded {len(queries)} queries from eval dataset")
        except Exception:
            print(f"Using default benchmark queries")

    try:
        from src.rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        return

    results = []
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Running: {query[:60]}...")
        start = time.time()
        try:
            response = pipeline.run(query)
            elapsed_ms = (time.time() - start) * 1000
            results.append({
                "query": query,
                "latency_ms": round(elapsed_ms, 2),
                "sources": len(response.sources),
                "confidence": response.confidence,
                "success": True,
            })
            print(f"  ✓ {elapsed_ms:.0f}ms | confidence={response.confidence} | sources={len(response.sources)}")
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            results.append({"query": query, "latency_ms": round(elapsed_ms, 2), "success": False, "error": str(e)})
            print(f"  ✗ Failed: {e}")

    if results:
        successful = [r for r in results if r["success"]]
        avg_latency = sum(r["latency_ms"] for r in successful) / len(successful) if successful else 0
        print(f"\n{'='*60}")
        print(f"Benchmark Results:")
        print(f"  Total queries: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Avg latency: {avg_latency:.0f}ms")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
