#!/usr/bin/env python3
"""Generate an evaluation dataset from indexed documents."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SAMPLE_QA_PAIRS = [
    {"question": "What is microservices architecture?", "expected_answer": "Microservices architecture decomposes applications into small, independent services that can be deployed and scaled independently.", "category": "architecture"},
    {"question": "What is an API Gateway?", "expected_answer": "An API Gateway serves as the single entry point for all client requests, handling routing, authentication, and load balancing.", "category": "infrastructure"},
    {"question": "What is the database-per-service pattern?", "expected_answer": "Each microservice owns its data and has its own database, ensuring loose coupling and independent scalability.", "category": "data"},
    {"question": "What is a vector database?", "expected_answer": "A vector database stores high-dimensional embeddings for efficient semantic search and retrieval operations.", "category": "ai"},
    {"question": "What is RAG?", "expected_answer": "RAG (Retrieval-Augmented Generation) combines document retrieval with language model generation to ground responses in factual documents.", "category": "ai"},
    {"question": "What is Docker?", "expected_answer": "Docker is a containerization platform that packages applications with their dependencies for consistent deployment across environments.", "category": "infrastructure"},
    {"question": "What is Kubernetes?", "expected_answer": "Kubernetes is an open-source container orchestration system for automating deployment, scaling, and management of containerized applications.", "category": "infrastructure"},
    {"question": "What are the advantages of microservices?", "expected_answer": "Microservices offer independent deployability, technology flexibility, fault isolation, and easier scaling of individual components.", "category": "architecture"},
]

def main():
    from src.config.settings import get_settings
    settings = get_settings()

    output_path = Path(settings.EVAL_DATASET_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating evaluation dataset...")
    print(f"Output: {output_path}")

    with open(output_path, "w") as f:
        json.dump(SAMPLE_QA_PAIRS, f, indent=2)

    print(f"Generated {len(SAMPLE_QA_PAIRS)} Q&A pairs")
    print(f"Dataset saved to: {output_path}\n")

if __name__ == "__main__":
    main()
