from __future__ import annotations
"""Evaluation framework for RAG responses."""

import json
from datetime import datetime
from pathlib import Path
from typing import List
from pydantic import BaseModel


class EvalResult(BaseModel):
    question: str
    answer: str
    expected: str
    correctness: float  # 0-1
    relevance: float  # 0-1
    hallucination_rate: float  # 0-1


class EvalReport(BaseModel):
    total: int
    correct: int
    avg_relevance: float
    avg_correctness: float
    timestamp: str


class Evaluator:
    """Evaluates RAG answers using string overlap metrics."""

    def evaluate_answer(self, question: str, answer: str, expected: str) -> EvalResult:
        """Evaluate a single answer against expected output."""
        answer_words = set(answer.lower().split())
        expected_words = set(expected.lower().split())

        # Correctness: overlap between answer and expected
        if expected_words:
            overlap = answer_words & expected_words
            correctness = len(overlap) / len(expected_words)
        else:
            correctness = 0.0

        # Relevance: how much of the answer relates to the question
        question_words = set(question.lower().split()) - {"what", "how", "why", "is", "the", "a", "an"}
        if question_words:
            q_overlap = answer_words & question_words
            relevance = min(1.0, len(q_overlap) / len(question_words))
        else:
            relevance = 0.5

        # Hallucination rate: words in answer not in expected or question
        all_source_words = expected_words | question_words
        if answer_words:
            unsupported = answer_words - all_source_words - {
                "the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "and", "or"
            }
            hallucination_rate = len(unsupported) / len(answer_words)
        else:
            hallucination_rate = 0.0

        return EvalResult(
            question=question,
            answer=answer,
            expected=expected,
            correctness=round(min(1.0, correctness), 3),
            relevance=round(min(1.0, relevance), 3),
            hallucination_rate=round(min(1.0, hallucination_rate), 3),
        )

    def run_evaluation_suite(self, dataset_path: str) -> EvalReport:
        """Run evaluation on a dataset file."""
        from ..rag.pipeline import RAGPipeline

        path = Path(dataset_path)
        if not path.exists():
            return EvalReport(
                total=0,
                correct=0,
                avg_relevance=0.0,
                avg_correctness=0.0,
                timestamp=datetime.utcnow().isoformat(),
            )

        with open(path) as f:
            dataset = json.load(f)

        pipeline = RAGPipeline()
        results: List[EvalResult] = []

        for item in dataset:
            question = item.get("question", "")
            expected = item.get("expected_answer", item.get("answer", ""))
            if not question:
                continue

            try:
                rag_response = pipeline.run(question)
                result = self.evaluate_answer(question, rag_response.answer, expected)
                results.append(result)
            except Exception:
                pass

        if not results:
            return EvalReport(
                total=0,
                correct=0,
                avg_relevance=0.0,
                avg_correctness=0.0,
                timestamp=datetime.utcnow().isoformat(),
            )

        correct = sum(1 for r in results if r.correctness > 0.5)
        avg_relevance = sum(r.relevance for r in results) / len(results)
        avg_correctness = sum(r.correctness for r in results) / len(results)

        return EvalReport(
            total=len(results),
            correct=correct,
            avg_relevance=round(avg_relevance, 3),
            avg_correctness=round(avg_correctness, 3),
            timestamp=datetime.utcnow().isoformat(),
        )
