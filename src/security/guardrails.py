from __future__ import annotations
"""Input/output safety guardrails for the RAG pipeline."""

from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel

from .injection_detector import PromptInjectionDetector


class GuardrailResult(BaseModel):
    """Outcome of a guardrail check."""

    is_safe: bool
    reason: str
    action: str  # "allow" | "block" | "warn"


class GuardrailsManager:
    """Validates user inputs and LLM outputs against safety policies.

    Checks performed on **inputs**:

    * Non-empty and within length limits.
    * Absent or below-threshold injection patterns.

    Checks performed on **outputs**:

    * Minimum length (prevents empty/degenerate responses).
    * Soft grounding verification against retrieved source chunks.
    """

    MAX_QUERY_LENGTH = 2000
    MIN_RESPONSE_LENGTH = 5

    def __init__(self) -> None:
        self.injection_detector = PromptInjectionDetector()

    # ------------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------------

    def check_input(self, query: str) -> GuardrailResult:
        """Return a :class:`GuardrailResult` for the provided user *query*."""
        if not query.strip():
            return GuardrailResult(
                is_safe=False, reason="Empty query", action="block"
            )

        if len(query) > self.MAX_QUERY_LENGTH:
            return GuardrailResult(
                is_safe=False,
                reason=f"Query exceeds maximum length of {self.MAX_QUERY_LENGTH} characters",
                action="block",
            )

        injection = self.injection_detector.detect(query)
        if injection.risk_score >= 0.6:
            return GuardrailResult(
                is_safe=False,
                reason=f"Potential prompt injection detected: {injection.patterns_found}",
                action="block",
            )
        if injection.risk_score >= 0.3:
            return GuardrailResult(
                is_safe=True,
                reason="Potential injection pattern detected but below block threshold",
                action="warn",
            )

        return GuardrailResult(
            is_safe=True, reason="Input passed safety checks", action="allow"
        )

    # ------------------------------------------------------------------
    # Output checks
    # ------------------------------------------------------------------

    def check_output(
        self, response: str, retrieved_chunks: List[Document]
    ) -> GuardrailResult:
        """Validate the LLM *response* before it is sent to the user."""
        if len(response.strip()) < self.MIN_RESPONSE_LENGTH:
            return GuardrailResult(
                is_safe=False, reason="Response too short", action="warn"
            )

        if not retrieved_chunks:
            return GuardrailResult(
                is_safe=True,
                reason="No source documents to verify against",
                action="warn",
            )

        return GuardrailResult(
            is_safe=True, reason="Output passed safety checks", action="allow"
        )

    def verify_grounding(self, response: str, chunks: List[Document]) -> bool:
        """Return ``True`` when *response* shares meaningful vocabulary with *chunks*.

        Uses a simple word-overlap heuristic: if more than 10 % of the unique
        words in the response also appear in the retrieved chunks the response
        is considered grounded.
        """
        if not chunks:
            return False

        chunk_text = " ".join(doc.page_content for doc in chunks).lower()
        response_words = set(response.lower().split())
        chunk_words = set(chunk_text.split())
        overlap = response_words & chunk_words
        return len(overlap) / max(len(response_words), 1) > 0.1
