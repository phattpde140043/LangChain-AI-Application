from __future__ import annotations
"""Prompt injection detection using pattern matching."""

from typing import List

from pydantic import BaseModel


class InjectionResult(BaseModel):
    """Result of an injection scan."""

    is_injection: bool
    risk_score: float  # 0.0 – 1.0
    patterns_found: List[str]


class PromptInjectionDetector:
    """Detects common prompt-injection attempts via substring matching.

    The ``risk_score`` is computed as ``min(1.0, matched_patterns * 0.3)``,
    giving:
    * 0.3  → 1 pattern  (warn)
    * 0.6  → 2 patterns (block)
    * 1.0  → 3+ patterns (block)
    """

    INJECTION_PATTERNS: List[str] = [
        "ignore previous",
        "ignore all",
        "reveal system prompt",
        "act as",
        "jailbreak",
        "forget your instructions",
        "pretend you are",
        "disregard",
        "override instructions",
        "ignore instructions",
        "system prompt",
        "developer mode",
    ]

    def detect(self, query: str) -> InjectionResult:
        """Scan *query* for known injection patterns.

        Args:
            query: Raw user input string.

        Returns:
            An :class:`InjectionResult` with ``is_injection``, ``risk_score``,
            and the list of matched pattern strings.
        """
        query_lower = query.lower()
        patterns_found = [p for p in self.INJECTION_PATTERNS if p in query_lower]
        risk_score = min(1.0, len(patterns_found) * 0.3)
        return InjectionResult(
            is_injection=len(patterns_found) > 0,
            risk_score=round(risk_score, 2),
            patterns_found=patterns_found,
        )
