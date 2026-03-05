from __future__ import annotations
"""Query expansion: generate alternative phrasings to improve retrieval recall."""

from typing import List, Optional

from ..monitoring.logger import get_logger

logger = get_logger(__name__)

EXPANSION_PROMPT = """Generate 2-3 alternative phrasings for the following search query to improve document retrieval.
Return only the alternative queries, one per line, without numbering or explanation.

Original query: {query}

Alternative phrasings:"""


class QueryExpander:
    """Uses an LLM to generate semantically equivalent query variants.

    Falls back gracefully to just the original query on any error so the
    rest of the pipeline is never blocked.
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm

    def expand_query(self, query: str, llm=None) -> List[str]:
        """Return the original query prepended to 2-3 alternative phrasings.

        Args:
            query: The original user query.
            llm:   Optional override LLM; falls back to ``self._llm``.

        Returns:
            A list starting with *query* followed by generated alternatives.
            On any failure the list contains only *query*.
        """
        active_llm = llm or self._llm
        if active_llm is None:
            return [query]

        try:
            prompt_text = EXPANSION_PROMPT.format(query=query)
            result = active_llm.invoke(prompt_text)
            raw = result.content if hasattr(result, "content") else str(result)

            alternatives: List[str] = []
            for line in raw.strip().splitlines():
                line = line.strip()
                # Skip empty lines or lines that just echo the original.
                if line and line.lower() != query.lower():
                    alternatives.append(line)

            # Keep at most 3 alternatives to avoid excessive API calls downstream.
            alternatives = alternatives[:3]
            return [query] + alternatives
        except Exception as e:
            logger.warning(f"Query expansion failed, using original query only: {e}")
            return [query]
