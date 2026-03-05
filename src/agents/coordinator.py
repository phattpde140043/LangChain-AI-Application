from __future__ import annotations
"""Coordinator agent that routes queries to specialized agents."""

from typing import Dict, List
from pydantic import BaseModel
from .base_agent import BaseAgent

INTENT_KEYWORDS = {
    "summarize": ["summarize", "summary", "overview", "brief", "tldr", "describe"],
    "compare": ["compare", "difference", "similarities", "vs", "versus", "contrast"],
    "analyze": ["analyze", "concepts", "insights", "extract", "key ideas", "understand"],
    "research": ["what", "how", "why", "explain", "tell me", "find", "search"],
}


class AgentResponse(BaseModel):
    answer: str
    agent_used: str
    sources: List[str] = []
    confidence: str = "medium"


class CoordinatorAgent(BaseAgent):
    name = "coordinator"
    description = "Routes queries to the appropriate specialized agent"

    def __init__(self):
        super().__init__()
        self._agents: Dict[str, BaseAgent] = {}

    def _get_agent(self, intent: str) -> BaseAgent:
        if intent not in self._agents:
            if intent == "summarize":
                from .summarization_agent import SummarizationAgent
                self._agents[intent] = SummarizationAgent()
            elif intent == "compare":
                from .comparison_agent import ComparisonAgent
                self._agents[intent] = ComparisonAgent()
            elif intent == "analyze":
                from .analysis_agent import AnalysisAgent
                self._agents[intent] = AnalysisAgent()
            else:
                from .research_agent import ResearchAgent
                self._agents[intent] = ResearchAgent()
        return self._agents[intent]

    def classify_intent(self, query: str) -> str:
        """Classify the query intent."""
        query_lower = query.lower()
        for intent, keywords in INTENT_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return intent
        return "research"

    def route(self, query: str, session_id: str = "default") -> AgentResponse:
        """Route query to appropriate agent and return structured response."""
        intent = self.classify_intent(query)
        agent = self._get_agent(intent)

        context = {"session_id": session_id}

        try:
            answer = agent.run(query, context=context)
            return AgentResponse(
                answer=answer,
                agent_used=agent.name,
                confidence="medium",
            )
        except Exception as e:
            from ..monitoring.logger import get_logger
            get_logger(__name__).error(f"Agent routing failed: {e}")
            return AgentResponse(
                answer=f"An error occurred: {str(e)}",
                agent_used=agent.name,
                confidence="low",
            )

    def run(self, query: str, context: Dict = None) -> str:
        session_id = (context or {}).get("session_id", "default")
        response = self.route(query, session_id=session_id)
        return response.answer
