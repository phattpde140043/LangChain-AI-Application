from __future__ import annotations
"""Plugin interface for tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from pydantic import BaseModel

TOOL_REGISTRY: Dict[str, Type["BaseTool"]] = {}


def register_tool(tool_class: Type["BaseTool"]) -> Type["BaseTool"]:
    """Decorator to register a tool in the registry."""
    TOOL_REGISTRY[tool_class.name] = tool_class
    return tool_class


class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str = ""
    description: str = ""
    input_schema: Type[BaseModel] = None

    @abstractmethod
    def _run(self, input_data: BaseModel) -> str:
        """Execute the tool with the given input."""
        ...

    def run(self, **kwargs) -> str:
        """Validate input and run the tool."""
        if self.input_schema:
            validated = self.input_schema(**kwargs)
            return self._run(validated)
        return self._run(kwargs)

    def to_langchain_tool(self):
        """Convert to a LangChain Tool object."""
        from langchain.tools import Tool
        return Tool(
            name=self.name,
            description=self.description,
            func=lambda x: self.run(query=x) if isinstance(x, str) else self.run(**x),
        )
