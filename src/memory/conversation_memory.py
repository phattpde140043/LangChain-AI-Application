from __future__ import annotations
"""Multi-user conversation memory management using LangChain core primitives."""

from typing import Dict, List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..monitoring.logger import get_logger

logger = get_logger(__name__)

MAX_SESSIONS = 100
MAX_MESSAGES_PER_SESSION = 50


class _InMemoryChatHistory(BaseChatMessageHistory):
    """Simple in-process list-backed chat history."""

    def __init__(self) -> None:
        self.messages: List[BaseMessage] = []

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


class MultiUserMemoryManager:
    """Per-session conversation memory with automatic LRU eviction.

    Sessions are stored in a plain dict.  When the number of live sessions
    reaches *MAX_SESSIONS* the oldest session is evicted before a new one is
    created.  Each session's history is also capped at *MAX_MESSAGES_PER_SESSION*
    messages to prevent unbounded memory growth.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, _InMemoryChatHistory] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_memory(self, session_id: str) -> _InMemoryChatHistory:
        """Return the chat history store for *session_id*.

        Creates a new store when the session does not exist yet, evicting the
        oldest session if the capacity limit has been reached.
        """
        if session_id not in self._sessions:
            if len(self._sessions) >= MAX_SESSIONS:
                oldest = next(iter(self._sessions))
                del self._sessions[oldest]
                logger.debug(f"Evicted oldest session '{oldest}' to free capacity.")
            self._sessions[session_id] = _InMemoryChatHistory()
            logger.debug(f"Created new memory session '{session_id}'.")
        return self._sessions[session_id]

    def add_user_message(self, session_id: str, content: str) -> None:
        """Append a human message to *session_id* and trim if necessary."""
        history = self.get_memory(session_id)
        history.add_messages([HumanMessage(content=content)])
        self.trim_session(session_id)

    def add_ai_message(self, session_id: str, content: str) -> None:
        """Append an AI message to *session_id* and trim if necessary."""
        history = self.get_memory(session_id)
        history.add_messages([AIMessage(content=content)])
        self.trim_session(session_id)

    def clear_memory(self, session_id: str) -> None:
        """Delete the memory for *session_id* (no-op if it does not exist)."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Cleared memory for session '{session_id}'.")

    def list_sessions(self) -> List[str]:
        """Return the IDs of all active sessions."""
        return list(self._sessions.keys())

    def get_history(self, session_id: str) -> List[Dict]:
        """Return the conversation history for *session_id* as a list of dicts.

        Each dict has ``role`` (``"human"`` or ``"assistant"``) and ``content``.
        Returns an empty list when the session does not exist.
        """
        if session_id not in self._sessions:
            return []
        history: List[Dict] = []
        for msg in self._sessions[session_id].messages:
            history.append(
                {
                    "role": "human" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content,
                }
            )
        return history

    def trim_session(self, session_id: str) -> None:
        """Trim the session history to the most recent *MAX_MESSAGES_PER_SESSION* entries."""
        if session_id not in self._sessions:
            return
        msgs = self._sessions[session_id].messages
        if len(msgs) > MAX_MESSAGES_PER_SESSION:
            self._sessions[session_id].messages = msgs[-MAX_MESSAGES_PER_SESSION:]
