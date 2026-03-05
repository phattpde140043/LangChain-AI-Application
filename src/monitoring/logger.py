from __future__ import annotations
"""Structured logging setup with request tracing support."""

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Optional

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


class LogContext:
    """Context manager that attaches a trace ID to all log records within its scope."""

    def __init__(self, trace_id: Optional[str] = None) -> None:
        self.trace_id = trace_id or str(uuid.uuid4())[:8]
        self._token = None

    def __enter__(self) -> "LogContext":
        self._token = _trace_id_var.set(self.trace_id)
        return self

    def __exit__(self, *args: object) -> None:
        _trace_id_var.reset(self._token)


class TraceFormatter(logging.Formatter):
    """Logging formatter that injects the current trace ID into every record."""

    def format(self, record: logging.LogRecord) -> str:
        record.trace_id = _trace_id_var.get() or "-"
        return super().format(record)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure the root logger with a structured format.

    Idempotent: subsequent calls return the already-configured root logger.
    """
    root = logging.getLogger()
    if root.handlers:
        return root

    handler = logging.StreamHandler(sys.stdout)
    formatter = TraceFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | trace=%(trace_id)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Falls back gracefully when the root logger has not been explicitly
    configured via :func:`setup_logging` — a basic StreamHandler is added so
    messages are never silently dropped.
    """
    logger = logging.getLogger(name)
    # Ensure at least one handler exists on the root so records surface.
    if not logging.getLogger().handlers:
        setup_logging()
    return logger
