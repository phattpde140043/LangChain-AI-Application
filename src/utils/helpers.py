from __future__ import annotations
"""Utility helper functions."""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict


def slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text to max_chars with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def load_json_file(path: str) -> Any:
    """Load and parse a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Any, path: str, indent: int = 2) -> None:
    """Save data to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def append_jsonl(record: Dict, path: str) -> None:
    """Append a dict as a JSON line to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def hash_text(text: str) -> str:
    """Compute SHA-256 hash of text."""
    return hashlib.sha256(text.encode()).hexdigest()


def extract_filename(path: str) -> str:
    """Extract filename without extension from a path."""
    return Path(path).stem


def timer(func):
    """Decorator to time function execution."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000
        return result, elapsed_ms

    return wrapper
