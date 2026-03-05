from __future__ import annotations
"""Rate limiting middleware using slowapi."""

from slowapi import Limiter
from slowapi.util import get_remote_address

def create_limiter(rate_limit: str = "60/minute") -> Limiter:
    """Create a rate limiter with the given rate."""
    return Limiter(key_func=get_remote_address, default_limits=[rate_limit])

# Module-level limiter instance (used as a decorator factory)
limiter = create_limiter()
