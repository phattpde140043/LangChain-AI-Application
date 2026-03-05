from __future__ import annotations
"""API Key authentication middleware."""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validates X-API-Key header if API_KEY is configured."""
    
    SKIP_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}
    
    def __init__(self, app, api_key: str = ""):
        super().__init__(app)
        self.api_key = api_key
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth if no API key configured
        if not self.api_key:
            return await call_next(request)
        
        # Skip auth for health and docs endpoints
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)
        
        # Validate API key
        provided_key = request.headers.get("X-API-Key", "")
        if provided_key != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Provide X-API-Key header."},
            )
        
        return await call_next(request)
