from __future__ import annotations
"""FastAPI application entry point."""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config.settings import get_settings
from ..monitoring.logger import setup_logging, get_logger
from .schemas import HealthResponse
from .routes import ask, summarize, compare, ingest, feedback, knowledge_base
from .middleware.auth import APIKeyMiddleware
from .middleware.rate_limit import limiter

settings = get_settings()
setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info(f"Starting LangChain AI Research Assistant (profile: {settings.ACTIVE_PROFILE})")
    logger.info(f"Vector DB: {settings.VECTOR_DB_TYPE} at {settings.VECTOR_DB_PATH}")
    logger.info(f"Model: {settings.MODEL_NAME} | Embeddings: {settings.EMBEDDING_MODEL}")
    yield
    logger.info("Shutting down LangChain AI Research Assistant")


app = FastAPI(
    title="LangChain AI Research Assistant",
    description="Production-grade AI research assistant powered by LangChain and RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Rate limiter state
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth middleware
app.add_middleware(APIKeyMiddleware, api_key=settings.API_KEY)

# Routers
app.include_router(ask.router, tags=["Research"])
app.include_router(summarize.router, tags=["Summarization"])
app.include_router(compare.router, tags=["Comparison"])
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(feedback.router, tags=["Feedback"])
app.include_router(knowledge_base.router, tags=["Knowledge Base"])


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        profile=settings.ACTIVE_PROFILE,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )
