from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI / LLM
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    MODEL_PROVIDER: str = Field(default="openai", description="LLM provider: openai or local")
    MODEL_NAME: str = Field(default="gpt-4o-mini", description="Default LLM model name")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="Embedding model name")

    # Vector Store
    VECTOR_DB_PATH: str = Field(default="./data/vector_store", description="Path to vector store")
    VECTOR_DB_TYPE: str = Field(default="chroma", description="Vector DB type: chroma or faiss")

    # RAG
    MAX_CONTEXT_TOKENS: int = Field(default=4000)
    RETRIEVAL_TOP_K: int = Field(default=5)
    CHUNK_SIZE: int = Field(default=800)
    CHUNK_OVERLAP: int = Field(default=100)

    # API Security
    API_KEY: str = Field(default="", description="API key for authentication (empty = disabled)")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)

    # Logging
    LOG_LEVEL: str = Field(default="INFO")

    # Redis
    REDIS_URL: str = Field(default="", description="Redis URL for caching")

    # Prompts & Profiles
    ACTIVE_RAG_PROMPT: str = Field(default="rag_prompt_v1")
    ACTIVE_PROFILE: str = Field(default="development")

    # Feature Flags
    ENABLE_HYBRID_RETRIEVAL: bool = Field(default=True)
    ENABLE_QUERY_EXPANSION: bool = Field(default=True)
    ENABLE_CONTEXT_COMPRESSION: bool = Field(default=False)
    ENABLE_SAFETY_CHECKS: bool = Field(default=True)
    ENABLE_COST_TRACKING: bool = Field(default=True)
    ENABLE_STREAMING: bool = Field(default=True)

    # Paths
    DATA_DIR: str = Field(default="./data")
    DOCUMENTS_DIR: str = Field(default="./documents")
    PROMPTS_DIR: str = Field(default="./prompts")
    EVAL_DATASET_PATH: str = Field(default="./data/eval_dataset.json")
    FEEDBACK_LOG_PATH: str = Field(default="./data/feedback.jsonl")
    METRICS_LOG_PATH: str = Field(default="./data/metrics.jsonl")
    COST_LOG_PATH: str = Field(default="./data/costs.jsonl")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def get_profile_settings(profile: str) -> Settings:
    """Return a validated Settings copy with profile-specific overrides applied."""
    base = get_settings()
    profile_overrides: dict[str, dict] = {
        "development": {"LOG_LEVEL": "DEBUG", "ENABLE_COST_TRACKING": False},
        "testing": {
            "ENABLE_SAFETY_CHECKS": False,
            "ENABLE_COST_TRACKING": False,
            "VECTOR_DB_TYPE": "faiss",
        },
        "production": {
            "LOG_LEVEL": "INFO",
            "ENABLE_SAFETY_CHECKS": True,
            "ENABLE_COST_TRACKING": True,
        },
    }
    overrides = profile_overrides.get(profile, {})
    return base.model_copy(update=overrides)
