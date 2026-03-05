from __future__ import annotations
"""Tests for settings configuration."""

import os
import pytest
from src.config.settings import Settings, get_profile_settings


def test_default_settings(monkeypatch):
    """Test that default settings are correct."""
    # Clear vars set by conftest to verify true class-level defaults
    monkeypatch.delenv("VECTOR_DB_TYPE", raising=False)
    monkeypatch.delenv("ACTIVE_PROFILE", raising=False)
    settings = Settings()
    assert settings.MODEL_NAME == "gpt-4o-mini"
    assert settings.EMBEDDING_MODEL == "text-embedding-3-small"
    assert settings.VECTOR_DB_TYPE == "chroma"
    assert settings.RETRIEVAL_TOP_K == 5
    assert settings.CHUNK_SIZE == 800
    assert settings.CHUNK_OVERLAP == 100


def test_feature_flags_default(monkeypatch):
    """Test that feature flags have correct defaults."""
    # Clear vars set by conftest to verify true class-level defaults
    monkeypatch.delenv("ENABLE_SAFETY_CHECKS", raising=False)
    monkeypatch.delenv("ENABLE_COST_TRACKING", raising=False)
    settings = Settings()
    assert settings.ENABLE_HYBRID_RETRIEVAL is True
    assert settings.ENABLE_QUERY_EXPANSION is True
    assert settings.ENABLE_CONTEXT_COMPRESSION is False
    assert settings.ENABLE_SAFETY_CHECKS is True
    assert settings.ENABLE_COST_TRACKING is True
    assert settings.ENABLE_STREAMING is True


def test_settings_override_from_env(monkeypatch):
    """Test that settings can be overridden via environment variables."""
    monkeypatch.setenv("MODEL_NAME", "gpt-4o")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
    monkeypatch.setenv("ENABLE_HYBRID_RETRIEVAL", "false")

    settings = Settings()
    assert settings.MODEL_NAME == "gpt-4o"
    assert settings.RETRIEVAL_TOP_K == 10
    assert settings.ENABLE_HYBRID_RETRIEVAL is False


def test_profile_development():
    """Test development profile settings."""
    settings = get_profile_settings("development")
    assert settings.LOG_LEVEL == "DEBUG"


def test_profile_testing():
    """Test testing profile settings."""
    settings = get_profile_settings("testing")
    assert settings.ENABLE_COST_TRACKING is False


def test_profile_production():
    """Test production profile settings."""
    settings = get_profile_settings("production")
    assert settings.ENABLE_SAFETY_CHECKS is True
    assert settings.ENABLE_COST_TRACKING is True
