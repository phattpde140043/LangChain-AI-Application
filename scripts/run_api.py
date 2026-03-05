#!/usr/bin/env python3
"""Script to start the FastAPI server."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from src.config.settings import get_settings

def main():
    settings = get_settings()
    print(f"\nStarting LangChain AI Research Assistant API")
    print(f"Profile: {settings.ACTIVE_PROFILE}")
    print(f"Model: {settings.MODEL_NAME}")
    print(f"Docs: http://localhost:8000/docs\n")

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=(settings.ACTIVE_PROFILE == "development"),
        log_level=settings.LOG_LEVEL.lower(),
    )

if __name__ == "__main__":
    main()
