"""
Minimal LangSmith integration helpers.
Provides optional tracing without changing runtime behavior when disabled.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from app.config import settings


def _noop_traceable(*args, **kwargs):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator


try:
    from langsmith import traceable as _langsmith_traceable
except Exception:
    _langsmith_traceable = _noop_traceable


def traceable(name: str, run_type: str = "chain", tags: list[str] | None = None):
    """Thin wrapper around LangSmith traceable with graceful fallback."""
    return _langsmith_traceable(name=name, run_type=run_type, tags=tags or [])


def configure_langsmith() -> bool:
    """Enable LangSmith tracing via environment when configured."""
    enabled = bool(settings.langsmith_tracing_enabled and settings.langsmith_api_key)
    if not enabled:
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key or ""
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    return True
