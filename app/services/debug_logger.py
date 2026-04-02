"""
Structured debug logging helpers for retrieval and memory pipeline tracing.
"""
from typing import Any


def log_step(step: str, data: Any) -> None:
    """Print a structured debug step block."""
    print(f"\n===== {step} =====")
    print(data)
