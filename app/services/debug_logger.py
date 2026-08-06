"""
Structured debug logging helpers for retrieval and memory pipeline tracing.

Step payloads routinely contain raw user text and retrieved memory, so they are
emitted only when step logging is explicitly enabled (development by default).
In production this collapses to a single DEBUG-level line with no payload, so
personal data never reaches the log stream and the output can be filtered by
level like any other logger.
"""
import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


def log_step(step: str, data: Any) -> None:
    """Emit a structured debug step. No-op unless step logging is enabled."""
    if not settings.step_logging_enabled:
        # Keep a payload-free breadcrumb so step ordering is still traceable.
        logger.debug("step=%s", step)
        return

    logger.debug("===== %s =====\n%s", step, data)
