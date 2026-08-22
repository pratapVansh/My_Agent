"""
In-process, per-client rate limiting.

Every LLM call, embedding call, and headless-browser run on this service costs
real money and real CPU. Without a limiter a single client can drive unbounded
spend against Groq/Cohere/Tavily or exhaust the worker pool.

Design notes
------------
* Sliding-window counters keyed by (client IP, bucket). Buckets are matched by
  path prefix so expensive endpoints get a tighter limit than ordinary reads.
* Deliberately dependency-free: no Redis, no new packages. State is per process,
  which is correct for the current single-worker deployment. Running multiple
  workers or instances requires a shared store — see docs/AUDIT_REPORT.md (M15).
* Limits are generous enough that normal interactive use is never throttled.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Iterable, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60.0

# Longest prefix wins, so ordering here is not significant.
_BUCKET_RULES: Tuple[Tuple[str, str], ...] = (
    # Login is the tightest bucket: it is the one endpoint where an attacker
    # gains something by trying repeatedly.
    ("/api/v1/auth/login", "auth"),
    ("/api/v1/auth/guest", "auth"),
    ("/api/v1/agents/query", "llm"),
    ("/api/v1/agents/memory/upload-pdf", "upload"),
    ("/api/v1/agents/memory/upload-text", "upload"),
    ("/api/v1/agents/tools/timetable/upload-pdf", "upload"),
    ("/api/v1/agents/tools/attendance/scrape", "expensive"),
    ("/api/v1/agents/tools/job-search", "llm"),
    ("/api/v1/agents/tools/email-draft", "llm"),
    ("/api/v1/voice/token", "expensive"),
)

_EXEMPT_PATHS = frozenset({"/", "/health", "/docs", "/openapi.json", "/redoc"})


def _limit_for(bucket: str) -> int:
    if bucket == "auth":
        return settings.rate_limit_auth_per_minute
    if bucket == "llm":
        return settings.rate_limit_llm_per_minute
    if bucket == "upload":
        return settings.rate_limit_upload_per_minute
    if bucket == "expensive":
        return settings.rate_limit_expensive_per_minute
    return settings.rate_limit_default_per_minute


def _classify(path: str) -> str:
    """Map a request path to its rate-limit bucket (longest prefix wins)."""
    best_bucket = "default"
    best_len = -1
    for prefix, bucket in _BUCKET_RULES:
        if path.startswith(prefix) and len(prefix) > best_len:
            best_bucket, best_len = bucket, len(prefix)
    return best_bucket


class SlidingWindowCounter:
    """Fixed-duration sliding window of request timestamps per key."""

    def __init__(self, window_seconds: float = _WINDOW_SECONDS) -> None:
        self._window = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def check_and_record(self, key: str, limit: int, now: float | None = None) -> Tuple[bool, int]:
        """
        Record a hit for `key` unless it would exceed `limit`.

        Returns (allowed, retry_after_seconds).
        """
        now = time.monotonic() if now is None else now
        window = self._hits[key]

        cutoff = now - self._window
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= limit:
            retry_after = max(1, int(self._window - (now - window[0])) + 1)
            return False, retry_after

        window.append(now)
        return True, 0

    def prune(self, now: float | None = None) -> None:
        """Drop keys with no recent activity so the map cannot grow forever."""
        now = time.monotonic() if now is None else now
        cutoff = now - self._window
        for key in [k for k, w in self._hits.items() if not w or w[-1] <= cutoff]:
            del self._hits[key]


def client_key(request: Request) -> str:
    """
    Identify the caller for limiting purposes.

    Prefers the left-most X-Forwarded-For entry because the service runs behind
    a platform proxy (Render), where request.client.host is the proxy itself.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests that exceed the per-bucket allowance for their client."""

    def __init__(self, app, exempt_paths: Iterable[str] = _EXEMPT_PATHS) -> None:
        super().__init__(app)
        self._counter = SlidingWindowCounter()
        self._exempt = frozenset(exempt_paths)
        self._requests_since_prune = 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.rate_limit_enabled or request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if path in self._exempt:
            return await call_next(request)

        bucket = _classify(path)
        limit = _limit_for(bucket)
        key = f"{client_key(request)}:{bucket}"

        allowed, retry_after = self._counter.check_and_record(key, limit)

        # Amortised cleanup keeps the counter map bounded without a timer task.
        self._requests_since_prune += 1
        if self._requests_since_prune >= 500:
            self._requests_since_prune = 0
            self._counter.prune()

        if not allowed:
            logger.warning(
                "Rate limit exceeded: client=%s bucket=%s limit=%d/min path=%s",
                client_key(request), bucket, limit, path,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "Too many requests. Please wait a moment and try again."
                    ),
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
