"""
Telling one LLM failure from another.

The retry loop's old defect was not that it retried too often — it was that it
could not see what it was retrying. Every rejection arrived as a bare
`Exception` carrying a formatted string, so a 429 (the window is closed for the
next 30 seconds), a 401 (the key is wrong and always will be) and a 503 (try
again shortly) were all answered identically: sleep one second, send it again.

For the 429 that is actively harmful. Re-sending into a closed window earns
another rejection, and each rejected request still counts against the account,
so the retry policy was a cause of the rate limiting rather than a response to
it. For the 401 it is pure waste. Only the 503 was ever the case backoff was
designed for.

This module answers the two questions the retry loop needs: *what kind of
failure is this*, and *how long did the provider say to wait*. Both are read
structurally — status code, then response headers — and fall back to matching
the message only when there is no structure to read, which is the situation in
tests and behind wrappers that stringify.
"""
from __future__ import annotations

import email.utils
import logging
import re
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMErrorKind(str, Enum):
    """What a provider rejection means for whether to try again."""

    RATE_LIMITED = "rate_limited"
    """429. The window is closed; the provider usually says for how long."""

    PERMANENT = "permanent"
    """Auth, a missing model, a malformed request. Retrying re-learns this."""

    TRANSIENT = "transient"
    """5xx, connection resets, timeouts. The case backoff exists for."""


# Status codes that no retry can fix. 400 is included deliberately: a malformed
# request is malformed on the second attempt too, and the loop that produced it
# will produce it again.
_PERMANENT_STATUSES = frozenset({400, 401, 403, 404, 405, 413, 422})

_RATE_LIMIT_PATTERNS = re.compile(
    r"\b(429|rate[ _-]?limit|too many requests|tokens? per minute|\btpm\b|\brpm\b)",
    re.IGNORECASE,
)
_PERMANENT_PATTERNS = re.compile(
    r"\b(401|403|404|invalid[ _-]api[ _-]key|unauthorized|authentication|"
    r"permission denied|model[ _-]not[ _-]found|does not exist|decommissioned)\b",
    re.IGNORECASE,
)

# "6", "6.5", "6.5s", "250ms", "2m" — Groq and its proxies use several of these.
_DURATION = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(ms|s|m|min)?\s*$", re.IGNORECASE)


def _status_code(exc: BaseException) -> Optional[int]:
    """The HTTP status behind an exception, however the SDK exposes it."""
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def _headers(exc: BaseException) -> Any:
    """Response headers if the exception carries a response, else None."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    return getattr(response, "headers", None)


def classify_llm_error(exc: BaseException) -> LLMErrorKind:
    """
    Whether `exc` should be waited out, given up on, or simply retried.

    Structure first. The message is consulted only when there is no status code
    to read — which is the case for anything that has been stringified on the
    way up, and for the fakes the test suite injects.
    """
    status = _status_code(exc)
    if status is not None:
        if status == 429:
            return LLMErrorKind.RATE_LIMITED
        if status in _PERMANENT_STATUSES:
            return LLMErrorKind.PERMANENT
        return LLMErrorKind.TRANSIENT

    # Class name before message: an SDK's own RateLimitError is a stronger
    # signal than any substring, and survives an unhelpful message.
    type_name = type(exc).__name__
    if "RateLimit" in type_name:
        return LLMErrorKind.RATE_LIMITED
    if "Authentication" in type_name or "PermissionDenied" in type_name:
        return LLMErrorKind.PERMANENT

    message = str(exc)
    if _RATE_LIMIT_PATTERNS.search(message):
        return LLMErrorKind.RATE_LIMITED
    if _PERMANENT_PATTERNS.search(message):
        return LLMErrorKind.PERMANENT
    return LLMErrorKind.TRANSIENT


def _parse_duration(raw: Any) -> Optional[float]:
    """Seconds from a Retry-After style value, or None if unparseable."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return max(0.0, float(raw))

    text = str(raw).strip()
    if not text:
        return None

    match = _DURATION.match(text)
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        if unit == "ms":
            value /= 1000.0
        elif unit in ("m", "min"):
            value *= 60.0
        return max(0.0, value)

    # RFC 7231 allows an HTTP-date instead of a delta. Rare from Groq, but a
    # date parsed as garbage would otherwise become "retry immediately".
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    try:
        import datetime as _dt

        now = _dt.datetime.now(_dt.timezone.utc)
        if when.tzinfo is None:
            when = when.replace(tzinfo=_dt.timezone.utc)
        return max(0.0, (when - now).total_seconds())
    except Exception:  # pragma: no cover — defensive
        return None


# Checked in order. `retry-after` is the standard; the x-ratelimit-reset pair is
# what Groq actually sends when a token budget rather than a request count is
# what ran out, and it is often the only one present.
_RETRY_AFTER_HEADERS = (
    "retry-after",
    "x-ratelimit-reset-tokens",
    "x-ratelimit-reset-requests",
)


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """
    How long the provider asked us to wait, or None if it did not say.

    Returning None rather than a default is deliberate: "the provider gave no
    guidance" and "the provider said zero" are different states, and the caller
    picks its own floor for the first one.
    """
    headers = _headers(exc)
    if headers is not None:
        for name in _RETRY_AFTER_HEADERS:
            raw = None
            try:
                raw = headers.get(name)
            except Exception:  # pragma: no cover — exotic header mappings
                raw = None
            seconds = _parse_duration(raw)
            if seconds is not None:
                return seconds

    # Some wrappers put it on the exception itself.
    for attr in ("retry_after", "retry_after_seconds"):
        seconds = _parse_duration(getattr(exc, attr, None))
        if seconds is not None:
            return seconds

    # Last resort: Groq states the wait in the message body of a 429
    # ("Please try again in 6.573s").
    match = re.search(
        r"try again in\s+([0-9]*\.?[0-9]+)\s*(ms|s|m|min)?", str(exc), re.IGNORECASE
    )
    if match:
        return _parse_duration(f"{match.group(1)}{match.group(2) or 's'}")

    return None
