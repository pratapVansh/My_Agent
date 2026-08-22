"""
JWT issuance and verification.

Design notes
------------
* Access and refresh tokens are signed with *different* secrets, so a stolen
  access token cannot be presented as a refresh token even if the `typ` claim
  were somehow bypassed.
* Refresh tokens rotate: redeeming one revokes it and issues a fresh pair. A
  replayed refresh token is therefore rejected, which turns token theft into a
  detectable event rather than indefinite access.
* Revocation is an in-process store keyed by `jti`. That is correct for the
  current single-worker deployment; running multiple instances requires a
  shared store (Redis) or the revocation list will not be seen by every worker.
  See docs/AUDIT_REPORT.md (M15) for the same constraint on LiveKit worker state.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt

from app.auth.models import Principal, Role, scopes_for_role
from app.config import settings

logger = logging.getLogger(__name__)

ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"


class AuthError(Exception):
    """Token could not be verified. Message is safe to return to the client."""


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
    access_expires_in: int      # seconds
    refresh_expires_in: int     # seconds


class RevocationStore:
    """
    Tracks revoked token ids until their natural expiry.

    Entries are dropped once the token would have expired anyway, so the store
    stays bounded without a background sweeper.
    """

    def __init__(self) -> None:
        self._revoked: Dict[str, float] = {}

    def revoke(self, token_id: str, expires_at_epoch: float) -> None:
        self._prune()
        self._revoked[token_id] = expires_at_epoch

    def is_revoked(self, token_id: str) -> bool:
        self._prune()
        return token_id in self._revoked

    def _prune(self) -> None:
        now = time.time()
        for token_id in [tid for tid, exp in self._revoked.items() if exp <= now]:
            del self._revoked[token_id]

    def clear(self) -> None:
        self._revoked.clear()


revocation_store = RevocationStore()


def _secret_for(token_type: str) -> str:
    secret = (
        settings.jwt_access_secret
        if token_type == ACCESS_TOKEN_TYPE
        else settings.jwt_refresh_secret
    )
    if not (secret or "").strip():
        # Refusing to sign is the only safe response: a default secret would
        # let anyone mint an owner token.
        raise AuthError("Authentication is not configured on this server.")
    return secret


def _encode(
    *,
    token_type: str,
    user_id: str,
    role: Role,
    scopes: frozenset[str],
    ttl: timedelta,
    session_id: Optional[str] = None,
) -> tuple[str, str, int]:
    """Return (token, jti, ttl_seconds)."""
    now = datetime.now(timezone.utc)
    token_id = uuid.uuid4().hex
    payload: Dict[str, Any] = {
        "sub": user_id,
        "role": role.value,
        "scopes": sorted(scopes),
        "typ": token_type,
        "jti": token_id,
        "iss": settings.jwt_issuer,
        "iat": now,
        "exp": now + ttl,
    }
    if session_id:
        payload["sid"] = session_id

    token = jwt.encode(payload, _secret_for(token_type), algorithm=settings.jwt_algorithm)
    return token, token_id, int(ttl.total_seconds())


def issue_token_pair(
    user_id: str,
    role: Role,
    session_id: Optional[str] = None,
) -> TokenPair:
    """Mint a fresh access + refresh pair for a verified identity."""
    scopes = scopes_for_role(role)

    refresh_days = (
        settings.owner_refresh_token_ttl_days
        if role is Role.OWNER
        else settings.guest_refresh_token_ttl_days
    )

    access_token, _, access_ttl = _encode(
        token_type=ACCESS_TOKEN_TYPE,
        user_id=user_id,
        role=role,
        scopes=scopes,
        ttl=timedelta(minutes=settings.access_token_ttl_minutes),
        session_id=session_id,
    )
    refresh_token, _, refresh_ttl = _encode(
        token_type=REFRESH_TOKEN_TYPE,
        user_id=user_id,
        role=role,
        scopes=scopes,
        ttl=timedelta(days=refresh_days),
        session_id=session_id,
    )

    logger.info("Issued token pair for user=%s role=%s", user_id, role.value)
    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        access_expires_in=access_ttl,
        refresh_expires_in=refresh_ttl,
    )


def decode_token(token: str, expected_type: str) -> Dict[str, Any]:
    """
    Verify a token's signature, expiry, issuer, type, and revocation status.

    Raises AuthError with a client-safe message on any failure.
    """
    try:
        payload = jwt.decode(
            token,
            _secret_for(expected_type),
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "sub", "jti", "typ"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Session expired.")
    except jwt.InvalidIssuerError:
        raise AuthError("Invalid token issuer.")
    except jwt.InvalidTokenError as exc:
        logger.warning("Rejected malformed token: %s", exc)
        raise AuthError("Invalid authentication token.")

    # Belt-and-braces: separate secrets already make cross-use impossible, but
    # an explicit type check keeps the intent obvious and survives a future
    # refactor that unifies the secrets.
    if payload.get("typ") != expected_type:
        raise AuthError("Invalid authentication token.")

    token_id = payload.get("jti")
    if token_id and revocation_store.is_revoked(token_id):
        raise AuthError("Session has been revoked. Please sign in again.")

    return payload


def principal_from_payload(payload: Dict[str, Any]) -> Principal:
    """
    Build a Principal from verified claims.

    Scopes are re-derived from the role rather than trusted from the token, so
    tightening GUEST_SCOPES takes effect immediately for tokens already issued
    instead of waiting for them to expire.
    """
    try:
        role = Role(payload.get("role", ""))
    except ValueError:
        raise AuthError("Invalid authentication token.")

    user_id = (payload.get("sub") or "").strip()
    if not user_id:
        raise AuthError("Invalid authentication token.")

    return Principal(
        user_id=user_id,
        role=role,
        scopes=scopes_for_role(role),
        token_id=payload.get("jti"),
        session_id=payload.get("sid"),
    )


def revoke_token(token: str, expected_type: str) -> None:
    """Best-effort revocation; an unverifiable token needs no revoking."""
    try:
        payload = jwt.decode(
            token,
            _secret_for(expected_type),
            algorithms=[settings.jwt_algorithm],
            options={"verify_exp": False, "verify_iss": False},
        )
    except (jwt.InvalidTokenError, AuthError):
        return

    token_id = payload.get("jti")
    if token_id:
        revocation_store.revoke(token_id, float(payload.get("exp", time.time())))
        logger.info("Revoked token jti=%s for user=%s", token_id, payload.get("sub"))


def new_guest_user_id() -> str:
    """
    Namespaced, unguessable identity for an anonymous visitor.

    Guests get their own user_id so their conversation history is isolated from
    the owner's and from each other's, while still satisfying the API's
    user_id format rules.
    """
    return f"guest-{uuid.uuid4().hex[:24]}"
