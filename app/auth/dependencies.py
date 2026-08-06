"""
Request-level authentication and authorization dependencies.

Every protected endpoint resolves its caller through `get_current_principal`,
which reads the JWT from an HttpOnly cookie (browsers) or an Authorization
header (scripts). A `user_id` in a request body, query string, or path is never
an identity source — at most it is cross-checked against the verified one and
rejected on mismatch.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from fastapi import Depends, HTTPException, Request, WebSocket, status

from app.auth.cookies import ACCESS_COOKIE, CSRF_COOKIE, CSRF_HEADER
from app.auth.jwt_service import (
    ACCESS_TOKEN_TYPE,
    AuthError,
    decode_token,
    principal_from_payload,
)
from app.auth.models import Principal, Scope
from app.auth.password import constant_time_equals
from app.config import settings

logger = logging.getLogger(__name__)

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Authentication required.",
    headers={"WWW-Authenticate": "Bearer"},
)

# Methods that can change state and therefore require CSRF proof.
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _extract_token(request: Request) -> tuple[Optional[str], bool]:
    """
    Pull the access token from the request.

    Returns (token, came_from_cookie). The flag matters because only cookie
    auth is vulnerable to CSRF — a bearer token is never sent automatically.
    """
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
        if token:
            return token, False

    cookie_token = request.cookies.get(ACCESS_COOKIE)
    if cookie_token:
        return cookie_token, True

    return None, False


def _verify_csrf(request: Request) -> None:
    """
    Double-submit CSRF check for cookie-authenticated mutations.

    A cross-site attacker can make the browser send cookies but cannot read
    them, so it cannot reproduce the cookie value in a custom header.
    """
    if not settings.csrf_protection_enabled:
        return
    if request.method.upper() not in _MUTATING_METHODS:
        return

    cookie_value = request.cookies.get(CSRF_COOKIE, "")
    header_value = request.headers.get(CSRF_HEADER, "")

    if not cookie_value or not header_value or not constant_time_equals(
        cookie_value, header_value
    ):
        logger.warning(
            "CSRF check failed for %s %s (cookie_present=%s, header_present=%s)",
            request.method, request.url.path, bool(cookie_value), bool(header_value),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed. Refresh the page and try again.",
        )


async def get_current_principal(request: Request) -> Principal:
    """Resolve and verify the caller. Raises 401 when unauthenticated."""
    token, from_cookie = _extract_token(request)
    if not token:
        raise _UNAUTHENTICATED

    if from_cookie:
        _verify_csrf(request)

    try:
        payload = decode_token(token, ACCESS_TOKEN_TYPE)
        principal = principal_from_payload(payload)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Make the principal available to logging/telemetry downstream.
    request.state.principal = principal
    return principal


async def get_optional_principal(request: Request) -> Optional[Principal]:
    """Resolve the caller if present, without requiring authentication."""
    try:
        return await get_current_principal(request)
    except HTTPException:
        return None


def require_scope(*required: Scope | str) -> Callable:
    """
    Dependency factory enforcing that the caller holds every listed scope.

    Returns 403 (not 401) on an authenticated caller lacking permission, so the
    frontend can distinguish "sign in" from "your account cannot do this".
    """

    async def _dependency(
        principal: Principal = Depends(get_current_principal),
    ) -> Principal:
        if not principal.has_all_scopes(*required):
            missing = [
                s.value if isinstance(s, Scope) else s
                for s in required
                if not principal.has_scope(s)
            ]
            logger.warning(
                "Authorization denied for user=%s role=%s missing=%s",
                principal.user_id, principal.role.value, missing,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Your account does not have permission to perform this action."
                ),
            )
        return principal

    return _dependency


async def require_owner(
    principal: Principal = Depends(get_current_principal),
) -> Principal:
    """Restrict an endpoint to the account that owns this assistant."""
    if not principal.is_owner:
        logger.warning(
            "Owner-only endpoint refused for user=%s role=%s",
            principal.user_id, principal.role.value,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action is restricted to the account owner.",
        )
    return principal


def resolve_user_id(principal: Principal, claimed_user_id: Optional[str] = None) -> str:
    """
    Return the authenticated user_id, rejecting any conflicting claim.

    Request bodies and paths still carry `user_id` for backward compatibility,
    but it is never authoritative. Supplying someone else's id is treated as an
    access attempt and refused loudly rather than silently ignored.
    """
    if claimed_user_id is not None:
        normalized = claimed_user_id.strip().lower()
        if normalized and normalized != principal.user_id.lower():
            logger.warning(
                "Rejected user_id mismatch: authenticated=%s claimed=%s",
                principal.user_id, normalized,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You cannot access another user's data.",
            )
    return principal.user_id


async def authenticate_websocket(websocket: WebSocket) -> Optional[Principal]:
    """
    Authenticate a WebSocket handshake.

    Browsers cannot set headers on a WebSocket, so the cookie sent with the
    upgrade request is the transport here. CSRF is covered by the Origin check
    performed by the caller before this runs.
    """
    token = websocket.cookies.get(ACCESS_COOKIE)

    if not token:
        # Fallback for non-browser clients that can set headers.
        header = websocket.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:].strip()

    if not token:
        return None

    try:
        payload = decode_token(token, ACCESS_TOKEN_TYPE)
        return principal_from_payload(payload)
    except AuthError as exc:
        logger.info("WebSocket authentication failed: %s", exc)
        return None
