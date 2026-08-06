"""
Authentication endpoints: login, guest session, refresh, logout, and identity.

Tokens are delivered as HttpOnly cookies by default. The response body also
carries them so non-browser clients (scripts, CI) can use bearer auth without
a cookie jar; browsers should ignore the body and rely on the cookies.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.auth.cookies import (
    CSRF_COOKIE,
    REFRESH_COOKIE,
    clear_auth_cookies,
    generate_csrf_token,
    set_auth_cookies,
)
from app.auth.dependencies import get_current_principal
from app.auth.jwt_service import (
    REFRESH_TOKEN_TYPE,
    AuthError,
    decode_token,
    issue_token_pair,
    new_guest_user_id,
    principal_from_payload,
    revoke_token,
)
from app.auth.models import Principal, Role
from app.auth.password import constant_time_equals, verify_password
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=256)


class SessionResponse(BaseModel):
    """Identity plus, for non-browser clients, the raw tokens."""

    user_id: str
    role: str
    scopes: list[str]
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    csrf_token: Optional[str] = None


class IdentityResponse(BaseModel):
    user_id: str
    role: str
    scopes: list[str]


def _session_response(
    response: Response,
    user_id: str,
    role: Role,
    session_id: Optional[str] = None,
) -> SessionResponse:
    """Mint a token pair, set cookies, and build the response body."""
    pair = issue_token_pair(user_id=user_id, role=role, session_id=session_id)
    csrf_token = set_auth_cookies(
        response,
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        access_max_age=pair.access_expires_in,
        refresh_max_age=pair.refresh_expires_in,
        csrf_token=generate_csrf_token(),
    )

    from app.auth.models import scopes_for_role

    return SessionResponse(
        user_id=user_id,
        role=role.value,
        scopes=sorted(scopes_for_role(role)),
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_in=pair.access_expires_in,
        csrf_token=csrf_token,
    )


@router.post("/login", response_model=SessionResponse)
async def login(payload: LoginRequest, response: Response, request: Request):
    """
    Authenticate the owner with username and password.

    Failures return one generic message and always perform a full password
    comparison, so neither the response text nor its timing reveals whether the
    username exists.
    """
    if not settings.is_owner_login_configured:
        logger.error("Login attempted but OWNER_USERNAME/OWNER_PASSWORD_HASH are unset")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Login is not configured on this server.",
        )

    username_ok = constant_time_equals(
        payload.username.strip().lower(),
        (settings.owner_username or "").strip().lower(),
    )
    # Always verify, even on username mismatch, to keep timing uniform.
    password_ok = verify_password(payload.password, settings.owner_password_hash)

    if not (username_ok and password_ok):
        client = request.client.host if request.client else "unknown"
        logger.warning("Failed login attempt for username=%s from %s", payload.username, client)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
        )

    logger.info("Owner login succeeded for user_id=%s", settings.owner_user_id)
    return _session_response(response, settings.owner_user_id, Role.OWNER)


@router.post("/guest", response_model=SessionResponse)
async def create_guest_session(response: Response):
    """
    Issue an anonymous, read-only session.

    This backs the public recruiter view. Guests receive their own namespaced
    user_id so their conversation history stays isolated from the owner's, and
    a scope set that excludes sending email, mutating memory, and everything
    academic (see GUEST_SCOPES).
    """
    if not settings.guest_sessions_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Guest access is disabled on this server.",
        )

    guest_id = new_guest_user_id()
    logger.info("Issued guest session user_id=%s", guest_id)
    return _session_response(response, guest_id, Role.GUEST)


@router.post("/refresh", response_model=SessionResponse)
async def refresh_session(request: Request, response: Response):
    """
    Exchange a refresh token for a new pair, rotating the old one.

    Rotation means a refresh token is single-use: replaying one that has
    already been redeemed fails, so theft degrades into a detectable error
    rather than indefinite access.
    """
    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if not refresh_token:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            refresh_token = header[7:].strip()

    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No refresh token provided.",
        )

    try:
        payload = decode_token(refresh_token, REFRESH_TOKEN_TYPE)
        principal = principal_from_payload(payload)
    except AuthError as exc:
        # Clear cookies so a client holding a dead token stops retrying.
        clear_auth_cookies(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    revoke_token(refresh_token, REFRESH_TOKEN_TYPE)

    logger.info("Rotated refresh token for user=%s", principal.user_id)
    return _session_response(
        response,
        principal.user_id,
        principal.role,
        session_id=principal.session_id,
    )


@router.post("/logout")
async def logout(request: Request, response: Response):
    """
    End the session: revoke the refresh token and clear all auth cookies.

    Deliberately unauthenticated — logging out must succeed even when the
    access token has already expired.
    """
    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if refresh_token:
        revoke_token(refresh_token, REFRESH_TOKEN_TYPE)

    clear_auth_cookies(response)
    return {"success": True, "message": "Signed out."}


@router.get("/me", response_model=IdentityResponse)
async def whoami(principal: Principal = Depends(get_current_principal)):
    """Return the verified identity behind the current session."""
    return IdentityResponse(
        user_id=principal.user_id,
        role=principal.role.value,
        scopes=sorted(principal.scopes),
    )


@router.get("/csrf")
async def rotate_csrf_token(request: Request, response: Response):
    """
    Hand the client a CSRF token when it has cookies but no readable token
    (for example after a page load in a fresh tab).
    """
    existing = request.cookies.get(CSRF_COOKIE)
    if existing:
        return {"csrf_token": existing}

    token = generate_csrf_token()
    response.set_cookie(
        CSRF_COOKIE,
        token,
        httponly=False,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        domain=settings.auth_cookie_domain,
        path="/",
    )
    return {"csrf_token": token}
