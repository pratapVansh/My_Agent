"""
Auth cookie transport and CSRF protection.

Why HttpOnly cookies rather than localStorage
---------------------------------------------
Tokens in localStorage are readable by any JavaScript on the page, so a single
XSS bug (including one in a third-party script) exfiltrates a long-lived
credential. HttpOnly cookies are invisible to JavaScript, so the same XSS can
at most ride the session while the page is open — it cannot steal the token.

The cost is that browsers attach cookies automatically, which reintroduces
CSRF. That is handled here with the double-submit pattern: a readable
`csrf_token` cookie must be echoed back in the `X-CSRF-Token` header on every
mutating request. An attacker's site can cause the browser to *send* cookies
but cannot *read* them, so it cannot produce the matching header.

Bearer tokens remain supported for non-browser clients (scripts, CI). Those are
CSRF-immune by construction because nothing is sent automatically.
"""
from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Response

from app.config import settings

ACCESS_COOKIE = "access_token"
REFRESH_COOKIE = "refresh_token"
CSRF_COOKIE = "csrf_token"
CSRF_HEADER = "X-CSRF-Token"

# The refresh cookie is only ever sent to the endpoints that rotate or clear
# it, so it is never exposed to ordinary API traffic.
REFRESH_COOKIE_PATH = "/api/v1/auth"


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    access_max_age: int,
    refresh_max_age: int,
    csrf_token: Optional[str] = None,
) -> str:
    """Attach the auth cookie set to `response`. Returns the CSRF token used."""
    csrf_token = csrf_token or generate_csrf_token()
    secure = settings.cookie_secure
    samesite = settings.cookie_samesite
    domain = settings.auth_cookie_domain

    response.set_cookie(
        ACCESS_COOKIE,
        access_token,
        max_age=access_max_age,
        httponly=True,
        secure=secure,
        samesite=samesite,
        domain=domain,
        path="/",
    )
    response.set_cookie(
        REFRESH_COOKIE,
        refresh_token,
        max_age=refresh_max_age,
        httponly=True,
        secure=secure,
        samesite=samesite,
        domain=domain,
        path=REFRESH_COOKIE_PATH,
    )
    # Deliberately NOT HttpOnly: the frontend must read this to echo it back.
    # It carries no authority on its own — it is only a proof that the caller
    # can read the cookie jar for this origin.
    response.set_cookie(
        CSRF_COOKIE,
        csrf_token,
        max_age=refresh_max_age,
        httponly=False,
        secure=secure,
        samesite=samesite,
        domain=domain,
        path="/",
    )
    return csrf_token


def clear_auth_cookies(response: Response) -> None:
    """Remove every auth cookie, matching the attributes used to set them."""
    domain = settings.auth_cookie_domain
    samesite = settings.cookie_samesite
    secure = settings.cookie_secure

    for name, path in (
        (ACCESS_COOKIE, "/"),
        (REFRESH_COOKIE, REFRESH_COOKIE_PATH),
        (CSRF_COOKIE, "/"),
    ):
        response.delete_cookie(
            name,
            path=path,
            domain=domain,
            samesite=samesite,
            secure=secure,
        )
