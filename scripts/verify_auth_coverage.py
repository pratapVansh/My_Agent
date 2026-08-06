"""
Verify that no route serves user data without authentication.

Walks the live FastAPI route table and inspects each endpoint's dependency
graph, rather than grepping source, so a route that merely *looks* protected
cannot pass. Fails with a non-zero exit code if any gap is found, which makes
it usable as a CI gate.

Usage:
    python scripts/verify_auth_coverage.py
"""
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.routing import APIRoute, APIWebSocketRoute  # noqa: E402

from app.auth.dependencies import (  # noqa: E402
    get_current_principal,
    require_owner,
)
from app.main import app  # noqa: E402

# Endpoints that are public by design.
PUBLIC_PATHS = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/docs/oauth2-redirect",
    "/api/v1/auth/login",       # issues a session; cannot require one
    "/api/v1/auth/guest",       # issues an anonymous session
    "/api/v1/auth/refresh",     # authenticates via refresh token itself
    "/api/v1/auth/logout",      # must work with an expired access token
    "/api/v1/auth/csrf",        # hands out a CSRF token, grants no access
    "/api/v1/agents/agents",    # static capability catalogue, no user data
}

# WebSockets authenticate inside the handler (a dependency cannot close the
# socket with a status code), so they are verified separately.
WEBSOCKET_PATHS = {"/api/v1/agents/stream"}

_AUTH_DEPENDENCIES = {get_current_principal, require_owner}


def route_is_protected(route: APIRoute) -> bool:
    """True when the route's dependency tree reaches an auth dependency."""
    for dependency in route.dependant.dependencies:
        if dependency.call in _AUTH_DEPENDENCIES:
            return True
        # require_scope(...) returns a closure that depends on
        # get_current_principal, so check one level deeper too.
        for nested in dependency.dependencies:
            if nested.call in _AUTH_DEPENDENCIES:
                return True
    return False


def main() -> None:
    unprotected: List[Tuple[str, str]] = []
    protected: List[Tuple[str, str]] = []
    public: List[str] = []
    websockets: List[str] = []

    for route in app.routes:
        if isinstance(route, APIWebSocketRoute):
            websockets.append(route.path)
            continue
        if not isinstance(route, APIRoute):
            continue

        methods = ",".join(sorted(route.methods - {"HEAD", "OPTIONS"}))

        if route.path in PUBLIC_PATHS:
            public.append(f"{methods} {route.path}")
            continue

        if route_is_protected(route):
            protected.append((methods, route.path))
        else:
            unprotected.append((methods, route.path))

    print("=" * 72)
    print("AUTHENTICATION COVERAGE")
    print("=" * 72)

    print(f"\nProtected ({len(protected)}):")
    for methods, path in sorted(protected, key=lambda r: r[1]):
        print(f"  [OK]   {methods:<18} {path}")

    print(f"\nPublic by design ({len(public)}):")
    for entry in sorted(public):
        print(f"  [PUB]  {entry}")

    print(f"\nWebSocket — authenticated in-handler ({len(websockets)}):")
    for path in sorted(websockets):
        marker = "OK" if path in WEBSOCKET_PATHS else "??"
        print(f"  [{marker}]   {path}")

    if unprotected:
        print(f"\nUNPROTECTED ({len(unprotected)}):")
        for methods, path in sorted(unprotected, key=lambda r: r[1]):
            print(f"  [FAIL] {methods:<18} {path}")
        print("\nRESULT: FAILED — the routes above serve requests without authentication.")
        sys.exit(1)

    unknown_ws = set(websockets) - WEBSOCKET_PATHS
    if unknown_ws:
        print(f"\nRESULT: FAILED — unreviewed WebSocket routes: {sorted(unknown_ws)}")
        sys.exit(1)

    print("\nRESULT: PASSED — every data-bearing route requires authentication.")


if __name__ == "__main__":
    main()
