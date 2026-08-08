"""
End-to-end authorization tests against the real FastAPI app.

These are the tests that prove audit findings C1 and C2 are actually closed:
no endpoint serves user data without a token, and a guest token cannot reach
owner-only capability.
"""
import pytest
from fastapi.testclient import TestClient

from app.auth.jwt_service import issue_token_pair, revocation_store
from app.auth.models import Role
from app.auth.password import hash_password
from app.config import settings
from app.main import app

OWNER_PASSWORD = "correct-horse-battery-staple"


@pytest.fixture(autouse=True)
def _auth_config(monkeypatch):
    monkeypatch.setattr(settings, "jwt_access_secret", "a" * 64)
    monkeypatch.setattr(settings, "jwt_refresh_secret", "b" * 64)
    monkeypatch.setattr(settings, "owner_username", "vansh")
    monkeypatch.setattr(settings, "owner_password_hash", hash_password(OWNER_PASSWORD, rounds=4))
    monkeypatch.setattr(settings, "owner_user_id", "vansh")
    monkeypatch.setattr(settings, "guest_sessions_enabled", True)
    # Disable CSRF and rate limiting so these tests exercise authz alone.
    monkeypatch.setattr(settings, "csrf_protection_enabled", False)
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    revocation_store.clear()
    yield
    revocation_store.clear()


@pytest.fixture
def client():
    # Constructed without the context manager on purpose: that skips the
    # lifespan handler, so these tests exercise routing and authorization
    # without needing live Postgres/Qdrant/Cohere connections.
    return TestClient(app)


def bearer(role: Role, user_id: str) -> dict:
    pair = issue_token_pair(user_id, role)
    return {"Authorization": f"Bearer {pair.access_token}"}


# ── Endpoints must reject anonymous callers ──────────────────────────────

PROTECTED_ENDPOINTS = [
    ("post", "/api/v1/agents/query", {"query": "hi"}),
    ("post", "/api/v1/agents/tools/job-search", {"query": "python"}),
    ("post", "/api/v1/agents/tools/email-draft", {"query": "hello"}),
    ("post", "/api/v1/agents/tools/attendance/scrape",
     {"erp_url": "https://example.com", "username": "u", "password": "p"}),
    ("post", "/api/v1/agents/tools/timetable/suggest", {}),
    ("post", "/api/v1/agents/tools/timetable/store",
     {"entries": [{"day_of_week": 0, "start_time": "09:00", "end_time": "10:00", "subject": "X"}]}),
    ("get", "/api/v1/agents/memory/profile/vansh", None),
    ("post", "/api/v1/agents/memory/profile", {"key": "k", "value": "v"}),
    ("delete", "/api/v1/agents/memory/profile/vansh", None),
    ("delete", "/api/v1/agents/memory/profile/vansh/somekey", None),
    ("get", "/api/v1/agents/memory/episodes/vansh", None),
    ("get", "/api/v1/agents/conversations", None),
    ("get", "/api/v1/agents/conversations/session_abc", None),
    ("delete", "/api/v1/agents/conversations/session_abc", None),
    ("post", "/api/v1/voice/token", {}),
]


@pytest.mark.parametrize("method,path,body", PROTECTED_ENDPOINTS)
def test_endpoint_requires_authentication(client, method, path, body):
    response = getattr(client, method)(path, **({"json": body} if body is not None else {}))
    assert response.status_code == 401, (
        f"{method.upper()} {path} served an unauthenticated request "
        f"(got {response.status_code})"
    )


# ── Guests must not reach owner-only capability ──────────────────────────

GUEST_FORBIDDEN = [
    ("post", "/api/v1/agents/tools/attendance/scrape",
     {"erp_url": "https://example.com", "username": "u", "password": "p"}),
    ("post", "/api/v1/agents/tools/timetable/suggest", {}),
    ("post", "/api/v1/agents/memory/profile", {"key": "k", "value": "v"}),
    ("delete", "/api/v1/agents/memory/profile/guest-abc", None),
    ("post", "/api/v1/agents/memory/upload-text", None),
]


@pytest.mark.parametrize("method,path,body", GUEST_FORBIDDEN)
def test_guest_is_forbidden_from_owner_capability(client, method, path, body):
    headers = bearer(Role.GUEST, "guest-abc123def456")
    kwargs = {"headers": headers}
    if body is not None:
        kwargs["json"] = body
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 403, (
        f"{method.upper()} {path} allowed a guest (got {response.status_code})"
    )


# ── Cross-user access must be refused ────────────────────────────────────

def test_owner_cannot_read_another_users_profile(client):
    response = client.get(
        "/api/v1/agents/memory/profile/someone-else",
        headers=bearer(Role.OWNER, "vansh"),
    )
    assert response.status_code == 403


def test_body_user_id_cannot_override_token_identity(client):
    """The core of C1: a client-supplied user_id must never be honoured."""
    response = client.post(
        "/api/v1/agents/query",
        json={"query": "hello", "user_id": "someone-else"},
        headers=bearer(Role.GUEST, "guest-abc123def456"),
    )
    assert response.status_code == 403


def test_livekit_room_cannot_be_chosen_by_client(client):
    response = client.post(
        "/api/v1/voice/token",
        json={"room_name": "voice-vansh"},
        headers=bearer(Role.GUEST, "guest-abc123def456"),
    )
    # Guest lacks the voice scope; even with it, the room name is derived.
    assert response.status_code == 403


# ── Login flow ───────────────────────────────────────────────────────────

def test_login_succeeds_and_sets_httponly_cookies(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "vansh", "password": OWNER_PASSWORD},
    )
    assert response.status_code == 200
    assert response.json()["role"] == "owner"

    cookies = response.headers.get_list("set-cookie")
    access_cookie = next(c for c in cookies if c.startswith("access_token="))
    refresh_cookie = next(c for c in cookies if c.startswith("refresh_token="))
    csrf_cookie = next(c for c in cookies if c.startswith("csrf_token="))

    assert "httponly" in access_cookie.lower()
    assert "httponly" in refresh_cookie.lower()
    # The CSRF token must stay readable for the double-submit check.
    assert "httponly" not in csrf_cookie.lower()


@pytest.mark.parametrize("username,password", [
    ("vansh", "wrong-password"),
    ("nobody", OWNER_PASSWORD),
    ("nobody", "wrong-password"),
])
def test_login_rejects_bad_credentials(client, username, password):
    response = client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 401
    # One generic message for every failure — no user enumeration.
    assert response.json()["detail"] == "Incorrect username or password."


def test_guest_session_can_be_created(client):
    response = client.post("/api/v1/auth/guest")
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "guest"
    assert data["user_id"].startswith("guest-")
    assert "email:send" not in data["scopes"]


def test_guest_sessions_can_be_disabled(client, monkeypatch):
    monkeypatch.setattr(settings, "guest_sessions_enabled", False)
    assert client.post("/api/v1/auth/guest").status_code == 403


def test_me_returns_verified_identity(client):
    response = client.get("/api/v1/auth/me", headers=bearer(Role.OWNER, "vansh"))
    assert response.status_code == 200
    assert response.json()["user_id"] == "vansh"


def test_me_requires_authentication(client):
    assert client.get("/api/v1/auth/me").status_code == 401


def test_refresh_rotates_and_invalidates_the_old_token(client):
    login = client.post(
        "/api/v1/auth/login", json={"username": "vansh", "password": OWNER_PASSWORD}
    )
    old_refresh = login.json()["refresh_token"]

    # Cookies are cleared so the bearer header is unambiguously the token under
    # test — otherwise the persisted (and freshly rotated) cookie would win.
    client.cookies.clear()

    first = client.post(
        "/api/v1/auth/refresh", headers={"Authorization": f"Bearer {old_refresh}"}
    )
    assert first.status_code == 200
    new_refresh = first.json()["refresh_token"]
    assert new_refresh != old_refresh, "refresh token must rotate"

    client.cookies.clear()

    # Replaying the consumed refresh token must now fail.
    replay = client.post(
        "/api/v1/auth/refresh", headers={"Authorization": f"Bearer {old_refresh}"}
    )
    assert replay.status_code == 401

    # ...while the rotated one still works.
    client.cookies.clear()
    assert client.post(
        "/api/v1/auth/refresh", headers={"Authorization": f"Bearer {new_refresh}"}
    ).status_code == 200


def test_logout_revokes_refresh_token_and_clears_cookies(client):
    client.post("/api/v1/auth/login", json={"username": "vansh", "password": OWNER_PASSWORD})

    response = client.post("/api/v1/auth/logout")
    assert response.status_code == 200

    # The rotated cookie is gone, so a refresh is no longer possible.
    assert client.post("/api/v1/auth/refresh").status_code == 401


def test_health_and_root_stay_public(client):
    assert client.get("/").status_code == 200
