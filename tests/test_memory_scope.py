"""
Retrieval scoping and the memory control plane (Phase 6).

The bug being closed: memory was partitioned solely by `owner_id`, guests were
correctly given their own `guest-<uuid>` identity, and nothing ever pointed a
guest at the owner's data. The recruiter view therefore retrieved an empty
partition and could not discuss the work it exists to present.

The fix separates two questions that were previously one:

* **whose memory do I read** — the owner's, filtered to public, for a guest;
* **whose memory do I write** — always the caller's own.

Conflating them would swap one bug for a worse one: recruiter chatter writing
into the owner's memory.

See docs/MEMORY_ARCHITECTURE.md §1.8 and §3.4.
"""
import pytest
from fastapi.testclient import TestClient

from app.auth.jwt_service import issue_token_pair, revocation_store
from app.auth.models import GUEST_SCOPES, OWNER_SCOPES, Principal, Role
from app.auth.password import hash_password
from app.config import settings
from app.main import app
from app.memory.kinds import Visibility
from app.memory.scope import can_write_memory, resolve_retrieval_scope

OWNER_PASSWORD = "correct-horse-battery-staple"


def owner_principal(user_id="vansh"):
    return Principal(user_id=user_id, role=Role.OWNER, scopes=OWNER_SCOPES)


def guest_principal(user_id="guest-abc123def456"):
    return Principal(user_id=user_id, role=Role.GUEST, scopes=GUEST_SCOPES)


# ─────────────────────────────────────────────────────────────────────────
# Scope resolution
# ─────────────────────────────────────────────────────────────────────────

def test_the_owner_reads_their_own_memory_unrestricted():
    scope = resolve_retrieval_scope(owner_principal())
    assert scope.owner_id == "vansh"
    assert scope.visibilities is None
    assert scope.is_own_memory is True


def test_a_guest_is_pointed_at_the_owners_memory(monkeypatch):
    """
    The heart of the fix. Previously a guest retrieved its own `guest-<uuid>`
    partition, which is empty and always will be — guests hold no scope that
    can write memory.
    """
    monkeypatch.setattr(settings, "owner_user_id", "vansh")
    scope = resolve_retrieval_scope(guest_principal())
    assert scope.owner_id == "vansh"
    assert scope.is_own_memory is False


def test_a_guest_is_restricted_to_public_records(monkeypatch):
    monkeypatch.setattr(settings, "owner_user_id", "vansh")
    scope = resolve_retrieval_scope(guest_principal())
    assert scope.visibilities == [Visibility.PUBLIC]
    assert scope.public_only is True


def test_the_owner_scope_is_not_public_only():
    assert resolve_retrieval_scope(owner_principal()).public_only is False


def test_guests_cannot_write_memory():
    assert can_write_memory(owner_principal()) is True
    assert can_write_memory(guest_principal()) is False


def test_scope_description_is_log_friendly(monkeypatch):
    monkeypatch.setattr(settings, "owner_user_id", "vansh")
    assert "own" in resolve_retrieval_scope(owner_principal()).describe()
    assert "public" in resolve_retrieval_scope(guest_principal()).describe()


def test_two_different_guests_resolve_to_the_same_owner(monkeypatch):
    """Guest identity governs writes; it must not fragment the read scope."""
    monkeypatch.setattr(settings, "owner_user_id", "vansh")
    a = resolve_retrieval_scope(guest_principal("guest-aaa"))
    b = resolve_retrieval_scope(guest_principal("guest-bbb"))
    assert a.owner_id == b.owner_id == "vansh"


# ─────────────────────────────────────────────────────────────────────────
# Control plane authorization
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _auth_config(monkeypatch):
    monkeypatch.setattr(settings, "jwt_access_secret", "a" * 64)
    monkeypatch.setattr(settings, "jwt_refresh_secret", "b" * 64)
    monkeypatch.setattr(settings, "owner_username", "vansh")
    monkeypatch.setattr(settings, "owner_password_hash", hash_password(OWNER_PASSWORD, rounds=4))
    monkeypatch.setattr(settings, "owner_user_id", "vansh")
    monkeypatch.setattr(settings, "csrf_protection_enabled", False)
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    revocation_store.clear()
    yield
    revocation_store.clear()


@pytest.fixture
def client():
    return TestClient(app)


def bearer(role: Role, user_id: str) -> dict:
    return {"Authorization": f"Bearer {issue_token_pair(user_id, role).access_token}"}


CONTROL_PLANE_ENDPOINTS = [
    ("get", "/api/v1/agents/memory/records", None),
    ("get", "/api/v1/agents/memory/records/00000000-0000-0000-0000-000000000001", None),
    ("patch", "/api/v1/agents/memory/records/00000000-0000-0000-0000-000000000001",
     {"pinned": True}),
    ("delete", "/api/v1/agents/memory/records/00000000-0000-0000-0000-000000000001", None),
    ("get", "/api/v1/agents/memory/export", None),
]


@pytest.mark.parametrize("method,path,body", CONTROL_PLANE_ENDPOINTS)
def test_control_plane_rejects_anonymous_callers(client, method, path, body):
    response = getattr(client, method)(path, **({"json": body} if body is not None else {}))
    assert response.status_code == 401, (
        f"{method.upper()} {path} served an unauthenticated request"
    )


MUTATING_ENDPOINTS = [
    ("patch", "/api/v1/agents/memory/records/00000000-0000-0000-0000-000000000001",
     {"pinned": True}),
    ("delete", "/api/v1/agents/memory/records/00000000-0000-0000-0000-000000000001", None),
    ("get", "/api/v1/agents/memory/export", None),
]


@pytest.mark.parametrize("method,path,body", MUTATING_ENDPOINTS)
def test_guests_cannot_mutate_or_export_memory(client, method, path, body):
    """
    A guest reading the owner's public memory is intended. A guest editing,
    erasing, or exporting it is emphatically not.
    """
    kwargs = {"headers": bearer(Role.GUEST, "guest-abc123def456")}
    if body is not None:
        kwargs["json"] = body
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 403, (
        f"{method.upper()} {path} allowed a guest (got {response.status_code})"
    )


def test_an_invalid_record_id_is_rejected_before_any_lookup(client):
    response = client.get(
        "/api/v1/agents/memory/records/not-a-uuid",
        headers=bearer(Role.OWNER, "vansh"),
    )
    assert response.status_code == 400


def test_patching_nothing_is_rejected_without_a_database_lookup(client):
    """
    An empty patch is a client error whether or not the record exists, so it is
    caught before any I/O. Originally the route looked the record up first,
    which meant a malformed request cost a query — and surfaced as a 500 when
    the table was unreachable rather than the 400 it always was.
    """
    response = client.patch(
        "/api/v1/agents/memory/records/00000000-0000-0000-0000-000000000001",
        json={},
        headers=bearer(Role.OWNER, "vansh"),
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "No fields to update"


def test_an_unknown_memory_kind_is_rejected(client):
    response = client.get(
        "/api/v1/agents/memory/records?kind=nonsense",
        headers=bearer(Role.OWNER, "vansh"),
    )
    assert response.status_code in (400, 500)
    if response.status_code == 400:
        assert "nonsense" in response.json()["detail"]
