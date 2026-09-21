"""
Identity-resolution tests (audit finding C1).

`user_id` is no longer validated as a client-supplied string — it is *derived*
from the verified token. These tests pin that behaviour: a claim that conflicts
with the authenticated identity must be refused, never silently ignored.
"""
import pytest
from fastapi import HTTPException

from app.auth.dependencies import resolve_user_id
from app.auth.models import GUEST_SCOPES, OWNER_SCOPES, Principal, Role
from app.config import settings


def owner(user_id: str = "vansh") -> Principal:
    return Principal(user_id=user_id, role=Role.OWNER, scopes=OWNER_SCOPES)


def guest(user_id: str = "guest-abc123") -> Principal:
    return Principal(user_id=user_id, role=Role.GUEST, scopes=GUEST_SCOPES)


def test_identity_comes_from_the_token_when_nothing_is_claimed():
    assert resolve_user_id(owner(), None) == "vansh"


def test_matching_claim_is_accepted():
    assert resolve_user_id(owner("vansh"), "vansh") == "vansh"


def test_claim_is_case_and_whitespace_insensitive():
    assert resolve_user_id(owner("vansh"), "  VANSH  ") == "vansh"


def test_empty_claim_falls_back_to_the_token():
    assert resolve_user_id(owner("vansh"), "") == "vansh"


@pytest.mark.parametrize("claimed", [
    "someone-else",
    "vansh2",
    "../../etc/passwd",
    "admin",
])
def test_conflicting_claim_is_refused(claimed):
    with pytest.raises(HTTPException) as exc:
        resolve_user_id(owner("vansh"), claimed)
    assert exc.value.status_code == 403


def test_guest_cannot_claim_the_owner_identity():
    """The exact escalation C1 described: assert someone else's id."""
    with pytest.raises(HTTPException) as exc:
        resolve_user_id(guest(), "vansh")
    assert exc.value.status_code == 403


def test_guest_keeps_its_own_namespaced_identity():
    assert resolve_user_id(guest("guest-abc123"), None) == "guest-abc123"


# ── Operator-supplied owner id is validated at startup ───────────────────

@pytest.mark.parametrize("bad_id", [
    "",
    "ab",                     # too short
    "../../etc/passwd",
    "user with spaces",
    "UPPERCASE",
    "trailing-",
    "x" * 200,
])
def test_invalid_owner_user_id_is_rejected_at_startup(monkeypatch, bad_id):
    monkeypatch.setattr(settings, "jwt_access_secret", "a" * 64)
    monkeypatch.setattr(settings, "jwt_refresh_secret", "b" * 64)
    monkeypatch.setattr(settings, "owner_username", "vansh")
    monkeypatch.setattr(settings, "owner_password_hash", "$2b$12$fake")
    monkeypatch.setattr(settings, "owner_user_id", bad_id)

    problems = settings.validate_auth_config()
    assert any("OWNER_USER_ID" in p for p in problems)


def test_valid_owner_user_id_passes_startup_validation(monkeypatch):
    monkeypatch.setattr(settings, "jwt_access_secret", "a" * 64)
    monkeypatch.setattr(settings, "jwt_refresh_secret", "b" * 64)
    monkeypatch.setattr(settings, "owner_username", "vansh")
    monkeypatch.setattr(settings, "owner_password_hash", "$2b$12$fake")
    monkeypatch.setattr(settings, "owner_user_id", "vansh")
    monkeypatch.setattr(settings, "auth_cookie_secure", True)

    assert settings.validate_auth_config() == []


def test_shared_signing_secret_is_rejected(monkeypatch):
    """Reusing one secret would let an access token be replayed as a refresh."""
    monkeypatch.setattr(settings, "jwt_access_secret", "s" * 64)
    monkeypatch.setattr(settings, "jwt_refresh_secret", "s" * 64)
    monkeypatch.setattr(settings, "owner_username", "vansh")
    monkeypatch.setattr(settings, "owner_password_hash", "$2b$12$fake")
    monkeypatch.setattr(settings, "owner_user_id", "vansh")

    problems = settings.validate_auth_config()
    assert any("must be different" in p for p in problems)


def test_short_signing_secret_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "jwt_access_secret", "short")
    monkeypatch.setattr(settings, "jwt_refresh_secret", "b" * 64)
    monkeypatch.setattr(settings, "owner_username", "vansh")
    monkeypatch.setattr(settings, "owner_password_hash", "$2b$12$fake")
    monkeypatch.setattr(settings, "owner_user_id", "vansh")

    problems = settings.validate_auth_config()
    assert any("too short" in p for p in problems)
