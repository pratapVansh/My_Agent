"""JWT issuance, verification, rotation, and revocation (audit finding C1)."""
import time
from datetime import timedelta

import jwt
import pytest

from app.auth import jwt_service
from app.auth.jwt_service import (
    ACCESS_TOKEN_TYPE,
    REFRESH_TOKEN_TYPE,
    AuthError,
    decode_token,
    issue_token_pair,
    new_guest_user_id,
    principal_from_payload,
    revocation_store,
    revoke_token,
)
from app.auth.models import GUEST_SCOPES, OWNER_SCOPES, Role, Scope
from app.config import settings


@pytest.fixture(autouse=True)
def _configured_secrets(monkeypatch):
    monkeypatch.setattr(settings, "jwt_access_secret", "a" * 64)
    monkeypatch.setattr(settings, "jwt_refresh_secret", "b" * 64)
    revocation_store.clear()
    yield
    revocation_store.clear()


def test_owner_token_carries_full_scopes():
    pair = issue_token_pair("vansh", Role.OWNER)
    principal = principal_from_payload(decode_token(pair.access_token, ACCESS_TOKEN_TYPE))

    assert principal.user_id == "vansh"
    assert principal.is_owner
    assert principal.scopes == OWNER_SCOPES
    assert principal.has_scope(Scope.EMAIL_SEND)


def test_guest_token_is_restricted():
    pair = issue_token_pair(new_guest_user_id(), Role.GUEST)
    principal = principal_from_payload(decode_token(pair.access_token, ACCESS_TOKEN_TYPE))

    assert principal.is_guest
    assert principal.scopes == GUEST_SCOPES
    assert principal.has_scope(Scope.CHAT)
    assert principal.has_scope(Scope.PROFILE_READ)
    # The capabilities that make C2 critical must be absent.
    assert not principal.has_scope(Scope.EMAIL_SEND)
    assert not principal.has_scope(Scope.PROFILE_WRITE)
    assert not principal.has_scope(Scope.TOOLS_SCRAPE)
    assert not principal.has_scope(Scope.MEMORY_WRITE)
    assert not principal.has_scope(Scope.ATTENDANCE_READ)


def test_access_token_cannot_be_used_as_refresh_token():
    """Separate signing secrets make cross-use impossible."""
    pair = issue_token_pair("vansh", Role.OWNER)

    with pytest.raises(AuthError):
        decode_token(pair.access_token, REFRESH_TOKEN_TYPE)


def test_refresh_token_cannot_be_used_as_access_token():
    pair = issue_token_pair("vansh", Role.OWNER)

    with pytest.raises(AuthError):
        decode_token(pair.refresh_token, ACCESS_TOKEN_TYPE)


def test_token_signed_with_wrong_secret_is_rejected():
    forged = jwt.encode(
        {
            "sub": "vansh", "role": "owner", "typ": ACCESS_TOKEN_TYPE,
            "jti": "x", "iss": settings.jwt_issuer,
            "iat": int(time.time()), "exp": int(time.time()) + 600,
        },
        "attacker-secret",
        algorithm="HS256",
    )

    with pytest.raises(AuthError):
        decode_token(forged, ACCESS_TOKEN_TYPE)


def test_expired_token_is_rejected(monkeypatch):
    token, _, _ = jwt_service._encode(
        token_type=ACCESS_TOKEN_TYPE,
        user_id="vansh",
        role=Role.OWNER,
        scopes=OWNER_SCOPES,
        ttl=timedelta(seconds=-5),
    )

    with pytest.raises(AuthError, match="expired"):
        decode_token(token, ACCESS_TOKEN_TYPE)


def test_wrong_issuer_is_rejected():
    forged = jwt.encode(
        {
            "sub": "vansh", "role": "owner", "typ": ACCESS_TOKEN_TYPE,
            "jti": "x", "iss": "some-other-service",
            "iat": int(time.time()), "exp": int(time.time()) + 600,
        },
        settings.jwt_access_secret,
        algorithm="HS256",
    )

    with pytest.raises(AuthError):
        decode_token(forged, ACCESS_TOKEN_TYPE)


def test_revoked_refresh_token_is_rejected():
    """Rotation must make a redeemed refresh token single-use."""
    pair = issue_token_pair("vansh", Role.OWNER)
    assert decode_token(pair.refresh_token, REFRESH_TOKEN_TYPE)

    revoke_token(pair.refresh_token, REFRESH_TOKEN_TYPE)

    with pytest.raises(AuthError, match="revoked"):
        decode_token(pair.refresh_token, REFRESH_TOKEN_TYPE)


def test_scopes_are_derived_from_role_not_trusted_from_token():
    """A tampered scope claim must not grant capability."""
    escalated = jwt.encode(
        {
            "sub": "guest-abc", "role": "guest",
            "scopes": [Scope.EMAIL_SEND.value, Scope.PROFILE_WRITE.value],
            "typ": ACCESS_TOKEN_TYPE, "jti": "x", "iss": settings.jwt_issuer,
            "iat": int(time.time()), "exp": int(time.time()) + 600,
        },
        settings.jwt_access_secret,
        algorithm="HS256",
    )

    principal = principal_from_payload(decode_token(escalated, ACCESS_TOKEN_TYPE))

    assert principal.scopes == GUEST_SCOPES
    assert not principal.has_scope(Scope.EMAIL_SEND)


def test_unknown_role_is_rejected():
    token = jwt.encode(
        {
            "sub": "x", "role": "superadmin", "typ": ACCESS_TOKEN_TYPE,
            "jti": "x", "iss": settings.jwt_issuer,
            "iat": int(time.time()), "exp": int(time.time()) + 600,
        },
        settings.jwt_access_secret,
        algorithm="HS256",
    )

    with pytest.raises(AuthError):
        principal_from_payload(decode_token(token, ACCESS_TOKEN_TYPE))


def test_guest_ids_are_unique_and_well_formed():
    ids = {new_guest_user_id() for _ in range(50)}
    assert len(ids) == 50
    for guest_id in ids:
        assert guest_id.startswith("guest-")
        assert 3 <= len(guest_id) <= 128
