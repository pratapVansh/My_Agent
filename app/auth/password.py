"""
Password hashing and verification (bcrypt).

The owner's password exists only as a bcrypt hash in configuration; the
plaintext is never stored, logged, or transmitted anywhere but the login body.
"""
from __future__ import annotations

import logging
import secrets

import bcrypt

logger = logging.getLogger(__name__)

# bcrypt truncates silently at 72 bytes; reject longer input rather than let a
# user believe a 200-character passphrase is fully protecting the account.
_MAX_PASSWORD_BYTES = 72

# Verified against when no owner is configured, so a login attempt costs the
# same time whether or not the account exists (no user-enumeration signal).
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password-for-timing", bcrypt.gensalt(rounds=12))


def hash_password(password: str, rounds: int = 12) -> str:
    """Return a bcrypt hash suitable for OWNER_PASSWORD_HASH."""
    encoded = password.encode("utf-8")
    if len(encoded) > _MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password must be at most {_MAX_PASSWORD_BYTES} bytes "
            "(bcrypt truncates beyond that)."
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=rounds)).decode("utf-8")


def verify_password(password: str, password_hash: str | None) -> bool:
    """
    Check a password against its hash in constant time.

    Always performs a real bcrypt comparison — including when no hash is
    configured — so response timing cannot reveal whether an account exists.
    """
    encoded = password.encode("utf-8")[:_MAX_PASSWORD_BYTES]

    if not (password_hash or "").strip():
        bcrypt.checkpw(encoded, _DUMMY_HASH)
        return False

    try:
        return bcrypt.checkpw(encoded, password_hash.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        # Malformed hash in configuration — fail closed and make it visible.
        logger.error("OWNER_PASSWORD_HASH is not a valid bcrypt hash: %s", exc)
        bcrypt.checkpw(encoded, _DUMMY_HASH)
        return False


def constant_time_equals(left: str, right: str) -> bool:
    """Timing-safe string comparison for tokens and usernames."""
    return secrets.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
