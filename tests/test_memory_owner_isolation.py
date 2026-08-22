"""
Can a guest read the owner's private memory? Proven, not assumed.

The previous pass added a retrieval-scope component to the memory cache key and
recorded an open question alongside it: the v1 retrieval path still queries by
`user_id` rather than by `memory_owner_id`, and that was flagged rather than
resolved. This file resolves it.

The finding is that the v1 path is safe, and safe for a reason worth stating
precisely, because the reason is the opposite of what "the scope is not applied"
sounds like:

    `memory_owner_id` is *never used to widen a read*. Every v1 retrieval leg
    queries the caller's own `user_id`, and `user_id` comes from the verified
    token. A guest therefore reads the guest's own partition — which is empty,
    and which guests hold no scope to write to.

So the gap is over-restriction, not leakage: the recruiter view cannot see the
owner's public résumé (the thing `RetrievalScope` was built to enable), and no
guest can see anything of the owner's at all. That is the safe direction to be
wrong in, and it is why no fix is applied here — a fix would *widen* access,
which is a feature change, not a hardening.

These tests pin the property so that a later change implementing owner-scoped
v1 reads cannot land without deliberately confronting them.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.auth.jwt_service import issue_token_pair, revocation_store
from app.auth.models import Principal, Role
from app.auth.password import hash_password
from app.config import settings
from app.main import app
from app.memory.kinds import Visibility
from app.memory.memory_cache import memory_cache
from app.memory.memory_manager import MemoryManager
from app.memory.scope import resolve_retrieval_scope
from app.memory.sources import QueryCategory

OWNER_ID = "vansh"
OWNER_PASSWORD = "correct-horse-battery-staple"


@pytest.fixture(autouse=True)
def _auth_config(monkeypatch):
    monkeypatch.setattr(settings, "jwt_access_secret", "a" * 64)
    monkeypatch.setattr(settings, "jwt_refresh_secret", "b" * 64)
    monkeypatch.setattr(settings, "owner_username", OWNER_ID)
    monkeypatch.setattr(settings, "owner_password_hash", hash_password(OWNER_PASSWORD, rounds=4))
    monkeypatch.setattr(settings, "owner_user_id", OWNER_ID)
    monkeypatch.setattr(settings, "guest_sessions_enabled", True)
    monkeypatch.setattr(settings, "csrf_protection_enabled", False)
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    revocation_store.clear()
    memory_cache.clear()
    yield
    revocation_store.clear()
    memory_cache.clear()


# ── The scope resolver ───────────────────────────────────────────────────

def test_a_guest_is_scoped_to_the_owners_public_records_only():
    guest = Principal(user_id="guest-abc123", role=Role.GUEST, scopes=frozenset())
    scope = resolve_retrieval_scope(guest)

    assert scope.owner_id == OWNER_ID
    assert scope.visibilities == [Visibility.PUBLIC]
    assert scope.is_own_memory is False
    assert scope.public_only is True


def test_the_owner_reads_their_own_memory_at_every_visibility():
    owner = Principal(user_id=OWNER_ID, role=Role.OWNER, scopes=frozenset())
    scope = resolve_retrieval_scope(owner)

    assert scope.owner_id == OWNER_ID
    assert scope.visibilities is None
    assert scope.is_own_memory is True


# ── What the v1 legs actually query ──────────────────────────────────────

class _SpyManager(MemoryManager):
    """Records the user_id each retrieval leg is asked for."""

    def __init__(self):
        super().__init__()
        self.queried_ids: list[str] = []
        outer = self

        class _ShortTerm:
            async def get_recent_context(self, user_id, **kw):
                outer.queried_ids.append(user_id)
                return []

            async def get_profile_facts(self, user_id, **kw):
                outer.queried_ids.append(user_id)
                return [{"key": "cgpa", "value": "8.9"}]

            async def get_recent_episodes(self, user_id, **kw):
                outer.queried_ids.append(user_id)
                return []

        class _Smart:
            async def retrieve_preferences(self, user_id, **kw):
                outer.queried_ids.append(user_id)
                return []

        class _LongTerm:
            async def search_all(self, user_id, query, limit=5, sections=None):
                outer.queried_ids.append(user_id)
                return {"resume": {}, "skills": [], "projects": []}

        self.short_term = _ShortTerm()
        self.smart = _Smart()
        self.long_term = _LongTerm()


async def test_v1_retrieval_never_widens_to_the_owner_partition():
    """
    The property that makes the v1 path safe.

    Even handed `memory_owner_id="vansh"` and the owner's visibilities, every
    leg is queried for the *caller's* id. `memory_owner_id` is used for the
    cache key and for nothing that reads data.
    """
    manager = _SpyManager()

    await manager.retrieve_context(
        user_id="guest-abc123",
        session_id="iso-1",
        query="what is my CGPA",
        category=QueryCategory.PROFILE_EDUCATION.value,
        memory_owner_id=OWNER_ID,
        visibilities=[Visibility.PUBLIC],
    )

    assert manager.queried_ids, "no retrieval leg ran"
    assert set(manager.queried_ids) == {"guest-abc123"}, (
        f"a leg read a partition other than the caller's: {set(manager.queried_ids)}"
    )
    assert OWNER_ID not in manager.queried_ids


async def test_the_owner_reads_their_own_partition():
    manager = _SpyManager()

    await manager.retrieve_context(
        user_id=OWNER_ID,
        session_id="iso-2",
        query="what is my CGPA",
        category=QueryCategory.PROFILE_EDUCATION.value,
        memory_owner_id=OWNER_ID,
        visibilities=None,
    )

    assert set(manager.queried_ids) == {OWNER_ID}


# ── The cache cannot bridge the two ──────────────────────────────────────

async def test_an_owner_cache_entry_is_not_served_to_a_guest():
    """
    Two independent barriers, and the test asserts the outcome rather than
    either mechanism: the key carries the caller's id *and* the retrieval
    scope, so an owner's entry is unreachable from a guest request even if a
    future change makes the two share a user_id.
    """
    owner_manager = _SpyManager()
    await owner_manager.retrieve_context(
        user_id=OWNER_ID, session_id="iso-3", query="what is my CGPA",
        category=QueryCategory.PROFILE_EDUCATION.value,
        memory_owner_id=OWNER_ID, visibilities=None,
    )

    guest_manager = _SpyManager()
    guest_view = await guest_manager.retrieve_context(
        user_id="guest-abc123", session_id="iso-4", query="what is my CGPA",
        category=QueryCategory.PROFILE_EDUCATION.value,
        memory_owner_id=OWNER_ID, visibilities=[Visibility.PUBLIC],
    )

    # The guest's own legs ran — it did not read a cached owner result.
    assert guest_manager.queried_ids, "the guest was served from cache"
    assert set(guest_manager.queried_ids) == {"guest-abc123"}
    assert guest_view["profile_facts"] == [{"key": "cgpa", "value": "8.9"}]


def test_the_scope_key_separates_owner_from_guest():
    owner_key = MemoryManager._retrieval_scope_key(OWNER_ID, None, "PROFILE_EDUCATION")
    guest_key = MemoryManager._retrieval_scope_key(
        OWNER_ID, [Visibility.PUBLIC], "PROFILE_EDUCATION"
    )
    assert owner_key != guest_key


def test_the_scope_key_renders_visibility_enums_stably():
    """
    Order and representation must not vary, or the same scope produces two keys
    and the cache silently stops working.
    """
    a = MemoryManager._retrieval_scope_key(
        OWNER_ID, [Visibility.PUBLIC, Visibility.PRIVATE], "X"
    )
    b = MemoryManager._retrieval_scope_key(
        OWNER_ID, [Visibility.PRIVATE, Visibility.PUBLIC], "X"
    )
    assert a == b
    assert "public" in a and "private" in a


# ── The identity a request runs under cannot be claimed ──────────────────

def _bearer(role: Role, user_id: str) -> dict:
    pair = issue_token_pair(user_id, role)
    return {"Authorization": f"Bearer {pair.access_token}"}


def test_a_guest_cannot_claim_the_owners_identity_in_the_body():
    """
    The load-bearing check behind everything above.

    Every v1 leg reads `user_id`, and `user_id` comes from `resolve_user_id`.
    If a caller could put someone else's id in the request body, the isolation
    proven above would be bypassed in one line without touching the memory
    layer at all.
    """
    client = TestClient(app)

    response = client.post(
        "/api/v1/agents/query",
        headers=_bearer(Role.GUEST, "guest-abc123"),
        json={"query": "what is my CGPA", "user_id": OWNER_ID, "session_id": "iso-5"},
    )

    assert response.status_code == 403


def test_an_anonymous_caller_reaches_no_memory_at_all():
    client = TestClient(app)

    response = client.post(
        "/api/v1/agents/query",
        json={"query": "what is my CGPA", "session_id": "iso-6"},
    )

    assert response.status_code in (401, 403)
