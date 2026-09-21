"""
The durable action system against a real PostgreSQL server.

Everything P0 rests on is SQL that had only ever been *compiled*. The claim
that two workers cannot both send the same email is a claim about
`DELETE … RETURNING`. The claim that a restart cannot re-send is a claim about
a partial unique index. The claim that an approved action replays exactly what
was previewed is a claim about JSONB round-tripping. A Python double that
mimics those semantics proves the tests agree with themselves; it does not
prove Postgres agrees with them.

So these run against a dedicated database, and they are opt-in because they are
destructive: `TRUNCATE` between cases, and `take_expired` issues a table-wide
`DELETE`. `tests/support/postgres.py` holds the guards that keep them aimed at
`my_agent_test` and nowhere else — including one that asks the server
`SELECT current_database()` after connecting, because every other check tests
intent and only that one tests reality.

Run them with:

    POSTGRES_INTEGRATION_TESTS=1 pytest -m postgres

Without the flag the database fixtures skip and the suite stays offline. The
safety-guard tests at the top of this file run either way — the logic that
prevents a destructive run from reaching the wrong database should never be
conditional on remembering to enable it.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace

import pytest
from sqlalchemy import text

from app.agents.actions import ActionGateway, compute_content_hash
from app.config import settings
from app.domain.audit import (
    AuditEntry,
    AuditStatus,
    AuditWriteError,
    PostgresAuditLog,
    digest,
)
from app.domain.pending_actions import (
    PostgresPendingActionStore,
    StoredAction,
)
from app.tools.contract import Effect
from tests.support import register_confirmable
from tests.support import postgres as pg_support

OWNER = "owner@example.com"
INTRUDER = "intruder@example.com"
CONVO = "conv-pg"


# ═══════════════════════════════════════════════════════════════════════════
# 0. The guards — these run with or without a database
# ═══════════════════════════════════════════════════════════════════════════

def test_the_application_database_is_refused_outright():
    """
    The single most important assertion in this file.

    These tests truncate tables and issue table-wide deletes. Aimed at the
    development database, one run would destroy real outstanding
    confirmations — so the app's own database name is rejected before a
    connection is ever attempted.
    """
    with pytest.raises(pg_support.UnsafeTestDatabase) as exc:
        pg_support.assert_safe_target(settings.postgres_db)
    assert "application's own database" in str(exc.value)


@pytest.mark.parametrize("name", ["my_agent", "production", "postgres", "trace", ""])
def test_databases_without_a_test_suffix_are_refused(name):
    """
    A typo must not be enough. The `_test` suffix is a second, independent
    barrier: even a name that is not the app database is refused unless it
    announces itself as disposable.
    """
    with pytest.raises(pg_support.UnsafeTestDatabase):
        pg_support.assert_safe_target(name)


def test_the_dedicated_database_is_accepted():
    pg_support.assert_safe_target("my_agent_test")


def test_the_test_database_never_defaults_to_the_application_database(monkeypatch):
    """
    There is no code path from "unconfigured" to "the real database". With the
    environment variable unset the answer is the hardcoded test name, never
    `settings.postgres_db`.
    """
    monkeypatch.delenv(pg_support.DB_ENV, raising=False)
    resolved = pg_support.test_database_name()
    assert resolved == pg_support.DEFAULT_TEST_DB
    assert resolved != settings.postgres_db


def test_integration_tests_are_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv(pg_support.ENABLE_FLAG, raising=False)
    assert pg_support.integration_enabled() is False
    monkeypatch.setenv(pg_support.ENABLE_FLAG, "1")
    assert pg_support.integration_enabled() is True


def test_only_the_two_owned_tables_are_ever_in_scope():
    """Truncation is by explicit name, so nothing else can be caught by it."""
    assert set(pg_support.OWNED_TABLES) == {"pending_action", "action_audit"}


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures — everything below needs the database
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
async def db():
    """A pool onto the dedicated test database, with the schema applied."""
    if not pg_support.integration_enabled():
        pytest.skip(
            f"set {pg_support.ENABLE_FLAG}=1 to run PostgreSQL integration tests"
        )
    database = await pg_support.connect()
    await database.create_schema(pg_support.migration_statements())
    await database.truncate()
    try:
        yield database
    finally:
        await database.truncate()
        await database.dispose()


@pytest.fixture
async def stores(db):
    """The real Postgres implementations, over the test pool."""
    return {
        "audit": PostgresAuditLog(session_maker=db.session_maker),
        "pending": PostgresPendingActionStore(session_maker=db.session_maker),
    }


@pytest.fixture
def gateway(stores):
    return ActionGateway(audit=stores["audit"], pending=stores["pending"])


class _SendRecorder:
    def __init__(self):
        self.sent: list[dict] = []

    async def __call__(self, args):
        self.sent.append(dict(args))
        return {"success": True, "message_id": f"m{len(self.sent)}"}

    @property
    def count(self) -> int:
        return len(self.sent)


def _spec(recorder, effect=Effect.EXTERNAL_WRITE, name="send_email"):
    preview = "To: alice@example.com\nSubject: Hi\n\nHello."
    register_confirmable(name, recorder, effect=effect, preview=preview)
    return {"callable": recorder, "effect": effect, "preview": preview}


async def hold(gateway, recorder, *, owner=OWNER, convo=CONVO, args=None):
    return await gateway.intercept(
        tool="send_email",
        spec=_spec(recorder),
        arguments=args or {"to_email": "alice@example.com", "subject": "Hi",
                           "body": "Hello."},
        owner_id=owner,
        conversation_id=convo,
    )


def _stored(**overrides) -> StoredAction:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    base = dict(
        id=str(uuid.uuid4()),
        token_hash=digest(uuid.uuid4().hex),
        audit_id=str(uuid.uuid4()),
        owner_id=OWNER,
        conversation_id=CONVO,
        tool="send_email",
        effect=Effect.EXTERNAL_WRITE.value,
        arguments={"to_email": "alice@example.com", "subject": "Hi", "body": "Hello."},
        preview="To: alice@example.com",
        content_hash="c" * 64,
        idempotency_key=uuid.uuid4().hex,
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    base.update(overrides)
    return StoredAction(**base)


pytestmark = pytest.mark.postgres


# ═══════════════════════════════════════════════════════════════════════════
# 1. The migration really applies
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_migration_creates_both_tables(db):
    async with db.engine.connect() as conn:
        tables = {
            r[0] for r in (await conn.execute(text(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ))).all()
        }
    assert {"pending_action", "action_audit"} <= tables


async def test_the_partial_unique_index_exists(db):
    """
    The durable idempotency guarantee is this index and nothing else. If the
    migration ever stops creating it, the guarantee silently becomes advisory.
    """
    async with db.engine.connect() as conn:
        row = await conn.scalar(text(
            "SELECT indexdef FROM pg_indexes "
            "WHERE indexname = 'uq_action_audit_executed_idempotency'"
        ))
    assert row is not None, "the partial unique index was not created"
    assert "UNIQUE" in row.upper()
    assert "executed" in row


async def test_the_migration_is_idempotent(db):
    """Applying it twice must not fail — deployments re-run migrations."""
    await db.create_schema(pg_support.migration_statements())
    await db.create_schema(pg_support.migration_statements())


# ═══════════════════════════════════════════════════════════════════════════
# 2. The real store round-trips
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_stored_action_round_trips_through_postgres(db, stores):
    action = _stored()
    await stores["pending"].put(action)

    loaded = await stores["pending"].peek(action.token_hash)

    assert loaded is not None
    assert loaded.tool == action.tool
    assert loaded.owner_id == OWNER
    # JSONB must return the arguments intact — this is what gets replayed.
    assert loaded.arguments == action.arguments
    assert loaded.preview == action.preview
    assert loaded.content_hash == action.content_hash
    assert loaded.idempotency_key == action.idempotency_key


async def test_claiming_removes_the_row(db, stores):
    action = _stored()
    await stores["pending"].put(action)

    claimed = await stores["pending"].claim(action.token_hash)
    assert claimed is not None
    assert await stores["pending"].peek(action.token_hash) is None
    assert await db.count("pending_action") == 0


async def test_listing_is_scoped_to_owner_and_conversation(db, stores):
    await stores["pending"].put(_stored(owner_id=OWNER, conversation_id=CONVO))
    await stores["pending"].put(_stored(owner_id=INTRUDER, conversation_id=CONVO))
    await stores["pending"].put(_stored(owner_id=OWNER, conversation_id="other"))

    mine = await stores["pending"].list_for(CONVO, OWNER)

    assert len(mine) == 1
    assert mine[0].owner_id == OWNER


async def test_expired_rows_are_not_listed(db, stores):
    from datetime import datetime, timedelta, timezone

    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    await stores["pending"].put(_stored(expires_at=past))

    assert await stores["pending"].list_for(CONVO, OWNER) == []
    assert await db.count("pending_action") == 1, "the row exists, it is just not offered"


async def test_take_expired_removes_only_lapsed_rows(db, stores):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    await stores["pending"].put(_stored(expires_at=now - timedelta(seconds=1)))
    live = _stored(expires_at=now + timedelta(minutes=5))
    await stores["pending"].put(live)

    swept = await stores["pending"].take_expired()

    assert len(swept) == 1
    assert await stores["pending"].peek(live.token_hash) is not None


# ═══════════════════════════════════════════════════════════════════════════
# 3. The audit log against real SQL
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_audit_lifecycle_persists(db, stores):
    entry = AuditEntry(
        audit_id=str(uuid.uuid4()), owner_id=OWNER, conversation_id=CONVO,
        effect=Effect.EXTERNAL_WRITE.value, tool="send_email",
        arguments_hash="a" * 64, preview_hash="p" * 64, token_hash="t" * 64,
        idempotency_key=uuid.uuid4().hex,
    )
    await stores["audit"].record_pending(entry)

    loaded = await stores["audit"].get(entry.audit_id)
    assert loaded.status is AuditStatus.PENDING
    assert loaded.confirmed_at is None

    await stores["audit"].mark(entry.audit_id, AuditStatus.CONFIRMED)
    assert (await stores["audit"].get(entry.audit_id)).confirmed_at is not None

    await stores["audit"].mark(entry.audit_id, AuditStatus.EXECUTED, outcome="ok")
    final = await stores["audit"].get(entry.audit_id)
    assert final.status is AuditStatus.EXECUTED
    assert final.executed_at is not None


async def test_the_database_refuses_two_executed_rows_for_one_key(db, stores):
    """
    The partial unique index, exercised rather than assumed. This is the
    constraint that makes "already sent" a fact rather than a hope.
    """
    key = uuid.uuid4().hex
    first = AuditEntry(
        audit_id=str(uuid.uuid4()), owner_id=OWNER, conversation_id=CONVO,
        effect=Effect.EXTERNAL_WRITE.value, tool="send_email",
        arguments_hash="a" * 64, preview_hash="p" * 64, token_hash="t1" + "0" * 62,
        idempotency_key=key,
    )
    second = replace(first, audit_id=str(uuid.uuid4()), token_hash="t2" + "0" * 62)

    await stores["audit"].record_pending(first)
    await stores["audit"].record_pending(second)
    await stores["audit"].mark(first.audit_id, AuditStatus.EXECUTED)

    with pytest.raises(AuditWriteError):
        await stores["audit"].mark(second.audit_id, AuditStatus.EXECUTED)

    assert await stores["audit"].was_executed(key) is True


async def test_a_failed_row_does_not_hold_the_idempotency_slot(db, stores):
    """
    The deliberate asymmetry: an explicit failure releases the claim, so a
    demonstrably undelivered send stays retryable.
    """
    key = uuid.uuid4().hex
    entry = AuditEntry(
        audit_id=str(uuid.uuid4()), owner_id=OWNER, conversation_id=CONVO,
        effect=Effect.EXTERNAL_WRITE.value, tool="send_email",
        arguments_hash="a" * 64, preview_hash="p" * 64, token_hash="t" * 64,
        idempotency_key=key,
    )
    await stores["audit"].record_pending(entry)
    await stores["audit"].mark(entry.audit_id, AuditStatus.FAILED, error="smtp")

    assert await stores["audit"].was_executed(key) is False


# ═══════════════════════════════════════════════════════════════════════════
# 4. Surviving a restart — separate pools, shared database
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_pending_action_survives_a_repository_restart(db):
    """
    Two independent pools share nothing but the database, which is the closest
    a single process can get to "the old worker is gone".
    """
    recorder = _SendRecorder()

    before = ActionGateway(
        audit=PostgresAuditLog(session_maker=db.session_maker),
        pending=PostgresPendingActionStore(session_maker=db.session_maker),
    )
    held = await hold(before, recorder)
    assert recorder.count == 0

    # ── a genuinely separate connection pool ─────────────────────────────
    other = await pg_support.connect()
    try:
        after = ActionGateway(
            audit=PostgresAuditLog(session_maker=other.session_maker),
            pending=PostgresPendingActionStore(session_maker=other.session_maker),
        )
        outstanding = await after.pending_for(CONVO, OWNER)
        assert len(outstanding) == 1
        assert "alice@example.com" in outstanding[0].preview
        assert outstanding[0].token == "", "the raw token must not be recoverable"
        assert outstanding[0].handle == digest(held.data["confirmation_token"])

        result = await after.confirm_and_execute(
            handle=outstanding[0].handle, owner_id=OWNER
        )
        assert result.ok is True
        assert recorder.count == 1
    finally:
        await other.dispose()


async def test_a_completed_action_cannot_be_repeated_after_a_restart(db):
    """The T8 acceptance criterion, now against real SQL."""
    recorder = _SendRecorder()
    args = {"to_email": "alice@example.com", "subject": "Hi", "body": "Hello."}

    before = ActionGateway(
        audit=PostgresAuditLog(session_maker=db.session_maker),
        pending=PostgresPendingActionStore(session_maker=db.session_maker),
    )
    held = await hold(before, recorder, args=args)
    await before.confirm_and_execute(held.data["confirmation_token"], owner_id=OWNER)
    assert recorder.count == 1

    other = await pg_support.connect()
    try:
        after = ActionGateway(
            audit=PostgresAuditLog(session_maker=other.session_maker),
            pending=PostgresPendingActionStore(session_maker=other.session_maker),
        )
        repeat = await hold(after, recorder, args=args)
        assert repeat.is_pending is False
        assert repeat.data.get("already_executed") is True
        assert recorder.count == 1
    finally:
        await other.dispose()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Concurrency — the claim that only SQL can settle
# ═══════════════════════════════════════════════════════════════════════════

async def test_concurrent_claims_from_separate_pools_yield_exactly_one_winner(db, stores):
    """
    `DELETE … RETURNING` at the store level, across independent connections.

    This is the mechanism the cross-worker guarantee rests on, and until now it
    had only ever been asserted against a Python dict holding a lock.
    """
    action = _stored()
    await stores["pending"].put(action)

    workers = [await pg_support.connect(pool_size=2) for _ in range(6)]
    try:
        claims = await asyncio.gather(*[
            PostgresPendingActionStore(session_maker=w.session_maker).claim(
                action.token_hash
            )
            for w in workers
        ])
    finally:
        for w in workers:
            await w.dispose()

    winners = [c for c in claims if c is not None]
    assert len(winners) == 1, f"{len(winners)} workers claimed the same action"
    assert await db.count("pending_action") == 0


async def test_concurrent_confirmations_send_exactly_one_email(db):
    """
    The whole stack under contention: six gateways, six pools, one action.

    Exactly one email, exactly one executed audit row — the property a user
    would notice being wrong.
    """
    recorder = _SendRecorder()

    prep = ActionGateway(
        audit=PostgresAuditLog(session_maker=db.session_maker),
        pending=PostgresPendingActionStore(session_maker=db.session_maker),
    )
    held = await hold(prep, recorder)
    handle = digest(held.data["confirmation_token"])

    workers = [await pg_support.connect(pool_size=2) for _ in range(6)]
    try:
        gateways = [
            ActionGateway(
                audit=PostgresAuditLog(session_maker=w.session_maker),
                pending=PostgresPendingActionStore(session_maker=w.session_maker),
            )
            for w in workers
        ]
        results = await asyncio.gather(*[
            g.confirm_and_execute(handle=handle, owner_id=OWNER) for g in gateways
        ], return_exceptions=True)
    finally:
        for w in workers:
            await w.dispose()

    successes = [r for r in results if getattr(r, "ok", False)]
    assert len(successes) == 1, f"{len(successes)} confirmations succeeded"
    assert recorder.count == 1, f"{recorder.count} emails were sent"

    async with db.engine.connect() as conn:
        executed = await conn.scalar(text(
            "SELECT count(*) FROM action_audit WHERE status = 'executed'"
        ))
    assert executed == 1


async def test_concurrent_cancellation_is_atomic(db, stores):
    """Only one cancel may win, for the same reason only one claim may."""
    recorder = _SendRecorder()
    gateway = ActionGateway(audit=stores["audit"], pending=stores["pending"])
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    workers = [await pg_support.connect(pool_size=2) for _ in range(4)]
    try:
        gateways = [
            ActionGateway(
                audit=PostgresAuditLog(session_maker=w.session_maker),
                pending=PostgresPendingActionStore(session_maker=w.session_maker),
            )
            for w in workers
        ]
        outcomes = await asyncio.gather(*[
            g.cancel(handle=handle, owner_id=OWNER) for g in gateways
        ])
    finally:
        for w in workers:
            await w.dispose()

    assert sum(1 for o in outcomes if o) == 1
    assert recorder.count == 0
    assert await db.count("pending_action") == 0


async def test_a_claim_racing_a_cancel_never_both_succeed(db, stores):
    recorder = _SendRecorder()
    gateway = ActionGateway(audit=stores["audit"], pending=stores["pending"])
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    a = await pg_support.connect(pool_size=2)
    b = await pg_support.connect(pool_size=2)
    try:
        confirming = ActionGateway(
            audit=PostgresAuditLog(session_maker=a.session_maker),
            pending=PostgresPendingActionStore(session_maker=a.session_maker),
        )
        cancelling = ActionGateway(
            audit=PostgresAuditLog(session_maker=b.session_maker),
            pending=PostgresPendingActionStore(session_maker=b.session_maker),
        )
        confirmed, cancelled = await asyncio.gather(
            confirming.confirm_and_execute(handle=handle, owner_id=OWNER),
            cancelling.cancel(handle=handle, owner_id=OWNER),
        )
    finally:
        await a.dispose()
        await b.dispose()

    assert not (confirmed.ok and cancelled), "the action was both sent and cancelled"
    assert recorder.count == (1 if confirmed.ok else 0)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Security properties, on real SQL
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_wrong_owner_cannot_consume_the_action(db, stores):
    recorder = _SendRecorder()
    gateway = ActionGateway(audit=stores["audit"], pending=stores["pending"])
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    refused = await gateway.confirm_and_execute(handle=handle, owner_id=INTRUDER)

    assert refused.is_error is True
    assert recorder.count == 0
    # Critically the row survives — a hostile probe must not destroy it.
    assert await db.count("pending_action") == 1
    ok = await gateway.confirm_and_execute(handle=handle, owner_id=OWNER)
    assert ok.ok is True


async def test_an_expired_action_cannot_execute(db, stores):
    recorder = _SendRecorder()
    gateway = ActionGateway(
        ttl_seconds=0.05, audit=stores["audit"], pending=stores["pending"]
    )
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    await asyncio.sleep(0.2)
    refused = await gateway.confirm_and_execute(handle=handle, owner_id=OWNER)

    assert refused.is_error is True
    assert recorder.count == 0
    assert await db.count("pending_action") == 0


async def test_tampered_arguments_are_rejected(db, stores):
    """
    The row is on disk, so its arguments are reachable in a way they never were
    in memory. The hash is recomputed from the row's own contents, so an edit
    that leaves the stored hash alone is caught.
    """
    recorder = _SendRecorder()
    gateway = ActionGateway(audit=stores["audit"], pending=stores["pending"])
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    async with db.engine.begin() as conn:
        await conn.execute(text(
            "UPDATE pending_action "
            "SET arguments = jsonb_set(arguments, '{to_email}', '\"attacker@evil.com\"') "
            "WHERE token_hash = :h"
        ), {"h": handle})

    refused = await gateway.confirm_and_execute(handle=handle, owner_id=OWNER)

    assert refused.is_error is True
    assert recorder.count == 0
    assert await db.count("pending_action") == 0, "a tampered row must be destroyed"


async def test_a_tampered_content_hash_is_rejected(db, stores):
    recorder = _SendRecorder()
    gateway = ActionGateway(audit=stores["audit"], pending=stores["pending"])
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    refused = await gateway.confirm_and_execute(
        handle=handle, owner_id=OWNER, content_hash="0" * 64
    )

    assert refused.is_error is True
    assert recorder.count == 0
    assert await db.count("pending_action") == 1


async def test_a_row_naming_an_unknown_tool_cannot_execute(db, stores):
    recorder = _SendRecorder()
    gateway = ActionGateway(audit=stores["audit"], pending=stores["pending"])
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    async with db.engine.begin() as conn:
        await conn.execute(text(
            "UPDATE pending_action SET tool = 'rm_minus_rf' WHERE token_hash = :h"
        ), {"h": handle})

    refused = await gateway.confirm_and_execute(handle=handle, owner_id=OWNER)

    assert refused.is_error is True
    assert recorder.count == 0


async def test_a_replayed_confirmation_is_refused(db, stores):
    recorder = _SendRecorder()
    gateway = ActionGateway(audit=stores["audit"], pending=stores["pending"])
    held = await hold(gateway, recorder)
    token = held.data["confirmation_token"]

    first = await gateway.confirm_and_execute(token, owner_id=OWNER)
    second = await gateway.confirm_and_execute(token, owner_id=OWNER)

    assert first.ok is True
    assert second.is_error is True
    assert recorder.count == 1


async def test_idempotency_prevents_a_duplicate_after_completion(db, stores):
    recorder = _SendRecorder()
    gateway = ActionGateway(audit=stores["audit"], pending=stores["pending"])
    args = {"to_email": "alice@example.com", "subject": "Hi", "body": "Hello."}

    held = await hold(gateway, recorder, args=args)
    await gateway.confirm_and_execute(held.data["confirmation_token"], owner_id=OWNER)
    assert recorder.count == 1

    repeat = await hold(gateway, recorder, args=args)

    assert repeat.is_pending is False
    assert repeat.data.get("already_executed") is True
    assert recorder.count == 1


async def test_the_raw_token_is_never_written_to_the_database(db, stores):
    """Read straight out of both tables — the secret must appear in neither."""
    recorder = _SendRecorder()
    gateway = ActionGateway(audit=stores["audit"], pending=stores["pending"])
    held = await hold(gateway, recorder)
    token = held.data["confirmation_token"]

    async with db.engine.connect() as conn:
        pending_rows = (await conn.execute(text(
            "SELECT token_hash::text, arguments::text, preview FROM pending_action"
        ))).all()
        audit_rows = (await conn.execute(text(
            "SELECT token_hash, arguments_hash, preview_hash FROM action_audit"
        ))).all()

    blob = str(pending_rows) + str(audit_rows)
    assert token not in blob
    assert digest(token) in blob


# ═══════════════════════════════════════════════════════════════════════════
# 7. Cleanup discipline
# ═══════════════════════════════════════════════════════════════════════════

async def test_truncation_empties_only_the_owned_tables(db, stores):
    await stores["pending"].put(_stored())
    assert await db.count("pending_action") == 1

    await db.truncate()

    assert await db.count("pending_action") == 0
    assert await db.count("action_audit") == 0


async def test_counting_an_unowned_table_is_refused(db):
    with pytest.raises(pg_support.UnsafeTestDatabase):
        await db.count("chat_history")
