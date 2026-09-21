"""
The durable record, and the guarantees that depend on it existing.

Two properties here are the reason the audit log was built, and both are
asserted against behaviour rather than against the presence of a row:

  * **A restart cannot re-send.** The gateway's process memory is gone; the
    record is not. A fresh gateway sharing the same store must refuse an action
    the old one completed. That is the acceptance criterion, and
    `test_a_restart_cannot_repeat_a_completed_action` is where it lives.

  * **No record, no effect.** If the log cannot be written the action does not
    run — not "runs and logs a warning". An unauditable irreversible action is
    treated exactly like a forbidden one, at every point in the lifecycle where
    the write could fail.

The in-memory store used throughout enforces the same unique-executed
constraint as the Postgres partial index, so a test that would violate it fails
here rather than in production.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agents.actions import ActionGateway, action_gateway, compute_content_hash
from app.agents.confirmation import resolve
from app.domain.audit import (
    AuditStatus,
    AuditWriteError,
    InMemoryAuditLog,
    digest,
)
from app.tools.contract import Effect
from tests.support import (
    drive, final, register_confirmable, state, stub_services, tool_call,
)

OWNER = "owner@example.com"
INTRUDER = "intruder@example.com"
CONVO = "conv-audit"


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
    """
    A spec, plus the registry entry the executor will actually resolve.

    Both are needed now: `intercept` reads the spec, while `confirm_and_execute`
    rebuilds the callable from the tool *name*. Registering keeps the recorder
    on the path production takes rather than beside it — and without it a
    confirmed action would reach the real SMTP sender.
    """
    preview = "To: alice@example.com\nSubject: Hi\n\nHello."
    register_confirmable(name, recorder, effect=effect, preview=preview)
    return {"callable": recorder, "effect": effect, "preview": preview}


@pytest.fixture
def store():
    return InMemoryAuditLog()


@pytest.fixture
def waiting_room():
    from app.domain.pending_actions import InMemoryPendingActionStore

    return InMemoryPendingActionStore()


@pytest.fixture
def gateway(store, waiting_room):
    """
    Both stores injected. A gateway missing either reaches for Postgres and —
    correctly — refuses everything, which would make these tests pass for the
    wrong reason.
    """
    return ActionGateway(audit=store, pending=waiting_room)


async def _hold(gateway, recorder, *, owner=OWNER, convo=CONVO, args=None,
                effect=Effect.EXTERNAL_WRITE):
    return await gateway.intercept(
        tool="send_email",
        spec=_spec(recorder, effect),
        arguments=args or {"to_email": "alice@example.com", "subject": "Hi",
                           "body": "Hello."},
        owner_id=owner,
        conversation_id=convo,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. The lifecycle
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_pending_action_creates_an_audit_row(gateway, store):
    recorder = _SendRecorder()
    await _hold(gateway, recorder)

    entry = store.only()
    assert entry.status is AuditStatus.PENDING
    assert entry.tool == "send_email"
    assert entry.effect == Effect.EXTERNAL_WRITE.value
    assert entry.owner_id == OWNER
    assert entry.conversation_id == CONVO
    assert entry.idempotency_key
    assert entry.created_at is not None
    assert entry.confirmed_at is None
    assert entry.executed_at is None
    assert recorder.count == 0


async def test_the_row_exists_before_the_action_is_offered(gateway, store):
    """
    Written before a token exists anywhere a user could act on. An action that
    is proposed and abandoned is therefore as visible as one that was sent.
    """
    recorded = {}

    original = store.record_pending

    async def watch(entry):
        recorded["pending_before_token"] = True
        return await original(entry)

    store.record_pending = watch
    await _hold(gateway, _SendRecorder())
    assert recorded.get("pending_before_token") is True


async def test_confirmation_updates_the_row(gateway, store):
    recorder = _SendRecorder()
    held = await _hold(gateway, recorder)

    await gateway.confirm_and_execute(
        held.data["confirmation_token"], owner_id=OWNER
    )

    entry = store.only()
    assert entry.status is AuditStatus.EXECUTED
    assert entry.confirmed_at is not None
    assert entry.executed_at is not None
    assert recorder.count == 1


async def test_a_successful_execution_records_executed(gateway, store):
    held = await _hold(gateway, _SendRecorder())
    await gateway.confirm_and_execute(held.data["confirmation_token"], owner_id=OWNER)

    assert store.by_status(AuditStatus.EXECUTED)
    assert store.only().error is None


async def test_a_failed_execution_records_failed(gateway, store):
    async def refuse(args):
        raise RuntimeError("smtp refused the message")

    held = await gateway.intercept(
        tool="send_email",
        spec=_spec(refuse),
        arguments={"to_email": "a@b.com"},
        owner_id=OWNER, conversation_id=CONVO,
    )
    result = await gateway.confirm_and_execute(
        held.data["confirmation_token"], owner_id=OWNER
    )

    entry = store.only()
    assert result.is_error is True
    assert entry.status is AuditStatus.FAILED
    assert "smtp refused" in (entry.error or "")


async def test_a_tool_reporting_its_own_failure_records_failed(gateway, store):
    async def reports_failure(args):
        return {"success": False, "error": "mailbox full"}

    held = await gateway.intercept(
        tool="send_email",
        spec=_spec(reports_failure),
        arguments={"to_email": "a@b.com"},
        owner_id=OWNER, conversation_id=CONVO,
    )
    await gateway.confirm_and_execute(held.data["confirmation_token"], owner_id=OWNER)

    assert store.only().status is AuditStatus.FAILED


async def test_cancellation_records_cancelled(gateway, store):
    recorder = _SendRecorder()
    held = await _hold(gateway, recorder)

    assert await gateway.cancel(held.data["confirmation_token"], owner_id=OWNER) is True

    assert store.only().status is AuditStatus.CANCELLED
    assert recorder.count == 0


async def test_expiration_records_expired(store, waiting_room):
    """
    A TTL short enough to lapse, long enough to survive the eviction that runs
    inside `intercept`. With a zero TTL the action is already expired when it
    is inserted and never becomes pending at all — a degenerate case covered
    separately below.
    """
    expiring = ActionGateway(ttl_seconds=0.02, audit=store, pending=waiting_room)
    recorder = _SendRecorder()
    held = await _hold(expiring, recorder)

    assert held.is_pending is True
    assert store.only().status is AuditStatus.PENDING

    await asyncio.sleep(0.05)
    swept = await expiring.sweep_expired()

    assert swept == 1
    assert store.only().status is AuditStatus.EXPIRED
    assert recorder.count == 0
    assert await expiring.pending_for(CONVO, OWNER) == []


async def test_an_already_expired_action_is_never_offered(store, waiting_room):
    """
    A zero TTL lapses the instant it is written.

    It is never *offered* — `pending_for` filters on expiry, so no one can act
    on it — and the audit row settles at `expired` once swept. The row is not
    updated at write time because expiry is now a property of the stored
    `expires_at`, evaluated on read, rather than something a process notices
    while holding the action in memory.
    """
    expiring = ActionGateway(ttl_seconds=0.0, audit=store, pending=waiting_room)
    recorder = _SendRecorder()
    await _hold(expiring, recorder)

    assert await expiring.pending_for(CONVO, OWNER) == []
    assert recorder.count == 0

    assert await expiring.sweep_expired() == 1
    assert store.only().status is AuditStatus.EXPIRED


# ═══════════════════════════════════════════════════════════════════════════
# 2. Fail closed
# ═══════════════════════════════════════════════════════════════════════════

async def test_an_audit_outage_prevents_the_action_being_offered(gateway, store):
    """The acceptance criterion's second half: no database, no email."""
    recorder = _SendRecorder()
    store.failing = True

    result = await _hold(gateway, recorder)

    assert result.is_error is True
    assert result.is_pending is False
    assert recorder.count == 0
    assert await gateway.pending_for(CONVO, OWNER) == []


async def test_an_audit_outage_at_confirmation_prevents_execution(gateway, store):
    recorder = _SendRecorder()
    held = await _hold(gateway, recorder)

    store.failing = True
    result = await gateway.confirm_and_execute(
        held.data["confirmation_token"], owner_id=OWNER
    )

    assert result.is_error is True
    assert recorder.count == 0


async def test_an_audit_outage_at_the_claim_prevents_execution(gateway, store):
    """
    The last gate. The approval was recorded; the executed claim then fails, so
    the tool must not run — a send with no record of having happened is exactly
    what this whole change exists to prevent.
    """
    recorder = _SendRecorder()
    held = await _hold(gateway, recorder)

    # confirm() writes CONFIRMED, then confirm_and_execute writes EXECUTED.
    store.fail_next(1)
    original_mark = store.mark
    calls = {"n": 0}

    async def fail_on_claim(audit_id, status, **kw):
        calls["n"] += 1
        if status is AuditStatus.EXECUTED:
            raise AuditWriteError("simulated outage claiming the action")
        return await original_mark(audit_id, status, **kw)

    store._fail_next = 0
    store.mark = fail_on_claim

    result = await gateway.confirm_and_execute(
        held.data["confirmation_token"], owner_id=OWNER
    )

    assert result.is_error is True
    assert recorder.count == 0, "the tool ran without a durable claim"


async def test_an_unreadable_audit_prevents_the_idempotency_check(gateway, store):
    """
    Not knowing whether something already ran is not permission to run it.
    """
    recorder = _SendRecorder()
    store.failing = True

    result = await _hold(gateway, recorder)

    assert result.is_error is True
    assert recorder.count == 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. Durable idempotency — the acceptance criterion
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_restart_cannot_repeat_a_completed_action(store, waiting_room):
    """
    THE acceptance criterion.

    A confirmed send completes. The process dies — modelled by discarding the
    gateway entirely, since its pending map and every in-process set go with
    it. A new gateway comes up against the same durable record and must refuse
    the same action.
    """
    recorder = _SendRecorder()

    before = ActionGateway(audit=store, pending=waiting_room)
    held = await _hold(before, recorder)
    await before.confirm_and_execute(held.data["confirmation_token"], owner_id=OWNER)
    assert recorder.count == 1

    # ── process restart ──────────────────────────────────────────────────
    del before
    after = ActionGateway(audit=store, pending=waiting_room)

    assert await after.pending_for(CONVO, OWNER) == [], "pending state should not survive"

    repeat = await _hold(after, recorder)

    assert repeat.is_pending is False
    assert repeat.data.get("already_executed") is True
    assert recorder.count == 1, "the same action executed twice across a restart"


async def test_a_second_replica_sees_the_same_completion(store, waiting_room):
    """
    Two gateways alive at once, as two workers would be. The record is shared;
    the process memory is not.
    """
    recorder = _SendRecorder()
    replica_a = ActionGateway(audit=store, pending=waiting_room)
    replica_b = ActionGateway(audit=store, pending=waiting_room)

    held = await _hold(replica_a, recorder)
    await replica_a.confirm_and_execute(held.data["confirmation_token"], owner_id=OWNER)
    assert recorder.count == 1

    on_b = await _hold(replica_b, recorder)
    assert on_b.data.get("already_executed") is True
    assert recorder.count == 1


async def test_the_executed_constraint_is_enforced_by_the_store(store, waiting_room):
    """
    Mirrors the Postgres partial unique index. Two executed rows for one
    idempotency key is a state the database will not hold, and neither will
    the double used to stand in for it.
    """
    recorder = _SendRecorder()
    gateway = ActionGateway(audit=store, pending=waiting_room)

    first = await _hold(gateway, recorder)
    await gateway.confirm_and_execute(first.data["confirmation_token"], owner_id=OWNER)

    # Force a second pending row for the identical action, bypassing the
    # gateway's own check, then try to claim it.
    entry = next(iter(store.entries.values()))
    from app.domain.audit import AuditEntry

    duplicate = AuditEntry(
        audit_id="00000000-0000-0000-0000-0000000000ff",
        owner_id=OWNER, conversation_id=CONVO,
        effect=entry.effect, tool=entry.tool,
        arguments_hash=entry.arguments_hash, preview_hash=entry.preview_hash,
        token_hash="deadbeef", idempotency_key=entry.idempotency_key,
    )
    await store.record_pending(duplicate)

    with pytest.raises(AuditWriteError):
        await store.mark(duplicate.audit_id, AuditStatus.EXECUTED)


async def test_a_failed_action_may_be_retried(store, waiting_room):
    """
    The deliberate asymmetry. A crash mid-call leaves the claim standing and
    nothing retries; an *explicit* failure releases it, because a send that
    demonstrably did not happen should not be blocked forever.
    """
    attempts = {"n": 0}

    async def flaky(args):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient smtp error")
        return {"success": True, "message_id": "m1"}

    gateway = ActionGateway(audit=store, pending=waiting_room)
    spec = _spec(flaky)
    args = {"to_email": "a@b.com"}

    first = await gateway.intercept(tool="send_email", spec=spec, arguments=args,
                                    owner_id=OWNER, conversation_id=CONVO)
    failed = await gateway.confirm_and_execute(
        first.data["confirmation_token"], owner_id=OWNER)
    assert failed.is_error is True

    second = await gateway.intercept(tool="send_email", spec=spec, arguments=args,
                                     owner_id=OWNER, conversation_id=CONVO)
    assert second.is_pending is True, "a demonstrably failed send should be retryable"

    ok = await gateway.confirm_and_execute(
        second.data["confirmation_token"], owner_id=OWNER)
    assert ok.ok is True
    assert attempts["n"] == 2


async def test_duplicate_confirmation_executes_only_once(gateway, store):
    recorder = _SendRecorder()
    held = await _hold(gateway, recorder)
    token = held.data["confirmation_token"]

    first = await gateway.confirm_and_execute(token, owner_id=OWNER)
    second = await gateway.confirm_and_execute(token, owner_id=OWNER)

    assert first.ok is True
    assert second.is_error is True
    assert recorder.count == 1
    assert len(store.by_status(AuditStatus.EXECUTED)) == 1


async def test_concurrent_confirmations_produce_one_executed_row(gateway, store):
    recorder = _SendRecorder()
    held = await _hold(gateway, recorder)
    token = held.data["confirmation_token"]

    results = await asyncio.gather(
        gateway.confirm_and_execute(token, owner_id=OWNER),
        gateway.confirm_and_execute(token, owner_id=OWNER),
        gateway.confirm_and_execute(token, owner_id=OWNER),
    )

    assert sum(1 for r in results if r.ok) == 1
    assert recorder.count == 1
    assert len(store.by_status(AuditStatus.EXECUTED)) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 4. Security properties, now with a record
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_wrong_owner_cannot_confirm_and_leaves_the_row_pending(gateway, store):
    recorder = _SendRecorder()
    held = await _hold(gateway, recorder)

    result = await gateway.confirm_and_execute(
        held.data["confirmation_token"], owner_id=INTRUDER
    )

    assert result.is_error is True
    assert recorder.count == 0
    assert store.only().status is AuditStatus.PENDING
    assert store.only().confirmed_at is None


async def test_a_tampered_content_hash_cannot_execute(gateway, store):
    recorder = _SendRecorder()
    held = await _hold(gateway, recorder)

    result = await gateway.confirm_and_execute(
        held.data["confirmation_token"], owner_id=OWNER, content_hash="0" * 64
    )

    assert result.is_error is True
    assert recorder.count == 0
    assert store.only().status is AuditStatus.PENDING


async def test_the_raw_confirmation_token_is_never_stored(gateway, store):
    recorder = _SendRecorder()
    held = await _hold(gateway, recorder)
    token = held.data["confirmation_token"]

    entry = store.only()
    assert entry.token_hash == digest(token)
    assert entry.token_hash != token

    # Nowhere in the serialised row, under any field.
    serialised = repr(entry) + str(entry.summary())
    assert token not in serialised


async def test_arguments_and_preview_are_stored_only_as_digests(gateway, store):
    recorder = _SendRecorder()
    await gateway.intercept(
        tool="send_email",
        spec=_spec(recorder),
        arguments={"to_email": "secret@example.com", "subject": "Confidential",
                   "body": "Bank details: 12345678"},
        owner_id=OWNER, conversation_id=CONVO,
    )

    entry = store.only()
    blob = repr(entry)
    assert "secret@example.com" not in blob
    assert "Bank details" not in blob
    assert "12345678" not in blob
    assert len(entry.arguments_hash) == 64
    assert len(entry.preview_hash) == 64


async def test_the_preview_hash_matches_what_was_previewed(gateway, store):
    held = await _hold(gateway, _SendRecorder())
    action = await gateway.get(held.data["confirmation_token"])
    assert store.only().preview_hash == digest(action.preview)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Reads and local writes stay out of the log
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("effect", [Effect.READ, Effect.LOCAL_WRITE])
async def test_harmless_effects_are_not_audited(effect, store, waiting_room):
    """
    Auditing every résumé lookup would bury the rows that matter. The log is
    for actions that leave the system or cannot be undone.
    """
    from tests.support import ScriptedAgent

    tool = _SendRecorder()
    agent = ScriptedAgent([
        tool_call("harmless", x=1),
        final("done"),
    ])
    await agent.execute_reasoning_loop(
        state=state("do the harmless thing"),
        base_system_prompt="p",
        tools={"harmless": {"callable": tool, "effect": effect}},
    )

    assert tool.count == 1, "a harmless tool should just run"
    assert store.entries == {}


# ═══════════════════════════════════════════════════════════════════════════
# 6. DESTRUCTIVE actions
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_destructive_action_is_audited_through_its_lifecycle(gateway, store):
    deleted: list[str] = []

    async def forget(args):
        deleted.append(args["key"])
        return {"success": True}

    held = await gateway.intercept(
        tool="forget_preference",
        spec=_spec(forget, effect=Effect.DESTRUCTIVE, name="forget_preference"),
        arguments={"key": "preferred_tone"},
        owner_id=OWNER, conversation_id=CONVO,
    )

    assert store.only().status is AuditStatus.PENDING
    assert store.only().effect == Effect.DESTRUCTIVE.value
    assert deleted == []

    await gateway.confirm_and_execute(held.data["confirmation_token"], owner_id=OWNER)

    assert deleted == ["preferred_tone"]
    assert store.only().status is AuditStatus.EXECUTED


async def test_a_rejected_destructive_action_deletes_nothing(gateway, store):
    deleted: list[str] = []

    async def forget(args):
        deleted.append(args["key"])
        return {"success": True}

    held = await gateway.intercept(
        tool="forget_preference",
        spec=_spec(forget, effect=Effect.DESTRUCTIVE, name="forget_preference"),
        arguments={"key": "x"},
        owner_id=OWNER, conversation_id=CONVO,
    )
    await gateway.cancel(held.data["confirmation_token"], owner_id=OWNER)

    assert deleted == []
    assert store.only().status is AuditStatus.CANCELLED


# ═══════════════════════════════════════════════════════════════════════════
# 7. The real email registry, SMTP stubbed
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_real_email_agent_produces_a_full_audit_trail(monkeypatch, audit_store):
    """
    End to end on production code: the real EmailAgent registry, the shared
    gateway, the conversational confirmation route. Only SMTP is replaced.
    """
    from app.agents.email_agent import EmailAgent

    recorded = stub_services(monkeypatch)

    prepared, _ = await drive(
        EmailAgent(),
        [tool_call("send_email", to_email="alice@example.com",
                   subject="Interview follow-up", body="Thanks for your time."),
         final("Ready for your approval.")],
        state("send this email to alice", user_id=OWNER, session_id=CONVO),
    )

    assert recorded.emails_sent == 0
    entry = audit_store.only()
    assert entry.status is AuditStatus.PENDING
    assert entry.tool == "send_email"
    assert "alice@example.com" not in repr(entry), "recipient leaked into the record"

    # Ambiguity changes nothing.
    await resolve(state("okay", user_id=OWNER, session_id=CONVO))
    assert recorded.emails_sent == 0
    assert audit_store.only().status is AuditStatus.PENDING

    # An explicit yes completes the trail.
    outcome = await resolve(state("yes", user_id=OWNER, session_id=CONVO))
    assert outcome.executed is True
    assert recorded.emails_sent == 1

    entry = audit_store.only()
    assert entry.status is AuditStatus.EXECUTED
    assert entry.confirmed_at is not None
    assert entry.executed_at is not None

    # And the record blocks a repeat.
    again, _ = await drive(
        EmailAgent(),
        [tool_call("send_email", to_email="alice@example.com",
                   subject="Interview follow-up", body="Thanks for your time."),
         final("...")],
        state("send it again", user_id=OWNER, session_id=CONVO),
    )
    assert recorded.emails_sent == 1


async def test_the_real_email_agent_sends_nothing_when_the_audit_is_down(
    monkeypatch, audit_store,
):
    """The other acceptance criterion, on production code."""
    from app.agents.email_agent import EmailAgent

    recorded = stub_services(monkeypatch)
    audit_store.failing = True

    result, _ = await drive(
        EmailAgent(),
        [tool_call("send_email", to_email="alice@example.com",
                   subject="Hi", body="Hello."),
         final("...")],
        state("send this email", user_id=OWNER, session_id=CONVO),
    )

    assert recorded.emails_sent == 0
    assert result["pending_actions"] == []
    assert await action_gateway.pending_for(CONVO, OWNER) == []


# ═══════════════════════════════════════════════════════════════════════════
# 8. Querying the history
# ═══════════════════════════════════════════════════════════════════════════

async def test_history_is_scoped_to_the_owner(gateway, store):
    await _hold(gateway, _SendRecorder(), owner=OWNER)
    await _hold(gateway, _SendRecorder(), owner=INTRUDER,
                args={"to_email": "z@z.com", "subject": "x", "body": "y"})

    mine = await store.history(OWNER)
    theirs = await store.history(INTRUDER)

    assert len(mine) == 1 and mine[0].owner_id == OWNER
    assert len(theirs) == 1 and theirs[0].owner_id == INTRUDER


async def test_an_action_can_be_found_by_its_token_digest(gateway, store):
    held = await _hold(gateway, _SendRecorder())
    token = held.data["confirmation_token"]

    found = await store.find_by_token_hash(digest(token))
    assert found is not None
    assert found.tool == "send_email"
