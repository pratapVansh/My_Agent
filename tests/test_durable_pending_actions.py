"""
Confirmations that outlive the process that asked for them.

Two deployments were quietly broken while pending actions lived in a
dictionary. A restart between "shall I send this?" and "yes" lost the action.
Two workers behind a load balancer lost it roughly half the time, because the
approval landed on the replica that had never heard of it. Both failed closed —
correct, and still the wrong behaviour.

Moving the wait into Postgres fixes both, and introduces exactly one new attack
surface worth worrying about: a row that decides what code runs. The answer is
that it does not. A stored action holds a tool *name*, and the callable is
rebuilt from `confirmable_tools` at confirmation time — so a tampered row can
only claim to be one of the tools this build already ships, and the effect,
scope and behaviour all come from the application rather than from the
database. `test_a_stored_action_cannot_introduce_new_behaviour` is where that
is pinned.

Every security property from T2 and T3 is re-asserted here against the durable
path, because a guarantee that held in memory and not on disk would be worse
than no guarantee at all — it would look tested.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agents.actions import ActionGateway
from app.agents.confirmable_tools import CONFIRMABLE_TOOLS, resolve_confirmable_tool
from app.agents.confirmation import resolve
from app.domain.audit import AuditStatus, InMemoryAuditLog, digest
from app.domain.pending_actions import (
    InMemoryPendingActionStore,
    PendingStoreError,
    StoredAction,
)
from app.tools.contract import Effect, ToolStatus
from tests.support import (
    drive,
    final,
    register_confirmable,
    state,
    stub_services,
    tool_call,
)

OWNER = "owner@example.com"
INTRUDER = "intruder@example.com"
CONVO = "conv-durable"
OTHER_CONVO = "conv-other"


class _SendRecorder:
    def __init__(self):
        self.sent: list[dict] = []

    async def __call__(self, args):
        self.sent.append(dict(args))
        return {"success": True, "message_id": f"m{len(self.sent)}"}

    @property
    def count(self) -> int:
        return len(self.sent)


@pytest.fixture
def disk():
    """The durable stores — shared across every 'process' in a test."""
    return {"audit": InMemoryAuditLog(), "pending": InMemoryPendingActionStore()}


def boot(disk, **kwargs) -> ActionGateway:
    """Start a 'process': a fresh gateway over the same durable stores."""
    return ActionGateway(audit=disk["audit"], pending=disk["pending"], **kwargs)


def _spec(recorder, effect=Effect.EXTERNAL_WRITE, name="send_email"):
    preview = "To: alice@example.com\nSubject: Hi\n\nHello."
    register_confirmable(name, recorder, effect=effect, preview=preview)
    return {"callable": recorder, "effect": effect, "preview": preview}


async def hold(gateway, recorder, *, owner=OWNER, convo=CONVO, args=None,
               effect=Effect.EXTERNAL_WRITE, name="send_email"):
    return await gateway.intercept(
        tool=name,
        spec=_spec(recorder, effect, name),
        arguments=args or {"to_email": "alice@example.com", "subject": "Hi",
                           "body": "Hello."},
        owner_id=owner,
        conversation_id=convo,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Surviving a restart
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_pending_action_survives_a_restart(disk):
    """
    The headline. Prepared by one process, approved by its successor — which
    is what a deploy, a crash, or an ordinary worker recycle looks like.
    """
    recorder = _SendRecorder()

    before = boot(disk)
    await hold(before, recorder)
    assert recorder.count == 0

    # ── process restart ──────────────────────────────────────────────────
    del before
    after = boot(disk)

    outstanding = await after.pending_for(CONVO, OWNER)
    assert len(outstanding) == 1, "the action did not survive the restart"
    assert outstanding[0].tool == "send_email"
    assert "alice@example.com" in outstanding[0].preview

    result = await after.confirm_and_execute(
        handle=outstanding[0].handle, owner_id=OWNER,
    )
    assert result.ok is True
    assert recorder.count == 1


async def test_the_raw_token_does_not_survive_but_the_action_does(disk):
    """
    Only the digest is stored, so a reloaded action carries no token. The
    handle is what refers to it — a reference, not a bearer credential, since
    owner binding still applies.
    """
    recorder = _SendRecorder()
    before = boot(disk)
    held = await hold(before, recorder)
    raw_token = held.data["confirmation_token"]

    after = boot(disk)
    reloaded = (await after.pending_for(CONVO, OWNER))[0]

    assert reloaded.token == ""
    assert reloaded.handle == digest(raw_token)
    assert raw_token not in repr(reloaded)


async def test_the_original_token_still_works_after_a_restart(disk):
    """A UI holding the raw token can still post it back; it hashes to the same row."""
    recorder = _SendRecorder()
    before = boot(disk)
    held = await hold(before, recorder)

    after = boot(disk)
    result = await after.confirm_and_execute(
        held.data["confirmation_token"], owner_id=OWNER,
    )
    assert result.ok is True
    assert recorder.count == 1


async def test_a_conversation_can_approve_across_a_restart(disk, monkeypatch):
    """
    End to end through the real conversational route: the action is prepared,
    the process restarts, and "yes" still lands.
    """
    from app.agents.actions import action_gateway
    from app.agents.email_agent import EmailAgent

    recorded = stub_services(monkeypatch)
    action_gateway.use_audit_log(disk["audit"])
    action_gateway.use_pending_store(disk["pending"])

    await drive(
        EmailAgent(),
        [tool_call("send_email", to_email="alice@example.com",
                   subject="Hi", body="Hello."),
         final("Ready for your approval.")],
        state("send this to alice", user_id=OWNER, session_id=CONVO),
    )
    assert recorded.emails_sent == 0

    # A new process picks up the same durable stores.
    action_gateway.use_pending_store(disk["pending"])
    action_gateway.use_audit_log(disk["audit"])

    outcome = await resolve(state("yes", user_id=OWNER, session_id=CONVO))
    assert outcome.executed is True
    assert recorded.emails_sent == 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. Cross-worker confirmation
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_second_worker_can_confirm_what_the_first_prepared(disk):
    """
    The load-balancer case. Worker A never tells worker B anything; the shared
    row is the entire channel between them.
    """
    recorder = _SendRecorder()
    worker_a = boot(disk)
    worker_b = boot(disk)

    await hold(worker_a, recorder)

    on_b = await worker_b.pending_for(CONVO, OWNER)
    assert len(on_b) == 1, "worker B cannot see the action worker A prepared"

    result = await worker_b.confirm_and_execute(handle=on_b[0].handle, owner_id=OWNER)
    assert result.ok is True
    assert recorder.count == 1

    # And worker A now agrees it is gone.
    assert await worker_a.pending_for(CONVO, OWNER) == []


async def test_only_one_worker_wins_a_concurrent_confirmation(disk):
    """
    Two workers confirming the same action at the same instant. The claim is a
    single atomic delete, so exactly one can win — and exactly one email goes.
    """
    recorder = _SendRecorder()
    prep = boot(disk)
    held = await hold(prep, recorder)
    handle = digest(held.data["confirmation_token"])

    workers = [boot(disk) for _ in range(4)]
    results = await asyncio.gather(*[
        w.confirm_and_execute(handle=handle, owner_id=OWNER) for w in workers
    ])

    assert sum(1 for r in results if r.ok) == 1
    assert recorder.count == 1
    assert len(disk["audit"].by_status(AuditStatus.EXECUTED)) == 1


async def test_a_worker_cannot_see_another_conversation(disk):
    recorder = _SendRecorder()
    a = boot(disk)
    await hold(a, recorder, convo=CONVO)

    b = boot(disk)
    assert await b.pending_for(OTHER_CONVO, OWNER) == []
    assert await b.pending_for(CONVO, INTRUDER) == []


# ═══════════════════════════════════════════════════════════════════════════
# 3. Reconstruction is not deserialisation
# ═══════════════════════════════════════════════════════════════════════════

def test_no_callable_is_ever_persisted(disk):
    """
    The property that makes a durable action safe at all. `StoredAction` has no
    field that could hold code.
    """
    fields = set(StoredAction.__dataclass_fields__)
    assert "callable" not in fields and "callable_" not in fields
    assert "fn" not in fields
    # Everything stored is data.
    assert fields == {
        "id", "token_hash", "audit_id", "owner_id", "conversation_id",
        "tool", "effect", "arguments", "preview", "content_hash",
        "idempotency_key", "created_at", "expires_at",
    }


async def test_a_stored_action_cannot_introduce_new_behaviour(disk):
    """
    A row naming a tool this build does not ship cannot execute. The registry
    is the only source of callables, so the worst a tampered row can do is name
    one of the tools already present.
    """
    recorder = _SendRecorder()
    gateway = boot(disk)
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    # Tamper: repoint the row at a tool that does not exist.
    stored = await disk["pending"].peek(handle)
    from dataclasses import replace as _replace

    await disk["pending"].put(_replace(stored, tool="rm_minus_rf"))

    result = await gateway.confirm_and_execute(handle=handle, owner_id=OWNER)

    assert result.is_error is True
    assert recorder.count == 0
    assert disk["audit"].only().status is AuditStatus.FAILED


async def test_a_row_cannot_downgrade_its_own_effect(disk):
    """
    A tampered row claiming READ must not thereby escape the gate. The effect
    used for execution comes from the rebuilt registry entry, and a mismatch
    with the stored one is refused outright.
    """
    recorder = _SendRecorder()
    gateway = boot(disk)
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    stored = await disk["pending"].peek(handle)
    from dataclasses import replace as _replace

    await disk["pending"].put(_replace(stored, effect=Effect.READ.value))

    result = await gateway.confirm_and_execute(handle=handle, owner_id=OWNER)

    assert result.is_error is True
    assert recorder.count == 0


def test_every_confirmable_tool_is_resolvable():
    """
    Drift guard. Anything the gateway may hold must be reconstructible, or a
    restart would strand it permanently.
    """
    for name in CONFIRMABLE_TOOLS:
        spec = resolve_confirmable_tool(name, OWNER)
        assert spec is not None, name
        assert callable(spec["callable"]), name
        assert spec["effect"].requires_confirmation, name
        assert spec.get("preview") is not None, name


def test_an_unknown_tool_name_resolves_to_nothing():
    assert resolve_confirmable_tool("definitely_not_a_tool", OWNER) is None
    assert resolve_confirmable_tool("", OWNER) is None


async def test_the_real_agents_only_gate_resolvable_tools():
    """
    The registry the agents offer and the one the resolver uses must agree.
    They are the same factories, and this proves it for the live registries.
    """
    from app.agents.email_agent import EmailAgent
    from app.agents.profile_agent import ProfileAgent
    from tests.support import capture_registry

    for agent in (EmailAgent(), ProfileAgent()):
        tools = await capture_registry(agent, state(user_id=OWNER))
        for name, spec in tools.items():
            if spec.get("effect") and spec["effect"].requires_confirmation:
                assert resolve_confirmable_tool(name, OWNER) is not None, (
                    f"{agent.name} gates '{name}' but it cannot be reconstructed"
                )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Every security property, on the durable path
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_wrong_owner_cannot_confirm_and_the_action_survives(disk):
    """
    Crucially the action is *not* consumed by a failed attempt. Validating
    after claiming would let anyone holding a handle destroy someone else's
    outstanding action — a denial of service dressed up as a failed login.
    """
    recorder = _SendRecorder()
    gateway = boot(disk)
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    refused = await gateway.confirm_and_execute(handle=handle, owner_id=INTRUDER)

    assert refused.is_error is True
    assert recorder.count == 0
    assert len(await gateway.pending_for(CONVO, OWNER)) == 1, "the owner lost their action"

    # The real owner can still approve it.
    ok = await gateway.confirm_and_execute(handle=handle, owner_id=OWNER)
    assert ok.ok is True
    assert recorder.count == 1


async def test_a_tampered_content_hash_cannot_execute_and_preserves_the_action(disk):
    recorder = _SendRecorder()
    gateway = boot(disk)
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    refused = await gateway.confirm_and_execute(
        handle=handle, owner_id=OWNER, content_hash="0" * 64,
    )

    assert refused.is_error is True
    assert recorder.count == 0
    assert len(await gateway.pending_for(CONVO, OWNER)) == 1


async def test_tampered_arguments_change_the_hash_and_are_refused(disk):
    """
    The row is on disk now, so its arguments are within an attacker's reach in
    a way they never were in memory.

    Note what does *not* catch this: comparing the presented hash against the
    stored one. Edit the recipient and leave the hash alone and those two still
    agree — with each other, while describing different actions. So the hash is
    recomputed from the row's own contents at confirmation time, and a row that
    fails its own integrity check is refused and destroyed.
    """
    recorder = _SendRecorder()
    gateway = boot(disk)
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])
    approved_hash = held.data["content_hash"]

    stored = await disk["pending"].peek(handle)
    from dataclasses import replace as _replace

    tampered = dict(stored.arguments)
    tampered["to_email"] = "attacker@evil.com"
    await disk["pending"].put(_replace(stored, arguments=tampered))

    refused = await gateway.confirm_and_execute(
        handle=handle, owner_id=OWNER, content_hash=approved_hash,
    )

    assert refused.is_error is True
    assert recorder.count == 0
    # Destroyed rather than left available for another attempt.
    assert await gateway.pending_for(CONVO, OWNER) == []
    assert disk["audit"].only().status is AuditStatus.FAILED


async def test_a_row_that_recomputes_its_own_hash_is_still_caught_by_the_user(disk):
    """
    The other half of the pair. An attacker who edits the arguments *and*
    recomputes the row's hash passes the integrity check — but the hash the
    user approved no longer matches, and that is the check that stops it.
    """
    recorder = _SendRecorder()
    gateway = boot(disk)
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])
    approved_hash = held.data["content_hash"]

    stored = await disk["pending"].peek(handle)
    from dataclasses import replace as _replace

    from app.agents.actions import compute_content_hash as _hash

    tampered_args = dict(stored.arguments)
    tampered_args["to_email"] = "attacker@evil.com"
    await disk["pending"].put(_replace(
        stored,
        arguments=tampered_args,
        content_hash=_hash(stored.tool, tampered_args, stored.preview),
    ))

    refused = await gateway.confirm_and_execute(
        handle=handle, owner_id=OWNER, content_hash=approved_hash,
    )

    assert refused.is_error is True
    assert recorder.count == 0


async def test_an_expired_action_is_refused_and_recorded(disk):
    recorder = _SendRecorder()
    gateway = boot(disk, ttl_seconds=0.02)
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    await asyncio.sleep(0.05)

    refused = await gateway.confirm_and_execute(handle=handle, owner_id=OWNER)

    assert refused.is_error is True
    assert recorder.count == 0
    assert disk["audit"].only().status is AuditStatus.EXPIRED


async def test_an_expired_action_is_not_offered_after_a_restart(disk):
    recorder = _SendRecorder()
    before = boot(disk, ttl_seconds=0.02)
    await hold(before, recorder)

    await asyncio.sleep(0.05)
    after = boot(disk)

    assert await after.pending_for(CONVO, OWNER) == []


async def test_a_replayed_confirmation_is_refused(disk):
    recorder = _SendRecorder()
    gateway = boot(disk)
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    first = await gateway.confirm_and_execute(handle=handle, owner_id=OWNER)
    second = await gateway.confirm_and_execute(handle=handle, owner_id=OWNER)

    assert first.ok is True
    assert second.is_error is True
    assert recorder.count == 1


async def test_a_replay_after_a_restart_is_refused(disk):
    """Single-use survives the process, because the row is gone from the store."""
    recorder = _SendRecorder()
    before = boot(disk)
    held = await hold(before, recorder)
    handle = digest(held.data["confirmation_token"])
    await before.confirm_and_execute(handle=handle, owner_id=OWNER)

    after = boot(disk)
    replayed = await after.confirm_and_execute(handle=handle, owner_id=OWNER)

    assert replayed.is_error is True
    assert recorder.count == 1


async def test_cancellation_survives_and_removes_the_action(disk):
    recorder = _SendRecorder()
    before = boot(disk)
    held = await hold(before, recorder)
    handle = digest(held.data["confirmation_token"])

    assert await before.cancel(handle=handle, owner_id=OWNER) is True

    after = boot(disk)
    assert await after.pending_for(CONVO, OWNER) == []
    assert disk["audit"].only().status is AuditStatus.CANCELLED
    assert recorder.count == 0


async def test_a_stranger_cannot_cancel_your_action(disk):
    recorder = _SendRecorder()
    gateway = boot(disk)
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    assert await gateway.cancel(handle=handle, owner_id=INTRUDER) is False
    assert len(await gateway.pending_for(CONVO, OWNER)) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. Fail closed when the store is unavailable
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_pending_store_outage_prevents_the_action_being_offered(disk):
    recorder = _SendRecorder()
    gateway = boot(disk)
    disk["pending"].failing = True

    result = await hold(gateway, recorder)

    assert result.is_error is True
    assert result.is_pending is False
    assert recorder.count == 0
    # The audit row is closed out rather than left stuck at `pending`.
    assert disk["audit"].only().status is AuditStatus.FAILED


async def test_a_pending_store_outage_prevents_confirmation(disk):
    recorder = _SendRecorder()
    gateway = boot(disk)
    held = await hold(gateway, recorder)
    handle = digest(held.data["confirmation_token"])

    disk["pending"].failing = True
    result = await gateway.confirm_and_execute(handle=handle, owner_id=OWNER)

    assert result.is_error is True
    assert recorder.count == 0


async def test_an_unreadable_store_offers_nothing(disk):
    """
    With no way to know what is outstanding, offering nothing is the only safe
    answer — and it means a "yes" falls through to ordinary conversation rather
    than approving something unknown.
    """
    gateway = boot(disk)
    await hold(gateway, _SendRecorder())

    disk["pending"].failing = True
    assert await gateway.pending_for(CONVO, OWNER) == []


# ═══════════════════════════════════════════════════════════════════════════
# 6. Reads and local writes are untouched
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("effect", [Effect.READ, Effect.LOCAL_WRITE])
async def test_harmless_effects_never_reach_the_store(effect, disk):
    from tests.support import ScriptedAgent

    tool = _SendRecorder()
    agent = ScriptedAgent([tool_call("harmless", x=1), final("done")])
    await agent.execute_reasoning_loop(
        state=state("do the harmless thing"),
        base_system_prompt="p",
        tools={"harmless": {"callable": tool, "effect": effect}},
    )

    assert tool.count == 1
    assert disk["pending"].count == 0
    assert disk["audit"].entries == {}
