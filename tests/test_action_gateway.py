"""
The action gateway, tested against the component it exists to contain.

The threat model here is not an attacker on the network. It is the language
model: the part of the system most easily steered by text it read somewhere,
holding `send_email` in its tool registry. So most of what follows is written
from its point of view — a model that has decided to send, and every route it
might take to get there.

The acceptance criterion is one sentence, and the tests are organised to
falsify it:

    no sequence of model outputs causes an EXTERNAL_WRITE or DESTRUCTIVE
    action without an explicit, valid, user-bound confirmation.

`_SendRecorder` is the instrument throughout. It stands in for SMTP and counts
deliveries, so "was it sent" is a number rather than an inference — the
distinction that matters when the failure mode under test is *sending twice*.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from app.agents.actions import (
    ActionGateway,
    ActionPreview,
    Denial,
    build_preview,
    compute_content_hash,
)
from app.domain.audit import digest
from app.tools.contract import Effect, ErrorKind, ToolResult, ToolStatus
from tests.support import ScriptedAgent, register_confirmable


OWNER = "owner@example.com"
INTRUDER = "someone-else@example.com"
CONVERSATION = "conv-1"


class _SendRecorder:
    """Stands in for the SMTP sender. Counts real deliveries."""

    def __init__(self):
        self.sent: list[dict] = []

    async def __call__(self, args):
        self.sent.append(dict(args))
        return {"success": True, "message_id": f"msg-{len(self.sent)}"}

    @property
    def count(self) -> int:
        return len(self.sent)


def _send_spec(recorder, preview="To: alice@example.com\nSubject: Hi\n\nHello."):
    """
    A spec, plus the registry entry `confirm_and_execute` will resolve.

    Execution rebuilds the callable from the tool *name*, so a recorder passed
    only in the spec would be bypassed and the real sender would run.
    """
    register_confirmable("send_email", recorder,
                         effect=Effect.EXTERNAL_WRITE, preview=preview)
    return {
        "callable": recorder,
        "effect": Effect.EXTERNAL_WRITE,
        "preview": preview,
        "description": "send an email",
    }


@pytest.fixture
def gateway(audit_store, pending_store):
    """
    A private gateway backed by the same in-memory stores the shared one uses.

    Both are explicit rather than defaulted: a gateway constructed without them
    reaches for Postgres and — correctly — refuses every consequential action,
    which would make these tests pass for entirely the wrong reason.
    """
    return ActionGateway(audit=audit_store, pending=pending_store)


def _run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Interception — the tool is never reached
# ═══════════════════════════════════════════════════════════════════════════

def test_an_external_write_is_held_and_never_executed(gateway):
    """The core claim, at its simplest."""
    recorder = _SendRecorder()
    result = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(recorder),
        arguments={"to_email": "alice@example.com", "subject": "Hi", "body": "Hello."},
        owner_id=OWNER,
        conversation_id=CONVERSATION,
    ))

    assert result.status is ToolStatus.PENDING_CONFIRMATION
    assert result.is_pending is True
    assert result.ok is False
    assert recorder.count == 0, "the send tool ran during interception"


def test_a_destructive_action_is_held_too(gateway):
    deleted = []

    async def forget(args):
        deleted.append(args["key"])
        return {"success": True}

    result = _run(gateway.intercept(
        tool="forget_preference",
        spec={"callable": forget, "effect": Effect.DESTRUCTIVE},
        arguments={"key": "preferred_tone"},
        owner_id=OWNER,
    ))

    assert result.is_pending is True
    assert deleted == []


@pytest.mark.parametrize("effect", [Effect.READ, Effect.LOCAL_WRITE])
def test_reads_and_local_writes_are_not_gated(effect, gateway):
    assert gateway.requires_confirmation(effect) is False


@pytest.mark.parametrize("effect", [Effect.EXTERNAL_WRITE, Effect.DESTRUCTIVE])
def test_consequential_effects_are_gated(effect, gateway):
    assert gateway.requires_confirmation(effect) is True


async def test_the_pending_action_carries_everything_needed_to_review_it(gateway):
    result = (await gateway.intercept(
        tool="send_email",
        spec=_send_spec(_SendRecorder()),
        arguments={"to_email": "alice@example.com", "subject": "Hi", "body": "Hello."},
        owner_id=OWNER,
        conversation_id=CONVERSATION,
    ))
    action = await gateway.get(result.data["confirmation_token"])

    assert action.tool == "send_email"
    assert action.effect is Effect.EXTERNAL_WRITE
    assert action.arguments["to_email"] == "alice@example.com"
    assert action.preview
    assert action.idempotency_key
    assert action.content_hash
    assert action.owner_id == OWNER
    assert action.expires_at > action.created_at

    # Reloaded from the store, so the raw token is gone — only its digest was
    # persisted. The handle is what refers to it from here on.
    assert action.token == ""
    assert action.handle == digest(result.data["confirmation_token"])


def test_the_observation_forbids_claiming_the_action_happened(gateway):
    """
    The model is told, in the observation itself, that nothing ran. This is a
    belt to the gateway's braces — enforcement does not depend on it, but a
    model that reports "email sent" when none was is its own kind of harm.
    """
    result = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(_SendRecorder()),
        arguments={"to_email": "a@b.com", "subject": "s", "body": "b"},
        owner_id=OWNER,
    ))
    text = result.observation()
    assert "NOT executed" in text
    assert "explicit approval" in text
    assert "PREVIEW" in text


# ═══════════════════════════════════════════════════════════════════════════
# 2. Valid confirmation — exactly once
# ═══════════════════════════════════════════════════════════════════════════

def test_a_valid_confirmation_sends_exactly_once(gateway):
    recorder = _SendRecorder()
    held = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(recorder),
        arguments={"to_email": "alice@example.com", "subject": "Hi", "body": "Hello."},
        owner_id=OWNER,
    ))
    token = held.data["confirmation_token"]

    result = _run(gateway.confirm_and_execute(token, owner_id=OWNER))

    assert result.ok is True
    assert recorder.count == 1
    assert recorder.sent[0]["to_email"] == "alice@example.com"


def test_confirmation_may_also_assert_the_content_hash(gateway):
    """Belt and braces: the caller can re-present what it displayed."""
    recorder = _SendRecorder()
    held = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(recorder),
        arguments={"to_email": "a@b.com", "subject": "s", "body": "b"},
        owner_id=OWNER,
    ))
    result = _run(gateway.confirm_and_execute(
        held.data["confirmation_token"],
        owner_id=OWNER,
        content_hash=held.data["content_hash"],
    ))
    assert result.ok is True
    assert recorder.count == 1


def test_execution_replays_the_resolved_arguments_not_the_models(gateway):
    """
    A preview builder that resolves a draft decides what gets sent. If the
    executed arguments were the model's originals, the user would approve one
    email and a different one would go out.
    """
    recorder = _SendRecorder()

    async def resolving_preview(args):
        return ActionPreview(
            "To: real@example.com\nSubject: Resolved\n\nResolved body.",
            {"to_email": "real@example.com", "subject": "Resolved", "body": "Resolved body."},
        )

    register_confirmable("send_email", recorder, effect=Effect.EXTERNAL_WRITE,
                         preview=resolving_preview)
    held = _run(gateway.intercept(
        tool="send_email",
        spec={"callable": recorder, "effect": Effect.EXTERNAL_WRITE,
              "preview": resolving_preview},
        arguments={"draft_id": 7},          # all the model supplied
        owner_id=OWNER,
    ))
    _run(gateway.confirm_and_execute(held.data["confirmation_token"], owner_id=OWNER))

    assert recorder.sent[0]["to_email"] == "real@example.com"
    assert recorder.sent[0]["body"] == "Resolved body."


# ═══════════════════════════════════════════════════════════════════════════
# 3. Token attacks
# ═══════════════════════════════════════════════════════════════════════════

def test_an_unknown_token_is_refused(gateway):
    outcome = _run(gateway.confirm("not-a-real-token", owner_id=OWNER))
    assert outcome.approved is False
    assert outcome.denial is Denial.UNKNOWN_TOKEN


def test_an_expired_token_is_refused_and_nothing_is_sent(gateway, audit_store, pending_store):
    recorder = _SendRecorder()
    short = ActionGateway(ttl_seconds=0.0, audit=audit_store, pending=pending_store)
    held = _run(short.intercept(
        tool="send_email",
        spec=_send_spec(recorder),
        arguments={"to_email": "a@b.com", "subject": "s", "body": "b"},
        owner_id=OWNER,
    ))
    result = _run(short.confirm_and_execute(
        held.data["confirmation_token"], owner_id=OWNER
    ))

    assert result.is_error is True
    assert recorder.count == 0


def test_expiry_is_evaluated_against_the_clock_not_a_flag(gateway):
    from app.agents.actions import _utcnow

    held = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(_SendRecorder()),
        arguments={"to_email": "a@b.com", "subject": "s", "body": "b"},
        owner_id=OWNER,
    ))
    token = held.data["confirmation_token"]
    future = _utcnow() + timedelta(seconds=10_000)

    outcome = _run(gateway.confirm(token, owner_id=OWNER, now=future))
    assert outcome.approved is False
    assert outcome.denial is Denial.EXPIRED


def test_a_token_cannot_be_reused(gateway):
    """Single-use. The second attempt must not deliver a second email."""
    recorder = _SendRecorder()
    held = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(recorder),
        arguments={"to_email": "a@b.com", "subject": "s", "body": "b"},
        owner_id=OWNER,
    ))
    token = held.data["confirmation_token"]

    first = _run(gateway.confirm_and_execute(token, owner_id=OWNER))
    second = _run(gateway.confirm_and_execute(token, owner_id=OWNER))

    assert first.ok is True
    assert second.is_error is True
    assert recorder.count == 1


def test_a_token_is_consumed_even_when_execution_fails(gateway):
    """
    A failed send must not leave a live token. Otherwise whoever holds it gets
    a second attempt at an external action the user approved exactly once.
    """
    async def failing_send(args):
        raise RuntimeError("smtp refused")

    held = _run(gateway.intercept(
        tool="send_email",
        spec={"callable": failing_send, "effect": Effect.EXTERNAL_WRITE},
        arguments={"to_email": "a@b.com", "subject": "s", "body": "b"},
        owner_id=OWNER,
    ))
    token = held.data["confirmation_token"]

    first = _run(gateway.confirm_and_execute(token, owner_id=OWNER))
    assert first.is_error is True

    replay = _run(gateway.confirm(token, owner_id=OWNER))
    assert replay.approved is False


def test_concurrent_confirmations_of_one_token_send_once(gateway):
    """
    The race. Two confirmations arriving together must not both pass the
    single-use check — which is why the record is removed under the lock
    before any execution begins, rather than marked afterwards.
    """
    recorder = _SendRecorder()
    held = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(recorder),
        arguments={"to_email": "a@b.com", "subject": "s", "body": "b"},
        owner_id=OWNER,
    ))
    token = held.data["confirmation_token"]

    async def race():
        return await asyncio.gather(
            gateway.confirm_and_execute(token, owner_id=OWNER),
            gateway.confirm_and_execute(token, owner_id=OWNER),
            gateway.confirm_and_execute(token, owner_id=OWNER),
        )

    results = _run(race())
    assert sum(1 for r in results if r.ok) == 1
    assert recorder.count == 1


# ═══════════════════════════════════════════════════════════════════════════
# 4. Content binding
# ═══════════════════════════════════════════════════════════════════════════

def test_a_mismatched_content_hash_is_refused(gateway):
    recorder = _SendRecorder()
    held = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(recorder),
        arguments={"to_email": "alice@example.com", "subject": "Hi", "body": "Hello."},
        owner_id=OWNER,
    ))

    result = _run(gateway.confirm_and_execute(
        held.data["confirmation_token"],
        owner_id=OWNER,
        content_hash="0" * 64,
    ))

    assert result.is_error is True
    assert recorder.count == 0


def test_changing_the_recipient_produces_a_different_action(gateway):
    """
    Approving a draft to Alice cannot send to Bob: the second action hashes
    differently, and no token was ever issued for it.
    """
    recorder = _SendRecorder()
    to_alice = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(recorder),
        arguments={"to_email": "alice@example.com", "subject": "Hi", "body": "Hello."},
        owner_id=OWNER,
    ))
    to_bob = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(recorder),
        arguments={"to_email": "bob@example.com", "subject": "Hi", "body": "Hello."},
        owner_id=OWNER,
    ))

    assert to_alice.data["content_hash"] != to_bob.data["content_hash"]
    assert to_alice.data["confirmation_token"] != to_bob.data["confirmation_token"]

    # Alice's token cannot approve Bob's action.
    result = _run(gateway.confirm_and_execute(
        to_alice.data["confirmation_token"],
        owner_id=OWNER,
        content_hash=to_bob.data["content_hash"],
    ))
    assert result.is_error is True
    assert recorder.count == 0


def test_the_content_hash_covers_the_preview_text_as_well():
    """Approval is bound to what the user actually read."""
    args = {"to_email": "a@b.com", "subject": "s", "body": "b"}
    one = compute_content_hash("send_email", args, "To: a@b.com")
    two = compute_content_hash("send_email", args, "To: attacker@evil.com")
    assert one != two


def test_the_content_hash_is_stable_for_the_same_action():
    args = {"to_email": "a@b.com", "subject": "s"}
    assert compute_content_hash("send_email", args, "p") == compute_content_hash(
        "send_email", dict(reversed(list(args.items()))), "p"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. User binding
# ═══════════════════════════════════════════════════════════════════════════

def test_another_user_cannot_confirm_your_action(gateway):
    recorder = _SendRecorder()
    held = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(recorder),
        arguments={"to_email": "a@b.com", "subject": "s", "body": "b"},
        owner_id=OWNER,
    ))

    outcome = _run(gateway.confirm(held.data["confirmation_token"], owner_id=INTRUDER))
    assert outcome.approved is False
    assert outcome.denial is Denial.WRONG_USER
    assert recorder.count == 0


def test_an_empty_owner_cannot_confirm(gateway):
    """
    An action prepared without an owner is unconfirmable rather than
    confirmable-by-anyone. Empty-matching-empty would make every unowned action
    approvable by an unauthenticated caller.
    """
    held = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(_SendRecorder()),
        arguments={"to_email": "a@b.com", "subject": "s", "body": "b"},
        owner_id="",
    ))
    outcome = _run(gateway.confirm(held.data["confirmation_token"], owner_id=""))
    assert outcome.approved is False
    assert outcome.denial is Denial.WRONG_USER


def test_only_the_owner_can_cancel(gateway):
    held = _run(gateway.intercept(
        tool="send_email",
        spec=_send_spec(_SendRecorder()),
        arguments={"to_email": "a@b.com", "subject": "s", "body": "b"},
        owner_id=OWNER,
    ))
    token = held.data["confirmation_token"]

    assert _run(gateway.cancel(token, owner_id=INTRUDER)) is False
    assert _run(gateway.cancel(token, owner_id=OWNER)) is True
    assert _run(gateway.confirm(token, owner_id=OWNER)).denial is Denial.UNKNOWN_TOKEN


async def test_pending_actions_are_scoped_to_owner_and_conversation(gateway):
    (await gateway.intercept(
        tool="send_email", spec=_send_spec(_SendRecorder()),
        arguments={"to_email": "a@b.com", "subject": "s", "body": "b"},
        owner_id=OWNER, conversation_id=CONVERSATION,
    ))
    assert len(await gateway.pending_for(CONVERSATION, OWNER)) == 1
    assert await gateway.pending_for(CONVERSATION, INTRUDER) == []
    assert await gateway.pending_for("other-conv", OWNER) == []


# ═══════════════════════════════════════════════════════════════════════════
# 6. Reflect / retry duplication
# ═══════════════════════════════════════════════════════════════════════════

def test_an_already_completed_action_is_not_performed_again(gateway):
    """
    Requirement 10, and the reason `_executed` exists. The reflect loop re-runs
    a specialist after a failure; if that specialist already sent, the second
    run must not send again.
    """
    recorder = _SendRecorder()
    args = {"to_email": "a@b.com", "subject": "s", "body": "b"}

    held = _run(gateway.intercept(
        tool="send_email", spec=_send_spec(recorder), arguments=args, owner_id=OWNER,
    ))
    _run(gateway.confirm_and_execute(held.data["confirmation_token"], owner_id=OWNER))
    assert recorder.count == 1

    # The retry: identical request, same arguments.
    again = _run(gateway.intercept(
        tool="send_email", spec=_send_spec(recorder), arguments=args, owner_id=OWNER,
    ))

    assert again.is_pending is False
    assert again.data.get("already_executed") is True
    assert recorder.count == 1


def test_a_genuinely_different_action_is_still_offered(gateway):
    """The dedupe must not block a second, legitimately different email."""
    recorder = _SendRecorder()
    first = {"to_email": "a@b.com", "subject": "s", "body": "b"}
    second = {"to_email": "c@d.com", "subject": "s", "body": "b"}

    held = _run(gateway.intercept(
        tool="send_email", spec=_send_spec(recorder), arguments=first, owner_id=OWNER,
    ))
    _run(gateway.confirm_and_execute(held.data["confirmation_token"], owner_id=OWNER))

    other = _run(gateway.intercept(
        tool="send_email", spec=_send_spec(recorder), arguments=second, owner_id=OWNER,
    ))
    assert other.is_pending is True


def test_a_stale_token_for_a_completed_action_is_refused(gateway):
    """Two tokens minted for one action; using the second after the first ran."""
    recorder = _SendRecorder()
    args = {"to_email": "a@b.com", "subject": "s", "body": "b"}

    one = _run(gateway.intercept(
        tool="send_email", spec=_send_spec(recorder), arguments=args, owner_id=OWNER))
    two = _run(gateway.intercept(
        tool="send_email", spec=_send_spec(recorder), arguments=args, owner_id=OWNER))

    _run(gateway.confirm_and_execute(one.data["confirmation_token"], owner_id=OWNER))
    outcome = _run(gateway.confirm(two.data["confirmation_token"], owner_id=OWNER))

    assert outcome.approved is False
    assert outcome.denial is Denial.ALREADY_EXECUTED
    assert recorder.count == 1


# ═══════════════════════════════════════════════════════════════════════════
# 7. Preview safety
# ═══════════════════════════════════════════════════════════════════════════

def test_building_a_preview_never_runs_the_tool(gateway):
    """Requirement 11, asserted directly."""
    recorder = _SendRecorder()
    _run(build_preview(
        _send_spec(recorder),
        tool="send_email",
        effect=Effect.EXTERNAL_WRITE,
        arguments={"to_email": "a@b.com"},
    ))
    assert recorder.count == 0


def test_an_invalid_preview_blocks_the_action_entirely(gateway):
    """A precondition failure yields an error, not an unreviewable pending."""
    recorder = _SendRecorder()

    async def reject(args):
        return ActionPreview.invalid("to_email is required.")

    result = _run(gateway.intercept(
        tool="send_email",
        spec={"callable": recorder, "effect": Effect.EXTERNAL_WRITE, "preview": reject},
        arguments={},
        owner_id=OWNER,
    ))

    assert result.is_error is True
    assert "to_email is required" in result.error.message
    assert result.is_pending is False
    assert recorder.count == 0


def test_a_raising_preview_builder_fails_closed(gateway):
    """A broken builder must never degrade into 'no preview, send anyway'."""
    recorder = _SendRecorder()

    async def explode(args):
        raise RuntimeError("db down")

    result = _run(gateway.intercept(
        tool="send_email",
        spec={"callable": recorder, "effect": Effect.EXTERNAL_WRITE, "preview": explode},
        arguments={"to_email": "a@b.com"},
        owner_id=OWNER,
    ))
    assert result.is_error is True
    assert recorder.count == 0


def test_a_tool_with_no_preview_builder_still_gets_a_complete_one(gateway):
    result = _run(gateway.intercept(
        tool="mystery_action",
        spec={"callable": _SendRecorder(), "effect": Effect.DESTRUCTIVE},
        arguments={"target": "everything", "force": True},
        owner_id=OWNER,
    ))
    assert result.is_pending is True
    assert "mystery_action" in result.preview
    assert "target" in result.preview and "everything" in result.preview


def test_a_synchronous_preview_builder_is_supported(gateway):
    result = _run(gateway.intercept(
        tool="send_email",
        spec={"callable": _SendRecorder(), "effect": Effect.EXTERNAL_WRITE,
              "preview": lambda args: f"Send to {args['to_email']}"},
        arguments={"to_email": "a@b.com"},
        owner_id=OWNER,
    ))
    assert result.preview == "Send to a@b.com"


# ═══════════════════════════════════════════════════════════════════════════
# 8. Fail-safe on undeclared effects
# ═══════════════════════════════════════════════════════════════════════════

def test_a_tool_with_no_declared_effect_is_gated(gateway):
    """Requirement 7: an omission must not become permission."""
    recorder = _SendRecorder()
    result = _run(gateway.intercept(
        tool="undeclared_tool",
        spec={"callable": recorder},          # no effect key at all
        arguments={"x": 1},
        owner_id=OWNER,
    ))
    assert result.is_pending is True
    assert recorder.count == 0


def test_a_tool_with_an_unreadable_effect_is_gated(gateway):
    recorder = _SendRecorder()
    result = _run(gateway.intercept(
        tool="weird_tool",
        spec={"callable": recorder, "effect": "SEND_IT_NOW"},
        arguments={"x": 1},
        owner_id=OWNER,
    ))
    assert result.is_pending is True
    assert recorder.count == 0


def test_a_gated_spec_without_a_callable_is_an_error_not_a_pending(gateway):
    result = _run(gateway.intercept(
        tool="broken",
        spec={"effect": Effect.EXTERNAL_WRITE},
        arguments={},
        owner_id=OWNER,
    ))
    assert result.is_error is True


# ═══════════════════════════════════════════════════════════════════════════
# 9. End to end through the real reasoning loop
# ═══════════════════════════════════════════════════════════════════════════

# Shared with the contract and confirmation suites — see tests/support.
_ScriptedAgent = ScriptedAgent


def _state():
    return {
        "user_input": "send it",
        "user_id": OWNER,
        "session_id": CONVERSATION,
        "conversation_history": [],
    }


@pytest.fixture(autouse=True)
def _clean_global_gateway():
    """The loop uses the process-wide gateway; isolate each test."""
    from app.agents.actions import action_gateway
    action_gateway.reset()
    yield
    action_gateway.reset()


def test_the_loop_cannot_send_an_email(gateway):
    """The headline case, through the real `execute_reasoning_loop`."""
    recorder = _SendRecorder()
    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"send_email","tool_input":'
        '{"to_email":"alice@example.com","subject":"Hi","body":"Hello."}}',
        '{"type":"final","content":"I have prepared the email for your approval."}',
    ])

    result = _run(agent.execute_reasoning_loop(
        state=_state(),
        base_system_prompt="p",
        tools={"send_email": _send_spec(recorder)},
        max_iterations=3,
    ))

    assert recorder.count == 0
    assert len(result["pending_actions"]) == 1
    assert result["pending_actions"][0].is_pending is True
    # Nothing ran, so nothing may be reported as evidence.
    assert result["tools_with_evidence"] == []


def test_a_model_hammering_send_email_still_sends_nothing(gateway):
    """
    The adversarial case named in the acceptance criterion: a model that keeps
    asking. Every attempt mints a pending action; none of them delivers.
    """
    recorder = _SendRecorder()
    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"send_email","tool_input":{"to_email":"a@b.com","subject":"1","body":"x"}}',
        '{"type":"tool_call","tool":"send_email","tool_input":{"to_email":"a@b.com","subject":"2","body":"x"}}',
        '{"type":"tool_call","tool":"send_email","tool_input":{"to_email":"a@b.com","subject":"3","body":"x"}}',
        '{"type":"tool_call","tool":"send_email","tool_input":{"to_email":"a@b.com","subject":"4","body":"x"}}',
        '{"type":"final","content":"..."}',
    ])

    result = _run(agent.execute_reasoning_loop(
        state=_state(),
        base_system_prompt="p",
        tools={"send_email": _send_spec(recorder)},
        max_iterations=6,
    ))

    assert recorder.count == 0
    assert len(result["pending_actions"]) >= 1


def test_a_model_claiming_confirmation_in_its_arguments_is_ignored(gateway):
    """
    Confirmation is a token the gateway issued, not a field a model can set.
    Passing `confirmed: true` is just another argument — and one that changes
    the content hash, so it cannot even collide with a real approval.
    """
    recorder = _SendRecorder()
    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"send_email","tool_input":'
        '{"to_email":"a@b.com","subject":"s","body":"b",'
        '"confirmed":true,"user_approved":true,"skip_confirmation":true}}',
        '{"type":"final","content":"..."}',
    ])

    result = _run(agent.execute_reasoning_loop(
        state=_state(), base_system_prompt="p",
        tools={"send_email": _send_spec(recorder)},
    ))

    assert recorder.count == 0
    assert result["pending_actions"][0].is_pending is True


def test_read_tools_are_untouched_by_the_gateway(gateway):
    """Requirement 8: no behaviour change for the 21 read tools."""
    calls = []

    async def get_skills(args):
        calls.append(args)
        return {"success": True, "count": 2, "skills": ["python", "rust"]}

    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"get_skills","tool_input":{}}',
        '{"type":"final","content":"You know Python and Rust."}',
    ])
    result = _run(agent.execute_reasoning_loop(
        state=_state(), base_system_prompt="p",
        tools={"get_skills": {"callable": get_skills, "effect": Effect.READ}},
    ))

    assert len(calls) == 1
    assert result["tools_used"] == ["get_skills"]
    assert result["answerability"] == "ANSWERABLE"
    assert result["pending_actions"] == []


def test_local_writes_execute_without_confirmation(gateway):
    """Requirement 9: drafting is not gated — only sending is."""
    drafted = []

    async def email_draft(args):
        drafted.append(args)
        return {"success": True, "draft": {"subject": "Hi", "body": "Hello"}}

    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"email_draft","tool_input":{"query":"write to alice"}}',
        '{"type":"final","content":"Here is your draft."}',
    ])
    result = _run(agent.execute_reasoning_loop(
        state=_state(), base_system_prompt="p",
        tools={"email_draft": {"callable": email_draft, "effect": Effect.LOCAL_WRITE}},
    ))

    assert len(drafted) == 1
    assert result["pending_actions"] == []
    assert result["answerability"] == "ANSWERABLE"


def test_the_full_draft_then_confirm_then_send_journey(gateway):
    """
    The intended shape end to end: draft freely, send only on approval, and
    exactly one delivery at the end of it.
    """
    from app.agents.actions import action_gateway

    recorder = _SendRecorder()
    drafted = []

    async def email_draft(args):
        drafted.append(args)
        return {"success": True, "draft": {"subject": "Hi", "body": "Hello."}}

    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"email_draft","tool_input":{"query":"write to alice"}}',
        '{"type":"tool_call","tool":"send_email","tool_input":'
        '{"to_email":"alice@example.com","subject":"Hi","body":"Hello."}}',
        '{"type":"final","content":"Ready to send — please confirm."}',
    ])

    result = _run(agent.execute_reasoning_loop(
        state=_state(), base_system_prompt="p",
        tools={
            "email_draft": {"callable": email_draft, "effect": Effect.LOCAL_WRITE},
            "send_email": _send_spec(recorder),
        },
        max_iterations=4,
    ))

    assert len(drafted) == 1          # drafting ran
    assert recorder.count == 0        # sending did not
    pending = result["pending_actions"][0]

    sent = _run(action_gateway.confirm_and_execute(
        pending.data["confirmation_token"], owner_id=OWNER,
    ))
    assert sent.ok is True
    assert recorder.count == 1
    assert recorder.sent[0]["to_email"] == "alice@example.com"


def test_a_second_agent_run_after_a_send_does_not_resend(gateway):
    """
    The reflect loop, end to end: the specialist runs again after the send
    already happened. It must be told so, not repeat it.
    """
    from app.agents.actions import action_gateway

    recorder = _SendRecorder()
    args = '{"to_email":"a@b.com","subject":"s","body":"b"}'

    first = _ScriptedAgent([
        '{"type":"tool_call","tool":"send_email","tool_input":' + args + '}',
        '{"type":"final","content":"awaiting approval"}',
    ])
    held = _run(first.execute_reasoning_loop(
        state=_state(), base_system_prompt="p",
        tools={"send_email": _send_spec(recorder)},
    ))
    _run(action_gateway.confirm_and_execute(
        held["pending_actions"][0].data["confirmation_token"], owner_id=OWNER,
    ))
    assert recorder.count == 1

    retry = _ScriptedAgent([
        '{"type":"tool_call","tool":"send_email","tool_input":' + args + '}',
        '{"type":"final","content":"..."}',
    ])
    result = _run(retry.execute_reasoning_loop(
        state=_state(), base_system_prompt="p",
        tools={"send_email": _send_spec(recorder)},
    ))

    assert recorder.count == 1
    assert result["tool_results"][0].data.get("already_executed") is True
