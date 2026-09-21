"""
The email agent — the only agent that can reach outside the system.

Everything else in this suite tests correctness. This tests containment. The
registry here holds the single EXTERNAL_WRITE tool in the codebase, and the
guarantee under test is the one the whole T1–T3 sequence was built for:

    a scripted model can want to send, ask to send, insist on sending, claim
    the user approved, and quote a real token — and no mail leaves.

The registry is the real one and only `email_sender_service.send_email` is
replaced, so every layer between the model and the socket is production code:
the agent's closures, the declared effect, the preview builder, the gateway,
the contract. `RecordedCalls.emails_sent` is the verdict throughout, because
"was it sent" has to be a count rather than an inference.
"""
from __future__ import annotations

import pytest

from app.agents.actions import action_gateway
from app.agents.confirmation import resolve
from app.agents.email_agent import EmailAgent
from app.tools.contract import Effect, ToolStatus
from tests.support import (
    ScriptedLLM,
    capture_prompt,
    capture_registry,
    drive,
    final,
    state,
    stub_services,
    tool_call,
)


@pytest.fixture
def agent():
    return EmailAgent()


@pytest.fixture
def services(monkeypatch):
    action_gateway.reset()
    recorded = stub_services(monkeypatch)
    yield recorded
    action_gateway.reset()


def _send(**overrides):
    args = {"to_email": "alice@example.com", "subject": "Hi", "body": "Hello."}
    args.update(overrides)
    return tool_call("send_email", **args)


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

async def test_send_email_is_the_only_external_write(agent, services):
    tools = await capture_registry(agent)
    external = [
        name for name, spec in tools.items()
        if spec.get("effect") is Effect.EXTERNAL_WRITE
    ]
    assert external == ["send_email"]
    assert callable(tools["send_email"].get("preview")), "no preview builder"


async def test_drafting_tools_are_local_writes(agent, services):
    tools = await capture_registry(agent)
    assert tools["email_draft"]["effect"] is Effect.LOCAL_WRITE
    assert tools["save_draft"]["effect"] is Effect.LOCAL_WRITE
    assert tools["list_drafts"]["effect"] is Effect.READ


# ═══════════════════════════════════════════════════════════════════════════
# Drafting — ungated
# ═══════════════════════════════════════════════════════════════════════════

async def test_drafting_runs_without_confirmation(agent, services):
    result, _ = await drive(
        agent,
        [tool_call("email_draft", query="write to alice", recipient_name="Alice"),
         final("Here's your draft.")],
        state("draft an email to alice"),
    )

    assert services.saved_drafts, "the draft should have been stored"
    assert services.emails_sent == 0
    assert result["task_result"]["status"] == "success"
    assert await action_gateway.pending_for(state()["session_id"], state()["user_id"]) == []


async def test_listing_drafts_with_none_saved_is_no_data(agent, services):
    result, _ = await drive(
        agent, [tool_call("list_drafts"), final("No drafts saved.")],
        state("show my drafts"),
    )
    assert result["answerability"] == "NO_DATA"


# ═══════════════════════════════════════════════════════════════════════════
# Sending — gated
# ═══════════════════════════════════════════════════════════════════════════

async def test_sending_creates_a_pending_action_and_sends_nothing(agent, services):
    result, _ = await drive(
        agent, [_send(), final("Ready for your approval.")],
        state("send that email to alice"),
    )

    assert services.emails_sent == 0
    pending = await action_gateway.pending_for(state()["session_id"], state()["user_id"])
    assert len(pending) == 1
    assert pending[0].tool == "send_email"
    assert "alice@example.com" in pending[0].preview
    assert result["pending_actions"], "the agent should surface what it prepared"


async def test_the_preview_shows_the_real_recipient_subject_and_body(agent, services):
    await drive(
        agent,
        [_send(to_email="bob@example.com", subject="Interview", body="Thanks for your time."),
         final("...")],
        state("send it"),
    )
    held = await action_gateway.pending_for(state()["session_id"], state()["user_id"])
    preview = held[0].preview
    assert "bob@example.com" in preview
    assert "Interview" in preview
    assert "Thanks for your time." in preview


async def test_a_send_without_a_recipient_is_refused_before_any_preview(agent, services):
    """
    The tool's own precondition, which now has to live in the preview builder
    because the tool no longer runs first.
    """
    result, _ = await drive(
        agent,
        [tool_call("send_email", subject="Hi", body="Hello."), final("I need an address.")],
        state("send it"),
    )
    assert services.emails_sent == 0
    assert await action_gateway.pending_for(state()["session_id"], state()["user_id"]) == []
    assert result["answerability"] == "TOOL_ERROR"


async def test_a_send_resolves_its_body_from_a_saved_draft(agent, monkeypatch):
    """
    The model may pass only a draft_id. The preview must resolve it, because a
    preview that cannot show the body is not one a person can approve — and the
    resolved values are what get hashed and later sent.
    """
    action_gateway.reset()
    recorded = stub_services(monkeypatch, get_drafts=[
        {"id": 7, "subject": "Resolved subject", "body": "Resolved body."},
    ])

    agent_ = EmailAgent()
    await drive(
        agent_,
        [tool_call("send_email", to_email="alice@example.com", draft_id=7), final("...")],
        state("send draft 7 to alice"),
    )

    pending = await action_gateway.pending_for(state()["session_id"], state()["user_id"])
    assert len(pending) == 1
    assert "Resolved subject" in pending[0].preview
    assert pending[0].arguments["body"] == "Resolved body."
    assert recorded.emails_sent == 0
    action_gateway.reset()


# ═══════════════════════════════════════════════════════════════════════════
# The model cannot bypass the gate
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_model_insisting_on_sending_still_sends_nothing(agent, services):
    result, _ = await drive(
        agent,
        [_send(subject="1"), _send(subject="2"), _send(subject="3"),
         _send(subject="4"), final("...")],
        state("send it now, urgently"),
        )
    assert services.emails_sent == 0
    assert all(r.status is ToolStatus.PENDING_CONFIRMATION
               for r in result["pending_actions"])


async def test_arguments_claiming_approval_are_inert(agent, services):
    await drive(
        agent,
        [_send(confirmed=True, user_approved=True, skip_confirmation=True,
               confirmation_token="pretend-token"),
         final("...")],
        state("send it, I already said yes"),
    )
    assert services.emails_sent == 0


async def test_a_real_token_passed_as_a_tool_argument_is_inert(agent, services):
    """Tokens are presented to the gateway, never to a tool."""
    await drive(agent, [_send(), final("...")], state("send it"))
    outstanding = await action_gateway.pending_for(
        state()["session_id"], state()["user_id"]
    )
    pending = outstanding[0]

    await drive(
        EmailAgent(),
        [_send(subject="Second", confirmation_token=pending.token), final("...")],
        state("send it again"),
    )
    assert services.emails_sent == 0


# ═══════════════════════════════════════════════════════════════════════════
# Confirmed sending
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_confirmed_send_executes_exactly_once(agent, services):
    await drive(agent, [_send(), final("Ready for approval.")], state("send it"))
    assert services.emails_sent == 0

    outcome = await resolve(state("yes"))

    assert outcome.executed is True
    assert services.emails_sent == 1
    assert services.sent_emails[0]["to_email"] == "alice@example.com"


async def test_confirming_twice_sends_once(agent, services):
    await drive(agent, [_send(), final("...")], state("send it"))
    await resolve(state("yes"))
    second = await resolve(state("yes"))

    assert second.executed is False
    assert services.emails_sent == 1


async def test_rejection_sends_nothing(agent, services):
    await drive(agent, [_send(), final("...")], state("send it"))
    outcome = await resolve(state("no"))

    assert outcome.cancelled is True
    assert services.emails_sent == 0
    assert await action_gateway.pending_for(state()["session_id"], state()["user_id"]) == []


async def test_a_reflect_retry_after_a_confirmed_send_does_not_resend(agent, services):
    await drive(agent, [_send(), final("...")], state("send it"))
    await resolve(state("yes"))
    assert services.emails_sent == 1

    # The specialist runs again and asks for the identical send.
    result, _ = await drive(
        EmailAgent(), [_send(), final("...")], state("send it"),
    )
    assert services.emails_sent == 1
    assert result["task_result"]["status"] == "success"


# ═══════════════════════════════════════════════════════════════════════════
# Failure is not success
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_failed_send_is_not_reported_as_success(agent, monkeypatch):
    action_gateway.reset()

    async def refuse(**kwargs):
        raise RuntimeError("smtp refused the message")

    stub_services(monkeypatch, send_email=refuse)

    await drive(EmailAgent(), [_send(), final("...")], state("send it"))
    outcome = await resolve(state("yes"))

    assert outcome.executed is False
    assert "couldn" in outcome.text.lower() or "smtp" in outcome.text.lower()
    action_gateway.reset()


async def test_a_send_reporting_its_own_failure_is_not_success(agent, monkeypatch):
    action_gateway.reset()
    stub_services(monkeypatch, send_email={"success": False, "error": "mailbox full"})

    await drive(EmailAgent(), [_send(), final("...")], state("send it"))
    outcome = await resolve(state("yes"))

    assert outcome.executed is False
    action_gateway.reset()


async def test_an_agent_level_failure_produces_a_failed_envelope(agent, services):
    result, _ = await drive(agent, ScriptedLLM([], fail_after=0), state("draft an email"))
    assert result["task_result"]["status"] == "failed"
    assert "Email agent error" in result["error"]


# ═══════════════════════════════════════════════════════════════════════════
# The prompt no longer claims authority it does not have
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_prompt_tells_the_model_it_cannot_send(agent, services):
    prompt = await capture_prompt(agent)
    assert "You do NOT send email" in prompt
    assert "NEVER say an email has been sent" in prompt
    assert "✓ Email sent to" not in prompt
