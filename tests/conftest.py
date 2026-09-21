"""
Shared fixtures.

One thing here is load-bearing rather than convenient. The action gateway now
fails closed on an audit error: if the record cannot be written, a consequential
action does not run. In production that record is Postgres. In a unit test there
is no Postgres, so without this fixture every EXTERNAL_WRITE test would pass for
the wrong reason — nothing sent, because the database was unreachable, rather
than because the gate held.

So the suite installs an in-memory audit log on the shared gateway. It enforces
the same unique-executed constraint the Postgres partial index does, which keeps
the tests honest about the guarantee they are asserting.

Production is unaffected: `app.domain.audit.audit_log` remains the Postgres
implementation, and nothing here touches it.
"""
from __future__ import annotations

import pytest

from app.agents.actions import action_gateway
from app.domain.audit import InMemoryAuditLog
from app.domain.pending_actions import InMemoryPendingActionStore


@pytest.fixture(autouse=True)
def audit_store(pending_store):
    """A fresh in-memory audit record for every test, on the shared gateway."""
    store = InMemoryAuditLog()
    previous = action_gateway.audit
    action_gateway.use_audit_log(store)
    try:
        yield store
    finally:
        action_gateway.use_audit_log(previous)


@pytest.fixture(autouse=True)
def confirmable_registry():
    """
    Restore the real confirmable-tool registry after every test.

    Tests point a name at a double via `tests.support.register_confirmable`;
    this makes that override strictly test-scoped, so one test's fake
    `send_email` can never be the thing another test — or production — resolves.
    """
    from app.agents import confirmable_tools

    original = dict(confirmable_tools.CONFIRMABLE_TOOLS)
    try:
        yield confirmable_tools.CONFIRMABLE_TOOLS
    finally:
        confirmable_tools.CONFIRMABLE_TOOLS.clear()
        confirmable_tools.CONFIRMABLE_TOOLS.update(original)


@pytest.fixture(autouse=True)
def pending_store():
    """
    A fresh in-memory waiting room for every test.

    Both stores must be swapped, not just the audit one: the gateway now keeps
    outstanding actions in Postgres too, so a test left pointing at the real
    store would refuse every confirmation because the database is unreachable —
    passing for the wrong reason.
    """
    store = InMemoryPendingActionStore()
    previous = action_gateway.pending_store
    action_gateway.use_pending_store(store)
    try:
        yield store
    finally:
        store.reset()
        action_gateway.use_pending_store(previous)


class SMTPContactedInTests(BaseException):
    """
    A test opened a real SMTP connection. See `_no_real_smtp`.

    Deliberately a `BaseException`. `EmailSenderService.send_email` ends in a
    bare `except Exception`, which would otherwise catch this and turn a test
    that reached a mail server into a quiet `{"success": False}` — the failure
    would be contained but invisible, which is how the original bug survived.
    Inheriting from `BaseException` puts it past every ordinary handler in the
    application, so the test fails loudly and names the cause.
    """


@pytest.fixture(scope="session", autouse=True)
def _no_real_smtp():
    """
    Make the suite's offline guarantee structural rather than a convention.

    Every email test stubs the send — at `_send_sync`, at the tool callable, or
    through `register_confirmable`. Each of those is a stub a test has to
    remember, and one test forgot: `test_a_token_is_consumed_even_when_execution
    _fails` passed a failing callable in a spec, but the gateway rebuilds the
    callable from the tool *name*, so the **real** `send_email` ran. With
    placeholder credentials the real send returned an error and the assertion
    held for the wrong reason. Once `.env` held working credentials the same
    test sent live mail to `a@b.com` and then failed.

    So the guarantee is enforced one layer below every one of those doubles, at
    the socket boundary itself. Nothing in the suite constructs `smtplib.SMTP`,
    so replacing it costs nothing and cannot mask a stub that is working; it
    only fires for a path that would otherwise have reached a mail server.

    Session-scoped and autouse: a guard a test can opt out of is not a guard.
    """
    import smtplib

    def _refuse(kind):
        def _blocked(*args, **kwargs):
            host = args[0] if args else kwargs.get("host", "?")
            raise SMTPContactedInTests(
                f"This test tried to open a real SMTP connection "
                f"({kind} to {host!r}). The suite runs offline.\n"
                f"Stub the send instead — `_send_sync` for the service, or "
                f"`tests.support.register_confirmable` when the gateway will "
                f"resolve `send_email` by name."
            )
        return _blocked

    originals = {name: getattr(smtplib, name) for name in ("SMTP", "SMTP_SSL")}
    for name in originals:
        setattr(smtplib, name, _refuse(name))
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(smtplib, name, value)
