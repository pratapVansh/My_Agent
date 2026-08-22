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
