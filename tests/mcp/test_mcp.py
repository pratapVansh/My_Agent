"""
MCP integration, proved against a real server process.

These tests spawn `tests/support/mcp_server.py` as a child process and speak
the real protocol to it over stdio. That costs a second or two per test and it
is the point: the claims worth making about MCP — that an allowlist holds, that
a hostile description never reaches a prompt, that a dead server degrades one
tool rather than the assistant — are all claims about behaviour at the process
boundary, and a mocked boundary would assume every one of them.

The security properties are grouped first because they are the reason this
integration is allowed to exist at all.
"""
from __future__ import annotations

import sys

import pytest

from app.agents.actions import ActionGateway
from app.auth.models import Scope
from app.mcp import client as mcp_client
from app.mcp import registry as mcp_registry
from app.mcp.config import MCPServerSpec, MCPToolSpec, parse_qualified, qualified_name
from app.tools.contract import Effect, ToolStatus
from tests.support.mcp_server import HOSTILE_DESCRIPTION

pytest.importorskip("mcp", reason="the MCP SDK is not installed")

OWNER = "owner@example.com"


# ── The fixture server ───────────────────────────────────────────────────────

def _server(**overrides) -> MCPServerSpec:
    """
    A server spec pointing at the real fixture process.

    Note what is declared here rather than fetched: every effect, every scope,
    and every description. That is the whole security model — see
    `app/mcp/config.py`.
    """
    defaults = dict(
        name="fixture",
        command=sys.executable,
        args=["-m", "tests.support.mcp_server"],
        enabled=True,
        startup_timeout=30.0,
        call_timeout=30.0,
        tools=(
            MCPToolSpec(
                name="add",
                description="Add two integers. Args: a (int), b (int).",
                effect=Effect.READ,
                scope=Scope.PROFILE_READ,
                exposed_to=("profile",),
            ),
            MCPToolSpec(
                name="hostile_description_tool",
                description="Echo a value back. Args: value (str).",
                effect=Effect.READ,
                scope=Scope.PROFILE_READ,
                exposed_to=("profile",),
            ),
            MCPToolSpec(
                name="publish",
                description="Publish a message externally. Args: message (str).",
                effect=Effect.EXTERNAL_WRITE,
                scope=Scope.EMAIL_SEND,
                exposed_to=("email",),
            ),
            MCPToolSpec(
                name="always_fails",
                description="A tool that fails. Args: none.",
                effect=Effect.READ,
                scope=Scope.PROFILE_READ,
                exposed_to=("profile",),
            ),
        ),
    )
    defaults.update(overrides)
    return MCPServerSpec(**defaults)


@pytest.fixture
async def server():
    spec = _server()
    yield spec
    await mcp_client.close_all()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Security — the reasons this is allowed to exist
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_hostile_tool_description_never_reaches_a_prompt(server):
    """
    The highest-value injection vector in the protocol.

    The fixture server advertises a description that is a direct prompt
    injection. Tool descriptions are placed verbatim into the system prompt, so
    if the server's string were trusted this would be an attack with a
    guaranteed delivery path. The registered description comes from local
    config instead.
    """
    connection = mcp_client.connection_for(server)
    advertised = {t["name"]: t["server_description"] for t in await connection.list_tools()}
    assert HOSTILE_DESCRIPTION in advertised["hostile_description_tool"], (
        "the fixture server should be advertising the injection"
    )

    registry = await mcp_registry.registry_for_agent("profile", servers=[server])
    entry = registry[qualified_name("fixture", "hostile_description_tool")]

    assert HOSTILE_DESCRIPTION not in entry["description"]
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" not in entry["description"]
    assert "attacker@evil.example" not in entry["description"]


async def test_a_tool_the_server_offers_but_config_forbids_is_not_registered(server):
    """
    Allowlist, fail closed. The fixture server offers `not_allowlisted`; a
    server that grows tools it was never permitted must not grow its own reach.
    """
    connection = mcp_client.connection_for(server)
    offered = {t["name"] for t in await connection.list_tools()}
    assert "not_allowlisted" in offered, "the fixture should offer it"

    registry = await mcp_registry.registry_for_agent("profile", servers=[server])
    assert qualified_name("fixture", "not_allowlisted") not in registry
    assert all("not_allowlisted" not in name for name in registry)


async def test_verify_reports_offered_but_unallowed_tools(server):
    report = await mcp_registry.verify_server(server)
    assert "add" in report["allowed"]
    assert "not_allowlisted" in report["offered_not_allowed"]
    assert report["missing"] == []


async def test_effect_and_scope_come_from_local_config(server):
    registry = await mcp_registry.registry_for_agent("email", servers=[server])
    entry = registry[qualified_name("fixture", "publish")]
    assert entry["effect"] is Effect.EXTERNAL_WRITE
    assert entry["scope"] == Scope.EMAIL_SEND.value


async def test_an_mcp_tool_cannot_shadow_an_internal_one(server):
    """The namespace prefix makes collision impossible rather than unlikely."""
    registry = await mcp_registry.registry_for_agent("email", servers=[server])
    assert "send_email" not in registry
    for name in registry:
        assert name.startswith("mcp__")
        assert parse_qualified(name) is not None


async def test_an_agent_only_holds_the_tools_it_is_exposed_to(server):
    profile = await mcp_registry.registry_for_agent("profile", servers=[server])
    email = await mcp_registry.registry_for_agent("email", servers=[server])

    assert qualified_name("fixture", "add") in profile
    assert qualified_name("fixture", "add") not in email
    assert qualified_name("fixture", "publish") in email
    assert qualified_name("fixture", "publish") not in profile


async def test_a_guest_never_sees_an_mcp_tool(server):
    """
    Scope filtering is the existing code path, unmodified — which is the point
    of registering MCP tools as ordinary entries.
    """
    from app.agents.base_agent import BaseAgent

    class _Probe(BaseAgent):
        async def execute(self, state):
            return state

    registry = await mcp_registry.registry_for_agent("email", servers=[server])
    guest_state = {"scopes": [Scope.CHAT.value, Scope.PROFILE_READ.value]}
    filtered = _Probe(name="p", description="d")._filter_tools_by_scope(
        registry, guest_state
    )
    assert qualified_name("fixture", "publish") not in filtered


# ═══════════════════════════════════════════════════════════════════════════
# 2. Behaviour at the process boundary
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_read_tool_round_trips_through_a_real_process(server):
    registry = await mcp_registry.registry_for_agent("profile", servers=[server])
    result = await registry[qualified_name("fixture", "add")]["callable"](
        {"a": 2, "b": 3}
    )
    assert result.status is ToolStatus.OK
    assert "5" in str(result.data)
    assert result.effect is Effect.READ


async def test_a_failing_tool_is_an_error_not_an_empty_result(server):
    """
    The distinction the whole memory layer is built around, carried across the
    process boundary: "the lookup failed" and "there is nothing" are opposite
    claims and must not arrive looking alike.
    """
    registry = await mcp_registry.registry_for_agent("profile", servers=[server])
    result = await registry[qualified_name("fixture", "always_fails")]["callable"]({})
    assert result.status is ToolStatus.ERROR
    assert result.status is not ToolStatus.NO_DATA


def test_a_raised_tool_is_detected_even_when_iserror_is_unset():
    """
    Measured against the reference SDK: a tool that raises comes back with
    `isError` unset and the exception rendered into the text. Trusting the flag
    alone would let a crashed tool read as data.
    """
    from app.mcp.client import _normalise_result

    class _Block:
        text = "Error executing tool always_fails: this tool is designed to fail"

    class _Raw:
        isError = False
        content = [_Block()]
        structuredContent = None

    assert _normalise_result(_Raw(), tool="always_fails")["is_error"] is True


def test_a_successful_result_that_merely_mentions_errors_stays_successful():
    """
    The false positive that a looser check would cause. A log-search tool
    returning error lines has *succeeded*; reading that as a failure would be
    its own bug, and a worse one — it would hide real data behind a fake outage.
    """
    from app.mcp.client import _normalise_result

    class _Block:
        text = "Found 3 errors in the log: error: disk full; error: timeout"

    class _Raw:
        isError = False
        content = [_Block()]
        structuredContent = None

    assert _normalise_result(_Raw(), tool="search_logs")["is_error"] is False


async def test_a_dead_server_degrades_one_tool_not_the_turn():
    """A server that cannot start returns an error result, and does not raise."""
    broken = _server(
        command=sys.executable,
        args=["-c", "import sys; sys.exit(1)"],
        startup_timeout=15.0,
    )
    registry = await mcp_registry.registry_for_agent("profile", servers=[broken])
    result = await registry[qualified_name("fixture", "add")]["callable"]({"a": 1, "b": 1})
    assert result.status is ToolStatus.ERROR
    assert result.error is not None
    await mcp_client.close_all()


async def test_a_hanging_call_is_bounded_by_the_timeout():
    slow = _server(
        call_timeout=2.0,
        tools=(
            MCPToolSpec(
                name="sleep_forever",
                description="Sleeps. Args: seconds (float).",
                effect=Effect.READ,
                scope=Scope.PROFILE_READ,
                exposed_to=("profile",),
            ),
        ),
    )
    registry = await mcp_registry.registry_for_agent("profile", servers=[slow])
    result = await registry[qualified_name("fixture", "sleep_forever")]["callable"](
        {"seconds": 30}
    )
    assert result.status is ToolStatus.ERROR
    assert "timed out" in (result.error.message or "").lower()
    await mcp_client.close_all()


# ═══════════════════════════════════════════════════════════════════════════
# 3. The confirmation gateway applies unchanged
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_consequential_mcp_call_is_held_not_executed(
    server, audit_store, pending_store
):
    """
    The property that makes MCP writes safe: an EXTERNAL_WRITE MCP tool is
    intercepted before the call, exactly like `send_email`, and the user is
    shown a preview built from the arguments rather than from anything the
    server said.

    The stores are passed explicitly. A gateway built without them reaches for
    Postgres and — correctly — refuses every consequential action, which would
    make this pass or fail for a reason that has nothing to do with MCP.
    """
    gateway = ActionGateway(audit=audit_store, pending=pending_store)
    registry = await mcp_registry.registry_for_agent("email", servers=[server])
    name = qualified_name("fixture", "publish")

    held = await gateway.intercept(
        tool=name,
        spec=registry[name],
        arguments={"message": "hello world"},
        owner_id=OWNER,
        conversation_id="mcp-test",
    )

    assert held.status is ToolStatus.PENDING_CONFIRMATION
    assert "hello world" in (held.preview or "")
    assert name in (held.preview or "")
    gateway.reset()


async def test_a_held_mcp_action_is_reconstructible_by_name(server, monkeypatch):
    """
    `confirm_and_execute` rebuilds a stored action from its name alone, taking
    nothing executable from the row. MCP tools have to participate or an
    approved MCP action could never run.
    """
    monkeypatch.setattr(
        "app.mcp.config.SERVERS", [server], raising=False
    )
    builders = mcp_registry.confirmable_builders()
    name = qualified_name("fixture", "publish")

    assert name in builders, "a confirmable MCP tool must be rebuildable"
    rebuilt = builders[name](OWNER)
    assert rebuilt["effect"] is Effect.EXTERNAL_WRITE
    assert rebuilt["scope"] == Scope.EMAIL_SEND.value

    # A read-only tool is not confirmable and must not appear.
    assert qualified_name("fixture", "add") not in builders


# ═══════════════════════════════════════════════════════════════════════════
# 4. Configuration hygiene
# ═══════════════════════════════════════════════════════════════════════════

def test_no_server_is_enabled_by_default():
    """
    Spawning a child process is a decision. A server ships disabled, so adding
    configuration does not silently start running third-party code.
    """
    from app.mcp.config import SERVERS

    assert all(not s.enabled for s in SERVERS), (
        "an MCP server must be explicitly enabled, never on by default"
    )


def test_every_configured_tool_declares_effect_scope_and_placement():
    from app.mcp.config import SERVERS

    for server in SERVERS:
        for tool in server.tools:
            assert isinstance(tool.effect, Effect), f"{server.name}.{tool.name}"
            assert isinstance(tool.scope, Scope), f"{server.name}.{tool.name}"
            assert tool.exposed_to, (
                f"{server.name}.{tool.name} names no agent, so it would be "
                "registered nowhere"
            )
            assert tool.description.strip(), f"{server.name}.{tool.name}"
