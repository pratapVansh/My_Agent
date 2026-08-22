"""
Turning remote MCP tools into ordinary registry entries.

The integration point is deliberately unremarkable: an MCP tool becomes a dict
with `description`, `callable`, `effect` and `scope` — the exact shape
`base_agent` already expects — and from that moment on nothing downstream
knows or cares that the call leaves the process.

That is the design goal, not an accident of implementation. Every safety
mechanism in this system is attached to the registry entry rather than to the
tool's identity, so an MCP tool inherits all of them by construction:

    scope filtering       `base_agent._filter_tools_by_scope` removes tools the
                          caller lacks the scope for — a guest never learns an
                          MCP tool exists.

    confirmation          `execute_reasoning_loop` routes any entry whose
                          declared effect requires confirmation through
                          `action_gateway.intercept`. An EXTERNAL_WRITE MCP
                          tool is held with a preview and a content-bound
                          token exactly like `send_email`.

    typed results         the callable returns a `ToolResult`, so NO_DATA and
                          ERROR stay distinguishable and a transport failure
                          cannot read as an empty result.

    grounding             a category's required tools are named locally; an
                          MCP tool counts as evidence only if it actually ran
                          and returned something.

    audit                 the gateway writes one `action_audit` row per
                          consequential execution, keyed by the qualified
                          `mcp__server__tool` name.

The alternative — a parallel "MCP tool" path with its own permission checks —
would mean every future safety property has to be implemented twice, and the
second implementation is the one that gets forgotten.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional

from app.mcp import config as mcp_config
from app.mcp.client import MCPUnavailable, connection_for
from app.mcp.config import MCPServerSpec, MCPToolSpec
from app.tools.contract import ErrorKind, ToolResult

logger = logging.getLogger(__name__)


def _build_callable(
    server: MCPServerSpec, spec: MCPToolSpec, qualified: str
) -> Callable:
    """
    The tool callable handed to an agent.

    Returns a `ToolResult` rather than raising, because a remote server being
    down is an ordinary operating condition and the reasoning loop already
    knows how to describe a failed tool honestly. What it must never do is
    return something that reads as "found nothing".
    """

    async def call_mcp_tool(tool_input: Optional[Mapping[str, Any]] = None) -> ToolResult:
        arguments = dict(tool_input or {})
        connection = connection_for(server)
        try:
            payload = await connection.call(spec.name, arguments)
        except MCPUnavailable as exc:
            logger.warning("MCP call failed (%s): %s", qualified, exc)
            return ToolResult.failed(
                str(exc),
                kind=ErrorKind.TIMEOUT if "timed out" in str(exc) else ErrorKind.EXCEPTION,
                effect=spec.effect,
                retryable=True,
                tool=qualified,
            )
        except Exception as exc:
            logger.exception("Unexpected MCP failure (%s)", qualified)
            return ToolResult.failed(
                f"{type(exc).__name__}: {exc}",
                kind=ErrorKind.EXCEPTION,
                effect=spec.effect,
                tool=qualified,
            )

        # The server reporting its own failure. Treated as an error, not as
        # data — same as a local tool returning {"success": False}.
        if payload.get("is_error"):
            return ToolResult.failed(
                str(payload.get("text") or "the MCP server reported an error"),
                kind=ErrorKind.TOOL_REPORTED,
                effect=spec.effect,
                tool=qualified,
            )

        body = {k: v for k, v in payload.items() if k != "is_error"}
        if not body:
            return ToolResult.no_data(effect=spec.effect, tool=qualified, raw=payload)

        return ToolResult.success(
            body,
            effect=spec.effect,
            tool=qualified,
            raw=payload,
            # A consequential MCP call gets a preview built from the arguments
            # the caller actually supplied. The gateway will not hold an action
            # it cannot describe, and the server's own text is not trusted to
            # describe it.
            preview=(
                _preview(qualified, arguments)
                if spec.effect.requires_confirmation else None
            ),
        )

    call_mcp_tool.__name__ = f"mcp_{server.name}_{spec.name}"
    return call_mcp_tool


def _preview(qualified: str, arguments: Mapping[str, Any]) -> str:
    """
    What the user is shown before approving a consequential MCP call.

    Built from the arguments, locally. A preview assembled from anything the
    server said would let a server describe one action and perform another —
    which is the exact substitution the content hash exists to prevent.
    """
    lines = [f"Run the external tool `{qualified}` with:"]
    if not arguments:
        lines.append("  (no arguments)")
    for key, value in sorted(arguments.items()):
        rendered = str(value)
        if len(rendered) > 300:
            rendered = rendered[:297] + "..."
        lines.append(f"  {key}: {rendered}")
    lines.append("This calls a service outside this application.")
    return "\n".join(lines)


def build_entry(server: MCPServerSpec, spec: MCPToolSpec) -> Dict[str, Any]:
    """One registry entry, in the shape every agent already understands."""
    qualified = mcp_config.qualified_name(server.name, spec.name)
    return {
        # Local text. The server's own description never reaches a prompt —
        # see the module docstring in `app/mcp/config.py`.
        "description": spec.description,
        "callable": _build_callable(server, spec, qualified),
        "effect": spec.effect,
        "scope": spec.scope.value,
        "mcp": {"server": server.name, "tool": spec.name},
    }


async def verify_server(server: MCPServerSpec) -> Dict[str, Any]:
    """
    Compare what a server offers against what it is allowed to offer.

    Run at startup or by an operator. Reports three things worth knowing and
    reports them separately, because they mean different things:

        allowed    configured and present — the working set
        missing    configured but the server does not offer it. A typo, a
                   version drift, or a server that has changed under you.
        offered    present but not configured. Not an error and not
                   registered; it is the queue of things a human may choose to
                   allow. A server growing tools it was never permitted is
                   exactly what an allowlist is for.
    """
    connection = connection_for(server)
    advertised = await connection.list_tools()
    advertised_names = {t["name"] for t in advertised}
    configured = set(server.allowlist)

    return {
        "server": server.name,
        "allowed": sorted(configured & advertised_names),
        "missing": sorted(configured - advertised_names),
        "offered_not_allowed": sorted(advertised_names - configured),
    }


async def registry_for_agent(
    agent_name: str, *, servers: Optional[List[MCPServerSpec]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Every MCP tool this agent is permitted to hold, keyed by qualified name.

    Never raises. A server that will not start costs its own tools and nothing
    else — the agent keeps its internal registry and answers what it can, which
    is the same degradation the memory layer already applies when a store is
    unreachable.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for server in (servers if servers is not None else mcp_config.enabled_servers()):
        for spec in server.tools:
            if agent_name not in spec.exposed_to:
                continue
            out[mcp_config.qualified_name(server.name, spec.name)] = build_entry(
                server, spec
            )
    return out


def confirmable_builders() -> Dict[str, Callable[[str], Dict[str, Any]]]:
    """
    Rebuilders for MCP tools that require confirmation.

    `action_gateway.confirm_and_execute` reconstructs a held action from its
    *name*, deliberately taking nothing executable from the stored row. MCP
    tools have to participate in that or an approved MCP action could never
    run. The rebuild goes through the same local config as the original
    registration, so a tampered row can still only name one of the tools this
    application already permits.
    """
    builders: Dict[str, Callable[[str], Dict[str, Any]]] = {}
    for server in mcp_config.enabled_servers():
        for spec in server.tools:
            if not spec.effect.requires_confirmation:
                continue
            qualified = mcp_config.qualified_name(server.name, spec.name)

            def make(_server=server, _spec=spec):
                def build(_owner_id: str) -> Dict[str, Any]:
                    return build_entry(_server, _spec)
                return build

            builders[qualified] = make()
    return builders


__all__ = [
    "build_entry",
    "confirmable_builders",
    "registry_for_agent",
    "verify_server",
]
