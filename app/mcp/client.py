"""
Talking to an MCP server process, and surviving it misbehaving.

One `MCPConnection` per configured server, each owning a child process spoken
to over stdio. The connection is lazy: nothing is spawned until a tool from
that server is actually called, so a configured-but-unused server costs
nothing.

Everything here treats the server as untrusted *and* unreliable, which are
different problems with different fixes:

  untrusted   its results are data, never instructions. Handled by the
              contract coercion in `registry.py` and by never letting the
              server's own strings into a prompt (`config.py`).

  unreliable  it can hang, die mid-call, or never start. Every call is bounded
              by a timeout, a dead connection is torn down rather than reused,
              and a failure becomes a `ToolResult` error rather than an
              exception that kills the turn. A broken MCP server degrades one
              tool, not the assistant.

The distinction the memory layer insists on applies here too: a tool call that
*failed* is not a tool call that found *nothing*. A timeout against a calendar
server must never become "you have no meetings".
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from app.mcp.config import MCPServerSpec

logger = logging.getLogger(__name__)


class MCPUnavailable(RuntimeError):
    """The server could not be reached or did not answer in time."""


class MCPConnection:
    """A lazily-started stdio connection to one MCP server."""

    def __init__(self, spec: MCPServerSpec):
        self.spec = spec
        self._session: Any = None
        self._stack: Optional[AsyncExitStack] = None
        self._lock = asyncio.Lock()
        self._advertised: Dict[str, str] = {}
        """Tool name → the server's own description. Kept for diagnostics
        only; never placed in a prompt. See `config.py`."""

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        """
        Start the server process and initialise the session.

        Guarded by a lock so two concurrent tool calls cannot spawn two
        processes — the same race the Deepgram bridge had, with the same fix.
        """
        async with self._lock:
            if self._session is not None:
                return
            await self._connect_locked()

    async def _connect_locked(self) -> None:
        # Imported here rather than at module scope so that the MCP SDK is a
        # dependency only of deployments that actually configure a server.
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        stack = AsyncExitStack()
        try:
            params = StdioServerParameters(
                command=self.spec.command,
                args=list(self.spec.args),
                env=dict(self.spec.env) if self.spec.env else None,
            )
            read, write = await asyncio.wait_for(
                stack.enter_async_context(stdio_client(params)),
                timeout=self.spec.startup_timeout,
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(
                session.initialize(), timeout=self.spec.startup_timeout
            )
        except Exception as exc:
            await _quiet_aclose(stack)
            raise MCPUnavailable(
                f"MCP server '{self.spec.name}' failed to start: {exc}"
            ) from exc

        self._stack = stack
        self._session = session
        logger.info("MCP server '%s' connected", self.spec.name)

    async def close(self) -> None:
        async with self._lock:
            await self._close_locked()

    async def _close_locked(self) -> None:
        stack, self._stack = self._stack, None
        self._session = None
        if stack is not None:
            await _quiet_aclose(stack)
            logger.info("MCP server '%s' disconnected", self.spec.name)

    # ── Use ──────────────────────────────────────────────────────────────────

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        What the server says it offers.

        Advertising is not permission — `registry.py` intersects this with the
        local allowlist. This exists so an operator can see what a server
        offers and decide what to allow, and so a tool named in config but
        missing from the server is a loud warning rather than a mystery.
        """
        await self.connect()
        try:
            listing = await asyncio.wait_for(
                self._session.list_tools(), timeout=self.spec.call_timeout
            )
        except Exception as exc:
            await self.close()
            raise MCPUnavailable(
                f"MCP server '{self.spec.name}' failed to list tools: {exc}"
            ) from exc

        out: List[Dict[str, Any]] = []
        for tool in getattr(listing, "tools", []) or []:
            name = getattr(tool, "name", "")
            description = getattr(tool, "description", "") or ""
            self._advertised[name] = description
            out.append({
                "name": name,
                "server_description": description,
                "input_schema": getattr(tool, "inputSchema", None),
            })
        return out

    async def call(self, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a tool and return a plain dict.

        Raises `MCPUnavailable` on transport failure — never returns something
        that could be mistaken for an empty result. A caller that cannot tell a
        dead server from an empty calendar will eventually tell a user they are
        free when they are not.
        """
        await self.connect()
        try:
            raw = await asyncio.wait_for(
                self._session.call_tool(tool, arguments or {}),
                timeout=self.spec.call_timeout,
            )
        except asyncio.TimeoutError as exc:
            await self.close()
            raise MCPUnavailable(
                f"MCP tool '{tool}' on '{self.spec.name}' timed out after "
                f"{self.spec.call_timeout:.0f}s"
            ) from exc
        except Exception as exc:
            await self.close()
            raise MCPUnavailable(
                f"MCP tool '{tool}' on '{self.spec.name}' failed: {exc}"
            ) from exc

        return _normalise_result(raw, tool=tool)


async def _quiet_aclose(stack: AsyncExitStack) -> None:
    """
    Close a stack without letting teardown noise replace the real error.

    An MCP server that died mid-session frequently raises again on close, and
    that second exception is never the interesting one.
    """
    try:
        await stack.aclose()
    except Exception as exc:
        logger.debug("MCP teardown raised (ignored): %s", exc)


def _normalise_result(raw: Any, *, tool: str = "") -> Dict[str, Any]:
    """
    Flatten an MCP `CallToolResult` into a plain dict.

    `isError` is honoured as the server reporting its own failure, which the
    contract layer then reads as an error rather than as data — the same
    treatment a local tool returning `{"success": False}` gets.

    **`isError` alone is not sufficient, and that was measured rather than
    assumed.** Against the reference SDK (v2), a tool that *raises* comes back
    with `isError` unset and the exception rendered into the text content as
    ``Error executing tool <name>: ...``. Trusting the flag by itself would
    make a crashed tool indistinguishable from a successful one whose content
    happens to describe a failure — and the model would then read "Error
    executing tool" as data and answer from it.

    So the SDK's own error envelope is recognised as a second signal. The match
    is deliberately narrow: anchored at the start, and it must name the tool
    that was actually called. Sniffing for the word "error" anywhere in a
    result would be the wrong fix — a log-search tool returning error lines is
    a *successful* call, and misreading that as a failure is its own bug.

    A server that returns exactly this SDK-generated prefix while having
    succeeded is lying about its own failure, which no client-side check can
    or should try to unpick.
    """
    is_error = bool(getattr(raw, "isError", False))

    texts: List[str] = []
    structured: Any = getattr(raw, "structuredContent", None)

    for block in (getattr(raw, "content", None) or []):
        text = getattr(block, "text", None)
        if text:
            texts.append(str(text))

    joined = "\n".join(texts)
    if not is_error and tool and joined.startswith(f"Error executing tool {tool}:"):
        is_error = True

    payload: Dict[str, Any] = {"is_error": is_error}
    if structured is not None:
        payload["structured"] = structured
    if joined:
        payload["text"] = joined
    return payload


# ── Connection registry ──────────────────────────────────────────────────────

_CONNECTIONS: Dict[str, MCPConnection] = {}


def connection_for(spec: MCPServerSpec) -> MCPConnection:
    existing = _CONNECTIONS.get(spec.name)
    if existing is not None and existing.spec is spec:
        return existing
    connection = MCPConnection(spec)
    _CONNECTIONS[spec.name] = connection
    return connection


async def close_all() -> None:
    """Shut every server down. Called from the application's lifespan."""
    for connection in list(_CONNECTIONS.values()):
        await connection.close()
    _CONNECTIONS.clear()


__all__ = [
    "MCPConnection",
    "MCPUnavailable",
    "close_all",
    "connection_for",
]
