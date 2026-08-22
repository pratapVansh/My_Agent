"""
MCP support: third-party tools, under the same rules as internal ones.

MCP earns its place for external services with real auth surfaces and
independently versioned APIs — GitHub, Gmail, Calendar, web search. It is the
wrong tool for anything reading this application's own memory, which never
leaves the process and would gain only a serialization boundary and an
exfiltration path.

Architecturally the host sits *behind* the capability registry and *under* the
action gateway, never beside them. The three modules split along the line that
matters:

    config     what exists and what it may do — declared locally, the
               security boundary
    client     one child process per server, treated as untrusted and
               unreliable
    registry   the adapter that makes a remote tool indistinguishable from a
               local one, so every existing guarantee applies unchanged

Read `config.py` first; it explains why nothing the server says about itself is
believed.
"""
from app.mcp.client import MCPConnection, MCPUnavailable, close_all, connection_for
from app.mcp.config import (
    MCPServerSpec,
    MCPToolSpec,
    MCP_PREFIX,
    SERVERS,
    enabled_servers,
    find_server,
    parse_qualified,
    qualified_name,
)
from app.mcp.registry import (
    build_entry,
    confirmable_builders,
    registry_for_agent,
    verify_server,
)

__all__ = [
    "MCPConnection",
    "MCPServerSpec",
    "MCPToolSpec",
    "MCPUnavailable",
    "MCP_PREFIX",
    "SERVERS",
    "build_entry",
    "close_all",
    "confirmable_builders",
    "connection_for",
    "enabled_servers",
    "find_server",
    "parse_qualified",
    "qualified_name",
    "registry_for_agent",
    "verify_server",
]
