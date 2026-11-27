# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assembly of in-process Claude Agent SDK MCP servers."""

from __future__ import annotations

from typing import Any, Callable, Iterable


def build_sdk_mcp_server(
    sdk: Any,
    server_name: str,
    tools: Iterable[tuple[str, str, dict[str, Any], Callable[..., Any]]],
) -> Any:
    """Build an in-process MCP server from ``(name, description, schema, handler)`` tools.

    Args:
        sdk: The imported ``claude_agent_sdk`` module.
        server_name: The name the server is registered under.
        tools: The tools it serves.

    Returns:
        The SDK's server config, for ``ClaudeAgentOptions.mcp_servers``.
    """
    decorated = [sdk.tool(name, description, schema)(handler) for name, description, schema, handler in tools]
    return sdk.create_sdk_mcp_server(server_name, "1.0.0", decorated)
