# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assembly of in-process Claude Agent SDK MCP servers."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)


def _resolve_sdk(sdk_module: Any | None) -> Any | None:
    """Resolve the Claude Agent SDK module."""
    if sdk_module is not None:
        return sdk_module
    try:
        return importlib.import_module("claude_agent_sdk")
    except ImportError:
        return None


def build_sdk_mcp_server(
    server_name: str,
    tools: Iterable[tuple[str, str, dict[str, Any], Callable[..., Any]]],
    *,
    sdk_module: Any | None = None,
    tool_factory: Callable[..., Any] | None = None,
    server_factory: Callable[..., Any] | None = None,
) -> Any | None:
    """Build an in-process MCP server from ``(name, description, schema, handler)`` tools.

    Returns ``None`` when the SDK (or an explicit factory) is unavailable.
    """
    sdk = _resolve_sdk(sdk_module)
    if tool_factory is None:
        tool_factory = getattr(sdk, "tool", None)
    if server_factory is None:
        server_factory = getattr(sdk, "create_sdk_mcp_server", None)
    if tool_factory is None or server_factory is None:
        log.info("%s MCP server unavailable (sdk=%s).", server_name, getattr(sdk, "__name__", "<none>"))
        return None
    decorated = [tool_factory(name, description, schema)(handler) for name, description, schema, handler in tools]
    return server_factory(server_name, "1.0.0", decorated)
