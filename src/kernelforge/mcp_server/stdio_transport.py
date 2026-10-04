# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Newline-delimited JSON-RPC transport shared by the stdio MCP servers.

Both ``pr_stdio_server`` and ``probe_stdio_server`` speak the same wire protocol
to the same clients; only the tool set behind ``tools/call`` differs. Keeping the
envelope here means a protocol fix -- an error code, a method the client starts
sending -- lands once instead of drifting between two copies, and the read loop
gets one set of tests rather than one tested copy and one untested one.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any


class InvalidParamsError(ValueError):
    """Invalid agent-supplied MCP tool arguments."""


def write_message(payload: dict[str, Any]) -> None:
    """Write one newline-delimited JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()


def write_error(request_id: Any, code: int, message: str) -> None:
    """Write one JSON-RPC error response."""
    write_message(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    )


async def dispatch_envelope(
    method: str,
    params: dict[str, Any],
    *,
    server_name: str,
    tool_definitions: list[dict[str, Any]],
    handle_tool_call: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Answer one supported MCP request, delegating ``tools/call`` to the server."""
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion") or "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": server_name, "version": "0.1.0"},
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": tool_definitions}
    if method == "tools/call":
        # Not ``or {}``: a falsy-but-wrong value such as [] would coerce to an
        # empty object and slip past the type check below.
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise InvalidParamsError("tools/call arguments must be an object")
        return await handle_tool_call(str(params.get("name") or ""), arguments)
    if method in {"resources/list", "prompts/list"}:
        return {"resources": []} if method == "resources/list" else {"prompts": []}
    if method in {"logging/setLevel", "shutdown"}:
        return {}
    raise NotImplementedError(f"unsupported MCP method: {method}")


async def serve(
    dispatch: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
) -> None:
    """Serve JSON-RPC requests until stdin closes or an exit notification arrives."""
    while True:
        raw = await asyncio.to_thread(sys.stdin.buffer.readline)
        if not raw:
            return
        try:
            message = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            write_error(None, -32700, "Parse error")
            continue
        if not isinstance(message, dict):
            write_error(None, -32600, "Invalid Request")
            continue
        method = str(message.get("method") or "")
        request_id = message.get("id")
        if method == "exit":
            return
        if request_id is None:
            continue
        params = message.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            write_error(request_id, -32602, "params must be an object")
            continue
        try:
            result = await dispatch(method, params)
            write_message({"jsonrpc": "2.0", "id": request_id, "result": result})
        except NotImplementedError as exc:
            write_error(request_id, -32601, str(exc))
        except InvalidParamsError as exc:
            write_error(request_id, -32602, str(exc))
        except Exception as exc:  # noqa: BLE001 - convert failures to JSON-RPC
            write_error(request_id, -32603, f"{type(exc).__name__}: {exc}")
