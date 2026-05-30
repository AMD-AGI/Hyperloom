#!/usr/bin/env python3
"""MCP server exposing Hyperloom agent dispatch tools.

This is a stdio-based MCP server that the orchestrator (running via Claude Code CLI)
can use to dispatch specialist agents, check their status, and collect results.

Tools exposed:
  - dispatch_agents: Launch specialist sub-agents (CPU or GPU)
  - check_agents: Poll status of all dispatched agents
  - collect_agent_results: Read a completed agent's report and patches
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


SESSION_DIR = os.environ.get("HYPERLOOM_SESSION_DIR", "")


def _read_jsonrpc() -> dict | None:
    """Read a JSON-RPC message from stdin (Content-Length framed)."""
    headers = {}
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        if ":" in line:
            key, val = line.split(":", 1)
            headers[key.strip().lower()] = val.strip()

    length = int(headers.get("content-length", 0))
    if length == 0:
        return None
    body = sys.stdin.read(length)
    return json.loads(body)


def _write_jsonrpc(msg: dict) -> None:
    """Write a JSON-RPC message to stdout (Content-Length framed)."""
    body = json.dumps(msg)
    header = f"Content-Length: {len(body)}\r\n\r\n"
    sys.stdout.write(header)
    sys.stdout.write(body)
    sys.stdout.flush()


def _respond(req_id: Any, result: dict) -> None:
    _write_jsonrpc({"jsonrpc": "2.0", "id": req_id, "result": result})


def _error(req_id: Any, code: int, message: str) -> None:
    _write_jsonrpc({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


TOOLS = [
    {
        "name": "dispatch_agents",
        "description": (
            "Dispatch specialist agents to work on optimization tasks in parallel. "
            "CPU-only agents launch immediately; GPU agents are allocated from the pool. "
            "Each agent runs as an independent Claude Code process with full tool access. "
            "Use this instead of doing optimization work directly — specialists are better "
            "at focused tasks like kernel tuning, config exploration, and profiling analysis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "List of tasks to dispatch",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Detailed description of what the agent should do",
                            },
                            "task_summary": {
                                "type": "string",
                                "description": "Short 1-line summary for tracking",
                            },
                            "needs_gpu": {
                                "type": "boolean",
                                "description": "Whether this task requires GPU access",
                                "default": False,
                            },
                            "gpu_count": {
                                "type": "integer",
                                "description": "Number of GPUs needed (if needs_gpu=true)",
                                "default": 1,
                            },
                            "role": {
                                "type": "string",
                                "description": "Agent role: specialist, kernel, profiler, config, critic",
                                "default": "specialist",
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["critical", "high", "normal", "low"],
                                "default": "normal",
                            },
                            "kb_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Knowledge base domains to inject (e.g., ['vllm', 'rocm', 'moe'])",
                            },
                        },
                        "required": ["task_description", "task_summary"],
                    },
                },
                "model": {
                    "type": "string",
                    "description": "Model for specialist agents (default: from env AGENT_MODEL)",
                },
                "timeout_minutes": {
                    "type": "integer",
                    "description": "Max runtime per agent in minutes",
                    "default": 120,
                },
            },
            "required": ["tasks"],
        },
    },
    {
        "name": "check_agents",
        "description": (
            "Check status of all dispatched specialist agents. Returns active (still running), "
            "completed (finished with results), and dead (crashed/timed out) agents. "
            "Call this periodically to monitor progress and know when to collect results."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "collect_agent_results",
        "description": (
            "Collect results from a completed specialist agent: completion report, "
            "config changes, patches, and new knowledge discovered. Use the patches "
            "to apply optimizations and re-benchmark."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "ID of the completed agent to collect results from",
                },
            },
            "required": ["agent_id"],
        },
    },
]


def handle_initialize(req_id: Any, params: dict) -> None:
    _respond(req_id, {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {
            "name": "hyperloom-dispatch",
            "version": "1.0.0",
        },
    })


def handle_tools_list(req_id: Any) -> None:
    _respond(req_id, {"tools": TOOLS})


def handle_tool_call(req_id: Any, params: dict) -> None:
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    session_dir = SESSION_DIR
    if not session_dir:
        _respond(req_id, {"content": [{"type": "text", "text": "ERROR: HYPERLOOM_SESSION_DIR not set"}]})
        return

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hyperloom.orchestrator import execute_tool

    result_text = execute_tool(tool_name, arguments, session_dir)
    _respond(req_id, {"content": [{"type": "text", "text": result_text}]})


def main() -> None:
    """Main MCP server loop — reads JSON-RPC messages and responds."""
    while True:
        msg = _read_jsonrpc()
        if msg is None:
            break

        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            handle_initialize(req_id, params)
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            handle_tools_list(req_id)
        elif method == "tools/call":
            handle_tool_call(req_id, params)
        elif method == "shutdown":
            if req_id is not None:
                _respond(req_id, {})
            break
        else:
            if req_id is not None:
                _error(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()
