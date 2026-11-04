# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AI work socket against a real MCP service: a call in flight keeps the session's heartbeat fresh."""

from __future__ import annotations

import asyncio
import os
import secrets
import sys
import time
from pathlib import Path

from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
from hyperloom.orchestrator.specialists.agent_runner import (
    LOST_AFTER_SEC,
    AgentLaunchSpec,
    LocalAgentRunner,
    McpEndpoint,
)
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST
from hyperloom_multibranch.toolbox.mcp_tools import SPECIALIST_TOKEN_ENV, McpSpecialistTokens

# A stand-in for the agent CLI: it reads the MCP file the runner passed with --mcp-config, expands ${VAR} in the
# header from its own environment the way the CLIs do, and makes one blocking tool call.
AGENT = """
import json, os, pathlib, re, sys, time, urllib.request
time.sleep(1.0)
config = json.loads(pathlib.Path(sys.argv[-1]).read_text())
server = config["mcpServers"]["tools"]
header = re.sub(r"\\$\\{(\\w+)\\}", lambda m: os.environ[m.group(1)], server["headers"]["Authorization"])
pathlib.Path("call_started").write_text("1")
body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "block", "arguments": {}}}
request = urllib.request.Request(server["url"], json.dumps(body).encode(), {"Authorization": header})
with urllib.request.urlopen(request, timeout=20) as response:
    pathlib.Path("call_status").write_text(str(response.status))
pathlib.Path("specialist_done.json").write_text("{}")
"""


async def test_the_mcp_services_beat_keeps_a_session_in_a_call_from_being_judged_lost(tmp_path: Path) -> None:
    server = McpServer(
        McpServerConfig(host_token="host-" + secrets.token_hex(12)), campaign="camp", heartbeat_seconds=0.1
    )
    server.register(
        "block",
        schema={"type": "object"},
        handler=lambda arguments, ctx: time.sleep(1.2) or {"done": True},
        scopes={SCOPE_SPECIALIST},
    )
    server.start()
    workspace = tmp_path / "w"
    workspace.mkdir()
    (workspace / "prompt.md").write_text("go\n", encoding="utf-8")
    runner = LocalAgentRunner(
        session_dir=tmp_path / "session", poll_interval_seconds=0.05, version_probe=lambda a, e: "x"
    )
    try:
        grant = McpSpecialistTokens(server).issue("s1", frozenset({SCOPE_SPECIALIST}))
        assert grant.token_env == SPECIALIST_TOKEN_ENV
        spec = AgentLaunchSpec(
            task_id="s1",
            workspace=workspace,
            argv=(sys.executable, "-c", AGENT),
            env={"PATH": os.environ.get("PATH", ""), grant.token_env: grant.token},
            cwd=workspace,
            gpus=0,
            deadline_unix=time.time() + 6 * 3600,
            done_paths=(workspace / "specialist_done.json",),
            heartbeat_path=workspace / "heartbeat.json",
            mcp_servers=(McpEndpoint("tools", grant.url, grant.token_env),),
            heartbeat_sink=grant.heartbeat_sink,
        )
        handle = await runner.launch(spec)
        # The agent has been silent for longer than the lost window.
        old = time.time() - (LOST_AFTER_SEC + 600)
        for path in (spec.heartbeat_path, workspace / "process.log"):
            path.touch()
            os.utime(path, (old, old))
        assert (await runner.poll(handle)).state == "lost"

        while not (workspace / "call_started").exists():
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.6)  # inside the 1.2 s call: the service has beaten several times
        status = await runner.poll(handle)
        assert status.state == "running"
        assert status.last_heartbeat_unix is not None and time.time() - status.last_heartbeat_unix < 2.0

        await handle.session.task
        assert (workspace / "call_status").read_text() == "200"  # the Bearer header carried the real token
        await runner.collect(handle)
        stamp = spec.heartbeat_path.stat().st_mtime
        await asyncio.sleep(0.5)
        assert spec.heartbeat_path.stat().st_mtime == stamp  # no call, no beat
    finally:
        server.stop()
