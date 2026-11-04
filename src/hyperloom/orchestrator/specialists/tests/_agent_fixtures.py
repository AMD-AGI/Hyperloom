# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared builders for the agent runner tests."""

from __future__ import annotations

import asyncio
import secrets
import sys
import time
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.specialists.agent_runner import (
    AgentLaunchSpec,
    AgentStatus,
    LocalAgentRunner,
    RunnerHost,
    McpEndpoint,
)

TOKEN = "tok-" + secrets.token_hex(12)  # built at run time: no key-shaped literal in the repo
TOKEN_ENV = "HYPERLOOM_SPECIALIST_MCP_TOKEN"


def child_argv(code: str) -> tuple[str, ...]:
    """A python child standing in for the agent CLI (its argv[0] is not codex, so it is a claude backend)."""
    return (sys.executable, "-c", code)


def make_spec(
    workspace: Path,
    *,
    task_id: str = "task-1",
    code: str = "import time; time.sleep(60)",
    deadline_unix: float | None = None,
    env: dict[str, str] | None = None,
    mcp: bool = True,
    **overrides: Any,
) -> AgentLaunchSpec:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "worktree").mkdir(exist_ok=True)
    (workspace / "prompt.md").write_text("do the task\n", encoding="utf-8")
    base_env = {"PATH": "/usr/bin:/bin", "HOME": str(workspace), TOKEN_ENV: TOKEN}
    base_env.update(env or {})
    fields: dict[str, Any] = {
        "task_id": task_id,
        "workspace": workspace,
        "argv": child_argv(code),
        "env": base_env,
        "cwd": workspace,
        "gpus": 0,
        "deadline_unix": deadline_unix if deadline_unix is not None else time.time() + 6 * 3600,
        "done_paths": (workspace / "specialist_done.json",),
        "heartbeat_path": workspace / "heartbeat.json",
        "mcp_servers": (McpEndpoint("tools", "http://127.0.0.1:9/mcp", TOKEN_ENV),) if mcp else (),
    }
    fields.update(overrides)
    return AgentLaunchSpec(**fields)


def make_runner(session_dir: Path, **kwargs: Any) -> LocalAgentRunner:
    kwargs.setdefault("poll_interval_seconds", 0.05)
    kwargs.setdefault("version_probe", lambda argv0, env: "test-1.0")
    host_keys = ("monotonic", "wall", "popen", "cgroup_root", "proc_root", "sleep")
    host = {key: kwargs.pop(key) for key in host_keys if key in kwargs}
    host.setdefault("cgroup_root", session_dir / "no-cgroup")
    kwargs["host"] = RunnerHost(**host)
    return LocalAgentRunner(session_dir=session_dir, **kwargs)


async def wait_finished(runner: LocalAgentRunner, handle: Any, timeout: float = 20.0) -> AgentStatus:
    end = time.time() + timeout
    while time.time() < end:
        status = await runner.poll(handle)
        if status.state != "running" and handle.session.task.done():
            return status
        await asyncio.sleep(0.02)
    raise AssertionError("session did not finish")


def read_all_files(root: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
