# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hermes backend must honor KernelForge's shared policy contract."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from kernelforge.agent_backends.base import (
    AgentProviderUnavailableError,
    AgentRunSpec,
    AgentRuntimeConfig,
    AgentToolPolicy,
)
from kernelforge.agent_backends.hermes import HermesBackend


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "driver.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "driver.py"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("writable", "toolsets"),
    [(False, "todo"), (True, "terminal,file")],
)
async def test_hermes_backend_maps_writable_policy_and_guards_workspace(
    tmp_path: Path,
    monkeypatch,
    writable: bool,
    toolsets: str,
) -> None:
    calls: list[tuple[list[str], dict]] = []
    _init_repo(tmp_path)
    monkeypatch.setattr("kernelforge.agent_backends.hermes._running_in_container", lambda: True)

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr("kernelforge.agent_backends.hermes.subprocess.run", fake_run)
    runtime = AgentRuntimeConfig(
        provider="hermes",
        model="gpt-5.6-sol",
        executable="/runtime/hermes",
        sandbox_mode="bypass",
        options={"external_sandbox": True},
    )
    backend = HermesBackend(runtime)
    spec = AgentRunSpec(
        system_prompt="SYS",
        user_prompt="USER",
        cwd=str(tmp_path),
        writable=writable,
        tool_policy=(None if writable else AgentToolPolicy(write=False, shell=False)),
    )

    result = await backend.run(spec)

    argv, _ = calls[0]
    assert result.text == "done"
    assert "--safe-mode" in argv
    assert argv[argv.index("--toolsets") + 1] == toolsets
    assert "--yolo" not in argv
    assert backend.capabilities.workspace_guard
    assert backend.capabilities.sandbox is False


@pytest.mark.asyncio
async def test_hermes_backend_rejects_unimplemented_native_sandbox(tmp_path: Path) -> None:
    backend = HermesBackend(
        AgentRuntimeConfig(
            provider="hermes",
            model="gpt-5.6-sol",
            executable="/runtime/hermes",
            sandbox_mode="workspace-write",
        )
    )
    spec = AgentRunSpec(system_prompt="SYS", user_prompt="USER", cwd=str(tmp_path), writable=True)

    with pytest.raises(AgentProviderUnavailableError, match="external sandbox"):
        await backend.run(spec)


@pytest.mark.asyncio
async def test_hermes_backend_requires_declared_external_sandbox(tmp_path: Path) -> None:
    backend = HermesBackend(
        AgentRuntimeConfig(
            provider="hermes",
            model="gpt-5.6-sol",
            executable="/runtime/hermes",
            sandbox_mode="bypass",
        )
    )
    spec = AgentRunSpec(system_prompt="SYS", user_prompt="USER", cwd=str(tmp_path), writable=True)

    with pytest.raises(AgentProviderUnavailableError, match="external_sandbox=true"):
        await backend.run(spec)


@pytest.mark.asyncio
async def test_hermes_backend_redacts_stderr_and_secret_values(tmp_path: Path, monkeypatch) -> None:
    _init_repo(tmp_path)
    monkeypatch.setattr("kernelforge.agent_backends.hermes._running_in_container", lambda: True)

    def fake_run(argv, **kwargs):
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="Authorization: Bearer top-secret-value API_KEY=also-secret",
        )

    monkeypatch.setattr("kernelforge.agent_backends.hermes.subprocess.run", fake_run)
    backend = HermesBackend(
        AgentRuntimeConfig(
            provider="hermes",
            model="gpt-5.6-sol",
            executable="/runtime/hermes",
            sandbox_mode="bypass",
            options={"external_sandbox": True},
        )
    )
    spec = AgentRunSpec(
        system_prompt="SYS",
        user_prompt="USER",
        cwd=str(tmp_path),
        env={"PRIVATE_TOKEN": "top-secret-value"},
    )

    with pytest.raises(AgentProviderUnavailableError) as exc_info:
        await backend.run(spec)

    message = str(exc_info.value)
    assert "top-secret-value" not in message
    assert "also-secret" not in message
    assert "[REDACTED]" in message
