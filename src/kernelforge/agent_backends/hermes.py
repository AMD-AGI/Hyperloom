# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hermes Agent backend for KernelForge's existing AgentRunSpec."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentProviderUnavailableError,
    AgentRunResult,
    AgentRunSpec,
    AgentRuntimeConfig,
)
from kernelforge.agent_backends.workspace_guard import WorkspaceGuard
from kernelforge.knowledge.experience_reader import sanitize_read_error


def _running_in_container() -> bool:
    """Require a concrete outer-container marker for writable Hermes tools."""

    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False
    return any(marker in cgroup for marker in ("docker", "kubepods", "containerd", "podman", "lxc"))


@dataclass
class HermesBackend:
    """Run one KernelForge session through a dedicated Hermes profile."""

    runtime: AgentRuntimeConfig
    name: str = "hermes"
    capabilities: AgentCapabilities = AgentCapabilities(
        writable=True,
        resumable=False,
        probe=True,
        requires_workspace_cwd=True,
        session_env=True,
        workspace_guard=True,
    )

    def _executable(self) -> str:
        value = self.runtime.executable.strip() or shutil.which("hermes") or ""
        if not value:
            raise AgentProviderUnavailableError("Hermes executable not found")
        return value

    def preflight(self) -> None:
        self._executable()
        if self.runtime.sandbox_mode.strip().lower() != "bypass":
            raise AgentProviderUnavailableError(
                "Hermes CLI has no native OS sandbox; sandbox_mode must be 'bypass' only when the operator "
                "already provides an external sandbox"
            )
        if self.runtime.options.get("external_sandbox") is not True:
            raise AgentProviderUnavailableError(
                "Hermes terminal/file sessions require options.external_sandbox=true inside an outer container"
            )
        if not _running_in_container():
            raise AgentProviderUnavailableError(
                "Hermes external_sandbox=true requires a verifiable outer container runtime"
            )

    def probe(self, *, cwd: str = "", usage: Any = None) -> None:
        del cwd, usage
        self.preflight()

    async def run(self, spec: AgentRunSpec, usage: Any = None) -> AgentRunResult:
        del usage
        resolved = spec.resolved(self.runtime)
        self.preflight()
        if resolved.progress_log is not None:
            resolved.progress_log.append("progress: Hermes CLI returns final output only")
        profile = str(self.runtime.options.get("profile") or "hyperloomfaithful").strip()
        provider = str(self.runtime.options.get("provider") or "openai-codex").strip()
        policy = (
            "You may use terminal and file tools, but only inside the declared workspace and only for target files."
            if resolved.writable
            else "This is a read-only reasoning turn. You have no local filesystem or terminal tools."
        )
        prompt = "\n\n".join(part for part in (resolved.system_prompt.strip(), policy, resolved.user_prompt) if part)
        toolsets = "terminal,file" if resolved.writable else "todo"
        argv = [
            self._executable(),
            "--profile",
            profile,
            "--provider",
            provider,
            "--model",
            resolved.model,
            "--safe-mode",
            "--toolsets",
            toolsets,
            "-z",
            prompt,
        ]
        child_env = dict(os.environ)
        child_env.update(resolved.env)
        cwd = Path(resolved.cwd).expanduser().resolve()
        cwd.mkdir(parents=True, exist_ok=True)
        guard = WorkspaceGuard(resolved, dirty_baseline_default=True)
        guard.prepare()

        def _invoke() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                argv,
                cwd=cwd,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=resolved.timeout_sec,
                check=False,
            )

        try:
            completed = await asyncio.to_thread(_invoke)
        except subprocess.TimeoutExpired as exc:
            with contextlib.suppress(Exception):
                guard.rollback()
            raise AgentProviderUnavailableError(f"Hermes session timed out after {resolved.timeout_sec}s") from exc
        except OSError as exc:
            with contextlib.suppress(Exception):
                guard.rollback()
            raise AgentProviderUnavailableError(f"Hermes session could not start: {exc}") from exc
        if completed.returncode != 0:
            with contextlib.suppress(Exception):
                guard.rollback()
            secret_values = tuple(
                value
                for key, value in child_env.items()
                if value
                and any(
                    fragment in key.upper() for fragment in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
                )
            )
            detail = sanitize_read_error(
                RuntimeError((completed.stderr or "").strip()[-1000:]),
                secrets=secret_values,
            )
            raise AgentProviderUnavailableError(f"Hermes session exited rc={completed.returncode}: {detail}")
        try:
            actual_changes = guard.verify()
        except Exception:
            with contextlib.suppress(Exception):
                guard.rollback()
            raise
        result = AgentRunResult(
            text=completed.stdout or "",
            subtype="success",
            num_turns=1,
            end_reason="agent_stopped",
        )
        result.file_changes = actual_changes
        result.target_edit_count = guard.count_target_edits()
        result.edit_count = result.target_edit_count
        return result


__all__ = ["HermesBackend"]
