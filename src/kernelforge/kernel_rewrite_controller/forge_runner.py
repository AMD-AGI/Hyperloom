# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run one named-kernel ``forge-loop`` as an isolated subprocess."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.kernel_rewrite_controller.contracts import KernelRewriteTask
from kernelforge.kernel_rewrite_controller.worktree import OperatorWorktree

_RESULT_SENTINEL = "__FORGE_RESULT__"
_TERMINATE_GRACE_SEC = 5.0
_CHECKPOINT_POLL_SEC = 1.0


@dataclass(frozen=True)
class ForgeLoopInvocation:
    """Resolved arguments and artifacts for one forge-loop subprocess."""

    command: tuple[str, ...]
    workspace: Path
    result_json: Path
    deadline_unix: float


@dataclass(frozen=True)
class ForgeLoopOutcome:
    """Normalized completion state for one forge-loop subprocess."""

    returncode: int
    stdout: str
    stderr: str
    result: dict[str, Any] | None
    timed_out: bool
    command: tuple[str, ...]

    @property
    def best_commit(self) -> str:
        payload = self.result or {}
        direct = str(payload.get("best_commit") or "").strip()
        if direct:
            return direct
        checkpoint = payload.get("checkpoint")
        return str(checkpoint.get("best_commit") or "").strip() if isinstance(checkpoint, dict) else ""

    @property
    def improved(self) -> bool:
        return bool((self.result or {}).get("improved")) and bool(self.best_commit)


def _local_src_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _child_environment() -> dict[str, str]:
    env = dict(os.environ)
    src_root = str(_local_src_root())
    current = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    env["PYTHONPATH"] = os.pathsep.join([src_root, *(entry for entry in current if entry != src_root)])
    return env


def build_forge_loop_invocation(
    task: KernelRewriteTask,
    *,
    task_dir: Path,
    worktree: OperatorWorktree,
    deadline_unix: float,
) -> ForgeLoopInvocation:
    """Map one controller task onto the existing named-kernel forge-loop CLI."""
    driver = (Path(task_dir).resolve() / task.driver_path).resolve()
    result_json = Path(task_dir).resolve() / "forge-result.json"
    remaining_hours = max(0.0, (float(deadline_unix) - time.time()) / 3600.0)
    max_hours = max(1.0, remaining_hours)
    experiment_id = f"controller-{hashlib.sha256(task.operator_id.encode('utf-8')).hexdigest()[:16]}"
    command = [
        sys.executable,
        "-m",
        "kernelforge.cli",
        "forge-loop",
        "--workspace",
        str(worktree.workspace),
        "--kernel",
        str(worktree.kernel_path),
        "--driver",
        str(driver),
        "--max-hours",
        str(max_hours),
        "--deadline-unix",
        str(float(deadline_unix)),
        "--git-branch",
        worktree.branch,
        "--gpu-type",
        task.identity.gpu,
        "--kernel-backend",
        task.identity.backend,
        "--framework",
        task.identity.framework,
        "--operator-name",
        task.operator_name,
        "--producer",
        task.identity.producer,
        "--experiments-dir",
        str(worktree.workspace / "forge_experiments"),
        "--experiment-id",
        experiment_id,
        "--experience-id",
        task.operator_id,
        "--result-json",
        str(result_json),
    ]
    if worktree.source_files:
        command.extend(["--source-files", ",".join(str(path) for path in worktree.source_files)])
    if task.target_functions:
        command.extend(["--target-functions", ",".join(task.target_functions)])
    return ForgeLoopInvocation(
        command=tuple(command),
        workspace=worktree.workspace,
        result_json=result_json,
        deadline_unix=float(deadline_unix),
    )


def _terminate_process_group(process: subprocess.Popen) -> tuple[str, str]:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        return process.communicate(timeout=_TERMINATE_GRACE_SEC)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        return process.communicate()


def _read_result(path: Path, stdout: str) -> dict[str, Any] | None:
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
    parts = stdout.split(_RESULT_SENTINEL)
    if len(parts) >= 3:
        try:
            payload = json.loads(parts[-2])
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _notify_checkpoint(callback: Callable[[], None] | None) -> None:
    if callback is None:
        return
    with contextlib.suppress(Exception):
        callback()


def run_forge_loop(
    invocation: ForgeLoopInvocation,
    *,
    on_checkpoint: Callable[[], None] | None = None,
) -> ForgeLoopOutcome:
    """Run forge-loop until completion or the controller task deadline."""
    remaining = invocation.deadline_unix - time.time()
    if remaining <= 0:
        return ForgeLoopOutcome(
            returncode=-1,
            stdout="",
            stderr="controller deadline reached before forge-loop started",
            result=_read_result(invocation.result_json, ""),
            timed_out=True,
            command=invocation.command,
        )
    process = subprocess.Popen(
        list(invocation.command),
        cwd=invocation.workspace,
        env=_child_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    while True:
        remaining = invocation.deadline_unix - time.time()
        if remaining <= 0:
            timed_out = True
            stdout, stderr = _terminate_process_group(process)
            break
        try:
            stdout, stderr = process.communicate(timeout=min(remaining, _CHECKPOINT_POLL_SEC))
            break
        except subprocess.TimeoutExpired:
            _notify_checkpoint(on_checkpoint)
    _notify_checkpoint(on_checkpoint)
    return ForgeLoopOutcome(
        returncode=int(process.returncode if process.returncode is not None else -1),
        stdout=stdout,
        stderr=stderr,
        result=_read_result(invocation.result_json, stdout),
        timed_out=timed_out,
        command=invocation.command,
    )


__all__ = [
    "ForgeLoopInvocation",
    "ForgeLoopOutcome",
    "build_forge_loop_invocation",
    "run_forge_loop",
]
