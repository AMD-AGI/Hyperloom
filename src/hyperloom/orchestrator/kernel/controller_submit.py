# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launch the KernelForge rewrite controller as one bounded subprocess tree."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from hyperloom.common.llm_attribution import inject_env
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    run_with_session_kill,
)

_TERMINAL_STATUSES = {
    "completed",
    "no_opportunity",
    "no_result",
    "partial",
    "failed",
}


def build_controller_command(
    *,
    handoff_dir: Path,
    output_dir: Path,
    budget_minutes: float,
) -> list[str]:
    """Build the only Hyperloom-to-Controller CLI contract."""
    return [
        sys.executable,
        "-m",
        "kernelforge.cli",
        "kernel-rewrite-controller",
        "--handoff-dir",
        str(Path(handoff_dir).resolve()),
        "--budget-minutes",
        f"{float(budget_minutes):.9g}",
        "--output-dir",
        str(Path(output_dir).resolve()),
    ]


def _controller_environment() -> dict[str, str]:
    env = dict(os.environ)
    src_root = str(Path(__file__).resolve().parents[3])
    existing = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    env["PYTHONPATH"] = os.pathsep.join([src_root, *(entry for entry in existing if entry != src_root)])
    env.setdefault("PYTHONUNBUFFERED", "1")
    inject_env(
        env,
        component="kernel_rewrite_controller",
        operation="analyze_and_optimize_kernels",
    )
    return env


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _published_patch_count(output_dir: Path) -> int:
    root = Path(output_dir) / "result" / "patches"
    if not root.is_dir():
        return 0
    return sum(
        1
        for entry in root.iterdir()
        if not entry.name.startswith(".")
        and entry.is_dir()
        and (entry / "change.patch").is_file()
        and (entry / "report.md").is_file()
    )


def read_controller_result(
    *,
    output_dir: Path,
    returncode: int,
    timed_out: bool,
    stderr: str = "",
) -> dict[str, Any]:
    """Normalize durable Controller state after normal exit or hard timeout."""
    output = Path(output_dir).resolve()
    state_path = output / "controller" / "state.json"
    result = _read_json(state_path)
    patch_count = _published_patch_count(output)
    status = str(result.get("status") or "").strip().lower()
    if status not in _TERMINAL_STATUSES:
        status = "partial" if patch_count else "failed"
    normalized = {
        **result,
        "status": status,
        "returncode": int(returncode),
        "timed_out": bool(timed_out),
        "killed_by_hyperloom": bool(timed_out),
        "patch_count": patch_count,
        "output_dir": str(output),
        "controller_state_path": str(state_path),
        "summary_path": str(output / "result" / "summary.md"),
        "patches_root": str(output / "result" / "patches"),
    }
    if stderr:
        normalized["stderr_tail"] = stderr[-2000:]
    if timed_out:
        normalized["reason"] = str(normalized.get("reason") or "controller exceeded Hyperloom hard timeout")
    elif returncode != 0 and not normalized.get("reason"):
        normalized["reason"] = f"controller exited with return code {returncode}"
    return normalized


def run_controller_subprocess(
    *,
    handoff_dir: Path,
    output_dir: Path,
    budget_minutes: float,
    hard_timeout_sec: int,
    on_output: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run Controller and reclaim its complete descendant process tree."""
    command = build_controller_command(
        handoff_dir=handoff_dir,
        output_dir=output_dir,
        budget_minutes=budget_minutes,
    )
    timed_out = False
    try:
        completed = run_with_session_kill(
            command,
            env=_controller_environment(),
            timeout=max(1, int(hard_timeout_sec)),
            text=True,
            on_output=on_output,
        )
        returncode = int(completed.returncode)
        stderr = str(completed.stderr or "")
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = -1
        stderr = str(error.stderr or "")
    return read_controller_result(
        output_dir=output_dir,
        returncode=returncode,
        timed_out=timed_out,
        stderr=stderr,
    )


__all__ = [
    "build_controller_command",
    "read_controller_result",
    "run_controller_subprocess",
]
