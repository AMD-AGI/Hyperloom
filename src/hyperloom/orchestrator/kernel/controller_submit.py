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

from kernelforge.kernel_rewrite_controller import ControllerLayout, published_operator_dirs

from hyperloom.common.llm_attribution import inject_env
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    RoundWatch,
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
        component="forge",
        operation="analyze_and_optimize_kernels",
    )
    return env


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_controller_result(
    *,
    output_dir: Path,
    returncode: int,
    timed_out: bool,
    stderr: str = "",
) -> dict[str, Any]:
    """Normalize durable Controller state after normal exit or hard timeout."""
    layout = ControllerLayout(output_dir)
    result = _read_json(layout.controller_state)
    patch_count = len(published_operator_dirs(layout))
    status = str(result.get("status") or "").strip().lower()
    if status not in _TERMINAL_STATUSES:
        status = "partial" if patch_count else "failed"
    normalized = {
        **result,
        "status": status,
        "returncode": int(returncode),
        "timed_out": bool(timed_out),
        "patch_count": patch_count,
        "output_dir": str(layout.output_dir),
        "controller_state_path": str(layout.controller_state),
        "summary_path": str(layout.summary_md),
        "patches_root": str(layout.patches_root),
    }
    if stderr:
        normalized["stderr_tail"] = stderr[-2000:]
    if timed_out:
        normalized["reason"] = str(normalized.get("reason") or "controller exceeded Hyperloom hard timeout")
    elif returncode != 0 and not normalized.get("reason"):
        normalized["reason"] = f"controller exited with return code {returncode}"
    return normalized


def record_controller_llm_usage(*, result: dict[str, Any], session_dir: Path) -> int:
    """Append one ``llm_calls`` row per model call the Controller paid for."""
    rows: list[Any] = []
    for key in ("forge_llm_usage", "analysis_llm_usage"):
        listed = result.get(key)
        if isinstance(listed, list):
            rows.extend(listed)
    appended = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            from hyperloom.inference_optimizer.trace.llm_trace import (
                LLMCallRecord,
                append_llm_call,
            )

            append_llm_call(
                session_dir=Path(session_dir),
                record=LLMCallRecord(
                    session_id=Path(session_dir).name,
                    component="forge",
                    task_id=str(row.get("operator_id") or "") or None,
                    model=str(row.get("model") or "") or None,
                    input_tokens=row.get("input_tokens"),
                    output_tokens=row.get("output_tokens"),
                    cache_creation_input_tokens=row.get("cache_creation_input_tokens"),
                    cache_read_input_tokens=row.get("cache_read_input_tokens"),
                ),
            )
        except ValueError:  # a malformed usage row is skipped
            continue
        appended += 1
    return appended


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
            watch=RoundWatch(timeout=max(1, int(hard_timeout_sec))),
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
    "record_controller_llm_usage",
    "run_controller_subprocess",
]
