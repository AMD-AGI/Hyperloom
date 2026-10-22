# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dispatch `kernelforge forge-loop` over a materialized SIKL run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hyperloom.sikl.workspace import Run

RESULT_FILENAME = "forge_result.json"


def build_command(
    run: Run,
    *,
    kernel_backend: str,
    gpu_target: str,
    max_hours: float,
    git_branch: str,
    model: str = "",
) -> list[str]:
    """The argv that optimizes this run's kernel."""
    command = [
        sys.executable,
        "-m",
        "kernelforge.cli",
        "forge-loop",
        "--kernel",
        str(run.kernel),
        "--driver",
        str(run.driver),
        "--workspace",
        str(run.workspace),
        "--experiments-dir",
        str(run.experiments),
        "--result-json",
        str(run.experiments / RESULT_FILENAME),
        "--program-md-file",
        str(run.program_md),
        "--kernel-backend",
        kernel_backend,
        "--gpu-target",
        gpu_target,
        "--max-hours",
        str(max_hours),
        "--git-branch",
        git_branch,
    ]
    if model:
        command += ["--model", model]
    return command


def dispatch(command: list[str], *, cwd: Path) -> int:
    """Run forge-loop in its own process group and return its exit status."""
    completed = subprocess.run(command, cwd=cwd, start_new_session=True)
    return completed.returncode
