# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The committed task workspace the in-session gate tests build their gates over."""

from __future__ import annotations

import subprocess
from pathlib import Path


def committed_task_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """A repository holding a driver, a task config, a runner and one kernel source; returns it and the kernel."""
    workspace = tmp_path / "ws"
    scripts = workspace / "scripts"
    source = workspace / "aiter" / "csrc"
    scripts.mkdir(parents=True)
    source.mkdir(parents=True)

    (workspace / "config.yaml").write_text("task_type: image_kernel\n")
    (workspace / "forge_driver.py").write_text("print('driver')\n")
    (scripts / "task_runner.py").write_text("print('runner')\n")
    kernel = source / "kernel.cu"
    kernel.write_text("__global__ void kernel() {}\n")
    # The gate diffs the session's edits against HEAD, so its workspace is a repo that has one.
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "base"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    return workspace, kernel
