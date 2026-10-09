# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared low-level git subprocess primitives for framework / integrate_patch."""

from __future__ import annotations

import subprocess
from pathlib import Path

from hyperloom.common.git_safety import safe_directory_args

__all__ = ["_git_head_sha", "_run_git", "_run_git_cp"]


def _run_git(args: list[str], *, timeout: float = 120.0) -> tuple[bool, str, str]:
    """Run ``git <args>`` capturing output; returns ``(ok, stdout, stderr)``, never raises."""
    try:
        cp = subprocess.run(
            ["git", *safe_directory_args(args)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, "", f"git spawn/timeout failed: {exc!r}"
    if cp.returncode != 0:
        return False, cp.stdout or "", (cp.stderr or "").strip()
    return True, cp.stdout or "", cp.stderr or ""


def _run_git_cp(
    args: list[str],
    *,
    timeout: float = 120.0,
    cwd: str | Path | None = None,
    input: str | None = None,
) -> subprocess.CompletedProcess | None:
    """Run ``git <args>`` returning the raw CompletedProcess, or None on spawn/timeout."""
    try:
        return subprocess.run(
            ["git", *safe_directory_args(args, cwd=cwd)],
            cwd=cwd,
            input=input,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _git_head_sha(root: str | Path) -> str:
    """Return ``root``'s HEAD commit, or ``""`` when the read failed (spawn failure, timeout, non-zero exit)."""
    cp = _run_git_cp(["-C", str(root), "rev-parse", "HEAD"], timeout=30.0)
    if cp is None or cp.returncode != 0:
        return ""
    return (cp.stdout or "").strip()
