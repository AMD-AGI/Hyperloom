# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared low-level git subprocess primitives for framework / integrate_patch.

``_run_git`` → ``(ok, stdout, stderr)``; ``_run_git_cp`` → the raw
CompletedProcess (or None on spawn/timeout) for callers that must inspect
returncode. Both carry a ``safe.directory`` exception for the target checkout.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from hyperloom.common.git_safety import safe_directory_args

__all__ = ["_run_git", "_run_git_cp"]


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
    input: str | None = None,  # noqa: A002 - mirrors subprocess.run's keyword
) -> subprocess.CompletedProcess | None:
    """Run ``git <args>`` returning the raw CompletedProcess, or None on spawn/timeout.

    ``cwd`` locates the checkout for callers that do not pass ``-C``; ``input``
    feeds a patch on stdin.
    """
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
