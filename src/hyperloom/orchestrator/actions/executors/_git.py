# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared low-level git subprocess primitives for framework / integrate_patch.

``_run_git`` → ``(ok, stdout, stderr)``; ``_run_git_cp`` → the raw
CompletedProcess (or None on spawn/timeout) for callers that must inspect
returncode (e.g. the stash family's rc==128 "not a git repository" case).
"""

from __future__ import annotations

import subprocess

__all__ = ["_run_git", "_run_git_cp"]


def _run_git(args: list[str], *, timeout: float = 120.0) -> tuple[bool, str, str]:
    """Run ``git <args>`` capturing output; returns ``(ok, stdout, stderr)``, never raises."""
    try:
        cp = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, "", f"git spawn/timeout failed: {exc!r}"
    if cp.returncode != 0:
        return False, cp.stdout or "", (cp.stderr or "").strip()
    return True, cp.stdout or "", cp.stderr or ""


def _run_git_cp(args: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess | None:
    """Run ``git <args>`` returning the raw CompletedProcess, or None on spawn/timeout."""
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
