# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared low-level git subprocess primitives for framework / integrate_patch.

``_run_git`` → ``(ok, stdout, stderr)``; ``_run_git_cp`` → the raw
CompletedProcess (or None on spawn/timeout) for callers that must inspect
returncode.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["_run_git", "_run_git_cp"]


def _repo_root(target: str) -> str | None:
    """Nearest ancestor of ``target`` holding a ``.git`` entry, else None.

    ``.git`` is a file in linked worktrees, so existence is the test. Returns
    None outside any checkout, where no exception should be invented.
    """
    try:
        current = Path(target).expanduser().resolve()
    except OSError:
        return None
    for candidate in (current, *current.parents):
        try:
            if (candidate / ".git").exists():
                return str(candidate)
        except OSError:
            continue
    return None


def _safe_directory_args(args: list[str]) -> list[str]:
    """Prepend a ``safe.directory`` exception for the repo ``args`` targets.

    A bind-mounted checkout owned by another uid makes git refuse every
    operation on it, reads included. git honours ``safe.directory`` from command
    config, resolves ownership against the repository root rather than the
    ``-C`` path, and ignores a ``-c`` placed after the subcommand.
    """
    try:
        target = args[args.index("-C") + 1]
    except (ValueError, IndexError):
        return args
    root = _repo_root(target)
    if root is None:
        return args
    return ["-c", f"safe.directory={root}", *args]


def _run_git(args: list[str], *, timeout: float = 120.0) -> tuple[bool, str, str]:
    """Run ``git <args>`` capturing output; returns ``(ok, stdout, stderr)``, never raises."""
    try:
        cp = subprocess.run(
            ["git", *_safe_directory_args(args)],
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


def _run_git_cp(args: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess | None:
    """Run ``git <args>`` returning the raw CompletedProcess, or None on spawn/timeout."""
    try:
        return subprocess.run(
            ["git", *_safe_directory_args(args)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
