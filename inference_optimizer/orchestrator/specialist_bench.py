# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Worktree git helpers for patch-authoring specialists.

Migrated out of the retired ``dynamic_action_tools`` module. Exposes the
worktree-scoped ``git`` helpers (self-check apply, cumulative-diff capture,
hard reset) used by :class:`SpecialistRunner`.

The legacy ``run_bench`` in-loop micro-bench tool (and its capped/stubbed
``BENCH_REGISTRY`` whitelist) has been removed: GPU specialists now run real
serving + benchmark / autotune loops on their own leased cards (see
``specialist_rebench`` and the opened-up iron rules), so a capped, serving-
forbidden micro-bench box is no longer the validation path.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


def _error(reason: str, **extra: Any) -> dict[str, Any]:
    """Build a failure result envelope.

    Args:
        reason: Human-readable failure reason.
        **extra: Additional fields to merge into the envelope.

    Returns:
        Dict with ``ok=False`` plus the reason and any extra fields.
    """
    return {"ok": False, "reason": reason, **extra}


def _ok(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a success result envelope.

    Args:
        payload: Optional fields to merge into the envelope.

    Returns:
        Dict with ``ok=True`` plus any payload fields.
    """
    out: dict[str, Any] = {"ok": True}
    if payload:
        out.update(payload)
    return out


_PATCH_PATH_RE = re.compile(r"^(?:---|\+\+\+) (?:a|b)/(?P<path>.+)$", re.M)


def apply_patch_in_worktree(
    worktree: Path,
    patch_text: str,
) -> dict[str, Any]:
    """Try ``git apply`` inside the worktree (self-check); not committed.

    Args:
        worktree: Worktree the patch is applied inside.
        patch_text: The unified-diff text to apply.

    Returns:
        A result dict with ``applied`` on success, or an error dict for an
        empty patch, missing worktree, path escape, or git-apply failure.
    """
    if not patch_text or not patch_text.strip():
        return _error("empty_patch")
    worktree = Path(worktree)
    if not worktree.is_dir():
        return _error("worktree_missing", path=str(worktree))
    for hit in _PATCH_PATH_RE.finditer(patch_text):
        cand = hit.group("path").strip()
        if cand.startswith("/") or ".." in Path(cand).parts:
            return _error("patch_path_escapes_worktree", offending=cand)
    try:
        proc = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=str(worktree),
            input=patch_text,
            text=True,
            capture_output=True,
            timeout=20.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return _error("git_apply_failed_to_spawn", detail=repr(exc))
    if proc.returncode != 0:
        return _error(
            "git_apply_rejected",
            stderr_tail=(proc.stderr or "").strip()[-2000:],
        )
    try:
        proc2 = subprocess.run(
            ["git", "apply", "-"],
            cwd=str(worktree),
            input=patch_text,
            text=True,
            capture_output=True,
            timeout=20.0,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _error("git_apply_timed_out", detail=repr(exc))
    if proc2.returncode != 0:
        return _error(
            "git_apply_unexpected_failure_after_check",
            stderr_tail=(proc2.stderr or "").strip()[-2000:],
        )
    return _ok({"applied": True})


def capture_worktree_cumulative_diff(worktree: Path) -> str | None:
    """Return ``git diff HEAD`` output for ``worktree``.

    * ``""``     — clean worktree.
    * ``<diff>`` — uncommitted-change diff.
    * ``None``   — git failure / not a repo; callers skip the
                   cumulative-diff check rather than aborting.

    Args:
        worktree: Worktree to capture the cumulative diff from.

    Returns:
        The ``git diff HEAD`` output, ``""`` for a clean worktree, or ``None``
        on git failure / not a repo.
    """
    worktree = Path(worktree)
    if not worktree.is_dir():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(worktree), "diff", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout or ""


def reset_worktree(worktree: Path) -> None:
    """Discard uncommitted changes + untracked files in ``worktree``.

    Args:
        worktree: Worktree to hard-reset and clean.
    """
    worktree = Path(worktree)
    if not worktree.is_dir():
        return
    try:
        subprocess.run(
            ["git", "reset", "--hard"],
            cwd=str(worktree),
            capture_output=True,
            timeout=20.0,
            check=False,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=str(worktree),
            capture_output=True,
            timeout=20.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


__all__ = [
    "apply_patch_in_worktree",
    "capture_worktree_cumulative_diff",
    "reset_worktree",
]
