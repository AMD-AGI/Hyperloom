# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared non-git patch apply / revert primitives.

Extracted from ``integrate_patch`` so both the EXPLORE specialist executor
and the FRAMEWORK-phase per-candidate executor can share the same
backup-based apply/revert channel without git.

Public surface
--------------
* :func:`_is_git_tree`          — probe whether a directory is inside a git work-tree.
* :func:`_apply_patch_no_git`   — apply a unified diff with POSIX ``patch``, backing up targets.
* :func:`_revert_patches_no_git` — restore backed-up targets (reverse of apply).

Supporting constants / helpers used by both callers:

* :data:`_P_LEVELS`         — ``-p`` strip levels tried in priority order.
* :data:`_PATCH_DEV_NULL`   — the sentinel ``/dev/null`` path in diff headers.
* :func:`_strip_path_prefix` — drop leading path components like ``git apply -p<n>``.
* :func:`_is_within`        — containment check (both paths pre-resolved).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ...specialists.patch_safety import patch_file_targets

log = logging.getLogger(__name__)


# Candidate ``-p`` strip levels, tried in priority order.  ``-p1`` is the
# git-native default and stays first for backward-compat; specialists author
# patches with heterogeneous path prefixes (``a/vllm/...`` -> -p1,
# ``b/_aiter_ops.py`` -> -p0/-p2, full absolute
# ``b/usr/local/lib/python3.12/dist-packages/vllm/...`` -> -p7), so we must
# auto-detect rather than assume a single level.
_P_LEVELS: tuple[int, ...] = (1, 0, 2, 3, 4, 5, 6, 7, 8)

_PATCH_DEV_NULL = "/dev/null"


def _strip_path_prefix(path: str, level: int) -> str:
    """Drop ``level`` leading path components (mimics ``git apply -p<level>``).

    Args:
        path: The diff-header path to strip.
        level: The number of leading components to drop (``<= 0`` is a no-op).

    Returns:
        The path with ``level`` leading components removed (or the basename
        when there are not enough components).
    """
    if level <= 0:
        return path
    parts = path.split("/")
    return "/".join(parts[level:]) if len(parts) > level else parts[-1]


def _is_within(child: Path, root: Path) -> bool:
    """True iff ``child`` is ``root`` or nested under it (both pre-resolved)."""
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def _is_git_tree(path: Path) -> bool:
    """True when ``path`` is inside an initialised git work tree."""
    try:
        cp = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return cp.returncode == 0 and cp.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _apply_patch_no_git(
    framework_root: Path,
    patch_path: Path,
    backup_root: Path,
) -> tuple[bool, str, list[dict[str, Any]]]:
    """Apply ``patch_path`` into ``framework_root`` without git, backing up targets.

    Uses the ``patch`` CLI (POSIX standard) with automatic ``-p`` strip-level
    detection via a dry-run pass, then backs up each target file before
    mutating, mirroring the artifact backup scheme.

    Args:
        framework_root: The source-tree root to apply into (need not be a git repo).
        patch_path: The unified-diff patch file to apply.
        backup_root: Directory under which target backups are written.

    Returns:
        A ``(ok, err, backups)`` triple: ``ok`` is ``True`` on success, ``err``
        is a human-readable failure description, and ``backups`` is a list of
        per-file backup records in the same format as :meth:`_apply_artifacts`
        (``target``, ``backup_path`` or ``None`` when the file was created).
    """
    # Detect strip level via dry-run.
    detected_level: int | None = None
    for lvl in _P_LEVELS:
        try:
            cp = subprocess.run(
                ["patch", f"-p{lvl}", "--dry-run", "-i", str(patch_path)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                cwd=str(framework_root),
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return False, f"patch CLI unavailable or timed out: {exc}", []
        if cp.returncode == 0:
            detected_level = lvl
            break
    if detected_level is None:
        return False, f"patch --dry-run failed at all strip levels for {patch_path.name}", []

    # Resolve target files to back up before mutation.
    try:
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return False, f"cannot read patch file: {exc}", []

    framework_root_resolved = framework_root.resolve()
    backup_root.mkdir(parents=True, exist_ok=True)
    backups: list[dict[str, Any]] = []
    for old, new in patch_file_targets(patch_text):
        target_raw = new if (new and new != _PATCH_DEV_NULL) else old
        if not target_raw or target_raw == _PATCH_DEV_NULL:
            continue
        rel_target = Path(_strip_path_prefix(target_raw, detected_level))
        if rel_target.is_absolute() or ".." in rel_target.parts:
            return False, f"patch target escapes framework root: {target_raw}", backups
        target = (framework_root_resolved / rel_target).resolve()
        if not _is_within(target, framework_root_resolved):
            return False, f"patch target escapes framework root: {target_raw}", backups
        existed = target.exists()
        record: dict[str, Any] = {"target": str(target), "existed": existed, "backup_path": None}
        if existed:
            bak = backup_root / f"{len(backups):03d}_{target.name}.bak"
            try:
                shutil.copy2(target, bak)
                record["backup_path"] = str(bak)
            except OSError as exc:
                return False, f"backup of {target} failed: {exc}", backups
        backups.append(record)

    # Apply for real.
    try:
        cp2 = subprocess.run(
            ["patch", f"-p{detected_level}", "-i", str(patch_path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            cwd=str(framework_root),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"patch apply failed: {exc}", backups
    if cp2.returncode != 0:
        return False, cp2.stderr.strip() or cp2.stdout.strip(), backups
    return True, "", backups


def _revert_patches_no_git(backups: list[dict[str, Any]]) -> None:
    """Restore or remove files recorded in ``backups`` (reverse of :func:`_apply_patch_no_git`).

    Iterates in reverse so multi-file patches unwind in the correct order.
    Errors are logged but never raised — best-effort, matching :meth:`_revert_artifacts`.

    Args:
        backups: The per-file backup records produced by :func:`_apply_patch_no_git`.
    """
    for record in reversed(backups):
        target = Path(record["target"])
        bak = record.get("backup_path")
        try:
            if bak:
                shutil.copy2(bak, target)
            elif target.exists():
                target.unlink()
        except OSError as exc:
            log.warning("integrate_patch: no-git revert failed for %s: %s", target, exc)


__all__ = [
    "_P_LEVELS",
    "_PATCH_DEV_NULL",
    "_apply_patch_no_git",
    "_is_git_tree",
    "_is_within",
    "_revert_patches_no_git",
    "_strip_path_prefix",
]
