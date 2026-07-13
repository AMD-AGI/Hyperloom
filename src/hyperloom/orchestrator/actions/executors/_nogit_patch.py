# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared non-git patch apply / revert primitives.

Extracted from ``integrate_patch`` so both the EXPLORE specialist executor
and the FRAMEWORK-phase per-candidate executor can share the same
backup-based apply/revert channel without git.

Public surface
--------------
* :func:`_is_git_tree`           — probe whether a directory is inside a git work-tree.
* :func:`_apply_patch_no_git`    — apply a unified diff with POSIX ``patch``, backing up targets.
* :func:`_revert_patches_no_git` — restore backed-up targets (reverse of apply).

Supporting constants / helpers used by both callers:

* :data:`_P_LEVELS`         — ``-p`` strip levels tried in priority order.
* :data:`_PATCH_DEV_NULL`   — the sentinel ``/dev/null`` path in diff headers.
* :func:`_strip_path_prefix` — drop leading path components like ``git apply -p<n>``.
* :func:`_is_within`        — containment check (both paths pre-resolved).

Backup naming (#3 fix)
----------------------
Backup files are named ``<patch_stem>__<rel_flat>__<seq:04d>.bak`` where
``rel_flat`` is the target's relative path with ``/`` replaced by ``__`` and
``seq`` is a caller-supplied global offset (``seq_offset``) plus the record
index within this apply call.  Callers that accumulate backups across multiple
``_apply_patch_no_git`` calls (``integrate_patch``, ``framework_agent``) pass
``seq_offset=len(existing_backups)`` so backup names are globally unique within
a shared ``backup_root`` directory even when different patches touch files with
the same basename.

Rename / move revert (#4 fix)
------------------------------
When a patch hunk renames a file (``---`` old path ≠ ``+++`` new path, neither
is ``/dev/null``), ``_apply_patch_no_git`` backs up *both* the old source file
(so its content can be restored on revert) *and* tracks the new destination
(so it can be deleted on revert).  Each backup record carries a ``revert_action``
key:

* ``"restore"`` — copy ``backup_path`` back to ``target`` (modified/created files).
* ``"delete"``  — remove ``target`` (the rename destination, which did not exist
  before the patch).
* ``"restore_old"`` — copy ``backup_path`` back to ``target`` (the rename source,
  which existed before the patch and must be put back).

``_revert_patches_no_git`` dispatches on ``revert_action`` (falling back to the
original ``backup_path``-present → restore / absent → delete heuristic for
records written by older code).
"""

from __future__ import annotations

import logging
import re
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

# Characters unsafe in filenames (replaced with ``_`` in rel_flat).
_UNSAFE_NAME_RE = re.compile(r"[/\\:<>\"?*|]")


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


def _bak_name(patch_stem: str, rel_target: Path, seq: int) -> str:
    """Return a backup filename unique within a shared ``backup_root``.

    Encodes both the patch identity and the target path so two patches that
    touch a file with the same basename never collide.

    Args:
        patch_stem: ``patch_path.stem`` of the originating patch file.
        rel_target: Relative path of the target within ``framework_root``.
        seq: Globally unique sequence number (caller-maintained).

    Returns:
        A safe filename string ending in ``.bak``.
    """
    flat = _UNSAFE_NAME_RE.sub("_", str(rel_target))
    safe_stem = _UNSAFE_NAME_RE.sub("_", patch_stem)
    return f"{safe_stem}__{flat}__{seq:04d}.bak"


def _apply_patch_no_git(
    framework_root: Path,
    patch_path: Path,
    backup_root: Path,
    *,
    seq_offset: int = 0,
) -> "tuple[bool, str, list[dict[str, Any]], Any]":
    """Apply ``patch_path`` into ``framework_root`` without git, backing up targets.

    Uses the ``patch`` CLI (POSIX standard) with automatic ``-p`` strip-level
    detection via a dry-run pass, then backs up each target file before
    mutating, mirroring the artifact backup scheme.

    Backup names are globally unique across multiple calls sharing the same
    ``backup_root`` when callers pass ``seq_offset=len(accumulated_backups)``
    (see module docstring for the naming scheme).

    Rename/move hunks (``---`` old ≠ ``+++`` new, neither ``/dev/null``) are
    handled completely: the old source file is backed up so revert can restore
    it; the new destination is tracked so revert can delete it.

    Args:
        framework_root: The source-tree root to apply into (need not be a git repo).
        patch_path: The unified-diff patch file to apply.
        backup_root: Directory under which target backups are written.
        seq_offset: Starting sequence number for backup filenames within this call.
            Pass ``len(accumulated_backups)`` when reusing a ``backup_root`` across
            multiple ``_apply_patch_no_git`` calls to avoid name collisions.

    Returns:
        A ``(ok, err, backups, feedback)`` four-tuple: ``ok`` is ``True`` on
        success, ``err`` is a human-readable failure description, ``backups``
        is a list of per-file backup records, and ``feedback`` is an
        :class:`~._apply_feedback.ApplyFeedback` instance on failure or
        ``None`` on success.  Backup record fields:

        * ``"target"`` — absolute path of the affected file.
        * ``"existed"`` — whether the file existed before the patch.
        * ``"backup_path"`` — path of the saved copy (``None`` for new files).
        * ``"revert_action"`` — one of ``"restore"``, ``"delete"``, or
          ``"restore_old"`` (see module docstring).

    .. note::
        Existing callers that unpack only three items can use
        ``ok, err, backups, *_ = _apply_patch_no_git(...)`` for zero-change
        compatibility.
    """
    from ._apply_feedback import ApplyFeedback, read_patch_source_context

    # Detect strip level via dry-run; accumulate stderr per level for feedback.
    detected_level: int | None = None
    dry_run_stderrs: list[str] = []
    tried_levels: list[int] = []
    for lvl in _P_LEVELS:
        tried_levels.append(lvl)
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
            err_msg = f"patch CLI unavailable or timed out: {exc}"
            feedback = ApplyFeedback(
                patch=str(patch_path),
                channel="nogit",
                tried_levels=tried_levels,
                stderr=err_msg,
            )
            return False, err_msg, [], feedback
        dry_run_stderrs.append(f"-p{lvl}: {cp.stderr.strip()}" if cp.stderr.strip() else f"-p{lvl}: (no stderr)")
        if cp.returncode == 0:
            detected_level = lvl
            break
    if detected_level is None:
        combined_stderr = "\n".join(dry_run_stderrs)
        err_msg = f"patch --dry-run failed at all strip levels for {patch_path.name}"
        try:
            patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
            source_ctx = read_patch_source_context(patch_text, framework_root, radius=50)
        except Exception:  # noqa: BLE001
            source_ctx = ""
        feedback = ApplyFeedback(
            patch=str(patch_path),
            channel="nogit",
            tried_levels=tried_levels,
            stderr=combined_stderr,
            source_context=source_ctx,
        )
        return False, err_msg, [], feedback

    # Resolve target files to back up before mutation.
    try:
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return False, f"cannot read patch file: {exc}", []

    framework_root_resolved = framework_root.resolve()
    backup_root.mkdir(parents=True, exist_ok=True)
    backups: list[dict[str, Any]] = []
    patch_stem = patch_path.stem

    def _resolve_target(raw: str) -> tuple[Path | None, Path | None, str]:
        """Resolve a raw diff-header path to (rel, abs, error).

        Returns ``(None, None, err_msg)`` on failure.
        """
        rel = Path(_strip_path_prefix(raw, detected_level))  # type: ignore[arg-type]
        if rel.is_absolute() or ".." in rel.parts:
            return None, None, f"patch target escapes framework root: {raw}"
        abs_path = (framework_root_resolved / rel).resolve()
        if not _is_within(abs_path, framework_root_resolved):
            return None, None, f"patch target escapes framework root: {raw}"
        return rel, abs_path, ""

    def _backup_existing(abs_path: Path, rel: Path, action: str) -> tuple[dict[str, Any] | None, str]:
        """Copy ``abs_path`` to a uniquely named backup and return the record."""
        seq = seq_offset + len(backups)
        bak = backup_root / _bak_name(patch_stem, rel, seq)
        try:
            shutil.copy2(abs_path, bak)
        except OSError as exc:
            return None, f"backup of {abs_path} failed: {exc}"
        return {
            "target": str(abs_path),
            "existed": True,
            "backup_path": str(bak),
            "revert_action": action,
        }, ""

    for old_raw, new_raw in patch_file_targets(patch_text):
        is_create = old_raw == _PATCH_DEV_NULL or not old_raw
        is_delete = new_raw == _PATCH_DEV_NULL or not new_raw
        is_rename = (
            not is_create
            and not is_delete
            and old_raw != new_raw
        )

        if is_create:
            # New file created by patch: track for deletion on revert.
            if not new_raw or new_raw == _PATCH_DEV_NULL:
                continue
            rel_new, abs_new, err = _resolve_target(new_raw)
            if err:
                return False, err, backups
            backups.append({
                "target": str(abs_new),
                "existed": False,
                "backup_path": None,
                "revert_action": "delete",
            })

        elif is_delete:
            # Existing file deleted by patch: back it up to restore on revert.
            if not old_raw or old_raw == _PATCH_DEV_NULL:
                continue
            rel_old, abs_old, err = _resolve_target(old_raw)
            if err:
                return False, err, backups
            if abs_old.exists():
                rec, err = _backup_existing(abs_old, rel_old, "restore")  # type: ignore[arg-type]
                if err:
                    return False, err, backups
                backups.append(rec)  # type: ignore[arg-type]
            else:
                backups.append({
                    "target": str(abs_old),
                    "existed": False,
                    "backup_path": None,
                    "revert_action": "delete",
                })

        elif is_rename:
            # Rename/move: back up old source (to restore on revert) and
            # track new destination (to delete on revert).
            rel_old, abs_old, err = _resolve_target(old_raw)
            if err:
                return False, err, backups
            rel_new, abs_new, err = _resolve_target(new_raw)
            if err:
                return False, err, backups
            # Back up old source so it can be restored on revert.
            if abs_old.exists():  # type: ignore[union-attr]
                rec, err = _backup_existing(abs_old, rel_old, "restore_old")  # type: ignore[arg-type]
                if err:
                    return False, err, backups
                backups.append(rec)  # type: ignore[arg-type]
            # Track new destination for deletion on revert.
            backups.append({
                "target": str(abs_new),
                "existed": False,
                "backup_path": None,
                "revert_action": "delete",
            })

        else:
            # Modification: back up existing target to restore on revert.
            target_raw = new_raw if (new_raw and new_raw != _PATCH_DEV_NULL) else old_raw
            if not target_raw or target_raw == _PATCH_DEV_NULL:
                continue
            rel_t, abs_t, err = _resolve_target(target_raw)
            if err:
                return False, err, backups
            if abs_t.exists():  # type: ignore[union-attr]
                rec, err = _backup_existing(abs_t, rel_t, "restore")  # type: ignore[arg-type]
                if err:
                    return False, err, backups
                backups.append(rec)  # type: ignore[arg-type]
            else:
                backups.append({
                    "target": str(abs_t),
                    "existed": False,
                    "backup_path": None,
                    "revert_action": "delete",
                })

    # Apply for real.  Use --reject to write .rej files for failed hunks so we
    # can collect them for reauthor feedback rather than silently losing them.
    rej_dir = backup_root / "rej"
    rej_dir.mkdir(parents=True, exist_ok=True)
    try:
        cp2 = subprocess.run(
            ["patch", f"-p{detected_level}", "--reject-file=-", "-i", str(patch_path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            cwd=str(framework_root),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        err_msg = f"patch apply failed: {exc}"
        feedback = ApplyFeedback(
            patch=str(patch_path),
            channel="nogit",
            tried_levels=tried_levels,
            stderr=err_msg,
        )
        return False, err_msg, backups, feedback
    if cp2.returncode != 0:
        # Collect any .rej files that were left next to the target files.
        rejected_hunks = _collect_rej_files(framework_root, patch_path)
        apply_stderr = cp2.stderr.strip() or cp2.stdout.strip()
        # Build source context from the patch text.
        source_ctx = ""
        try:
            patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
            source_ctx = read_patch_source_context(patch_text, framework_root, radius=50)
        except Exception:  # noqa: BLE001
            pass
        feedback = ApplyFeedback(
            patch=str(patch_path),
            channel="nogit",
            tried_levels=[detected_level],
            stderr=apply_stderr,
            rejected_hunks=rejected_hunks,
            source_context=source_ctx,
        )
        return False, apply_stderr, backups, feedback
    return True, "", backups, None


def _collect_rej_files(framework_root: Path, patch_path: Path) -> str:
    """Collect ``.rej`` reject files left by a failed ``patch`` apply.

    The POSIX ``patch`` tool writes ``<target>.rej`` alongside each file that
    had failing hunks.  This helper scans ``framework_root`` for any ``.rej``
    file whose mtime is recent (within 60 s), reads them all, then removes
    them so they do not interfere with future apply attempts.

    Args:
        framework_root: The source-tree root that was patched into.
        patch_path: The patch file that was applied (used for logging only).

    Returns:
        A concatenated string of all ``.rej`` file contents, or ``""`` when
        none were found.
    """
    import time

    cutoff = time.time() - 60.0
    parts: list[str] = []
    try:
        for rej in sorted(framework_root.rglob("*.rej")):
            try:
                if rej.stat().st_mtime >= cutoff:
                    content = rej.read_text(encoding="utf-8", errors="replace").strip()
                    if content:
                        parts.append(f"# {rej.relative_to(framework_root)}\n{content}")
                    rej.unlink(missing_ok=True)
            except OSError:
                # Best-effort scan: skip unreadable/racing .rej files silently.
                continue
    except Exception:  # noqa: BLE001
        log.debug("_collect_rej_files: scan failed for %s", patch_path, exc_info=True)
    return "\n\n".join(parts)


def _revert_patches_no_git(backups: list[dict[str, Any]]) -> None:
    """Restore or remove files recorded in ``backups`` (reverse of :func:`_apply_patch_no_git`).

    Iterates in reverse so multi-file patches unwind in the correct order.
    Dispatches on the ``revert_action`` field when present; falls back to the
    legacy heuristic (``backup_path`` present → restore, absent → delete) for
    records produced by older code.  Errors are logged but never raised —
    best-effort, matching :meth:`_revert_artifacts`.

    Args:
        backups: The per-file backup records produced by :func:`_apply_patch_no_git`.
    """
    for record in reversed(backups):
        target = Path(record["target"])
        bak = record.get("backup_path")
        action = record.get("revert_action")
        try:
            if action == "restore" or (action is None and bak):
                # Modified / deleted file: restore from backup.
                if bak:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bak, target)
            elif action == "restore_old":
                # Rename source: restore the original file at its old path.
                if bak:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bak, target)
            elif action == "delete" or (action is None and not bak):
                # New / rename-destination file: remove it.
                if target.exists():
                    target.unlink()
        except OSError as exc:
            log.warning("integrate_patch: no-git revert failed for %s: %s", target, exc)


__all__ = [
    "_P_LEVELS",
    "_PATCH_DEV_NULL",
    "_apply_patch_no_git",
    "_collect_rej_files",
    "_is_git_tree",
    "_is_within",
    "_revert_patches_no_git",
    "_strip_path_prefix",
]
