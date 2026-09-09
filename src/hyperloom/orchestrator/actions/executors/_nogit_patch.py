# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared non-git patch apply / revert primitives."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess

from hyperloom.common.git_safety import safe_directory_args
from pathlib import Path
from typing import Any

from ...specialists.patch_safety import patch_file_targets

log = logging.getLogger(__name__)


# Candidate ``-p`` strip levels, tried in priority order (``-p1`` first).
_P_LEVELS: tuple[int, ...] = (1, 0, 2, 3, 4, 5, 6, 7, 8)

_PATCH_DEV_NULL = "/dev/null"

# Characters unsafe in filenames (replaced with ``_`` in rel_flat).
_UNSAFE_NAME_RE = re.compile(r"[/\\:<>\"?*|]")

# A git ``index <old>..<new>`` header whose *old* blob hash is all zeros.
_ZERO_OLD_INDEX_RE = re.compile(r"^index 0+\.\.")


def _old_path_after_index(lines: list[str], start: int) -> str | None:
    """Return the ``--- `` path token of the file block containing ``lines[start]``."""
    for line in lines[start + 1 :]:
        if line.startswith("--- "):
            return line[4:].strip().split("\t")[0]
        if line.startswith("diff --git ") or line.startswith("@@"):
            return None
    return None


def _sanitize_git_index_lines(patch_text: str) -> tuple[str, int]:
    """Drop git ``index`` lines whose all-zero old blob contradicts the ``---`` header."""
    lines = patch_text.splitlines(keepends=True)
    kept: list[str] = []
    dropped = 0
    for idx, line in enumerate(lines):
        if _ZERO_OLD_INDEX_RE.match(line):
            old_path = _old_path_after_index(lines, idx)
            if old_path is not None and old_path != _PATCH_DEV_NULL:
                dropped += 1
                continue
        kept.append(line)
    if not dropped:
        return patch_text, 0
    return "".join(kept), dropped


def _strip_path_prefix(path: str, level: int) -> str:
    """Drop ``level`` leading path components (mimics ``git apply -p<level>``)."""
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
            ["git", *safe_directory_args(["-C", str(path), "rev-parse", "--is-inside-work-tree"])],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return cp.returncode == 0 and cp.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _reverse_applies_cleanly(framework_root: Path, patch_path: Path) -> bool:
    """True when ``patch_path`` is already fully applied in ``framework_root``."""
    for lvl in _P_LEVELS:
        try:
            cp = subprocess.run(
                ["patch", f"-p{lvl}", "-R", "--dry-run", "-i", str(patch_path)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                cwd=str(framework_root),
                stdin=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        if cp.returncode == 0:
            return True
    return False


def _bak_name(patch_stem: str, rel_target: Path, seq: int) -> str:
    """Return a backup filename unique within a shared ``backup_root``."""
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
    """Apply ``patch_path`` into ``framework_root`` without git, backing up targets."""
    from ._apply_feedback import ApplyFeedback, read_patch_source_context

    try:
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        err_msg = f"cannot read patch file: {exc}"
        return (
            False,
            err_msg,
            [],
            ApplyFeedback(patch=str(patch_path), channel="nogit", tried_levels=[], stderr=err_msg),
        )

    # Feed the CLI a copy with contradicting index headers removed; keep the original path in feedback so advisories
    # point at what the author wrote.
    patch_input = patch_path
    sanitized_text, dropped_index_lines = _sanitize_git_index_lines(patch_text)
    if dropped_index_lines:
        backup_root.mkdir(parents=True, exist_ok=True)
        patch_input = backup_root / f"{patch_path.stem}.sanitized.diff"
        patch_input.write_text(sanitized_text, encoding="utf-8")
        log.info(
            "nogit patch: dropped %d placeholder git index line(s) from %s that contradicted the --- header",
            dropped_index_lines,
            patch_path.name,
        )

    # Detect strip level via dry-run; accumulate stderr per level for feedback.
    detected_level: int | None = None
    dry_run_stderrs: list[str] = []
    tried_levels: list[int] = []
    for lvl in _P_LEVELS:
        tried_levels.append(lvl)
        try:
            cp = subprocess.run(
                ["patch", f"-p{lvl}", "--dry-run", "-i", str(patch_input)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                cwd=str(framework_root),
                # ``patch`` prompts ("Assume -R? [n]") on an already-applied hunk; without a closed stdin it can
                # inherit the parent's and block until the 60s timeout.
                stdin=subprocess.DEVNULL,
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
        # Before reporting failure, distinguish "does not apply" from "already applied".
        if _reverse_applies_cleanly(framework_root, patch_input):
            log.info(
                "nogit patch: %s is already fully applied (clean reverse dry-run); treating as a no-op",
                patch_path.name,
            )
            return True, "", [], None
        combined_stderr = "\n".join(dry_run_stderrs)
        err_msg = f"patch --dry-run failed at all strip levels for {patch_path.name}"
        try:
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

    def _fail(err_message: str, recs: list[dict[str, Any]]) -> "tuple[bool, str, list[dict[str, Any]], Any]":
        """Return the canonical 4-tuple failure result with structured feedback."""
        return (
            False,
            err_message,
            recs,
            ApplyFeedback(
                patch=str(patch_path),
                channel="nogit",
                tried_levels=tried_levels,
                stderr=err_message,
            ),
        )

    framework_root_resolved = framework_root.resolve()
    backup_root.mkdir(parents=True, exist_ok=True)
    backups: list[dict[str, Any]] = []
    patch_stem = patch_path.stem

    def _resolve_target(raw: str) -> tuple[Path | None, Path | None, str]:
        """Resolve a raw diff-header path to (rel, abs, error)."""
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
            mode = abs_path.stat().st_mode & 0o7777
            shutil.copy2(abs_path, bak)
        except OSError as exc:
            return None, f"backup of {abs_path} failed: {exc}"
        return {
            "target": str(abs_path),
            "existed": True,
            "backup_path": str(bak),
            "revert_action": action,
            "mode": mode,
        }, ""

    for old_raw, new_raw in patch_file_targets(patch_text):
        is_create = old_raw == _PATCH_DEV_NULL or not old_raw
        is_delete = new_raw == _PATCH_DEV_NULL or not new_raw
        is_rename = not is_create and not is_delete and old_raw != new_raw

        if is_create:
            # New file created by patch: track for deletion on revert.
            if not new_raw or new_raw == _PATCH_DEV_NULL:
                continue
            rel_new, abs_new, err = _resolve_target(new_raw)
            if err:
                return _fail(err, backups)
            backups.append(
                {
                    "target": str(abs_new),
                    "existed": False,
                    "backup_path": None,
                    "revert_action": "delete",
                }
            )

        elif is_delete:
            # Existing file deleted by patch: back it up to restore on revert.
            if not old_raw or old_raw == _PATCH_DEV_NULL:
                continue
            rel_old, abs_old, err = _resolve_target(old_raw)
            if err:
                return _fail(err, backups)
            if abs_old.exists():
                rec, err = _backup_existing(abs_old, rel_old, "restore")  # type: ignore[arg-type]
                if err:
                    return _fail(err, backups)
                backups.append(rec)  # type: ignore[arg-type]
            else:
                backups.append(
                    {
                        "target": str(abs_old),
                        "existed": False,
                        "backup_path": None,
                        "revert_action": "delete",
                    }
                )

        elif is_rename:
            # Rename/move: back up old source (to restore on revert) and track new destination (to delete on revert).
            rel_old, abs_old, err = _resolve_target(old_raw)
            if err:
                return _fail(err, backups)
            rel_new, abs_new, err = _resolve_target(new_raw)
            if err:
                return _fail(err, backups)
            # Back up old source so it can be restored on revert.
            if abs_old.exists():  # type: ignore[union-attr]
                rec, err = _backup_existing(abs_old, rel_old, "restore_old")  # type: ignore[arg-type]
                if err:
                    return _fail(err, backups)
                backups.append(rec)  # type: ignore[arg-type]
            # Track new destination for deletion on revert.
            backups.append(
                {
                    "target": str(abs_new),
                    "existed": False,
                    "backup_path": None,
                    "revert_action": "delete",
                }
            )

        else:
            # Modification: back up existing target to restore on revert.
            target_raw = new_raw if (new_raw and new_raw != _PATCH_DEV_NULL) else old_raw
            if not target_raw or target_raw == _PATCH_DEV_NULL:
                continue
            rel_t, abs_t, err = _resolve_target(target_raw)
            if err:
                return _fail(err, backups)
            if abs_t.exists():  # type: ignore[union-attr]
                rec, err = _backup_existing(abs_t, rel_t, "restore")  # type: ignore[arg-type]
                if err:
                    return _fail(err, backups)
                backups.append(rec)  # type: ignore[arg-type]
            else:
                backups.append(
                    {
                        "target": str(abs_t),
                        "existed": False,
                        "backup_path": None,
                        "revert_action": "delete",
                    }
                )

    # Apply for real.
    rej_dir = backup_root / "rej"
    rej_dir.mkdir(parents=True, exist_ok=True)
    try:
        cp2 = subprocess.run(
            ["patch", f"-p{detected_level}", "--reject-file=-", "-i", str(patch_input)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            cwd=str(framework_root),
            stdin=subprocess.DEVNULL,
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
        # Collect any .rej files left next to the target files.
        rejected_hunks = _collect_rej_files(framework_root, patch_path)
        apply_stderr = cp2.stderr.strip() or cp2.stdout.strip()
        source_ctx = ""
        try:
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
    """Collect ``.rej`` reject files left by a failed ``patch`` apply."""
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
                # Best-effort scan: skip unreadable/racing .rej files.
                continue
    except Exception:  # noqa: BLE001
        log.debug("_collect_rej_files: scan failed for %s", patch_path, exc_info=True)
    return "\n\n".join(parts)


def _revert_patches_no_git(
    backups: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Restore or remove files recorded in ``backups`` (reverse of :func:`_apply_patch_no_git`)."""
    errors: list[str] = []
    for record in reversed(backups):
        target = Path(record["target"])
        bak = record.get("backup_path")
        action = record.get("revert_action")
        mode = record.get("mode")
        try:
            if action in ("restore", "restore_old") or (action is None and bak):
                # Modified / deleted / rename-source file: restore from backup.
                if bak:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bak, target)
                    if mode is not None:
                        target.chmod(mode)
                    if not target.is_file():
                        errors.append(f"restore: {target} missing after copy")
                    elif target.read_bytes() != Path(bak).read_bytes():
                        errors.append(f"restore: {target} content mismatch after copy")
            elif action == "delete" or (action is None and not bak):
                # New / rename-destination file: remove it.
                if target.exists():
                    target.unlink()
                if target.exists() or target.is_symlink():
                    errors.append(f"delete: {target} still exists after unlink")
        except OSError as exc:
            errors.append(f"{target}: {exc}")
            log.warning("nogit revert failed for %s: %s", target, exc)
    return not errors, errors


__all__ = [
    "_P_LEVELS",
    "_PATCH_DEV_NULL",
    "_apply_patch_no_git",
    "_collect_rej_files",
    "_is_git_tree",
    "_is_within",
    "_revert_patches_no_git",
    "_sanitize_git_index_lines",
    "_strip_path_prefix",
]
