# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared non-git patch apply / revert primitives.

Lets the specialist and upstream-PR patch
executor share one backup-based apply/revert channel without git.

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
* :func:`_sanitize_git_index_lines` — drop ``index`` headers that contradict ``---``.

Placeholder git index headers
-----------------------------
Unlike ``git apply``, GNU ``patch`` honours the ``index <old>..<new>`` line and
reads an all-zero *old* blob hash as a file creation. Specialists write
placeholder hashes, so a modification hunk can arrive as
``index 0000000..1111111`` alongside ``--- a/path``, and ``patch`` then refuses
it with ``... which already exists!``. :func:`_sanitize_git_index_lines` drops
such contradicting lines before the CLI sees the patch; genuine creations
(``--- /dev/null``) keep theirs.

Backup naming
-------------
Backup files are named ``<patch_stem>__<rel_flat>__<seq:04d>.bak`` where
``rel_flat`` is the target's relative path with ``/`` replaced by ``__`` and
``seq`` is a caller-supplied offset (``seq_offset``) plus the record index.
Callers that accumulate backups across multiple ``_apply_patch_no_git`` calls
pass ``seq_offset=len(existing_backups)`` so names are globally unique within a
shared ``backup_root`` even when patches touch files with the same basename.

Rename / move revert
--------------------
For a rename hunk (``---`` old path != ``+++`` new path, neither ``/dev/null``),
``_apply_patch_no_git`` backs up the old source file (to restore on revert) and
tracks the new destination (to delete on revert). Each backup record carries a
``revert_action`` key:

* ``"restore"`` — copy ``backup_path`` back to ``target`` (modified/created files).
* ``"delete"``  — remove ``target`` (the rename destination, which did not exist
  before the patch).
* ``"restore_old"`` — copy ``backup_path`` back to ``target`` (the rename source,
  which existed before the patch and must be put back).

``_revert_patches_no_git`` dispatches on ``revert_action`` (falling back to the
``backup_path``-present → restore / absent → delete heuristic for older records).
"""

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
# Specialists author patches with heterogeneous path prefixes, so we auto-detect
# rather than assume a single level.
_P_LEVELS: tuple[int, ...] = (1, 0, 2, 3, 4, 5, 6, 7, 8)

_PATCH_DEV_NULL = "/dev/null"

# Characters unsafe in filenames (replaced with ``_`` in rel_flat).
_UNSAFE_NAME_RE = re.compile(r"[/\\:<>\"?*|]")

# A git ``index <old>..<new>`` header whose *old* blob hash is all zeros.
_ZERO_OLD_INDEX_RE = re.compile(r"^index 0+\.\.")


def _old_path_after_index(lines: list[str], start: int) -> str | None:
    """Return the ``--- `` path token of the file block containing ``lines[start]``.

    Scans forward from an ``index`` line to that block's ``--- `` header,
    stopping at the next ``diff --git`` header or the first hunk marker so a
    later block's header is never attributed to this one.

    Args:
        lines: The patch text split into lines.
        start: Index of the ``index`` line to resolve.

    Returns:
        The raw pre-image path token, or ``None`` when the block has no
        ``--- `` header.
    """
    for line in lines[start + 1 :]:
        if line.startswith("--- "):
            return line[4:].strip().split("\t")[0]
        if line.startswith("diff --git ") or line.startswith("@@"):
            return None
    return None


def _sanitize_git_index_lines(patch_text: str) -> tuple[str, int]:
    """Drop git ``index`` lines whose all-zero old blob contradicts the ``---`` header.

    GNU ``patch`` reads an all-zero *old* blob hash as "this hunk creates the
    file" and then refuses the hunk with ``The next patch would create the file
    X, which already exists!`` -- even though the accompanying ``--- a/X``
    header says X is being *modified*. ``git apply`` ignores the index line
    entirely, so such a patch applies through the git channel and fails only
    here, which makes the failure look like a bad patch rather than a header
    disagreement.

    Specialists emit placeholder index lines rather than real blob hashes, so
    the contradiction is common enough to absorb rather than reject. The
    ``---``/``+++`` headers are the authoritative unified-diff surface and GNU
    ``patch`` does not need the index line, so a contradicting one is dropped.

    A genuine creation hunk carries ``--- /dev/null`` and keeps its index line,
    so real file creations are unaffected.

    Args:
        patch_text: The unified-diff text to sanitize.

    Returns:
        A ``(sanitized_text, dropped_count)`` pair. When nothing contradicts,
        ``dropped_count`` is ``0`` and the text is returned unmodified.
    """
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
    """True when ``patch_path`` is already fully applied in ``framework_root``.

    A reverse dry-run (``patch -R --dry-run``) succeeds only when every hunk's
    *post*-state is already present in the tree — i.e. the tree is exactly what
    a successful forward apply would have produced. That makes it the reliable
    already-applied probe: POSIX ``patch`` exits non-zero for both "does not
    apply" and "previously applied", so the forward exit code alone cannot tell
    the two apart.

    Strictly a probe: ``--dry-run`` is passed at every level, so the tree is
    never mutated. Partial overlap is correctly rejected — a patch whose hunks
    are only *partly* present fails the reverse check and stays a real failure.

    Args:
        framework_root: The source-tree root the patch targets.
        patch_path: The unified-diff patch file to probe.

    Returns:
        ``True`` when some strip level reverse-applies cleanly.
    """
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

    Rename/move hunks back up the old source file (to restore on revert) and
    track the new destination (to delete on revert).

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

    # Feed the CLI a copy with contradicting index headers removed; keep the
    # original path in feedback so advisories point at what the author wrote.
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
                # ``patch`` prompts ("Assume -R? [n]") on an already-applied
                # hunk; without a closed stdin it can inherit the parent's and
                # block until the 60s timeout.
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
        # Before reporting failure, distinguish "does not apply" from "already
        # applied". A specialist commonly writes a superset patch AND the subset
        # it contains (e.g. an FLA-layout revert plus that same revert bundled
        # with a config fix). Applying the superset makes every later subset
        # hunk a no-op, and POSIX ``patch`` reports that as "Reversed (or
        # previously applied) patch detected ... Skipping patch" with a non-zero
        # exit -- indistinguishable, at the exit code, from a patch that simply
        # does not fit. Treating it as a hard failure aborted the whole apply and
        # reverted a combo that was in fact fully and correctly applied.
        #
        # A clean *reverse* dry-run is the unambiguous already-applied probe: it
        # succeeds only when every hunk's post-state is already present in the
        # tree, which is exactly the state a forward apply would produce. In that
        # case the apply is a satisfied no-op -- report success with no backups
        # (the patch that really made those edits owns the backups needed for a
        # correct revert).
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
            # Rename/move: back up old source (to restore on revert) and
            # track new destination (to delete on revert).
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

    # Apply for real. ``--reject-file=-`` discards rejects, so the failure path's
    # ``.rej`` sweep only picks up sidecars left by other tooling.
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
                # Best-effort scan: skip unreadable/racing .rej files.
                continue
    except Exception:  # noqa: BLE001
        log.debug("_collect_rej_files: scan failed for %s", patch_path, exc_info=True)
    return "\n\n".join(parts)


def _revert_patches_no_git(
    backups: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Restore or remove files recorded in ``backups`` (reverse of :func:`_apply_patch_no_git`).

    Iterates in reverse so multi-file patches unwind in the correct order.
    Dispatches on the ``revert_action`` field when present; falls back to the
    legacy heuristic (``backup_path`` present → restore, absent → delete) for
    records produced by older code. Every path is re-read after the restore, so
    a partial restore is reported rather than mistaken for success.

    Args:
        backups: The per-file backup records produced by :func:`_apply_patch_no_git`.

    Returns:
        A ``(ok, errors)`` tuple; ``ok`` is ``True`` only when every record
        restored and verified, and ``errors`` carries one entry per failure.
    """
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
