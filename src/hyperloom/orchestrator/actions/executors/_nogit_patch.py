# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared non-git patch apply primitives."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess

from hyperloom.common.git_safety import safe_directory_args
from hyperloom.common.unified_diff import parse_patch_manifest, strip_path_components
from pathlib import Path
from typing import Any, NamedTuple

from ...delivery.ledger import append_record, mark_prepared
from ...delivery import file_digest as _file_digest

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
        if line.startswith(("diff --git ", "@@")):
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

    A reverse dry-run (``patch -R --dry-run``) is the only probe POSIX
    ``patch`` offers: its forward exit code is non-zero for both "does not
    apply" and "previously applied". On its own it is not proof of exactness:
    ``patch`` matches with fuzz and at an offset, so it answers yes for a tree
    that merely resembles the post-state. Fuzz is disabled, and a hunk that
    only matched by shifting is rejected -- an offset means the surrounding
    lines are not the ones the patch was written against, so the tree is not
    the post-image even though every hunk found a home.

    Strictly a probe: ``--dry-run`` is passed at every level, so the tree is
    never mutated.

    Args:
        framework_root: The source-tree root the patch targets.
        patch_path: The unified-diff patch file to probe.

    Returns:
        ``True`` when the tree exactly holds the patch's post-state.
    """
    for lvl in _P_LEVELS:
        try:
            cp = subprocess.run(
                # ``--fuzz=0``: an approximate context match would answer yes
                # for a tree that only resembles the post-state.
                ["patch", f"-p{lvl}", "-R", "--fuzz=0", "--dry-run", "-i", str(patch_path)],
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
            if "offset" in cp.stdout:
                log.info(
                    "nogit patch: %s reverses only at an offset in %s; not treating it as applied",
                    patch_path.name,
                    framework_root,
                )
                return False
            return True
    return False


class _StripLevelProbe(NamedTuple):
    """What the forward ``patch --dry-run`` sweep over the candidate ``-p`` levels found."""

    #: The first level that applied cleanly, or ``None`` when none did.
    level: int | None
    tried_levels: list[int]
    #: One line per level probed, for the feedback a failed apply carries back to the author.
    stderrs: list[str]
    #: Why the sweep stopped early; empty when every candidate level was probed.
    cli_error: str


def _detect_strip_level(framework_root: Path, patch_path: Path) -> _StripLevelProbe:
    """Probe the candidate ``-p`` levels and report the first one a forward dry run accepts."""
    tried_levels: list[int] = []
    stderrs: list[str] = []
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
                # ``patch`` prompts ("Assume -R? [n]") on an already-applied hunk; without a closed stdin it can
                # inherit the parent's and block until the 60s timeout.
                stdin=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return _StripLevelProbe(None, tried_levels, stderrs, f"patch CLI unavailable or timed out: {exc}")
        stderrs.append(f"-p{lvl}: {cp.stderr.strip()}" if cp.stderr.strip() else f"-p{lvl}: (no stderr)")
        if cp.returncode == 0:
            return _StripLevelProbe(lvl, tried_levels, stderrs, "")
    return _StripLevelProbe(None, tried_levels, stderrs, "")


def _bak_name(patch_stem: str, rel_target: Path, seq: int) -> str:
    """Return a backup filename unique within a shared ``backup_root``."""
    flat = _UNSAFE_NAME_RE.sub("_", str(rel_target))
    safe_stem = _UNSAFE_NAME_RE.sub("_", patch_stem)
    return f"{safe_stem}__{flat}__{seq:04d}.bak"


def _rejected_hunks(reject_path: Path) -> str:
    """The hunk bodies ``patch`` could not apply; empty when it failed before rejecting any."""
    if not reject_path.exists():
        return ""
    return reject_path.read_text(encoding="utf-8", errors="replace").strip()


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

    probe = _detect_strip_level(framework_root, patch_input)
    detected_level, tried_levels = probe.level, probe.tried_levels
    if probe.cli_error:
        feedback = ApplyFeedback(
            patch=str(patch_path),
            channel="nogit",
            tried_levels=tried_levels,
            stderr=probe.cli_error,
        )
        return False, probe.cli_error, [], feedback
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
            if not mark_prepared(backup_root):
                err_msg = "backup prepare record could not be persisted"
                return (
                    False,
                    err_msg,
                    [],
                    ApplyFeedback(patch=str(patch_path), channel="nogit", tried_levels=tried_levels, stderr=err_msg),
                )
            log.info(
                "nogit patch: %s is already fully applied (clean reverse dry-run); treating as a no-op",
                patch_path.name,
            )
            return True, "", [], None
        combined_stderr = "\n".join(probe.stderrs)
        err_msg = f"patch --dry-run failed at all strip levels for {patch_path.name}"
        source_ctx = read_patch_source_context(patch_text, framework_root, radius=50)
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
        rel = Path(strip_path_components(raw, detected_level))  # type: ignore[arg-type]
        if rel.is_absolute() or ".." in rel.parts:
            return None, None, f"patch target escapes framework root: {raw}"
        abs_path = (framework_root_resolved / rel).resolve()
        if not _is_within(abs_path, framework_root_resolved):
            return None, None, f"patch target escapes framework root: {raw}"
        return rel, abs_path, ""

    def _record(record: dict[str, Any]) -> str:
        """Persist a backup record and add it to the in-memory list.

        Returns:
            An error string when the record could not be persisted; the caller
            must then abort rather than mutate an unrecorded file.
        """
        if not append_record(backup_root, record):
            return f"backup ledger write failed for {record.get('target')}"
        backups.append(record)
        return ""

    def _backup_existing(abs_path: Path, rel: Path) -> tuple[dict[str, Any] | None, str]:
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
            "revert_action": "restore",
            "mode": mode,
            "pre_image_sha256": _file_digest(abs_path),
        }, ""

    try:
        manifest = parse_patch_manifest(patch_text)
    except ValueError as exc:
        return _fail(f"unparseable patch: {exc}", backups)

    for entry in manifest:
        rel, abs_path, err = _resolve_target(entry.path)
        if err:
            return _fail(err, backups)
        if entry.is_new or not abs_path.exists():
            record = {"target": str(abs_path), "existed": False, "backup_path": None, "revert_action": "delete"}
        else:
            record, err = _backup_existing(abs_path, rel)
            if err:
                return _fail(err, backups)
        err = _record(record)
        if err:
            return _fail(err, backups)

    if not mark_prepared(backup_root):
        return _fail("backup prepare record could not be persisted", backups)

    # Rejects land under the backup root, where no other apply writes; ``--no-backup-if-mismatch`` drops ``.orig``.
    reject_path = backup_root / f"{patch_stem}.rej"
    apply_argv = ["patch", f"-p{detected_level}", f"--reject-file={reject_path}", "--no-backup-if-mismatch"]
    try:
        cp2 = subprocess.run(
            [*apply_argv, "-i", str(patch_input)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            cwd=str(framework_root),
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return _fail(f"patch apply failed: {exc}", backups)
    if cp2.returncode != 0:
        rejected_hunks = _rejected_hunks(reject_path)
        apply_stderr = cp2.stderr.strip() or cp2.stdout.strip()
        source_ctx = read_patch_source_context(patch_text, framework_root, radius=50)
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


__all__ = [
    "_P_LEVELS",
    "_PATCH_DEV_NULL",
    "_apply_patch_no_git",
    "_is_git_tree",
    "_is_within",
    "_sanitize_git_index_lines",
]
