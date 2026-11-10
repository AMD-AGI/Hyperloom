# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Snapshot, restore and discard the live framework source a fusion attempt edits."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Sequence

from kernelforge.utils.git import git

from .emit import _git_tracks, _is_fused_module_name

log = logging.getLogger("forge_fusion")


_FUSED_INVENTORY = ".fused_siblings"


def _read_fused_inventory(snapshot_dir: Path) -> set[str] | None:
    """Names of the fused-looking modules that existed BEFORE authoring."""
    listing = snapshot_dir / _FUSED_INVENTORY
    try:
        raw = listing.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # Unreadable and undecodable are the same answer here: this runs while cleaning up, so it must not raise, and
        # an inventory it cannot trust means it deletes nothing.
        return None
    return {line.strip() for line in raw.splitlines() if line.strip()}


def _needs_discard(exported_ok: bool, artifacts) -> bool:
    """Whether the framework is still carrying changes nobody exported."""
    if not exported_ok:
        return True
    return not (artifacts and artifacts.patch)


def _discard_failed_attempt(
    repo_root: str,
    source_file: str,
    out: Path,
    pristine_dir: str,
    extra_files: Sequence[str] = (),
) -> None:
    """Put the framework back as it was after a run that produced nothing usable."""
    if not repo_root or not source_file or not pristine_dir:
        return
    _snapshot_fusion_source(repo_root, source_file, out, subdir=".failed", extra_files=extra_files)
    for path in [source_file, *extra_files]:
        if path:
            _reset_fusion_source(repo_root, path, pristine_dir=pristine_dir)

    for candidate in _author_created_modules(source_file, pristine_dir):
        with contextlib.suppress(OSError):
            candidate.unlink()
            log.info("discarded author-created module %s", candidate.name)


def _author_created_modules(source_file: str, pristine_dir: str) -> list[Path]:
    """Fused-looking modules that appeared beside the source during this run."""
    if not source_file or not pristine_dir:
        return []
    pre_existing = _read_fused_inventory(Path(pristine_dir))
    if pre_existing is None:
        return []
    model_dir = Path(source_file).parent
    if not model_dir.is_dir():
        return []
    # The inventory lists the source's SIBLINGS, so the source is absent from it by construction -- and a model file
    # can itself be named like a fused module (``fused_moe.py``).
    source_resolved = Path(source_file).resolve()
    found: list[Path] = []
    for candidate in sorted(model_dir.glob("*.py")):
        if not _is_fused_module_name(candidate.name) or candidate.name in pre_existing:
            continue
        if candidate.resolve() == source_resolved:
            continue
        found.append(candidate)
    return found


def _snapshot_fusion_source(
    repo_root: str,
    source_file: str,
    out: Path,
    subdir: str = ".pristine",
    extra_files: Sequence[str] = (),
) -> str:
    """Copy the pristine framework source (pre-authoring) into ``out/<subdir>/<rel>``.

    ``extra_files`` are the further call-site files a multi-file fusion edits. They
    are snapshotted too so the non-git export can diff them; without a snapshot
    such a file diffs against nothing and exports as a bogus whole-file creation.
    """
    if not source_file or not Path(source_file).is_file():
        return ""
    pdir = out / subdir
    root = Path(repo_root).resolve() if repo_root else None

    def _rel(p: Path) -> str:
        if root:
            with contextlib.suppress(ValueError):
                return str(p.resolve().relative_to(root))
        return p.name

    def _snap(f: Path) -> bool:
        dest = pdir / _rel(f)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            return True
        except OSError as exc:
            log.warning("could not snapshot pristine fusion source %s: %s", f, exc)
            return False

    # The MAIN source snapshot is mandatory: without it, export would diff a missing snapshot ("") vs the live edited
    # file and emit the whole file as a bogus "new file".
    src = Path(source_file)
    if not _snap(src):
        return ""

    # The same reasoning applies to every other file this fusion edits, but one of them failing is not fatal: the
    # primary is what decides whether a patch can be produced at all.
    for path in extra_files:
        if path and Path(path).is_file() and Path(path).resolve() != src.resolve():
            _snap(Path(path))

    # Also snapshot any pre-existing *_fused*/*_fusion* module beside it, so export can tell an author-created NEW
    # module from a pre-existing framework file that merely matches the marker.
    model_dir = src.parent
    if model_dir.is_dir():
        siblings = [
            f for f in sorted(model_dir.glob("*.py")) if _is_fused_module_name(f.name) and f.resolve() != src.resolve()
        ]
        with contextlib.suppress(OSError):
            (pdir / _FUSED_INVENTORY).write_text("".join(f"{f.name}\n" for f in siblings), encoding="utf-8")
        for f in siblings:
            _snap(f)
    return str(pdir)


def _reset_fusion_source(repo_root: str, source_file: str, pristine_dir: str = "") -> None:
    """Revert the tracked model source file to its committed baseline (best-effort)."""
    import subprocess

    if not repo_root or not source_file:
        return
    # Use the SAME rel scheme as _snapshot_fusion_source (basename fallback when the source is not under repo_root) so
    # the pristine restore below can find the snap.
    try:
        rel = str(Path(source_file).resolve().relative_to(Path(repo_root).resolve()))
        rel_is_repo_relative = True
    except ValueError:
        rel = Path(source_file).name
        rel_is_repo_relative = False
    # Decide by whether the source file is git-TRACKED, NOT merely inside a work tree — a pip framework under a
    # project-local venv/site-packages is untracked, so `git checkout` is a no-op there and we must restore from the
    # snapshot. (Aligned with export_artifacts / restore_exported_changes.)
    if not _git_tracks(repo_root, source_file):
        if pristine_dir:
            snap = Path(pristine_dir) / rel
            if snap.is_file():
                with contextlib.suppress(OSError):
                    Path(source_file).write_text(snap.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        return  # untracked: nothing git can revert
    if not rel_is_repo_relative:
        return  # git checkout below needs a repo-relative path
    try:
        status = git(
            "-C",
            repo_root,
            "status",
            "--porcelain",
            "--",
            rel,
            check=False,
            timeout=30,
        )
        src = Path(repo_root) / rel
        if status.returncode == 0 and status.stdout.strip() and src.is_file():
            backup = (
                Path(os.environ.get("USER_DATA_PATH") or "/tmp")
                / "forge_fusion"
                / "source_backups"
                / str(time.time_ns())
                / rel
            )
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, backup)
            log.warning("backed up dirty fusion source before reset: %s", backup)
        git("-C", repo_root, "checkout", "--", rel, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("could not reset %s: %s", rel, exc)
