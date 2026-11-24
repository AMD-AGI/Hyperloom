# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atomic, all-or-nothing landing of a content-addressed snapshot inside its authorized deploy roots."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, NamedTuple

from hyperloom_common.unified_diff import PatchOp, PatchTarget

from .patch_jit_cache import _within_root


def path_hash(path: Path) -> str:
    """Return a short, stable hash for ``path`` to disambiguate backups."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


class PatchRefused(Exception):
    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(str(result.get("error") or "kernel patch refused"))
        self.result = result


def contained_dest(
    repo_root: Path,
    rel_path: str,
    *,
    deploy_roots: Iterable[Path] | None = None,
) -> Path:
    """Resolve ``rel_path`` under its coordinate and authorized deploy roots.

    ``deploy_roots=None`` keeps the coordinate-only check for legacy callers; an
    empty iterable rejects every destination.
    """
    if rel_path.startswith("/"):
        raise ValueError(f"absolute path not allowed in patch: {rel_path}")
    if ".." in Path(rel_path).parts:
        raise ValueError(f"'..' not allowed in patch path: {rel_path}")
    root = repo_root.resolve()
    dest = (root / rel_path).resolve()
    try:
        dest.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"patch path escapes repo root {root}: {rel_path}") from exc
    if deploy_roots is not None:
        allowed = [Path(path).resolve() for path in deploy_roots]
        if not any(_within_root(dest, allowed_root) for allowed_root in allowed):
            roots_text = ", ".join(str(path) for path in allowed) or "<none>"
            raise ValueError(
                f"patch path is outside authorized deploy roots: {rel_path}; authorized roots: {roots_text}"
            )
    return dest


def apply_snapshot(
    *,
    descriptors: list[PatchTarget],
    snapshot_dir: str | Path,
    repo_root: str | Path,
    backup_dir: str | Path,
    deploy_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Apply a content-addressed snapshot atomically (all-or-nothing).

    Each descriptor asserts a final filesystem state rather than replaying a diff,
    and the whole set pre-flights before any write: a failure restores every
    touched path from backup, so the repo is never left partially applied and
    there is no fuzzy fallback.
    """
    root = Path(repo_root)
    allowed_roots = None if deploy_roots is None else [Path(path) for path in deploy_roots]
    roots = _SnapshotRoots(
        snapshot_dir=Path(snapshot_dir),
        repo_root=root,
        deploy_roots=allowed_roots,
        symlink_roots=[root] if allowed_roots is None else allowed_roots,
    )
    try:
        staged = _stage_snapshot_writes(descriptors, roots)
    except PatchRefused as refusal:
        return refusal.result

    backups: list[dict[str, Any]] = []
    touched: list[str] = []
    for write in staged:
        try:
            backups.append(_back_up_before_write(write, Path(backup_dir)))
            _land_staged_write(write, roots)
        except (OSError, ValueError) as exc:
            _restore_live_tree(backups)
            return {
                "status": "failed",
                "error": f"apply failed at {write.descriptor.path}: {exc}; repo restored",
                "path": write.descriptor.path,
            }
        touched.append(str(write.dest))

    return {"status": "ok", "source_backups": backups, "touched": touched}


class _SnapshotRoots(NamedTuple):
    snapshot_dir: Path
    repo_root: Path
    deploy_roots: list[Path] | None
    symlink_roots: list[Path]


class _StagedWrite(NamedTuple):
    descriptor: PatchTarget
    dest: Path
    source: Path | None


def _stage_snapshot_writes(descriptors: list[PatchTarget], roots: _SnapshotRoots) -> list[_StagedWrite]:
    staged: list[_StagedWrite] = []
    try:
        for desc in descriptors:
            dest = contained_dest(roots.repo_root, desc.path, deploy_roots=roots.deploy_roots)
            source: Path | None = None
            if desc.op is PatchOp.WRITE:
                source = roots.snapshot_dir / desc.path
                _refuse_unwritable_snapshot_source(source, dest, desc.path, roots.symlink_roots)
            staged.append(_StagedWrite(desc, dest, source))
    except ValueError as exc:
        raise PatchRefused({"status": "failed", "error": str(exc), "path": "<pre-flight>"}) from exc
    return staged


def _refuse_unwritable_snapshot_source(
    source: Path,
    dest: Path,
    rel_path: str,
    symlink_roots: list[Path],
) -> None:
    if not source.is_symlink():
        if not source.exists():
            raise PatchRefused(
                {"status": "failed", "error": f"snapshot missing content for {rel_path}", "path": rel_path}
            )
        return
    link_target = os.readlink(source)
    resolved_link = (Path(link_target) if os.path.isabs(link_target) else dest.parent / link_target).resolve()
    if any(_within_root(resolved_link, allowed_root) for allowed_root in symlink_roots):
        return
    raise PatchRefused(
        {
            "status": "failed",
            "error": (f"snapshot symlink target is outside authorized deploy roots: {rel_path} -> {link_target}"),
            "path": rel_path,
        }
    )


def _back_up_before_write(write: _StagedWrite, backup_dir: Path) -> dict[str, Any]:
    dest = write.dest
    existed = dest.exists() or dest.is_symlink()
    if not existed and write.descriptor.op is not PatchOp.DELETE:
        return {"path": str(dest), "backup_path": None, "disposition": "added"}
    backup_path = None
    if existed:
        backup_dest = backup_dir / "source" / f"{path_hash(dest)}_{dest.name}"
        backup_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, backup_dest, follow_symlinks=False)
        backup_path = str(backup_dest)
    disposition = "deleted" if write.descriptor.op is PatchOp.DELETE else "modified"
    return {"path": str(dest), "backup_path": backup_path, "disposition": disposition}


def _land_staged_write(write: _StagedWrite, roots: _SnapshotRoots) -> None:
    dest = write.dest
    if write.descriptor.op is PatchOp.DELETE:
        dest.unlink(missing_ok=True)
        return
    # Re-validate containment right before the write (close TOCTOU window).
    contained_dest(roots.repo_root, write.descriptor.path, deploy_roots=roots.deploy_roots)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        dest.unlink()
    if write.source.is_symlink():
        dest.symlink_to(os.readlink(write.source))
        return
    shutil.copy2(write.source, dest, follow_symlinks=False)
    if write.descriptor.mode:
        try:
            dest.chmod(int(write.descriptor.mode, 8))
        except (ValueError, OSError):
            # Restoring mode bits is best-effort.
            pass


def _restore_live_tree(backups: list[dict[str, Any]]) -> None:
    for entry in reversed(backups):
        live = Path(entry["path"])
        if entry["disposition"] == "added":
            live.unlink(missing_ok=True)
            continue
        # modified / deleted -> restore old bytes from backup
        backup_path = entry.get("backup_path")
        if backup_path and Path(backup_path).exists():
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, live, follow_symlinks=False)
