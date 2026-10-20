# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The live tree a kernel patch edits and the backup tree that can undo it.

Owns where a patch may write, where the bytes it replaces are kept, which paths
read back from an untrusted manifest may be copied from or deleted, and how a
content-addressed snapshot lands all-or-nothing.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, NamedTuple

from hyperloom.common.unified_diff import PatchOp, PatchTarget


def _safe_name(value: str) -> str:
    """Sanitize a filename component to a safe, short identifier."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned[:80] or "kernel_agent"


def _path_hash(path: Path) -> str:
    """Return a short, stable hash for ``path`` to disambiguate backups."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


_MAX_BACKUP_DIR_ATTEMPTS: int = 200


def _claim_backup_dir(backup_root: Path, kernel_id: str, target: Path) -> Path:
    """Create a backup directory this apply alone owns.

    ``mkdir(exist_ok=False)`` makes the claim atomic against a concurrent caller,
    which a constant or absent ``kernel_id`` would otherwise let reuse a name.
    """
    base = Path(backup_root) / f"{_safe_name(kernel_id or target.stem)}_{_path_hash(target)}"
    for suffix in range(1, _MAX_BACKUP_DIR_ATTEMPTS + 1):
        candidate = base if suffix == 1 else base.with_name(f"{base.name}-{suffix}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"_claim_backup_dir: {base} still taken after {_MAX_BACKUP_DIR_ATTEMPTS} suffixes")


def _copy_to_backup(path: Path, backup_dir: Path, group: str) -> dict[str, str]:
    """Copy a file into the backup tree and return the manifest entry."""
    dst = backup_dir / group / f"{_path_hash(path)}_{path.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return {"path": str(path), "backup_path": str(dst)}


def _contained_dest(
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


def _within_root(path: Path, root: Path) -> bool:
    """Return True when ``path`` (symlinks resolved) stays inside ``root``."""
    try:
        p = path.resolve()
        r = root.resolve()
    except (OSError, RuntimeError):
        return False
    return p == r or p.is_relative_to(r)


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
    staged = _stage_snapshot_writes(descriptors, roots)
    if staged.failure is not None:
        return staged.failure

    backups: list[dict[str, Any]] = []
    touched: list[str] = []
    for write in staged.writes:
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
    """Where a snapshot's contents come from and which roots they may land in."""

    snapshot_dir: Path
    repo_root: Path
    deploy_roots: list[Path] | None
    symlink_roots: list[Path]


class _StagedWrite(NamedTuple):
    """One descriptor's resolved destination and, for a write, its snapshot source."""

    descriptor: PatchTarget
    dest: Path
    source: Path | None


class _StagedSnapshot(NamedTuple):
    """A pre-flighted snapshot: the writes to replay, or the failure that stopped it."""

    writes: list[_StagedWrite]
    failure: dict[str, Any] | None


def _stage_snapshot_writes(descriptors: list[PatchTarget], roots: _SnapshotRoots) -> _StagedSnapshot:
    """Validate every destination and snapshot source before the live tree is touched."""
    staged: list[_StagedWrite] = []
    try:
        for desc in descriptors:
            dest = _contained_dest(roots.repo_root, desc.path, deploy_roots=roots.deploy_roots)
            source: Path | None = None
            if desc.op is PatchOp.WRITE:
                source = roots.snapshot_dir / desc.path
                refusal = _unwritable_snapshot_source(source, dest, desc.path, roots.symlink_roots)
                if refusal is not None:
                    return _StagedSnapshot([], refusal)
            staged.append(_StagedWrite(desc, dest, source))
    except ValueError as exc:
        return _StagedSnapshot([], {"status": "failed", "error": str(exc), "path": "<pre-flight>"})
    return _StagedSnapshot(staged, None)


def _unwritable_snapshot_source(
    source: Path,
    dest: Path,
    rel_path: str,
    symlink_roots: list[Path],
) -> dict[str, Any] | None:
    """Why ``source`` may not be written to ``dest``, or ``None`` when it may."""
    if not source.is_symlink() and not source.exists():
        return {
            "status": "failed",
            "error": f"snapshot missing content for {rel_path}",
            "path": rel_path,
        }
    if not source.is_symlink():
        return None
    link_target = os.readlink(source)
    resolved_link = (Path(link_target) if os.path.isabs(link_target) else dest.parent / link_target).resolve()
    if any(_within_root(resolved_link, allowed_root) for allowed_root in symlink_roots):
        return None
    return {
        "status": "failed",
        "error": (f"snapshot symlink target is outside authorized deploy roots: {rel_path} -> {link_target}"),
        "path": rel_path,
    }


def _back_up_before_write(write: _StagedWrite, backup_dir: Path) -> dict[str, Any]:
    """Copy the live file aside and describe how a revert must undo this write."""
    dest = write.dest
    existed = dest.exists() or dest.is_symlink()
    if not existed and write.descriptor.op is not PatchOp.DELETE:
        return {"path": str(dest), "backup_path": None, "disposition": "added"}
    backup_path = None
    if existed:
        backup_dest = backup_dir / "source" / f"{_path_hash(dest)}_{dest.name}"
        backup_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, backup_dest, follow_symlinks=False)
        backup_path = str(backup_dest)
    disposition = "deleted" if write.descriptor.op is PatchOp.DELETE else "modified"
    return {"path": str(dest), "backup_path": backup_path, "disposition": disposition}


def _land_staged_write(write: _StagedWrite, roots: _SnapshotRoots) -> None:
    """Put one descriptor's asserted final state on the live tree."""
    dest = write.dest
    if write.descriptor.op is PatchOp.DELETE:
        dest.unlink(missing_ok=True)
        return
    # Re-validate containment right before the write (close TOCTOU window).
    _contained_dest(roots.repo_root, write.descriptor.path, deploy_roots=roots.deploy_roots)
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
    """Roll the live tree back to v0 from recorded backups."""
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


class _BackupDeletion(NamedTuple):
    """The backups finalize removed, and the ones it refused to touch."""

    deleted: list[str]
    issues: list[dict[str, Any]]


def _delete_recorded_backups(candidates: set[str], backup_root: Path) -> _BackupDeletion:
    """Delete each recorded backup that really lives under ``backup_root``."""
    deleted: list[str] = []
    issues: list[dict[str, Any]] = []
    for raw in sorted(path for path in candidates if path):
        path = Path(raw)
        if path.resolve() == backup_root:
            issues.append(
                {
                    "kind": "untrusted_backup",
                    "path": str(path),
                    "error": "backup path equals manifest directory",
                }
            )
            continue
        if not _within_root(path, backup_root):
            issues.append({"kind": "untrusted_backup", "path": str(path)})
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            deleted.append(str(path))
        except OSError as exc:
            issues.append(
                {
                    "kind": "delete_failed",
                    "path": str(path),
                    "error": str(exc),
                }
            )
    return _BackupDeletion(deleted, issues)


def _revert_backup_trusted(backup_path: str, backup_root: Path) -> bool:
    """True when a manifest ``backup_path`` is a real backup under ``backup_root``.

    The manifest JSON is untrusted at revert time, so every ``copy2`` source must
    stay inside the apply-time backup tree or a tampered entry could read an
    arbitrary host file.
    """
    if not backup_path:
        return False
    return _within_root(Path(backup_path), backup_root)
