"""Path constraints for kernel patch apply/revert on inference pods (stdlib only).

Shared by ``kernel_node_ops.py`` (Infera SSH) and ``kernel_patch_multinode.py``
(RayJob). Keeps backups under ``$HYPERLOOM_MN_KERNEL_BACKUP_DIR`` (default
``/var/kernel_patch_backups``), and hosts the atomic write both apply paths use.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_DEFAULT_KERNEL_BACKUP_ROOT = "/var/kernel_patch_backups"


def resolve_kernel_backup_root() -> Path:
    """Resolve the allowed kernel backup directory on the pod.

    Returns:
        Path: Absolute backup root from ``$HYPERLOOM_MN_KERNEL_BACKUP_DIR``.
    """
    raw = (os.environ.get("HYPERLOOM_MN_KERNEL_BACKUP_DIR") or _DEFAULT_KERNEL_BACKUP_ROOT).strip()
    return Path(raw).resolve()


def _path_under_root(path: Path, root: Path) -> bool:
    """Return whether ``path`` is ``root`` or nested under ``root``.

    Args:
        path: Path to test (need not exist).
        root: Allowed root directory.

    Returns:
        bool: True when ``path`` resolves under ``root``.
    """
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return path.resolve() == root.resolve()


def assert_backup_dir_allowed(backup_dir: Path) -> None:
    """Raise ValueError when ``backup_dir`` is outside the kernel backup root.

    Args:
        backup_dir: Directory where pre-patch backups are written.

    Raises:
        ValueError: When the directory is outside the allowed backup root.
    """
    root = resolve_kernel_backup_root()
    if not _path_under_root(backup_dir.resolve(), root):
        raise ValueError(f"backup_dir {backup_dir} not under {root}")


def assert_backup_path_allowed(backup: Path) -> None:
    """Raise ValueError when ``backup`` is outside the kernel backup root.

    Args:
        backup: Backup file path recorded by a prior apply.

    Raises:
        ValueError: When the backup path is outside the allowed backup root.
    """
    root = resolve_kernel_backup_root()
    if not _path_under_root(backup.resolve(), root):
        raise ValueError(f"backup_path {backup} not under {root}")


def assert_aiter_jit_build_allowed(jit_build: Path) -> None:
    """Validate an AITER ``jit/build`` path before recursive mutation."""
    resolved = jit_build.resolve()
    if (
        resolved.name != "build"
        or resolved.parent.name != "jit"
        or resolved.parent.parent.name != "aiter"
        or not (resolved.parent / "__init__.py").is_file()
        or not (resolved.parent.parent / "__init__.py").is_file()
    ):
        raise ValueError(f"invalid AITER jit/build path: {jit_build}")


def invalidate_aiter_jit_build(
    jit_build: Path | None,
    backup_dir: Path,
    backup_name: str,
) -> dict:
    """Move one pod's stale AITER JIT cache aside before patched serving."""
    if jit_build is None:
        return {"status": "skipped", "reason": "no jit_build_dir supplied"}
    assert_aiter_jit_build_allowed(jit_build)
    assert_backup_dir_allowed(backup_dir)
    resolved = jit_build.resolve()
    if not resolved.exists() or not any(resolved.iterdir()):
        return {"status": "clean", "src": str(resolved)}
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{backup_name}_jit_build"
    assert_backup_path_allowed(backup)
    if backup.exists():
        raise ValueError(f"JIT backup already exists: {backup}")
    shutil.move(str(resolved), str(backup))
    return {
        "status": "ok",
        "src": str(resolved),
        "backup_path": str(backup),
    }


def restore_aiter_jit_build(record: dict) -> dict:
    """Remove candidate JIT output and restore a pod's baseline cache."""
    if not isinstance(record, dict) or record.get("status") not in {"ok", "clean"}:
        return {"status": "skipped", "reason": "no JIT invalidation record"}
    src = Path(str(record.get("src") or ""))
    assert_aiter_jit_build_allowed(src)
    if record.get("status") == "clean":
        if src.exists():
            shutil.rmtree(src)
        return {"status": "restored_clean", "restored_to": str(src)}
    backup = Path(str(record.get("backup_path") or ""))
    assert_backup_path_allowed(backup)
    if not backup.exists():
        raise FileNotFoundError(f"JIT backup does not exist: {backup}")
    if src.exists():
        shutil.rmtree(src)
    src.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(backup), str(src))
    return {"status": "restored", "restored_to": str(src)}


def finalize_patch_records(records: list[dict]) -> dict:
    """Delete source/JIT backups after a patch becomes the accepted baseline."""
    deleted: list[str] = []
    jit_backups: set[str] = set()
    for record in records:
        backup_raw = str(record.get("backup_path") or "").strip()
        if backup_raw:
            backup = Path(backup_raw)
            assert_backup_path_allowed(backup)
            if backup.is_file():
                backup.unlink()
                deleted.append(str(backup))
        jit_record = record.get("jit_backup")
        if isinstance(jit_record, dict):
            jit_backup = str(jit_record.get("backup_path") or "").strip()
            if jit_backup:
                jit_backups.add(jit_backup)
    for backup_raw in sorted(jit_backups):
        backup = Path(backup_raw)
        assert_backup_path_allowed(backup)
        if backup.is_dir():
            shutil.rmtree(backup)
            deleted.append(str(backup))
    return {"status": "finalized", "deleted": deleted}


def atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` atomically (tmp file in-dir + ``os.replace``).

    Args:
        target (Path): Destination file path (parent dirs are created).
        data (bytes): Bytes to write.

    Raises:
        OSError: If writing the temp file or replacing the target fails; the
            temp file is removed first.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                # Temp file already gone; the original error is re-raised below.
                pass
        raise
