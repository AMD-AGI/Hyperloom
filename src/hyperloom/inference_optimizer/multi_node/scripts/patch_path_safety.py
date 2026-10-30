"""Path constraints for kernel patch apply/revert on inference pods (stdlib only).

Shared by ``kernel_node_ops.py`` (Infera SSH) and ``kernel_patch_multinode.py``
(RayJob). Keeps backups under ``$HYPERLOOM_MN_KERNEL_BACKUP_DIR`` (default
``/var/kernel_patch_backups``), and hosts the single-pod apply/revert
transaction both drivers run.
"""

from __future__ import annotations

import base64
import hashlib
import os
import py_compile
import shutil
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import aiter_jit_cache as _jit_cache
except ModuleNotFoundError as exc:
    if exc.name != "aiter_jit_cache":
        raise
    from hyperloom.common import aiter_jit_cache as _jit_cache

DEFAULT_KERNEL_BACKUP_ROOT = "/var/kernel_patch_backups"


def resolve_kernel_backup_root() -> Path:
    """Resolve the allowed kernel backup directory on the pod.

    Returns:
        Path: Absolute backup root from ``$HYPERLOOM_MN_KERNEL_BACKUP_DIR``.
    """
    raw = (os.environ.get("HYPERLOOM_MN_KERNEL_BACKUP_DIR") or DEFAULT_KERNEL_BACKUP_ROOT).strip()
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
    """Validate the pod's runtime AITER destination before recursive mutation."""
    package = jit_build.parent.parent
    if "AITER_JIT_DIR" not in os.environ and not (
        package.name == "aiter"
        and jit_build.parent.name == "jit"
        and (package / "__init__.py").is_file()
        and (jit_build.parent / "__init__.py").is_file()
    ):
        package = _jit_cache.resolve_package_root()
    expected = _jit_cache.resolve_jit_build_dir(package)
    if expected is None or not _jit_cache.trusted_jit_build_dir(jit_build, expected):
        raise ValueError(f"invalid AITER jit/build path: {jit_build}")


def invalidate_aiter_jit_build(
    jit_build: Path | None,
    backup_dir: Path,
    backup_name: str,
) -> dict:
    """Invalidate build and all serving modules through the shared transaction."""
    if jit_build is None:
        return {"status": "skipped", "reason": "no jit_build_dir supplied"}
    assert_aiter_jit_build_allowed(jit_build)
    assert_backup_dir_allowed(backup_dir)
    transaction_dir = backup_dir / backup_name
    assert_backup_dir_allowed(transaction_dir)
    result = _jit_cache.invalidate_jit_cache(jit_build.resolve(), transaction_dir)
    if result.get("status") == "failed":
        raise OSError(result["error"])
    return result


def restore_aiter_jit_build(record: dict) -> dict:
    """Adapt shared restoration to the pod's exception and status contract."""
    if not isinstance(record, dict) or record.get("status") not in {"ok", "clean"}:
        return {"status": "skipped", "reason": "no JIT invalidation record"}
    src = Path(str(record.get("src") or ""))
    assert_aiter_jit_build_allowed(src)
    for key in ("backup_path", "modules_backup_path"):
        if record.get(key):
            backup = Path(record[key])
            assert_backup_path_allowed(backup)
            if not backup.exists():
                raise FileNotFoundError(f"JIT backup does not exist: {backup}")
    result = _jit_cache.restore_jit_cache(record, src, resolve_kernel_backup_root())
    if result.get("status") == "failed":
        raise OSError(result["error"])
    if result.get("status") == "ok":
        result["status"] = "restored_clean" if record["status"] == "clean" else "restored"
    return result


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
            for key in ("backup_path", "modules_backup_path"):
                jit_backup = str(jit_record.get(key) or "").strip()
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


def _safe_name(value: str) -> str:
    """Sanitize a string into a filesystem-safe filename token."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned[:80] or "patch"


def apply_patch_on_pod(target: Path, patch_b64: str, backup_dir: Path, kernel_id: str, jit_build_dir: str) -> dict:
    """Back up ``target``, invalidate the AITER JIT cache, write the patch, and compile-check ``.py`` targets.

    Raises on any failure; a failed write or compile restores the source and the
    JIT cache before re-raising. Returns the apply record the controller persists.
    """
    host = socket.gethostname()
    assert_backup_dir_allowed(backup_dir)
    if not target.is_file():
        raise FileNotFoundError(f"target_path does not exist on pod {host}: {target}")
    try:
        data = base64.b64decode(patch_b64.encode("ascii"))
    except ValueError as exc:
        raise ValueError(f"patch_b64 not valid base64: {exc!r}") from exc
    backup_dir.mkdir(parents=True, exist_ok=True)
    path_hash = hashlib.sha256(str(target).encode("utf-8")).hexdigest()[:16]
    backup_stem = f"{_safe_name(kernel_id or target.stem)}_{path_hash}_{host}_{time.time_ns()}"
    backup_path = backup_dir / f"{backup_stem}.bak"
    shutil.copy2(target, backup_path)
    jit_backup = invalidate_aiter_jit_build(Path(jit_build_dir) if jit_build_dir else None, backup_dir, backup_stem)
    compile_result: dict[str, Any] = {"status": "skipped", "reason": "non-py target"}
    try:
        atomic_write_bytes(target, data)
        if target.suffix.lower() == ".py":
            py_compile.compile(str(target), doraise=True)
            compile_result = {"status": "ok"}
    except Exception:
        shutil.copy2(backup_path, target)
        restore_aiter_jit_build(jit_backup)
        raise
    return {
        "host": host,
        "target_path": str(target),
        "backup_path": str(backup_path),
        "wrote_bytes": len(data),
        "compile": compile_result,
        "jit_backup": jit_backup,
    }


def revert_patch_records(records: list[dict]) -> dict:
    """Restore every source and JIT backup recorded for one pod, newest first; raises on the first failure."""
    if not records:
        raise ValueError("empty revert records")
    restored: list[str] = []
    jit_records: list[dict] = []
    for record in reversed(records):
        target = Path(str(record.get("target_path") or ""))
        backup = Path(str(record.get("backup_path") or ""))
        if not backup.is_file():
            raise FileNotFoundError(f"backup missing: {backup}")
        assert_backup_path_allowed(backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, target)
        restored.append(str(target))
        jit_record = record.get("jit_backup")
        if isinstance(jit_record, dict) and jit_record.get("status") in {"ok", "clean"}:
            jit_records.append(jit_record)
    jit_restore = {"status": "skipped", "reason": "no JIT backup"}
    if jit_records:
        first = jit_records[0]
        if any(record != first for record in jit_records[1:]):
            raise ValueError("conflicting JIT backup records for one pod")
        jit_restore = restore_aiter_jit_build(first)
    return {"status": "restored", "restored_targets": restored, "jit_restore": jit_restore}
