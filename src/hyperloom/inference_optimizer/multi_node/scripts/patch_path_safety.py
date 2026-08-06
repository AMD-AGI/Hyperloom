"""Path constraints for kernel patch apply/revert on inference pods (stdlib only).

Shared by ``kernel_node_ops.py`` (Infera SSH) and ``kernel_patch_multinode.py``
(RayJob). Keeps patch targets under framework install roots and backups under
``$HYPERLOOM_MN_KERNEL_BACKUP_DIR`` (default ``/var/kernel_patch_backups``), and
hosts the atomic write both apply paths use to land a patched file.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_DEFAULT_KERNEL_BACKUP_ROOT = "/var/kernel_patch_backups"

# Superset of orchestrator.framework.paths._STATIC_PATCH_FALLBACK_ROOTS
# (adds the /sgl-workspace/* image roots).
_DEFAULT_PATCH_TARGET_ROOTS: tuple[str, ...] = (
    "/sgl-workspace/aiter/",
    "/sgl-workspace/sglang/",
    "/sgl-workspace/vllm/",
    "/app/ATOM/atom/",
    "/app/xDiT/",
    "/opt/venv/lib/python3.10/site-packages/aiter/",
    "/opt/venv/lib/python3.10/site-packages/sglang/",
    "/opt/venv/lib/python3.10/site-packages/vllm/",
    "/opt/venv/lib/python3.10/site-packages/atom/",
    "/opt/venv/lib/python3.12/site-packages/aiter/",
    "/opt/venv/lib/python3.12/site-packages/sglang/",
    "/opt/venv/lib/python3.12/site-packages/vllm/",
    "/opt/venv/lib/python3.12/site-packages/atom/",
    "/usr/local/lib/python3.12/dist-packages/aiter/",
    "/usr/local/lib/python3.12/dist-packages/sglang/",
    "/usr/local/lib/python3.12/dist-packages/vllm/",
    "/usr/local/lib/python3.12/dist-packages/atom/",
    "/usr/local/lib/python3.10/dist-packages/aiter/",
    "/usr/local/lib/python3.10/dist-packages/sglang/",
    "/usr/local/lib/python3.10/dist-packages/vllm/",
    "/usr/local/lib/python3.10/dist-packages/atom/",
    "/aiter_meta/csrc/",
)


def _normalize_root(path: str) -> str:
    """Normalize a root path to a trailing-slash form.

    Args:
        path: Raw path string.

    Returns:
        str: Stripped path with a trailing slash, or empty when blank.
    """
    p = str(path or "").strip()
    if not p:
        return ""
    return p if p.endswith("/") else f"{p}/"


def _merge_roots(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Merge root groups, dropping blanks and duplicates.

    Args:
        *groups: One or more ordered root groups.

    Returns:
        tuple[str, ...]: De-duplicated roots in first-seen order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for root in group:
            if root and root not in seen:
                seen.add(root)
                out.append(root)
    return tuple(out)


def resolve_patch_target_roots() -> tuple[str, ...]:
    """Return allowed framework roots for patch targets.

    Merges static defaults with ``$INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS``.

    Returns:
        tuple[str, ...]: Normalized framework root prefixes.
    """
    env = os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", "").strip()
    env_roots = tuple(_normalize_root(p) for p in env.split(":") if p.strip()) if env else ()
    return _merge_roots(_DEFAULT_PATCH_TARGET_ROOTS, env_roots)


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


def assert_target_path_allowed(target: Path, *, must_exist: bool = False) -> None:
    """Raise ValueError when ``target`` is outside framework patch roots.

    Args:
        target: Pod-side file path to patch or restore.
        must_exist: When true, require that ``target`` is an existing file.

    Raises:
        ValueError: When the path is disallowed or missing (if required).
    """
    resolved = target.resolve()
    if must_exist and not resolved.is_file():
        raise ValueError(f"target_path does not exist: {target}")
    roots = [Path(r.rstrip("/")).resolve() for r in resolve_patch_target_roots() if r]
    for root in roots:
        if _path_under_root(resolved, root):
            return
    raise ValueError(f"target_path {target} not under framework patch roots")


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


def assert_revert_paths_allowed(target: Path, backup: Path) -> None:
    """Validate revert target and backup paths before restoring from backup.

    Args:
        target: Pod-side file path to restore.
        backup: Recorded backup file from the matching apply.

    Raises:
        ValueError: When either path is outside its allowed root.
    """
    assert_target_path_allowed(target, must_exist=False)
    assert_backup_path_allowed(backup)


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
    if src.exists():
        shutil.rmtree(src)
    if record.get("status") == "clean":
        return {"status": "restored_clean", "restored_to": str(src)}
    backup = Path(str(record.get("backup_path") or ""))
    assert_backup_path_allowed(backup)
    if not backup.exists():
        raise FileNotFoundError(f"JIT backup does not exist: {backup}")
    src.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(backup), str(src))
    return {"status": "restored", "restored_to": str(src)}


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
