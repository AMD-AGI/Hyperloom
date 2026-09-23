# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve and transact AITER runtime JIT caches without importing AITER.

Callers authorize package locations and backup roots. This stdlib-only module
owns cache scope, backup integrity, and filesystem operations, and can also be
shipped as a standalone file to inference workers.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def resolve_jit_build_dir(package_root: str | Path | None) -> Path | None:
    """Return the runtime's build path, or None when its cache is unavailable.

    Explicit overrides retain whitespace and literal tildes, matching AITER.
    The readonly-package fallback must already exist: AITER's first import
    copies the package cache there, which can reintroduce invalidated modules.
    """
    if package_root is None:
        return None
    if "AITER_JIT_DIR" in os.environ:
        override = os.environ["AITER_JIT_DIR"]
        return Path(override).absolute() / "build" if override else None
    jit = Path(package_root) / "jit"
    if not os.access(jit, os.W_OK):
        jit = Path.home() / ".aiter" / "jit"
        if not jit.is_dir():
            return None
    return jit / "build"


def trusted_jit_build_dir(path: str | Path, expected: str | Path) -> bool:
    """Check build-leaf identity against a caller-authorized destination.

    Parent symlinks are allowed; build-leaf aliases are not, because module
    cleanup operates on the lexical parent rather than the resolved leaf.
    """
    try:
        path, expected = Path(path), Path(expected)
        return (
            path.name == expected.name == "build"
            and not path.is_symlink()
            and not expected.is_symlink()
            and path.parent.resolve() == expected.parent.resolve()
            and path.resolve() == expected.resolve()
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _module_scope(modules: tuple[str, ...] | list[str] | None) -> list[str] | None:
    if modules is None:
        return None
    if not isinstance(modules, (tuple, list)) or any(
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name) for name in modules
    ):
        raise ValueError("module scope must be None or a sequence of module stems")
    return list(dict.fromkeys(modules))


def _serving_modules(jit: Path, scope: list[str] | None) -> list[Path]:
    if scope == [] or not jit.exists():
        return []
    return sorted(path for path in jit.iterdir() if path.suffix == ".so" and (scope is None or path.stem in scope))


def _require_regular_modules(modules: list[Path]) -> None:
    if any(not module.is_file() or module.is_symlink() or module.suffix != ".so" for module in modules):
        raise ValueError("AITER serving modules must be regular .so files")


def _move_cache(jit_build: Path, backup_dir: Path, modules: list[Path], record: dict[str, Any]) -> list[str]:
    stamp = time.time_ns()
    backup_path = backup_dir / f"jit_build_{stamp}"
    modules_path = backup_dir / f"jit_modules_{stamp}"
    moved: list[tuple[Path, Path]] = []
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        if any(path.exists() or path.is_symlink() for path in (backup_path, modules_path)):
            raise FileExistsError(f"JIT backup already exists: {backup_path}")
        if record["build_existed"]:
            shutil.move(str(jit_build), str(backup_path))
            moved.append((jit_build, backup_path))
            record["backup_path"] = str(backup_path)
        if modules:
            modules_path.mkdir()
            record["modules_backup_path"] = str(modules_path)
            for module in modules:
                destination = modules_path / module.name
                shutil.move(str(module), str(destination))
                moved.append((module, destination))
    except (OSError, shutil.Error) as exc:
        rollback_errors = []
        for original, saved in reversed(moved):
            try:
                shutil.move(str(saved), str(original))
            except (OSError, shutil.Error) as restore_error:
                rollback_errors.append(str(restore_error))
        record.update(status="failed", error=f"JIT cache invalidation failed: {exc}")
        return rollback_errors
    return []


def invalidate_jit_cache(
    jit_build: str | Path,
    backup_dir: str | Path,
    modules: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Back up build/ and selected top-level .so files before recompilation.

    None selects every module; a tuple selects exact stems; () only moves
    build/. The JSON-compatible record separates module_scope (the requested
    stems) from module_names (the actual backed-up filenames). Expected I/O or
    validation errors return status=failed, including any rollback errors.
    """
    jit_build, backup_dir = Path(jit_build), Path(backup_dir)
    record: dict[str, Any] = {"src": str(jit_build)}
    try:
        if not trusted_jit_build_dir(jit_build, jit_build):
            raise ValueError(f"JIT build leaf must be named build and must not be a symlink: {jit_build}")
        scope = _module_scope(modules)
        serving = _serving_modules(jit_build.parent, scope)
        _require_regular_modules(serving)
        build_existed = jit_build.exists()
        if build_existed and not jit_build.is_dir():
            raise ValueError(f"JIT build path must be a directory: {jit_build}")
        if backup_dir.resolve().is_relative_to(jit_build.resolve()):
            raise ValueError("JIT backup directory must not be inside the build directory")
        record.update(
            status="ok" if build_existed or serving else "clean",
            build_existed=build_existed,
            modules_invalidated=scope is None or bool(scope),
            module_scope=scope,
            module_names=[module.name for module in serving],
            moved_at=datetime.now(timezone.utc).isoformat(),
        )
        if build_existed or serving:
            rollback_errors = _move_cache(jit_build, backup_dir, serving, record)
            if record["status"] == "failed":
                record["rollback_errors"] = rollback_errors
    except (OSError, shutil.Error, ValueError, RuntimeError) as exc:
        record.update(status="failed", error=f"JIT cache invalidation failed: {exc}", rollback_errors=[])
    return record


def _record_scope(record: dict[str, Any]) -> list[str] | None:
    if "module_scope" in record:
        return _module_scope(record["module_scope"])
    invalidated = record.get("modules_invalidated", False)
    if invalidated is True:
        return None
    if invalidated is not False:
        raise ValueError("invalid legacy modules_invalidated flag")
    return []


def _backup_directory(raw: str, src: Path, backup_root: str | Path | None) -> Path | None:
    if not raw:
        return None
    saved = Path(raw)
    resolved = saved.resolve()
    if backup_root is not None:
        root = Path(backup_root).resolve()
        if resolved == root or not resolved.is_relative_to(root):
            raise ValueError(f"untrusted jit/build backup path: {saved}")
    if resolved.is_relative_to(src.resolve()) or src.resolve().is_relative_to(resolved):
        raise ValueError(f"JIT backup overlaps the restore destination: {saved}")
    if saved.is_symlink() or not saved.is_dir():
        raise ValueError(f"backup path missing or not a regular directory: {saved}")
    return saved


def _restore_inventory(
    record: dict[str, Any], src: Path, backup_root: str | Path | None, scope: list[str] | None
) -> tuple[Path | None, list[Path]]:
    backup_raw = str(record.get("backup_path") or "")
    build_existed = record.get("build_existed", bool(backup_raw) or record["status"] != "clean")
    if not isinstance(build_existed, bool) or build_existed != bool(backup_raw):
        raise ValueError("incomplete or inconsistent JIT build backup record")
    backup = _backup_directory(backup_raw, src, backup_root)
    saved_modules = _backup_directory(str(record.get("modules_backup_path") or ""), src, backup_root)
    modules = sorted(saved_modules.iterdir()) if saved_modules is not None else []
    _require_regular_modules(modules)
    names = record.get("module_names", [module.name for module in modules])
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("invalid serving-module inventory")
    if sorted(names) != sorted(module.name for module in modules):
        raise ValueError("incomplete serving-module backup")
    if scope is not None and any(module.stem not in scope for module in modules):
        raise ValueError("serving-module backup is outside the requested scope")
    return backup, modules


def restore_jit_cache(
    record: dict[str, Any], expected_jit_build_dir: str | Path, backup_root: str | Path | None
) -> dict[str, Any]:
    """Restore a baseline after validating the entire record and backup inventory.

    Selected restores delete only requested stems, including modules absent at
    invalidation time. Legacy modules_invalidated=True means full scope; records
    without scope or that flag are build-only. A legacy clean record needs no
    backup. Module copies precede the build move so copy failure leaves the build
    backup available for retry. Expected errors return status=failed.
    """
    if not isinstance(record, dict) or record.get("status") not in {"ok", "clean"}:
        return {"status": "skipped", "reason": "no backup recorded"}
    src_raw = str(record.get("src") or "")
    if not src_raw:
        return {"status": "skipped", "reason": "incomplete backup record"}
    src = Path(src_raw)
    try:
        if not trusted_jit_build_dir(src, expected_jit_build_dir):
            raise ValueError(f"jit/build src {src} does not match expected dir {expected_jit_build_dir}")
        scope = _record_scope(record)
        backup, modules = _restore_inventory(record, src, backup_root, scope)
        candidates = _serving_modules(src.parent, scope)
        if any(module.is_dir() and not module.is_symlink() for module in candidates):
            raise ValueError("candidate serving module must not be a directory")
        for module in candidates:
            module.unlink()
        if src.exists():
            shutil.rmtree(src)
        src.parent.mkdir(parents=True, exist_ok=True)
        for module in modules:
            shutil.copy2(module, src.parent / module.name)
        if backup is not None:
            shutil.move(str(backup), str(src))
    except (OSError, shutil.Error, ValueError, RuntimeError) as exc:
        return {"status": "failed", "error": f"JIT cache restore failed: {exc}", "src": str(src)}
    return {"status": "ok", "restored_to": str(src)}
