# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AITER JIT build and cpp_itfs runtime-cache invalidation, restore and verification for kernel patches."""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from common import aiter_jit_cache
from hyperloom.common.timeutil import now_iso


_ALLOWED_KERNEL_DEPLOY_PACKAGES = frozenset({"aiter", "aiter_meta", "sglang", "vllm"})
_EDITABLE_AITER_ROOT = Path("/sgl-workspace/aiter")


def _within_root(path: Path, root: Path) -> bool:
    """Return True when ``path`` (symlinks resolved) stays inside ``root``."""
    try:
        p = path.resolve()
        r = root.resolve()
    except (OSError, RuntimeError):
        return False
    return p == r or p.is_relative_to(r)


# aiter JIT cache invalidation around rebuilds (setup.py develop won't invalidate jit/build/ .so).
# aiter device sources ship either in-tree (``/aiter/csrc/``) or in the sibling
# split wheel (``/aiter_meta/csrc/``); both feed the same importable ``aiter``
# JIT build, so the rebuild gates must recognise both layouts.
_AITER_CSRC_MARKERS = ("/aiter/csrc/", "/aiter_meta/csrc/")

# SGLang's in-tree JIT kernels are compiled by tvm-ffi on first import into a
# cache directory keyed on a hash of the source closure, so a patched source
# builds under a new key instead of colliding with the baseline module.
_SGLANG_JIT_SOURCE_MARKER = "/sglang/kernels/jit/"


def _installed_kernel_package(
    target_file: Path,
) -> tuple[Path, str] | None:
    """Return ``(site-packages root, package name)`` for an allowed framework."""
    # Preserve the lexical install path: venv/site-packages is commonly a
    # symlink, and resolving it here would make strategy classification disagree
    # with the path matching performed by _detect_strategy.
    target = target_file.absolute()
    for parent in target.parents:
        if parent.name in _ALLOWED_KERNEL_DEPLOY_PACKAGES and parent.parent.name in {"site-packages", "dist-packages"}:
            return parent.parent, parent.name
    return None


def _installed_aiter_runtime_root(target_file: Path) -> Path | None:
    """Return the package root shared by installed ``aiter`` and ``aiter_meta``.

    AITER wheels split runtime Python/JIT code into ``aiter`` and device sources
    into the sibling ``aiter_meta`` package. Both are runtime-JIT inputs and must
    share one deployment/rebuild strategy.
    """
    installed = _installed_kernel_package(target_file)
    if installed is None or installed[1] not in {"aiter", "aiter_meta"}:
        return None
    return installed[0]


def _aiter_checkout_root(target_file: Path) -> Path | None:
    """Identify a source checkout through the installed runtime, not its basename."""
    package = aiter_jit_cache.resolve_package_root()
    if package is None or package.parent.name in {"site-packages", "dist-packages"}:
        return None
    if not (package / "__init__.py").is_file() or not (package / "jit" / "__init__.py").is_file():
        return None
    meta = os.environ.get("AITER_META_DIR", "").strip()
    source_root = Path(meta).absolute() if meta else package.parent
    if not (source_root / "csrc").is_dir():
        source_root = package.parent
    if (source_root / "csrc").is_dir() and _within_root(target_file, source_root):
        return source_root
    if _within_root(target_file, package):
        return package.parent
    return None


def _target_is_in_aiter_csrc(target_file: Path) -> bool:
    """Report whether a file belongs to AITER device sources or codegen.

    Args:
        target_file: The file path to test.

    Returns:
        ``True`` under ``aiter/csrc/``, ``aiter_meta/csrc/``, or the ``csrc/``
        tree of the checkout identified by the installed runtime package.
    """
    norm = str(target_file).replace(os.sep, "/")
    checkout = _aiter_checkout_root(target_file)
    return any(marker in norm for marker in _AITER_CSRC_MARKERS) or (
        checkout is not None and _within_root(target_file, checkout / "csrc")
    )


def _target_is_sglang_jit_source(target_file: Path) -> bool:
    """Report whether a file lives under SGLang's in-tree JIT source tree.

    Matches both the editable checkout
    (``/sgl-workspace/sglang/python/sglang/kernels/jit/...``) and an installed
    wheel (``.../site-packages/sglang/kernels/jit/...``).

    Args:
        target_file: The file path to test.

    Returns:
        ``True`` if the path is under a ``sglang/kernels/jit/`` directory.
    """
    norm = str(target_file).replace(os.sep, "/").lower()
    return _SGLANG_JIT_SOURCE_MARKER in norm


def _target_uses_aiter_jit(target_file: Path) -> bool:
    """Return whether a source belongs to an installed/editable AITER runtime."""
    normalized = str(target_file).replace(os.sep, "/")
    return (
        _installed_aiter_runtime_root(target_file) is not None
        or "/sgl-workspace/aiter/" in normalized
        or _aiter_checkout_root(target_file) is not None
    )


def _aiter_jit_build_dir() -> Path | None:
    """Resolve AITER's build directory through the shared runtime resolver.

    Returns:
        The build path under ``AITER_JIT_DIR``, the package JIT directory, or
        the initialized user cache; ``None`` when the package/cache is unavailable.
    """
    return aiter_jit_cache.resolve_jit_build_dir(aiter_jit_cache.resolve_package_root())


def _invalidate_aiter_jit_build(
    target_file: Path,
    backup_dir: Path,
    *,
    jit_build_dir: Path | None = None,
) -> dict[str, Any]:
    """Back up AITER's build tree and all serving modules for a source patch.

    Args:
        target_file: The file being patched (gates the operation).
        backup_dir: Parent of the build-tree and serving-module backups.
        jit_build_dir: Authoritative JIT build path from the apply strategy.
            Falls back to runtime discovery for non-runtime-JIT strategies.

    Returns:
        The shared transaction record: ``ok`` when cache state was moved,
        ``clean`` when neither build nor modules existed, ``skipped``, or ``failed``.
    """
    if jit_build_dir is None and not _target_is_in_aiter_csrc(target_file):
        return {
            "status": "skipped",
            "reason": ("target is outside aiter csrc and no runtime-JIT build dir was supplied"),
        }
    jit_build = jit_build_dir or _aiter_jit_build_dir()
    if jit_build is None:
        return {
            "status": "skipped",
            "reason": "AITER runtime JIT directory unavailable; correct AITER_JIT_DIR and initialize the runtime cache before applying patches",
        }
    return aiter_jit_cache.invalidate_jit_cache(jit_build, backup_dir)


def _trusted_aiter_jit_build_dir(path: Path) -> bool:
    """Validate a strategy-persisted AITER JIT destination.

    Accepts both shapes apply can pin: a wheel install under site-packages and
    the editable checkout :data:`_EDITABLE_AITER_ROOT`.
    """
    root = _installed_aiter_runtime_root(path) or _EDITABLE_AITER_ROOT
    package = root / "aiter"
    if aiter_jit_cache.trusted_jit_build_dir(path, package / "jit" / "build"):
        return (package / "__init__.py").is_file() and (package / "jit" / "__init__.py").is_file()
    package = aiter_jit_cache.resolve_package_root()
    expected = aiter_jit_cache.resolve_jit_build_dir(package)
    return (
        package is not None
        and expected is not None
        and aiter_jit_cache.trusted_jit_build_dir(path, expected)
        and (package / "__init__.py").is_file()
        and (package / "jit" / "__init__.py").is_file()
    )


def _restore_aiter_jit_build(
    jit_build_backup: dict[str, Any],
    *,
    expected_jit_build_dir: str | Path | None = None,
    backup_root: str | Path,
) -> dict[str, Any]:
    """Restore AITER build and serving-module state within the recorded scope.

    The shared transaction validates backups before removing candidate artifacts.
    Legacy build-only records leave serving modules untouched.

    Args:
        jit_build_backup: The backup record returned by
            :func:`_invalidate_aiter_jit_build`.
        expected_jit_build_dir: Apply-time strategy path. Runtime-JIT restores
            use this instead of rediscovering aiter from the current process.
        backup_root: Apply manifest's backup directory. The recorded backup
            must remain contained under this root.

    Returns:
        A status dict with ``status`` of ``ok``, ``skipped``, or ``failed``.
    """
    if not isinstance(jit_build_backup, dict) or jit_build_backup.get("status") not in {"ok", "clean"}:
        return {"status": "skipped", "reason": "no backup recorded"}
    expected = Path(expected_jit_build_dir) if expected_jit_build_dir else _aiter_jit_build_dir()
    if expected is None or not _trusted_aiter_jit_build_dir(expected):
        return {"status": "failed", "error": f"untrusted strategy jit/build dir: {expected}"}
    return aiter_jit_cache.restore_jit_cache(jit_build_backup, expected, backup_root)


# aiter cpp_itfs kernels are runtime-compiled into parameter-keyed caches.
_AITER_CPP_ITFS_MARKERS = ("/aiter/csrc/cpp_itfs/", "/aiter_meta/csrc/cpp_itfs/")
_MD_NAME_RE = re.compile(r"""(?m)^\s*MD_NAME\s*=\s*["']([^"']+)["']""")


def _target_is_in_aiter_cpp_itfs(target_file: Path) -> bool:
    """Report whether a file lives under an ``aiter/csrc/cpp_itfs/`` tree.

    Strict subset of :func:`_target_is_in_aiter_csrc`: these are the
    runtime-compiled kernels whose served ``.so`` lives in
    ``$HOME/.aiter/build`` rather than in ``<aiter>/jit/build`` or the
    statically-linked wheel. Matches the editable checkout
    (``/sgl-workspace/aiter/csrc/cpp_itfs/...``), the dist-packages layout
    (``.../aiter/csrc/cpp_itfs/...``), and the split-wheel layout
    (``.../aiter_meta/csrc/cpp_itfs/...``).

    Args:
        target_file: The file path to test.

    Returns:
        ``True`` if the path is under an ``aiter/csrc/cpp_itfs/`` directory.
    """
    norm = str(target_file).replace(os.sep, "/")
    checkout = _aiter_checkout_root(target_file)
    return any(marker in norm for marker in _AITER_CPP_ITFS_MARKERS) or (
        checkout is not None and _within_root(target_file, checkout / "csrc" / "cpp_itfs")
    )


def _aiter_cpp_itfs_build_dir() -> Path:
    """Resolve aiter's cpp_itfs runtime ``BUILD_DIR``.

    Mirrors aiter ``csrc/cpp_itfs/utils.py``: ``$AITER_ROOT_DIR/build`` with
    ``$AITER_ROOT_DIR`` defaulting to ``$HOME/.aiter``. Honouring both env
    vars keeps non-default deployments + unit tests correct without importing
    aiter into this standalone tool.

    Returns:
        The resolved cpp_itfs ``build`` directory path.
    """
    root = os.environ.get("AITER_ROOT_DIR", "").strip()
    if not root:
        home = Path(os.environ.get("HOME", "~")).expanduser()
        root = str(home / ".aiter")
    return Path(root) / "build"


def _cpp_itfs_module_names(target_file: Path) -> list[str]:
    """Collect ``MD_NAME`` prefixes for the cpp_itfs module(s) a source feeds.

    The cpp_itfs ``.py`` driver next to the patched source declares
    ``MD_NAME = "pa_ragged"`` (etc.), which becomes the
    ``<md_name>_<hash>`` runtime-cache folder prefix. A single shared
    ``.cuh`` (e.g. ``pa_kernels.cuh``) is pulled into several drivers in the
    same directory, so we collect EVERY ``MD_NAME`` declared in the target's
    directory.

    Args:
        target_file: The patched source file whose directory is scanned.

    Returns:
        The discovered ``MD_NAME`` prefixes. An empty list tells the caller to
        fall back to clearing the whole cpp_itfs build root.
    """
    names: set[str] = set()
    search_dir = target_file.parent
    try:
        py_files = sorted(search_dir.glob("*.py"))
    except OSError:
        py_files = []
    if target_file.suffix.lower() == ".py":
        py_files = list(dict.fromkeys([target_file, *py_files]))
    for py in py_files:
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _MD_NAME_RE.finditer(text):
            names.add(match.group(1))
    return sorted(names)


def _invalidate_aiter_cpp_itfs_cache(
    target_file: Path,
    backup_dir: Path,
    *,
    build_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Move aiter cpp_itfs runtime-cache dirs aside so the re-baseline server
    runtime-recompiles the patched kernel from clean state.

    No-op (``skipped``, ``is_cpp_itfs=False``) for non-cpp_itfs targets. For
    cpp_itfs targets the affected ``<build_dir>/<md_name>_*`` dirs (scoped by
    ``MD_NAME`` when determinable, else every child of the build root) are
    MOVED into ``backup_dir/cpp_itfs_cache/`` so the operation is reversible.
    The record always carries ``is_cpp_itfs`` + ``build_dir`` +
    ``module_names`` + ``invalidated_unix`` (even when nothing was cached
    yet) so integrate can later verify a fresh rebuild actually landed.

    Args:
        target_file: The file being patched (gates the operation).
        backup_dir: Directory to move affected cache dirs into.
        build_dir_override: Test-only override for the cpp_itfs build root.

    Returns:
        A status dict (``ok`` / ``skipped`` / ``failed``) recording what was
        moved and the invalidation metadata.
    """
    if not _target_is_in_aiter_cpp_itfs(target_file):
        return {
            "status": "skipped",
            "is_cpp_itfs": False,
            "reason": "target not under aiter/csrc/cpp_itfs/",
        }
    build_dir = build_dir_override or _aiter_cpp_itfs_build_dir()
    module_names = _cpp_itfs_module_names(target_file)
    scope = "module" if module_names else "build_root"
    record: dict[str, Any] = {
        "is_cpp_itfs": True,
        "build_dir": str(build_dir),
        "module_names": module_names,
        "scope": scope,
        "invalidated_at": now_iso(timespec="auto"),
        "invalidated_unix": time.time(),
    }
    if not build_dir.exists():
        # Nothing cached yet -> the re-baseline server builds fresh.
        record.update(
            {
                "status": "skipped",
                "reason": "cpp_itfs build dir does not exist",
                "moved": [],
            }
        )
        return record
    try:
        if module_names:
            to_move: list[Path] = []
            for md in module_names:
                to_move.extend(p for p in build_dir.glob(f"{md}_*") if p.is_dir())
        else:
            to_move = [p for p in build_dir.iterdir() if p.is_dir()]
    except OSError as exc:
        record.update({"status": "failed", "error": f"failed to scan {build_dir}: {exc}"})
        return record
    # De-dup: a shared .cuh can match overlapping MD_NAME globs.
    to_move = sorted({p.resolve() for p in to_move})
    if not to_move:
        record.update(
            {
                "status": "skipped",
                "reason": "no matching cpp_itfs cache entries",
                "moved": [],
            }
        )
        return record
    cache_backup_root = backup_dir / "cpp_itfs_cache"
    moved: list[dict[str, str]] = []
    try:
        cache_backup_root.mkdir(parents=True, exist_ok=True)
        for src in to_move:
            dst = cache_backup_root / src.name
            if dst.exists():
                record.update(
                    {
                        "status": "failed",
                        "error": f"cpp_itfs cache backup path already exists: {dst}",
                        "moved": moved,
                    }
                )
                return record
            shutil.move(str(src), str(dst))
            moved.append({"src": str(src), "backup_path": str(dst)})
    except (OSError, shutil.Error) as exc:
        record.update(
            {
                "status": "failed",
                "error": f"shutil.move failed: {exc}",
                "moved": moved,
            }
        )
        return record
    record.update({"status": "ok", "moved": moved})
    return record


def verify_cpp_itfs_rebuilt(cache_backup: dict[str, Any]) -> dict[str, Any]:
    """Assert the re-baseline actually runtime-recompiled the patched kernel.

    After :func:`_invalidate_aiter_cpp_itfs_cache` moved the stale cache
    aside, a faithful re-baseline must repopulate
    ``<build_dir>/<md_name>_*/lib.so`` (the served binary) with an mtime at
    or after the invalidation. If no such fresh ``lib.so`` exists, the server
    dlopened a stale binary (or the kernel was never exercised) and the
    measured gain is meaningless -- the integrate KEEP/REVERT gate uses this
    to flag/abort instead of scoring a stale binary.

    Args:
        cache_backup: The invalidation record from
            :func:`_invalidate_aiter_cpp_itfs_cache`.

    Returns:
        A dict ``{"verified": bool, ...}``. ``verified`` is ``True`` for
        non-cpp_itfs targets so the caller's gate is a strict no-op off the
        cpp_itfs path.
    """
    if not isinstance(cache_backup, dict) or not cache_backup.get("is_cpp_itfs"):
        return {"verified": True, "status": "skipped", "reason": "non-cpp_itfs target"}
    build_dir_raw = cache_backup.get("build_dir", "")
    since = float(cache_backup.get("invalidated_unix") or 0.0)
    module_names = list(cache_backup.get("module_names") or [])
    # Path("") is Path("."), so an empty/missing build_dir must be rejected
    # before globbing — otherwise verify walks the process CWD.
    if not str(build_dir_raw).strip():
        return {
            "verified": False,
            "status": "stale",
            "reason": f"cpp_itfs build dir absent after re-baseline: {build_dir_raw}",
        }
    build_dir = Path(build_dir_raw)
    if not build_dir.exists():
        return {
            "verified": False,
            "status": "stale",
            "reason": f"cpp_itfs build dir absent after re-baseline: {build_dir}",
        }
    globs = [f"{md}_*/lib.so" for md in module_names] or ["*/lib.so"]
    fresh: list[str] = []
    for pattern in globs:
        for so in build_dir.glob(pattern):
            try:
                mtime = so.stat().st_mtime
            except OSError:
                continue
            # 1s slack absorbs build-dir-create vs lib.so-flush ordering.
            if mtime + 1.0 >= since:
                fresh.append(str(so))
    if fresh:
        return {"verified": True, "status": "ok", "fresh_lib_so": sorted(set(fresh))[:8]}
    return {
        "verified": False,
        "status": "stale",
        "reason": ("no freshly-built cpp_itfs lib.so found after re-baseline; served binary is stale"),
        "build_dir": str(build_dir),
        "module_names": module_names,
    }
