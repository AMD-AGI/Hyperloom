#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Apply an optimized kernel file with source/artifact backup and fast revert."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


COMPILED_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".hip"}
PYTHON_SOURCE_SUFFIXES = {".py"}
COMPILED_ARTIFACT_SUFFIXES = {".so", ".co", ".hsaco"}
TEXT_ARTIFACT_SUFFIXES = {".txt", ".md", ".markdown", ".log", ".patch", ".diff"}
# Fallback when ``inference_optimizer`` is not on ``sys.path`` (standalone CLI).
_FALLBACK_KNOWN_TARGET_ROOTS: tuple[str, ...] = (
    "/sgl-workspace/aiter/",
    "/sgl-workspace/sglang/",
    "/sgl-workspace/vllm/",
    "/opt/venv/lib/python3.10/site-packages/aiter/",
    "/opt/venv/lib/python3.10/site-packages/sglang/",
    "/opt/venv/lib/python3.10/site-packages/vllm/",
    "/usr/local/lib/python3.12/dist-packages/aiter/",
    "/usr/local/lib/python3.12/dist-packages/sglang/",
    "/usr/local/lib/python3.12/dist-packages/vllm/",
    "/usr/local/lib/python3.10/dist-packages/aiter/",
    "/usr/local/lib/python3.10/dist-packages/sglang/",
    "/usr/local/lib/python3.10/dist-packages/vllm/",
)

_CACHED_KNOWN_TARGET_ROOTS: tuple[str, ...] | None = None


def known_target_roots() -> tuple[str, ...]:
    """Resolved framework roots (importlib/glob when orchestrator is importable)."""
    global _CACHED_KNOWN_TARGET_ROOTS
    if _CACHED_KNOWN_TARGET_ROOTS is not None:
        return _CACHED_KNOWN_TARGET_ROOTS
    try:
        from inference_optimizer.orchestrator.framework_paths import (
            resolve_patch_target_roots,
        )

        _CACHED_KNOWN_TARGET_ROOTS = resolve_patch_target_roots()
    except ImportError:
        _CACHED_KNOWN_TARGET_ROOTS = _FALLBACK_KNOWN_TARGET_ROOTS
    return _CACHED_KNOWN_TARGET_ROOTS


# Backward-compat alias for tests / external imports.
KNOWN_TARGET_ROOTS = _FALLBACK_KNOWN_TARGET_ROOTS


# Pod-local multi-node backup dir; survives sglang restarts.
# Overridable via $HYPERLOOM_MN_KERNEL_BACKUP_DIR for tests.
_MN_POD_BACKUP_DIR_DEFAULT = "/var/kernel_patch_backups"

# Multi-node signal file (nodes >= 2) written by multi_node create-rayjob.
# $MULTI_NODE_STATE_FILE wins; env override keeps test runs isolated.
_MN_STATE_FILE_DEFAULT = "/tmp/multi_node_state.json"


def _mn_state_path() -> Path:
    """Resolve where ``inference_optimizer.multi_node`` dropped its state."""
    return Path(os.environ.get("MULTI_NODE_STATE_FILE", _MN_STATE_FILE_DEFAULT))


# Legacy module attribute kept for direct importers; runtime uses _mn_state_path.
_MN_STATE_FILE = Path(_MN_STATE_FILE_DEFAULT)


def _is_multi_node() -> bool:
    """True iff a multi-node RayJob is active (nodes >= 2); missing/unreadable → False."""
    state_path = _mn_state_path()
    try:
        if not state_path.is_file():
            return False
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return int(data.get("nodes") or 0) >= 2
    except (OSError, ValueError):
        return False


def _dispatch_multinode_apply(
    *,
    target_file: Path,
    patch_path: Path,
    kernel_id: str,
    backup_dir_on_pod: str,
    timeout_sec: int = 180,
) -> dict[str, Any]:
    """Fan the patch out to every pod (head + workers); return parsed JSON.

    Raises RuntimeError on subprocess failure / non-JSON / pod status != ok so
    the caller can roll back the sandbox-local copy.
    """
    cmd = [
        sys.executable, "-m", "inference_optimizer.multi_node",
        "apply-patch",
        "--patch-file", str(patch_path),
        "--target-path", str(target_file),
        "--backup-dir", backup_dir_on_pod,
        "--kernel-id", kernel_id or "",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"multi-node apply-patch returned rc={proc.returncode}: "
            f"stderr={(proc.stderr or '')[-2000:]!r}"
        )
    try:
        parsed = json.loads(proc.stdout.strip().splitlines()[-1]) \
            if proc.stdout.strip().startswith("{") is False \
            else json.loads(proc.stdout)
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"multi-node apply-patch stdout not JSON: {exc!r}; "
            f"stdout_tail={(proc.stdout or '')[-2000:]!r}"
        ) from exc
    if str(parsed.get("status", "")).lower() != "ok":
        raise RuntimeError(
            f"multi-node apply-patch reported status={parsed.get('status')!r}: "
            f"failures={parsed.get('failures')!r}"
        )
    return parsed


def _dispatch_multinode_revert(
    *,
    target_path: str,
    backup_map: dict[str, str],
    timeout_sec: int = 120,
) -> dict[str, Any]:
    """Restore the original file on every pod that received the apply (best-effort)."""
    cmd = [
        sys.executable, "-m", "inference_optimizer.multi_node",
        "revert-patch",
        "--target-path", str(target_path),
        "--backup-map-json", json.dumps(backup_map, sort_keys=True),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_sec,
    )
    out = (proc.stdout or "").strip()
    try:
        parsed = json.loads(out) if out.startswith("{") else {}
    except json.JSONDecodeError:
        parsed = {}
    if proc.returncode != 0 or str(parsed.get("status", "")).lower() != "ok":
        # Warn-only: sandbox revert already won.
        sys.stderr.write(
            f"WARN multi-node revert-patch rc={proc.returncode} "
            f"status={parsed.get('status')!r} "
            f"stderr_tail={(proc.stderr or '')[-1000:]!r}\n"
        )
    return parsed or {
        "status": "partial",
        "returncode": proc.returncode,
        "stdout_tail": out[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def _now() -> str:
    """Return the current UTC timestamp as an ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    """Sanitize a filename component to a safe, short identifier."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned[:80] or "kernel"


def _path_hash(path: Path) -> str:
    """Return a short, stable hash for ``path`` to disambiguate backups."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def _copy_to_backup(path: Path, backup_dir: Path, group: str) -> dict[str, str]:
    """Copy a file into the backup tree and return the manifest entry."""
    dst = backup_dir / group / f"{_path_hash(path)}_{path.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return {"path": str(path), "backup_path": str(dst)}


def _source_text_looks_complete(text: str, suffix: str) -> bool:
    """Heuristically check that the patch text is a full source file."""
    stripped = text.strip()
    if not stripped or "```" in stripped:
        return False
    if suffix == ".py":
        try:
            compile(stripped + "\n", "<optimized_kernel>", "exec")
        except SyntaxError:
            return False
        return any(
            marker in stripped
            for marker in ("def ", "class ", "import ", "@triton.jit", "torch.")
        )
    if suffix in COMPILED_SOURCE_SUFFIXES:
        return any(
            marker in stripped
            for marker in (
                "#include", "__global__", "__device__", "extern ", "namespace ",
                "template", "void ", "int ", "float ", "half", "torch::",
            )
        )
    return False


def _validate_patch_source(patch: Path, target: Path) -> None:
    patch_suffix = patch.suffix.lower()
    target_suffix = target.suffix.lower()
    if patch_suffix in TEXT_ARTIFACT_SUFFIXES:
        raise ValueError(f"patch_path is not a complete source file: {patch}")
    if patch_suffix != target_suffix:
        raise ValueError(
            f"patch suffix {patch_suffix or '<none>'} does not match "
            f"target suffix {target_suffix or '<none>'}"
        )
    try:
        text = patch.read_text(encoding="utf-8", errors="replace")
    except UnicodeDecodeError as exc:
        raise ValueError(f"patch_path is not text source: {patch}") from exc
    if not _source_text_looks_complete(text, target_suffix):
        raise ValueError(f"patch_path does not look like a complete {target_suffix} source file: {patch}")
    target_text = target.read_text(encoding="utf-8", errors="replace")
    _validate_replacement_compatibility(text, target_text, target)
    if target_suffix == ".py":
        py_compile.compile(str(patch), doraise=True)


def _host_entry_functions(source_text: str) -> set[str]:
    names: set[str] = set()
    pattern = re.compile(
        r"(?m)^\s*(?!__global__)(?:template\s*<[^>]+>\s*)?"
        r"(?:void|int|bool|float|double|auto|at::Tensor|torch::Tensor)\s+"
        r"([A-Za-z_]\w*)\s*\("
    )
    for match in pattern.finditer(source_text):
        name = match.group(1)
        if name.startswith("_") or name in {"main"}:
            continue
        names.add(name)
    return names


def _validate_replacement_compatibility(patch_text: str, target_text: str, target: Path) -> None:
    if "PYBIND11_MODULE" in patch_text and "PYBIND11_MODULE" not in target_text:
        raise ValueError(
            "patch creates a standalone PYBIND11 module but target is a framework source file"
        )
    if "TORCH_LIBRARY" in patch_text and "TORCH_LIBRARY" not in target_text:
        raise ValueError(
            "patch creates standalone TORCH_LIBRARY registration absent from target"
        )
    if "namespace aiter" in target_text and "namespace aiter" not in patch_text:
        raise ValueError("patch does not preserve namespace aiter")

    required = _host_entry_functions(target_text)
    if required:
        present = _host_entry_functions(patch_text)
        missing = sorted(required - present)
        if missing:
            raise ValueError(
                f"patch does not preserve target host entry function(s) for {target}: "
                + ", ".join(missing[:12])
            )


def _clear_python_kernel_caches(target: Path) -> dict[str, Any]:
    removed: list[str] = []

    def remove_path(path: Path) -> None:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            removed.append(str(path))
        except OSError:
            return

    for cache_dir in target.parent.rglob("__pycache__"):
        remove_path(cache_dir)

    home = Path(os.environ.get("HOME", "~")).expanduser()
    for path in (
        home / ".triton" / "cache",
        home / ".cache" / "triton",
        home / ".cache" / "torch" / "inductor",
    ):
        if path.exists():
            remove_path(path)
    for pattern in ("/tmp/torchinductor_*", "/tmp/triton_*"):
        for path in Path("/tmp").glob(Path(pattern).name):
            if path.exists():
                remove_path(path)
    return {"status": "ok", "removed": removed}


# PR-K: aiter JIT cache invalidation around rebuilds (setup.py develop won't invalidate jit/build/ .so).
_AITER_CSRC_MARKER = "/aiter/csrc/"


def _target_is_in_aiter_csrc(target_file: Path) -> bool:
    """Return True iff ``target_file`` resides under any ``aiter/csrc/`` tree."""
    return _AITER_CSRC_MARKER in str(target_file).replace(os.sep, "/")


def _aiter_jit_build_dir() -> Path | None:
    """Return ``<aiter>/jit/build`` for the importable aiter, or ``None`` if absent."""
    try:
        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    aiter_pkg = Path(list(spec.submodule_search_locations)[0])
    return aiter_pkg / "jit" / "build"


def _invalidate_aiter_jit_build(
    target_file: Path,
    backup_dir: Path,
    *,
    jit_build_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Move aiter ``jit/build/`` aside so a post-rebuild first import re-JITs.

    Returns status ok / skipped / failed; ``jit_build_dir_override`` is test-only.
    """
    if not _target_is_in_aiter_csrc(target_file):
        return {"status": "skipped", "reason": "target not under aiter/csrc/"}
    jit_build = jit_build_dir_override or _aiter_jit_build_dir()
    if jit_build is None:
        return {"status": "skipped", "reason": "aiter package not importable"}
    if not jit_build.exists():
        return {"status": "skipped", "reason": "aiter jit/build/ does not exist"}
    try:
        is_empty = not any(jit_build.iterdir())
    except OSError as exc:
        return {
            "status": "failed",
            "error": f"failed to scan {jit_build}: {exc}",
            "src": str(jit_build),
        }
    if is_empty:
        return {"status": "skipped", "reason": "aiter jit/build/ is empty"}
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "jit_build"
    if backup_path.exists():
        return {
            "status": "failed",
            "error": f"jit/build backup path already exists: {backup_path}",
            "src": str(jit_build),
        }
    try:
        shutil.move(str(jit_build), str(backup_path))
    except (OSError, shutil.Error) as exc:
        return {
            "status": "failed",
            "error": f"shutil.move failed: {exc}",
            "src": str(jit_build),
            "backup_path": str(backup_path),
        }
    return {
        "status": "ok",
        "src": str(jit_build),
        "backup_path": str(backup_path),
        "moved_at": _now(),
    }


def _restore_aiter_jit_build(jit_build_backup: dict[str, Any]) -> dict[str, Any]:
    """Reverse of :func:`_invalidate_aiter_jit_build`: restore the backup, removing any regenerated dir first."""
    if not isinstance(jit_build_backup, dict) or jit_build_backup.get("status") != "ok":
        return {"status": "skipped", "reason": "no backup recorded"}
    src = Path(jit_build_backup.get("src", ""))
    backup_path = Path(jit_build_backup.get("backup_path", ""))
    if not src or not backup_path:
        return {"status": "skipped", "reason": "incomplete backup record"}
    if not backup_path.exists():
        return {
            "status": "skipped",
            "reason": f"backup path missing: {backup_path}",
        }
    if src.exists():
        try:
            shutil.rmtree(src)
        except OSError as exc:
            return {
                "status": "failed",
                "error": f"failed to clear regenerated jit/build/: {exc}",
                "src": str(src),
            }
    try:
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(backup_path), str(src))
    except (OSError, shutil.Error) as exc:
        return {
            "status": "failed",
            "error": f"shutil.move failed during restore: {exc}",
            "src": str(src),
            "backup_path": str(backup_path),
        }
    return {"status": "ok", "restored_to": str(src)}


# ---------------------------------------------------------------------------
# PR-K2: aiter cpp_itfs RUNTIME-compiled cache invalidation.
#
# Distinct from the ``@compile_ops`` jit/build cache handled above. aiter's
# ``csrc/cpp_itfs`` kernels (e.g. paged_attention -> ``pa_ragged``) are NOT
# produced by ``setup.py develop``; they are runtime-compiled on first call
# by ``compile_template_op`` (aiter ``csrc/cpp_itfs/utils.py``) into
# ``$AITER_ROOT_DIR/build/<md_name>_<md5(params)>/lib.so`` (default
# ``$HOME/.aiter/build``). The cache folder name hashes kernel *parameters*,
# NOT source content, so the pristine and the patched build of the same
# kernel collide on the SAME directory. ``compile_template_op`` rebuilds only
# when ``lib.so`` is missing (``not_built``), so after we patch the ``.cuh``
# and run the (no-op for this class) ``setup.py develop``, the next server
# reuses the STALE pristine ``lib.so``; the integrate re-baseline then
# measures ~0% and a genuinely-good kernel is flagged NEEDS_REVIEW / REVERT
# (observed -0.17% on a +2.5% paged_attention kernel; see GH #458).
#
# ``setup.py develop`` cannot refresh this kernel class, and the jit/build
# move above never touches ``$HOME/.aiter/build``. So for cpp_itfs targets we
# ALSO move the affected runtime-cache dirs aside before the rebuild step --
# scoped to the patched module's ``MD_NAME`` prefix(es) when determinable,
# else the whole cpp_itfs build root the scheduler uses -- so the re-baseline
# server runtime-recompiles the patched kernel from clean state.
# ``shutil.move`` keeps it reversible; :func:`_restore_aiter_cpp_itfs_cache`
# moves the backup back on revert. ``integrate_handler`` additionally sets
# ``AITER_REBUILD=1`` on the re-baseline server and gates KEEP on a verified
# fresh rebuild via :func:`verify_cpp_itfs_rebuilt`.
#
# Scope: ONLY aiter cpp_itfs targets. Non-cpp_itfs aiter targets, sglang and
# vllm keep their current behaviour bit-for-bit (this is a no-op for them).
# ---------------------------------------------------------------------------
_AITER_CPP_ITFS_MARKER = "/aiter/csrc/cpp_itfs/"
_MD_NAME_RE = re.compile(r"""(?m)^\s*MD_NAME\s*=\s*["']([^"']+)["']""")


def _target_is_in_aiter_cpp_itfs(target_file: Path) -> bool:
    """True iff ``target_file`` lives under any ``aiter/csrc/cpp_itfs/`` tree.

    Strict subset of :func:`_target_is_in_aiter_csrc`: these are the
    runtime-compiled kernels whose served ``.so`` lives in
    ``$HOME/.aiter/build`` rather than in ``<aiter>/jit/build`` or the
    statically-linked wheel. Matches both the editable checkout
    (``/sgl-workspace/aiter/csrc/cpp_itfs/...``) and the dist-packages
    layout (``.../aiter/csrc/cpp_itfs/...``).
    """
    return _AITER_CPP_ITFS_MARKER in str(target_file).replace(os.sep, "/")


def _aiter_cpp_itfs_build_dir() -> Path:
    """Resolve aiter's cpp_itfs runtime ``BUILD_DIR``.

    Mirrors aiter ``csrc/cpp_itfs/utils.py``: ``$AITER_ROOT_DIR/build`` with
    ``$AITER_ROOT_DIR`` defaulting to ``$HOME/.aiter``. Honouring both env
    vars keeps non-default deployments + unit tests correct without importing
    aiter into this standalone tool.
    """
    root = os.environ.get("AITER_ROOT_DIR", "").strip()
    if not root:
        home = Path(os.environ.get("HOME", "~")).expanduser()
        root = str(home / ".aiter")
    return Path(root) / "build"


def _cpp_itfs_module_names(target_file: Path) -> list[str]:
    """Best-effort ``MD_NAME`` prefix(es) for the cpp_itfs module(s) the
    patched source feeds.

    The cpp_itfs ``.py`` driver next to the patched source declares
    ``MD_NAME = "pa_ragged"`` (etc.), which becomes the
    ``<md_name>_<hash>`` runtime-cache folder prefix. A single shared
    ``.cuh`` (e.g. ``pa_kernels.cuh``) is pulled into several drivers in the
    same directory, so we collect EVERY ``MD_NAME`` declared in the target's
    directory. An empty result tells the caller to fall back to clearing the
    whole cpp_itfs build root.
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

    Returns one of ``ok`` / ``skipped`` / ``failed`` mirroring
    :func:`_invalidate_aiter_jit_build`. ``build_dir_override`` is a
    test-only hook.
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
        "invalidated_at": _now(),
        "invalidated_unix": time.time(),
    }
    if not build_dir.exists():
        # Nothing cached yet -> the re-baseline server will build fresh into
        # this dir on first kernel call. No stale binary to mask.
        record.update({
            "status": "skipped",
            "reason": "cpp_itfs build dir does not exist",
            "moved": [],
        })
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
        record.update({
            "status": "skipped",
            "reason": "no matching cpp_itfs cache entries",
            "moved": [],
        })
        return record
    cache_backup_root = backup_dir / "cpp_itfs_cache"
    moved: list[dict[str, str]] = []
    try:
        cache_backup_root.mkdir(parents=True, exist_ok=True)
        for src in to_move:
            dst = cache_backup_root / src.name
            if dst.exists():
                record.update({
                    "status": "failed",
                    "error": f"cpp_itfs cache backup path already exists: {dst}",
                    "moved": moved,
                })
                return record
            shutil.move(str(src), str(dst))
            moved.append({"src": str(src), "backup_path": str(dst)})
    except (OSError, shutil.Error) as exc:
        record.update({
            "status": "failed",
            "error": f"shutil.move failed: {exc}",
            "moved": moved,
        })
        return record
    record.update({"status": "ok", "moved": moved})
    return record


def _restore_aiter_cpp_itfs_cache(cache_backup: dict[str, Any]) -> dict[str, Any]:
    """Reverse :func:`_invalidate_aiter_cpp_itfs_cache` (revert path).

    Moves each backed-up cache dir back to its original location, removing
    any dir the re-baseline server regenerated there first so the pre-patch
    runtime cache is restored bit-for-bit.
    """
    if not isinstance(cache_backup, dict) or cache_backup.get("status") != "ok":
        return {"status": "skipped", "reason": "no cpp_itfs cache backup recorded"}
    moved = cache_backup.get("moved") or []
    if not moved:
        return {"status": "skipped", "reason": "nothing was moved"}
    restored: list[str] = []
    for entry in moved:
        src = Path(entry.get("src", ""))  # original cache location
        backup_path = Path(entry.get("backup_path", ""))
        if not str(src) or not str(backup_path) or not backup_path.exists():
            continue
        if src.exists():
            try:
                shutil.rmtree(src)
            except OSError as exc:
                return {
                    "status": "failed",
                    "error": f"failed to clear regenerated cache dir {src}: {exc}",
                }
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup_path), str(src))
        except (OSError, shutil.Error) as exc:
            return {
                "status": "failed",
                "error": f"shutil.move failed during restore: {exc}",
            }
        restored.append(str(src))
    return {"status": "ok", "restored": restored}


def verify_cpp_itfs_rebuilt(cache_backup: dict[str, Any]) -> dict[str, Any]:
    """Assert the re-baseline actually runtime-recompiled the patched kernel.

    After :func:`_invalidate_aiter_cpp_itfs_cache` moved the stale cache
    aside, a faithful re-baseline must repopulate
    ``<build_dir>/<md_name>_*/lib.so`` (the served binary) with an mtime at
    or after the invalidation. If no such fresh ``lib.so`` exists, the server
    dlopened a stale binary (or the kernel was never exercised) and the
    measured gain is meaningless -- the integrate KEEP/REVERT gate uses this
    to flag/abort instead of scoring a stale binary (GH #458 point 2).

    Returns ``{"verified": bool, ...}``. ``verified`` is True for
    non-cpp_itfs targets so the caller's gate is a strict no-op off the
    cpp_itfs path.
    """
    if not isinstance(cache_backup, dict) or not cache_backup.get("is_cpp_itfs"):
        return {"verified": True, "status": "skipped", "reason": "non-cpp_itfs target"}
    build_dir = Path(cache_backup.get("build_dir", ""))
    since = float(cache_backup.get("invalidated_unix") or 0.0)
    module_names = list(cache_backup.get("module_names") or [])
    if not str(build_dir) or not build_dir.exists():
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
        "reason": (
            "no freshly-built cpp_itfs lib.so found after re-baseline; "
            "served binary is stale"
        ),
        "build_dir": str(build_dir),
        "module_names": module_names,
    }


def _detect_strategy(target_file: Path, *, allow_unknown_target: bool) -> dict[str, Any]:
    target = str(target_file)
    lower = target.lower()
    roots = known_target_roots()
    if not allow_unknown_target and not any(root in lower for root in roots):
        raise ValueError(f"target_file is outside known reusable source roots: {target_file}")

    suffix = target_file.suffix.lower()
    compiled = suffix in COMPILED_SOURCE_SUFFIXES
    root = None
    rebuild_command: list[str] = []
    artifact_roots: list[Path] = []

    if "/sgl-workspace/aiter/" in lower:
        root = Path("/sgl-workspace/aiter")
        rebuild_command = ["/opt/venv/bin/python", "setup.py", "develop"]
        artifact_roots = [root]
    elif "/sgl-workspace/sglang/sgl-kernel/" in lower:
        root = Path("/sgl-workspace/sglang/sgl-kernel")
        rebuild_command = ["/opt/venv/bin/python", "-m", "pip", "install", "-e", "."]
        artifact_roots = [root]
    elif "/sgl-workspace/sglang/" in lower:
        root = Path("/sgl-workspace/sglang")
        rebuild_command = ["/opt/venv/bin/python", "-m", "pip", "install", "-e", "python"]
        artifact_roots = [root]
    elif "/sgl-workspace/vllm/" in lower:
        root = Path("/sgl-workspace/vllm")
        rebuild_command = ["/opt/venv/bin/python", "-m", "pip", "install", "-e", "."]
        artifact_roots = [root]
    elif allow_unknown_target:
        root = target_file.parent

    if suffix in PYTHON_SOURCE_SUFFIXES:
        compiled = False
        rebuild_command = []
        artifact_roots = []

    return {
        "compiled": compiled,
        "root": str(root) if root else "",
        "rebuild_command": rebuild_command,
        "artifact_roots": artifact_roots,
    }


def _discover_artifacts(
    roots: Iterable[Path],
    explicit_artifact_paths: Iterable[str] | None,
    *,
    max_artifacts: int = 400,
) -> list[Path]:
    seen: set[Path] = set()
    artifacts: list[Path] = []
    for raw in explicit_artifact_paths or []:
        p = Path(raw)
        if p.is_file() and p not in seen:
            artifacts.append(p)
            seen.add(p)
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if len(artifacts) >= max_artifacts:
                break
            if p.is_file() and p.suffix.lower() in COMPILED_ARTIFACT_SUFFIXES and p not in seen:
                artifacts.append(p)
                seen.add(p)
    return artifacts


def _run_rebuild(command: list[str], cwd: Path, timeout_sec: int) -> dict[str, Any]:
    if not command:
        return {"status": "skipped", "reason": "no rebuild command"}
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env={**os.environ, "PATH": f"/opt/venv/bin:{os.environ.get('PATH', '')}"},
    )
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "command": command,
        "cwd": str(cwd),
    }


def revert_kernel_patch(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    restored: list[str] = []
    for item in manifest.get("artifacts", []):
        src = Path(item["backup_path"])
        dst = Path(item["path"])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(str(dst))
    source_backup = manifest.get("source_backup") or {}
    if source_backup:
        src = Path(source_backup["backup_path"])
        dst = Path(source_backup["path"])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(str(dst))
            if dst.suffix.lower() in PYTHON_SOURCE_SUFFIXES:
                manifest["revert_cache_clear"] = _clear_python_kernel_caches(dst)

    # PR-K: restore aiter jit/build/ (before multi-node fan-out) if apply moved it aside.
    jit_build_backup = manifest.get("jit_build_backup") or {}
    if jit_build_backup.get("status") == "ok":
        jit_build_restore = _restore_aiter_jit_build(jit_build_backup)
        manifest["jit_build_restore"] = jit_build_restore
        if jit_build_restore.get("status") == "ok" and jit_build_restore.get("restored_to"):
            restored.append(str(jit_build_restore["restored_to"]))

    # PR-K2: restore the aiter cpp_itfs runtime cache moved aside during apply so a non-KEEP decision serves v0 (only present when apply moved cpp_itfs cache dirs).
    cpp_itfs_cache_backup = manifest.get("cpp_itfs_cache_backup") or {}
    if cpp_itfs_cache_backup.get("status") == "ok":
        cpp_itfs_cache_restore = _restore_aiter_cpp_itfs_cache(cpp_itfs_cache_backup)
        manifest["cpp_itfs_cache_restore"] = cpp_itfs_cache_restore
        if cpp_itfs_cache_restore.get("status") == "ok":
            restored.extend(cpp_itfs_cache_restore.get("restored", []))

    # Multi-node: fan-out a revert to every pod that received the apply (best-effort) so pod-side sglang loads v0 on next restart.
    multinode_info = manifest.get("multinode") or {}
    mn_revert: dict[str, Any] = {}
    if multinode_info and multinode_info.get("host_backup_map"):
        target_path = multinode_info.get("target_path") or (
            source_backup.get("path") if source_backup else ""
        )
        backup_map = multinode_info.get("host_backup_map") or {}
        if target_path and backup_map:
            try:
                mn_revert = _dispatch_multinode_revert(
                    target_path=target_path,
                    backup_map=backup_map,
                )
            except Exception as exc:  # noqa: BLE001
                mn_revert = {"status": "failed", "error": str(exc)}

    reverted_at = _now()
    manifest["status"] = "reverted"
    manifest["reverted_at"] = reverted_at
    manifest["restored_paths"] = restored
    if mn_revert:
        manifest["multinode_revert"] = mn_revert
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "status": "ok",
        "manifest_path": str(manifest_file),
        "restored_paths": restored,
        "reverted_at": reverted_at,
    }
    # Only attach multinode_revert when fan-out actually ran.
    if mn_revert:
        result["multinode_revert"] = mn_revert
    return result


def apply_kernel_patch(
    *,
    patch_path: str | Path,
    target_file: str | Path,
    backup_root: str | Path,
    kernel_id: str = "",
    artifact_paths: Iterable[str] | None = None,
    rebuild_command: list[str] | str | None = None,
    rebuild_timeout_sec: int = 1800,
    skip_rebuild: bool = False,
    allow_unknown_target: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    patch = Path(patch_path).resolve()
    target = Path(target_file).resolve()
    if not patch.is_file():
        return {"status": "failed", "error": f"patch_path does not exist: {patch}"}
    if not target.is_file():
        return {"status": "failed", "error": f"target_file does not exist: {target}"}

    try:
        strategy = _detect_strategy(target, allow_unknown_target=allow_unknown_target)
        _validate_patch_source(patch, target)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc)}

    backup_dir = Path(backup_root) / f"{_safe_name(kernel_id or target.stem)}_{_path_hash(target)}"
    manifest_path = backup_dir / "manifest.json"
    source_backup = _copy_to_backup(target, backup_dir, "source")
    artifacts: list[dict[str, str]] = []
    if strategy["compiled"]:
        found = _discover_artifacts(strategy["artifact_roots"], artifact_paths)
        artifacts = [_copy_to_backup(path, backup_dir, "artifacts") for path in found]

    manifest = {
        "status": "prepared",
        "kernel_id": kernel_id,
        "patch_path": str(patch),
        "target_file": str(target),
        "source_backup": source_backup,
        "artifacts": artifacts,
        "strategy": {
            "compiled": strategy["compiled"],
            "root": strategy["root"],
        },
        "created_at": _now(),
    }
    backup_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dry_run:
        return {"status": "ok", "dry_run": True, "manifest_path": str(manifest_path)}

    shutil.copy2(patch, target)
    cache_clear = (
        _clear_python_kernel_caches(target)
        if target.suffix.lower() in PYTHON_SOURCE_SUFFIXES else
        {"status": "skipped", "reason": "not a python source target"}
    )

    # Multi-node: fan-out the patch to every RayJob pod, else hard-revert the sandbox copy.
    multinode_info: dict[str, Any] = {}
    if _is_multi_node():
        pod_backup_dir = os.environ.get(
            "HYPERLOOM_MN_KERNEL_BACKUP_DIR", _MN_POD_BACKUP_DIR_DEFAULT,
        )
        try:
            mn_apply = _dispatch_multinode_apply(
                target_file=target,
                patch_path=patch,
                kernel_id=kernel_id,
                backup_dir_on_pod=pod_backup_dir,
            )
        except Exception as exc:  # noqa: BLE001
            # Pod fan-out failed: revert sandbox copy to v0.
            try:
                shutil.copy2(source_backup["backup_path"], target)
            except OSError:
                pass
            return {
                "status": "failed",
                "error": (
                    "multi-node apply fan-out failed; sandbox copy "
                    f"reverted to {source_backup['backup_path']}: {exc}"
                ),
                "manifest_path": str(manifest_path),
            }
        # Persist per-host backups {hostname: backup_path} so revert can find them.
        backup_map: dict[str, str] = {}
        for entry in mn_apply.get("per_node", []) or []:
            host = (entry.get("host") or "").strip()
            bp = (entry.get("backup_path") or "").strip()
            if host and bp:
                backup_map[host] = bp
        multinode_info = {
            "status": "ok",
            "target_path": str(target),
            "backup_dir_on_pod": pod_backup_dir,
            "host_backup_map": backup_map,
            "per_node": mn_apply.get("per_node", []),
        }
        # Persist multinode info now so a later rebuild failure reverts pod-side patches too.
        manifest["multinode"] = multinode_info
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    command: list[str] = []
    if isinstance(rebuild_command, str):
        command = ["/bin/bash", "-lc", rebuild_command]
    elif rebuild_command:
        command = list(rebuild_command)
    elif not skip_rebuild:
        command = list(strategy["rebuild_command"])

    rebuild = {"status": "skipped", "reason": "source-only patch or skip_rebuild=true"}
    jit_build_backup: dict[str, Any] = {
        "status": "skipped", "reason": "rebuild not run",
    }
    cpp_itfs_cache_backup: dict[str, Any] = {
        "status": "skipped", "reason": "rebuild not run", "is_cpp_itfs": False,
    }
    if strategy["compiled"] and not skip_rebuild:
        # PR-K: move aiter jit/build/ aside so post-rebuild import re-JITs cleanly.
        jit_build_backup = _invalidate_aiter_jit_build(target, backup_dir)
        if jit_build_backup.get("status") == "failed":
            # Refuse to rebuild against an inconsistent jit cache: restore v0 and bail.
            try:
                shutil.copy2(source_backup["backup_path"], target)
            except OSError:
                pass
            return {
                "status": "failed",
                "error_class": "aiter_jit_invalidation_failed",
                "error": (
                    "aiter jit/build/ invalidation failed: "
                    f"{jit_build_backup.get('error')}"
                ),
                "manifest_path": str(manifest_path),
                "jit_build_backup": jit_build_backup,
            }
        if jit_build_backup.get("status") == "ok":
            # Persist the backup record before rebuild so a failure can still restore.
            manifest["jit_build_backup"] = jit_build_backup
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        # PR-K2: aiter cpp_itfs kernels (e.g. paged_attention -> pa_ragged)
        # are runtime-compiled into $HOME/.aiter/build/<md_name>_<hash>/ on
        # first call, NOT by setup.py develop, and the dir name hashes
        # params (not source) so pristine + patched collide -> the next
        # server reuses the stale .so. Move the affected runtime-cache dirs
        # aside so the re-baseline recompiles from clean state (GH #458).
        # No-op for non-cpp_itfs targets (sglang / vllm / other aiter csrc).
        cpp_itfs_cache_backup = _invalidate_aiter_cpp_itfs_cache(target, backup_dir)
        if cpp_itfs_cache_backup.get("status") == "failed":
            # Refuse to re-baseline against a stale runtime cache: restore
            # source + jit/build (if moved) so on-disk state matches v0,
            # then bail rather than score a possibly-stale binary.
            try:
                shutil.copy2(source_backup["backup_path"], target)
            except OSError:
                pass
            if jit_build_backup.get("status") == "ok":
                _restore_aiter_jit_build(jit_build_backup)
            return {
                "status": "failed",
                "error_class": "aiter_cpp_itfs_invalidation_failed",
                "error": (
                    "aiter cpp_itfs runtime cache invalidation failed: "
                    f"{cpp_itfs_cache_backup.get('error')}"
                ),
                "manifest_path": str(manifest_path),
                "cpp_itfs_cache_backup": cpp_itfs_cache_backup,
            }
        if cpp_itfs_cache_backup.get("status") == "ok":
            # Persist BEFORE rebuild so a rebuild failure can restore the
            # moved-aside runtime cache via revert_kernel_patch.
            manifest["cpp_itfs_cache_backup"] = cpp_itfs_cache_backup
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        cwd = Path(strategy["root"] or target.parent)
        rebuild = _run_rebuild(command, cwd, rebuild_timeout_sec)
        if rebuild["status"] != "ok":
            revert = revert_kernel_patch(manifest_path)
            return {
                "status": "failed",
                "error": "rebuild failed; original source/artifacts restored",
                "manifest_path": str(manifest_path),
                "rebuild": rebuild,
                "revert": revert,
            }

    manifest["status"] = "applied"
    manifest["applied_at"] = _now()
    manifest["rebuild"] = rebuild
    manifest["cache_clear"] = cache_clear
    if jit_build_backup.get("status") in {"ok", "skipped"}:
        # Surface skipped reason too so manifest readers can audit it.
        manifest["jit_build_backup"] = jit_build_backup
    if cpp_itfs_cache_backup.get("status") in {"ok", "skipped"}:
        # Surface the cpp_itfs record (is_cpp_itfs + build_dir + module_names + invalidated_unix) so integrate can verify a rebuild and revert can restore the runtime cache.
        manifest["cpp_itfs_cache_backup"] = cpp_itfs_cache_backup
    # multinode block already persisted at fan-out time; don't rewrite it here.
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "target_file": str(target),
        "backup_dir": str(backup_dir),
        "compiled": bool(strategy["compiled"]),
        "artifact_count": len(artifacts),
        "cache_clear": cache_clear,
        "rebuild": rebuild,
        "jit_build_backup": jit_build_backup,
        "cpp_itfs_cache_backup": cpp_itfs_cache_backup,
    }
    # Only attach the multinode key when fan-out actually ran.
    if multinode_info:
        result["multinode"] = multinode_info
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply or revert a kernel patch")
    sub = parser.add_subparsers(dest="command", required=True)
    apply_p = sub.add_parser("apply")
    apply_p.add_argument("--patch-path", required=True)
    apply_p.add_argument("--target-file", required=True)
    apply_p.add_argument("--backup-root", required=True)
    apply_p.add_argument("--kernel-id", default="")
    apply_p.add_argument("--artifact-path", action="append", default=[])
    apply_p.add_argument("--rebuild-command", default="")
    apply_p.add_argument("--rebuild-timeout-sec", type=int, default=1800)
    apply_p.add_argument("--skip-rebuild", action="store_true")
    apply_p.add_argument("--allow-unknown-target", action="store_true")
    apply_p.add_argument("--dry-run", action="store_true")

    revert_p = sub.add_parser("revert")
    revert_p.add_argument("--manifest-path", required=True)

    args = parser.parse_args()
    if args.command == "revert":
        result = revert_kernel_patch(args.manifest_path)
    else:
        result = apply_kernel_patch(
            patch_path=args.patch_path,
            target_file=args.target_file,
            backup_root=args.backup_root,
            kernel_id=args.kernel_id,
            artifact_paths=args.artifact_path,
            rebuild_command=args.rebuild_command or None,
            rebuild_timeout_sec=args.rebuild_timeout_sec,
            skip_rebuild=args.skip_rebuild,
            allow_unknown_target=args.allow_unknown_target,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
