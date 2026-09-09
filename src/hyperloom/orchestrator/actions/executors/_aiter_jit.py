# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared aiter build-cache helpers and stale-lock cleanup."""

from __future__ import annotations

import csv
import importlib.util
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# < N .so files under aiter jit/ ⇒ COLD start (first-time JIT compile pending).
COLD_START_KERNEL_THRESHOLD = 20
# COLD-start benchmark cap, including first-time compilation and graph capture.
BASELINE_COLD_START_TIMEOUT_SEC = 9000

# Fallback probe paths for aiter's JIT cache dir; first existing wins.
AITER_JIT_PROBE_PATHS: tuple[str, ...] = (
    "/sgl-workspace/aiter/aiter/jit",
    "/sgl-workspace/aiter/aiter/jit/build",
    "/usr/local/lib/python3.10/dist-packages/aiter/jit",
    "/usr/local/lib/python3.12/dist-packages/aiter/jit",
    "/usr/local/lib/python3.10/site-packages/aiter/jit",
    "/usr/local/lib/python3.12/site-packages/aiter/jit",
    "/opt/venv/lib/python3.10/site-packages/aiter/jit",
    "/opt/venv/lib/python3.12/site-packages/aiter/jit",
)

# Fallback locations for cpp_itfs template builds.
AITER_CPP_BUILD_PROBE_PATHS: tuple[str, ...] = ("/root/.aiter/build",)

# Mtime gate (minutes) for the lock sweep.
AITER_LOCK_STALE_MINUTES = 5

# Process names that indicate an in-flight aiter/ninja compile. hipcc is a wrapper whose ``name`` can surface as
# ``perl``/``sh``, so we also match on the cmdline's first token (see ``_any_live_compiler``).
COMPILER_PROCESS_NAMES = frozenset(
    {
        "hipcc",
        "hipcc.bin",
        "ninja",
        "cc1plus",
        "clang",
        "clang++",
        "clang-cpp",
    }
)

# Lock file names left by aiter / ninja under the jit dir.
_LOCK_NAMES = {"lock", ".ninja_lock"}
_BATON_WAIT_MARKER = "waiting for baton release at"
_BATON_LOG_NAMES = {"server.log", "benchmark_stderr.log", "benchmark_stdout.log"}


def _resolve_aiter_jit_dir_dynamic() -> list[str]:
    """Locate aiter's ``jit/`` dir via Python's import machinery."""
    try:
        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError):  # noqa: BLE001 — aiter not importable
        return []
    if spec is None or not spec.origin:
        return []
    aiter_root = Path(spec.origin).parent
    return [
        str(aiter_root / "jit"),
        str(aiter_root / "jit" / "build"),
    ]


def probe_aiter_jit_cache() -> dict[str, Any]:
    """Inspect aiter's JIT cache and classify the next start as cold or warm."""
    info: dict[str, Any] = {
        "path": None,
        "kernel_count": 0,
        "size_mb": 0,
        "is_cold": None,
        "probe_status": "not_found",
    }
    candidates: list[str] = []
    override = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
    if override:
        candidates.append(override)
    candidates.extend(_resolve_aiter_jit_dir_dynamic())
    candidates.extend(AITER_JIT_PROBE_PATHS)

    try:
        chosen: Path | None = None
        for raw in candidates:
            path = Path(raw)
            if path.exists() and path.is_dir():
                chosen = path
                break
        if chosen is None:
            return info
        info["path"] = str(chosen)

        total_bytes = 0
        kernel_count = 0
        for so_path in chosen.rglob("*.so"):
            try:
                total_bytes += so_path.stat().st_size
                kernel_count += 1
            except OSError:
                continue
        info["kernel_count"] = kernel_count
        info["size_mb"] = total_bytes // (1024 * 1024)
        info["is_cold"] = kernel_count < COLD_START_KERNEL_THRESHOLD
        info["probe_status"] = "found"
        return info
    except Exception as exc:  # noqa: BLE001
        log.warning("aiter_jit: cache probe failed: %s", exc)
        info["probe_status"] = "error"
        info["is_cold"] = None
        return info


def _any_live_compiler(
    build_dirs: list[Path] | None = None,
) -> bool | None:
    """Return True if a compiler associated with the build trees is alive."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        normalized_dirs = [str(path.resolve()) for path in (build_dirs or [])]
        for proc in psutil.process_iter(["name", "cmdline", "cwd"]):
            try:
                info = proc.info
                name = (info.get("name") or "").strip()
                cmdline = info.get("cmdline") or []
                is_compiler = name in COMPILER_PROCESS_NAMES
                if cmdline:
                    first = os.path.basename(str(cmdline[0]).strip())
                    if first in COMPILER_PROCESS_NAMES:
                        is_compiler = True
                if not is_compiler:
                    continue
                if not normalized_dirs:
                    return True
                cwd = str(info.get("cwd") or "")
                command = "\0".join(str(arg) for arg in cmdline)
                if any(
                    cwd == build_dir or cwd.startswith(f"{build_dir}{os.sep}") or build_dir in command
                    for build_dir in normalized_dirs
                ):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as exc:  # noqa: BLE001 — enumeration failed entirely
        log.warning("aiter_jit: compiler-liveness scan failed: %s", exc)
        return None
    return False


def _dedupe_existing_dirs(candidates: list[Path], unreadable: list[str] | None = None) -> list[Path]:
    """Return existing candidate directories once, preserving priority."""
    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalized = candidate.expanduser().resolve()
        except OSError:
            normalized = candidate.expanduser().absolute()
        key = str(normalized)
        if key in seen:
            continue
        try:
            if not normalized.is_dir():
                continue
        except OSError as exc:
            # is_dir() re-raises EACCES (not in pathlib's ignored errnos), so a fallback under an unreadable root is
            # skipped rather than fatal.
            log.warning("aiter lock sweep: cannot inspect %s (%s); skipping", key, exc)
            if unreadable is not None:
                unreadable.append(key)
            continue
        seen.add(key)
        resolved.append(normalized)
    return resolved


def _resolve_lock_sweep_dirs(aiter_jit_dir: Path | None, unreadable: list[str] | None = None) -> list[Path]:
    """Resolve every active aiter build tree that may contain baton locks."""
    if aiter_jit_dir is not None:
        return [aiter_jit_dir]

    candidates: list[Path] = []
    aiter_root = os.environ.get("AITER_ROOT_DIR", "").strip()
    if aiter_root:
        candidates.append(Path(aiter_root) / "build")
    else:
        home = os.environ.get("HOME", "").strip()
        if home:
            candidates.append(Path(home) / ".aiter" / "build")
        candidates.extend(Path(path) for path in AITER_CPP_BUILD_PROBE_PATHS)

    aiter_jit_override = os.environ.get("AITER_JIT_DIR", "").strip()
    if aiter_jit_override:
        override_path = Path(aiter_jit_override)
        candidates.extend([override_path / "build", override_path])
    if aiter_root and aiter_jit_override:
        # Forge sets both variables for a private attempt.
        return _dedupe_existing_dirs(candidates, unreadable)
    override = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
    if override:
        override_path = Path(override)
        # Preserve the legacy explicit-override contract: callers use this variable to constrain a diagnostic/test
        # sweep to one tree.
        return _dedupe_existing_dirs([override_path / "build", override_path], unreadable)
    try:
        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        aiter_root = Path(spec.origin).parent
        candidates.append(aiter_root / "jit" / "build")
    candidates.extend(
        Path(path)
        for path in (
            "/sgl-workspace/aiter/aiter/jit/build",
            "/usr/local/lib/python3.10/dist-packages/aiter/jit/build",
            "/usr/local/lib/python3.12/dist-packages/aiter/jit/build",
            "/opt/venv/lib/python3.10/site-packages/aiter/jit/build",
            "/opt/venv/lib/python3.12/site-packages/aiter/jit/build",
        )
    )
    return _dedupe_existing_dirs(candidates, unreadable)


def _resolve_lock_sweep_dir(aiter_jit_dir: Path | None) -> Path | None:
    """Compatibility wrapper returning the first resolved build tree."""
    if aiter_jit_dir is None:
        override = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
        if override:
            override_path = Path(override)
            preferred = _dedupe_existing_dirs([override_path / "build", override_path])
            if preferred:
                return preferred[0]
    resolved = _resolve_lock_sweep_dirs(aiter_jit_dir)
    return resolved[0] if resolved else None


def clean_stale_aiter_locks(
    aiter_jit_dir: Path | None = None,
    stale_minutes: int = AITER_LOCK_STALE_MINUTES,
) -> dict[str, Any]:
    """Sweep aiter's JIT build trees for stale plain-file locks left by killed runs."""
    stats: dict[str, Any] = {
        "dir": None,
        "dirs": [],
        "scanned": 0,
        "deleted": 0,
        "skipped_fresh": 0,
        "errors": 0,
    }

    unreadable: list[str] = []
    resolved_dirs = _resolve_lock_sweep_dirs(aiter_jit_dir, unreadable)
    stats["errors"] += len(unreadable)
    stats["unreadable"] = unreadable
    if not resolved_dirs:
        return stats

    primary = _resolve_lock_sweep_dir(None) if aiter_jit_dir is None else resolved_dirs[0]
    stats["dir"] = str(primary or resolved_dirs[0])
    stats["dirs"] = [str(path) for path in resolved_dirs]

    threshold_seconds = float(stale_minutes) * 60.0
    now = time.time()
    for resolved in resolved_dirs:
        try:
            walker = os.walk(str(resolved))
            for root, _dirs, files in walker:
                for fname in files:
                    if not (fname in _LOCK_NAMES or fname.startswith("lock_")):
                        continue
                    stats["scanned"] += 1
                    fpath = Path(root) / fname
                    try:
                        age = now - fpath.stat().st_mtime
                    except OSError:
                        stats["errors"] += 1
                        continue
                    if age < threshold_seconds:
                        stats["skipped_fresh"] += 1
                        continue
                    try:
                        fpath.unlink()
                        stats["deleted"] += 1
                    except OSError:
                        stats["errors"] += 1
        except OSError:
            stats["errors"] += 1

    return stats


def sweep_stale_aiter_locks_if_dead(
    aiter_jit_dir: Path | None = None,
) -> dict[str, Any]:
    """Sweep orphaned aiter JIT locks, gated on no live compiler process."""
    resolved_dirs = _resolve_lock_sweep_dirs(aiter_jit_dir)
    alive = _any_live_compiler(resolved_dirs)
    if alive is True:
        return {
            "dir": None,
            "dirs": [],
            "scanned": 0,
            "deleted": 0,
            "skipped_fresh": 0,
            "errors": 0,
            "compiler_alive": True,
            "skipped_live": True,
        }
    if alive is None:
        stats = clean_stale_aiter_locks(
            aiter_jit_dir,
            stale_minutes=AITER_LOCK_STALE_MINUTES,
        )
        stats["compiler_alive"] = None
        return stats
    stats = clean_stale_aiter_locks(
        aiter_jit_dir,
        stale_minutes=AITER_LOCK_STALE_MINUTES,
    )
    stats["compiler_alive"] = False
    return stats


def find_aiter_baton_wait(
    search_root: Path,
    *,
    since_unix: float | None = None,
    max_files: int = 20,
    tail_bytes: int = 256 * 1024,
) -> dict[str, str] | None:
    """Find bounded log evidence of an aiter process blocked on a FileBaton."""
    try:
        candidates = [path for path in search_root.rglob("*") if path.is_file() and path.name in _BATON_LOG_NAMES]
    except OSError:
        return None

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    candidates.sort(key=_mtime, reverse=True)
    if since_unix is not None:
        candidates = [path for path in candidates if _mtime(path) >= since_unix]
    marker_lower = _BATON_WAIT_MARKER.lower()
    for path in candidates[:max_files]:
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - tail_bytes))
                text = handle.read().decode(errors="replace")
        except OSError:
            continue
        marker_index = text.lower().rfind(marker_lower)
        if marker_index < 0:
            continue
        line_end = text.find("\n", marker_index)
        excerpt_end = len(text) if line_end < 0 else min(len(text), line_end + 512)
        return {
            "log_path": str(path),
            "excerpt": text[marker_index:excerpt_end].strip(),
        }
    return None


COMPILED_REGISTRY_MARKER = "not present in the compiled registry"

# Serving modules whose codegen reads the matching AITER_CONFIG_* tune file.
AITER_ENV_TO_SERVING_MODULES: dict[str, tuple[str, ...]] = {
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": (
        "module_gemm_a8w8_blockscale_bpreshuffle",
        "module_gemm_a8w8_blockscale_bpreshuffle_cktile",
        "module_gemm_a8w8_blockscale_bpreshuffle_asm",
    ),
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": (
        "module_gemm_a8w8_blockscale",
        "module_gemm_a8w8_blockscale_cktile",
        "module_gemm_a8w8_blockscale_asm",
    ),
    "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE": (
        "module_gemm_a8w8_bpreshuffle",
        "module_gemm_a8w8_bpreshuffle_cktile",
    ),
    "AITER_CONFIG_GEMM_A8W8": ("module_gemm_a8w8",),
    "AITER_CONFIG_GEMM_A4W4": ("module_gemm_a4w4_blockscale",),
}


def is_aiter_jit_registry_mismatch(*texts: str) -> bool:
    """True when logs show a tuned CSV kernel name missing from the compiled .so."""
    blob = "\n".join(t for t in texts if t).lower()
    return COMPILED_REGISTRY_MARKER in blob


def csv_kernel_names(csv_path: Path) -> set[str]:
    """Return non-empty ``kernelName`` values from a tuned GEMM CSV."""
    names: set[str] = set()
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get("kernelName") or "").strip()
                if name:
                    names.add(name)
    except OSError:
        return set()
    return names


def serving_modules_cover_csv(jit_dir: Path, modules: tuple[str, ...], csv_path: Path) -> bool:
    """True when every CSV kernel name appears in at least one serving .so.

    Missing .so files mean the next start will compile from the current CSV, so
    that case is treated as covered.
    """
    names = csv_kernel_names(csv_path)
    if not names:
        return True
    blobs: list[bytes] = []
    for module in modules:
        so_path = jit_dir / f"{module}.so"
        if so_path.is_file():
            try:
                blobs.append(so_path.read_bytes())
            except OSError:
                return False
    if not blobs:
        return True
    joined = b"".join(blobs)
    return all(name.encode("utf-8") in joined for name in names)


def _resolve_serving_jit_dir() -> Path | None:
    """The aiter ``jit/`` directory that holds serving ``module_*.so`` files."""
    override = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.extend(Path(path) for path in _resolve_aiter_jit_dir_dynamic())
    candidates.extend(Path(path) for path in AITER_JIT_PROBE_PATHS)
    for path in candidates:
        if path.is_dir():
            return path
    return None


def _jit_build_dir(jit_dir: Path) -> Path:
    return jit_dir if jit_dir.name == "build" else jit_dir / "build"


def _unlink_serving_modules(jit_dir: Path, modules: tuple[str, ...]) -> list[str]:
    removed: list[str] = []
    for module in modules:
        so_path = jit_dir / f"{module}.so"
        if not so_path.is_file():
            continue
        try:
            so_path.unlink()
            removed.append(str(so_path))
        except OSError as exc:
            log.warning("failed to unlink serving so %s: %s", so_path, exc)
    return removed


def _invalidate_jit_build(jit_dir: Path, backup_dir: Path) -> dict[str, Any]:
    """Move ``jit/build`` aside so the next import re-runs codegen for the new CSV."""
    try:
        from hyperloom.agents.kernel.tools.apply_kernel_patch import _invalidate_aiter_jit_build
    except ImportError:
        build = _jit_build_dir(jit_dir)
        if not build.is_dir():
            return {"status": "clean", "reason": "aiter jit/build/ does not exist"}
        backup_dir.mkdir(parents=True, exist_ok=True)
        dest = backup_dir / f"jit_build_{time.time_ns()}"
        shutil.move(str(build), str(dest))
        return {"status": "ok", "src": str(build), "backup_path": str(dest)}
    return _invalidate_aiter_jit_build(
        target_file=jit_dir / "core.py",
        backup_dir=backup_dir,
        jit_build_dir=_jit_build_dir(jit_dir),
    )


def _modules_for_envs(envs: dict[str, str] | None) -> tuple[str, ...]:
    if not envs:
        modules: list[str] = []
        for names in AITER_ENV_TO_SERVING_MODULES.values():
            modules.extend(names)
        return tuple(dict.fromkeys(modules))
    modules: list[str] = []
    for key in envs:
        modules.extend(AITER_ENV_TO_SERVING_MODULES.get(str(key), ()))
    return tuple(dict.fromkeys(modules))


def prepare_serving_so_for_csvs(
    envs: dict[str, str],
    *,
    backup_dir: Path | None = None,
) -> dict[str, Any]:
    """Skip when serving .so already covers the CSV; otherwise unlink and re-JIT.

    Args:
        envs: ``AITER_CONFIG_*`` paths about to be used to start the server.
        backup_dir: Where to move ``jit/build`` if invalidation runs.

    Returns:
        A status dict with ``action`` of ``skip``, ``invalidate``, or ``noop``.
    """
    jit_dir = _resolve_serving_jit_dir()
    if jit_dir is None:
        return {"action": "noop", "reason": "aiter jit dir not found"}
    modules_needed: list[str] = []
    for env_var, csv_path_raw in envs.items():
        modules = AITER_ENV_TO_SERVING_MODULES.get(str(env_var), ())
        if not modules:
            continue
        csv_path = Path(str(csv_path_raw))
        if not csv_path.is_file():
            continue
        if serving_modules_cover_csv(jit_dir, modules, csv_path):
            continue
        modules_needed.extend(modules)
    modules_needed_t = tuple(dict.fromkeys(modules_needed))
    if not modules_needed_t:
        return {"action": "skip", "jit_dir": str(jit_dir)}
    dest = backup_dir or (jit_dir / "hyperloom_jit_backup")
    removed = _unlink_serving_modules(jit_dir, modules_needed_t)
    invalidation = _invalidate_jit_build(jit_dir, dest)
    log.info(
        "aiter serving so does not cover tuned CSV; unlinked %d module(s) jit_build=%s",
        len(removed),
        invalidation.get("status"),
    )
    return {
        "action": "invalidate",
        "jit_dir": str(jit_dir),
        "removed": removed,
        "jit_build": invalidation,
    }


def drop_serving_so_for_envs(
    envs: dict[str, str] | None = None,
    *,
    backup_dir: Path | None = None,
) -> dict[str, Any]:
    """Unlink serving GEMM .so files and move ``jit/build`` so a later start rebuilds."""
    jit_dir = _resolve_serving_jit_dir()
    if jit_dir is None:
        return {"action": "noop", "reason": "aiter jit dir not found"}
    modules = _modules_for_envs(envs)
    dest = backup_dir or (jit_dir / "hyperloom_jit_backup")
    removed = _unlink_serving_modules(jit_dir, modules)
    invalidation = _invalidate_jit_build(jit_dir, dest)
    return {
        "action": "invalidate",
        "jit_dir": str(jit_dir),
        "removed": removed,
        "jit_build": invalidation,
    }


def result_is_aiter_jit_registry_mismatch(result: dict[str, Any] | None) -> bool:
    """True when an integrate/baseline result is a compiled-registry miss."""
    if not isinstance(result, dict):
        return False
    if str(result.get("error_class") or "") == "aiter_jit_registry_mismatch":
        return True
    return is_aiter_jit_registry_mismatch(str(result.get("error") or ""))


__all__ = [
    "AITER_CPP_BUILD_PROBE_PATHS",
    "AITER_ENV_TO_SERVING_MODULES",
    "AITER_JIT_PROBE_PATHS",
    "AITER_LOCK_STALE_MINUTES",
    "BASELINE_COLD_START_TIMEOUT_SEC",
    "COLD_START_KERNEL_THRESHOLD",
    "COMPILER_PROCESS_NAMES",
    "COMPILED_REGISTRY_MARKER",
    "clean_stale_aiter_locks",
    "csv_kernel_names",
    "drop_serving_so_for_envs",
    "find_aiter_baton_wait",
    "is_aiter_jit_registry_mismatch",
    "prepare_serving_so_for_csvs",
    "probe_aiter_jit_cache",
    "result_is_aiter_jit_registry_mismatch",
    "serving_modules_cover_csv",
    "sweep_stale_aiter_locks_if_dead",
    "_any_live_compiler",
    "_resolve_aiter_jit_dir_dynamic",
    "_resolve_lock_sweep_dirs",
]
