# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared aiter build-cache helpers and stale-lock cleanup.

aiter has two independent runtime build trees:

* ``AITER_JIT_DIR`` / ``aiter/jit/build`` for ``compile_ops`` modules.
* ``AITER_ROOT_DIR/build`` (normally ``~/.aiter/build``) for ``cpp_itfs``
  template modules such as paged attention.

Both use zero-byte ``FileBaton`` locks with no owner pid and an unbounded
``wait()``. A build killed mid-compile therefore blocks every later process
that needs the same module.

This module resolves and sweeps both trees. Upstream lock files do not encode an
owner pid, so shared-cache locks are removed only after the conservative age
threshold and only when no compiler is alive. Forge-owned private caches carry
separate owner metadata and are cleaned by KernelForge. Build artifacts are
never deleted.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# < N .so files under aiter jit/ ⇒ COLD start (first-time JIT compile pending).
COLD_START_KERNEL_THRESHOLD = 20

# Fallback probe paths for aiter's JIT cache dir; first existing wins. Override
# via env `INFERENCE_OPTIMIZER_AITER_JIT_DIR` (tried first).
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

# Fallback locations for cpp_itfs template builds. ``AITER_ROOT_DIR`` and the
# current HOME are resolved dynamically before these paths.
AITER_CPP_BUILD_PROBE_PATHS: tuple[str, ...] = (
    "/root/.aiter/build",
)

# Mtime gate (minutes) for the lock sweep. Applied on every path, including the
# no-live-compiler one: a fresh lock's owner cannot be identified.
AITER_LOCK_STALE_MINUTES = 5

# Process names that indicate an in-flight aiter/ninja compile. hipcc is a
# wrapper whose ``name`` can surface as ``perl``/``sh``, so we also match on
# the cmdline's first token (see ``_any_live_compiler``).
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
    """Locate aiter's ``jit/`` dir via Python's import machinery.

    ``jit`` is preferred over ``jit/build``. Returns an ordered candidate list;
    empty if aiter not found.

    Returns:
        An ordered list of candidate aiter ``jit/`` directory paths, or an
        empty list when aiter cannot be located.
    """
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


def _any_live_compiler(
    build_dirs: list[Path] | None = None,
) -> bool | None:
    """Return True if a compiler associated with the build trees is alive.

    Used to decide whether an aiter JIT lock is dead (orphaned by a killed
    build) or held by a legitimate in-flight compile. When ``build_dirs`` is
    provided, a compiler counts only when its cwd or command line references
    one of those trees. This prevents an unrelated long-running ``hipcc`` on the
    same node from disabling cleanup forever.

    Returns ``None`` when process enumeration itself fails (psutil missing or
    erroring) — callers MUST treat ``None`` as "unknown" and refuse to delete
    on the proven-dead fast path.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        normalized_dirs = [
            str(path.resolve())
            for path in (build_dirs or [])
        ]
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
                    cwd == build_dir
                    or cwd.startswith(f"{build_dir}{os.sep}")
                    or build_dir in command
                    for build_dir in normalized_dirs
                ):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as exc:  # noqa: BLE001 — enumeration failed entirely
        log.warning("aiter_jit: compiler-liveness scan failed: %s", exc)
        return None
    return False


def _dedupe_existing_dirs(candidates: list[Path]) -> list[Path]:
    """Return existing candidate directories once, preserving priority."""
    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalized = candidate.expanduser().resolve()
        except OSError:
            normalized = candidate.expanduser().absolute()
        key = str(normalized)
        if key in seen or not normalized.is_dir():
            continue
        seen.add(key)
        resolved.append(normalized)
    return resolved


def _resolve_lock_sweep_dirs(aiter_jit_dir: Path | None) -> list[Path]:
    """Resolve every active aiter build tree that may contain baton locks.

    An explicit argument remains a single-directory test/diagnostic override.
    Automatic resolution covers both ``AITER_ROOT_DIR/build`` and
    ``AITER_JIT_DIR/build`` rather than stopping at the first JIT directory.
    """
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
        # Forge sets both variables for a private attempt. Treat the pair as an
        # authoritative namespace and never wander into shared fallback trees.
        return _dedupe_existing_dirs(candidates)
    override = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
    if override:
        override_path = Path(override)
        # Preserve the legacy explicit-override contract: callers use this
        # variable to constrain a diagnostic/test sweep to one tree.
        return _dedupe_existing_dirs([override_path / "build", override_path])
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
    return _dedupe_existing_dirs(candidates)


def _resolve_lock_sweep_dir(aiter_jit_dir: Path | None) -> Path | None:
    """Compatibility wrapper returning the first resolved build tree."""
    if aiter_jit_dir is None:
        override = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
        if override:
            override_path = Path(override)
            preferred = _dedupe_existing_dirs(
                [override_path / "build", override_path]
            )
            if preferred:
                return preferred[0]
    resolved = _resolve_lock_sweep_dirs(aiter_jit_dir)
    return resolved[0] if resolved else None


def clean_stale_aiter_locks(
    aiter_jit_dir: Path | None = None,
    stale_minutes: int = AITER_LOCK_STALE_MINUTES,
) -> dict[str, Any]:
    """Sweep aiter's JIT build trees for stale plain-file locks left by killed runs.

    Killed runs leave locks that block the next compile. Only deletes locks
    with mtime older than ``stale_minutes`` (default 5). Pass
    ``stale_minutes=0`` is reserved for a caller that independently owns the
    cache. Build tree resolution: caller arg (single-dir override) →
    ``AITER_ROOT_DIR``/build (else ``$HOME/.aiter/build`` plus
    ``AITER_CPP_BUILD_PROBE_PATHS``) together with ``AITER_JIT_DIR``, the pair
    being authoritative when both are set → $INFERENCE_OPTIMIZER_AITER_JIT_DIR
    → dynamic <aiter>/jit/build → legacy fallbacks. Every resolved tree is
    swept. Never raises (errors counted).

    Args:
        aiter_jit_dir (Path | None): Explicit JIT build dir; when ``None`` it
            is resolved from env / installed aiter / legacy fallbacks.
        stale_minutes (int): Minimum lock age in minutes before deletion
            (default 5).

    Returns:
        dict[str, Any]: A stats dict (``dir`` — primary tree, ``dirs`` — every
            swept tree, ``scanned``, ``deleted``, ``skipped_fresh``,
            ``errors``).
    """
    stats: dict[str, Any] = {
        "dir": None,
        "dirs": [],
        "scanned": 0,
        "deleted": 0,
        "skipped_fresh": 0,
        "errors": 0,
    }

    resolved_dirs = _resolve_lock_sweep_dirs(aiter_jit_dir)
    if not resolved_dirs:
        return stats

    primary = (
        _resolve_lock_sweep_dir(None)
        if aiter_jit_dir is None
        else resolved_dirs[0]
    )
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
    """Sweep orphaned aiter JIT locks, gated on no live compiler process.

    The build dirs may be node-global and shared across concurrent benchmarks,
    so a live ``hipcc``/``ninja`` may legitimately hold a lock. Upstream baton
    files have no owner pid, so process liveness alone cannot prove a fresh lock
    is orphaned. The policy is:

    * live compiler present (``True``) → skip entirely (don't disturb a real
      in-flight compile).
    * liveness unknown (``None``, psutil missing/errored) → fall back to the
      mtime-gated sweep so a genuinely ancient lock can still be reaped.
    * no live compiler (``False``) → remove only locks older than the stale-age
      threshold. Fresh locks remain protected because their owner is unknown.

    Returns the underlying stats dict augmented with ``compiler_alive`` and,
    on the skip path, ``skipped_live=True``.
    """
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
    """Find bounded log evidence of an aiter process blocked on a FileBaton.

    Only recent benchmark/server logs are inspected and only their tails are
    read, so classification remains cheap even when an integration workspace
    contains large logs from multiple attempts.
    """
    try:
        candidates = [
            path
            for path in search_root.rglob("*")
            if path.is_file() and path.name in _BATON_LOG_NAMES
        ]
    except OSError:
        return None
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    candidates.sort(key=_mtime, reverse=True)
    if since_unix is not None:
        candidates = [
            path for path in candidates if _mtime(path) >= since_unix
        ]
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


__all__ = [
    "AITER_CPP_BUILD_PROBE_PATHS",
    "AITER_JIT_PROBE_PATHS",
    "AITER_LOCK_STALE_MINUTES",
    "COLD_START_KERNEL_THRESHOLD",
    "COMPILER_PROCESS_NAMES",
    "clean_stale_aiter_locks",
    "find_aiter_baton_wait",
    "sweep_stale_aiter_locks_if_dead",
    "_any_live_compiler",
    "_resolve_aiter_jit_dir_dynamic",
    "_resolve_lock_sweep_dirs",
]
