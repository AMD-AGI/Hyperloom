# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared aiter JIT cache helpers: dir resolution, cold/warm constants, and
stale-lock cleanup.

aiter JIT-compiles GPU kernels on demand with ninja, guarding each module
build with a per-module file lock (aiter ``jit/core.py`` ``mp_lock`` +
``jit/utils/file_baton.py``). The lock is a zero-byte file (``O_CREAT|O_EXCL``,
no pid inside) and ``FileBaton.wait()`` spins forever with no timeout. When a
``hipcc`` build process is killed mid-compile (timeout / OOM / KILL_TASK) it
never releases its lock, so every later process compiling that module spins
forever — the failure that hung 10 production sessions (``build_count`` frozen,
server unreachable, robustness escalated).

The fix sweeps these orphaned locks before each cold server start, but ONLY
when no compiler process is alive (the jit dir is node-global and shared across
concurrent benchmarks, so a live ``hipcc`` may legitimately hold a lock). ninja
resumes the build incrementally from existing ``.o`` once the lock is gone, so
we delete only locks — never build artifacts.

This module is the single home for the logic so both ``cli.py`` (startup sweep)
and ``baseline.py`` (per-cold-start sweep) can import it without a circular
dependency (``cli.py`` already imports ``baseline``).
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

# Legacy fallback probe order for aiter's JIT cache dir, used only when
# find_spec("aiter") can't resolve aiter dynamically. First existing path
# wins. Override via env `INFERENCE_OPTIMIZER_AITER_JIT_DIR` (tried first).
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

# Default mtime gate (minutes) for the lock sweep. Used as the belt-and-
# suspenders fallback when compiler liveness is UNKNOWN; the proven-dead path
# (no live compiler) bypasses it with stale_minutes=0. 5 min sits above a
# cold-start MoE module build's lock churn, below the hang-suspicion cliff.
AITER_LOCK_STALE_MINUTES = 5

# Process names that indicate an in-flight aiter/ninja compile. hipcc is a
# perl/bash wrapper, so its ``name`` can surface as ``perl``/``sh`` — we also
# match on the cmdline's first token (see ``_any_live_compiler``).
COMPILER_PROCESS_NAMES = frozenset({
    "hipcc",
    "hipcc.bin",
    "ninja",
    "cc1plus",
    "clang",
    "clang++",
    "clang-cpp",
})

# Lock file names left by aiter / ninja under the jit dir.
_LOCK_NAMES = {"lock", ".ninja_lock"}


def _resolve_aiter_jit_dir_dynamic() -> list[str]:
    """Locate aiter's ``jit/`` dir via Python's import machinery.

    Counting at ``<aiter>/jit/`` reflects a warm wheel install (~80
    pre-built ``.so``); the legacy fixed ``jit/build`` list mis-reports
    every wheel install as COLD. Returns an ordered candidate list
    (``jit`` preferred over ``jit/build``); empty if aiter not found.
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


def _any_live_compiler() -> bool | None:
    """Return True if any aiter/ninja compiler process is alive, else False.

    Used to decide whether an aiter JIT lock is dead (orphaned by a killed
    build) or held by a legitimate in-flight compile. The lock file carries no
    pid, so we scan the node for live compiler processes instead.

    Returns ``None`` when process enumeration itself fails (psutil missing or
    erroring) — callers MUST treat ``None`` as "unknown" and refuse to delete
    on the proven-dead fast path.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                info = proc.info
                name = (info.get("name") or "").strip()
                if name in COMPILER_PROCESS_NAMES:
                    return True
                cmdline = info.get("cmdline") or []
                if cmdline:
                    first = os.path.basename(str(cmdline[0]).strip())
                    if first in COMPILER_PROCESS_NAMES:
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                continue
    except Exception as exc:  # noqa: BLE001 — enumeration failed entirely
        log.warning("aiter_jit: compiler-liveness scan failed: %s", exc)
        return None
    return False


def _resolve_lock_sweep_dir(aiter_jit_dir: Path | None) -> Path | None:
    """Resolve the build dir to sweep for locks: arg → env → dynamic → legacy.

    Returns the first existing directory, or ``None`` when nothing resolves.
    Mirrors the historical resolution order so the swept dir matches what
    aiter actually writes locks into. A non-None caller arg is trusted as-is
    (the ``os.walk`` of a nonexistent path simply yields nothing).
    """
    if aiter_jit_dir is not None:
        return aiter_jit_dir

    candidates: list[str] = []
    override = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
    if override:
        override_path = Path(override)
        candidates.extend([str(override_path), str(override_path / "build")])
    try:
        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        aiter_root = Path(spec.origin).parent
        candidates.append(str(aiter_root / "jit" / "build"))
    candidates.extend([
        "/sgl-workspace/aiter/aiter/jit/build",
        "/usr/local/lib/python3.10/dist-packages/aiter/jit/build",
        "/usr/local/lib/python3.12/dist-packages/aiter/jit/build",
        "/opt/venv/lib/python3.10/site-packages/aiter/jit/build",
        "/opt/venv/lib/python3.12/site-packages/aiter/jit/build",
    ])
    for cand in candidates:
        p = Path(cand)
        if p.is_dir():
            return p
    return None


def clean_stale_aiter_locks(
    aiter_jit_dir: Path | None = None,
    stale_minutes: int = AITER_LOCK_STALE_MINUTES,
) -> dict[str, Any]:
    """Sweep aiter's JIT build dir for stale plain-file locks left by killed runs.

    Killed runs leave locks that block the next compile (aiter's untimed
    FileBaton wait). Only deletes locks with mtime older than ``stale_minutes``
    (default 5; above cold-start MoE build time, below the hang-suspicion
    cliff). Pass ``stale_minutes=0`` only when liveness has proven the locks
    are orphaned (see ``sweep_stale_aiter_locks_if_dead``). Build dir
    resolution: caller arg → $INFERENCE_OPTIMIZER_AITER_JIT_DIR → dynamic
    <aiter>/jit/build → legacy fallbacks. Returns a stats dict; never raises
    (errors counted).
    """
    stats: dict[str, Any] = {
        "dir": None,
        "scanned": 0,
        "deleted": 0,
        "skipped_fresh": 0,
        "errors": 0,
    }

    resolved = _resolve_lock_sweep_dir(aiter_jit_dir)
    if resolved is None:
        return stats
    aiter_jit_dir = resolved

    stats["dir"] = str(aiter_jit_dir)

    threshold_seconds = float(stale_minutes) * 60.0
    now = time.time()
    try:
        walker = os.walk(str(aiter_jit_dir))
    except OSError:
        stats["errors"] += 1
        return stats

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

    return stats


def sweep_stale_aiter_locks_if_dead(
    aiter_jit_dir: Path | None = None,
) -> dict[str, Any]:
    """Sweep orphaned aiter JIT locks, gated on no live compiler process.

    The jit dir is node-global and shared across concurrent benchmarks, so a
    live ``hipcc``/``ninja`` may legitimately hold a lock. We therefore only
    delete when liveness is confidently ``False``:

    * live compiler present (``True``) → skip entirely (don't disturb a real
      in-flight compile).
    * liveness unknown (``None``, psutil missing/errored) → fall back to the
      mtime-gated sweep so a genuinely ancient lock can still be reaped.
    * no live compiler (``False``) → every lock is orphaned; sweep with
      ``stale_minutes=0`` (mtime gate not needed — liveness already proved it).

    Returns the underlying stats dict augmented with ``compiler_alive`` and,
    on the skip path, ``skipped_live=True``.
    """
    alive = _any_live_compiler()
    if alive is True:
        return {
            "dir": None,
            "scanned": 0,
            "deleted": 0,
            "skipped_fresh": 0,
            "errors": 0,
            "compiler_alive": True,
            "skipped_live": True,
        }
    if alive is None:
        stats = clean_stale_aiter_locks(
            aiter_jit_dir, stale_minutes=AITER_LOCK_STALE_MINUTES,
        )
        stats["compiler_alive"] = None
        return stats
    stats = clean_stale_aiter_locks(aiter_jit_dir, stale_minutes=0)
    stats["compiler_alive"] = False
    return stats


__all__ = [
    "AITER_JIT_PROBE_PATHS",
    "AITER_LOCK_STALE_MINUTES",
    "COLD_START_KERNEL_THRESHOLD",
    "COMPILER_PROCESS_NAMES",
    "clean_stale_aiter_locks",
    "sweep_stale_aiter_locks_if_dead",
    "_any_live_compiler",
    "_resolve_aiter_jit_dir_dynamic",
]
