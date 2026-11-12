"""Non-blocking watchers — poll a condition, tail logs, wake up when done.

Never block. Always stay active. Use these instead of sleep loops.

## Rules for cron watchers

A WatchGuard cron is Arbor infrastructure. It MUST NOT be deleted manually.
The cron runs until the condition is met, then self-terminates by writing
DONE to the guard file. The orchestrator reads the guard to know when to proceed.

Wrong:  CronDelete(id)              # never do this on a WatchGuard cron
Right:  WatchGuard.is_done(dir)     # check if the condition was met
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


# ── Status snapshot ──────────────────────────────────────────────────────────

def tail_log(log_path: str, container: str | None = None, n: int = 8) -> str:
    """Last N lines of a log file, from host or inside a container."""
    if container:
        r = subprocess.run(
            ["docker", "exec", container, "tail", f"-{n}", log_path],
            capture_output=True, text=True,
        )
        return r.stdout.strip() or "(empty)"
    p = Path(log_path)
    if not p.exists():
        return "(log not found)"
    lines = p.read_text().splitlines()
    return "\n".join(lines[-n:])


def check_http(url: str, timeout: int = 3) -> bool:
    """True if the URL responds 200."""
    r = subprocess.run(
        ["curl", "-sf", "--max-time", str(timeout), url],
        capture_output=True,
    )
    return r.returncode == 0


def _gpu_ids(value) -> list[int] | None:
    """``gpu_ids`` as a list of ints, from "0,1,2,3" or [0, 1, 2, 3] (a config may write either), as
    component.gpu_list reads it; None for none. A string reached the membership test below as it
    was, ``int in str`` would raise, and the server stop after a gate would crash the iteration."""
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple, set)) else str(value).split(",")
    ids = [int(g) for g in items if str(g).strip()]
    return ids or None


def _mask(env, *names: str) -> list[int] | None | bool:
    """The first of ``names`` set in ``env``, as a list of card numbers; None when none is set; False when it holds
    something else (a UUID), which no number maps to."""
    for name in names:
        value = str(env.get(name) or "").strip()
        if value:
            items = [x.strip() for x in value.split(",") if x.strip()]
            return [int(x) for x in items] if all(x.isdigit() for x in items) else False
    return None


def physical_gpu_ids(gpu_ids, env=None) -> list[int] | None:
    """``gpu_ids`` as rocm-smi numbers the node's cards (``GPU[n]``), under the device masks in ``env`` (this
    process's environment by default), which a host sets for the server; None when they cannot be mapped.

    rocm-smi lists every card of the node whatever the masks. A process sees the cards ROCR_VISIBLE_DEVICES leaves,
    numbered from 0, and HIP_VISIBLE_DEVICES (or CUDA_VISIBLE_DEVICES) picks among those. A host may give its lease
    either way: as the cards' own numbers (they are among the cards the masks leave, and stay as they are), or as the
    process's device numbers, from 0, which are mapped (Hyperloom's: ROCR_VISIBLE_DEVICES=2,3 with ids 0,1 is cards
    2 and 3). Read as card numbers, a second lease on a node would read the first's cards, busy with its bench,
    and book its candidates NOT MEASURED. Without masks the ids are the cards'."""
    ids = _gpu_ids(gpu_ids)
    if ids is None:
        return None
    import os
    env = os.environ if env is None else env
    outer = _mask(env, "ROCR_VISIBLE_DEVICES")
    inner = _mask(env, "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    if outer is None and inner is None:
        return ids
    if outer is False or inner is False:
        return None
    visible = outer if inner is None else (inner if outer is None else
                                           [outer[i] for i in inner if 0 <= i < len(outer)])
    if all(i in visible for i in ids):
        return ids
    if all(0 <= i < len(visible) for i in ids):
        return [visible[i] for i in ids]
    return None


def gpu_vram_used(gpu_ids: list[int] | str | None = None) -> dict[int, int]:
    """GPU VRAM used in GB, keyed by GPU index; ``gpu_ids`` in either form a config writes."""
    gpu_ids = _gpu_ids(gpu_ids)
    r = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True)
    used: dict[int, int] = {}
    current_gpu = None
    for line in r.stdout.splitlines():
        if "GPU[" in line and "Device Name" not in line:
            try:
                current_gpu = int(line.split("[")[1].split("]")[0])
            except Exception:
                pass
        if "Used Memory" in line and current_gpu is not None:
            if gpu_ids is None or current_gpu in gpu_ids:
                try:
                    used[current_gpu] = int(line.split(":")[-1].strip()) // (1024**3)
                except Exception:
                    pass
    return used


def report(
    health_url: str,
    log_path: str,
    container: str | None = None,
    gpu_ids: list[int] | None = None,
    log_lines: int = 8,
) -> str:
    """Single status snapshot — paste into a 60s cron watcher prompt."""
    ready = check_http(health_url)
    log = tail_log(log_path, container=container, n=log_lines)
    vram = gpu_vram_used(gpu_ids)
    vram_str = "  ".join(f"GPU{i}:{gb}GB" for i, gb in sorted(vram.items())) if vram else "N/A"

    lines = [
        f"{'✓ READY' if ready else '⧗ waiting'}  {health_url}",
        f"VRAM: {vram_str}",
        f"--- log tail ({log_path}) ---",
        log,
    ]
    return "\n".join(lines)


# ── WatchGuard — protected cron state ────────────────────────────────────────

class WatchGuard:
    """Tracks an active watcher cron. The cron MUST NOT be deleted externally.

    Lifecycle:
        guard = WatchGuard.start(session_dir, cron_id, label)
        # cron runs every minute, calls report(), writes DONE when ready
        if WatchGuard.is_done(session_dir):
            result = WatchGuard.result(session_dir)
    """

    GUARD_FILE = "watch_guard.json"
    DONE_FILE = "watch_done.json"

    @classmethod
    def start(cls, session_dir: str | Path, cron_id: str, label: str) -> None:
        """Record that a watcher cron is active. Call immediately after CronCreate."""
        p = Path(session_dir) / cls.GUARD_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "cron_id": cron_id,
            "label": label,
            "started_at": time.time(),
        }, indent=2))

    @classmethod
    def is_active(cls, session_dir: str | Path) -> bool:
        """True if a watcher cron is registered and not yet done."""
        guard = Path(session_dir) / cls.GUARD_FILE
        done = Path(session_dir) / cls.DONE_FILE
        return guard.exists() and not done.exists()

    @classmethod
    def is_done(cls, session_dir: str | Path) -> bool:
        """True if the watcher marked the condition as met."""
        return (Path(session_dir) / cls.DONE_FILE).exists()

    @classmethod
    def complete(cls, session_dir: str | Path, result: dict) -> None:
        """Called by the cron when the condition is met. Writes DONE and unregisters guard."""
        p = Path(session_dir)
        (p / cls.DONE_FILE).write_text(json.dumps({**result, "completed_at": time.time()}, indent=2))
        guard = p / cls.GUARD_FILE
        if guard.exists():
            guard.unlink()

    @classmethod
    def result(cls, session_dir: str | Path) -> dict:
        """Read the result written by the cron at completion."""
        p = Path(session_dir) / cls.DONE_FILE
        return json.loads(p.read_text()) if p.exists() else {}

    @classmethod
    def guard_info(cls, session_dir: str | Path) -> dict:
        """Return the guard metadata (cron_id, label, started_at)."""
        p = Path(session_dir) / cls.GUARD_FILE
        return json.loads(p.read_text()) if p.exists() else {}

    @classmethod
    def assert_no_active_guard(cls, session_dir: str | Path) -> None:
        """Raise if a watcher is still active. Call before proceeding past a wait point."""
        if cls.is_active(session_dir):
            info = cls.guard_info(session_dir)
            raise RuntimeError(
                f"WatchGuard active: '{info.get('label')}' (cron {info.get('cron_id')}) "
                f"is still running. Do NOT delete this cron. "
                f"Wait for WatchGuard.is_done() before proceeding."
            )
