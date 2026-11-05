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


def gpu_vram_used(gpu_ids: list[int] | None = None) -> dict[int, int]:
    """GPU VRAM used in GB, keyed by GPU index."""
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
