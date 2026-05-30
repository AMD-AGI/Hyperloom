"""File-locked GPU pool for local execution mode.

Manages GPU allocation on a single machine. Automatically detects
which GPUs are in use by the serving process and reserves the rest
for optimization agents.

Interface:
  pool.acquire(count, holder) -> list[int] | None
  pool.release(gpu_ids)       -> None
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_LOCK_DIR = "/tmp/hyperloom_gpu_pool"
STALE_TIMEOUT_S = 600  # 10 minutes without heartbeat = stale


class GPUPool:
    """File-locked GPU allocation pool.

    Each GPU is represented by a lock file. Allocation tracking is
    via simple JSON files — atomic enough for single-node use.
    """

    def __init__(
        self,
        total_gpus: list[int],
        reserved: list[int] | None = None,
        lock_dir: str = DEFAULT_LOCK_DIR,
    ):
        self._total = total_gpus
        self._reserved = set(reserved or [])
        self._available_set = [g for g in total_gpus if g not in self._reserved]
        self._lock_dir = Path(lock_dir)
        self._lock_dir.mkdir(parents=True, exist_ok=True)

    @property
    def total_gpus(self) -> list[int]:
        return list(self._total)

    @property
    def available_gpus(self) -> list[int]:
        """GPUs not reserved and not currently allocated."""
        allocated = self._get_allocated_gpu_ids()
        return [g for g in self._available_set if g not in allocated]

    @property
    def num_available(self) -> int:
        return len(self.available_gpus)

    def acquire(self, count: int, holder: str = "unknown") -> list[int] | None:
        """Try to acquire `count` GPUs. Returns list of GPU IDs or None."""
        self.cleanup_stale()
        available = self.available_gpus
        if len(available) < count:
            return None

        selected = available[:count]
        now = time.time()
        for gpu_id in selected:
            self._write_lock(gpu_id, holder, now)

        log.info("Allocated GPUs %s to %s", selected, holder)
        return selected

    def release(self, gpu_ids: list[int]) -> None:
        """Release specific GPU IDs."""
        for gpu_id in gpu_ids:
            lock_file = self._lock_dir / f"gpu_{gpu_id}.lock"
            if lock_file.exists():
                try:
                    lock_file.unlink()
                except OSError:
                    pass
        log.info("Released GPUs %s", gpu_ids)

    def release_by_holder(self, holder: str) -> list[int]:
        """Release all GPUs held by a specific holder. Returns freed IDs."""
        freed = []
        for gpu_id in self._total:
            lock_file = self._lock_dir / f"gpu_{gpu_id}.lock"
            if lock_file.exists():
                try:
                    data = json.loads(lock_file.read_text())
                    if data.get("holder") == holder:
                        lock_file.unlink()
                        freed.append(gpu_id)
                except (json.JSONDecodeError, OSError):
                    pass
        if freed:
            log.info("Released GPUs %s from holder %s", freed, holder)
        return freed

    def heartbeat(self, gpu_ids: list[int], holder: str) -> None:
        """Update heartbeat timestamp for held GPUs."""
        now = time.time()
        for gpu_id in gpu_ids:
            lock_file = self._lock_dir / f"gpu_{gpu_id}.lock"
            if lock_file.exists():
                try:
                    data = json.loads(lock_file.read_text())
                    if data.get("holder") == holder:
                        data["heartbeat"] = now
                        lock_file.write_text(json.dumps(data))
                except (json.JSONDecodeError, OSError):
                    pass

    def cleanup_stale(self) -> list[int]:
        """Remove allocations that haven't heartbeated in STALE_TIMEOUT_S."""
        freed = []
        now = time.time()
        for gpu_id in self._total:
            lock_file = self._lock_dir / f"gpu_{gpu_id}.lock"
            if lock_file.exists():
                try:
                    data = json.loads(lock_file.read_text())
                    if now - data.get("heartbeat", 0) > STALE_TIMEOUT_S:
                        lock_file.unlink()
                        freed.append(gpu_id)
                except (json.JSONDecodeError, OSError):
                    lock_file.unlink()
                    freed.append(gpu_id)
        if freed:
            log.info("Cleaned up stale GPU allocations: %s", freed)
        return freed

    def status(self) -> dict[str, Any]:
        """Return current pool status."""
        allocated: dict[int, dict] = {}
        for gpu_id in self._total:
            lock_file = self._lock_dir / f"gpu_{gpu_id}.lock"
            if lock_file.exists():
                try:
                    allocated[gpu_id] = json.loads(lock_file.read_text())
                except (json.JSONDecodeError, OSError):
                    pass
        return {
            "total": self._total,
            "reserved": sorted(self._reserved),
            "available": self.available_gpus,
            "allocated": allocated,
        }

    def _get_allocated_gpu_ids(self) -> set[int]:
        allocated = set()
        for gpu_id in self._total:
            lock_file = self._lock_dir / f"gpu_{gpu_id}.lock"
            if lock_file.exists():
                allocated.add(gpu_id)
        return allocated

    def _write_lock(self, gpu_id: int, holder: str, timestamp: float) -> None:
        lock_file = self._lock_dir / f"gpu_{gpu_id}.lock"
        data = {
            "gpu_id": gpu_id,
            "holder": holder,
            "allocated_at": timestamp,
            "heartbeat": timestamp,
        }
        lock_file.write_text(json.dumps(data))


# ─── Factory functions ─────────────────────────────────────────────────────────


def detect_serving_gpus() -> list[int]:
    """Detect GPUs currently used by a serving process (vLLM, SGLang, etc.)."""
    serving_gpus: set[int] = set()
    for var in ("SERVING_GPUS", "ROCR_VISIBLE_DEVICES"):
        val = os.environ.get(var, "")
        if val:
            try:
                serving_gpus = {int(x.strip()) for x in val.split(",") if x.strip()}
                break
            except ValueError:
                pass
    return sorted(serving_gpus)


def auto_pool(
    session_dir: str | None = None,
    total_gpus: int = 8,
    reserved: list[int] | None = None,
) -> GPUPool:
    """Create a GPUPool with auto-detected serving GPU reservation.

    If session_dir is provided, lock files go there for session isolation.
    Otherwise uses a global /tmp directory.
    """
    all_gpus = list(range(total_gpus))
    if reserved is None:
        reserved = detect_serving_gpus()

    lock_dir = DEFAULT_LOCK_DIR
    if session_dir:
        lock_dir = str(Path(session_dir) / ".gpu_pool")

    return GPUPool(total_gpus=all_gpus, reserved=reserved, lock_dir=lock_dir)
