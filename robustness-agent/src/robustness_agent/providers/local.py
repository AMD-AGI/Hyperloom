"""Local metrics provider — collects via shell commands on the sandbox host.

Used when Primus-Robust-Internal is not available (single-machine / dev mode).
GPU metrics are stored in a ring buffer for short-term trend analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from typing import Optional

from ..models import DiskSnapshot, FaultEvent, GpuSnapshot, ProcessInfo
from .base import MetricsProvider

log = logging.getLogger(__name__)


class _RingBuffer:
    """In-memory time-series buffer for local GPU metrics."""

    def __init__(self, max_age_seconds: int = 3600):
        self.max_age = max_age_seconds
        self._data: dict[int, deque[tuple[float, GpuSnapshot]]] = defaultdict(deque)

    def push(self, snapshot: GpuSnapshot) -> None:
        buf = self._data[snapshot.gpu_id]
        buf.append((snapshot.timestamp, snapshot))
        cutoff = snapshot.timestamp - self.max_age
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def get(self, gpu_id: int, window_seconds: int) -> list[GpuSnapshot]:
        if not self._data.get(gpu_id):
            return []
        latest_ts = self._data[gpu_id][-1][0]
        cutoff = latest_ts - window_seconds
        return [s for ts, s in self._data[gpu_id] if ts >= cutoff]


async def _run_cmd(cmd: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, stdout.decode(errors="replace")
    except asyncio.TimeoutError:
        log.warning("Command timed out: %s", cmd)
        return -1, ""
    except Exception as exc:
        log.warning("Command failed: %s — %s", cmd, exc)
        return -1, ""


class LocalProvider(MetricsProvider):
    """Collect metrics locally via rocm-smi / ps / df."""

    def __init__(self, history_seconds: int = 3600):
        self._ring = _RingBuffer(max_age_seconds=history_seconds)

    async def get_gpu_metrics(self, gpu_id: Optional[int] = None) -> list[GpuSnapshot]:
        snapshots = await self._collect_rocm_smi()
        if not snapshots:
            snapshots = await self._collect_nvidia_smi()
        now = time.time()
        for s in snapshots:
            s.timestamp = now
            self._ring.push(s)
        if gpu_id is not None:
            return [s for s in snapshots if s.gpu_id == gpu_id]
        return snapshots

    async def get_gpu_history(self, gpu_id: int, window_seconds: int) -> list[GpuSnapshot]:
        return self._ring.get(gpu_id, window_seconds)

    async def get_process_list(self) -> list[ProcessInfo]:
        code, output = await _run_cmd("ps aux --no-headers")
        if code != 0:
            return []
        results: list[ProcessInfo] = []
        for line in output.strip().splitlines():
            parts = line.split(None, 10)
            if len(parts) < 11:
                continue
            try:
                pid = int(parts[1])
                rss_kb = float(parts[5]) if parts[5].replace(".", "").isdigit() else 0
                state = parts[7]
                cmd = parts[10]
                results.append(ProcessInfo(
                    pid=pid, state=state, cmd=cmd, rss_mb=rss_kb / 1024.0,
                ))
            except (ValueError, IndexError):
                continue
        return results

    async def get_disk_usage(self, path: str = "/") -> list[DiskSnapshot]:
        code, output = await _run_cmd(f"df -BG {path}")
        if code != 0:
            return []
        results: list[DiskSnapshot] = []
        for line in output.strip().splitlines()[1:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                total = float(parts[1].rstrip("G"))
                used = float(parts[2].rstrip("G"))
                avail = float(parts[3].rstrip("G"))
                mount = parts[5]
                results.append(DiskSnapshot(
                    mount=mount, total_gb=total, used_gb=used, available_gb=avail,
                ))
            except (ValueError, IndexError):
                continue
        return results

    async def get_fault_events(self, since: float) -> list[FaultEvent]:
        return []

    async def check_available(self) -> bool:
        return True

    # -- internal collectors --

    async def _collect_rocm_smi(self) -> list[GpuSnapshot]:
        code, output = await _run_cmd(
            "rocm-smi --showuse --showmeminfo vram --showtemp --showpower --json",
        )
        if code != 0 or not output.strip():
            return []
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return []
        snapshots: list[GpuSnapshot] = []
        for key, info in data.items():
            if not key.startswith("card"):
                continue
            try:
                gpu_id = int(key.replace("card", ""))
                util = float(info.get("GPU use (%)", info.get("GPU Usage (%)", 0)))
                vram_used = float(info.get("VRAM Total Used Memory (B)", 0)) / (1024 * 1024)
                vram_total = float(info.get("VRAM Total Memory (B)", 0)) / (1024 * 1024)
                temp = float(info.get("Temperature (Sensor junction) (C)",
                                      info.get("Temperature (Sensor edge) (C)", 0)))
                power = float(info.get("Average Graphics Package Power (W)", 0))
                snapshots.append(GpuSnapshot(
                    gpu_id=gpu_id, utilization=util,
                    vram_used_mb=vram_used, vram_total_mb=vram_total,
                    temperature_c=temp, power_watts=power,
                ))
            except (ValueError, TypeError):
                continue
        return snapshots

    async def _collect_nvidia_smi(self) -> list[GpuSnapshot]:
        code, output = await _run_cmd(
            "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw --format=csv,noheader,nounits",
        )
        if code != 0 or not output.strip():
            return []
        snapshots: list[GpuSnapshot] = []
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                snapshots.append(GpuSnapshot(
                    gpu_id=int(parts[0]),
                    utilization=float(parts[1]),
                    vram_used_mb=float(parts[2]),
                    vram_total_mb=float(parts[3]),
                    temperature_c=float(parts[4]),
                    power_watts=float(parts[5]),
                ))
            except (ValueError, TypeError):
                continue
        return snapshots
