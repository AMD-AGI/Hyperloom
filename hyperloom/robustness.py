"""Robustness — health monitoring, hang detection, crash recovery.

Monitors running agents and GPU tasks for:
  - Process hangs (no heartbeat)
  - Server crashes
  - OOM kills
  - Unexpected throughput drops
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    HUNG = "hung"
    CRASHED = "crashed"
    OOM = "oom"


@dataclass
class HealthCheck:
    """Result of a health check."""

    status: HealthStatus
    message: str = ""
    action: str = ""  # "", "restart", "kill", "skip"
    details: dict[str, Any] | None = None


def check_process_health(pid: int) -> HealthCheck:
    """Check if a process is still alive and responsive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return HealthCheck(status=HealthStatus.CRASHED, message=f"PID {pid} not found")
    except PermissionError:
        return HealthCheck(status=HealthStatus.HEALTHY)

    return HealthCheck(status=HealthStatus.HEALTHY)


def check_server_health(url: str, timeout: int = 10) -> HealthCheck:
    """Check if an HTTP server is responding."""
    import urllib.request
    import urllib.error

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                return HealthCheck(status=HealthStatus.HEALTHY)
            return HealthCheck(
                status=HealthStatus.DEGRADED,
                message=f"Server returned {resp.status}",
            )
    except urllib.error.URLError as e:
        return HealthCheck(
            status=HealthStatus.CRASHED,
            message=f"Server unreachable: {e}",
            action="restart",
        )
    except Exception as e:
        return HealthCheck(
            status=HealthStatus.DEGRADED,
            message=str(e),
        )


def check_heartbeat(heartbeat_path: str, stale_timeout_s: float = 300) -> HealthCheck:
    """Check if a heartbeat file is fresh."""
    path = Path(heartbeat_path)
    if not path.exists():
        return HealthCheck(
            status=HealthStatus.DEGRADED,
            message="No heartbeat file found",
        )

    try:
        mtime = path.stat().st_mtime
        age = time.time() - mtime
        if age > stale_timeout_s:
            return HealthCheck(
                status=HealthStatus.HUNG,
                message=f"Heartbeat stale ({age:.0f}s > {stale_timeout_s}s)",
                action="kill",
            )
        return HealthCheck(status=HealthStatus.HEALTHY)
    except OSError as e:
        return HealthCheck(status=HealthStatus.DEGRADED, message=str(e))


def check_gpu_oom() -> HealthCheck:
    """Check dmesg/syslog for recent GPU OOM events."""
    try:
        result = subprocess.run(
            ["dmesg", "--since", "5 minutes ago"],
            capture_output=True, text=True, timeout=5,
        )
        if "Out of memory" in result.stdout or "oom" in result.stdout.lower():
            return HealthCheck(
                status=HealthStatus.OOM,
                message="GPU OOM detected in dmesg",
                action="restart",
            )
    except (subprocess.TimeoutExpired, OSError):
        pass
    return HealthCheck(status=HealthStatus.HEALTHY)


def kill_process_tree(pid: int) -> bool:
    """Kill a process and all its children."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(5)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        return True
    except (ProcessLookupError, PermissionError):
        return False


def cleanup_gpu_processes(
    parent_pid: int | None = None,
    grace_sec: float = 5.0,
) -> list[int]:
    """Kill orphaned GPU processes by discovering actual PIDs on the device.

    Strategy:
      1. Query GPU compute processes via nvidia-smi (works for CUDA and ROCm).
      2. If parent_pid is given, also collect its entire child tree.
      3. Send SIGTERM, wait grace_sec, then SIGKILL survivors.

    Returns list of PIDs that were killed.
    """
    pids_to_kill: set[int] = set()

    gpu_pids = _get_gpu_pids()
    pids_to_kill.update(gpu_pids)

    if parent_pid:
        children = _get_descendant_pids(parent_pid)
        pids_to_kill.update(children)

    my_pid = os.getpid()
    pids_to_kill.discard(my_pid)
    pids_to_kill.discard(1)

    killed: list[int] = []
    for pid in pids_to_kill:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except (ProcessLookupError, PermissionError):
            pass

    if killed and grace_sec > 0:
        time.sleep(grace_sec)

    for pid in killed:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    return killed


def _get_gpu_pids() -> set[int]:
    """Get PIDs of all processes currently using GPU compute."""
    pids: set[int] = set()

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.add(int(line))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not pids:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpids"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = line.split()
                    for part in parts:
                        if part.isdigit() and int(part) > 1:
                            pids.add(int(part))
                            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return pids


def _get_descendant_pids(parent: int) -> set[int]:
    """Get all descendant PIDs of a process using /proc."""
    descendants: set[int] = set()
    try:
        result = subprocess.run(
            ["ps", "--ppid", str(parent), "-o", "pid=", "--no-headers"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    child = int(line)
                    descendants.add(child)
                    descendants.update(_get_descendant_pids(child))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return descendants
