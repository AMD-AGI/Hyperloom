"""Process-level monitor — detects server/benchmark process anomalies."""

from __future__ import annotations

import logging
import time
from typing import Optional

from ..config import Config
from ..models import Alert, ProcessInfo, Severity
from ..providers.base import MetricsProvider

log = logging.getLogger(__name__)


class ProcessMonitor:
    """Track server and benchmark processes, detect zombies and stalls."""

    def __init__(self, config: Config, provider: MetricsProvider):
        self._config = config
        self._provider = provider
        self._server_seen_at: dict[str, float] = {}
        self._benchmark_started_at: Optional[float] = None

    async def check(self) -> list[Alert]:
        alerts: list[Alert] = []
        processes = await self._provider.get_process_list()

        server_alive = self._check_server_processes(processes, alerts)
        self._check_benchmark_processes(processes, alerts)
        self._check_zombie_processes(processes, alerts)

        if not server_alive:
            for pattern in self._config.server_process_patterns:
                if pattern in self._server_seen_at:
                    elapsed = time.time() - self._server_seen_at[pattern]
                    if elapsed < 120:
                        alerts.append(Alert(
                            check_name="server_disappeared",
                            severity=Severity.CRITICAL,
                            summary=f"Server process '{pattern}' disappeared "
                                    f"(last seen {elapsed:.0f}s ago)",
                            evidence={"pattern": pattern, "elapsed_s": elapsed},
                            timestamp=time.time(),
                        ))
        return alerts

    def notify_benchmark_started(self) -> None:
        self._benchmark_started_at = time.time()

    def notify_benchmark_ended(self) -> None:
        self._benchmark_started_at = None

    def _check_server_processes(
        self, processes: list[ProcessInfo], alerts: list[Alert],
    ) -> bool:
        found = False
        for pattern in self._config.server_process_patterns:
            matching = [p for p in processes if pattern in p.cmd]
            if matching:
                found = True
                self._server_seen_at[pattern] = time.time()
                for p in matching:
                    if p.state.startswith("Z"):
                        alerts.append(Alert(
                            check_name="server_zombie",
                            severity=Severity.CRITICAL,
                            summary=f"Server process is zombie: pid={p.pid} cmd={p.cmd[:80]}",
                            evidence={"pid": p.pid, "pattern": pattern},
                            timestamp=time.time(),
                        ))
        return found

    def _check_benchmark_processes(
        self, processes: list[ProcessInfo], alerts: list[Alert],
    ) -> None:
        if self._benchmark_started_at is None:
            return
        elapsed = time.time() - self._benchmark_started_at
        if elapsed < self._config.benchmark_timeout_s:
            return
        bench_procs = []
        for pattern in self._config.benchmark_process_patterns:
            bench_procs.extend(p for p in processes if pattern in p.cmd)
        if bench_procs:
            alerts.append(Alert(
                check_name="benchmark_timeout",
                severity=Severity.CRITICAL,
                summary=f"Benchmark still running after {elapsed:.0f}s "
                        f"(timeout={self._config.benchmark_timeout_s}s)",
                evidence={
                    "elapsed_s": elapsed,
                    "pids": [p.pid for p in bench_procs],
                },
                timestamp=time.time(),
            ))

    def _check_zombie_processes(
        self, processes: list[ProcessInfo], alerts: list[Alert],
    ) -> None:
        zombies = [p for p in processes if p.state.startswith("Z")]
        if len(zombies) > 5:
            alerts.append(Alert(
                check_name="excessive_zombies",
                severity=Severity.WARNING,
                summary=f"{len(zombies)} zombie processes detected",
                evidence={"count": len(zombies), "sample_pids": [z.pid for z in zombies[:5]]},
                timestamp=time.time(),
            ))
