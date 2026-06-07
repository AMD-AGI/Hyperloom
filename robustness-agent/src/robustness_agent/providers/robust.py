# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Primus-Robust-Internal provider — queries robust-analyzer REST API.

Used in cluster deployments where Primus-Robust-Internal is available.
Provides GPU/RDMA/fault metrics with 5s granularity and 30-day history
from VictoriaMetrics, plus fault events from PostgreSQL.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from ..models import DiskSnapshot, FaultEvent, GpuSnapshot, ProcessInfo
from .base import MetricsProvider

log = logging.getLogger(__name__)


class RobustProvider(MetricsProvider):
    """Query Primus-Robust-Internal robust-analyzer for metrics."""

    def __init__(self, analyzer_url: str, workload_uid: str = ""):
        self.base_url = analyzer_url.rstrip("/")
        self.workload_uid = workload_uid
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(15.0),
        )

    async def get_gpu_metrics(self, gpu_id: Optional[int] = None) -> list[GpuSnapshot]:
        query = f'workload_gpu_utilization{{workload_uid="{self.workload_uid}"}}'
        try:
            result = await self._promql_query(query)
            snapshots = self._parse_gpu_instant(result)
            if gpu_id is not None:
                return [s for s in snapshots if s.gpu_id == gpu_id]
            return snapshots
        except Exception as exc:
            log.warning("Failed to query GPU metrics from Robust: %s", exc)
            return []

    async def get_gpu_history(self, gpu_id: int, window_seconds: int) -> list[GpuSnapshot]:
        query = f'workload_gpu_utilization{{workload_uid="{self.workload_uid}",gpu_id="{gpu_id}"}}'
        try:
            now = time.time()
            result = await self._promql_range(query, now - window_seconds, now, step="5s")
            return self._parse_gpu_range(result, gpu_id)
        except Exception as exc:
            log.warning("Failed to query GPU history from Robust: %s", exc)
            return []

    async def get_process_list(self) -> list[ProcessInfo]:
        # Robust does not track application-level processes; always empty.
        return []

    async def get_disk_usage(self, path: str = "/") -> list[DiskSnapshot]:
        # Robust does not track disk usage; always empty.
        return []

    async def get_fault_events(self, since: float) -> list[FaultEvent]:
        try:
            resp = await self._client.get("/api/v1/faults", params={"since": int(since)})
            resp.raise_for_status()
            data = resp.json()
            return self._parse_faults(data)
        except Exception as exc:
            log.warning("Failed to query fault events from Robust: %s", exc)
            return []

    async def check_available(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=httpx.Timeout(3.0))
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()

    # -- PromQL helpers --

    async def _promql_query(self, query: str) -> dict[str, Any]:
        resp = await self._client.get(
            "/api/v1/query",
            params={"query": query},
        )
        resp.raise_for_status()
        return resp.json()

    async def _promql_range(
        self, query: str, start: float, end: float, step: str = "5s",
    ) -> dict[str, Any]:
        resp = await self._client.get(
            "/api/v1/query_range",
            params={"query": query, "start": start, "end": end, "step": step},
        )
        resp.raise_for_status()
        return resp.json()

    # -- parsers --

    def _parse_gpu_instant(self, data: dict[str, Any]) -> list[GpuSnapshot]:
        snapshots: list[GpuSnapshot] = []
        results = data.get("data", {}).get("result", [])
        for r in results:
            metric = r.get("metric", {})
            value = r.get("value", [0, "0"])
            try:
                snapshots.append(GpuSnapshot(
                    gpu_id=int(metric.get("gpu_id", 0)),
                    utilization=float(value[1]),
                    vram_used_mb=0, vram_total_mb=0,
                    temperature_c=0, power_watts=0,
                    timestamp=float(value[0]),
                ))
            except (ValueError, TypeError, IndexError):
                continue
        return snapshots

    def _parse_gpu_range(self, data: dict[str, Any], gpu_id: int) -> list[GpuSnapshot]:
        snapshots: list[GpuSnapshot] = []
        results = data.get("data", {}).get("result", [])
        for r in results:
            for ts, val in r.get("values", []):
                try:
                    snapshots.append(GpuSnapshot(
                        gpu_id=gpu_id,
                        utilization=float(val),
                        vram_used_mb=0, vram_total_mb=0,
                        temperature_c=0, power_watts=0,
                        timestamp=float(ts),
                    ))
                except (ValueError, TypeError):
                    continue
        return snapshots

    def _parse_faults(self, data: Any) -> list[FaultEvent]:
        if isinstance(data, dict):
            data = data.get("data", data.get("faults", []))
        if not isinstance(data, list):
            return []
        events: list[FaultEvent] = []
        for item in data:
            try:
                events.append(FaultEvent(
                    monitor_id=str(item.get("monitor_id", "")),
                    category=str(item.get("category", "")),
                    severity=str(item.get("severity", "")),
                    message=str(item.get("message", item.get("output", ""))),
                    timestamp=float(item.get("created_at", item.get("timestamp", 0))),
                    node=str(item.get("node_name", "")),
                ))
            except (ValueError, TypeError):
                continue
        return events
