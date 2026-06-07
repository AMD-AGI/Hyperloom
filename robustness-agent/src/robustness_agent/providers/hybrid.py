# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Hybrid provider — Robust when available, fallback to Local."""

from __future__ import annotations

import logging
from typing import Optional

from ..config import Config
from ..models import DiskSnapshot, FaultEvent, GpuSnapshot, ProcessInfo
from .base import MetricsProvider
from .local import LocalProvider
from .robust import RobustProvider

log = logging.getLogger(__name__)


class HybridProvider(MetricsProvider):
    """Prefer RobustProvider for GPU/fault metrics, always use LocalProvider
    for process/disk monitoring (Robust does not track those)."""

    def __init__(self, robust: RobustProvider, local: LocalProvider):
        self._robust = robust
        self._local = local
        self._robust_available = False

    async def probe(self) -> None:
        self._robust_available = await self._robust.check_available()
        if self._robust_available:
            log.info("Primus-Robust-Internal is available at %s", self._robust.base_url)
        else:
            log.warning("Primus-Robust-Internal not reachable, using local fallback")

    async def get_gpu_metrics(self, gpu_id: Optional[int] = None) -> list[GpuSnapshot]:
        if self._robust_available:
            result = await self._robust.get_gpu_metrics(gpu_id)
            if result:
                return result
        return await self._local.get_gpu_metrics(gpu_id)

    async def get_gpu_history(self, gpu_id: int, window_seconds: int) -> list[GpuSnapshot]:
        if self._robust_available:
            result = await self._robust.get_gpu_history(gpu_id, window_seconds)
            if result:
                return result
        return await self._local.get_gpu_history(gpu_id, window_seconds)

    async def get_process_list(self) -> list[ProcessInfo]:
        return await self._local.get_process_list()

    async def get_disk_usage(self, path: str = "/") -> list[DiskSnapshot]:
        return await self._local.get_disk_usage(path)

    async def get_fault_events(self, since: float) -> list[FaultEvent]:
        if self._robust_available:
            return await self._robust.get_fault_events(since)
        return []

    async def check_available(self) -> bool:
        return True


async def create_provider(config: Config) -> MetricsProvider:
    """Factory: build the right provider based on auto-detected config."""
    local = LocalProvider(history_seconds=config.local_metrics_history_s)

    if not config.robust_analyzer_url:
        log.info("robust-analyzer not reachable, using LocalProvider only")
        return local

    robust = RobustProvider(
        analyzer_url=config.robust_analyzer_url,
        workload_uid="",
    )
    hybrid = HybridProvider(robust=robust, local=local)
    await hybrid.probe()
    return hybrid
