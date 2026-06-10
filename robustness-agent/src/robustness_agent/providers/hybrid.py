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
        """Initialise the hybrid provider with both backends.

        Args:
            robust (RobustProvider): Preferred backend for GPU/fault metrics.
            local (LocalProvider): Fallback and source for process/disk data.
        """
        self._robust = robust
        self._local = local
        self._robust_available = False

    async def probe(self) -> None:
        """Detect whether the Robust backend is reachable and cache the result."""
        self._robust_available = await self._robust.check_available()
        if self._robust_available:
            log.info("Primus-Robust-Internal is available at %s", self._robust.base_url)
        else:
            log.warning("Primus-Robust-Internal not reachable, using local fallback")

    async def get_gpu_metrics(self, gpu_id: Optional[int] = None) -> list[GpuSnapshot]:
        """Return GPU snapshots, preferring Robust and falling back to Local.

        Args:
            gpu_id (Optional[int]): If given, restrict the result to this GPU.

        Returns:
            list[GpuSnapshot]: GPU snapshots from the first backend that yields
            data.
        """
        if self._robust_available:
            result = await self._robust.get_gpu_metrics(gpu_id)
            if result:
                return result
        return await self._local.get_gpu_metrics(gpu_id)

    async def get_gpu_history(self, gpu_id: int, window_seconds: int) -> list[GpuSnapshot]:
        """Return GPU history, preferring Robust and falling back to Local.

        Args:
            gpu_id (int): The GPU identifier to query.
            window_seconds (int): Width of the trailing window, in seconds.

        Returns:
            list[GpuSnapshot]: Snapshots from the first backend that yields
            data.
        """
        if self._robust_available:
            result = await self._robust.get_gpu_history(gpu_id, window_seconds)
            if result:
                return result
        return await self._local.get_gpu_history(gpu_id, window_seconds)

    async def get_process_list(self) -> list[ProcessInfo]:
        """Return the process list (always from the Local backend).

        Returns:
            list[ProcessInfo]: The local host's process list.
        """
        return await self._local.get_process_list()

    async def get_disk_usage(self, path: str = "/") -> list[DiskSnapshot]:
        """Return disk usage (always from the Local backend).

        Args:
            path (str): Filesystem path to inspect.

        Returns:
            list[DiskSnapshot]: The local disk usage entries.
        """
        return await self._local.get_disk_usage(path)

    async def get_fault_events(self, since: float) -> list[FaultEvent]:
        """Return fault events from Robust when available, else empty.

        Args:
            since (float): Lower-bound Unix timestamp for returned faults.

        Returns:
            list[FaultEvent]: Robust fault events, or an empty list when the
            Robust backend is unavailable.
        """
        if self._robust_available:
            return await self._robust.get_fault_events(since)
        return []

    async def check_available(self) -> bool:
        """Report provider availability.

        Returns:
            bool: Always ``True``; the hybrid provider always has a fallback.
        """
        return True


async def create_provider(config: Config) -> MetricsProvider:
    """Factory: build the right provider based on auto-detected config.

    Args:
        config (Config): Runtime configuration; ``robust_analyzer_url``
            decides whether a Robust-backed hybrid provider is built.

    Returns:
        MetricsProvider: A :class:`LocalProvider` when no analyzer URL is
        configured, otherwise a probed :class:`HybridProvider`.
    """
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
