# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Abstract base for metrics providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..models import DiskSnapshot, FaultEvent, GpuSnapshot, ProcessInfo


class MetricsProvider(ABC):
    """Unified interface for infrastructure metrics.

    Concrete implementations:
    - LocalProvider: shell commands (rocm-smi, ps, df, etc.)
    - RobustProvider: Primus-Robust-Internal analyzer REST API
    """

    @abstractmethod
    async def get_gpu_metrics(self, gpu_id: Optional[int] = None) -> list[GpuSnapshot]:
        """Return current GPU snapshots.

        Args:
            gpu_id (Optional[int]): If given, restrict the result to this GPU.

        Returns:
            list[GpuSnapshot]: The current GPU snapshots.
        """
        ...

    @abstractmethod
    async def get_gpu_history(self, gpu_id: int, window_seconds: int) -> list[GpuSnapshot]:
        """Return GPU snapshots over a trailing time window.

        Args:
            gpu_id (int): The GPU identifier to query.
            window_seconds (int): Width of the trailing window, in seconds.

        Returns:
            list[GpuSnapshot]: Snapshots within the window.
        """
        ...

    @abstractmethod
    async def get_process_list(self) -> list[ProcessInfo]:
        """Return the current process list.

        Returns:
            list[ProcessInfo]: One entry per tracked process.
        """
        ...

    @abstractmethod
    async def get_disk_usage(self, path: str = "/") -> list[DiskSnapshot]:
        """Return disk usage for a path.

        Args:
            path (str): Filesystem path to inspect.

        Returns:
            list[DiskSnapshot]: One entry per relevant mount.
        """
        ...

    @abstractmethod
    async def get_fault_events(self, since: float) -> list[FaultEvent]:
        """Return fault events newer than a timestamp.

        Args:
            since (float): Lower-bound Unix timestamp for returned faults.

        Returns:
            list[FaultEvent]: The matching fault events.
        """
        ...

    @abstractmethod
    async def check_available(self) -> bool:
        """Return True if this provider is operational."""
        ...
