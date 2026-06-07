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
        ...

    @abstractmethod
    async def get_gpu_history(self, gpu_id: int, window_seconds: int) -> list[GpuSnapshot]:
        ...

    @abstractmethod
    async def get_process_list(self) -> list[ProcessInfo]:
        ...

    @abstractmethod
    async def get_disk_usage(self, path: str = "/") -> list[DiskSnapshot]:
        ...

    @abstractmethod
    async def get_fault_events(self, since: float) -> list[FaultEvent]:
        ...

    @abstractmethod
    async def check_available(self) -> bool:
        """Return True if this provider is operational."""
        ...
