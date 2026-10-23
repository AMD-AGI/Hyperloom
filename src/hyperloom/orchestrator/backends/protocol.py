# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Process-local GPU measurement surface: real cards or a replay/oracle answerer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: How a GPU executor fails on a host without a ROCm device (matches eval harness).
NO_GPU_ERROR = "HIP error: no ROCm-capable device is detected"


@runtime_checkable
class GpuBackend(Protocol):
    """Launch, bench, patch, kernel, and profile work that needs a GPU or a replay of one."""

    name: str

    def launch(self, cmd: Sequence[str], **kwargs: Any) -> Any:
        """Run a server or tool process (``run_with_session_kill``'s keywords); Replay refuses."""

    async def run_grid(self, **kwargs: Any) -> list[Any]:
        """Bench a grid of variants; Real runs servers, Replay asks the oracle per variant."""

    def parse_eval_results(
        self,
        workspace: Path | str,
        framework: str | None = None,
        benchmark_mode: str = "",
    ) -> Mapping[str, Any]:
        """Read accuracy (or AgentX error-rate) from a finished workspace."""

    async def apply_patch(self, **kwargs: Any) -> Mapping[str, Any]:
        """Apply a framework/source patch that needs a GPU to validate."""

    async def kernel_job(self, **kwargs: Any) -> Mapping[str, Any]:
        """Run a GEAK / kernel-agent job."""

    async def profile(self, **kwargs: Any) -> Mapping[str, Any]:
        """Collect a GPU profile / roofline sample."""
