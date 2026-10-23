# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GpuBackend that delegates to the existing GPU-facing executors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class RealGpuBackend:
    """Production path: unchanged launch / grid / accuracy behaviour."""

    name = "real"

    def launch(self, cmd: Sequence[str], **kwargs: Any) -> Any:
        from hyperloom.orchestrator.actions.executors._subprocess_kill import run_with_session_kill

        return run_with_session_kill(list(cmd), **kwargs)

    async def run_grid(self, **kwargs: Any) -> list[Any]:
        from hyperloom.orchestrator.actions.executors import _grid_runner

        return await _grid_runner.run_grid(**kwargs)

    def parse_eval_results(
        self,
        workspace: Path | str,
        framework: str | None = None,
        benchmark_mode: str = "",
    ) -> Mapping[str, Any]:
        from hyperloom.orchestrator.actions.executors._accuracy_gate import parse_eval_results

        return parse_eval_results(workspace, framework=framework, benchmark_mode=benchmark_mode)

    async def apply_patch(self, **kwargs: Any) -> Mapping[str, Any]:
        raise NotImplementedError("RealGpuBackend.apply_patch: call IntegratePatchExecutor directly")

    async def kernel_job(self, **kwargs: Any) -> Mapping[str, Any]:
        raise NotImplementedError("RealGpuBackend.kernel_job: call kernel_agent executor directly")

    async def profile(self, **kwargs: Any) -> Mapping[str, Any]:
        raise NotImplementedError("RealGpuBackend.profile: call profile / roofline executors directly")
