# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU measurement backends (real hardware or oracle replay). Locked for RSI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .protocol import NO_GPU_ERROR, GpuBackend
from .real import RealGpuBackend
from .replay import ReplayGpuBackend, claim_oracle_fd, connect_oracle, oracle_fd_from_env

__all__ = [
    "NO_GPU_ERROR",
    "GpuBackend",
    "RealGpuBackend",
    "ReplayGpuBackend",
    "bind_backend_coordinator",
    "claim_oracle_fd",
    "connect_oracle",
    "ensure_gpu_backend",
    "get_gpu_backend",
    "install_gpu_backend",
    "install_gpu_backend_from_env",
    "oracle_fd_from_env",
    "parse_eval_results",
    "run_grid",
]

_backend: GpuBackend | None = None


def get_gpu_backend() -> GpuBackend:
    """The process backend; ``RealGpuBackend`` when none was installed."""
    global _backend
    if _backend is None:
        _backend = RealGpuBackend()
    return _backend


def install_gpu_backend(backend: GpuBackend) -> GpuBackend:
    """Replace the process backend."""
    global _backend
    _backend = backend
    return backend


def install_gpu_backend_from_env() -> GpuBackend:
    """Install from ``HYPERLOOM_GPU_BACKEND`` (``real`` | ``replay``) and the oracle variables."""
    kind = (os.environ.get("HYPERLOOM_GPU_BACKEND") or "real").strip().lower()
    if kind == "real":
        return install_gpu_backend(RealGpuBackend())
    if kind == "replay":
        scale = float(os.environ.get("HYPERLOOM_ORACLE_TIME_SCALE") or "1")
        return install_gpu_backend(ReplayGpuBackend(oracle_fd=oracle_fd_from_env(), time_scale=scale))
    raise ValueError(f"unknown HYPERLOOM_GPU_BACKEND={kind!r}; use real or replay")


def ensure_gpu_backend() -> GpuBackend:
    """The installed backend, else one installed from the environment.

    An oracle descriptor can be claimed once, so a backend an embedding process installed is kept.
    """
    if _backend is not None:
        return _backend
    return install_gpu_backend_from_env()


def bind_backend_coordinator(coordinator: Any) -> None:
    """Let a backend that answers from shared state (replay) read ``coordinator``'s."""
    bind = getattr(get_gpu_backend(), "bind_coordinator", None)
    if callable(bind):
        bind(coordinator)


async def run_grid(**kwargs: Any) -> list[Any]:
    """Bench a grid of variants on the backend installed when the grid runs (``_grid_runner.run_grid``'s keywords).

    The backend is looked up per call, not bound at import: an embedding process installs it after import.
    """
    return list(await get_gpu_backend().run_grid(**kwargs))


def parse_eval_results(workspace: Path | str, framework: str | None = None, benchmark_mode: str = "") -> dict[str, Any]:
    """Accuracy (or AgentX error rate) of a finished bench slot, as a dict the caller may extend."""
    return dict(get_gpu_backend().parse_eval_results(workspace, framework=framework, benchmark_mode=benchmark_mode))
