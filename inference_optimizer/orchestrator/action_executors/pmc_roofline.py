"""Dedicated PMC roofline ActionRunner.

Unlike ``profile``, this runner launches its own server process with
``LD_PRELOAD=librocprofiler-register.so``. The normal torch-profiler/TraceLens
server must stay preload-free because rocprofiler-sdk only allows one active
tool registration per process.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..roofline_integration import _as_cmd, run_isolated_pmc_roofline
from ..sub_agent_runner import RunnerContext


PMC_ROOFLINE_DEFAULT_TIMEOUT_SEC = 1800
_RAY_CONTEXT_ENV_KEYS = (
    "HYPERLOOM_PMC_ROOFLINE_IN_RAY",
    "RAY_ADDRESS",
    "RAY_JOB_ID",
    "RAY_RUNTIME_ENV_CREATE_WORKING_DIR",
)
_GPU_VISIBILITY_ENV_KEYS = ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _ray_context_present(params: dict[str, Any]) -> bool:
    return _truthy(params.get("ray_worker")) or any(
        os.environ.get(key) for key in _RAY_CONTEXT_ENV_KEYS
    )


class PMCRooflineExecutor:
    """Run PMC/roofline collection in an isolated server process."""

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        params = ctx.task.params or {}
        allow_direct_gpu = _truthy(params.get("allow_direct_gpu")) or _truthy(
            os.environ.get("HYPERLOOM_ALLOW_DIRECT_PMC_ROOFLINE")
        )
        if not allow_direct_gpu and not _ray_context_present(params):
            return {
                "status": "failed",
                "error_class": "ray_worker_required",
                "error": (
                    "pmc_roofline must run inside a RayJob/Ray worker so GPU "
                    "allocation is owned by Ray. Set task.params.ray_worker=true "
                    "inside the Ray job, or pass allow_direct_gpu=true only for "
                    "local developer debugging."
                ),
            }

        server_cmd = _as_cmd(params.get("server_cmd"))
        if not server_cmd:
            return {
                "status": "failed",
                "error_class": "missing_server_cmd",
                "error": "pmc_roofline requires task.params.server_cmd",
            }

        output_dir = Path(
            params.get("output_dir")
            or Path("/tmp") / f"pmc-roofline-{ctx.task.task_id[:8]}"
        )
        health_url = str(params.get("health_url") or "http://127.0.0.1:8000/health")
        benchmark_cmd = _as_cmd(params.get("benchmark_cmd")) or None
        env_overrides = {
            str(k): str(v)
            for k, v in dict(params.get("extra_envs") or {}).items()
        }
        if not _truthy(params.get("allow_device_override")):
            blocked = sorted(k for k in _GPU_VISIBILITY_ENV_KEYS if k in env_overrides)
            if blocked:
                return {
                    "status": "failed",
                    "error_class": "ray_gpu_visibility_override",
                    "error": (
                        "pmc_roofline must use the GPU visibility assigned by Ray; "
                        f"remove extra_envs overrides for {', '.join(blocked)} or "
                        "set allow_device_override=true for an explicit exception."
                    ),
                }

        result = run_isolated_pmc_roofline(
            session_dir=output_dir,
            server_cmd=server_cmd,
            health_url=health_url,
            benchmark_cmd=benchmark_cmd,
            duration_ms=int(params.get("duration_ms") or 15000),
            precision=str(params.get("precision") or "fp16"),
            startup_timeout_s=int(params.get("startup_timeout_s") or 600),
            env_overrides=env_overrides,
            profile_mode=str(params.get("profile_mode") or "launch"),
        )
        result.setdefault("output_dir", str(output_dir))
        result.setdefault("health_url", health_url)
        return result


pmc_roofline_executor = PMCRooflineExecutor()


__all__ = [
    "PMC_ROOFLINE_DEFAULT_TIMEOUT_SEC",
    "PMCRooflineExecutor",
    "pmc_roofline_executor",
]
