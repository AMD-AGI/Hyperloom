# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Ray-managed GPU execution backend."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.env import env_flag
from hyperloom.common.env_safety import scrub_benchmark_process_env
from hyperloom.common.visible_devices import hip_mask_within_rocr

log = logging.getLogger(__name__)

# Env vars Ray owns inside its workers; never let a caller override them.
_RAY_OWNED_VISIBLE_DEVICE_VARS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


def ray_exec_enabled() -> bool:
    """Whether serving and ``needs_gpu`` specialist work runs through the Ray backend.

    Multi-node is always OFF. On single-node ``INFERENCE_OPTIMIZER_RAY_EXEC`` decides when set; otherwise it is ON,
    and OFF under pytest.
    """
    from ._multi_node_env import is_multi_node

    if is_multi_node():
        return False
    return env_flag("INFERENCE_OPTIMIZER_RAY_EXEC", default=not os.environ.get("PYTEST_CURRENT_TEST"))


def ray_gpu_pending_limit() -> int:
    """Max in-flight (pending + running) GPU specialists admitted to Ray at once."""
    try:
        v = int(os.environ.get("INFERENCE_OPTIMIZER_RAY_GPU_PENDING_LIMIT", "4"))
    except (TypeError, ValueError):
        return 4
    return max(1, v)


def ray_serving_priority_enabled() -> bool:
    """Whether serving is prioritized over GPU research specialists (§3.4)."""
    return env_flag("INFERENCE_OPTIMIZER_RAY_SERVING_PRIORITY", default=True)


def serving_slot_busy() -> bool:
    """Best-effort check: is Ray's whole-machine ``serving_slot`` currently held?"""
    if not ray_exec_enabled():
        return False
    import ray

    if not ray.is_initialized():
        return False
    try:
        avail = ray.available_resources()
    except (ray.exceptions.RayError, OSError):
        return False
    return float(avail.get("serving_slot", 1.0)) < 1.0


def align_hip_with_ray_rocr(env: dict[str, str]) -> dict[str, str]:
    """Ray narrows only ``ROCR_VISIBLE_DEVICES`` per task; re-index HIP into that view, in place."""
    hip = hip_mask_within_rocr(env)
    if hip is not None:
        env["HIP_VISIBLE_DEVICES"] = hip
    return env


def _merge_worker_env(caller_env: dict[str, str] | None) -> dict[str, str]:
    """Merge caller env over the worker's env, preserving Ray's visible devices."""
    merged = dict(os.environ)
    for key, value in (caller_env or {}).items():
        if key in _RAY_OWNED_VISIBLE_DEVICE_VARS:
            continue
        merged[key] = value
    return scrub_benchmark_process_env(align_hip_with_ray_rocr(merged))


@dataclass(frozen=True)
class SessionKillRound:
    """One benchmark round a Ray worker runs under session-kill semantics.

    ``session_remaining_sec`` is a duration, re-anchored as a deadline on the worker's own clock.
    """

    cmd: list[str]
    env: dict[str, str] | None = None
    cwd: str | None = None
    timeout: int | float | None = None
    silence_timeout_sec: float | None = None
    server_log_path: str | None = None
    server_already_ready: bool = False
    session_remaining_sec: float | None = None


def _run_subprocess_worker(bench_round: SessionKillRound) -> tuple[int, str, str]:
    """Ray worker body: run the subprocess under session-kill semantics."""
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        session_remaining_to_deadline_sec,
    )
    from hyperloom.orchestrator.actions.executors._subprocess_kill import RoundWatch, run_with_session_kill

    worker_env = _merge_worker_env(bench_round.env)
    proc = run_with_session_kill(
        bench_round.cmd,
        env=worker_env,
        cwd=bench_round.cwd,
        watch=RoundWatch(
            timeout=bench_round.timeout,
            silence_timeout_sec=bench_round.silence_timeout_sec,
            server_log_path=bench_round.server_log_path,
            server_already_ready=bench_round.server_already_ready,
            session_deadline_sec=session_remaining_to_deadline_sec(bench_round.session_remaining_sec),
        ),
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


class RayExecutionBackend:
    """Thin wrapper that runs GPU/serving subprocesses inside Ray workers."""

    def __init__(self) -> None:
        self._ensured = False

    def ensure(self, num_gpus: int | None = None) -> None:
        """Ensure a Ray cluster is up and this process is connected."""
        if self._ensured:
            return
        from ._ray_runtime import (
            ensure_ray_cluster,
            quiet_ray_init,
        )

        resolved = _resolve_head_num_gpus(num_gpus)
        ensure_ray_cluster(num_gpus=resolved)
        quiet_ray_init(num_gpus=resolved)
        self._ensured = True

    def restart_local_head(self) -> None:
        """Replace the local head with one that declares ``serving_slot`` and reconnect to it."""
        from ._ray_runtime import restart_local_head_with_serving_slot

        self._ensured = False
        restart_local_head_with_serving_slot(num_gpus=_resolve_head_num_gpus(None))
        self._ensured = True


def _resolve_head_num_gpus(num_gpus: int | None) -> int | None:
    """Return the ``--num-gpus`` for a head this process starts: the argument, else the env override."""
    if num_gpus is not None:
        return num_gpus
    env_n = os.environ.get("INFERENCE_OPTIMIZER_RAY_NUM_GPUS", "").strip()
    return int(env_n) if env_n.isdigit() else None


def strip_visible_devices_from_config(config_path: Path | str) -> Path:
    """Drop ``benchmark.envs.*_VISIBLE_DEVICES`` from a benchmark YAML."""
    import yaml

    src = Path(config_path)
    try:
        with src.open(encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp) or {}
    except (OSError, yaml.YAMLError):
        return src
    envs = (cfg.get("benchmark") or {}).get("envs")
    if not isinstance(envs, dict):
        return src
    changed = False
    for key in _RAY_OWNED_VISIBLE_DEVICE_VARS:
        if key in envs:
            envs.pop(key, None)
            changed = True
    if not changed:
        return src
    out = src.with_name(f"{src.stem}.ray{src.suffix or '.yaml'}")
    try:
        with out.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(cfg, fp, sort_keys=False)
    except OSError:
        return src
    return out


# Process-wide singleton (lazy cluster ensure on first use).
_BACKEND: RayExecutionBackend | None = None


def get_ray_backend() -> RayExecutionBackend:
    """Return the process-wide :class:`RayExecutionBackend` singleton."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = RayExecutionBackend()
    return _BACKEND


def mark_ray_backend_unhealthy() -> None:
    """Disconnect the current Ray driver and force the next use to re-ensure."""
    try:
        import ray

        ray.shutdown()
    except (RuntimeError, OSError):
        pass
    if _BACKEND is not None:
        _BACKEND._ensured = False


__all__ = [
    "RayExecutionBackend",
    "ray_exec_enabled",
    "get_ray_backend",
    "mark_ray_backend_unhealthy",
    "strip_visible_devices_from_config",
]
