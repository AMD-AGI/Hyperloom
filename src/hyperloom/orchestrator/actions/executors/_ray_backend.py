# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Ray-managed GPU execution backend."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.env_safety import scrub_benchmark_process_env

log = logging.getLogger(__name__)

# Env vars Ray owns inside its workers; never let a caller override them.
_RAY_OWNED_VISIBLE_DEVICE_VARS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


def ray_exec_enabled() -> bool:
    """Return whether GPU/serving work should run through the Ray backend."""
    val = os.environ.get("INFERENCE_OPTIMIZER_RAY_EXEC", "").strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    # Unset: single-node forced ON, multi-node OFF.
    from ._multi_node_env import is_multi_node

    return not is_multi_node()


def _should_use_ray_backend() -> bool:
    """Like :func:`ray_exec_enabled` but stays OFF under pytest when unset."""
    val = os.environ.get("INFERENCE_OPTIMIZER_RAY_EXEC", "").strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    from ._multi_node_env import is_multi_node

    return not is_multi_node()


def ray_gpu_specialist_exec_enabled() -> bool:
    """Whether ``needs_gpu`` specialists route through the Ray backend."""
    from ._multi_node_env import is_multi_node

    return _should_use_ray_backend() and not is_multi_node()


def ray_gpu_pending_limit() -> int:
    """Max in-flight (pending + running) GPU specialists admitted to Ray at once."""
    try:
        v = int(os.environ.get("INFERENCE_OPTIMIZER_RAY_GPU_PENDING_LIMIT", "4"))
    except (TypeError, ValueError):
        return 4
    return max(1, v)


def ray_serving_priority_enabled() -> bool:
    """Whether serving is prioritized over GPU research specialists (§3.4)."""
    val = os.environ.get("INFERENCE_OPTIMIZER_RAY_SERVING_PRIORITY", "").strip().lower()
    return val not in {"0", "false", "no", "off"}


def serving_slot_busy() -> bool:
    """Best-effort check: is Ray's whole-machine ``serving_slot`` currently held?"""
    if not ray_gpu_specialist_exec_enabled():
        return False
    try:
        import ray  # noqa: PLC0415

        if not ray.is_initialized():
            return False
        avail = ray.available_resources()
        return float(avail.get("serving_slot", 1.0)) < 1.0
    except Exception:  # noqa: BLE001 — best-effort; never block dispatch
        return False


@dataclass
class SubprocessResult:
    """Declarative shape for a subprocess executed inside a Ray worker."""

    returncode: int
    stdout: str
    stderr: str


def _merge_worker_env(caller_env: dict[str, str] | None) -> dict[str, str]:
    """Merge caller env over the worker's env, preserving Ray's visible devices."""
    merged = dict(os.environ)
    for key, value in (caller_env or {}).items():
        if key in _RAY_OWNED_VISIBLE_DEVICE_VARS:
            continue
        merged[key] = value
    return scrub_benchmark_process_env(merged)


def _run_subprocess_worker(
    *,
    cmd: list[str],
    env: dict[str, str] | None,
    cwd: str | None,
    timeout_s: int | float | None,
    soft_deadline_sec: float | None,
    server_log_path: str | None,
    server_already_ready: bool,
    session_remaining_sec: float | None = None,
) -> tuple[int, str, str]:
    """Ray worker body: run the subprocess under session-kill semantics."""
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        session_remaining_to_deadline_sec,
    )
    from hyperloom.orchestrator.actions.executors.launch_backend import launch

    worker_env = _merge_worker_env(env)
    proc = launch(
        cmd,
        env=worker_env,
        cwd=cwd,
        timeout=timeout_s,
        soft_deadline_sec=soft_deadline_sec,
        server_log_path=server_log_path,
        server_already_ready=server_already_ready,
        session_deadline_sec=session_remaining_to_deadline_sec(session_remaining_sec),
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


class RayExecutionBackend:
    """Thin wrapper that runs GPU/serving subprocesses inside Ray workers."""

    def __init__(self) -> None:
        self._ensured = False

    def ensure(self, num_gpus: int | None = None, log_path: Path | None = None) -> None:
        """Ensure a Ray cluster is up and this process is connected."""
        if self._ensured:
            return
        from hyperloom.agents.kernel.tools.backends.ray_runtime import (
            ensure_ray_cluster,
            quiet_ray_init,
        )

        resolved = num_gpus
        if resolved is None:
            env_n = os.environ.get("INFERENCE_OPTIMIZER_RAY_NUM_GPUS", "").strip()
            resolved = int(env_n) if env_n.isdigit() else None
        ensure_ray_cluster(num_gpus=resolved, log_path=log_path)
        quiet_ray_init(num_gpus=resolved, log_path=log_path)
        self._ensured = True


def resolve_shared_artifact_root(session_dir: Path | str) -> Path:
    """Resolve the shared artifact root for per-task artifacts."""
    session_dir = Path(session_dir)
    mn_root = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    from ._multi_node_env import is_multi_node

    if is_multi_node() and mn_root:
        return Path(mn_root)
    return session_dir


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
        import ray  # noqa: PLC0415

        ray.shutdown()
    except Exception:  # noqa: BLE001 - recovery must never raise
        pass
    if _BACKEND is not None:
        _BACKEND._ensured = False


__all__ = [
    "RayExecutionBackend",
    "SubprocessResult",
    "_should_use_ray_backend",
    "get_ray_backend",
    "mark_ray_backend_unhealthy",
    "ray_exec_enabled",
    "resolve_shared_artifact_root",
    "strip_visible_devices_from_config",
]
