# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Ray-managed GPU execution backend (P0 skeleton).

Unifies GPU/serving/benchmark execution onto Ray so a single node is just a
1-node Ray cluster. Behind ``INFERENCE_OPTIMIZER_RAY_EXEC`` (default off); when
disabled, callers keep using the existing local-subprocess path.

Only GPU/serving-class work runs through here. CPU/LLM steps (orchestration,
critic, target_analysis, report, session_breakdown, trace_analyze) stay local.

Hard invariant (see ray_modify.plan.md §4.2): any GPU process must live inside a
Ray task/actor lease. This module runs ``run_with_session_kill`` *inside* a Ray
worker; the worker inherits Ray's ``*_VISIBLE_DEVICES`` assignment, and the
caller's env is overlaid **without** overriding those (forcing them breaks Ray's
GPU isolation and triggers ROCm ``set_visible_accelerator_ids`` errors).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Env vars Ray owns inside its workers; never let a caller override them.
_RAY_OWNED_VISIBLE_DEVICE_VARS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


def ray_exec_enabled() -> bool:
    """Return True when ``INFERENCE_OPTIMIZER_RAY_EXEC`` selects the Ray backend.

    Default off (grayscale) so every executor can be switched/rolled back
    independently.

    Returns:
        ``True`` when the Ray execution backend is enabled.
    """
    val = os.environ.get("INFERENCE_OPTIMIZER_RAY_EXEC", "0").strip().lower()
    return val in {"1", "true", "yes", "on"}


@dataclass
class SubprocessResult:
    """Result of a subprocess executed inside a Ray worker.

    Mirrors the ``(returncode, stdout, stderr)`` triple the local path returns
    so executor-side parsing (``extract_benchmark_measurement`` etc.) is reused
    unchanged.
    """

    returncode: int
    stdout: str
    stderr: str


def _merge_worker_env(caller_env: dict[str, str] | None) -> dict[str, str]:
    """Merge caller env over the worker's env, preserving Ray's visible devices.

    Ray sets ``*_VISIBLE_DEVICES`` in the worker process; those must win over any
    values the caller passes so GPU isolation is Ray's alone.

    Args:
        caller_env: The env the executor would have used locally (may be None).

    Returns:
        The merged env for the subprocess: worker ``os.environ`` (with Ray's
        device assignment) overlaid by ``caller_env`` minus the device vars.
    """
    merged = dict(os.environ)
    for key, value in (caller_env or {}).items():
        if key in _RAY_OWNED_VISIBLE_DEVICE_VARS:
            continue
        merged[key] = value
    return merged


def _run_subprocess_worker(
    *,
    cmd: list[str],
    env: dict[str, str] | None,
    cwd: str | None,
    timeout_s: int | float | None,
    soft_deadline_sec: float | None,
    server_log_path: str | None,
    server_already_ready: bool,
) -> tuple[int, str, str]:
    """Ray worker body: run the subprocess under session-kill semantics.

    Executes on a Ray worker where ``*_VISIBLE_DEVICES`` are already set by Ray.
    Reuses :func:`run_with_session_kill` so kill/soft-deadline behaviour matches
    the local path exactly.

    Args:
        cmd: The command to execute.
        env: Caller env (overlaid without touching Ray's device vars).
        cwd: Working directory for the subprocess.
        timeout_s: Hard timeout in seconds.
        soft_deadline_sec: Overtime soft deadline.
        server_log_path: Path to the server log for watchdog markers.
        server_already_ready: Start the soft clock from spawn (warm reuse).

    Returns:
        ``(returncode, stdout, stderr)``.
    """
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        run_with_session_kill,
    )

    worker_env = _merge_worker_env(env)
    proc = run_with_session_kill(
        cmd,
        env=worker_env,
        cwd=cwd,
        timeout=timeout_s,
        soft_deadline_sec=soft_deadline_sec,
        server_log_path=server_log_path,
        server_already_ready=server_already_ready,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


class RayExecutionBackend:
    """Thin wrapper that runs GPU/serving subprocesses inside Ray workers.

    The cluster is ensured lazily on first use (single node = 1-node cluster),
    reusing the kernel agent's hardened ``ensure_ray_cluster`` (fd limit /
    version-mismatch recovery / loopback dashboard).
    """

    def __init__(self) -> None:
        self._ensured = False
        self._started = False

    def ensure(self, num_gpus: int | None = None, log_path: Path | None = None) -> None:
        """Ensure a Ray cluster is up and this process is connected.

        Idempotent. Reuses ``ensure_ray_cluster`` + ``quiet_ray_init`` from the
        kernel Ray runtime.

        Args:
            num_gpus: GPU count for ``ray start``; ``None`` lets Ray auto-detect
                (override via ``INFERENCE_OPTIMIZER_RAY_NUM_GPUS``).
            log_path: Optional path to append Ray lifecycle output.
        """
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
        self._started = ensure_ray_cluster(num_gpus=resolved, log_path=log_path)
        quiet_ray_init(num_gpus=resolved, log_path=log_path)
        self._ensured = True

    async def run_subprocess(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        num_gpus: float = 0,
        resources: dict[str, float] | None = None,
        timeout_s: int | float | None = None,
        result_dir: str | None = None,  # noqa: ARG002 — reserved for artifact addressing (§4.6)
        soft_deadline_sec: float | None = None,
        server_log_path: str | None = None,
        server_already_ready: bool = False,
    ) -> SubprocessResult:
        """Run ``cmd`` inside a Ray task holding ``num_gpus`` + ``resources``.

        Ray queues the task until the requested GPUs / custom resources
        (e.g. ``serving_slot``) are free, then runs the subprocess on a worker
        with the corresponding devices made visible.

        Args:
            cmd: Command to execute.
            env: Caller env (device vars are dropped; Ray owns them).
            cwd: Working directory.
            num_gpus: GPUs the task leases (Ray schedules on availability).
            resources: Ray custom resources (e.g. ``{"serving_slot": 1}``).
            timeout_s: Hard timeout.
            result_dir: Reserved for §4.6 unified artifact addressing.
            soft_deadline_sec: Overtime soft deadline.
            server_log_path: Server log path for watchdog markers.
            server_already_ready: Warm-reuse soft-clock semantics.

        Returns:
            The subprocess :class:`SubprocessResult`.
        """
        self.ensure()
        import ray

        # Ray's decorated remote function is dynamically typed; treat as Any so
        # mypy does not try to check the .remote(**kwargs) call shape.
        decorator: Any = ray.remote(num_gpus=num_gpus, resources=resources or {})
        worker: Any = decorator(_run_subprocess_worker)
        ref = worker.remote(
            cmd=cmd,
            env=env,
            cwd=cwd,
            timeout_s=timeout_s,
            soft_deadline_sec=soft_deadline_sec,
            server_log_path=server_log_path,
            server_already_ready=server_already_ready,
        )
        rc, out, err = await asyncio.to_thread(ray.get, ref)
        return SubprocessResult(returncode=rc, stdout=out, stderr=err)


def resolve_shared_artifact_root(session_dir: Path | str) -> Path:
    """Resolve the session-level shared root for per-task artifacts (§4.6).

    Single-node: the local session dir (a Ray worker on the same host sees it).
    Multi-node: the shared filesystem root (``$HYPERLOOM_MN_PROFILE_TRACE_DIR``
    or the session dir under the shared mount) so a worker on another node can
    write artifacts the collector still resolves by relative path.

    Args:
        session_dir: The session directory.

    Returns:
        The shared artifact root path.
    """
    session_dir = Path(session_dir)
    mn_root = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    from ._multi_node_env import is_multi_node

    if is_multi_node() and mn_root:
        return Path(mn_root)
    return session_dir


# Process-wide singleton (lazy cluster ensure on first use).
_BACKEND: RayExecutionBackend | None = None


def get_ray_backend() -> RayExecutionBackend:
    """Return the process-wide :class:`RayExecutionBackend` singleton.

    Returns:
        The shared backend instance (created on first call).
    """
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = RayExecutionBackend()
    return _BACKEND


__all__ = [
    "RayExecutionBackend",
    "SubprocessResult",
    "get_ray_backend",
    "ray_exec_enabled",
    "resolve_shared_artifact_root",
]
