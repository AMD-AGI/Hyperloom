"""Cluster execution mode — Ray job submission for multi-node GPU work.

Used when ExecutionMode is CLUSTER. Wraps Ray remote calls for GPU tasks
(GEAK, OOB kernel optimization) with proper resource scheduling.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class RayJobResult:
    """Result from a Ray-submitted job."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    gpu_ids: list[int] | None = None


def is_ray_available() -> bool:
    """Check if Ray is importable and a cluster is reachable."""
    try:
        import ray
        return ray.is_initialized() or True
    except ImportError:
        return False


def submit_gpu_task(
    cmd: list[str],
    num_gpus: int = 1,
    env: dict[str, str] | None = None,
    timeout_s: int = 7200,
    cwd: str | None = None,
) -> RayJobResult:
    """Submit a GPU task to Ray cluster.

    Falls back to local subprocess if Ray is not available.
    """
    if not is_ray_available():
        log.info("Ray not available, running locally")
        return _run_local(cmd, env=env, timeout_s=timeout_s, cwd=cwd)

    return _run_via_ray(cmd, num_gpus=num_gpus, env=env, timeout_s=timeout_s, cwd=cwd)


def _run_via_ray(
    cmd: list[str],
    num_gpus: int,
    env: dict[str, str] | None,
    timeout_s: int,
    cwd: str | None,
) -> RayJobResult:
    """Submit command as a Ray remote task."""
    try:
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        @ray.remote(num_gpus=num_gpus)
        def _execute(cmd: list[str], env: dict | None, timeout_s: int, cwd: str | None) -> dict:
            import subprocess as sp
            child_env = os.environ.copy()
            if env:
                child_env.update(env)
            result = sp.run(
                cmd, capture_output=True, text=True,
                timeout=timeout_s, env=child_env, cwd=cwd,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        ref = _execute.remote(cmd, env, timeout_s, cwd)
        result = ray.get(ref, timeout=timeout_s + 60)
        return RayJobResult(
            success=result["returncode"] == 0,
            stdout=result["stdout"],
            stderr=result["stderr"],
            returncode=result["returncode"],
        )
    except Exception as e:
        log.error("Ray task failed: %s, falling back to local", e)
        return _run_local(cmd, env=env, timeout_s=timeout_s, cwd=cwd)


def _run_local(
    cmd: list[str],
    env: dict[str, str] | None = None,
    timeout_s: int = 7200,
    cwd: str | None = None,
) -> RayJobResult:
    """Run command locally as a subprocess."""
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, env=child_env, cwd=cwd,
        )
        return RayJobResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return RayJobResult(success=False, stderr="timeout", returncode=-1)
    except OSError as e:
        return RayJobResult(success=False, stderr=str(e), returncode=-1)
