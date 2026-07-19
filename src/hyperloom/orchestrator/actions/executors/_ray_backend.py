# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Ray-managed GPU execution backend.

Runs GPU/serving/benchmark subprocesses inside Ray tasks/actors so every GPU
process lives inside a Ray lease and inherits Ray's ``*_VISIBLE_DEVICES``
assignment. CPU/LLM steps stay on the local subprocess path.
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
    """Return whether GPU/serving work should run through the Ray backend.

    When ``INFERENCE_OPTIMIZER_RAY_EXEC`` is unset: ON for single-node, OFF for
    multi-node. The env var is an explicit override / emergency escape valve.
    """
    val = os.environ.get("INFERENCE_OPTIMIZER_RAY_EXEC", "").strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    # Unset: single-node forced ON, multi-node OFF (decisions 2 + 4).
    from ._multi_node_env import is_multi_node

    return not is_multi_node()


def _should_use_ray_backend() -> bool:
    """Like :func:`ray_exec_enabled` but stays OFF under pytest when unset.

    Tests run the local subprocess path by default; ``INFERENCE_OPTIMIZER_RAY_EXEC=1``
    still opts a specific test into the Ray path.
    """
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
    """Whether ``needs_gpu`` specialists route through the Ray backend.

    Mirrors the gate inside
    :func:`hyperloom.orchestrator.actions.executors._ray_serving.maybe_gpu_specialist_lease`
    (single-node + ``INFERENCE_OPTIMIZER_RAY_EXEC`` on + not the pytest default)
    so the dispatcher can pick the Ray admission path (§3.2 count-based pending
    limit, physical mutex owned by Ray ``num_gpus``) over the legacy SQLite
    physical-capacity hard gate. Multi-node / RAY_EXEC off / pytest keep the
    legacy SQLite pool.

    Returns:
        ``True`` when GPU specialists run through Ray on this (single) node.
    """
    from ._multi_node_env import is_multi_node

    return _should_use_ray_backend() and not is_multi_node()


def ray_gpu_pending_limit() -> int:
    """Max in-flight (pending + running) GPU specialists admitted to Ray at once.

    Backpressure so a burst of GPU specialists cannot flood the single-node Ray
    queue and starve serving (§3.2 / invariant §6.5). Ray still runs only as
    many as fit ``num_gpus`` concurrently; this bounds how many may be *queued*.
    Override via ``INFERENCE_OPTIMIZER_RAY_GPU_PENDING_LIMIT`` (default 4,
    floored at 1).

    Returns:
        The pending-admission ceiling (>= 1).
    """
    try:
        v = int(os.environ.get("INFERENCE_OPTIMIZER_RAY_GPU_PENDING_LIMIT", "4"))
    except (TypeError, ValueError):
        return 4
    return max(1, v)


def ray_serving_priority_enabled() -> bool:
    """Whether serving is prioritized over GPU research specialists (§3.4).

    When on (the default), the dispatcher pauses admitting NEW GPU research
    specialists while serving currently holds the whole-machine slot, so a
    research pile-up cannot starve serving. Disable with
    ``INFERENCE_OPTIMIZER_RAY_SERVING_PRIORITY=0``.

    Returns:
        ``True`` unless the env var explicitly disables it.
    """
    val = os.environ.get("INFERENCE_OPTIMIZER_RAY_SERVING_PRIORITY", "").strip().lower()
    return val not in {"0", "false", "no", "off"}


def serving_slot_busy() -> bool:
    """Best-effort check: is Ray's whole-machine ``serving_slot`` currently held?

    ``True`` when a serving benchmark (or a bench-capable specialist) currently
    holds the slot — i.e. serving is active. Used by the §3.4 serving-priority
    gate to defer new GPU research specialists. Any error (Ray not initialised,
    resource absent, probe failure) returns ``False`` so it NEVER blocks
    dispatch, and it is a no-op off the single-node Ray path.

    Returns:
        ``True`` only when Ray reports ``serving_slot`` availability below 1.
    """
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
        import ray

        ref = self._submit(
            cmd,
            env=env,
            cwd=cwd,
            num_gpus=num_gpus,
            resources=resources,
            timeout_s=timeout_s,
            soft_deadline_sec=soft_deadline_sec,
            server_log_path=server_log_path,
            server_already_ready=server_already_ready,
        )
        rc, out, err = await asyncio.to_thread(ray.get, ref)
        return SubprocessResult(returncode=rc, stdout=out, stderr=err)

    def run_subprocess_sync(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        num_gpus: float = 0,
        resources: dict[str, float] | None = None,
        timeout_s: int | float | None = None,
        result_dir: str | None = None,  # noqa: ARG002 — reserved for §4.6
        soft_deadline_sec: float | None = None,
        server_log_path: str | None = None,
        server_already_ready: bool = False,
    ) -> SubprocessResult:
        """Blocking variant of :meth:`run_subprocess` for sync call sites.

        Used by ``_run_magpie`` (already invoked via ``asyncio.to_thread``), so a
        blocking ``ray.get`` here does not stall the event loop.

        Returns:
            The subprocess :class:`SubprocessResult`.
        """
        import ray

        ref = self._submit(
            cmd,
            env=env,
            cwd=cwd,
            num_gpus=num_gpus,
            resources=resources,
            timeout_s=timeout_s,
            soft_deadline_sec=soft_deadline_sec,
            server_log_path=server_log_path,
            server_already_ready=server_already_ready,
        )
        rc, out, err = ray.get(ref)
        return SubprocessResult(returncode=rc, stdout=out, stderr=err)

    def _submit(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None,
        cwd: str | None,
        num_gpus: float,
        resources: dict[str, float] | None,
        timeout_s: int | float | None,
        soft_deadline_sec: float | None,
        server_log_path: str | None,
        server_already_ready: bool,
    ) -> Any:
        """Ensure the cluster and submit the worker task; return its ObjectRef.

        Returns:
            The Ray ``ObjectRef`` for the submitted worker task.
        """
        self.ensure()
        import ray

        # Ray's decorated remote function is dynamically typed; treat as Any so
        # mypy does not check the .remote(**kwargs) call shape.
        decorator: Any = ray.remote(num_gpus=num_gpus, resources=resources or {})
        worker: Any = decorator(_run_subprocess_worker)
        return worker.remote(
            cmd=cmd,
            env=env,
            cwd=cwd,
            timeout_s=timeout_s,
            soft_deadline_sec=soft_deadline_sec,
            server_log_path=server_log_path,
            server_already_ready=server_already_ready,
        )


def resolve_shared_artifact_root(session_dir: Path | str) -> Path:
    """Resolve the shared artifact root for per-task artifacts.

    Single-node: the local session dir. Multi-node: ``$HYPERLOOM_MN_PROFILE_TRACE_DIR``
    (or the session dir) so workers on other nodes can still write artifacts.
    """
    session_dir = Path(session_dir)
    mn_root = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    from ._multi_node_env import is_multi_node

    if is_multi_node() and mn_root:
        return Path(mn_root)
    return session_dir


def strip_visible_devices_from_config(config_path: Path | str) -> Path:
    """Drop ``benchmark.envs.*_VISIBLE_DEVICES`` from a benchmark YAML.

    Ray sets device vars in the worker; leaving them in the YAML would override
    Ray's assignment. Best-effort: returns the original path on any parse/write
    error, and is a no-op when no device var is present.
    """
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
    "_should_use_ray_backend",
    "get_ray_backend",
    "ray_exec_enabled",
    "resolve_shared_artifact_root",
    "strip_visible_devices_from_config",
]
