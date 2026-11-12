# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Ray-managed GPU execution backend."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from hyperloom_common.env import env_flag
from hyperloom.common.benchmark_config import BenchmarkPoint, BenchmarkPointError
from hyperloom.common.env_safety import scrub_benchmark_process_env
from hyperloom.common.visible_devices import hip_mask_within_rocr
from inference_suite import EXIT_REFUSED
from inference_suite.contracts import SuiteConfig

from . import _ray_runtime

if TYPE_CHECKING:
    from ._agentx_checkpoint import SuiteRoundCheckpoint

log = logging.getLogger(__name__)

# Env vars Ray owns inside its workers; never let a caller override them.
_RAY_OWNED_VISIBLE_DEVICE_VARS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)
#: The variable Ray narrows to the GPUs it placed a worker on.
_RAY_PLACED_GPUS_VAR = _RAY_OWNED_VISIBLE_DEVICE_VARS[0]
_UTF8 = "utf-8"


def ray_exec_enabled() -> bool:
    """Whether serving and ``needs_gpu`` specialist work runs through the Ray backend.

    Multi-node is always OFF. On single-node ``INFERENCE_OPTIMIZER_RAY_EXEC`` decides when set; otherwise it is ON,
    and OFF under pytest. ON also needs a Ray head that answers a driver (``ray_head_answers_driver``): one that does
    not turns this OFF, as if the flag were ``0``, until the head is probed again after a reconnect or a head start.
    """
    from ._multi_node_env import is_multi_node

    if is_multi_node():
        return False
    if not env_flag("INFERENCE_OPTIMIZER_RAY_EXEC", default=not os.environ.get("PYTEST_CURRENT_TEST")):
        return False
    return _ray_runtime.ray_head_answers_driver()


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
    session_remaining_sec: float | None = None
    checkpoint: SuiteRoundCheckpoint | None = None

    def run(self) -> tuple[int, str, str]:
        """Ray worker body: run the subprocess under session-kill semantics."""
        from hyperloom.orchestrator.actions.executors._subprocess_kill import (
            RoundWatch,
            run_with_session_kill,
            session_remaining_to_deadline_sec,
        )

        proc = run_with_session_kill(
            self.cmd,
            env=_merge_worker_env(self.env),
            cwd=self.cwd,
            watch=RoundWatch(
                timeout=self.timeout,
                session_deadline_sec=session_remaining_to_deadline_sec(self.session_remaining_sec),
                checkpoint=self.checkpoint,
            ),
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""


@dataclass(frozen=True)
class SuiteLeasedRound:
    """One inference_suite round a serving lease's actor runs, on the GPUs Ray placed the actor on.

    ``config_path`` holds the round's config. Only the actor knows which GPUs Ray gave it, so it names them in the
    config before the round starts; the round masks its server to them and samples them. The round holds itself to its
    own deadlines, so ``timeout`` and ``session_remaining_sec`` only backstop the process running it. ``checkpoint``
    stops the round from outside it, the way a cancel does, when it trails its reference.
    """

    config_path: str
    run_dir: str
    env: dict[str, str]
    timeout: float
    session_remaining_sec: float | None = None
    checkpoint: SuiteRoundCheckpoint | None = None

    @property
    def cmd(self) -> list[str]:
        return [sys.executable, "-m", "inference_suite", "run", "--config", self.config_path, "--run-dir", self.run_dir]

    def run(self) -> tuple[int, str, str]:
        """Ray worker body: pin the config to the actor's GPUs, then run the round's CLI under session-kill."""
        refusal = _pin_to_placed_gpus(Path(self.config_path))
        if refusal is not None:
            return EXIT_REFUSED, "", refusal
        return SessionKillRound(
            self.cmd,
            env=self.env,
            timeout=self.timeout,
            session_remaining_sec=self.session_remaining_sec,
            checkpoint=self.checkpoint,
        ).run()


def _pin_to_placed_gpus(config_path: Path) -> str | None:
    """Name the GPUs Ray placed this worker on as the round's server GPUs; why not, when the round cannot run there."""
    placed = os.environ.get(_RAY_PLACED_GPUS_VAR, "").strip()
    if not placed:
        # Ray narrowed nothing, so the GPUs the config names are the machine's own.
        return None
    try:
        config = SuiteConfig.from_document(yaml.safe_load(config_path.read_text(encoding=_UTF8)))
        gpus = tuple(int(gpu) for gpu in placed.split(","))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return f"config_invalid: the leased round's config cannot be pinned to GPUs {placed}: {exc}"
    if len(gpus) != len(config.server.gpus):
        return f"config_invalid: Ray placed the round on GPUs {placed}, its server runs on {len(config.server.gpus)}"
    pinned = replace(config, server=replace(config.server, gpus=gpus))
    config_path.write_text(yaml.safe_dump(pinned.to_document(), sort_keys=False), encoding=_UTF8)
    return None


#: A round either the lease's actor runs: a benchmark subprocess, or an inference_suite round.
LeasedRound = SessionKillRound | SuiteLeasedRound


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
        with src.open(encoding=_UTF8) as fp:
            cfg = yaml.safe_load(fp) or {}
        owned = [key for key in _RAY_OWNED_VISIBLE_DEVICE_VARS if key in BenchmarkPoint.from_document(cfg).envs]
    except (OSError, yaml.YAMLError, BenchmarkPointError):
        return src
    if not owned:
        return src
    for key in owned:
        del cfg["benchmark"]["envs"][key]
    out = src.with_name(f"{src.stem}.ray{src.suffix or '.yaml'}")
    try:
        with out.open("w", encoding=_UTF8) as fp:
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
    """Disconnect the current Ray driver and force the next use to re-ensure, probing the head before it reconnects."""
    try:
        import ray

        ray.shutdown()
    except (RuntimeError, OSError):
        pass
    # The head may have stopped answering drivers since it was probed, and the re-ensure connects in-process.
    _ray_runtime.forget_ray_head_verdict()
    if _BACKEND is not None:
        _BACKEND._ensured = False


__all__ = [
    "LeasedRound",
    "RayExecutionBackend",
    "SessionKillRound",
    "SuiteLeasedRound",
    "ray_exec_enabled",
    "get_ray_backend",
    "mark_ray_backend_unhealthy",
    "strip_visible_devices_from_config",
]
