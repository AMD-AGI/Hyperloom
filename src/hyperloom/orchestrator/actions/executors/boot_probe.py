# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Answer "does this combo boot?" without measuring how fast it is.

The action boots a server, holds it, checks health, asks for one short
completion and tears it down. It goes through the same launch backend and the
same classifier as a baseline, so it produces the same
:class:`~hyperloom.common.bringup.BootObservation` at the round slot's usual
path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hyperloom.common.bringup import BootObservation, LadderStage
from hyperloom.common.model_paths import resolve_session_model_path

from ...bringup import (
    observe_bringup,
    write_boot_observation,
)
from ...bringup.env_preflight import (
    ENV_FAULT,
    FAULT,
    UNAVAILABLE,
    check_environment,
    env_fault_observation,
)
from ...loop.sub_agent_runner import RunnerContext
from ._workload_envs import (
    FrameworkScriptMismatchError,
    default_baseline_config,
    materialize_config_with_envs,
)
from .launch_backend import launch

log = logging.getLogger(__name__)

#: The deepest milestone this probe can witness: it makes the server generate,
#: but what the ladder classifier can place off a server's own log stops at a
#: serving HTTP front end.
PROBE_CEILING = LadderStage.HTTP_READY

#: Wall-clock bound on the whole probe, sized for a cold weight load on a large
#: checkpoint.
DEFAULT_TIMEOUT_SEC = 3600


def probe_would_inform(observation: BootObservation | None) -> bool:
    """Whether running the probe would tell the round anything it does not know.

    Args:
        observation: The boot observation the round already recorded, or
            ``None`` when it has none.

    Returns:
        bool: True only when the round has no answer yet -- a boot already
        witnessed at :data:`PROBE_CEILING`, or already placed at the stage it
        stopped at, is an answer a second boot of the same combo only copies.
    """
    if observation is None:
        return True
    if observation.stage_reached >= PROBE_CEILING:
        return False
    return observation.stage_failed is None


class BootProbeExecutor:
    """Runs one boot probe and records the observation it produced."""

    def __init__(self, *, session_dir: Path | str | None = None) -> None:
        """Bind the executor to the session its observations are written under.

        Args:
            session_dir: The session root, or ``None`` when the executor is
                driven directly and nothing is persisted under a session.
        """
        self.session_dir = Path(session_dir) if session_dir else None

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Boot the configured combo, probe it, and report the boot verdict.

        Args:
            ctx: Runner context carrying ``task.params`` and ``extra``.

        Returns:
            dict[str, Any]: ``status`` plus ``booted``, the observation's path,
            and -- when the host itself is at fault -- ``error_class``
            :data:`~hyperloom.orchestrator.bringup.env_preflight.ENV_FAULT`.
        """
        params = dict(ctx.task.params)
        output_dir = self._workspace(ctx, params)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = Path(params.get("config_path") or default_baseline_config())
        if not config_path.exists():
            return {
                "status": "failed",
                "error_class": "config_missing",
                "error": f"boot probe config not found: {config_path}",
                "output_dir": str(output_dir),
            }
        # The shipped YAML's `model`/`precision` and workload envs are the
        # fallbacks its own header calls OVERRIDDEN-at-runtime, so booting it
        # unrendered probes a combo this session never asked for and the ladder
        # places a wall that is not the run's. RUN_EVAL is forced off: the probe
        # asks whether the server comes up, never how well it answers.
        try:
            config_path = materialize_config_with_envs(
                config_path,
                output_dir,
                extra_envs={"RUN_EVAL": "false"},
                model_path=resolve_session_model_path(params=params, for_serving=True) or None,
                gpu_type=str(params.get("gpu_type") or "").strip().lower()
                or os.environ.get("GPU_TYPE", "").strip().lower(),
                out_name="boot_probe_config.with_envs.yaml",
            )
        except FrameworkScriptMismatchError as exc:
            return {
                "status": "failed",
                "error_class": "framework_script_mismatch",
                "error": str(exc),
                "output_dir": str(output_dir),
            }

        framework = str(params.get("framework") or "").strip().lower()
        env = self._launch_env(params)
        fault = self._environment_fault(
            framework=framework,
            params=params,
            env=env,
            output_dir=output_dir,
        )
        if fault is not None:
            return fault

        server_log = output_dir / "server.log"
        timeout_sec = int(params.get("timeout_sec") or DEFAULT_TIMEOUT_SEC)
        cmd = [
            sys.executable,
            "-m",
            "hyperloom.orchestrator.actions.executors.bypass_runner",
            "benchmark",
            "--benchmark-config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--run-mode",
            "local",
            "--phase",
            "boot_probe",
        ]
        started = time.time()
        try:
            proc = await asyncio.to_thread(
                launch,
                cmd,
                env=env,
                cwd=str(output_dir),
                timeout=timeout_sec,
                server_log_path=str(server_log),
            )
            returncode = int(proc.returncode)
            stdout, stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            # A reaped probe carries no wrapper streams; the server log is the
            # whole of the evidence, and it is still worth classifying.
            returncode, stdout, stderr = -1, "", ""

        observation_path = self._record(
            server_log,
            wrapper_stdout=stdout,
            wrapper_stderr=stderr,
            output_dir=output_dir,
        )
        booted = returncode == 0
        return {
            "status": "succeeded" if booted else "failed",
            "booted": booted,
            "returncode": returncode,
            "boot_observation_path": observation_path,
            "runtime_sec": max(0.0, time.time() - started),
            "output_dir": str(output_dir),
            "error": "" if booted else (stderr.strip()[-2000:] or "the boot probe did not reach a serving server"),
        }

    def _workspace(self, ctx: RunnerContext, params: dict[str, Any]) -> Path:
        """Resolve the slot this probe runs in.

        Args:
            ctx: The runner context.
            params: The task params.

        Returns:
            Path: The round workspace.
        """
        named = str(params.get("output_dir") or "").strip()
        if named:
            return Path(named)
        workspace = str(ctx.extra.get("workspace") or "").strip()
        if workspace:
            return Path(workspace)
        from hyperloom.inference_optimizer.session.session_paths import runs_dir

        return runs_dir(self.session_dir or Path.cwd(), "boot_probe", ctx.task.task_id)

    @staticmethod
    def _launch_env(params: dict[str, Any]) -> dict[str, str]:
        """Build the environment the probe's subprocess is launched with.

        Args:
            params: The task params, whose ``extra_envs`` layer the round's own
                overrides on top of the ambient environment.

        Returns:
            dict[str, str]: The launch environment.
        """
        from hyperloom.common.env_safety import build_benchmark_env

        overrides = {str(k): str(v) for k, v in (params.get("extra_envs") or {}).items()}
        return build_benchmark_env(overrides)

    @staticmethod
    def _serving_port(params: dict[str, Any], env: dict[str, str]) -> int:
        """Return the port the server would bind, ``0`` when none is named.

        A value that does not name a port leaves the port check with nothing to
        judge, which is an unavailable verdict; the launch itself will say what
        the framework makes of it.

        Args:
            params: The task params.
            env: The launch environment.

        Returns:
            int: The port, or ``0`` when none was named or it does not parse.
        """
        for raw in (params.get("port"), env.get("PORT")):
            text = str(raw or "").strip()
            if not text:
                continue
            try:
                return int(text)
            except ValueError:
                log.warning("boot probe: port check %s -- %r does not name a port", UNAVAILABLE, text)
                return 0
        return 0

    def _environment_fault(
        self,
        *,
        framework: str,
        params: dict[str, Any],
        env: dict[str, str],
        output_dir: Path,
    ) -> dict[str, Any] | None:
        """Return a terminal result when the host cannot host this round.

        Args:
            framework: Framework the round serves.
            params: The task params, naming the model and the port.
            env: The launch environment the probe resolved.
            output_dir: The round slot, for the observation artifact.

        Returns:
            dict | None: The terminal result, or ``None`` when the round may
            launch -- including every unavailable verdict, which must never
            cost a round. Nothing here raises: a preflight that cannot answer
            is an unavailable verdict, not an exception.
        """
        verdict = check_environment(
            framework=framework,
            model=str(params.get("model") or env.get("MODEL_PATH") or env.get("MODEL") or ""),
            port=self._serving_port(params, env),
            launch_env=env,
        )
        if verdict.status != FAULT:
            return None
        observation = env_fault_observation(verdict, session_dir=self.session_dir)
        ref = self._write(observation, output_dir)
        log.error("boot probe: the host cannot run this round (%s): %s", verdict.fault, verdict.detail)
        return {
            "status": "failed",
            "booted": False,
            "error_class": ENV_FAULT,
            "env_fault": verdict.fault,
            "error": f"{verdict.fault}: {verdict.detail}",
            "boot_observation_path": ref,
            "output_dir": str(output_dir),
        }

    def _record(
        self,
        server_log: Path,
        *,
        wrapper_stdout: str,
        wrapper_stderr: str,
        output_dir: Path,
    ) -> str:
        """Classify the probe's server log and persist the observation.

        Args:
            server_log: The log the probe's server wrote.
            wrapper_stdout: The launcher's stdout, fallback evidence.
            wrapper_stderr: The launcher's stderr, fallback evidence.
            output_dir: The round slot the artifact is indexed under.

        Returns:
            str: The observation artifact's path, empty when none was written.
        """
        from .baseline import read_bringup_log, server_child_elapsed_sec

        read = read_bringup_log(server_log)
        verdict = observe_bringup(
            server_log=read.text,
            server_elapsed_sec=server_child_elapsed_sec(read.text),
            wrapper_stderr=wrapper_stderr,
            wrapper_stdout=wrapper_stdout,
            session_dir=self.session_dir,
        )
        return self._write(verdict.observation, output_dir)

    def _write(self, observation: BootObservation, output_dir: Path) -> str:
        """Persist one observation under the session's bring-up reports.

        Args:
            observation: The observation to record.
            output_dir: The round slot the artifact is indexed under.

        Returns:
            str: The artifact path, empty when the executor has no session.
        """
        from .baseline import open_bringup_attempt

        if self.session_dir is None:
            return ""
        return write_boot_observation(
            observation,
            session_dir=self.session_dir,
            output_dir=output_dir,
            attempt=open_bringup_attempt(output_dir),
        )


__all__ = [
    "DEFAULT_TIMEOUT_SEC",
    "PROBE_CEILING",
    "BootProbeExecutor",
    "probe_would_inform",
]
