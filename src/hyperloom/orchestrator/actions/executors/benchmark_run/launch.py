# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Attempt setup and the measured round's result of ``BenchmarkRunExecutor``."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom_common.env import is_truthy
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import agentx_active
from hyperloom.common.env_safety import redact_secret_values
from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import BaselineInvocation
from hyperloom.inference_optimizer.trace.task_progress import heartbeat_while_output_flows
from inference_suite.contracts import FailedStep, LifecycleMode
from .._grid_runner import sanitize_result_dir, sanitize_script_name
from .._launch_evidence import RequestedLaunch, build_launch_evidence, persist_measurement_evidence
from .._workload_envs import (
    BenchmarkTarget,
    FrameworkScriptMismatchError,
    LaunchLayer,
    MaterializeOptions,
    materialize_benchmark_config,
)
from ..benchmark_result import (
    extract_benchmark_measurement,
)
from .._failed_step import failed_step_error_class
from .._round_folder import RoundFolder, round_report_path
from .._suite.rounds import SuiteRound, SuiteRoundRefused, SuiteRoute, deadline_unix, round_lifecycle, run_suite_round
from ..agentx_point import ServerArgvInvalidError
from .bringup import open_bringup_attempt, read_bringup_log, server_child_elapsed_sec
from .records import (
    _BENCHMARK_MODE_ATTR,
    BenchmarkAttempt,
    _BenchmarkRound,
    _BenchProcess,
    _round_post_ready_sec,
    _should_establish_quality_ref,
)

if TYPE_CHECKING:
    from ....loop.sub_agent_runner import RunnerContext

log = logging.getLogger(__name__)

#: What a round reaped by the detokenizer-stall watchdog reports: the server came up, then went silent.
_DETOKENIZER_STALL_ERROR = (
    "server reported ready but emitted no log output (hung engine / detokenizer stall); "
    "reaped by the detokenizer-stall watchdog. See server.log."
)


async def _prepare_aiter_serving_so(extra_envs: dict[str, Any]) -> None:
    """Rebuild the serving ``.so`` before boot when the CSVs it loads name kernels it lacks.

    sglang starts against whatever ``get_config_file`` resolves each ``AITER_CONFIG_*`` to,
    and a kernel the compiled module never registered raises from inside graph capture. The
    classifier downstream names that ``aiter_jit_registry_mismatch`` but cannot undo it, so
    the round is simply lost -- and PRELUDE's first measurement and every FRAMEWORK variant
    boot through here. The GEMM integrate lane already runs this check before its own boots.

    It is not conditioned on the round carrying a tuned CSV: a variable the round leaves
    unset is exactly the case where aiter merges the model overlays, which is where the
    kernel that fails the boot comes from.

    Nothing restores what this drops, so it is parked on the cache's own filesystem (the
    preflight's default) rather than copied into the run directory, which sits on shared NFS.
    It runs under this host's preflight lock (:mod:`.._aiter_serving_once`), waiting for another
    session's preflight or the GEMM tuning lane, and builds nothing: the boot's servers build what
    it parked, at their first call of it.

    The decision is logged as a warning, the level ``hyperloom optimize`` logs at without ``-v``:
    a move, every time; that nothing moved, once per process.

    Args:
        extra_envs: The round's environment, whose ``AITER_CONFIG_*`` values decide which
            branch of aiter's resolution each table takes.
    """
    csv_envs = {
        str(key): str(value)
        for key, value in (extra_envs or {}).items()
        if str(key).startswith("AITER_CONFIG_") and str(value).strip()
    }
    environ = {**os.environ, **{str(key): str(value) for key, value in (extra_envs or {}).items()}}
    from .._aiter_serving_once import prepare_serving_so_once

    try:
        preflight = await asyncio.to_thread(prepare_serving_so_once, csv_envs, environ)
    except OSError as exc:
        # A jit directory this cannot read is not a reason to lose the measurement.
        log.warning("baseline_executor: aiter serving .so preflight failed: %s", exc)
        return
    action, jit_dir = preflight.outcome.get("action"), preflight.outcome.get("jit_dir")
    if action == "skip" and jit_dir not in _COVERED_SAID:
        _COVERED_SAID.add(jit_dir)
        log.warning("baseline_executor: %s", preflight.summary())
    elif action == "invalidate":
        log.warning("baseline_executor: %s", preflight.summary())


#: The jit directories this process has said once already that the serving ``.so`` covers its boots' tables.
_COVERED_SAID: set[str | None] = set()


def _attach_baseline_launch_evidence(
    result: dict[str, Any],
    *,
    config_path: Path,
    output_dir: Path,
    framework: str,
) -> None:
    """Persist the declared and observed server identity, and its runtime findings, for a baseline result."""
    from .._grid_runner import _measurement_server_log_path

    workspace = Path(str(result.get("workspace") or "")) if result.get("workspace") else None
    actual_log = _measurement_server_log_path(output_dir / "server.log", workspace, slot=output_dir)
    result["server_log_path"] = actual_log or ""
    evidence = build_launch_evidence(
        RequestedLaunch(config_path, framework), actual_server_log=actual_log, slot=output_dir
    )
    result["launch_evidence"] = evidence
    result["launch_evidence_path"] = persist_measurement_evidence(
        evidence, actual_server_log=actual_log, slot=output_dir
    )


def _attempt_failure(error_class: str, exc: Exception, output_dir: Path) -> dict[str, Any]:
    """The failed result of an attempt refused before any round ran."""
    return {
        "status": "failed",
        "error_class": error_class,
        "error": str(exc),
        "output_dir": str(output_dir),
    }


def _attempt_launch_layer(params: dict[str, Any], server_args: str, extra_envs: dict[str, Any]) -> LaunchLayer:
    """The server args, envs and removals the attempt layers onto the benchmark config."""
    return LaunchLayer(
        extra_server_args=server_args,
        extra_envs=extra_envs,
        remove_args=params.get("remove_args"),
        unset_envs=params.get("unset_envs"),
        args_mode=str(params.get("args_mode") or "append"),
    )


def _attempt_target(params: dict[str, Any], resolved_model: str, override_script: str | None) -> BenchmarkTarget:
    """The model, GPU type and benchmark script the attempt's config targets."""
    return BenchmarkTarget(
        model_path=resolved_model,
        # gpu_type: task.params > $GPU_TYPE.
        gpu_type=str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower(),
        benchmark_script=override_script,
    )


def _attempt_options(ctx: RunnerContext, shared_state: Any, drop_moe_runner_backend: bool) -> MaterializeOptions:
    """How the attempt's config is materialized: quality-reference role, MoE-runner drop, FlyDSL, AgentX, grading."""
    params = ctx.task.params
    return MaterializeOptions(
        # Only a genuine ``baseline`` task may establish/overwrite the quality reference;
        # ``replay_warm_recipe`` reuses this executor but must compare against the pure baseline reference
        # rather than redefine it.
        establish_quality_ref=_should_establish_quality_ref(getattr(ctx.task, "kind", ""), params),
        drop_moe_runner_backend=drop_moe_runner_backend,
        flydsl_source_dirs=is_truthy(params.get("flydsl_source_dirs")),
        agentx_mode=agentx_active(benchmark_mode=getattr(shared_state, _BENCHMARK_MODE_ATTR, "")),
        grading=getattr(shared_state, "grading", None),
    )


def _suite_round_failure(
    round_: _BenchmarkRound, finished: _BenchProcess, measurement: dict[str, Any], warnings: list[str]
) -> dict[str, Any]:
    """The failed result of an inference_suite round: the step it stopped at, or a load that measured nothing."""
    step = measurement.get("failed_step")
    if step is not None:
        error_class, error = failed_step_error_class(FailedStep(step)), measurement.get("failure_detail") or step
    else:
        error_class, error = "invalid_measurement", "the round report holds no positive throughput and served requests"
    report_path = round_report_path(finished.workspace)
    return {
        "status": "failed",
        "error_class": error_class,
        "error": redact_secret_values(error),
        "returncode": finished.returncode,
        "run_status": measurement.get("run_status"),
        "output_dir": str(round_.output_dir),
        "workspace": str(finished.workspace),
        "report_path": str(report_path),
        "subprocess_runtime_sec": round(finished.runtime_sec, 2),
        "boot_observation_path": finished.boot_observation_ref,
        "nonfatal_warnings": warnings,
        **round_.capture_meta,
    }


class _BenchmarkLaunchMixin:
    """Attempt setup and the measured round's result of a benchmark run."""

    async def _run_once(
        self,
        ctx: RunnerContext,
        *,
        force_drop_moe_runner_backend: bool = False,
        recorder: Any = None,
        run_index: int = 0,
    ) -> dict[str, Any]:
        """Resolve what this baseline attempt runs with, write its config, and run its rounds."""
        params = ctx.task.params
        config_path = self._attempt_config_path(params)
        shared_state = ctx.extra.get("shared_state") or self.shared_state
        fw, server_args = self._settle_attempt_server_args(
            params,
            shared_state,
            drop_moe_runner_backend=force_drop_moe_runner_backend,
        )
        output_dir = self._resolve_workspace(ctx, "baseline")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Orchestration-supplied script + result_dir overrides.
        try:
            override_script = sanitize_script_name(params.get("benchmark_script"))
            override_result_dir = sanitize_result_dir(params.get("result_dir"))
        except ValueError as exc:
            return _attempt_failure("bad_param", exc, output_dir)
        base_extra_envs = self._attempt_extra_envs(params)
        # Model path: unified resolver (params → $MODEL_PATH → SharedState), then serving-path normalization
        # (HL_MODEL_BASE / HF cache).
        resolved_model = resolve_session_model_path(
            params=params,
            state_model_path=str(getattr(shared_state, "model_path", "") or ""),
        )
        await _prepare_aiter_serving_so(base_extra_envs)
        try:
            config_path = materialize_benchmark_config(
                config_path,
                output_dir,
                launch=_attempt_launch_layer(params, server_args, base_extra_envs),
                target=_attempt_target(params, resolved_model, override_script),
                options=_attempt_options(ctx, shared_state, force_drop_moe_runner_backend),
            )
        except (FrameworkScriptMismatchError, ServerArgvInvalidError) as exc:
            return _attempt_failure(getattr(exc, "error_class", "framework_script_mismatch"), exc, output_dir)
        from ..agentx_point import activate_runtime

        activate_runtime(config_path, params.get("runtime_override") or {})
        # Report the invocation now, with the config final and the server not
        # yet booted. This frame is the only one that knows the args as a fact:
        # the one-shot eager fallback and the MoE-runner drop have both had
        # their say by here, and after the launch the same answer can only be
        # guessed at by parsing the server's own log back.
        if recorder is not None:
            recorder.record_invocation(
                run_index=run_index,
                invocation=BaselineInvocation(
                    framework_args=server_args,
                    extra_envs=base_extra_envs,
                    config_path=config_path,
                    framework=fw,
                    model_path=resolved_model,
                    args_mode=str(params.get("args_mode") or "append"),
                ),
            )
        return await self._run_attempt_rounds(
            BenchmarkAttempt(
                timeout_sec=self._resolve_timeout(params),
                override_result_dir=override_result_dir,
                resolved_model=resolved_model,
                materialized_config_path=config_path,
                inferencex_path=os.environ.get("INFERENCEX_PATH", "").strip(),
                effective_extra_server_args=server_args,
                params=params,
                ctx=ctx,
                recorder=recorder,
                run_index=run_index,
            ),
            output_dir,
        )

    def _attempt_config_path(self, params: dict[str, Any]) -> Path:
        """The benchmark config the attempt starts from: the task's, else the executor's default."""
        config_path = Path(params.get("config_path") or self.default_config_path or self._resolve_default_config())
        if not config_path.exists():
            raise FileNotFoundError(f"baseline config not found: {config_path}")
        return config_path

    def _attempt_extra_envs(self, params: dict[str, Any]) -> dict[str, Any]:
        """The attempt's base environment: the task's envs, with the accuracy eval off when this executor skips it."""
        base_extra_envs = dict(params.get("extra_envs") or {})
        if not self.runs_accuracy_eval:
            base_extra_envs["RUN_EVAL"] = "false"
        return base_extra_envs

    async def _restart_multi_node_server(
        self,
        round_: _BenchmarkRound,
        *,
        server_args: str,
        resolved_model: str,
    ) -> dict[str, Any] | None:
        from .._multi_node_server_lifecycle import RoundServer, ServerRestartFailed, restart_server_for_round

        if round_.ctx_extra.get("mn_round_restarted"):
            return None
        try:
            # Merge the session base UNDER the per-task args (last-wins) so a multi-node per-round restart
            # carries the same base flags the single-node materialized YAML does.
            from hyperloom.inference_optimizer.grid_server_args import merge_server_args
            from hyperloom.orchestrator.state.base_launch import session_server_base

            server_base = session_server_base()
            # Fold in the operator ``--server-args`` (``INFERENCE_OPTIMIZER_SERVER_ARGS``).
            operator_args = os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "").strip()
            base = server_base.args
            if operator_args:
                base = merge_server_args(base, operator_args) if base else operator_args
            # Base on ``server_args`` (carries the one-shot cuda-graph eager-fallback flag when armed) so the MN
            # per-round restart keeps that fallback too.
            env = dict(server_base.env)
            # PD knobs auto-resolved by the helper from $PD_* env, falling back to state.json.
            await restart_server_for_round(
                RoundServer.from_env(
                    merge_server_args(base, server_args) if base else server_args, resolved_model or None
                ),
                extra_env=env or None,
            )
        except ServerRestartFailed as exc:
            return {
                "status": "failed",
                "error_class": "mn_server_restart_failed",
                "error": str(exc),
                "output_dir": str(round_.output_dir),
            }
        return None

    async def _suite_round_result(
        self, round_: _BenchmarkRound, route: SuiteRoute, *, warmup: bool, warnings: Sequence[str] = ()
    ) -> dict[str, Any]:
        """Run ``round_`` on inference_suite and read the run_dir it wrote into the round's result.

        ``warnings`` are what the passes before this round left for its result to carry.
        """
        # A lone round owns its server; a cold-start warmup boots the one the rounds after it reuse.
        mode = (
            LifecycleMode.REUSE if round_.server_already_ready else LifecycleMode.BOOT if warmup else LifecycleMode.OWN
        )
        request = SuiteRound(
            config_path=round_.config_path,
            round_dir=round_.output_dir,
            lifecycle=round_lifecycle(round_.output_dir, mode),
            warmup=warmup,
            deadline_unix=deadline_unix(round_.session_deadline_sec),
            serving_lease=round_.serving_lease,
        )
        attempt_index = open_bringup_attempt(round_.output_dir)
        started_unix = time.time()
        try:
            async with heartbeat_while_output_flows(unit="baseline_round", label="benchmark") as activity:
                ran = await asyncio.to_thread(run_suite_round, route, request, on_poll=activity.note)
        except SuiteRoundRefused as exc:
            log.error("baseline_executor: round refused before it ran: %s", exc)
            return _attempt_failure(exc.error_class, exc, round_.output_dir) | round_.capture_meta
        layout = ran.layout
        if ran.report.failed_step is FailedStep.FLAG_UNKNOWN:
            self._record_argv_refusal(round_, attempt_index, ran.report.failure_detail or "")
        else:
            await asyncio.to_thread(
                self._observe_bringup, round_, attempt_index, layout.server_log, ran.report.failure_detail or ""
            )
        RoundFolder.read(layout.root).write_gpu_metrics(round_.output_dir)
        measurement = extract_benchmark_measurement(None, workspace=layout.root, subprocess_started_unix=started_unix)
        finished = _BenchProcess(
            returncode=0 if ran.report.failed_step is None else 1,
            stdout="",
            stderr="",
            runtime_sec=max(0.0, time.time() - started_unix),
            started_unix=started_unix,
            boot_observation_ref=round_.capture_meta["boot_observation_path"],
            workspace=layout.root,
        )
        # The run_dir is where this round's server log, ready stamp and eval results are.
        ran_round = round_._replace(result_dir=layout.root, watchdog_server_log=str(layout.server_log))
        warnings = [*warnings, *(measurement.pop("nonfatal_warnings", []) or [])]
        if ran.report.failed_step is not None or not self._measurement_succeeded(measurement):
            return _suite_round_failure(ran_round, finished, measurement, warnings)
        result = await self._succeeded_round_result(ran_round, finished, measurement, warnings)
        self._stamp_round_quality(ran_round, result, measurement, warnings)
        return result

    def _record_argv_refusal(self, round_: _BenchmarkRound, attempt_index: int, detail: str) -> None:
        """Record the parser's refusal at the parse rung, where enablement reads it as a terminal no patch repairs."""
        from ....bringup import argv_invalid_observation, write_boot_observation

        round_.capture_meta["boot_observation_path"] = write_boot_observation(
            argv_invalid_observation(detail, session_dir=self.session_dir),
            session_dir=self.session_dir,
            output_dir=round_.output_dir,
            attempt=attempt_index,
        )
        round_.capture_meta["boot_observation_degraded"] = ""

    def _observe_bringup(
        self,
        round_: _BenchmarkRound,
        attempt_index: int,
        log_path: Path,
        wrapper_stderr: str = "",
        wrapper_stdout: str = "",
    ) -> str:
        """Classify the round's bring-up from its server log, record the observation, and return the log read."""
        from ....bringup import observe_bringup, write_boot_observation

        read = read_bringup_log(log_path)
        verdict = observe_bringup(
            server_log=read.text,
            server_elapsed_sec=server_child_elapsed_sec(read.text),
            wrapper_stderr=wrapper_stderr,
            wrapper_stdout=wrapper_stdout,
            session_dir=self.session_dir,
        )
        round_.capture_meta["boot_observation_path"] = write_boot_observation(
            verdict.observation,
            session_dir=self.session_dir,
            output_dir=round_.output_dir,
            attempt=attempt_index,
        )
        round_.capture_meta["boot_observation_degraded"] = read.degraded
        return read.text

    async def _succeeded_round_result(
        self,
        round_: _BenchmarkRound,
        finished: _BenchProcess,
        measurement: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any]:
        """The result of a round that measured, with the launch evidence of the server it measured attached."""
        report_path = round_report_path(finished.workspace)
        result = {
            "status": "succeeded",
            **measurement,
            "nonfatal_warnings": warnings,
            "returncode": finished.returncode,
            "output_dir": str(round_.output_dir),
            "result_dir": str(round_.result_dir),
            "report_path": str(report_path) if report_path.exists() else None,
            "workspace": str(finished.workspace),
            # Materialized YAML for THIS baseline.
            "materialized_config": str(round_.materialized_config_path),
            # The round's wall-clock (success path only).
            "subprocess_runtime_sec": round(finished.runtime_sec, 2),
            # That wall-clock after the boot: the benchmark, and the accuracy eval when this round ran one.
            "post_ready_runtime_sec": _round_post_ready_sec(
                None if round_.watchdog_server_log is None else Path(round_.watchdog_server_log).parent,
                started_unix=finished.started_unix,
                runtime_sec=finished.runtime_sec,
            ),
            # Authoritative (materialized-config) view of whether the serving lm-eval ran this run.
            "run_eval_disabled": bool(round_.run_eval_disabled),
            # Where this round's boot observation landed, and whether the log it was classified from could be read.
            "boot_observation_path": finished.boot_observation_ref,
            "boot_observation_degraded": round_.capture_meta.get("boot_observation_degraded", ""),
        }
        await asyncio.to_thread(
            _attach_baseline_launch_evidence,
            result,
            config_path=round_.materialized_config_path,
            output_dir=round_.output_dir,
            framework=round_.framework,
        )
        return result
