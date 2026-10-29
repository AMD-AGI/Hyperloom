# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Attempt setup, server-argv preflight and the measured round's result of ``BenchmarkRunExecutor``."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom_common.env import is_truthy
from hyperloom_common.jsonio import read_json
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import agentx_active
from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import BaselineInvocation
from ....measurement.runtime_findings import persist_runtime_findings, scan_server_log
from .._grid_runner import sanitize_result_dir, sanitize_script_name
from .._launch_evidence import RequestedLaunch, build_launch_evidence, persist_launch_evidence
from .._subprocess_kill import DETOKENIZER_STALL_RETURNCODE, clear_server_ready_stamp
from .._workload_envs import (
    VLLM_SOURCE_ROOT_ENVS,
    BenchmarkTarget,
    FrameworkScriptMismatchError,
    LaunchLayer,
    MaterializeOptions,
    materialize_benchmark_config,
)
from ..benchmark_result import (
    extract_benchmark_measurement,
    harvest_leaked_artifacts,
    select_run_workspace,
    snapshot_workspaces,
)
from ..magpie_agentx import ServerArgvInvalidError
from .bringup import _boot_failure, open_bringup_attempt, read_bringup_log, server_child_elapsed_sec
from .records import (
    _BENCHMARK_MODE_ATTR,
    BenchmarkAttempt,
    _BenchmarkRound,
    _BenchProcess,
    _round_post_ready_sec,
    _should_establish_quality_ref,
    _stopped_round_result,
)
from .rounds import _BENCHMARK_REPORT_NAME

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

    Args:
        extra_envs: The round's environment, whose ``AITER_CONFIG_*`` values decide which
            branch of aiter's resolution each table takes.
    """
    csv_envs = {
        str(key): str(value)
        for key, value in (extra_envs or {}).items()
        if str(key).startswith("AITER_CONFIG_") and str(value).strip()
    }
    from .._aiter_jit import prepare_serving_so_for_csvs

    try:
        outcome = await asyncio.to_thread(prepare_serving_so_for_csvs, csv_envs)
    except OSError as exc:
        # A jit directory this cannot read is not a reason to lose the measurement.
        log.warning("baseline_executor: aiter serving .so preflight failed: %s", exc)
        return
    if isinstance(outcome, dict) and outcome.get("action") == "invalidate":
        log.info(
            "baseline_executor: the CSVs this boot loads outran the serving .so; dropped %d module(s) for rebuild",
            len(outcome.get("removed") or []),
        )


def _attach_baseline_launch_evidence(
    result: dict[str, Any],
    *,
    config_path: Path,
    output_dir: Path,
    framework: str,
) -> None:
    """Persist the declared and observed server identity for a baseline result."""
    from .._grid_runner import _measurement_server_log_path

    workspace = Path(str(result.get("workspace") or "")) if result.get("workspace") else None
    actual_log = _measurement_server_log_path(output_dir / "server.log", workspace, slot=output_dir)
    result["server_log_path"] = actual_log or ""
    if result.get("benchmark_backend") == "magpie_agentx":
        from hyperloom.inference_optimizer.agentx.report import native_launch_evidence

        evidence = native_launch_evidence(result, config_path, actual_server_log=actual_log)
        result["launch_evidence"] = evidence
        result["launch_evidence_path"] = persist_launch_evidence(evidence, slot=output_dir)
        return
    evidence = build_launch_evidence(
        RequestedLaunch(config_path, framework), actual_server_log=actual_log, slot=output_dir
    )
    result["launch_evidence"] = evidence
    result["launch_evidence_path"] = persist_launch_evidence(evidence, slot=output_dir)
    declared_env = evidence["requested_server_env"].keys() - set(VLLM_SOURCE_ROOT_ENVS)
    report = scan_server_log(actual_log, evidence["framework"], declared_env=declared_env)
    persist_runtime_findings(report, slot=output_dir)


class _BenchmarkLaunchMixin:
    """Attempt setup, server-argv preflight and the measured round's result of a benchmark run."""

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
        config_path = Path(params.get("config_path") or self.default_config_path or self._resolve_default_config())
        if not config_path.exists():
            raise FileNotFoundError(f"baseline config not found: {config_path}")
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
            return {
                "status": "failed",
                "error_class": "bad_param",
                "error": str(exc),
                "output_dir": str(output_dir),
            }
        base_extra_envs = dict(params.get("extra_envs") or {})
        if not self.runs_accuracy_eval:
            base_extra_envs["RUN_EVAL"] = "false"
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
                launch=LaunchLayer(
                    extra_server_args=server_args,
                    extra_envs=base_extra_envs,
                    remove_args=params.get("remove_args"),
                    unset_envs=params.get("unset_envs"),
                    args_mode=str(params.get("args_mode") or "append"),
                ),
                target=BenchmarkTarget(
                    model_path=resolved_model,
                    # gpu_type: task.params > $GPU_TYPE (cli.py canonicalizes mi325x->mi300x).
                    gpu_type=str(params.get("gpu_type") or "").strip().lower()
                    or os.environ.get("GPU_TYPE", "").strip().lower(),
                    benchmark_script=override_script,
                ),
                options=MaterializeOptions(
                    # Only a genuine ``baseline`` task may establish/overwrite the quality reference;
                    # ``replay_warm_recipe`` reuses this executor but must compare against the pure baseline reference
                    # rather than redefine it.
                    establish_quality_ref=_should_establish_quality_ref(getattr(ctx.task, "kind", ""), params),
                    drop_moe_runner_backend=force_drop_moe_runner_backend,
                    flydsl_source_dirs=is_truthy(params.get("flydsl_source_dirs")),
                    agentx_mode=agentx_active(benchmark_mode=getattr(shared_state, _BENCHMARK_MODE_ATTR, "")),
                    grading=getattr(shared_state, "grading", None),
                ),
            )
        except (FrameworkScriptMismatchError, ServerArgvInvalidError) as exc:
            return {
                "status": "failed",
                "error_class": getattr(exc, "error_class", "framework_script_mismatch"),
                "error": str(exc),
                "output_dir": str(output_dir),
            }
        from ..magpie_agentx import activate_runtime

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

    def _preflight_server_argv(self, round_: _BenchmarkRound, attempt: int) -> dict[str, Any] | None:
        """Ask the installed framework's parser about this round's argv.

        Args:
            round_: The round about to launch; its capture_meta is echoed onto a refusal.
            attempt: This attempt's index within the round slot.

        Returns:
            dict | None: A terminal failure result when the argv is refused;
            ``None`` when the round may proceed -- including every unavailable
            verdict, which must never cost a round.
        """
        from ..magpie_agentx import native_config

        if native_config(round_.config_path):
            return None
        from ....bringup import (
            ARGV_INVALID,
            argv_invalid_observation,
            check_server_argv,
            write_boot_observation,
        )
        from ....bringup.argv_preflight import OK, PARSED_AFTER_DROP, UNAVAILABLE, ArgvRepair
        from .._server_argv import config_launch_env, config_server_argv, reseal_config_argv

        sealed = config_server_argv(round_.config_path)
        if not sealed.tokenized or not sealed.argv:
            return None

        effective_launch_env = config_launch_env(round_.config_path, round_.env)
        probe_argv = sealed.argv
        if round_.framework.strip().lower() == "vllm" and effective_launch_env.get("PROFILE") == "1":
            probe_argv = (
                "--profiler-config.profiler",
                "torch",
                "--profiler-config.torch_profiler_dir",
                str(round_.output_dir / "torch_trace"),
                *probe_argv,
            )

        # ``_resolve_shared_state`` is typed loosely and callers inject partial
        # doubles, so the round's repair ledger may not be present at all.
        enablement = getattr(self._resolve_shared_state(), "enablement", None)
        spent: list[str] = enablement.argv_repairs if enablement is not None else []
        verdict = check_server_argv(
            framework=round_.framework,
            argv=probe_argv,
            text=sealed.text,
            launch_env=effective_launch_env,
            repair=ArgvRepair(
                digest=sealed.digest,
                spent=spent,
            ),
        )

        if verdict.status == UNAVAILABLE:
            log.info(
                "argv preflight unavailable (%s): %s; the launch remains the only verdict",
                verdict.reason,
                verdict.detail,
            )
            return None
        if verdict.status == OK:
            if verdict.reason == PARSED_AFTER_DROP:
                reseal_config_argv(round_.config_path, verdict.text)
                if verdict.repaired_digest:
                    spent.append(verdict.repaired_digest)
                round_.capture_meta["server_argv_dropped"] = list(verdict.dropped)
            return None

        observation = argv_invalid_observation(verdict, session_dir=self.session_dir)
        round_.capture_meta["boot_observation_path"] = write_boot_observation(
            observation,
            session_dir=self.session_dir,
            output_dir=round_.output_dir,
            attempt=attempt,
        )
        excerpt = observation.excerpt
        log.error(
            "argv preflight: %s refused the server argv (%s); not launching",
            round_.framework or "the framework",
            verdict.reason,
        )
        return {
            "status": "failed",
            "error_class": ARGV_INVALID,
            "error": f"{round_.framework or 'framework'} rejected the server argv ({verdict.reason}): {verdict.detail}",
            "output_dir": str(round_.output_dir),
            "enablement_launch_log": excerpt.text if excerpt is not None else "",
            "server_argv": list(verdict.argv),
            **round_.capture_meta,
        }

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
            # Merge the reference base UNDER the per-task args (last-wins) so a multi-node per-round restart
            # carries the same reference flags the single-node materialized YAML does.
            from hyperloom.inference_optimizer.grid_server_args import merge_server_args
            from .._workload_envs import resolve_reference_launch

            ref_args, ref_envs, _ = resolve_reference_launch()
            # Fold in the operator ``--server-args`` (``INFERENCE_OPTIMIZER_SERVER_ARGS``).
            operator_args = os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "").strip()
            base = ref_args
            if operator_args:
                base = merge_server_args(base, operator_args) if base else operator_args
            # Base on ``server_args`` (carries the one-shot cuda-graph eager-fallback flag when armed) so the MN
            # per-round restart keeps that fallback too.
            env = {str(key): str(value) for key, value in ref_envs.items()}
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

    async def _measured_round_result(self, round_: _BenchmarkRound, round_warnings: list[str]) -> dict[str, Any]:
        from .._multi_node_env import is_multi_node

        workspaces_before = snapshot_workspaces(round_.output_dir)
        started_unix = time.time()
        # Anchor the Magpie parent process cwd to the per-task output_dir.
        round_.output_dir.mkdir(parents=True, exist_ok=True)
        # A reused output_dir may still hold a previous attempt's server.log, whose terminal init markers would be
        # read as this attempt's, and the ready stamp beside it would make this attempt's boot look like it never
        # happened.
        server_log = round_.output_dir / "server.log"
        attempt_index = open_bringup_attempt(round_.output_dir)
        clear_server_ready_stamp(str(server_log))
        refusal = self._preflight_server_argv(round_, attempt_index)
        if refusal is not None:
            return refusal
        ready = bool(round_.server_already_ready or round_.ctx_extra.get("server_already_ready") or is_multi_node())
        if self.benchmark_watchdog and not ready:
            from .._aiter_jit import sweep_stale_aiter_locks_if_dead

            if deleted := (await asyncio.to_thread(sweep_stale_aiter_locks_if_dead)).get("deleted"):
                log.warning("baseline_executor: reaped %d orphaned aiter JIT lock(s) before server launch", deleted)
        try:
            finished = await self._launch_benchmark(round_, started_unix, server_ready=ready)
        except subprocess.TimeoutExpired as exc:
            # A reaped timeout carries no wrapper streams; the server log is the whole of the evidence.
            await asyncio.to_thread(self._record_bringup, round_, attempt_index, workspaces_before)
            reaped = {"error_class": "timeout", "error": f"baseline benchmark exceeded {round_.timeout_sec}s: {exc}"}
            return self._reaped_round_failure(
                round_.output_dir, workspaces_before, started_unix, {**reaped, **round_.capture_meta}
            )
        server_log_text = await asyncio.to_thread(
            self._record_bringup, round_, attempt_index, workspaces_before, finished.stderr, finished.stdout
        )
        boot_ref = round_.capture_meta["boot_observation_path"]
        finished = finished._replace(server_log_text=server_log_text, boot_observation_ref=boot_ref)
        stopped = _stopped_round_result(
            finished.returncode,
            round_label="measured round",
            runtime_sec=finished.runtime_sec,
            output_dir=round_.output_dir,
            capture_meta=round_.capture_meta,
        )
        if stopped is not None:
            return stopped
        # Detokenizer-stall watchdog reap: the server came up healthy but went silent for the stall grace window
        # (hung engine / wedged detokenizer).
        if finished.returncode == DETOKENIZER_STALL_RETURNCODE:
            log.warning(
                "baseline_executor: detokenizer-stall watchdog reaped run "
                "(server ready but log went silent); error_class=detokenizer_stall."
            )
            reaped = {
                "error_class": "detokenizer_stall",
                "returncode": finished.returncode,
                "error": _DETOKENIZER_STALL_ERROR,
                "subprocess_runtime_sec": round(finished.runtime_sec, 2),
            }
            return self._reaped_round_failure(
                round_.output_dir, workspaces_before, started_unix, {**reaped, **round_.capture_meta}
            )
        # Read the server log back before the harvest below: harvesting can copy a leaked ``server.log`` into
        # ``output_dir`` and change what this classification sees.
        boot = _boot_failure(finished, server_log)
        workspace = select_run_workspace(round_.output_dir, known_before=workspaces_before)
        return await self._round_result_from_workspace(
            round_, finished._replace(workspace=workspace), boot, round_warnings
        )

    def _record_bringup(
        self,
        round_: _BenchmarkRound,
        attempt_index: int,
        workspaces_before: Any,
        wrapper_stderr: str = "",
        wrapper_stdout: str = "",
    ) -> str:
        from ....bringup import observe_bringup, write_boot_observation

        # Magpie writes the server log into the workspace it creates, not to $SERVER_LOG.
        workspace = select_run_workspace(round_.output_dir, known_before=workspaces_before)
        log_path = workspace / "server.log" if workspace is not None else round_.output_dir / "server.log"
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

    @staticmethod
    def _reaped_round_failure(
        output_dir: Path,
        workspaces_before: Any,
        started_unix: float,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        destination = select_run_workspace(output_dir, known_before=workspaces_before) or output_dir
        harvested = harvest_leaked_artifacts(destination, subprocess_started_unix=started_unix)
        return {
            "status": "failed",
            "output_dir": str(output_dir),
            "harvested_artifacts": [str(dst) for _, dst in harvested],
            "nonfatal_warnings": [f"harvested_leaked_artifact:{src}" for src, _ in harvested],
            **fields,
        }

    async def _round_result_from_workspace(
        self,
        round_: _BenchmarkRound,
        finished: _BenchProcess,
        boot: tuple[str, str] | None,
        round_warnings: list[str],
    ) -> dict[str, Any]:
        workspace = finished.workspace
        # Always-on artifact harvest: copy wrapper-side leaks into the task workspace so failure-path diagnostics
        # survive; mtime gating rejects stale prior-run leaks.
        harvested = harvest_leaked_artifacts(
            workspace if workspace is not None else round_.output_dir,
            subprocess_started_unix=finished.started_unix,
        )
        if harvested:
            log.info(
                "baseline_executor: harvested %d leaked artifact(s) into workspace: %s",
                len(harvested),
                ", ".join(str(src.name) for src, _ in harvested),
            )
        if workspace is None:
            return self._no_workspace_failure(round_, finished, boot, harvested)
        report_path = workspace / _BENCHMARK_REPORT_NAME
        report = read_json(report_path, None, require_dict=True)
        measurement = extract_benchmark_measurement(
            report,
            config_path=round_.config_path,
            workspace=workspace,
            subprocess_started_unix=finished.started_unix,
        )
        warnings = round_warnings + list(measurement.pop("nonfatal_warnings", []) or [])
        warnings.extend(f"harvested_leaked_artifact:{src}" for src, _ in harvested)
        failure = self._failed_round_result(round_, finished, boot, measurement, warnings)
        if failure is not None:
            return failure
        result = {
            "status": "succeeded",
            **measurement,
            "nonfatal_warnings": warnings,
            "returncode": finished.returncode,
            "output_dir": str(round_.output_dir),
            "result_dir": str(round_.result_dir),
            "report_path": str(report_path) if report_path.exists() else None,
            "workspace": str(workspace),
            # Materialized YAML for THIS baseline.
            "materialized_config": str(round_.materialized_config_path),
            # Magpie subprocess wall-clock (success path only).
            "subprocess_runtime_sec": round(finished.runtime_sec, 2),
            # That wall-clock after the boot: the benchmark, and the accuracy eval when this round ran one.
            "post_ready_runtime_sec": _round_post_ready_sec(
                round_.watchdog_server_log,
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
        self._stamp_round_quality(round_, result, report, warnings)
        return result
