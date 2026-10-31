# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Round guards, cold-start passes and failed-round results of ``BenchmarkRunExecutor``."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from hyperloom.common.env_safety import redact_secret_values
from hyperloom.common.gain_math import incremental_gain_pct
from hyperloom.common.perf_metric import is_agentx_mode
from hyperloom_common.timeutil import now_iso
from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import (
    ROUND_ACCURACY,
    ROUND_MEASURE,
    ROUND_WARMUP,
    BaselineRound,
)
from hyperloom.inference_optimizer.grid_server_args import remove_moe_runner_backend_arg
from hyperloom.inference_optimizer.trace.task_progress import report_progress
from ....measurement.integrate_performance import assess_integrate_performance
from ....state.failure_evidence import (
    MAGPIE_NONZERO_INVALID_MEASUREMENT,
    SUBPROCESS_NONZERO,
    nonzero_output_signature,
)
from ...stop_attribution import SESSION_TIME_EXHAUSTED_CLASS, STOPPED_BY_THE_RUN
from .. import _server_lifecycle as _lifecycle
from .._accuracy_gate import post_ready_sec_without_eval
from ..benchmark_result import served_complete_protocol
from ..bypass_engine import lifecycle_meta_file, lifecycle_pid_file
from .bringup import _classify_subprocess_error, _disable_cuda_graph_flag, _with_cuda_graph_disabled
from .records import (
    _ACCURACY_ROUND_DIR,
    _BENCHMARK_MODE_ATTR,
    _MEASURE_ROUND_DIR,
    _WARMUP_ROUND_DIR,
    BenchmarkAttempt,
    _BenchmarkRound,
    _BenchProcess,
    _cold_anchor_from_warmup,
    _requested_framework,
    _unlaunched_round_result,
)
from .warm_patches import (
    _apply_warm_patches,
    _rollback_warm_kernel_apply_results,
    _stamp_warm_patch_outcome,
    _warm_patches_applied,
)

log = logging.getLogger(__name__)

#: The file Magpie writes its own measurement to, inside the workspace it creates.
_BENCHMARK_REPORT_NAME = "benchmark_report.json"


# Above this cold-start delta the measured round is unlikely to have settled either: the observed pathological case
# climbed 14,202 -> 19,374 -> 22,425 tok/s across three rounds of one unchanged config, i.e. +36% into round 2 and
# another +16% into round 3.
_COLD_START_DELTA_WARN_PCT = 25.0


class _BenchmarkRoundsMixin:
    """Round guards, cold-start passes and failed-round results of a benchmark run."""

    def _eager_fallback_armed(self, shared_state: Any | None = None) -> bool:
        """Peek the one-shot eager fallback flag WITHOUT consuming it."""
        state = self._resolve_shared_state(shared_state)
        return bool(getattr(state, "baseline_eager_fallback", False))

    def _consume_eager_fallback(self, shared_state: Any | None = None) -> bool:
        """Consume the one-shot cuda-graph eager fallback flag from SharedState."""
        state = self._resolve_shared_state(shared_state)
        if not getattr(state, "baseline_eager_fallback", False):
            return False
        state.baseline_eager_fallback = False
        state.save(self.session_dir)
        return True

    @staticmethod
    def _warn_on_cold_start_delta(
        result: dict[str, Any],
        warmup_tput: Any,
    ) -> None:
        """Warn when the measured round climbed far enough above the warm-up to suggest a server still ramping."""
        warm = float(warmup_tput or 0.0)
        measured = float(result.get("output_throughput") or 0.0)
        if measured <= 0 or (delta_pct := incremental_gain_pct(measured, warm)) is None:
            return
        if delta_pct <= _COLD_START_DELTA_WARN_PCT:
            return
        result.setdefault("nonfatal_warnings", [])
        result["nonfatal_warnings"].append("baseline_cold_start_delta_high")
        log.warning(
            "baseline_executor: measured round is %.1f%% above the warm-up round "
            "(%.1f -> %.1f tok/s); the server may still have been ramping, so the "
            "anchor every later gain is graded against may be low",
            delta_pct,
            warm,
            measured,
        )

    def _settle_attempt_server_args(
        self,
        params: dict[str, Any],
        shared_state: Any,
        *,
        drop_moe_runner_backend: bool,
    ) -> tuple[str, str]:
        # One-shot cuda-graph eager fallback: a prior baseline armed state.baseline_eager_fallback.
        server_args = str(params.get("extra_server_args") or "")
        fw = _requested_framework(params) or ""
        if not fw and self._eager_fallback_armed(shared_state):
            log.warning(
                "baseline_executor: eager fallback is armed but framework is "
                "unknown; leaving the one-shot armed (not consuming) so a "
                "later baseline with a known framework can apply it",
            )
        elif fw and self._consume_eager_fallback(shared_state):
            server_args = _with_cuda_graph_disabled(server_args, fw)
            log.warning(
                "baseline_executor: retrying with %s after a prior cuda-graph capture failure (framework=%s)",
                _disable_cuda_graph_flag(fw),
                fw,
            )
        if drop_moe_runner_backend:
            server_args = remove_moe_runner_backend_arg(server_args)
        if server_args or "extra_server_args" in params:
            # Keep the task envelope aligned with the materialized runtime so Roofline fingerprints record one-shot
            # eager fallback accurately.
            params["extra_server_args"] = server_args
        return fw, server_args

    def _save_warm_replay_pending(self, state: Any, **fields: Any) -> bool:
        state.warm_replay_pending = {**(state.warm_replay_pending or {}), **fields}
        try:
            state.save(self.session_dir)
        except OSError:
            log.warning("baseline_executor: warm replay pending state persist failed", exc_info=True)
            return False
        return True

    def _prepare_round_patches(
        self,
        params: dict[str, Any],
        output_dir: Path,
        shared_state: Any,
    ) -> list[dict[str, str]] | dict[str, Any]:

        def _persist_recipe_snapshot(tree_records: list[dict[str, Any]]) -> bool:
            if shared_state is None:
                return True
            primary = tree_records[0] if tree_records else {}
            return self._save_warm_replay_pending(
                shared_state,
                status="preparing_required_recipe",
                # The trees are the authority; the singular fields describe the primary one so a resumed legacy
                # reader still finds it.
                recipe_patch_trees=tree_records,
                recipe_patch_target=str(primary.get("root") or ""),
                recipe_patch_pre_sha=str(primary.get("pre_sha") or ""),
                recipe_patch_snapshot_manifest=primary.get("snapshot_manifest"),
            )

        # An overlay recording its own root needs no resolved target, so this only refuses when nothing names a tree
        # at all.
        if params.get("patches") and not any(
            str((entry or {}).get("framework_root") or "").strip()
            for entry in params["patches"]
            if isinstance(entry, dict)
        ):
            return {"status": "failed", "failure": "missing_target_repo", "patches": [], "applied": []}
        application = _apply_warm_patches(params, output_dir, before_mutation=_persist_recipe_snapshot)
        if isinstance(application, Mapping):
            if application.get("status") == "failed":
                return application
            if shared_state is not None:
                self._save_warm_replay_pending(
                    shared_state,
                    status="benchmarking",
                    recipe_patch_trees=list(application.get("trees") or []),
                    recipe_patch_target=str(application.get("target_repo") or ""),
                    recipe_patch_pre_sha=str(application.get("pre_sha") or ""),
                    recipe_patch_snapshot_manifest=application.get("snapshot_manifest"),
                    recipe_patch_statuses=list(application.get("patches") or []),
                )
        if applied := _warm_patches_applied(application):
            log.info(
                "baseline_executor: prepared %d warm-replay code patches: %s",
                len(applied),
                [patch["patch_file"] for patch in applied],
            )
        return application

    def _roll_back_failed_patches(
        self,
        patch_application: Mapping[str, Any],
        params: Mapping[str, Any],
        output_dir: Path,
        live_shared_state: Any,
    ) -> dict[str, Any]:
        kernel_rollback = _rollback_warm_kernel_apply_results(
            params.get("warm_kernel_apply_results"),
            params.get("warm_kernel_snapshots"),
        )
        recipe_rollback = patch_application.get("rollback") or {
            "ok": not patch_application.get("snapshot_manifest"),
            "errors": [],
        }
        rollback_errors = [
            *list(recipe_rollback.get("errors") or []),
            *list(kernel_rollback.get("errors") or []),
        ]
        rolled_back = bool(recipe_rollback.get("ok") and kernel_rollback.get("ok"))
        rollback_result = {
            "ok": rolled_back,
            "recipe": recipe_rollback,
            "kernel": kernel_rollback,
            "errors": rollback_errors,
        }
        if live_shared_state is not None:
            if rolled_back:
                # A successful rollback leaves nothing pending, so this replaces the record rather than merging.
                live_shared_state.warm_replay_pending = {}
                self._save_warm_replay_pending(live_shared_state)
            else:
                live_shared_state.set_stop_reason("warm_replay_rollback_failed")
                self._save_warm_replay_pending(
                    live_shared_state,
                    status="rollback_failed",
                    rollback_errors=rollback_errors,
                )
        return {
            "status": "required_patch_failed" if rolled_back else "required_patch_rollback_failed",
            "error_class": "required_patch_failed" if rolled_back else "warm_replay_rollback_failed",
            "error": str(patch_application.get("failure") or "required recipe timeline patch failed"),
            "required_patch_failure": patch_application,
            "failed_patch_ref": patch_application.get("failed_ref"),
            "warm_kernel_rolled_back": bool(kernel_rollback.get("ok")),
            "warm_replay_rollback": rollback_result,
            "workspace": str(output_dir),
        }

    def _round_refused_before_ignition(
        self,
        attempt: BenchmarkAttempt,
        output_dir: Path,
        *,
        double_run: bool,
        ctx_extra: dict[str, Any],
    ) -> dict[str, Any] | None:
        ignitable, evidence = self._round_affordable_before_ignition(double_run=double_run, ctx_extra=ctx_extra)
        if ignitable:
            return None
        stopped_result = _unlaunched_round_result(
            STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS],
            round_label="baseline round",
            output_dir=output_dir,
            capture_meta={
                "materialized_config": str(attempt.materialized_config_path),
                "run_eval_disabled": bool(attempt.run_eval_disabled),
            },
        )
        stopped_result["budget_shortfall"] = evidence
        if evidence.get("baseline_failures_this_leg"):
            log.warning(
                "baseline_executor: the baseline has failed %d time(s) this leg and PRELUDE has spent "
                "everything outside the %.0fs optimization reserve (%.0fs usable), so no retry "
                "is booted. The phase closes on this same line.",
                evidence.get("baseline_failures_this_leg", 0),
                evidence.get("optimization_reserve_sec", 0.0),
                evidence.get("session_usable_sec", 0.0),
            )
        elif evidence.get("one_more_measurement_sec"):
            log.warning(
                "baseline_executor: this round (%.0fs) and one variant to read "
                "against it (%.0fs) need %.0fs, and %.0fs is left (bound=%s), so "
                "nothing is booted. A baseline no variant can follow is a "
                "denominator with no numerator; the anchor this session already "
                "measured stands.",
                evidence.get("round_sec", 0.0),
                evidence.get("one_more_measurement_sec", 0.0),
                evidence.get("expected_cost_sec", 0.0),
                evidence.get("affordable_sec", 0.0),
                evidence.get("bound", ""),
            )
        else:
            log.warning(
                "baseline_executor: this round needs %.0fs and only %.0fs is left "
                "(bound=%s), so nothing is booted. The anchor this session already "
                "measured stands.",
                evidence.get("expected_cost_sec", 0.0),
                evidence.get("affordable_sec", 0.0),
                evidence.get("bound", ""),
            )
        return stopped_result

    async def _run_guarded_round(
        self,
        attempt: BenchmarkAttempt,
        label: str,
        round_dir: Path,
        *,
        port: Any,
        **config: Any,
    ) -> dict[str, Any]:
        round_cfg = self._write_lifecycle_config(
            attempt.materialized_config_path,
            round_dir,
            cleanup=config.pop("cleanup"),
            port=port,
            run_eval=config.pop("run_eval"),
        )
        return await self._run_reported_round(
            attempt,
            label=label,
            config_path=round_cfg,
            output_dir=round_dir,
            **config,
        )

    async def _run_cold_start_guarded_rounds(
        self,
        attempt: BenchmarkAttempt,
        output_dir: Path,
        lifecycle: Mapping[str, Any],
        application: list[dict[str, str]] | dict[str, Any],
        *,
        defer_accuracy: bool,
    ) -> dict[str, Any]:
        params = attempt.params
        warmup_dir = output_dir / _WARMUP_ROUND_DIR
        log.info(
            "baseline_executor: cold-start guard — warmup round (discarded, boots persistent server) in %s",
            warmup_dir,
        )
        # A replayed KB config is promoted onto ``current_best`` and becomes the reference every later measurement
        # in the session is taken against, so it may not be adopted on throughput alone.
        replayed = str(getattr(attempt.ctx.task, "kind", "") or "") == "replay_warm_recipe"
        # A round that re-attaches to the server the warmup left up rather than booting one of its own.
        hot_round = {"port": lifecycle["port"], "server_already_ready": True}
        warmup_eval = True if replayed and not defer_accuracy else None
        # Each warmup starts a fresh benchmark cap; the session deadline never resets.
        warmup_result = await self._run_guarded_round(
            attempt,
            ROUND_WARMUP,
            warmup_dir,
            port=lifecycle["port"],
            cleanup=False,
            run_eval=warmup_eval,
            warmup_round=True,
        )
        if warmup_result.get("status") != "succeeded":
            # Warmup failure almost certainly recurs, so skip the measured round.
            warmup_result.setdefault("nonfatal_warnings", []).append("baseline_warmup_round_failed")
            log.warning(
                "baseline_executor: warmup round failed (error_class=%s); skipping measured round",
                warmup_result.get("error_class"),
            )
            _stamp_warm_patch_outcome(warmup_result, application, params)
            return warmup_result
        warmup_tput = warmup_result.get("output_throughput")
        await report_progress(
            unit="baseline_round",
            label="warmup",
            index=1,
            total=2,
            status="succeeded",
            output_throughput=warmup_tput,
            runtime_sec=warmup_result.get("subprocess_runtime_sec"),
        )
        if not defer_accuracy and (cold := self._cold_anchor_when_hot_pass_unaffordable(warmup_result, attempt)):
            return cold
        measure_dir = output_dir / _MEASURE_ROUND_DIR
        log.info(
            "baseline_executor: cold-start guard — measured baseline round in %s "
            "(warmup tput=%.1f tok/s discarded, reusing hot server)",
            measure_dir,
            warmup_tput or 0.0,
        )
        # Round 2 (measured): re-attach to the hot server (client only).
        result = await self._run_guarded_round(
            attempt, ROUND_MEASURE, measure_dir, cleanup=not defer_accuracy, run_eval=False, **hot_round
        )
        _stamp_warm_patch_outcome(result, application, params)
        if result.get("status") != "succeeded":
            if result.get("error_class") != SESSION_TIME_EXHAUSTED_CLASS:
                return result
            # The gate before this pass admitted it and the run's clock took it anyway -- the pass overran what it
            # was priced at.
            log.warning(
                "baseline_executor: the run's clock stopped the measured round mid-flight, so the warmup "
                "stands as the baseline. It is the cold anchor a single-round baseline would have produced, "
                "and the marker below says so.",
            )
            dropped = {"reason": "measure_round_reaped_by_the_run", "measure_round_error": result.get("error")}
            return _cold_anchor_from_warmup(warmup_result, dropped=dropped)
        self._record_measured_against_warmup(result, warmup_result)
        if defer_accuracy:
            await self._settle_deferred_accuracy(attempt, output_dir, hot_round, result)
        return result

    def _cold_anchor_when_hot_pass_unaffordable(
        self,
        warmup_result: dict[str, Any],
        attempt: BenchmarkAttempt,
    ) -> dict[str, Any] | None:
        affordable, evidence = self._measure_round_affordable(
            warmup_runtime_sec=warmup_result.get("subprocess_runtime_sec"),
            warmup_post_ready_sec=post_ready_sec_without_eval(warmup_result),
            ctx_extra=getattr(attempt.ctx, "extra", None) or {},
        )
        if affordable:
            return None
        if evidence.get("one_more_measurement_sec"):
            why = "a hot pass (%.0fs) and one variant to read against it (%.0fs) need %.0fs" % (
                evidence.get("measure_round_sec", 0.0),
                evidence.get("one_more_measurement_sec", 0.0),
                evidence.get("expected_cost_sec", 0.0),
            )
        else:
            why = "a hot pass needs %.0fs" % (evidence.get("expected_cost_sec", 0.0),)
        log.warning(
            "baseline_executor: %s, and %.0fs is left (bound=%s), so the hot "
            "pass is not run. It would have bought a denominator nothing "
            "could then be compared to, and its own overtime anchor would "
            "have gone unused. Keeping the warmup as the baseline; it is the "
            "cold anchor a single-round baseline would have produced, and "
            "the GPU time it cost is already spent. The marker below says "
            "the figure is cold so the session's later gains can be read "
            "against a known-depressed denominator.",
            why,
            evidence.get("affordable_sec", 0.0),
            evidence.get("bound", ""),
        )
        return _cold_anchor_from_warmup(warmup_result, dropped=evidence)

    def _record_measured_against_warmup(self, result: dict[str, Any], warmup_result: dict[str, Any]) -> None:
        result.setdefault("nonfatal_warnings", []).append("baseline_double_run_discarded_first")
        # The measured pass re-attaches to the server launched by the warmup pass.
        for evidence_field in ("launch_evidence", "launch_evidence_path", "server_log_path"):
            if warmup_result.get(evidence_field):
                result[evidence_field] = warmup_result[evidence_field]
        warmup_evidence = result.get("launch_evidence")
        if isinstance(warmup_evidence, Mapping):
            measured_evidence = dict(warmup_evidence)
            measured_evidence["warm_reuse"] = {
                **dict(warmup_evidence.get("warm_reuse") or {}),
                "reused_ready_server": True,
                "provenance": "warmup_round",
                "source_server_log_path": str(result.get("server_log_path") or ""),
            }
            result["launch_evidence"] = measured_evidence
        warmup_tput = warmup_result.get("output_throughput")
        warmup_runtime = warmup_result.get("subprocess_runtime_sec")
        result["warmup_round_tput"] = warmup_tput
        self._warn_on_cold_start_delta(result, warmup_tput)
        # The Coordinator promotes ``subprocess_runtime_sec`` into the explore soft-kill anchor.
        if isinstance(warmup_runtime, (int, float)) and warmup_runtime > 0:
            result["measure_round_runtime_sec"] = result.get("subprocess_runtime_sec")
            result["subprocess_runtime_sec"] = round(float(warmup_runtime), 2)
            # The split belongs to round 1 for the same reason its wall-clock does: round 2 re-attached, so it
            # has no boot to separate and its own reading says nothing about what booting this workload costs.
            result["post_ready_runtime_sec"] = warmup_result.get("post_ready_runtime_sec")
        hot = result.get("output_throughput") or 0.0
        cold = warmup_tput or 0.0
        log.info(
            "baseline_executor: cold-start guard — measured "
            "baseline=%.1f tok/s (warmup=%.1f tok/s discarded; "
            "artifact would have been +%.0f%%)",
            hot,
            cold,
            ((hot / cold - 1.0) * 100.0) if cold > 0 else 0.0,
        )

    async def _settle_deferred_accuracy(
        self,
        attempt: BenchmarkAttempt,
        output_dir: Path,
        hot_round: Mapping[str, Any],
        result: dict[str, Any],
    ) -> None:
        params = attempt.params
        hot = float(result.get("output_throughput") or 0.0)
        keep_policy = params.get("post_measure_accuracy_keep_policy")
        if keep_policy is None:
            try:
                min_tput = float(params.get("post_measure_accuracy_min_tput", 0.0) or 0.0)
            except (TypeError, ValueError):
                min_tput = 0.0
            run_accuracy = hot >= min_tput
            skipped = {
                "status": "skipped",
                "reason": "throughput_below_threshold",
                "minimum_tput": min_tput,
                "observed_tput": hot,
            }
        else:
            ctx_extra = getattr(attempt.ctx, "extra", None) or {}
            performance = assess_integrate_performance(
                ctx_extra.get("shared_state") or self.shared_state,
                result,
                **keep_policy,
            )
            graded = performance.graded
            run_accuracy = performance.decision == "KEEP"
            skipped = {
                "status": "skipped",
                "reason": (
                    "intvty_regression"
                    if graded.graded_on_intvty and graded.verdict == "REVERT"
                    else "performance_keep_not_eligible"
                ),
                "graded_objective": graded.objective,
                "candidate": graded.candidate,
                "reference": graded.reference,
                "base_tput": keep_policy["base_tput"],
                "gain_pct": performance.gain_pct,
                "stack_incremental_gain_pct": performance.stack_incremental_gain_pct,
                "degrade_reason": graded.degrade_reason,
            }
        if not run_accuracy:
            result["accuracy_stage"] = skipped
            return
        accuracy_result = await self._run_guarded_round(
            replace(attempt, run_eval_disabled=False),
            ROUND_ACCURACY,
            output_dir / _ACCURACY_ROUND_DIR,
            cleanup=True,
            run_eval=True,
            **hot_round,
        )
        result["accuracy_stage"] = {
            "status": accuracy_result.get("status"),
            "error_class": accuracy_result.get("error_class"),
            "workspace": accuracy_result.get("workspace"),
        }
        if accuracy_result.get("status") != "succeeded":
            result.setdefault("nonfatal_warnings", []).append("post_measure_accuracy_failed")
            return
        for key in ("accuracy", "accuracy_task", "accuracy_metric", "accuracy_source"):
            if accuracy_result.get(key) is not None:
                result[key] = accuracy_result[key]

    def _write_lifecycle_config(
        self,
        base_config_path: Path,
        dest_dir: Path,
        *,
        cleanup: bool,
        port: int,
        run_eval: bool | None = None,
    ) -> Path:
        """Render a per-round YAML injecting ``benchmark.server_lifecycle``."""
        with Path(base_config_path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        bench = cfg.setdefault("benchmark", {})
        # The rounds of one attempt share the server's pid files in the slot above their own dirs.
        _lifecycle.inject_lifecycle(
            bench,
            cleanup=cleanup,
            pid_dir=dest_dir.parent,
            port=port,
        )
        if run_eval is not None:
            bench.setdefault("envs", {})["RUN_EVAL"] = "true" if run_eval else "false"
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = Path(dest_dir) / "baseline_lifecycle.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return out

    async def _pre_start_cleanup(
        self,
        *,
        pid_dir: Path,
        framework: str,
        port: int,
    ) -> None:
        """Remove the pid/json files a previous round left for this framework and port."""
        for p in (lifecycle_pid_file(pid_dir, framework, port), lifecycle_meta_file(pid_dir, framework, port)):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    async def _run_reported_round(
        self,
        attempt: BenchmarkAttempt,
        *,
        label: str,
        config_path: Path,
        output_dir: Path,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Announce a benchmark round before it blocks, run it, and record it."""
        recorder = attempt.recorder
        await report_progress(unit="baseline_round", label=label, status="started")
        started_at = now_iso("seconds")
        started_monotonic = time.monotonic()

        def _record(result: dict[str, Any]) -> None:
            if recorder is None:
                return
            recorder.record_round(
                run_index=attempt.run_index,
                round_=BaselineRound(
                    label=label,
                    started_at=started_at,
                    duration_sec=round(time.monotonic() - started_monotonic, 3),
                    timeout_sec=attempt.timeout_sec,
                    result=result,
                ),
            )

        try:
            result = await self._run_single_benchmark(
                attempt,
                config_path=config_path,
                output_dir=output_dir,
                **overrides,
            )
        except BaseException as exc:
            # A round that raised still happened, and its row is the only place the timeline can say which round the
            # action died in.
            _record({"status": "failed", "error_class": type(exc).__name__, "error": repr(exc)})
            raise
        _record(result)
        return result

    def _no_workspace_failure(
        self,
        round_: _BenchmarkRound,
        finished: _BenchProcess,
        boot: tuple[str, str] | None,
        harvested: Any,
    ) -> dict[str, Any]:
        failure_extras = {
            "output_dir": str(round_.output_dir),
            "harvested_artifacts": [str(dst) for _, dst in harvested],
            **round_.capture_meta,
        }
        # Magpie never created a benchmark_* workspace, so the wrapper never wrote server.log.
        captured = redact_secret_values(finished.stderr + finished.stdout)
        if captured.strip():
            try:
                log_file = round_.output_dir / "baseline_stderr.log"
                log_file.write_text(captured, encoding="utf-8")
                failure_extras["stderr_log_path"] = str(log_file)
            except OSError as exc:
                log.warning("baseline_executor: failed to persist stderr log: %s", exc)
        if boot is not None:
            error_class, error = boot
            return {
                "status": "failed",
                "error_class": error_class,
                "returncode": finished.returncode,
                "error": redact_secret_values(error),
                **failure_extras,
            }
        if finished.returncode != 0:
            tail = redact_secret_values((finished.stderr or finished.stdout)[-2000:])
            err_class = _classify_subprocess_error(finished.runtime_sec, tail)
            return {
                "status": "failed",
                "error_class": err_class,
                "returncode": finished.returncode,
                "subprocess_runtime_sec": round(finished.runtime_sec, 2),
                "error": tail,
                "output_signature": nonzero_output_signature(err_class, finished.stdout, finished.stderr),
                **failure_extras,
            }
        return {
            "status": "failed",
            "error_class": "no_workspace",
            "error": "Magpie completed but produced no benchmark_* workspace",
            **failure_extras,
        }

    def _unmeasured_round_failure(
        self,
        finished: _BenchProcess,
        boot: tuple[str, str] | None,
        measurement: dict[str, Any],
        report_path: Path,
        tail: str,
    ) -> tuple[str, str]:
        """Name a round that left no usable measurement, from the strongest evidence it has."""
        if boot is not None:
            return boot
        if finished.returncode != 0:
            error_class = _classify_subprocess_error(finished.runtime_sec, tail)
            if error_class == SUBPROCESS_NONZERO and served_complete_protocol(measurement):
                # The run's own artifact records every requested request as served, so the exit
                # happened after the workload rather than during it. That is what the exit code
                # cannot say on its own, and it is the difference between a harness that refused
                # the result and a process that died mid-round.
                error_class = MAGPIE_NONZERO_INVALID_MEASUREMENT
            return error_class, tail
        if not report_path.exists():
            return "no_report", f"{_BENCHMARK_REPORT_NAME} missing under {report_path.parent}"
        return "invalid_measurement", "benchmark report did not contain positive throughput and completed requests"

    def _failed_round_result(
        self,
        round_: _BenchmarkRound,
        finished: _BenchProcess,
        boot: tuple[str, str] | None,
        measurement: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any] | None:
        workspace = round_.output_dir if finished.workspace is None else finished.workspace
        report_path = workspace / _BENCHMARK_REPORT_NAME
        tail = (finished.stderr or finished.stdout)[-2000:]
        failed = {
            "status": "failed",
            "returncode": finished.returncode,
            "output_dir": str(round_.output_dir),
            "workspace": str(workspace),
            "report_path": str(report_path) if report_path.exists() else None,
            "reported_success": measurement.get("reported_success"),
            "subprocess_runtime_sec": round(finished.runtime_sec, 2),
            "nonfatal_warnings": warnings,
            **round_.capture_meta,
        }
        if not self._measurement_succeeded(measurement):
            error_class, error = self._unmeasured_round_failure(finished, boot, measurement, report_path, tail)
            return {
                **failed,
                "error_class": error_class,
                "error": redact_secret_values(error),
                "output_signature": nonzero_output_signature(error_class, finished.stdout, finished.stderr),
            }
        if finished.returncode == 0:
            return None
        nonzero_error = redact_secret_values(tail)
        if not served_complete_protocol(measurement):
            return {
                **failed,
                "error_class": "magpie_nonzero_after_valid_measurement",
                "error": nonzero_error,
            }
        # The round served every request the client recorded as requested, so the exit code is not evidence against
        # the measurement. The cause stays on the result rather than only in the log.
        warnings.append(f"nonzero_rc_after_complete_protocol:{finished.returncode}")
        log.warning(
            "baseline_executor: magpie exited %d after serving its whole protocol (%s of %s requests); "
            "keeping the measurement: %s",
            finished.returncode,
            measurement.get("completed_requests"),
            measurement.get("requested_requests"),
            nonzero_error,
        )
        return None

    def _stamp_round_quality(
        self,
        round_: _BenchmarkRound,
        result: dict[str, Any],
        report: dict[str, Any] | None,
        warnings: list[str],
    ) -> None:
        from hyperloom.inference_optimizer import framework_registry

        from .._accuracy_gate import eval_probe_summary, parse_eval_results, read_eval_probe

        # Parse accuracy eval results (GSM8K for serving, or the image-quality gate for scriptable frameworks).
        eval_framework = (report or {}).get("framework") or os.environ.get("FRAMEWORK") or None
        shared_state = round_.ctx_extra.get("shared_state") or self.shared_state
        benchmark_mode = str(getattr(shared_state, _BENCHMARK_MODE_ATTR, "") or "")
        # RUN_EVAL gates ONLY the serving lm-eval GSM8K run.
        if (
            round_.run_eval_disabled
            and not framework_registry.is_scriptable(eval_framework)
            and not is_agentx_mode(benchmark_mode)
        ):
            # Serving RUN_EVAL was off this run (a profile round or a staged warmup round), so lm-eval did not
            # execute and there is no fresh accuracy to read.
            log.info(
                "baseline_executor: RUN_EVAL disabled this run (serving); skipping accuracy parse (no lm-eval executed)"
            )
        else:
            # Search from ``$RESULT_DIR`` so serving runs survive benchmark_lib.sh moving/cleaning
            # ``$EVAL_RESULT_DIR`` and scriptable quality gates still resolve from Magpie's benchmark reports.
            eval_data = parse_eval_results(round_.result_dir, framework=eval_framework, benchmark_mode=benchmark_mode)
            if eval_data.get("accuracy") is not None:
                result["accuracy"] = eval_data["accuracy"]
                result["accuracy_task"] = eval_data.get("task", "gsm8k")
                result["accuracy_metric"] = eval_data.get("metric", "")
                result["accuracy_source"] = eval_data.get("source_file", "")
                log.info("baseline_executor: accuracy=%.4f (%s)", result["accuracy"], result["accuracy_task"])
            else:
                log.warning("baseline_executor: accuracy eval not found: %s", eval_data.get("error", "unknown"))
            # Records why the score is ~0; the score itself is already correct.
            if eval_probe := read_eval_probe(round_.result_dir):
                result["eval_probe"] = eval_probe
                log.warning("baseline_executor: %s", eval_probe_summary(eval_probe))
        log.info(
            "baseline_executor: %s %.1f %s (output) e2el=%.1fms",
            "success_with_warning" if warnings else "success",
            result["output_throughput"],
            framework_registry.throughput_unit(eval_framework),
            result["e2el_mean_ms"] or 0.0,
        )
