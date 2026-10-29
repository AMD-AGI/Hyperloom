# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``baseline`` ActionRunner — runs Magpie SGLang benchmark.

``BenchmarkRunExecutor`` is the benchmark round shared by every action that runs one;
``BaselineExecutor`` adds the baseline timeline event, the benchmark watchdog and the
InferenceX eval patching.
"""

from __future__ import annotations
import asyncio
import logging
import os
import subprocess
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from common.coerce import optional_positive_float
from common.env import is_truthy
from hyperloom.common.perf_metric import agentx_active, with_output_per_gpu
from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import (
    ROUND_SINGLE,
    BaselineRequest,
    make_baseline_recorder,
)
from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM
from hyperloom.inference_optimizer import inferencex_checkout
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ...loop.sub_agent_runner import RunnerContext
from hyperloom.inference_optimizer.trace.task_progress import heartbeat_while_output_flows
from ...phases import machine_state as _phase_state
from ..stop_attribution import (
    SESSION_TIME_EXHAUSTED_CLASS,
    STOPPED_BY_THE_RUN,
)
from . import _server_lifecycle as _lifecycle

# The grid module is the namespace the helpers both benching arms share ended up in: how a sentinel returncode reads
# back, how a round's cap is clamped to the budget, how the two session bounds are resolved, and the hygiene every
# launch needs.
from ._grid_runner import (
    SessionDirField,
    session_grid_bounds,
    sync_benchmark_timeout,
)
from ._subprocess_kill import (
    AGENTX_PREFLIGHT_ERROR_CLASS,
    RoundWatch,
    resolve_benchmark_timeouts,
    run_with_session_kill,
    session_deadline_to_remaining_sec,
)
from ._accuracy_gate import materialized_run_eval_disabled

from ._workload_envs import (
    _client_tokenizer_mode,
    default_baseline_config,
    prepare_agentx_runtime,
)
from ._inferencex_patcher import (
    ensure_benchmark_lib_eval_dest_patched,
    ensure_benchmark_lib_eval_start_patched,
    ensure_benchmark_serving_seed_patched,
    ensure_eval_probe_patched,
    ensure_eval_unbound_outputs_patched,
    eval_probe_targets_exist,
    failed_patch_anchors,
    failed_patch_anchors_in,
)
from ._agentx_prefix_cache import refuse_cacheless_agentx_baseline
from ._magpie_patcher import client_tokenizer_hook_installed, ensure_eval_concurrency_compat
from .benchmark_config import apply_benchmark_round_seed
from .benchmark_result import (
    double_run_requested,
)
from .benchmark_backend import build_benchmark_command
from .benchmark_run.accuracy import _EVAL_FAILURE_MARKERS, _BaselineAccuracyMixin
from .benchmark_run.launch import _BenchmarkLaunchMixin
from .benchmark_run.records import (
    _BENCHMARK_MODE_ATTR,
    BenchmarkAttempt,
    _BenchmarkRound,
    _BenchProcess,
    _benchmark_process_env,
    _config_framework,
    _requested_framework,
    _resolve_result_dir,
    _set_materialized_run_eval,
    _should_establish_quality_ref,
    _stopped_round_result,
    _unlaunched_round_result,
    _watchdog_server_log_path,
)
from .benchmark_run.rounds import _BenchmarkRoundsMixin
from .benchmark_run.warm_patches import _revert_legacy_warm_patch_trees, _stamp_warm_patch_outcome


log = logging.getLogger(__name__)

#: The producer label the baseline event's fragments are written under.
_RECORDER_PRODUCER = "orchestrator"

# The measured pass ran as the first traffic against a freshly restarted server, because the warmup that exists to
# drive it did not.
_MN_WARMUP_DID_NOT_WARM_WARNING = "baseline_mn_warmup_did_not_run"


# Markers identifying a MoE quant scheme with no implementation for the ``--moe-runner-backend`` in use:
# ``create_moe_runner`` falls through without building a runner and the first forward pass dies (e.g. Quark MXFP4 on
# triton).
_MOE_RUNNER_MISSING_MARKERS = (
    "has no attribute 'runner'",
    'has no attribute "runner"',
)
_MOE_SCHEME_CONTEXT_MARKERS = (
    "create_moe_runner",
    "moe_runner",
    "_moe.py",
    "fused_moe",
    "/moe/",
)


def _round_headroom_sec(state: Any) -> tuple[float | None, dict[str, Any]]:
    """Seconds this round's budget may still spend, and the numbers behind it."""
    if state is None:
        return None, {"reason": "no_session_state"}
    usable_sec = _phase_state.session_usable_seconds(state)
    if usable_sec is None:
        return None, {"reason": "unbounded_session_budget"}
    return usable_sec, {"bound": "session_usable", "affordable_sec": round(usable_sec, 1)}


def _a_use_must_follow_the_round(state: Any) -> bool:
    """Whether this round is only worth running if something can be measured after it."""
    phase = str(getattr(state, "phase", "") or "").strip().upper()
    return phase == _phase_state.PHASE_PRELUDE


class BenchmarkRunExecutor(_BaselineAccuracyMixin, _BenchmarkLaunchMixin, _BenchmarkRoundsMixin):
    """One benchmark action: materialize the config, launch the backend, retry, parse the report."""

    benchmark_watchdog = False
    runs_accuracy_eval = True
    default_timeout_sec: int | None = None
    session_dir = SessionDirField()

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        shared_state: Any | None = None,
    ):
        """Initialize the executor with launch defaults."""
        # Backend-aware interpreter: bypass uses a plain python3, magpie uses the Magpie-importable venv.
        from .benchmark_backend import resolve_benchmark_interpreter

        self.magpie_python = magpie_python or resolve_benchmark_interpreter()
        # None = resolve from $FRAMEWORK at call time; explicit fixture path wins.
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.session_dir = session_dir
        self.shared_state = shared_state

    def _resolve_default_config(self) -> Path:
        """Resolve the benchmark YAML the round runs when the task names none."""
        raise NotImplementedError

    def _resolve_workspace(self, ctx: RunnerContext, action: str) -> Path:
        """Pick the per-task workspace dir."""
        params = ctx.task.params or {}
        if params.get("output_dir"):
            return Path(params["output_dir"])
        extra = getattr(ctx, "extra", None) or {}
        if extra.get("workspace"):
            return Path(extra["workspace"])
        return runs_dir(self.session_dir, action, ctx.task.task_id)

    def _resolve_shared_state(self, shared_state: Any | None = None) -> Any:
        """Resolve the live SharedState for a session-scoped flag read/write."""
        state = shared_state or self.shared_state
        if state is None:
            from ...state.shared_state import SharedState

            state = SharedState.load_or_init(self.session_dir)
        return state

    def _measurement_succeeded(self, measurement: dict[str, Any]) -> bool:
        return measurement.get("valid_measurement") is True

    def _resolve_timeout(self, params: dict[str, Any]) -> float:
        """Benchmark rounds share one cap; other executor purposes keep their budgets."""
        if self.benchmark_watchdog:
            return resolve_benchmark_timeouts()[1]
        return float(params.get("timeout_sec") or self.default_timeout_sec)

    def _after_materialize_config(
        self,
        config_path: Path,
        output_dir: Path,
    ) -> dict[str, Any] | None:
        """Hook after YAML materialization, before launch; a returned dict aborts the round with it."""
        return None

    @staticmethod
    def _is_eval_rooted_failure(result: dict[str, Any]) -> bool:
        """Whether a failed baseline result was caused by the accuracy eval."""
        return BenchmarkRunExecutor._failure_carries_markers(result, _EVAL_FAILURE_MARKERS)

    @staticmethod
    def _is_moe_runner_rooted_failure(result: dict[str, Any]) -> bool:
        """Whether a failed baseline died on the MoE runner backend in use."""
        return BenchmarkRunExecutor._failure_carries_markers(
            result,
            _MOE_RUNNER_MISSING_MARKERS,
            _MOE_SCHEME_CONTEXT_MARKERS,
        )

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the action inside the session the context names."""
        from hyperloom.inference_optimizer.session.session_binding import named_session_scope

        # Only a context that names its session binds one.
        with named_session_scope((getattr(ctx, "extra", None) or {}).get("session_dir")):
            return await self._run(ctx)

    async def _run(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the action inside the bound session."""
        return await self._run_retrying(ctx)

    async def _run_pass(
        self,
        ctx: RunnerContext,
        *,
        recorder: Any,
        attempt_reason: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one pass through :meth:`_run_once`, recorded as its own row."""
        index = recorder.begin_run(attempt_reason=attempt_reason) if recorder is not None else 0
        try:
            result = await self._run_once(ctx, recorder=recorder, run_index=index, **kwargs)
        except BaseException as exc:
            if recorder is not None:
                recorder.end_run(
                    run_index=index,
                    result={"status": "failed", "error_class": type(exc).__name__, "error": repr(exc)},
                )
            raise
        if recorder is not None:
            recorder.end_run(run_index=index, result=result)
        return result

    async def _run_attempt_rounds(self, attempt: BenchmarkAttempt, output_dir: Path) -> dict[str, Any]:
        params = attempt.params
        config_path = attempt.materialized_config_path
        extra = getattr(attempt.ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or self.shared_state
        # MLPerf prepares its delegated client off-loop; native AgentX is entirely managed by Magpie.
        agx_err = await asyncio.to_thread(
            prepare_agentx_runtime,
            env=os.environ,
            inferencex_path=attempt.inferencex_path,
            config_path=config_path,
            active=agentx_active(benchmark_mode=getattr(shared_state, _BENCHMARK_MODE_ATTR, "")),
        )
        if agx_err:
            agentx = {"error_class": AGENTX_PREFLIGHT_ERROR_CLASS, "error": agx_err}
            return {"status": "failed", **agentx, "output_dir": str(output_dir)}
        if (hook_result := self._after_materialize_config(config_path, output_dir)) is not None:
            hook_result.setdefault("materialized_config", str(config_path))
            hook_result.setdefault("output_dir", str(output_dir))
            return hook_result
        # Cold-start "warmup artifact" guard: the freshly-booted server's first benchmark window pays one-time cold
        # costs that inflate later gains into fictitious "improvements".
        lifecycle = _lifecycle.resolve_lifecycle_params(config_path)
        double_run_wanted = double_run_requested(params)
        double_run = double_run_wanted and lifecycle["eligible"]
        defer_accuracy = is_truthy(params.get("defer_accuracy_until_after_measure"))
        if defer_accuracy and double_run:
            # Only the lifecycle path can reuse the hot server for a staged accuracy round.
            _set_materialized_run_eval(config_path, enabled=False)
        # Whether THIS run actually executes lm-eval, read back from the materialized config the subprocess
        # consumes -- the single source of truth.
        attempt = replace(attempt, run_eval_disabled=materialized_run_eval_disabled(config_path))
        # Asked before the lease, because a round that will not be run should not hold a GPU while being refused.
        refused = self._round_refused_before_ignition(attempt, output_dir, double_run=double_run, ctx_extra=extra)
        if refused is not None:
            return refused
        application = self._prepare_round_patches(params, output_dir, shared_state)
        if isinstance(application, Mapping) and application.get("status") == "failed":
            return self._roll_back_failed_patches(application, params, output_dir, shared_state)
        # Ray-managed GPU execution (§12 T1): one held Ray lease (``num_gpus=TP``) spans this baseline's benchmark
        # rounds — a double-run's warmup + measure reuse one persistent server, so both must run under the same lease.
        from ._grid_runner import _num_gpus_for_config
        from ._ray_serving import maybe_serving_lease

        lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(config_path))
        attempt = replace(attempt, serving_lease=lease)
        # Deep-clean any lingering server right before this round actually boots -- after every earlier gate that can
        # still bail out without starting a server (budget, warm-patch failure), so a round that will not run does not
        # pay for a scan it gets no benefit from.
        framework, port = lifecycle["framework"], lifecycle["port"]
        await self._pre_start_cleanup(pid_dir=output_dir, framework=framework, port=port)
        try:
            if double_run:
                guarded = self._run_cold_start_guarded_rounds
                return await guarded(attempt, output_dir, lifecycle, application, defer_accuracy=defer_accuracy)
            if double_run_wanted:
                log.info(
                    "baseline_executor: cold-start double-run not eligible (%s); running single round.",
                    lifecycle["reason"],
                )
            result = await self._run_reported_round(
                attempt, label=ROUND_SINGLE, config_path=config_path, output_dir=output_dir
            )
            _stamp_warm_patch_outcome(result, application, params)
            return result
        finally:
            if double_run:
                # Defensive teardown so no persistent server leaks.
                _lifecycle.teardown_lifecycle_server(pid_dir=output_dir, framework=framework, port=port)
            else:
                _lifecycle.stop_agentx_servers(output_dir)
            # Revert warm-replay patches to prevent state leakage into subsequent tasks that reuse the same
            # InferenceX checkout. A required timeline's tree is promoted by prelude after this returns, so it must
            # stay patched; reverting it here handed prelude a clean tree and silently lost the replay.
            if application and not isinstance(application, Mapping):
                _revert_legacy_warm_patch_trees(params)
            if lease is not None:
                lease.close()

    def _round_affordable_before_ignition(
        self,
        *,
        double_run: bool,
        ctx_extra: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Whether the budget holds a whole round *and a use for it*, before anything boots."""
        state = (ctx_extra or {}).get("shared_state") or self.shared_state
        return self._round_affordable(
            state,
            round_sec=_phase_state.baseline_round_cost_sec(state, double_run=double_run),
        )

    @staticmethod
    def _round_affordable(state: Any, *, round_sec: float | None) -> tuple[bool, dict[str, Any]]:
        """Whether the budget holds a round costing ``round_sec`` and a use for it."""
        cold_sec = _phase_state.measured_seconds(state, "baseline_runtime_sec")
        if cold_sec is None or round_sec is None:
            return True, {"reason": "no_measured_round_to_predict_from"}
        use_sec = 0.0
        if _a_use_must_follow_the_round(state):
            # Without the split, a variant is priced at a whole cold round, which is what it is: a boot and a
            # benchmark that pays the compile.
            use_sec = _phase_state.one_more_measurement_sec(state) or cold_sec
        headroom_sec, evidence = _round_headroom_sec(state)
        if headroom_sec is None:
            return True, evidence
        cost = round_sec + use_sec
        priced = {
            "expected_cost_sec": round(cost, 1),
            "round_sec": round(round_sec, 1),
            "one_more_measurement_sec": round(use_sec, 1),
            **evidence,
        }
        return headroom_sec >= cost, priced

    def _measure_round_affordable(
        self,
        *,
        warmup_runtime_sec: Any,
        warmup_post_ready_sec: Any = None,
        ctx_extra: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Whether the budget covers the measured round *and a use for it*."""
        state = (ctx_extra or {}).get("shared_state") or self.shared_state
        headroom_sec, evidence = _round_headroom_sec(state)
        if headroom_sec is None:
            return True, evidence
        warmup_sec = optional_positive_float(warmup_runtime_sec)
        priced_by = "session_hot_pass"
        benchmark_sec = _phase_state.measured_seconds(state, "baseline_warm_runtime_sec")
        if benchmark_sec is None:
            priced_by = "warmup_post_ready"
            benchmark_sec = optional_positive_float(warmup_post_ready_sec)
        if benchmark_sec is None or warmup_sec is None:
            return True, {"reason": "no_measured_benchmark_to_predict_from", **evidence}
        use_sec = 0.0
        if _a_use_must_follow_the_round(state):
            use_sec = _phase_state.one_more_measurement_sec(state) or warmup_sec
        cost = benchmark_sec + use_sec
        priced = {
            "expected_cost_sec": round(cost, 1),
            "priced_by": priced_by,
            "measure_round_sec": round(benchmark_sec, 1),
            "one_more_measurement_sec": round(use_sec, 1),
            **evidence,
        }
        return headroom_sec >= cost, priced

    async def _mn_warmup_pass(self, round_: _BenchmarkRound, round_warnings: list[str]) -> dict[str, Any] | None:
        """Run the discarded multi-node client warmup against the restarted server."""
        # A multi-node round is two client passes of the same shape against a server the round did not boot, and the
        # session's measured figure is one of them -- the round's wall-clock is taken after this pass -- so the pair
        # costs twice it.
        state = round_.ctx_extra.get("shared_state") or self.shared_state
        one_pass_sec = _phase_state.measured_seconds(state, "baseline_runtime_sec")
        affordable, evidence = self._round_affordable(
            state,
            round_sec=None if one_pass_sec is None else one_pass_sec * 2.0,
        )
        if not affordable:
            log.warning(
                "baseline_executor: a multi-node round is two passes needing %.0fs "
                "and %.0fs is left (bound=%s), so neither is launched. The anchor "
                "this session already measured stands.",
                evidence.get("expected_cost_sec", 0.0),
                evidence.get("affordable_sec", 0.0),
                evidence.get("bound", ""),
            )
            refused = _unlaunched_round_result(
                STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS],
                round_label="multi-node round",
                output_dir=round_.output_dir,
                capture_meta=round_.capture_meta,
            )
            refused["budget_shortfall"] = evidence
            return refused
        warm_dir = round_.output_dir / "mn_warmup"
        started_unix = time.time()
        # The measurement is discarded, but the returncode is not: this pass is a full benchmark round, so a stop here
        # ends the baseline round.
        warm_rc: int | None = None
        try:
            warm_dir.mkdir(parents=True, exist_ok=True)
            warm_cmd = [str(warm_dir) if c == str(round_.output_dir) else c for c in round_.cmd]
            warm_env = dict(round_.env)
            # The pass runs on the server the measured pass reuses, so it must not send the measured prompts.
            apply_benchmark_round_seed(round_.config_path, warm_env, warmup=True)
            warm_env["RESULT_DIR"] = str(warm_dir)
            warm_env["EVAL_RESULT_DIR"] = str(warm_dir / "eval_output")
            warm_env["SERVER_LOG"] = str(warm_dir / "server.log")
            warm_env["GPU_METRICS_CSV"] = str(warm_dir / "gpu_metrics.csv")
            async with heartbeat_while_output_flows(
                unit="baseline_round",
                label="mn_warmup",
            ) as warm_activity:
                warm_proc = await asyncio.to_thread(
                    run_with_session_kill,
                    warm_cmd,
                    env=warm_env,
                    cwd=str(warm_dir),
                    watch=RoundWatch(
                        timeout=round_.timeout_sec,
                        silence_timeout_sec=round_.silence_timeout_sec,
                        server_already_ready=True,
                        server_log_path=_watchdog_server_log_path(warm_dir, round_.framework),
                        session_deadline_sec=round_.session_deadline_sec,
                    ),
                    on_output=warm_activity.note,
                )
            warm_rc = warm_proc.returncode
            log.info("baseline_executor: MN warmup pass done (discarded) rc=%s", warm_rc)
        except subprocess.TimeoutExpired as exc:
            log.warning("baseline_executor: MN warmup pass hit its own hang backstop (ignored): %r", exc)
            round_warnings.append(_MN_WARMUP_DID_NOT_WARM_WARNING)
        except (OSError, ValueError) as exc:
            log.warning("baseline_executor: MN warmup pass failed (ignored): %r", exc)
            round_warnings.append(_MN_WARMUP_DID_NOT_WARM_WARNING)
        return _stopped_round_result(
            warm_rc,
            round_label="multi-node warmup pass",
            runtime_sec=max(0.0, time.time() - started_unix),
            output_dir=round_.output_dir,
            capture_meta=round_.capture_meta,
        )

    async def _run_single_benchmark(
        self,
        attempt: BenchmarkAttempt,
        *,
        config_path: Path,
        output_dir: Path,
        server_already_ready: bool = False,
        warmup_round: bool = False,
    ) -> dict[str, Any]:
        """Run one Magpie benchmark subprocess and parse its result; ``warmup_round`` marks a discarded round."""
        from ._multi_node_env import (
            is_multi_node,
            log_mn_banner,
            magpie_remote_env,
            mn_bench_warmup_enabled,
        )
        from .magpie_agentx import bind_source_evidence

        await asyncio.to_thread(bind_source_evidence, config_path, attempt.params.get("patches") or [])
        result_dir = _resolve_result_dir(output_dir, attempt.override_result_dir)
        env = _benchmark_process_env(config_path, output_dir, result_dir, attempt.inferencex_path)
        # Multi-node (--nodes >= 2): inject MAGPIE_RUN_PHASE=client + BENCHMARK_BASE_URL so Magpie skips its server
        # launch and targets the RayJob head.
        env.update(magpie_remote_env())
        apply_benchmark_round_seed(config_path, env, warmup=warmup_round)
        ctx_extra = getattr(attempt.ctx, "extra", None) or {}
        # The session's wall-clock budget, resolved per round rather than once per task: a baseline runs up to three
        # of them, and a warmup that overran has already spent budget the ones after it were counting on.
        session_deadline_sec, _ = session_grid_bounds(ctx_extra.get("shared_state") or self.shared_state)
        silence_timeout_sec = None
        timeout_sec = attempt.timeout_sec
        if self.benchmark_watchdog:
            silence_timeout_sec, timeout_sec = resolve_benchmark_timeouts()
            env["PYTHONUNBUFFERED"] = "1"
            sync_benchmark_timeout(config_path, timeout_sec)
        # The materialized YAML is authoritative for the framework: params and $FRAMEWORK are both optional on a
        # baseline task.
        framework = (
            _config_framework(attempt.materialized_config_path) or (_requested_framework(attempt.params) or "").lower()
        )
        round_ = _BenchmarkRound(
            config_path=config_path,
            materialized_config_path=attempt.materialized_config_path,
            output_dir=output_dir,
            result_dir=result_dir,
            ctx=attempt.ctx,
            ctx_extra=ctx_extra,
            cmd=build_benchmark_command(python_exe=self.magpie_python, config_path=config_path, output_dir=output_dir),
            env=env,
            framework=framework,
            watchdog_server_log=_watchdog_server_log_path(output_dir, framework),
            # Config/eval-contract facts echoed onto every failure result so an eval-rooted failure can be
            # fingerprinted and re-run by enablement.
            capture_meta={
                "materialized_config": str(attempt.materialized_config_path),
                "result_dir": str(result_dir),
                "run_eval_disabled": bool(attempt.run_eval_disabled),
            },
            timeout_sec=timeout_sec,
            silence_timeout_sec=silence_timeout_sec,
            session_deadline_sec=session_deadline_sec,
            serving_lease=attempt.serving_lease,
            server_already_ready=server_already_ready,
            run_eval_disabled=attempt.run_eval_disabled,
        )
        # Multi-node only: restart sglang/vllm per round for a fresh server.
        restart_failure = await self._restart_multi_node_server(
            round_, server_args=attempt.effective_extra_server_args, resolved_model=attempt.resolved_model
        )
        if restart_failure is not None:
            return restart_failure
        log_mn_banner("baseline_executor", log, output_dir=str(output_dir))
        log.info("baseline_executor: launching Magpie cmd=%s output_dir=%s", round_.cmd, output_dir)
        round_warnings: list[str] = []
        if is_multi_node() and mn_bench_warmup_enabled() and not ctx_extra.get("mn_round_restarted"):
            warmup = await self._mn_warmup_pass(round_, round_warnings)
            if warmup is not None:
                return warmup
        return await self._measured_round_result(round_, round_warnings)

    async def _launch_benchmark(
        self,
        round_: _BenchmarkRound,
        started_unix: float,
        *,
        server_ready: bool,
    ) -> _BenchProcess:
        if round_.serving_lease is not None:
            # Ray-managed GPU execution (§12 T1): run inside the lease's actor (holds num_gpus across this run's
            # rounds).
            from ._ray_backend import SessionKillRound, strip_visible_devices_from_config

            # No liveness callback is possible here: the round runs inside a Ray actor in another process
            # (potentially on another node) and only its final ``(rc, stdout, stderr)`` crosses back, so there is
            # nothing local to call per line of child output.
            returncode, stdout, stderr = await asyncio.to_thread(
                round_.serving_lease.run_session_kill,
                SessionKillRound(
                    build_benchmark_command(
                        python_exe=self.magpie_python,
                        config_path=strip_visible_devices_from_config(round_.config_path),
                        output_dir=round_.output_dir,
                    ),
                    env=round_.env,
                    cwd=str(round_.output_dir),
                    timeout=round_.timeout_sec,
                    silence_timeout_sec=round_.silence_timeout_sec,
                    server_already_ready=server_ready,
                    server_log_path=round_.watchdog_server_log,
                    session_remaining_sec=session_deadline_to_remaining_sec(round_.session_deadline_sec),
                ),
            )
        else:
            async with heartbeat_while_output_flows(unit="baseline_round", label="benchmark") as activity:
                proc = await asyncio.to_thread(
                    run_with_session_kill,
                    round_.cmd,
                    env=round_.env,
                    cwd=str(round_.output_dir),
                    watch=RoundWatch(
                        timeout=round_.timeout_sec,
                        silence_timeout_sec=round_.silence_timeout_sec,
                        server_already_ready=server_ready,
                        server_log_path=round_.watchdog_server_log,
                        session_deadline_sec=round_.session_deadline_sec,
                    ),
                    on_output=activity.note,
                )
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        return _BenchProcess(
            returncode,
            stdout or "",
            stderr or "",
            max(0.0, time.time() - started_unix),
            started_unix,
        )


class BaselineExecutor(BenchmarkRunExecutor):
    """Class form for tests / DI; ``baseline_executor`` is the bare callable."""

    benchmark_watchdog = True

    def _resolve_default_config(self) -> Path:
        """Resolve the benchmark YAML the round runs when the task names none."""
        return default_baseline_config()

    @staticmethod
    def _client_script_from_config(config_path: Path) -> str | None:
        """Name the client script this round will run, or ``None`` if unknown.

        Magpie selects ``<framework>_<runner_type>.sh``. Naming it keeps the
        tokenizer check to the script that matters: the multimodal variants carry
        the same ``--result-dir`` marker with a different client call, and judging
        them would let an unrelated shape veto a workload whose own script is fine.
        """
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return None
        bench = (cfg.get("benchmark") if isinstance(cfg, dict) else {}) or {}
        override = str(bench.get("benchmark_script") or "").strip()
        if override:
            return override
        framework = str(bench.get("framework") or "").strip().lower()
        runner = str(bench.get("runner_type") or "").strip().lower()
        if not framework or not runner:
            return None
        return f"{framework}_{runner}.sh"

    @staticmethod
    def _model_from_config(config_path: Path) -> str:
        """Read the benchmark model path out of the materialized config."""
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return ""
        bench = cfg.get("benchmark") if isinstance(cfg, dict) else {}
        return str((bench or {}).get("model") or "").strip()

    @staticmethod
    def _inferencex_root_from_config(config_path: Path) -> str:
        """Resolve the InferenceX checkout the subprocess will ``cd`` into."""
        try:
            cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            bench = cfg.get("benchmark") if isinstance(cfg, dict) else {}
            path = str((bench or {}).get("inferencex_path") or "").strip()
        except (OSError, yaml.YAMLError):
            path = ""
        return path or os.environ.get("INFERENCEX_PATH", "").strip()

    def _after_materialize_config(
        self,
        config_path: Path,
        output_dir: Path,
    ) -> dict[str, Any] | None:
        """Hook after YAML materialization, before launch."""
        from .magpie_agentx import native_config

        if native_config(config_path):
            return None
        ix_root = self._inferencex_root_from_config(config_path)
        if ix_root:
            if invalid := inferencex_checkout.unusable_reason(ix_root):
                return {"status": "failed", "error_class": "inferencex_checkout_invalid", "error": invalid}
            ensure_benchmark_lib_eval_dest_patched(Path(ix_root))
            ensure_benchmark_lib_eval_start_patched(Path(ix_root))
        # Best-effort: without it the warmup round pre-fills the measured round's prefix cache, which inflates the
        # anchor but never breaks a round.
        ensure_benchmark_serving_seed_patched(Path(ix_root) if ix_root else None)
        # Required custom tokenizers must reach the throughput client as well as eval.
        tok_mode = _client_tokenizer_mode(self._model_from_config(config_path))
        if tok_mode:
            script_name = self._client_script_from_config(config_path)
            if not script_name or not client_tokenizer_hook_installed(script_name, inferencex_dir=ix_root or None):
                msg = (
                    f"the benchmark client script {script_name or '<unresolved>'!r} cannot be told to load the "
                    f"{tok_mode!r} tokenizer: it has no --tokenizer-mode hook "
                    f"(inferencex={ix_root or '<unset>'}). Use a Magpie revision at or above "
                    "1eb3df69de9cb26cb5c8109e43371dd1abb9bbed. This model's client resolves the "
                    "checkpoint through HF AutoConfig, which cannot map its model_type, so it "
                    "would die before issuing a request and the round would be graded a boot "
                    "failure with the server up."
                )
                log.error("baseline_executor: %s", msg)
                return {
                    "status": "failed",
                    "error_class": "client_tokenizer_unpatchable",
                    "error": msg,
                }
        if materialized_run_eval_disabled(config_path):
            return None
        return self._eval_patch_failure(ix_root)

    def _eval_patch_failure(self, ix_root: str) -> dict[str, Any] | None:
        # Target present but unpatchable is a hard stop; target absent is an unrecognized layout, which warns
        # rather than failing every eval run.
        probe_root = Path(ix_root) if ix_root else None
        # Best-effort, unlike the probe below: without it a refused connection ends the round on
        # UnboundLocalError instead of on the connection, which is worse reporting of a round that
        # was already going to fail -- not a reason to refuse to start one.
        ensure_eval_unbound_outputs_patched(probe_root)
        if not ensure_eval_probe_patched(probe_root):
            msg = (
                "the generation-pathology probe is not installed "
                "(utils/evals/patches/lm_eval_sitecustomize.py, inferencex="
                f"{ix_root or '<unset>'}, INFERENCEX_PATH="
                f"{os.environ.get('INFERENCEX_PATH', '') or '<unset>'}). "
                "A model that never emits EOS will run the accuracy eval to "
                "the full max_tokens budget on every sample and consume the "
                "entire baseline timeout."
            )
            if eval_probe_targets_exist(probe_root):
                log.error("baseline_executor: %s", msg)
                return {
                    "status": "failed",
                    "error_class": "eval_probe_unpatchable",
                    "error": msg,
                }
            log.warning("baseline_executor: %s", msg)
        # Fail LOUDLY (never warn-and-continue) when the fatal eval flag cannot be removed AND this run is meant
        # to execute lm-eval: the benchmark is guaranteed to abort in run_lm_eval, and the accuracy gate then
        # stops the whole session.
        try:
            compat_ok = ensure_eval_concurrency_compat(inferencex_dir=ix_root or None)
        except OSError as exc:
            log.error("baseline_executor: eval-concurrency compat patch failed for %s: %s", ix_root, exc)
            compat_ok = False
        if not compat_ok:
            msg = (
                "accuracy eval cannot run: the redundant "
                "'--concurrent-requests' flag could not be removed from the "
                "Magpie benchmark scripts (MAGPIE_PATH="
                f"{os.environ.get('MAGPIE_PATH', '') or '<unset>'}) and/or "
                "InferenceX's run_lm_eval arg parser could not be made to "
                f"tolerate it (inferencex={ix_root or '<unset>'}). "
                "InferenceX resolves eval concurrency from "
                "EVAL_CONCURRENT_REQUESTS (fallback CONC); the flag is "
                "rejected as 'Unknown parameter: --concurrent-requests' and "
                "aborts the benchmark before any results*.json is written. "
                "Fix the run_eval line (or re-run install.sh against the "
                "Magpie tree that is actually imported at run time) — do "
                "NOT work around this with RUN_EVAL=false."
            )
            log.error("baseline_executor: %s", msg)
            return {
                "status": "failed",
                "error_class": "eval_concurrency_flag_unpatchable",
                "error": msg,
            }
        return self._eval_patch_anchors_result(ix_root)

    # Scoped to the line-replacement patches this hook applies.
    _EVAL_HOOK_ANCHORS = ("eval_dest", "eval_start")
    # Of those, the one whose silent absence corrupts the accuracy gate rather than degrading it: without
    # ``eval_dest`` the results file lands in the cwd and no score is ever parsed.
    _EVAL_CRITICAL_ANCHORS = ("eval_dest",)

    def _eval_patch_anchors_result(self, ix_root: str | None) -> dict[str, Any] | None:
        """Fail the launch when an eval-critical patch can no longer be applied.

        The ``ensure_*`` calls above report a miss as ``False`` and no caller
        reads it, so an upstream edit that moves an anchor takes the patch
        offline silently -- the run still looks healthy and only the symptom (no
        score) shows up much later. This is the same failure mode that took the
        probe offline before it was re-homed to a real file. Checked only when
        this run executes lm-eval; anchors that merely degrade something are
        logged, not fatal.

        Scoped to the checkout Magpie will benchmark. Env-wide discovery also
        reaches the InferenceX bundled with the installed Magpie, which a run
        that pins ``benchmark.inferencex_path`` never executes — judging it
        aborts every eval run on rot in a tree nothing here reads.

        Args:
            ix_root: The resolved InferenceX root for this run, if any.

        Returns:
            An early-return failure dict, or ``None`` to proceed.
        """
        rotted = failed_patch_anchors_in(ix_root) if ix_root else failed_patch_anchors(None)
        broken = [s for s in rotted if s.name in self._EVAL_HOOK_ANCHORS]
        if not broken:
            return None
        for status in broken:
            log.error("baseline_executor: InferenceX patch anchor broken — %s", status.describe())
        fatal = [s for s in broken if s.name in self._EVAL_CRITICAL_ANCHORS]
        if not fatal:
            return None
        msg = (
            "accuracy eval cannot run: Hyperloom redirects lm-eval's results file "
            "by matching exact upstream text, and that text is no longer there "
            f"(inferencex={ix_root or '<unset>'}). Broken: "
            + "; ".join(s.describe() for s in fatal)
            + ". Re-anchor the patch in _inferencex_patcher.py against the "
            "checkout in use, or pin INFERENCEX_REF back to a revision it "
            "matches. Continuing would leave every results*.json in the "
            "benchmark's cwd, where the accuracy parser never looks, so the gate "
            "would see no score at all — do NOT work around this with "
            "RUN_EVAL=false."
        )
        log.error("baseline_executor: %s", msg)
        return {
            "status": "failed",
            "error_class": "inferencex_patch_anchor_broken",
            "error": msg,
        }

    async def _run(self, ctx: RunnerContext) -> dict[str, Any]:
        """Open the baseline event, run the action, and close it either way."""
        params = ctx.task.params or {}
        streak, total = self._failure_counters(ctx)
        recorder = make_baseline_recorder(
            self._resolve_sink(ctx),
            BaselineRequest(
                task_id=str(getattr(ctx.task, "task_id", "") or ""),
                task_kind=str(getattr(ctx.task, "kind", "") or ""),
                reason=str(params.get("reason") or ""),
                framework=self._resolve_framework(ctx),
                establishes_quality_ref=_should_establish_quality_ref(getattr(ctx.task, "kind", ""), params),
                params=params,
                failure_streak_before=streak,
                total_failures_before=total,
            ),
            owns_event=not str(params.get(INLINE_EVENT_PARAM) or ""),
        )
        try:
            result = await self._run_retrying(ctx, recorder=recorder)
            shared_state = (getattr(ctx, "extra", None) or {}).get("shared_state")
            is_agentx = agentx_active(benchmark_mode=getattr(shared_state, _BENCHMARK_MODE_ATTR, ""))
            if _should_establish_quality_ref(getattr(ctx.task, "kind", ""), params) and is_agentx:
                refuse_cacheless_agentx_baseline(result)
        except BaseException as exc:
            if recorder is not None:
                recorder.finish_crashed(exc)
            raise
        if recorder is not None:
            recorder.finish(with_output_per_gpu(result, getattr(shared_state, "tp", None)))
        return result

    def _resolve_sink(self, ctx: RunnerContext) -> Any:
        """Decide which event this measurement's rows belong to."""
        from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import baseline_event_id
        from hyperloom.inference_optimizer.breakdown.recorder.event_sink import EventSink
        from hyperloom.inference_optimizer.session.session_binding import session_is_bound

        params = ctx.task.params or {}
        if not session_is_bound():
            log.warning(
                "baseline timeline: no session bound; this measurement's whole event will "
                "be missing from the breakdown. The coordinator binds at startup, so this "
                "means either that never happened or the context did not name a session"
            )
            return None
        inline = str(params.get(INLINE_EVENT_PARAM) or "")
        if inline:
            return EventSink(inline, producer=_RECORDER_PRODUCER)
        state = self._resolve_shared_state((getattr(ctx, "extra", None) or {}).get("shared_state"))
        event = baseline_event_id(
            str(getattr(state, "phase", "") or "unphased"),
            int(getattr(state, "macro_cycle", 0) or 0),
        )
        return EventSink(event, producer=_RECORDER_PRODUCER)

    def _failure_counters(self, ctx: RunnerContext) -> tuple[int | None, int | None]:
        """The session's baseline failure counts as this measurement starts.

        Read here, at the dispatch, because the counters are advanced by the
        write-back once this action has returned -- so the event cannot see its
        own effect on them, and the value it can see is the one that says what
        this attempt was dispatched into.

        Args:
            ctx (RunnerContext): The runner context.

        Returns:
            tuple[int | None, int | None]: The consecutive-failure streak and
                the session total, or ``(None, None)`` when no state is bound.
        """
        with suppress(AttributeError, TypeError, ValueError):
            state = self._resolve_shared_state((getattr(ctx, "extra", None) or {}).get("shared_state"))
            return (
                int(getattr(state, "baseline_failure_streak", 0) or 0),
                int(getattr(state, "baseline_total_failures", 0) or 0),
            )
        return (None, None)

    def _resolve_framework(self, ctx: RunnerContext) -> str:
        """Name the serving framework this measurement runs against."""
        params = ctx.task.params or {}
        with suppress(AttributeError, TypeError, ValueError):
            state = self._resolve_shared_state((getattr(ctx, "extra", None) or {}).get("shared_state"))
            return str(
                params.get("framework") or getattr(state, "framework", "") or os.environ.get("FRAMEWORK", "")
            ).strip()
        return str(params.get("framework") or os.environ.get("FRAMEWORK", "")).strip()


baseline_executor = BaselineExecutor()


__all__ = [
    "BaselineExecutor",
    "BenchmarkRunExecutor",
    "baseline_executor",
]
