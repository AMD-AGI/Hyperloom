# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``baseline`` ActionRunner — runs the benchmark rounds of a baseline measurement.

Every round runs on inference_suite with the base launch its session froze (:func:`._suite.rounds.suite_route`).
``BenchmarkRunExecutor`` is the benchmark round shared by every action that runs one; ``BaselineExecutor`` adds the
baseline timeline event and the benchmark watchdog.
"""

from __future__ import annotations
import asyncio
import logging
import os
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from hyperloom_common.coerce import optional_positive_float
from hyperloom_common.env import is_truthy
from hyperloom.common.perf_metric import agentx_active, with_output_per_gpu
from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import (
    ROUND_SINGLE,
    BaselineRequest,
    make_baseline_recorder,
)
from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ...loop.sub_agent_runner import RunnerContext
from hyperloom.inference_optimizer.trace.task_progress import heartbeat_while_output_flows
from ...phases import machine_state as _phase_state
from ..stop_attribution import (
    SESSION_TIME_EXHAUSTED_CLASS,
    STOPPED_BY_THE_RUN,
)
from inference_suite.contracts import FailedStep

from .grid._grid_base import GridRound, run_grid_suite_round
from ._suite.held_test import HeldPass, warms_up
from ._suite.rounds import SuiteRoundRefused, SuiteRoute, suite_route

# The grid module is the namespace the helpers both benching arms share ended up in: how a sentinel returncode reads
# back, how a round's cap is clamped to the budget, how the two session bounds are resolved, and the hygiene every
# launch needs.
from .grid._grid_inputs import SessionDirField, session_grid_bounds
from .grid._grid_base import sync_benchmark_timeout
from ._subprocess_kill import (
    AGENTX_PREFLIGHT_ERROR_CLASS,
    resolve_benchmark_timeouts,
)
from ._accuracy_gate import materialized_run_eval_disabled

from .workload._workload_envs import (
    default_baseline_config,
)
from .workload._agentx_prefix_cache import refuse_cacheless_agentx_baseline
from .benchmark_config import prepare_agentx_runtime
from .benchmark_run.accuracy import _EVAL_FAILURE_MARKERS, _BaselineAccuracyMixin
from .benchmark_run.launch import _BenchmarkLaunchMixin
from .benchmark_run.records import (
    _BENCHMARK_MODE_ATTR,
    BenchmarkAttempt,
    _BenchmarkRound,
    _config_framework,
    _requested_framework,
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
#: The multi-node warmup pass: its directory below the round's and its progress label.
_MN_WARMUP = "mn_warmup"
_PROGRESS_UNIT = "baseline_round"


def _unaffordable_mn_round_result(round_: _BenchmarkRound, evidence: dict[str, Any]) -> dict[str, Any]:
    """The stopped result of a multi-node round whose two passes the session budget cannot afford."""
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


class BenchmarkRunExecutor(_BaselineAccuracyMixin, _BenchmarkLaunchMixin, _BenchmarkRoundsMixin):
    """One benchmark action: materialize the config, launch the backend, parse the report."""

    benchmark_watchdog = False
    runs_accuracy_eval = True
    default_timeout_sec: int | None = None
    session_dir = SessionDirField()

    def __init__(
        self,
        *,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        shared_state: Any | None = None,
    ):
        """Initialize the executor with launch defaults."""
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

    def _prepare_point(self, config_path: Path, output_dir: Path) -> None:
        """Hook that finishes the materialized point before its rounds run it."""

    @staticmethod
    def _is_eval_rooted_failure(result: dict[str, Any]) -> bool:
        """Whether a failed baseline result was caused by the accuracy eval."""
        if result.get("error_class") == FailedStep.EVAL_FAILED.value:
            return True
        return BenchmarkRunExecutor._failure_carries_markers(result, _EVAL_FAILURE_MARKERS)

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the action inside the session the context names."""
        from hyperloom.inference_optimizer.session.session_binding import named_session_scope

        # Only a context that names its session binds one.
        with named_session_scope((getattr(ctx, "extra", None) or {}).get("session_dir")):
            return await self._run(ctx)

    async def _run(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the action inside the bound session."""
        return await self._run_with_accuracy(ctx)

    async def _run_pass(self, ctx: RunnerContext, *, recorder: Any) -> dict[str, Any]:
        """Run :meth:`_run_once`, recorded as the action's run row."""
        if recorder is None:
            return await self._run_once(ctx)
        recorder.begin_run()
        try:
            result = await self._run_once(ctx, recorder=recorder)
        except BaseException as exc:
            recorder.end_run(result={"status": "failed", "error_class": type(exc).__name__, "error": repr(exc)})
            raise
        recorder.end_run(result=result)
        return result

    async def _attempt_refusal(self, attempt: BenchmarkAttempt, output_dir: Path, shared_state: Any) -> dict | None:
        """Why the attempt fails before any round: its AgentX client cannot run it."""
        config_path = attempt.materialized_config_path
        # MLPerf preflights its harness off-loop.
        agx_err = await asyncio.to_thread(
            prepare_agentx_runtime,
            env=os.environ,
            config_path=config_path,
            active=agentx_active(benchmark_mode=getattr(shared_state, _BENCHMARK_MODE_ATTR, "")),
        )
        if agx_err:
            agentx = {"error_class": AGENTX_PREFLIGHT_ERROR_CLASS, "error": agx_err}
            return {"status": "failed", **agentx, "output_dir": str(output_dir)}
        return None

    async def _run_attempt_rounds(self, attempt: BenchmarkAttempt, output_dir: Path) -> dict[str, Any]:
        params = attempt.params
        config_path = attempt.materialized_config_path
        extra = getattr(attempt.ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or self.shared_state
        await asyncio.to_thread(self._prepare_point, config_path, output_dir)
        try:
            attempt = replace(attempt, suite=suite_route(self.session_dir))
        except SuiteRoundRefused as exc:
            log.error("baseline_executor: round refused before it ran: %s", exc)
            return {
                "status": "failed",
                "error_class": exc.error_class,
                "error": str(exc),
                "output_dir": str(output_dir),
            }
        if (refusal := await self._attempt_refusal(attempt, output_dir, shared_state)) is not None:
            return refusal
        # Cold-start "warmup artifact" guard: the freshly-booted server's first benchmark window pays one-time cold
        # costs that inflate later gains into fictitious "improvements". A shape capture records one round and turns
        # the guard off.
        double_run = warms_up(config_path) and (
            "baseline_double_run" not in params or is_truthy(params["baseline_double_run"])
        )
        # Whether THIS run actually executes lm-eval, read back from the materialized config the subprocess
        # consumes -- the single source of truth.
        attempt = replace(attempt, run_eval_disabled=materialized_run_eval_disabled(config_path))
        refused = self._round_refused_before_ignition(attempt, output_dir, double_run=double_run, ctx_extra=extra)
        if refused is not None:
            return refused
        application = self._prepare_round_patches(params, output_dir, shared_state)
        if isinstance(application, Mapping) and application.get("status") == "failed":
            return self._roll_back_failed_patches(application, params, output_dir, shared_state)
        try:
            if double_run:
                return await self._run_cold_start_guarded_rounds(attempt, output_dir, application)
            result = await self._run_reported_round(attempt, label=ROUND_SINGLE, output_dir=output_dir)
            _stamp_warm_patch_outcome(result, application, params)
            return result
        finally:
            # Revert warm-replay patches to prevent state leakage into subsequent tasks that reuse the same
            # InferenceX checkout. A required timeline's tree is promoted by prelude after this returns, so it must
            # stay patched; reverting it here handed prelude a clean tree and silently lost the replay.
            if application and not isinstance(application, Mapping):
                _revert_legacy_warm_patch_trees(params)

    def _round_affordable_before_ignition(
        self,
        *,
        double_run: bool,
        ctx_extra: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Whether the budget holds a whole round *and a use for it*, before anything boots."""
        state = (ctx_extra or {}).get("shared_state") or self.shared_state
        past_the_line, reserve_evidence = _phase_state.a_retry_past_the_reserve_line(state)
        if past_the_line:
            return False, reserve_evidence
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
        if _phase_state.a_use_must_follow_the_round(state):
            # Without the split, a variant is priced at a whole cold round, which is what it is: a boot and a
            # benchmark that pays the compile.
            use_sec = _phase_state.one_more_measurement_sec(state) or cold_sec
        headroom_sec, evidence = _phase_state.round_headroom_sec(state)
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
        self, warmup_result: Mapping[str, Any], *, ctx_extra: dict[str, Any] | None = None
    ) -> tuple[bool, dict[str, Any]]:
        """Whether the budget covers the measured round *and a use for it*, priced from the cold *warmup_result*."""
        state = (ctx_extra or {}).get("shared_state") or self.shared_state
        headroom_sec, evidence = _phase_state.round_headroom_sec(state)
        if headroom_sec is None:
            return True, evidence
        warmup_sec = optional_positive_float(warmup_result.get("subprocess_runtime_sec"))
        session_hot_sec = _phase_state.measured_seconds(state, "baseline_warm_runtime_sec")
        priced_by = "warmup_post_ready" if session_hot_sec is None else "session_hot_pass"
        benchmark_sec = _phase_state.hot_pass_cost_sec(session_hot_sec, warmup_result.get("post_ready_runtime_sec"))
        if benchmark_sec is None or warmup_sec is None:
            return True, {"reason": "no_measured_benchmark_to_predict_from", **evidence}
        use_sec = 0.0
        if _phase_state.a_use_must_follow_the_round(state):
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

    async def _mn_warmup_pass(
        self, round_: _BenchmarkRound, round_warnings: list[str], suite: SuiteRoute
    ) -> dict[str, Any] | None:
        """Run the discarded multi-node client warmup against the restarted server on ``suite``."""
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
            return _unaffordable_mn_round_result(round_, evidence)
        warm_dir = round_.output_dir / _MN_WARMUP
        started_unix = time.time()
        warm_rc = await self._mn_suite_warmup(round_, warm_dir, suite, round_warnings)
        return _stopped_round_result(
            warm_rc,
            round_label="multi-node warmup pass",
            runtime_sec=max(0.0, time.time() - started_unix),
            output_dir=round_.output_dir,
            capture_meta=round_.capture_meta,
        )

    async def _mn_suite_warmup(
        self, round_: _BenchmarkRound, warm_dir: Path, suite: SuiteRoute, round_warnings: list[str]
    ) -> int:
        """Run the warmup pass as an external inference_suite round; a pass that fails warms nothing."""
        warm_round = GridRound(
            config_path=round_.config_path,
            output_dir=warm_dir,
            timeout_sec=int(round_.timeout_sec),
            server_already_ready=True,
            session_deadline_sec=round_.session_deadline_sec,
            warmup=True,
        )
        try:
            async with heartbeat_while_output_flows(unit=_PROGRESS_UNIT, label=_MN_WARMUP) as activity:
                ended = await asyncio.to_thread(
                    run_grid_suite_round,
                    suite,
                    warm_round,
                    env_drop=frozenset(),
                    on_poll=activity.note,
                )
        except SuiteRoundRefused as exc:
            log.warning("baseline_executor: MN warmup pass refused (ignored): %s", exc)
            round_warnings.append(_MN_WARMUP_DID_NOT_WARM_WARNING)
            return 1
        log.info("baseline_executor: MN warmup pass done (discarded) rc=%s", ended.returncode)
        if ended.returncode:
            round_warnings.append(_MN_WARMUP_DID_NOT_WARM_WARNING)
        return ended.returncode

    async def _run_single_benchmark(
        self, attempt: BenchmarkAttempt, *, output_dir: Path, held: HeldPass | None = None
    ) -> dict[str, Any]:
        """Run one benchmark round of the attempt's config and parse its result.

        ``held`` is the round's place in the test that holds its server; without one the round owns its server.
        """
        from ._multi_node_env import is_multi_node, log_mn_banner, mn_bench_warmup_enabled
        from .agentx_point import bind_source_evidence

        config_path = attempt.materialized_config_path
        await asyncio.to_thread(bind_source_evidence, config_path, attempt.params.get("patches") or [])
        ctx_extra = getattr(attempt.ctx, "extra", None) or {}
        # The session's wall-clock budget, resolved per round rather than once per task: a baseline runs up to two
        # of them, and a warmup that overran has already spent budget the one after it was counting on.
        session_deadline_sec, _ = session_grid_bounds(ctx_extra.get("shared_state") or self.shared_state)
        timeout_sec = resolve_benchmark_timeouts()[1] if self.benchmark_watchdog else attempt.timeout_sec
        sync_benchmark_timeout(config_path, timeout_sec)
        # The materialized YAML is authoritative for the framework: params and $FRAMEWORK are both optional on a
        # baseline task.
        framework = _config_framework(config_path) or (_requested_framework(attempt.params) or "").lower()
        round_ = _BenchmarkRound(
            config_path=config_path,
            output_dir=output_dir,
            ctx_extra=ctx_extra,
            framework=framework,
            watchdog_server_log=_watchdog_server_log_path(output_dir, framework),
            # Config/eval-contract facts echoed onto every failure result so an eval-rooted failure can be
            # fingerprinted and re-run by enablement.
            capture_meta={
                "materialized_config": str(config_path),
                "run_eval_disabled": bool(attempt.run_eval_disabled),
            },
            timeout_sec=timeout_sec,
            session_deadline_sec=session_deadline_sec,
            held=held,
            run_eval_disabled=attempt.run_eval_disabled,
        )
        # Multi-node only: restart sglang/vllm per round for a fresh server.
        restart_failure = await self._restart_multi_node_server(
            round_, server_args=attempt.effective_extra_server_args, resolved_model=attempt.resolved_model
        )
        if restart_failure is not None:
            return restart_failure
        log_mn_banner("baseline_executor", log, output_dir=str(output_dir))
        round_warnings: list[str] = []
        if is_multi_node() and mn_bench_warmup_enabled() and not ctx_extra.get("mn_round_restarted"):
            warmup = await self._mn_warmup_pass(round_, round_warnings, attempt.suite)
            if warmup is not None:
                return warmup
        return await self._suite_round_result(round_, attempt.suite, warnings=round_warnings)


class BaselineExecutor(BenchmarkRunExecutor):
    """Class form for tests / DI; ``baseline_executor`` is the bare callable."""

    benchmark_watchdog = True

    def _resolve_default_config(self) -> Path:
        """Resolve the benchmark YAML the round runs when the task names none."""
        return default_baseline_config()

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
            result = await self._run_with_accuracy(ctx, recorder=recorder)
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
