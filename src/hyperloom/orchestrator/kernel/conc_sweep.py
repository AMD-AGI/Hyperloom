# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Concurrency sweep over the CONC ladder."""

from __future__ import annotations
from hyperloom.orchestrator.actions.executors._workload_envs import prepare_grid_benchmark_config

import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from hyperloom.common import io as _common_io
from hyperloom.common.gain_math import conc_pair_comparison
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import GRADED_INTVTY, GRADED_OUTPUT, is_agentx_mode
from hyperloom.common.timeutil import now_iso, utc_now_compact
from hyperloom.inference_optimizer.breakdown.recorder.conc_sweep_event import (
    GRID_MODE_DEFAULT,
    GRID_REQUESTED,
    STAGE_BUDGET_SKIP,
    STAGE_SERVER_RESTART,
    STRATEGY_SERVER_RESTART,
    ArmStrategy,
    SweepAnchor,
    SweepBudget,
    SweepEnvironment,
    SweepPlan,
    SweepRung,
)
from hyperloom.inference_optimizer.grading import resolved_grading
from hyperloom.inference_optimizer.session.session_paths import reports_dir, runs_root
from ..actions.executors._grid_runner import (
    GridVariant,
    VariantResult,
    run_grid,
    session_grid_bounds,
    variant_rounds,
)
from ..actions.executors._subprocess_kill import resolve_benchmark_timeouts, session_deadline_to_remaining_sec
from ..actions.cancel_channel import current_cancel_scope
from ..actions.stop_attribution import (
    ORCHESTRATOR_CANCELLED_CLASS,
    SESSION_TIME_EXHAUSTED_CLASS,
    STOPPED_BY_THE_RUN,
    StoppedByTheRun,
)
from ..actions.executors._workload_envs import (
    FrameworkScriptMismatchError,
    default_baseline_config,
)
from ..actions.executors._proposal_identity import controls_of, is_executable, normalize_proposal
from hyperloom.inference_optimizer.roofline_ceiling import (
    ComputeTarget,
    DecodeShape,
    compute_compute_bound_ceiling_tok_per_sec,
    compute_theoretical_peak_output_tok_per_sec,
    load_model_meta,
    select_peak_and_bound,
)
from ..state.shared_state import SharedState


log = logging.getLogger(__name__)


def _grading_of(state: Any) -> tuple[str, float | None]:
    """The axis this sweep draws its speedups on, and the noise band its guard reads.

    Resolved through ``resolved_grading`` so the curve and the promotions in one session cannot end up on
    different axes. The environment-derived ``graded_metric_key`` diverges two ways: it never sees the axis
    recorded at seed, so a resume whose shell lost ``HYPERLOOM_PERF_METRIC`` redraws the curve on output; and it
    has no scriptable carve-out, so an image framework -- which reports no interactivity axis at all -- would
    compare every rung on a field it never measures and report the whole sweep as failed.
    """
    on_intvty, noise_pct = resolved_grading(state)
    return (GRADED_INTVTY if on_intvty else GRADED_OUTPUT), noise_pct


SCHEMA_VERSION = "1.0"

# Default ladders, one per workload (override via ``--conc-sweep-concs``).
DEFAULT_CONCS: list[int] = [256, 128, 64, 32, 16, 8, 4, 2]
AGENTX_DEFAULT_CONCS: list[int] = [1, 4, 8, 10, 14, 20, 28]


def default_concs_for_mode(benchmark_mode: Any = "") -> list[int]:
    """The ladder a mode sweeps when the operator names none."""
    return list(AGENTX_DEFAULT_CONCS if is_agentx_mode(benchmark_mode) else DEFAULT_CONCS)


# Multiplier applied to each CONC for NUM_PROMPTS.
DEFAULT_NUM_PROMPTS_FACTOR = 5

# Total wall-clock budget (seconds); override via ``--conc-sweep-total-budget-sec``.
DEFAULT_TOTAL_BUDGET_SEC = 9000


@dataclass(frozen=True)
class _Arm:
    """One side of the comparison: the launch recipe every rung of its ladder runs with."""

    name: str
    args: str = ""
    envs: dict[str, str] = field(default_factory=dict)
    overlay: str = ""
    controls: Mapping[str, Any] = field(default_factory=dict)


def _optimized_arm(state: SharedState) -> _Arm | None:
    """The arm for the retained config or kernel overlay, or ``None`` when there is nothing to compare."""
    cb = state.current_best or {}
    config = normalize_proposal(cb)
    overlay = str(cb.get("final_overlay") or "").strip()
    if not (is_executable(config) or overlay):
        return None
    return _Arm(
        name="optimized",
        args=config["extra_args"],
        envs=config["extra_envs"],
        overlay=overlay,
        controls=controls_of(config),
    )


def _budget_skip_result(variant: GridVariant) -> VariantResult:
    """Synthetic VariantResult for a budget-exhausted variant; ``skipped`` status distinguishes \"out of time\" from \"Magpie crashed\"."""
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status="skipped",
        output_throughput=None,
        request_throughput=None,
        total_token_throughput=None,
        error="conc_sweep total budget exhausted before this variant ran",
        error_class="budget_exhausted",
        note=variant.note,
    )


_SWEEP_BUDGET_STOP = StoppedByTheRun(
    error_class="budget_exhausted",
    interrupted="conc_sweep total budget exhausted while this round was running",
    never_started="conc_sweep total budget exhausted before this round ran",
    ends_the_batch=True,
)


# The orchestrator stopped the sweep's action from outside -- a phase transition or a shutdown. Every rung after the
# one it interrupted is recorded under this cause and never run.
_CANCEL_STOP = STOPPED_BY_THE_RUN[ORCHESTRATOR_CANCELLED_CLASS]


def _deadline_skip_result(variant: GridVariant, stopped: StoppedByTheRun) -> VariantResult:
    """Keep the owner of the deadline on every rung that never ran."""
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status="skipped",
        error=stopped.never_started,
        error_class=stopped.error_class,
        note=variant.note,
    )


def _point_from_variant(v: VariantResult, *, arm: str) -> dict[str, Any]:
    """Flatten a ``VariantResult`` into one row of the curve."""
    envs = v.extra_envs or {}
    # The grid sets CONC from an int, and a curve is only readable if every
    # rung names the concurrency it was measured at -- so a value that will not
    # parse is a bug to surface, not a rung to file under zero.
    conc = int(envs.get("CONC", "0"))
    # aiperf reports the total; the other parsers pass through whatever the framework named, leaving it null on a run
    # that measured both halves.
    total = v.total_token_throughput
    if total is None and v.input_throughput is not None and v.output_throughput is not None:
        total = v.input_throughput + v.output_throughput
    return {
        "arm": arm,
        "conc": conc,
        "status": v.status,
        "output_throughput": v.output_throughput,
        "request_throughput": v.request_throughput,
        "total_token_throughput": total,
        "input_throughput": v.input_throughput,
        "e2e_norm_intvty_p90": v.intvty_p90,
        "tpot_p90_ms": v.tpot_p90_ms,
        "ttft_mean_ms": v.ttft_mean_ms,
        "e2el_mean_ms": v.e2el_mean_ms,
        "duration_seconds": v.duration_seconds,
        "completed_requests": v.completed_requests,
        "error": v.error,
        "error_class": v.error_class,
        "killed_overtime": v.killed_overtime,
        "estimated_output_throughput": v.estimated_output_throughput,
        "workspace": v.workspace,
        "report_path": v.report_path,
    }


def _budget_limited_without_valid_pair(
    *,
    budget_exhausted: bool,
    summary: dict[str, Any],
    baseline_points: list[dict[str, Any]],
    optimized_points: list[dict[str, Any]],
) -> bool:
    """Return true when budget gating, not benchmark failure, prevented all pairs."""
    if not budget_exhausted or int(summary.get("successful_pairs") or 0) > 0:
        return False
    points = baseline_points + optimized_points
    if not points:
        return False
    saw_budget_skip = False
    for point in points:
        status = str(point.get("status") or "").lower()
        error_class = str(point.get("error_class") or "")
        if error_class in {"budget_exhausted", SESSION_TIME_EXHAUSTED_CLASS}:
            saw_budget_skip = True
            continue
        if status not in ("succeeded", "skipped"):
            return False
    return saw_budget_skip


def _write_csv(csv_path: Path, points: list[dict[str, Any]]) -> None:
    """One row per (arm, conc) — flat columns for spreadsheet pivots."""
    columns = [
        "arm",
        "conc",
        "status",
        "output_throughput",
        "request_throughput",
        "total_token_throughput",
        "input_throughput",
        "e2e_norm_intvty_p90",
        "tpot_p90_ms",
        "ttft_mean_ms",
        "e2el_mean_ms",
        "duration_seconds",
        "completed_requests",
        "error_class",
        "error",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for p in points:
            writer.writerow({k: p.get(k) for k in columns})


@dataclass(frozen=True)
class _ArmCurves:
    """The measured points of both arms of a sweep."""

    baseline: list[dict[str, Any]]
    optimized: list[dict[str, Any]]


def _build_roofline_ceiling(
    state: SharedState,
    curves: _ArmCurves,
    *,
    concs: list[int],
    isl: int,
    osl: int,
) -> dict[str, Any] | None:
    """Per-conc decode roofline alongside the measured curves."""
    model_path = str(getattr(state, "model_path", "") or "")
    precision = str(getattr(state, "precision", "") or "") or "bf16"
    meta = load_model_meta(model_path, precision_hint=precision)
    if meta is None:
        return None
    gpu_type = str(getattr(state, "gpu_type", "") or "")
    num_gpus = int(getattr(state, "tp", 0) or 0)
    if not gpu_type or num_gpus <= 0:
        return None

    t_cmp = compute_compute_bound_ceiling_tok_per_sec(
        ComputeTarget(gpu_type=gpu_type, num_gpus=num_gpus, precision_tag=precision),
        active_weight_bytes=meta.active_weight_bytes,
        weight_bytes=meta.weight_bytes,
        weight_dtype_bytes=meta.weight_dtype_bytes,
    )

    by_conc_b = {p["conc"]: p for p in curves.baseline}
    by_conc_o = {p["conc"]: p for p in curves.optimized}

    rows: list[dict[str, Any]] = []
    for c in concs:
        t_mem = compute_theoretical_peak_output_tok_per_sec(
            meta,
            DecodeShape(isl=isl, osl=osl, concurrency=c),
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            kv_dtype_bytes=meta.weight_dtype_bytes,
        )
        t_peak, bound_kind = select_peak_and_bound(t_mem, t_cmp)
        # Local import avoids a module-level import cycle.
        from hyperloom.inference_optimizer.roofline_snapshot import within_roofline_pct

        def _mbu_pct(measured: Any) -> float | None:
            """Express a measured throughput as a percent of peak."""
            if not isinstance(measured, (int, float)) or measured <= 0:
                return None
            return within_roofline_pct(peak=float(t_peak), achieved=float(measured))

        # Output throughput on both arms regardless of the graded axis, and not a bug to be aligned with it: the
        # peak above is a memory-bandwidth-derived ceiling on output tokens per second, so MBU is only meaningful
        # against the same quantity.
        bt = (by_conc_b.get(c) or {}).get("output_throughput")
        ot = (by_conc_o.get(c) or {}).get("output_throughput")
        rows.append(
            {
                "conc": c,
                "t_mem_tok_s": round(t_mem, 2),
                "t_cmp_tok_s": round(t_cmp, 2),
                "t_peak_tok_s": round(t_peak, 2),
                "bound_kind": bound_kind,
                "mbu_baseline_pct": _mbu_pct(bt),
                "mbu_optimized_pct": _mbu_pct(ot),
            }
        )

    return {
        "schema_version": 1,
        "source": "roofline_ceiling.py",
        "gpu_type": gpu_type,
        "precision": precision,
        "tp": num_gpus,
        "isl": isl,
        "osl": osl,
        "model_meta": {
            "weight_bytes": meta.weight_bytes,
            "active_weight_bytes": meta.active_weight_bytes,
            "num_experts": meta.num_experts,
            "experts_per_tok": meta.experts_per_tok,
            "expert_weight_bytes": meta.expert_weight_bytes,
            "num_layers": meta.num_layers,
            "num_kv_heads": meta.num_kv_heads,
            "head_dim": meta.head_dim,
            "weight_dtype_bytes": meta.weight_dtype_bytes,
        },
        "rows": rows,
    }


def _arm_status(results: list[VariantResult]) -> str:
    """The status an arm reports from the rungs it actually ran.

    An arm that measured some rungs and lost others is ``degraded`` rather
    than either extreme: the curve it produced is real and shorter than the
    ladder it was asked for.
    """
    statuses = {str(result.status or "").lower() for result in results}
    if not statuses:
        return "skipped"
    if "succeeded" in statuses:
        return "degraded" if statuses - {"succeeded"} else "succeeded"
    if statuses <= {"skipped"}:
        return "skipped"
    return "failed"


def _order_concs_desc(concs: list[int]) -> list[int]:
    """Return a strictly descending, deduplicated copy of the CONC ladder."""
    return sorted(set(concs), reverse=True)


def _build_arm_grid(
    arm: _Arm,
    concs_desc: list[int],
    *,
    isl: int,
    osl: int,
    num_prompts_factor: int,
) -> list[GridVariant]:
    """Build a single-arm grid in descending CONC order."""
    out: list[GridVariant] = []
    for conc in concs_desc:
        num_prompts = max(int(conc) * int(num_prompts_factor), int(conc))
        envs = dict(arm.envs)
        envs.update(
            {
                "CONC": str(conc),
                "ISL": str(isl),
                "OSL": str(osl),
                "NUM_PROMPTS": str(num_prompts),
            }
        )
        envs["RUN_EVAL"] = "false"
        variant = GridVariant(
            name=f"{arm.name}_conc{conc}",
            extra_server_args=arm.args,
            extra_envs=envs,
            note=f"arm={arm.name} conc={conc} isl={isl} osl={osl}",
            **arm.controls,
        )
        variant.overlay_pythonpath = arm.overlay  # type: ignore[attr-defined]
        out.append(variant)
    return out


_SESSION_RESERVE = "session_deadline_reserve"


@dataclass
class _Budget:
    """Whether the sweep's time has run out, and why."""

    skip_reason: str = ""
    remaining_sec: float = 0.0

    @property
    def exhausted(self) -> bool:
        return bool(self.skip_reason)

    def exhaust(self, reason: str, remaining_sec: float) -> None:
        self.skip_reason = reason
        self.remaining_sec = remaining_sec

    def spend_on_session_reserve(self) -> None:
        self.exhaust(_SESSION_RESERVE, 0.0)


@dataclass(frozen=True)
class _RungBounds:
    """The deadline a rung runs under and the cost ``run_grid`` admits it against."""

    deadline_sec: float | None
    expected_sec: float | None


@dataclass(frozen=True)
class _RungNote:
    """How one rung came to run: ``stage`` is a ``STAGE_*`` value and the seconds fields are wall clock."""

    stage: str
    start_time: str = ""
    wall_duration_sec: float | None = None
    budget_remaining_sec: float | None = None


@dataclass
class _SweepRun:
    """One conc sweep's fixed inputs, and the progress both of its arms add to.

    ``results`` and ``budget`` are shared across arms on purpose: every incremental flush reports the whole sweep so
    far, and a budget stop in either arm has to stop every rung after it. ``session_deadline_sec`` is the earlier monotonic
    sweep/session boundary, named by ``deadline_stop`` and distinct from the fixed per-spawn ``benchmark_timeout_sec``;
    ``grid_deadline_sec`` is the session's own deadline alone (``None`` when the session has none), the line no rung may
    be extended past.
    """

    state: SharedState
    session_dir: Path
    workspace: Path
    base_yaml_path: Path
    model_path: str
    gpu_type: str
    benchmark_script: str | None
    isl: int
    osl: int
    concs: list[int]
    num_prompts_factor: int
    optimized: _Arm
    baseline: _Arm
    benchmark_timeout_sec: float
    session_deadline_sec: float | None
    variant_expected_sec: float | None
    deadline_stop: StoppedByTheRun
    grid_deadline_sec: float | None
    started_at: float
    total_budget_sec: int | None
    json_path: Path
    csv_path: Path
    recorder: Any = None
    results: list[VariantResult] = field(default_factory=list)
    budget: _Budget = field(default_factory=_Budget)
    cancel_reason: str = ""

    @property
    def cancelled(self) -> bool:
        """Whether the orchestrator has stopped this sweep's action."""
        return bool(self.cancel_reason)

    def observe_cancel(self, results: Sequence[VariantResult] = ()) -> bool:
        """Latch a cancel seen on the action's scope or on a rung it reaped; return whether the sweep is cancelled.

        A cancelled scope stays cancelled, so every rung after it would be reaped the moment it started. Latching here
        is what lets the ladder stop instead of reading each of those reaps as a rung that failed.
        """
        if self.cancelled:
            return True
        scope = current_cancel_scope()
        if scope is not None and scope.cancelled:
            self.cancel_reason = scope.reason or ORCHESTRATOR_CANCELLED_CLASS
        elif any(result.error_class == ORCHESTRATOR_CANCELLED_CLASS for result in results):
            self.cancel_reason = ORCHESTRATOR_CANCELLED_CLASS
        return self.cancelled

    def stop_named_by(self, result: VariantResult | None) -> StoppedByTheRun | None:
        """The run-side stop a rung's result names, or ``None`` for a rung that ran or failed on its own."""
        if result is None:
            return None
        if result.error_class == self.deadline_stop.error_class:
            return self.deadline_stop
        if result.error_class == ORCHESTRATOR_CANCELLED_CLASS:
            return _CANCEL_STOP
        return None

    @property
    def concs_desc(self) -> list[int]:
        return _order_concs_desc(self.concs)

    def arm_grid(self, arm: _Arm) -> list[GridVariant]:
        return _build_arm_grid(
            arm, self.concs_desc, isl=self.isl, osl=self.osl, num_prompts_factor=self.num_prompts_factor
        )

    def arm_results(self, arm_name: str) -> list[VariantResult]:
        prefix = f"{arm_name}_"
        return [result for result in self.results if result.name.startswith(prefix)]

    def record_rung(self, arm_name: str, result: VariantResult, note: _RungNote) -> None:
        """Record one rung on the sweep's event, if the sweep is being recorded.

        The result is flattened into the same point the report writes, so the
        recorded curve and the written one cannot differ.
        """
        if self.recorder is None:
            return
        point = _point_from_variant(result, arm=arm_name)
        self.recorder.record_variant(
            arm_name,
            SweepRung(
                stage=note.stage,
                conc=point.get("conc"),
                point=point,
                num_prompts=(result.extra_envs or {}).get("NUM_PROMPTS"),
                start_time=note.start_time,
                wall_duration_sec=note.wall_duration_sec,
                granted_cap_sec=self.benchmark_timeout_sec,
                budget_remaining_sec=note.budget_remaining_sec,
            ),
        )

    def close_arm(self, arm_name: str, exc: BaseException | None) -> None:
        """Close one arm's row, whichever of its ladder's paths it leaves by.

        An arm that raised mid-ladder says so once, rather than reporting a ladder
        it never finished as the status its own results happen to add up to.
        """
        if self.recorder is None:
            return
        if exc is None:
            self.recorder.finish_arm(arm_name, status=_arm_status(self.arm_results(arm_name)))
        else:
            self.recorder.fail_arm(arm_name, exc)

    async def run_rung(
        self,
        variant: GridVariant,
        *,
        serving_lease: Any,
        bounds: _RungBounds | None = None,
    ) -> list[VariantResult]:
        """Run one rung from the sweep's base config; a deadline stop inside it exhausts the sweep's budget.

        ``bounds`` replaces the sweep's own deadline and admission cost for this rung (see :meth:`partner_bounds`).
        """
        if bounds is None:
            bounds = _RungBounds(self.session_deadline_sec, self.variant_expected_sec)
        if self.observe_cancel():
            return [_deadline_skip_result(variant, _CANCEL_STOP)]
        if self.budget.exhausted:
            return [_deadline_skip_result(variant, self.deadline_stop)]
        results = await run_grid(
            base_yaml_path=self.base_yaml_path,
            base_extra_args="",
            grid=[variant],
            output_root=self.workspace,
            model_path=self.model_path,
            gpu_type=self.gpu_type,
            benchmark_script=self.benchmark_script,
            serving_lease=serving_lease,
            session_deadline_sec=bounds.deadline_sec,
            variant_expected_sec=bounds.expected_sec,
            deadline_stop=self.deadline_stop,
        )
        self.observe_cancel(results)
        if any(result.error_class == self.deadline_stop.error_class for result in results):
            self._exhaust_on_deadline(session_deadline_to_remaining_sec(self.session_deadline_sec))
        return results

    def _exhaust_on_deadline(self, remaining: float | None) -> None:
        """Stop the sweep on its deadline, naming whether the time ran out or only cannot fit what comes next."""
        reason = self.deadline_stop.error_class
        if self.deadline_stop == _SWEEP_BUDGET_STOP:
            reason = (
                "total_budget_exhausted"
                if remaining is not None and remaining <= 0
                else "insufficient_remaining_for_variant"
            )
        self.budget.exhaust(reason, max(0.0, remaining) if remaining is not None else 0.0)

    def admit_level(self, arms: int) -> None:
        """Stop the sweep before a CONC level when what is left cannot admit one rung of each of ``arms`` arms.

        ``run_grid`` admits a rung against ``variant_expected_sec`` times the benchmark passes it runs
        (:func:`variant_rounds`); a level runs one such rung per arm, so it starts only on ``arms`` times that, and a
        level the estimate says would end with one arm measured is not started. A first rung that overruns the estimate
        can still leave its partner refused by ``run_grid``'s own gate. With no estimate a level starts
        while any time is left, as a rung does; a sweep already stopped leaves its rungs to record why.
        """
        if self.budget.exhausted or self.cancelled or self.session_deadline_sec is None:
            return
        remaining = float(self.session_deadline_sec) - time.monotonic()
        rung_sec = None if self.variant_expected_sec is None else float(self.variant_expected_sec) * variant_rounds()
        if remaining > 0 and (rung_sec is None or remaining >= arms * rung_sec):
            return
        log.warning(
            "conc_sweep: %.0fs left cannot admit a rung of each of %d arms (%s s each); not starting this level",
            max(0.0, remaining),
            arms,
            rung_sec,
        )
        self._exhaust_on_deadline(remaining)

    def partner_bounds(self) -> _RungBounds:
        """Bounds for the second arm of a level whose first arm has run, so the pair is never left half measured.

        The level was admitted on the estimate for both rungs; a first rung that overran it must not cost its partner
        the admission it was promised. The sweep's own budget deadline is extended to twice the time one rung is
        estimated to take, which clears ``run_grid``'s gate and still lets the partner be stopped if it hangs. The
        extension never passes the session's grid deadline, so the partner cannot run into the reserve the session keeps
        for closing; when the session clock is already the binding limit there is nothing to extend, and a partner the
        gate refuses is recorded as an unpaired level. A session that is closing or cancelled still stops it before it
        starts.
        """
        if self.session_deadline_sec is None or self.variant_expected_sec is None:
            return _RungBounds(self.session_deadline_sec, self.variant_expected_sec)
        rung_sec = float(self.variant_expected_sec) * variant_rounds()
        deadline = max(float(self.session_deadline_sec), time.monotonic() + 2 * rung_sec)
        if self.grid_deadline_sec is not None:
            deadline = min(deadline, float(self.grid_deadline_sec))
        if deadline > float(self.session_deadline_sec):
            log.info(
                "conc_sweep: extending the partner rung's deadline by %.0fs (sweep budget %.0fs left, session %s)",
                deadline - float(self.session_deadline_sec),
                max(0.0, float(self.session_deadline_sec) - time.monotonic()),
                "uncapped"
                if self.grid_deadline_sec is None
                else f"{self.grid_deadline_sec - time.monotonic():.0f}s left",
            )
        return _RungBounds(deadline, self.variant_expected_sec)

    def session_closing(self) -> bool:
        return bool(self.state.closing_phase or self.state.stop_reason)

    def _arm_section(self, arm: _Arm) -> dict[str, Any]:
        points = [_point_from_variant(result, arm=arm.name) for result in self.arm_results(arm.name)]
        points.sort(key=lambda p: p["conc"])
        return {"extra_server_args": arm.args, "extra_envs": arm.envs, "points": points}

    def payload(self, *, in_progress: bool) -> dict[str, Any]:
        """The ``conc_sweep_summary.json`` body for every rung recorded so far.

        An in-progress payload is the one rewritten between rungs; a final one settles its status from the pairs.
        """
        state = self.state
        baseline = self._arm_section(self.baseline)
        optimized = self._arm_section(self.optimized)
        metric_key, guard_noise_pct = _grading_of(state)
        comparison, summary = conc_pair_comparison(
            baseline["points"], optimized["points"], metric_key=metric_key, guard_noise_pct=guard_noise_pct
        )
        budget_limited_no_pair = not in_progress and _budget_limited_without_valid_pair(
            budget_exhausted=self.budget.exhausted,
            summary=summary,
            baseline_points=baseline["points"],
            optimized_points=optimized["points"],
        )
        if in_progress:
            status = "in_progress"
        elif self.cancelled:
            # Stopped from outside, not failed: whatever ended the phase is the reason the session records, and a
            # ladder it cut short is not evidence that the stack cannot serve.
            status = "cancelled"
        elif summary["successful_pairs"]:
            status = "succeeded"
        else:
            status = "skipped" if budget_limited_no_pair else "failed"
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "session_id": str(getattr(state, "session_id", "") or self.session_dir.name),
            "isl": self.isl,
            "osl": self.osl,
            "tp": int(getattr(state, "tp", 0) or 0),
            # Names the axis pair the points are drawn on, so a reader never has to infer it from whether
            # e2e_norm_intvty_p90 happens to be null.
            "benchmark_mode": str(getattr(state, "benchmark_mode", "") or ""),
            "concs_requested": list(self.concs),
            "baseline": baseline,
            "optimized": optimized,
            "comparison": comparison,
            "summary": summary,
            "workspace": self.workspace.as_posix(),
            "elapsed_sec": round(time.time() - self.started_at, 2),
            "total_budget_sec": self.total_budget_sec,
            "budget_exhausted": self.budget.exhausted,
        }
        if self.cancelled and not in_progress:
            payload["cancelled"] = True
            payload["cancel_reason"] = self.cancel_reason
        if budget_limited_no_pair:
            payload["was_skipped"] = True
            payload["skip_reason"] = "budget_exhausted_no_successful_pairs"
        if self.budget.exhausted:
            payload["budget_skip_reason"] = self.budget.skip_reason
            payload["budget_remaining_sec"] = round(self.budget.remaining_sec, 2)
        return payload

    def write(self, payload: dict[str, Any]) -> Exception | None:
        """Write ``payload`` as the sweep's report, carrying the paths it is written to."""
        payload["report_json_path"] = self.json_path.as_posix()
        payload["report_csv_path"] = self.csv_path.as_posix()
        return _flush_conc_sweep_report(payload, self.session_dir)

    def flush_progress(self) -> None:
        """Rewrite the report from the rungs so far, recording the pairs measured so far on the same beat."""
        payload = self.payload(in_progress=True)
        if self.recorder is not None:
            self.recorder.record_progress(comparison=payload["comparison"], summary=payload["summary"])
        self.write(payload)


@dataclass
class _Ladder:
    """The arms of one interleaved ladder, which of them have started, and the serving lease they share.

    An arm opens when its first rung is dispatched, so an arm whose every rung the sweep refused records the gate that
    refused it instead of a ladder it never climbed.
    """

    run: _SweepRun
    arms: list[_Arm]
    lease: Any = None
    opened: list[str] = field(default_factory=list)

    def open(self, arm: _Arm) -> None:
        """Start an arm's row the first time one of its rungs is dispatched."""
        if arm.name in self.opened:
            return
        self.opened.append(arm.name)
        recorder = self.run.recorder
        if recorder is None:
            return
        recorder.open_arm(arm.name, extra_server_args=arm.args, extra_envs=arm.envs)
        recorder.record_arm_grid(
            arm.name,
            rungs=[
                {
                    "name": variant.name,
                    "conc": variant.extra_envs.get("CONC"),
                    "num_prompts": variant.extra_envs.get("NUM_PROMPTS"),
                }
                for variant in self.run.arm_grid(arm)
            ],
        )
        recorder.record_arm_strategy(
            arm.name, ArmStrategy(strategy=STRATEGY_SERVER_RESTART, serving_lease_held=self.lease is not None)
        )

    def skip(self, arm_name: str, result: VariantResult, *, budget_remaining_sec: float | None) -> None:
        """Record a rung the sweep did not start; an arm that never opened keeps it off its rung rows."""
        self.run.results.append(result)
        if arm_name not in self.opened:
            return
        self.run.record_rung(
            arm_name, result, _RungNote(stage=STAGE_BUDGET_SKIP, budget_remaining_sec=budget_remaining_sec)
        )

    def close(self, failed: BaseException | None) -> None:
        """Close every arm on its rungs, or on the gate that refused it before it started.

        What raised out of a rung ends both ladders, not only the arm whose rung it was, so every arm says so.
        """
        run = self.run
        for arm in self.arms:
            if failed is not None or arm.name in self.opened:
                run.close_arm(arm.name, failed)
            elif run.recorder is None:
                continue
            elif run.cancelled:
                run.recorder.record_arm_refused(arm.name, reason=run.cancel_reason, remaining_sec=0.0)
            elif run.budget.exhausted:
                run.recorder.record_arm_refused(
                    arm.name, reason=run.budget.skip_reason, remaining_sec=run.budget.remaining_sec
                )


async def _run_level_rung(
    run: _SweepRun, ladder: _Ladder, arm: _Arm, variant: GridVariant, bounds: _RungBounds | None = None
) -> bool:
    """Run, or record as stopped, one arm's rung at the current level, each on a server of its own.

    The two arms alternate on the same GPUs, so no server outlives the rung it was started for. A rung the sweep can
    no longer start -- cancelled, closing, or out of budget -- is recorded as skipped under that cause and never boots.
    Returns whether the rung ran, so the level's next arm knows it is completing a pair.
    """
    rung_remaining = session_deadline_to_remaining_sec(run.session_deadline_sec)
    if run.observe_cancel():
        ladder.skip(arm.name, _deadline_skip_result(variant, _CANCEL_STOP), budget_remaining_sec=None)
        return False
    if run.session_closing():
        run.budget.spend_on_session_reserve()
        ladder.skip(arm.name, _budget_skip_result(variant), budget_remaining_sec=0.0)
        return False
    if run.budget.exhausted:
        ladder.skip(
            arm.name,
            _deadline_skip_result(variant, run.deadline_stop),
            budget_remaining_sec=run.budget.remaining_sec,
        )
        return False
    ladder.open(arm)
    rung_started_iso = now_iso("seconds")
    rung_started_at = time.time()
    try:
        results = await run.run_rung(variant, serving_lease=ladder.lease, bounds=bounds)
    except Exception as exc:  # noqa: BLE001
        results = [
            VariantResult(
                name=variant.name,
                extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                error=f"conc_sweep_rung: {exc}",
                error_class=type(exc).__name__,
            )
        ]
    rung_elapsed = round(time.time() - rung_started_at, 3)
    for result in results:
        run.results.append(result)
        stopped = run.stop_named_by(result) is not None
        run.record_rung(
            arm.name,
            result,
            _RungNote(
                stage=STAGE_BUDGET_SKIP if stopped else STAGE_SERVER_RESTART,
                start_time=rung_started_iso,
                wall_duration_sec=rung_elapsed,
                budget_remaining_sec=run.budget.remaining_sec if stopped else rung_remaining,
            ),
        )
    run.flush_progress()
    return True


async def _sweep_interleaved(run: _SweepRun, arms: list[_Arm]) -> None:
    """Sweep the ladder one CONC level at a time, high to low, running every arm's rung at a level before the next.

    A pair needs both arms measured at one CONC, so a budget that runs out keeps every level already finished. A level
    starts only when what is left admits a rung of every arm (:meth:`_SweepRun.admit_level`), so a level the estimate
    says it cannot finish is not started, and once its first arm has run the rest of the level is admitted whatever that
    rung cost (:meth:`_SweepRun.partner_bounds`), so a level is never left with one arm measured.
    """
    from ..actions.executors._grid_runner import _num_gpus_for_config
    from ..actions.executors._ray_serving import maybe_serving_lease

    grids = [run.arm_grid(arm) for arm in arms]
    # Ray-managed GPU execution: the arms take turns on the same GPUs, so one held lease (``num_gpus=TP``) spans the
    # whole sweep rather than one per arm, which two interleaved arms would hold at once.
    ladder = _Ladder(run=run, arms=arms, lease=maybe_serving_lease(num_gpus=_num_gpus_for_config(run.base_yaml_path)))
    failed: BaseException | None = None
    try:
        for level in range(len(run.concs_desc)):
            run.admit_level(len(arms))
            ran = False
            for arm, grid in zip(arms, grids):
                # Once an arm has run at this level its partner is admitted whatever the first rung cost.
                ran = await _run_level_rung(run, ladder, arm, grid[level], run.partner_bounds() if ran else None)
    except BaseException as exc:
        failed = exc
        raise
    finally:
        try:
            if ladder.lease is not None:
                ladder.lease.close()
        finally:
            ladder.close(failed)


def _flush_conc_sweep_report(payload: dict[str, Any], session_dir: Path) -> Exception | None:
    """Atomically write the conc-sweep summary JSON + CSV to the reports dir.

    Returns what stopped the write, so a caller that recorded the paths can say
    they are where the report was meant to go rather than where it is.
    """
    try:
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        json_path = Path(payload["report_json_path"])
        csv_path = Path(payload["report_csv_path"])
        _common_io.atomic_write_text(
            json_path,
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        )
        all_points: list[dict[str, Any]] = list((payload.get("baseline") or {}).get("points") or []) + list(
            (payload.get("optimized") or {}).get("points") or []
        )
        _write_csv(csv_path, all_points)
    except Exception as exc:
        log.warning("conc_sweep: _flush_conc_sweep_report failed", exc_info=True)
        return exc
    return None


def _skip(reason: str, **extras: Any) -> dict[str, Any]:
    """Build a non-fatal skip envelope. Reason is operator-readable."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "skipped",
        "skip_reason": reason,
    }
    payload.update(extras)
    return payload


def _declined(recorder: Any, reason: str, **extras: Any) -> dict[str, Any]:
    """Build a skip envelope and close the sweep's event on it.

    A sweep that declines is still a sweep that was dispatched. Routing every
    pre-flight refusal through here means none can be added later without the
    event learning about it.
    """
    payload = _skip(reason, **extras)
    if recorder is not None:
        recorder.record_declined(payload)
    return payload


def conc_sweep_declined_to_run(record: Mapping[str, Any] | None) -> bool:
    """Whether a conc-sweep record is one that never started a variant."""
    rec = record or {}
    return bool(rec.get("was_skipped")) and not rec.get("budget_exhausted")


async def run_conc_sweep(
    state: SharedState,
    session_dir: Path,
    *,
    concs: list[int] | None = None,
    total_budget_sec: int | None = DEFAULT_TOTAL_BUDGET_SEC,
    recorder: Any = None,
) -> dict[str, Any]:
    """Run the full conc-sweep SWEEP-phase action end-to-end (always returns a dict; never raises; no files written when skipped).

    ``concs`` of ``None`` uses the default ladder. ``total_budget_sec`` of
    ``None`` removes only the sweep budget, not the session deadline; ``<=0`` means the caller's clamp
    left no time and the sweep skips immediately. A ``None`` recorder records
    nothing, which is what a direct caller with no session bound wants.
    Returns a skip envelope when prerequisites are unmet.
    """
    _, benchmark_timeout_sec = resolve_benchmark_timeouts()
    session_deadline_sec, variant_expected_sec = session_grid_bounds(state)
    session_dir = Path(session_dir)
    # Whether the ladder was handed to the sweep or picked for the workload --
    # a distinction only this line can still see, since the two are the same
    # list one statement later.
    grid_source = GRID_REQUESTED if concs is not None else GRID_MODE_DEFAULT
    # ``None`` → default ladder; an explicit empty list short-circuits below.
    concs = list(concs) if concs is not None else default_concs_for_mode(getattr(state, "benchmark_mode", ""))
    isl = int(getattr(state, "isl", 0) or 0)
    osl = int(getattr(state, "osl", 0) or 0)
    baseline_tput = float(getattr(state, "baseline_tput", 0.0) or 0.0)

    optimized = _optimized_arm(state)

    if baseline_tput <= 0:
        return _declined(recorder, "no_baseline_tput")
    if isl <= 0 or osl <= 0:
        return _declined(recorder, "missing_workload_shape", isl=isl, osl=osl)
    if optimized is None:
        return _declined(recorder, "no_optimization_to_compare")
    if optimized.overlay:
        from hyperloom.inference_optimizer.runtime_environment import is_safe_path_entry
        from ..kernel.geak_config import geak_overlay_is_loadable

        if not is_safe_path_entry(optimized.overlay) or not geak_overlay_is_loadable(optimized.overlay):
            return _declined(recorder, "optimized_overlay_unavailable", final_overlay=optimized.overlay)
    if not concs:
        return _declined(recorder, "empty_conc_list")
    # A non-positive budget is "no time left", not "budget gate off": running the
    # ladder here would spend wall-clock the caller already accounted as gone.
    # No variant started, so this is a decline (see conc_sweep_declined_to_run)
    # and must not stamp ``budget_exhausted``.
    if total_budget_sec is not None and int(total_budget_sec) <= 0:
        return _declined(recorder, "no_time_budget_remaining", total_budget_sec=int(total_budget_sec))

    # Prefer the materialized baseline config; fall back to the shipped asset.
    base_yaml_raw = str(getattr(state, "baseline_config_path", "") or "").strip() or str(default_baseline_config())
    base_yaml_path = Path(base_yaml_raw)
    if not base_yaml_path.exists():
        return _declined(recorder, "baseline_config_missing", config_path=base_yaml_raw)

    task_id = f"conc_sweep_{utc_now_compact()}"
    workspace = runs_root(session_dir) / "conc_sweep" / task_id
    workspace.mkdir(parents=True, exist_ok=True)

    # Re-materialize (idempotent) in case we fell back to the shipped asset.
    resolved_model = resolve_session_model_path(
        state_model_path=str(getattr(state, "model_path", "") or ""),
        for_serving=True,
    )
    # Mirror the main flow (baseline/sweep/...): prefer $GPU_TYPE (cli.py canonicalizes mi325x/mi308x -> mi300x), fall
    # back to state.gpu_type, then canonicalize through _gpu_runner_type so the selected Magpie script is a shipped
    # runner (sglang_mi300x.sh), never the unshipped sglang_mi325x.sh.
    from hyperloom.inference_optimizer.gpu_types import _gpu_runner_type

    resolved_gpu = _gpu_runner_type(
        os.environ.get("GPU_TYPE", "").strip().lower() or str(getattr(state, "gpu_type", "") or "").strip().lower()
    )
    benchmark_script = state.accepted_baseline_script()
    try:
        base_yaml_path = prepare_grid_benchmark_config(
            base_yaml_path,
            workspace,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=benchmark_script,
            out_name="conc_sweep_base.with_envs.yaml",
            grading=getattr(state, "grading", None),
        )
    except FrameworkScriptMismatchError as exc:
        return _declined(
            recorder,
            "framework_script_mismatch",
            error_class="framework_script_mismatch",
            error=str(exc),
            workspace=str(workspace),
        )

    if recorder is not None:
        cb = state.current_best
        recorder.record_workload(
            session_id=getattr(state, "session_id", "") or session_dir.name,
            isl=isl,
            osl=osl,
            tp=getattr(state, "tp", 0),
            benchmark_mode=getattr(state, "benchmark_mode", ""),
        )
        recorder.record_anchor(
            SweepAnchor(
                baseline_tput=baseline_tput,
                anchor_tput=cb.get("tput"),
                tp=getattr(state, "tp", 0),
                variant_id=cb.get("variant_name"),
                action=cb.get("action"),
                extra_server_args=optimized.args,
                extra_envs=optimized.envs,
            )
        )
        recorder.record_environment(
            SweepEnvironment(
                sweep_task_id=task_id,
                workspace=workspace.as_posix(),
                model_path=resolved_model,
                gpu_type=resolved_gpu,
                base_config_path=base_yaml_path.as_posix(),
                report_json_path=(reports_dir(session_dir) / "conc_sweep_summary.json").as_posix(),
                report_csv_path=(reports_dir(session_dir) / "conc_sweep_raw.csv").as_posix(),
            )
        )

    started_at = time.time()
    deadline_stop = STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS]
    grid_deadline_sec = session_deadline_sec
    if total_budget_sec is not None:
        sweep_deadline_sec = time.monotonic() + total_budget_sec
        if session_deadline_sec is None or sweep_deadline_sec < session_deadline_sec:
            session_deadline_sec = sweep_deadline_sec
            deadline_stop = _SWEEP_BUDGET_STOP

    # Pre-compute report paths so incremental checkpoints carry them.
    rdir = reports_dir(session_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    json_path = rdir / "conc_sweep_summary.json"
    csv_path = rdir / "conc_sweep_raw.csv"

    if recorder is not None:
        recorder.record_budget(
            SweepBudget(
                declared_total_sec=total_budget_sec,
                granted_total_sec=total_budget_sec,
                rung_cost_sec=variant_expected_sec,
                raised=False,
                gate_active=total_budget_sec is not None,
                deadline=started_at + total_budget_sec if total_budget_sec is not None else None,
            )
        )

    run = _SweepRun(
        state=state,
        session_dir=session_dir,
        workspace=workspace,
        base_yaml_path=base_yaml_path,
        model_path=resolved_model,
        gpu_type=resolved_gpu,
        benchmark_script=benchmark_script,
        isl=isl,
        osl=osl,
        concs=concs,
        num_prompts_factor=DEFAULT_NUM_PROMPTS_FACTOR,
        optimized=optimized,
        baseline=_Arm(name="baseline"),
        benchmark_timeout_sec=benchmark_timeout_sec,
        session_deadline_sec=session_deadline_sec,
        variant_expected_sec=variant_expected_sec,
        deadline_stop=deadline_stop,
        grid_deadline_sec=grid_deadline_sec,
        started_at=started_at,
        total_budget_sec=total_budget_sec,
        json_path=json_path,
        csv_path=csv_path,
        recorder=recorder,
    )
    # Within a level the optimized arm runs first, then the baseline arm at the same CONC.
    arms_order = [run.optimized, run.baseline]
    log.info(
        "conc_sweep (interleaved): arms=%s per level, concs=%s isl=%d osl=%d total_budget=%s",
        ",".join(arm.name for arm in arms_order),
        run.concs_desc,
        isl,
        osl,
        f"{total_budget_sec}s" if total_budget_sec is not None else "unbounded",
    )
    if recorder is not None:
        recorder.record_plan(
            SweepPlan(
                concs_requested=concs,
                concs_ordered=run.concs_desc,
                grid_source=grid_source,
                num_prompts_factor=DEFAULT_NUM_PROMPTS_FACTOR,
                variant_timeout_sec=benchmark_timeout_sec,
                arms_order=[arm.name for arm in arms_order],
            )
        )
    await _sweep_interleaved(run, arms_order)

    payload = run.payload(in_progress=False)
    comparison, summary = payload["comparison"], payload["summary"]
    ceiling = _build_roofline_ceiling(
        state,
        _ArmCurves(baseline=payload["baseline"]["points"], optimized=payload["optimized"]["points"]),
        concs=concs,
        isl=isl,
        osl=osl,
    )
    if ceiling is not None:
        payload["roofline_ceiling"] = ceiling

    report_error = run.write(payload)

    if recorder is not None:
        if report_error is not None:
            recorder.record_fault(stage="report_write", exc=report_error)
        recorder.record_progress(comparison=comparison, summary=summary)
        recorder.finish(payload, stop_reason=getattr(state, "stop_reason", ""))

    log.info(
        "conc_sweep: done — status=%s successful_pairs=%d failed_pairs=%d best_speedup=%s%s",
        payload["status"],
        summary["successful_pairs"],
        summary["failed_pairs"],
        summary["best_speedup"],
        f" cancelled_by={run.cancel_reason}" if run.cancelled else "",
    )
    return payload


__all__ = [
    "AGENTX_DEFAULT_CONCS",
    "DEFAULT_CONCS",
    "DEFAULT_NUM_PROMPTS_FACTOR",
    "DEFAULT_TOTAL_BUDGET_SEC",
    "SCHEMA_VERSION",
    "_build_arm_grid",
    "_flush_conc_sweep_report",
    "_order_concs_desc",
    "conc_sweep_declined_to_run",
    "default_concs_for_mode",
    "run_conc_sweep",
]
