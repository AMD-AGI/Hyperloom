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
from dataclasses import dataclass, field, replace
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
    STAGE_BOOT,
    STAGE_BOOT_ATTEMPT,
    STAGE_BUDGET_SKIP,
    STAGE_REUSE,
    STAGE_SERVER_RESTART,
    STRATEGY_SERVER_RESTART,
    STRATEGY_SINGLE_SERVER,
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
    GridLaunch,
    GridRecipe,
    GridStop,
    GridVariant,
    VariantResult,
    run_grid,
)
from ..actions.executors._grid_runner import variant_conc
from ..actions.executors._subprocess_kill import resolve_benchmark_timeouts, session_deadline_to_remaining_sec
from ..actions.executors.benchmark_backend import resolve_backend
from ..actions.cancel_channel import current_cancel_scope
from ..actions.stop_attribution import (
    ORCHESTRATOR_CANCELLED_CLASS,
    SESSION_TIME_EXHAUSTED_CLASS,
    STOPPED_BY_THE_RUN,
    StoppedByTheRun,
)
from ..actions.executors._workload_envs import (
    BenchmarkTarget,
    FrameworkScriptMismatchError,
    LaunchLayer,
    MaterializeOptions,
    default_baseline_config,
    num_prompts_per_conc,
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


@dataclass(frozen=True)
class _WorkloadLadder:
    """A workload's default CONC ladder, and which way along it a rung gets cheaper.

    ``cost_grows_with_conc`` follows the workload's request window: a synthetic rung sends NUM_PROMPTS in proportion to
    its CONC, so a higher rung is more work; an agentic rung replays a fixed set of trajectories, so a higher rung runs
    the same work on more lanes and finishes sooner.
    """

    default_concs: list[int]
    cost_grows_with_conc: bool

    def order(self, concs: Sequence[int], anchor_conc: int) -> list[int]:
        """The deduplicated ladder as each arm runs it: the session's own CONC first, then the rest cheapest first.

        No server an arm boots reads CONC -- it only sets the client's concurrency -- so the order is the budget's to
        choose. The anchor pair is the one the session's result is graded on, and cheapest first fits the most pairs
        before a deadline ends the ladder.
        """
        rest = sorted(set(concs) - {anchor_conc}, reverse=not self.cost_grows_with_conc)
        return [anchor_conc, *rest] if anchor_conc in concs else rest


_SYNTHETIC_LADDER = _WorkloadLadder(DEFAULT_CONCS, cost_grows_with_conc=True)
_AGENTX_LADDER = _WorkloadLadder(AGENTX_DEFAULT_CONCS, cost_grows_with_conc=False)


def _workload_ladder(benchmark_mode: Any = "") -> _WorkloadLadder:
    return _AGENTX_LADDER if is_agentx_mode(benchmark_mode) else _SYNTHETIC_LADDER


def default_concs_for_mode(benchmark_mode: Any = "") -> list[int]:
    """The ladder a mode sweeps when the operator names none."""
    return list(_workload_ladder(benchmark_mode).default_concs)


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


# The stop for a first-arm rung its half of the budget reaps or never reaches, and for a second-arm rung whose partner
# never ran.
_PAIR_SHARE_STOP = StoppedByTheRun(
    error_class="budget_exhausted",
    interrupted="the first arm's half of the conc_sweep budget ran out while this round was running",
    never_started="the conc_sweep budget left no time to measure this concurrency on both arms",
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


def _raised_result(variant: GridVariant, stage: str, exc: Exception) -> VariantResult:
    """The failed result of a rung whose run raised ``exc`` at ``stage``."""
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status="failed",
        error=f"{stage}: {exc}",
        error_class=type(exc).__name__,
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


def _build_arm_grid(
    arm: _Arm,
    concs: list[int],
    *,
    isl: int,
    osl: int,
    num_prompts_factor: int,
) -> list[GridVariant]:
    """Build a single-arm grid in the order of ``concs``."""
    out: list[GridVariant] = []
    for conc in concs:
        num_prompts = int(conc) * int(num_prompts_factor)
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
_FIRST_ARM_SHARE = "first_arm_share_exhausted"


@dataclass(frozen=True)
class _Window:
    """The monotonic deadline a ladder runs under, the stop a round it reaps is recorded as, and the reason the
    report gives when it does."""

    deadline_sec: float | None
    stop: StoppedByTheRun
    exhausted_reason: str

    def remaining_sec(self) -> float | None:
        return session_deadline_to_remaining_sec(self.deadline_sec)

    def cannot_fit(self, least_sec: float | None) -> bool:
        """Whether a round that takes at least ``least_sec`` would outlast this window."""
        remaining_sec = self.remaining_sec()
        return least_sec is not None and remaining_sec is not None and remaining_sec < least_sec

    def whole(self) -> _ArmDeadlines:
        return _ArmDeadlines(self, self)

    def split_for_first_arm(self) -> _ArmDeadlines:
        """The deadlines of the arm that runs first: the whole window for its first rung, which has no pair yet to leave
        room for, and half of what is left now for each later one. With nothing left there is nothing to split."""
        remaining_sec = self.remaining_sec()
        if remaining_sec is None or remaining_sec <= 0:
            return self.whole()
        half = _Window(time.monotonic() + remaining_sec / 2, _PAIR_SHARE_STOP, _FIRST_ARM_SHARE)
        return _ArmDeadlines(self, half)


@dataclass(frozen=True)
class _ArmDeadlines:
    """The window an arm's first rung runs under, and the one each later rung does."""

    first_rung: _Window
    later_rungs: _Window

    def of_rung(self, idx: int) -> _Window:
        return self.first_rung if idx == 0 else self.later_rungs


@dataclass
class _Budget:
    """Whether a deadline has cut the sweep's ladder short, and why."""

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

    def run_out(self, window: _Window) -> None:
        self.exhaust(window.exhausted_reason, max(0.0, window.remaining_sec() or 0.0))


@dataclass(frozen=True)
class _RungNote:
    """How one rung came to run: ``stage`` is a ``STAGE_*`` value, and the seconds fields are wall clock."""

    stage: str
    start_time: str = ""
    wall_duration_sec: float | None = None
    budget_remaining_sec: float | None = None


@dataclass
class _SweepRun:
    """One conc sweep's fixed inputs, and the progress both of its arms add to.

    ``results`` and ``budget`` are shared across arms on purpose: every incremental flush reports the whole sweep so
    far, and the second arm runs the rungs the first one reached. ``window`` is the earlier of the sweep's and the
    session's monotonic deadlines, distinct from the fixed per-spawn ``benchmark_timeout_sec``.
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
    workload: _WorkloadLadder
    optimized: _Arm
    baseline: _Arm
    benchmark_timeout_sec: float
    window: _Window
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

    def stop_named_by(self, result: VariantResult, window: _Window) -> StoppedByTheRun | None:
        """The run-side stop a rung's result names, or ``None`` for a rung that ran or failed on its own."""
        if result.error_class == window.stop.error_class:
            return window.stop
        if result.error_class == ORCHESTRATOR_CANCELLED_CLASS:
            return _CANCEL_STOP
        return None

    @property
    def num_prompts_factor(self) -> int:
        """A synthetic rung sends the request window the session measured its anchor with, so the anchor rung costs and
        measures what the session's did. An agentic rung's window is its trajectory set; it never reads NUM_PROMPTS."""
        return num_prompts_per_conc(self.isl, self.osl)

    @property
    def ladder(self) -> list[int]:
        return self.workload.order(self.concs, int(self.state.conc or 0))

    def arm_grid(self, arm: _Arm) -> list[GridVariant]:
        return _build_arm_grid(arm, self.ladder, isl=self.isl, osl=self.osl, num_prompts_factor=self.num_prompts_factor)

    def arm_results(self, arm_name: str) -> list[VariantResult]:
        prefix = f"{arm_name}_"
        return [result for result in self.results if result.name.startswith(prefix)]

    def reached(self, arm: _Arm) -> list[int]:
        """The concurrencies ``arm`` measured or failed at: the rungs the other arm has to pair."""
        return [int(result.extra_envs["CONC"]) for result in self.arm_results(arm.name) if result.status != "skipped"]

    def record_never_run(self, arm_name: str, rungs: list[GridVariant], stop: StoppedByTheRun) -> None:
        """Record ``rungs`` as never run, under the deadline ``stop`` that kept them from running."""
        note = _RungNote(stage=STAGE_BUDGET_SKIP, budget_remaining_sec=self.budget.remaining_sec)
        for variant in rungs:
            skipped = _deadline_skip_result(variant, stop)
            self.results.append(skipped)
            self.record_rung(arm_name, skipped, note)

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

    def record_ran(self, arm_name: str, result: VariantResult, note: _RungNote, window: _Window) -> None:
        """Publish a rung's result; one a run-side stop names is recorded as a budget skip."""
        self.results.append(result)
        if self.stop_named_by(result, window) is not None:
            note = replace(note, stage=STAGE_BUDGET_SKIP, budget_remaining_sec=self.budget.remaining_sec)
        self.record_rung(arm_name, result, note)

    def record_boot(
        self,
        arm_name: str,
        attempted: list[GridVariant],
        failed: list[VariantResult],
        booted: GridVariant | None = None,
    ) -> None:
        """Record how the arm's boot attempts resolved: the rung that booted, or ``None`` when none came up."""
        if self.recorder is None:
            return
        self.recorder.record_arm_boot(
            arm_name,
            succeeded=booted is not None,
            booted_conc=variant_conc(booted) if booted is not None else None,
            attempted_concs=[variant_conc(variant) for variant in attempted],
            failed_concs=[variant_conc(fb) for fb in failed],
        )

    def refuse_arm(self, arm: _Arm, grid: list[GridVariant]) -> bool:
        """Skip every rung of ``arm`` when the sweep is cancelled or the session is closing; return whether it did."""
        if self.observe_cancel():
            skipped = [_deadline_skip_result(v, _CANCEL_STOP) for v in grid]
            reason, remaining_sec = ORCHESTRATOR_CANCELLED_CLASS, 0.0
        elif self.session_closing():
            self.budget.spend_on_session_reserve()
            skipped = [_budget_skip_result(v) for v in grid]
            reason, remaining_sec = _SESSION_RESERVE, 0.0
        else:
            return False
        self.results.extend(skipped)
        if self.recorder is not None:
            self.recorder.record_arm_refused(arm.name, reason=reason, remaining_sec=remaining_sec)
        return True

    def skip_rest(self, arm_name: str, rest: list[GridVariant]) -> bool:
        """Record ``rest`` as never run when the sweep is cancelled or the session is closing; return whether it did."""
        if self.observe_cancel():
            skipped = [_deadline_skip_result(v, _CANCEL_STOP) for v in rest]
            note = _RungNote(stage=STAGE_BUDGET_SKIP)
        elif self.session_closing():
            self.budget.spend_on_session_reserve()
            skipped = [_budget_skip_result(v) for v in rest]
            note = _RungNote(stage=STAGE_BUDGET_SKIP, budget_remaining_sec=0.0)
        else:
            return False
        for skip_r in skipped:
            self.results.append(skip_r)
            self.record_rung(arm_name, skip_r, note)
        return True

    def close_arm(self, arm_name: str, exc: BaseException | None) -> None:
        """Close one arm's row however its ladder ended, so an arm that raised mid-ladder says so once."""
        if self.recorder is None:
            return
        if exc is None:
            self.recorder.finish_arm(arm_name, status=_arm_status(self.arm_results(arm_name)))
        else:
            self.recorder.fail_arm(arm_name, exc)

    async def run_rung(
        self, variant: GridVariant, launch: GridLaunch, window: _Window, base: LaunchLayer
    ) -> VariantResult:
        """Run one rung from the sweep's base config under ``window``; a deadline stop inside it exhausts the budget."""
        if self.observe_cancel():
            return _deadline_skip_result(variant, _CANCEL_STOP)
        (result,) = await run_grid(
            grid=[variant],
            output_root=self.workspace,
            recipe=GridRecipe(
                base_yaml_path=self.base_yaml_path,
                base=base,
                target=BenchmarkTarget(
                    model_path=self.model_path,
                    gpu_type=self.gpu_type,
                    benchmark_script=self.benchmark_script,
                ),
            ),
            launch=launch,
            stop=GridStop(session_deadline_sec=window.deadline_sec, deadline_stop=window.stop),
        )
        self.observe_cancel([result])
        if result.error_class == window.stop.error_class:
            self.budget.run_out(window)
        return result

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


@dataclass(frozen=True)
class _ArmRound:
    """How one round reaches the arm's server: its launch, the layer it adds over the base config, and the label a
    round that raised is recorded under."""

    launch: GridLaunch
    label: str
    base: LaunchLayer = field(default_factory=LaunchLayer)


@dataclass
class _ArmServer:
    """The server one arm's ladder runs against, and how bringing it up went.

    ``lifecycle`` is whether the framework keeps one server across rungs; without it every rung starts its own. Until
    a round brings the persistent server up, each rung's round boots it.
    """

    arm_name: str
    pid_dir: Path
    port: int
    framework: str
    lifecycle: bool
    lease: Any = None
    attempted: list[GridVariant] = field(default_factory=list)
    failed: list[VariantResult] = field(default_factory=list)
    booted: GridVariant | None = None

    def _server_lifecycle(self, *, cleanup: bool) -> dict[str, Any]:
        return {"cleanup": cleanup, "pid_dir": str(self.pid_dir), "port": self.port}

    def boot_round(self) -> _ArmRound:
        return _ArmRound(
            GridLaunch(
                serving_lease=self.lease,
                server_lifecycle=self._server_lifecycle(cleanup=False),
                server_already_ready=False,
                warmup_before_measure=False,
                lifecycle_boot_only=True,
            ),
            label="single_server_boot",
            base=LaunchLayer(extra_envs=dict(resolve_backend().boot_phase_envs)),
        )

    def reuse_round(self, *, cleanup: bool) -> _ArmRound:
        return _ArmRound(
            GridLaunch(
                serving_lease=self.lease,
                server_lifecycle=self._server_lifecycle(cleanup=cleanup),
                server_already_ready=True,
                warmup_before_measure=False,
            ),
            label="single_server_reuse",
        )

    def restart_round(self) -> _ArmRound:
        return _ArmRound(GridLaunch(serving_lease=self.lease), label="server_restart")

    def teardown(self) -> None:
        from ..actions.executors._server_lifecycle import teardown_lifecycle_server

        teardown_lifecycle_server(pid_dir=self.pid_dir, framework=self.framework, port=self.port)

    def close(self, run: _SweepRun, exc: BaseException | None) -> None:
        """Take the arm's server down, release its lease and close its row, however its ladder ended."""
        if self.lifecycle:
            try:
                self.teardown()
            except Exception as teardown_exc:  # isolation boundary: one arm's server teardown, recorded as a fault
                # The last word on this arm's server: a teardown that failed here
                # leaves it alive past the arm that owned it.
                log.warning("conc_sweep single-server: arm=%s teardown failed", self.arm_name, exc_info=True)
                if run.recorder is not None:
                    run.recorder.record_fault(stage=f"teardown_lifecycle_server:{self.arm_name}", exc=teardown_exc)
            if self.attempted:
                run.record_boot(self.arm_name, self.attempted, self.failed, self.booted)
        if self.lease is not None:
            self.lease.close()
        run.close_arm(self.arm_name, exc)


@dataclass(frozen=True)
class _RungRun:
    """What one round of a rung returned, and the clock it ran on."""

    result: VariantResult
    start_time: str
    wall_duration_sec: float
    budget_remaining_sec: float | None

    def note(self, stage: str) -> _RungNote:
        return _RungNote(
            stage=stage,
            start_time=self.start_time,
            wall_duration_sec=self.wall_duration_sec,
            budget_remaining_sec=self.budget_remaining_sec,
        )


def _resolve_arm_strategy(run: _SweepRun, arm_name: str, *, serving_lease_held: bool) -> ArmStrategy:
    """How one arm runs its ladder: on one persistent server when its framework is lifecycle-eligible."""
    from ..actions.executors._server_lifecycle import resolve_lifecycle_params

    recorder = run.recorder
    # Resolve lifecycle params (port, framework) from the materialized config.
    lc_reason = "resolve_failed"
    try:
        lc_params = resolve_lifecycle_params(run.base_yaml_path)
        port = int(lc_params.get("port") or 8888)
        framework = str(lc_params.get("framework") or "")
        lc_eligible = bool(lc_params.get("eligible"))
        lc_reason = str(lc_params.get("reason") or "")
    except Exception as exc:  # isolation boundary: one arm's lifecycle resolution, recorded as a fault
        log.warning("conc_sweep single-server: resolve_lifecycle_params failed", exc_info=True)
        if recorder is not None:
            # Named per arm: both arms resolve, and two faults reading alike
            # would collapse into one row.
            recorder.record_fault(stage=f"resolve_lifecycle_params:{arm_name}", exc=exc)
        lc_eligible = False
        port = 8888
        framework = ""

    arm_strategy = ArmStrategy(
        strategy=STRATEGY_SINGLE_SERVER if lc_eligible else STRATEGY_SERVER_RESTART,
        reason=None if lc_eligible else "framework_not_lifecycle_eligible",
        lifecycle_eligible=lc_eligible,
        lifecycle_reason=lc_reason,
        port=port,
        framework=framework,
        serving_lease_held=serving_lease_held,
    )
    if recorder is not None:
        recorder.record_arm_strategy(arm_name, arm_strategy)
    return arm_strategy


async def _sweep_arm(
    run: _SweepRun, arm: _Arm, concs: Sequence[int], deadlines: _ArmDeadlines, unpaired: StoppedByTheRun
) -> None:
    """Run ``arm`` over the ladder's rungs at ``concs`` under ``deadlines``.

    The ladder's other rungs are recorded as never run under ``unpaired``, the stop that kept the first arm from them.
    """
    from ..actions.executors._grid_runner import _num_gpus_for_config
    from ..actions.executors._ray_serving import maybe_serving_lease

    grid = run.arm_grid(arm)
    if run.refuse_arm(arm, grid):
        return
    ladder = [variant for variant in grid if variant_conc(variant) in concs]
    recorder = run.recorder
    if recorder is not None:
        recorder.open_arm(arm.name, extra_server_args=arm.args, extra_envs=arm.envs)
        recorder.record_arm_grid(
            arm.name,
            rungs=[
                {
                    "name": variant.name,
                    "conc": variant.extra_envs.get("CONC"),
                    "num_prompts": variant.extra_envs.get("NUM_PROMPTS"),
                }
                for variant in ladder
            ],
        )
    run.record_never_run(arm.name, [variant for variant in grid if variant_conc(variant) not in concs], unpaired)
    if not ladder:
        run.close_arm(arm.name, None)
        return

    # Ray-managed GPU execution: one held Ray lease (``num_gpus=TP``) spans the arm's whole ladder -- its persistent
    # server's boot and every reuse round, or each per-rung restart -- so no GPU process outlives it.
    lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(run.base_yaml_path))
    strategy = _resolve_arm_strategy(run, arm.name, serving_lease_held=lease is not None)
    pid_dir = run.workspace / f"server_{arm.name}"
    pid_dir.mkdir(parents=True, exist_ok=True)
    server = _ArmServer(
        arm_name=arm.name,
        pid_dir=pid_dir,
        port=strategy.port,
        framework=strategy.framework,
        lifecycle=strategy.lifecycle_eligible,
        lease=lease,
    )
    failure: BaseException | None = None
    try:
        await _run_ladder(run, server, ladder, deadlines)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        server.close(run, failure)


async def _run_ladder(run: _SweepRun, server: _ArmServer, ladder: list[GridVariant], deadlines: _ArmDeadlines) -> None:
    """Run the rungs in order until a stop ends the ladder; the rungs it never starts are recorded as never run."""
    least_sec: float | None = None
    idx = 0
    while idx < len(ladder):
        if run.skip_rest(server.arm_name, ladder[idx:]):
            return
        window = deadlines.of_rung(idx)
        if window.cannot_fit(least_sec):
            run.budget.run_out(window)
            run.record_never_run(server.arm_name, ladder[idx:], window.stop)
            return
        result = await _run_rung_on(run, server, ladder[idx], window, is_last=idx == len(ladder) - 1)
        if result is None:
            # The round only brought the server up; the same rung's measuring round runs next.
            continue
        run.flush_progress()
        stop = run.stop_named_by(result, window)
        if stop is not None:
            run.record_never_run(server.arm_name, ladder[idx + 1 :], stop)
            return
        if idx > 0 and result.status == "succeeded":
            # Past the anchor the rungs run cheapest first, so none benchmarks faster than the one before it.
            least_sec = result.duration_seconds or least_sec
        idx += 1


async def _run_rung_on(
    run: _SweepRun, server: _ArmServer, variant: GridVariant, window: _Window, *, is_last: bool
) -> VariantResult | None:
    """Run one round of a rung on the arm's server and record it; ``None`` when the round only booted the server."""
    if not server.lifecycle:
        ran = await _run_round(run, variant, server.restart_round(), window)
        run.record_ran(server.arm_name, ran.result, ran.note(STAGE_SERVER_RESTART), window)
        return ran.result
    if server.booted is None:
        ran = await _run_round(run, variant, server.boot_round(), window)
        return None if _settle_boot(run, server, variant, ran, window) else ran.result
    ran = await _run_round(run, variant, server.reuse_round(cleanup=is_last), window)
    run.record_ran(server.arm_name, ran.result, ran.note(STAGE_REUSE), window)
    return ran.result


def _settle_boot(run: _SweepRun, server: _ArmServer, variant: GridVariant, ran: _RungRun, window: _Window) -> bool:
    """Record a boot round; return whether its rung still needs a measuring round on the server it brought up.

    Only a backend that boots without measuring leaves its rung to a round on the server. A boot that failed or was
    stopped takes its server down; one stopped by a deadline or a cancel is not a boot the server failed.
    """
    result = ran.result
    stopped = run.cancelled or run.stop_named_by(result, window) is not None
    if stopped or result.status in {"failed", "skipped"}:
        server.teardown()
        if not stopped:
            server.attempted.append(variant)
            server.failed.append(result)
        run.record_ran(server.arm_name, result, ran.note(STAGE_BOOT_ATTEMPT), window)
        return False
    server.attempted.append(variant)
    server.booted = variant
    if result.note == "server_lifecycle_boot_only":
        run.record_rung(server.arm_name, result, ran.note(STAGE_BOOT))
        return True
    run.record_ran(server.arm_name, result, ran.note(STAGE_BOOT), window)
    return False


async def _run_round(run: _SweepRun, variant: GridVariant, arm_round: _ArmRound, window: _Window) -> _RungRun:
    """Run one round of ``variant``; a round that raised becomes the rung's failed result, named by its label."""
    remaining_sec = window.remaining_sec()
    start_time = now_iso("seconds")
    started_at = time.time()
    try:
        result = await run.run_rung(variant, arm_round.launch, window, arm_round.base)
    except Exception as exc:  # isolation boundary: one round of an arm's ladder, recorded as its rung's failure
        log.warning("conc_sweep: %s round of %s raised %r", arm_round.label, variant.name, exc, exc_info=True)
        result = _raised_result(variant, arm_round.label, exc)
    return _RungRun(result, start_time, round(time.time() - started_at, 3), remaining_sec)


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
    except (OSError, ValueError, csv.Error) as exc:
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


def _decline_before_start(
    recorder: Any, state: SharedState, optimized: _Arm | None, concs: list[int], total_budget_sec: int | None
) -> dict[str, Any] | None:
    """The skip envelope for a sweep whose prerequisites are unmet, or ``None`` when it can start."""
    isl = int(getattr(state, "isl", 0) or 0)
    osl = int(getattr(state, "osl", 0) or 0)
    baseline_tput = float(getattr(state, "baseline_tput", 0.0) or 0.0)
    if baseline_tput <= 0:
        return _declined(recorder, "no_baseline_tput")
    if isl <= 0 or osl <= 0:
        return _declined(recorder, "missing_workload_shape", isl=isl, osl=osl)
    if optimized is None:
        return _declined(recorder, "no_optimization_to_compare")
    if optimized.overlay:
        from hyperloom.common.overlay import is_safe_path_entry
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
    return None


async def _sweep_arms(run: _SweepRun, grid_source: str) -> None:
    """Record the budget and plan, then run the baseline arm and the optimized arm on the rungs it reached.

    The first arm spends for both arms of each pair, so past its first rung it gets half of what is left. The baseline
    goes first because the optimized stack, faster at the anchor, replays its rungs in the half it left.
    """
    recorder = run.recorder
    total_budget_sec = run.total_budget_sec
    if recorder is not None:
        recorder.record_budget(
            SweepBudget(
                declared_total_sec=total_budget_sec,
                deadline=run.started_at + total_budget_sec if total_budget_sec is not None else None,
            )
        )
    log.info(
        "conc_sweep (single-server): arms=baseline,optimized concs=%s isl=%d osl=%d total_budget=%s",
        run.ladder,
        run.isl,
        run.osl,
        f"{total_budget_sec}s" if total_budget_sec is not None else "unbounded",
    )
    if recorder is not None:
        recorder.record_plan(
            SweepPlan(
                concs_requested=run.concs,
                concs_ordered=run.ladder,
                grid_source=grid_source,
                num_prompts_factor=run.num_prompts_factor,
                variant_timeout_sec=run.benchmark_timeout_sec,
                arms_order=[run.baseline.name, run.optimized.name],
            )
        )
    first_arm = run.window.split_for_first_arm()
    unpaired = first_arm.later_rungs.stop
    await _sweep_arm(run, run.baseline, run.ladder, first_arm, unpaired)
    await _sweep_arm(run, run.optimized, run.reached(run.baseline), run.window.whole(), unpaired)


def _finish_sweep(run: _SweepRun) -> dict[str, Any]:
    """Settle the final report from every recorded rung, write it and close the sweep's event on it."""
    recorder = run.recorder
    payload = run.payload(in_progress=False)
    comparison, summary = payload["comparison"], payload["summary"]
    ceiling = _build_roofline_ceiling(
        run.state,
        _ArmCurves(baseline=payload["baseline"]["points"], optimized=payload["optimized"]["points"]),
        concs=run.concs,
        isl=run.isl,
        osl=run.osl,
    )
    if ceiling is not None:
        payload["roofline_ceiling"] = ceiling

    report_error = run.write(payload)

    if recorder is not None:
        if report_error is not None:
            recorder.record_fault(stage="report_write", exc=report_error)
        recorder.record_progress(comparison=comparison, summary=summary)
        recorder.finish(payload, stop_reason=getattr(run.state, "stop_reason", ""))

    log.info(
        "conc_sweep: done — status=%s successful_pairs=%d failed_pairs=%d best_speedup=%s%s",
        payload["status"],
        summary["successful_pairs"],
        summary["failed_pairs"],
        summary["best_speedup"],
        f" cancelled_by={run.cancel_reason}" if run.cancelled else "",
    )
    return payload


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
    session_deadline_sec = state.grid_session_deadline_sec()
    session_dir = Path(session_dir)
    # Whether the ladder was handed to the sweep or picked for the workload --
    # a distinction only this line can still see, since the two are the same
    # list one statement later.
    grid_source = GRID_REQUESTED if concs is not None else GRID_MODE_DEFAULT
    # ``None`` → default ladder; an explicit empty list short-circuits below.
    workload = _workload_ladder(getattr(state, "benchmark_mode", ""))
    concs = list(concs) if concs is not None else list(workload.default_concs)
    optimized = _optimized_arm(state)
    declined = _decline_before_start(recorder, state, optimized, concs, total_budget_sec)
    if declined is not None:
        return declined
    # Positive past the decline above.
    isl = int(state.isl)
    osl = int(state.osl)
    baseline_tput = float(state.baseline_tput)

    # Prefer the materialized baseline config; fall back to the shipped asset.
    base_yaml_raw = str(getattr(state, "baseline_config_path", "") or "").strip() or str(default_baseline_config())
    base_yaml_path = Path(base_yaml_raw)
    if not base_yaml_path.exists():
        return _declined(recorder, "baseline_config_missing", config_path=base_yaml_raw)

    task_id = f"conc_sweep_{utc_now_compact()}"
    workspace = runs_root(session_dir) / "conc_sweep" / task_id
    workspace.mkdir(parents=True, exist_ok=True)

    # Re-materialize (idempotent) in case we fell back to the shipped asset.
    resolved_model = resolve_session_model_path(state_model_path=str(getattr(state, "model_path", "") or ""))
    # Mirror the main flow (baseline/sweep/...): prefer $GPU_TYPE (cli.py canonicalizes mi325x/mi308x -> mi300x), fall
    # back to state.gpu_type, then canonicalize through gpu_runner_type so the selected Magpie script is a shipped
    # runner (sglang_mi300x.sh), never the unshipped sglang_mi325x.sh.
    from hyperloom.inference_optimizer.gpu_types import gpu_runner_type

    resolved_gpu = gpu_runner_type(
        os.environ.get("GPU_TYPE", "").strip().lower() or str(getattr(state, "gpu_type", "") or "").strip().lower()
    )
    benchmark_script = state.accepted_baseline_script()
    try:
        base_yaml_path = prepare_grid_benchmark_config(
            base_yaml_path,
            workspace,
            target=BenchmarkTarget(
                model_path=resolved_model or None,
                gpu_type=resolved_gpu or None,
                benchmark_script=benchmark_script,
            ),
            options=MaterializeOptions(
                out_name="conc_sweep_base.with_envs.yaml",
                grading=getattr(state, "grading", None),
            ),
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
    window = _Window(
        session_deadline_sec, STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS], SESSION_TIME_EXHAUSTED_CLASS
    )
    if total_budget_sec is not None:
        sweep_deadline_sec = time.monotonic() + total_budget_sec
        if session_deadline_sec is None or sweep_deadline_sec < session_deadline_sec:
            window = _Window(sweep_deadline_sec, _SWEEP_BUDGET_STOP, "total_budget_exhausted")

    # Pre-compute report paths so incremental checkpoints carry them.
    rdir = reports_dir(session_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    json_path = rdir / "conc_sweep_summary.json"
    csv_path = rdir / "conc_sweep_raw.csv"

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
        workload=workload,
        optimized=optimized,
        baseline=_Arm(name="baseline"),
        benchmark_timeout_sec=benchmark_timeout_sec,
        window=window,
        started_at=started_at,
        total_budget_sec=total_budget_sec,
        json_path=json_path,
        csv_path=csv_path,
        recorder=recorder,
    )
    await _sweep_arms(run, grid_source)
    return _finish_sweep(run)


__all__ = [
    "AGENTX_DEFAULT_CONCS",
    "DEFAULT_CONCS",
    "DEFAULT_TOTAL_BUDGET_SEC",
    "SCHEMA_VERSION",
    "_build_arm_grid",
    "_flush_conc_sweep_report",
    "_workload_ladder",
    "conc_sweep_declined_to_run",
    "default_concs_for_mode",
    "run_conc_sweep",
]
