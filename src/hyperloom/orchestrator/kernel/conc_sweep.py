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
    session_grid_bounds,
)
from ..actions.executors._grid_runner import variant_benchmark_timeout_sec, variant_conc
from ..actions.executors._subprocess_kill import (
    resolve_benchmark_timeouts,
    session_deadline_to_remaining_sec,
)
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
        granted_cap_sec=variant_benchmark_timeout_sec(variant),
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
        granted_cap_sec=variant_benchmark_timeout_sec(variant),
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
        granted_cap_sec=variant_benchmark_timeout_sec(variant),
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

    def refuses_next_arm(self) -> bool:
        """A deadline stop refuses the next arm; the session reserve leaves each arm to skip its own rungs."""
        return self.exhausted and self.skip_reason != _SESSION_RESERVE


@dataclass(frozen=True)
class _RungNote:
    """How one rung came to run and whether it is part of the published curve.

    ``stage`` is a ``STAGE_*`` value, ``committed`` says whether the rung is part
    of the published curve, and the seconds fields are wall clock.
    """

    stage: str
    committed: bool = True
    start_time: str = ""
    wall_duration_sec: float | None = None
    budget_remaining_sec: float | None = None


@dataclass
class _SweepRun:
    """One conc sweep's fixed inputs, and the progress both of its arms add to.

    ``results`` and ``budget`` are shared across arms on purpose: every incremental flush reports the whole sweep so
    far, and a budget stop inside one arm has to refuse the next. ``session_deadline_sec`` is the earlier monotonic
    sweep/session boundary, named by ``deadline_stop`` and distinct from ``benchmark_timeout_sec``, the cap a rung
    that pins no workload of its own gets.
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
                committed=note.committed,
                num_prompts=(result.extra_envs or {}).get("NUM_PROMPTS"),
                start_time=note.start_time,
                wall_duration_sec=note.wall_duration_sec,
                # The cap the rung was actually held to, as its runner published it: each rung is granted
                # time sized to its own workload, and a cap derived again here from what the result happens
                # to carry would describe a run nobody made.
                granted_cap_sec=result.granted_cap_sec,
                budget_remaining_sec=note.budget_remaining_sec,
            ),
        )

    def record_ran(self, arm_name: str, results: list[VariantResult], note: _RungNote) -> None:
        """Publish the results a rung returned; one a run-side stop names is recorded as a budget skip."""
        for result in results:
            self.results.append(result)
            if self.stop_named_by(result) is not None:
                stopped = replace(note, stage=STAGE_BUDGET_SKIP, budget_remaining_sec=self.budget.remaining_sec)
                self.record_rung(arm_name, result, stopped)
            else:
                self.record_rung(arm_name, result, note)

    def record_boot(
        self,
        arm_name: str,
        attempted: list[GridVariant],
        failed: list[VariantResult],
        booted: GridVariant | None = None,
    ) -> None:
        """Record how the boot-retry-descend loop resolved: the rung it booted at, or ``None`` when none came up."""
        if self.recorder is None:
            return
        self.recorder.record_arm_boot(
            arm_name,
            succeeded=booted is not None,
            booted_conc=variant_conc(booted) if booted is not None else None,
            attempted_concs=[variant_conc(variant) for variant in attempted],
            failed_concs=[variant_conc(fb) for fb in failed],
        )

    def refuse_arm(self, arm: _Arm) -> bool:
        """Skip every rung of ``arm`` when the sweep is cancelled, out of budget or closing; return whether it did."""
        if self.observe_cancel():
            skipped = [_deadline_skip_result(v, _CANCEL_STOP) for v in self.arm_grid(arm)]
            reason, remaining_sec = ORCHESTRATOR_CANCELLED_CLASS, 0.0
        elif self.budget.refuses_next_arm():
            skipped = [_deadline_skip_result(v, self.deadline_stop) for v in self.arm_grid(arm)]
            reason, remaining_sec = self.budget.skip_reason, self.budget.remaining_sec
        elif self.session_closing():
            self.budget.spend_on_session_reserve()
            skipped = [_budget_skip_result(v) for v in self.arm_grid(arm)]
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

    def commit_failed_boots(self, arm_name: str, boots: list[VariantResult]) -> None:
        """Publish the boot attempts a lower rung's outcome has settled as part of the curve."""
        for failed in boots:
            self.results.append(failed)
            if self.recorder is not None:
                self.recorder.commit_variant(
                    arm_name,
                    stage=STAGE_BOOT_ATTEMPT,
                    conc=variant_conc(failed),
                    point=_point_from_variant(failed, arm=arm_name),
                )

    def close_arm(self, arm_name: str, exc: BaseException | None) -> None:
        """Close one arm's row, whichever of its ladder's paths it leaves by.

        An arm leaves by three: the restart ladder it was delegated to, the restart
        retry after every boot failed, and the reuse ladder. Closing here means an
        arm that raised mid-ladder says so once, rather than each path reporting a
        ladder it never finished as the status its own results happen to add up to.
        """
        if self.recorder is None:
            return
        if exc is None:
            self.recorder.finish_arm(arm_name, status=_arm_status(self.arm_results(arm_name)))
        else:
            self.recorder.fail_arm(arm_name, exc)

    async def run_rung(
        self, variant: GridVariant, launch: GridLaunch, base: LaunchLayer = LaunchLayer()
    ) -> list[VariantResult]:
        """Run one rung from the sweep's base config; a deadline stop inside it exhausts the sweep's budget."""
        if self.observe_cancel():
            return [_deadline_skip_result(variant, _CANCEL_STOP)]
        if self.budget.exhausted:
            return [_deadline_skip_result(variant, self.deadline_stop)]
        results = await run_grid(
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
            stop=GridStop(
                session_deadline_sec=self.session_deadline_sec,
                variant_expected_sec=self.variant_expected_sec,
                deadline_stop=self.deadline_stop,
            ),
        )
        self.observe_cancel(results)
        if any(result.error_class == self.deadline_stop.error_class for result in results):
            remaining = session_deadline_to_remaining_sec(self.session_deadline_sec)
            reason = self.deadline_stop.error_class
            if self.deadline_stop == _SWEEP_BUDGET_STOP:
                reason = (
                    "total_budget_exhausted"
                    if remaining is not None and remaining <= 0
                    else "insufficient_remaining_for_variant"
                )
            self.budget.exhaust(reason, max(0.0, remaining) if remaining is not None else 0.0)
        return results

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
class _ArmServer:
    """The server one arm's ladder runs against: where its pids live, how to reach it, and the Ray lease it holds."""

    arm_name: str
    pid_dir: Path
    port: int
    framework: str
    lease: Any = None

    def close(self, run: _SweepRun, exc: BaseException | None) -> None:
        """Release the arm's lease and close its row, whichever of its ladder's paths the arm leaves by."""
        if self.lease is not None:
            self.lease.close()
        run.close_arm(self.arm_name, exc)


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


async def _sweep_one_arm_single_server(run: _SweepRun, arm: _Arm) -> None:
    """Sweep one arm across all CONC values reusing a single persistent server.

    Boots the server on the highest CONC (Option A), then reuses it for all
    lower CONCs.  If boot fails, retries with the next lower CONC
    (boot-retry-descend).  Falls back to the legacy per-variant server-restart
    path (Option B) when all boot retries are exhausted.
    """
    from ..actions.executors._grid_runner import _num_gpus_for_config
    from ..actions.executors._ray_serving import maybe_serving_lease
    from ..actions.executors._server_lifecycle import teardown_lifecycle_server

    recorder = run.recorder
    arm_name = arm.name
    grid = run.arm_grid(arm)
    if recorder is not None:
        recorder.record_arm_grid(
            arm_name,
            rungs=[
                {
                    "name": variant.name,
                    "conc": variant.extra_envs.get("CONC"),
                    "num_prompts": variant.extra_envs.get("NUM_PROMPTS"),
                }
                for variant in grid
            ],
        )

    # Ray-managed GPU execution: one held Ray lease (``num_gpus=TP``) spans this arm's persistent server — boot +
    # every CONC reuse round, or the Option B per-variant restarts — so the shared server's whole lifetime is covered
    # by a single lease and no GPU process outlives it.
    arm_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(run.base_yaml_path))

    # Shared pid_dir for server reuse across all CONC variants in this arm.
    pid_dir = run.workspace / f"server_{arm_name}"
    pid_dir.mkdir(parents=True, exist_ok=True)

    arm_strategy = _resolve_arm_strategy(run, arm_name, serving_lease_held=arm_lease is not None)
    server = _ArmServer(
        arm_name=arm_name, pid_dir=pid_dir, port=arm_strategy.port, framework=arm_strategy.framework, lease=arm_lease
    )

    if not arm_strategy.lifecycle_eligible:
        # Framework does not support server_lifecycle — fall through to Option B (per-variant server restart via
        # normal run_grid).
        log.info(
            "conc_sweep single-server: arm=%s not lifecycle-eligible (%s); using per-variant server restart (Option B)",
            arm_name,
            arm_strategy.lifecycle_reason,
        )
        await _sweep_arm_option_b(run, server, grid)
        return

    # Boot-retry-descend: try each CONC from highest to lowest until boot succeeds.
    failed_boots: list[VariantResult] = []
    for boot_idx, boot_variant in enumerate(grid):
        log.info(
            "conc_sweep single-server: arm=%s boot attempt %d/%d (conc=%s)",
            arm_name,
            boot_idx + 1,
            len(grid),
            boot_variant.extra_envs.get("CONC", "?"),
        )
        server_lifecycle_boot = {
            "cleanup": False,
            "pid_dir": str(pid_dir),
            "port": server.port,
        }
        boot_started_iso = now_iso("seconds")
        boot_started_at = time.time()
        try:
            boot_results = await run.run_rung(
                boot_variant,
                GridLaunch(
                    serving_lease=arm_lease,
                    server_lifecycle=server_lifecycle_boot,
                    server_already_ready=False,
                    warmup_before_measure=False,
                    lifecycle_boot_only=True,
                ),
                base=LaunchLayer(extra_envs=dict(resolve_backend().boot_phase_envs)),
            )
        except Exception as exc:  # isolation boundary: one single-server boot rung
            log.warning(
                "conc_sweep single-server: arm=%s boot conc=%s raised %r; trying next lower conc",
                arm_name,
                boot_variant.extra_envs.get("CONC", "?"),
                exc,
                exc_info=True,
            )
            boot_results = [_raised_result(boot_variant, "single_server_boot", exc)]

        # run_rung returns one result for its one-variant grid.
        br = boot_results[0]
        boot_failed = br.status in {"failed", "skipped"}
        boot_elapsed = round(time.time() - boot_started_at, 3)

        boot_stop = run.stop_named_by(br)
        if boot_stop is None and boot_failed and run.cancelled:
            # A boot that failed under a cancel still ends the ladder: every lower rung would be reaped on start.
            boot_stop = _CANCEL_STOP
        if boot_stop is not None:
            if boot_stop is _CANCEL_STOP:
                log.info(
                    "conc_sweep single-server: arm=%s stopped at boot conc=%s (%s); not trying lower concs",
                    arm_name,
                    boot_variant.extra_envs.get("CONC", "?"),
                    run.cancel_reason,
                )
            run.commit_failed_boots(arm_name, failed_boots)
            run.record_ran(
                arm_name,
                [br, *[_deadline_skip_result(v, boot_stop) for v in grid[boot_idx + 1 :]]],
                _RungNote(stage=STAGE_BUDGET_SKIP, budget_remaining_sec=run.budget.remaining_sec),
            )
            try:
                teardown_lifecycle_server(pid_dir=pid_dir, framework=server.framework, port=server.port)
            finally:
                server.close(run, None)
            return

        if boot_failed:
            # Ensure server is torn down before retrying at a lower CONC.
            teardown_lifecycle_server(pid_dir=pid_dir, framework=server.framework, port=server.port)
            failed_boots.append(br)
            # Recorded now and uncommitted: a concurrency the server would not
            # come up at is the finding, and whether it counts toward the curve
            # is not known until a lower rung either boots or does not.
            run.record_rung(
                arm_name,
                br,
                _RungNote(
                    stage=STAGE_BOOT_ATTEMPT,
                    committed=False,
                    start_time=boot_started_iso,
                    wall_duration_sec=boot_elapsed,
                ),
            )
            continue

        boot_only = br.note == "server_lifecycle_boot_only"
        # Commit the higher-CONC failed boots (genuine capacity failures) first.
        run.commit_failed_boots(arm_name, failed_boots)
        if not boot_only:
            run.results.append(br)
        run.record_rung(
            arm_name, br, _RungNote(stage=STAGE_BOOT, start_time=boot_started_iso, wall_duration_sec=boot_elapsed)
        )
        run.record_boot(arm_name, grid[: boot_idx + 1], failed_boots, booted=boot_variant)
        # Incremental flush after boot point.
        run.flush_progress()

        # Server is up: sweep remaining CONCs by reuse.
        reuse_failure: BaseException | None = None
        try:
            await _sweep_by_reuse(run, server, grid[boot_idx:] if boot_only else grid[boot_idx + 1 :])
        except BaseException as exc:
            reuse_failure = exc
            raise
        finally:
            # Safety teardown — idempotent, no-op if already torn down.
            try:
                teardown_lifecycle_server(pid_dir=pid_dir, framework=server.framework, port=server.port)
            except Exception as exc:  # isolation boundary: one arm's server teardown, recorded as a fault
                # The last word on this arm's server: a teardown that failed here
                # leaves it alive past the arm that owned it.
                log.warning("conc_sweep single-server: arm=%s teardown failed", arm_name, exc_info=True)
                if recorder is not None:
                    recorder.record_fault(stage=f"teardown_lifecycle_server:{arm_name}", exc=exc)
            server.close(run, reuse_failure)
        return

    # Every CONC failed to boot the persistent server — retry the full grid via Option B (per-variant restart, no
    # lifecycle) which may succeed where persistent reuse could not.
    log.warning(
        "conc_sweep single-server: arm=%s all boot attempts failed; "
        "falling back to Option B (per-variant restart) for the full ladder",
        arm_name,
    )
    run.record_boot(arm_name, grid, failed_boots)
    if recorder is not None:
        # The whole ladder is retried per-rung, and those results supersede
        # the boot attempts rather than adding to them -- which is why the
        # attempts above stay uncommitted.
        recorder.record_arm_strategy(
            arm_name, replace(arm_strategy, strategy=STRATEGY_SERVER_RESTART, reason="all_boot_attempts_failed")
        )
    await _sweep_arm_option_b(run, server, grid)


async def _sweep_by_reuse(run: _SweepRun, server: _ArmServer, reuse_grid: list[GridVariant]) -> None:
    """Run the rungs below the boot rung against the arm's already-booted server; the last one tears it down."""
    arm_name = server.arm_name
    for r_idx, variant in enumerate(reuse_grid):
        _reuse_remaining = session_deadline_to_remaining_sec(run.session_deadline_sec)
        # Check cancel and session deadline before each reuse point.
        if run.skip_rest(arm_name, reuse_grid[r_idx:]):
            break

        is_last = r_idx == len(reuse_grid) - 1
        reuse_started_iso = now_iso("seconds")
        reuse_started_at = time.time()
        server_lifecycle_reuse = {
            "cleanup": is_last,
            "pid_dir": str(server.pid_dir),
            "port": server.port,
        }
        try:
            reuse_results = await run.run_rung(
                variant,
                GridLaunch(
                    serving_lease=server.lease,
                    server_lifecycle=server_lifecycle_reuse,
                    server_already_ready=True,
                    warmup_before_measure=False,
                ),
            )
        except Exception as exc:  # isolation boundary: one reuse rung on the shared server
            log.warning(
                "conc_sweep single-server: arm=%s reuse conc=%s raised %r",
                arm_name,
                variant.extra_envs.get("CONC", "?"),
                exc,
                exc_info=True,
            )
            reuse_results = [_raised_result(variant, "single_server_reuse", exc)]
        reuse_elapsed = round(time.time() - reuse_started_at, 3)
        run.record_ran(
            arm_name,
            reuse_results,
            _RungNote(
                stage=STAGE_REUSE,
                start_time=reuse_started_iso,
                wall_duration_sec=reuse_elapsed,
                budget_remaining_sec=_reuse_remaining,
            ),
        )
        # Incremental flush after each reuse point.
        run.flush_progress()


async def _sweep_arm_option_b(run: _SweepRun, server: _ArmServer, grid: list[GridVariant]) -> None:
    """Option B fallback: run each variant with its own server (legacy behaviour).

    Used when ``_sweep_one_arm_single_server`` detects the framework is not
    lifecycle-eligible or all boot retries are exhausted. The same monotonic
    deadline covers both arms and all retries. ``server.lease`` is ``None``
    when the arm runs on the local (non-Ray) path.
    """
    arm_name = server.arm_name
    arm_failure: BaseException | None = None
    try:
        for variant in grid:
            _ob_rem = session_deadline_to_remaining_sec(run.session_deadline_sec)
            if run.skip_rest(arm_name, [variant]):
                continue
            rung_started_iso = now_iso("seconds")
            rung_started_at = time.time()
            try:
                sub = await run.run_rung(variant, GridLaunch(serving_lease=server.lease))
            except Exception as exc:  # isolation boundary: one option-B rung on its own server
                log.warning(
                    "conc_sweep option B: arm=%s variant=%s raised %r", arm_name, variant.name, exc, exc_info=True
                )
                sub = [_raised_result(variant, "option_b", exc)]
            rung_elapsed = round(time.time() - rung_started_at, 3)
            run.record_ran(
                arm_name,
                sub,
                _RungNote(
                    stage=STAGE_SERVER_RESTART,
                    start_time=rung_started_iso,
                    wall_duration_sec=rung_elapsed,
                    budget_remaining_sec=_ob_rem,
                ),
            )
            run.flush_progress()
    except BaseException as exc:
        arm_failure = exc
        raise
    finally:
        server.close(run, arm_failure)


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
    return None


async def _sweep_arms(run: _SweepRun, grid_source: str) -> None:
    """Record the budget and plan, then run the optimized arm and the baseline arm in turn."""
    recorder = run.recorder
    total_budget_sec = run.total_budget_sec
    if recorder is not None:
        recorder.record_budget(
            SweepBudget(
                declared_total_sec=total_budget_sec,
                rung_cost_sec=run.variant_expected_sec,
                deadline=run.started_at + total_budget_sec if total_budget_sec is not None else None,
            )
        )
    log.info(
        "conc_sweep (single-server): arms=optimized,baseline concs=%s isl=%d osl=%d total_budget=%s",
        run.concs_desc,
        run.isl,
        run.osl,
        f"{total_budget_sec}s" if total_budget_sec is not None else "unbounded",
    )
    arms_order = [run.optimized, run.baseline]
    if recorder is not None:
        recorder.record_plan(
            SweepPlan(
                concs_requested=run.concs,
                concs_ordered=run.concs_desc,
                grid_source=grid_source,
                num_prompts_factor=DEFAULT_NUM_PROMPTS_FACTOR,
                variant_timeout_sec=run.benchmark_timeout_sec,
                arms_order=[arm.name for arm in arms_order],
            )
        )
    for arm in arms_order:
        if run.refuse_arm(arm):
            continue
        if recorder is not None:
            recorder.open_arm(arm.name, extra_server_args=arm.args, extra_envs=arm.envs)
        await _sweep_one_arm_single_server(run, arm)


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
    session_deadline_sec, variant_expected_sec = session_grid_bounds(state)
    session_dir = Path(session_dir)
    # Whether the ladder was handed to the sweep or picked for the workload --
    # a distinction only this line can still see, since the two are the same
    # list one statement later.
    grid_source = GRID_REQUESTED if concs is not None else GRID_MODE_DEFAULT
    # ``None`` → default ladder; an explicit empty list short-circuits below.
    concs = list(concs) if concs is not None else default_concs_for_mode(getattr(state, "benchmark_mode", ""))
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
    resolved_model = resolve_session_model_path(
        state_model_path=str(getattr(state, "model_path", "") or ""),
        for_serving=True,
    )
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
    deadline_stop = STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS]
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
