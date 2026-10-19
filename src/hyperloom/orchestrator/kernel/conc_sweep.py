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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

from hyperloom.common import io as _common_io
from hyperloom.common.gain_math import conc_pair_comparison
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import GRADED_INTVTY, GRADED_OUTPUT, is_agentx_mode
from hyperloom.common.timeutil import utc_now_compact
from hyperloom.inference_optimizer.breakdown.recorder.conc_sweep_event import (
    GRID_MODE_DEFAULT,
    GRID_REQUESTED,
    STAGE_BOOT_ATTEMPT,
    SweepAnchor,
    SweepBudget,
    SweepEnvironment,
    SweepPlan,
    SweepRung,
)
from hyperloom.inference_optimizer.grading import resolved_grading
from hyperloom.inference_optimizer.session.session_paths import reports_dir, runs_root
from ._conc_sweep_grid import (
    _Arm,
    _build_arm_grid,
    _budget_skip_result,
    _CANCEL_STOP,
    _deadline_skip_result,
    _optimized_arm,
    _order_concs_desc,
    _SWEEP_BUDGET_STOP,
)
from ._conc_sweep_ladder import _sweep_one_arm_single_server
from ..actions.executors._grid_runner import (
    GridVariant,
    VariantResult,
    run_grid,
    session_grid_bounds,
    variant_conc,
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


def _build_roofline_ceiling(
    state: SharedState,
    *,
    concs: list[int],
    isl: int,
    osl: int,
    baseline_points: list[dict[str, Any]],
    optimized_points: list[dict[str, Any]],
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

    by_conc_b = {p["conc"]: p for p in baseline_points}
    by_conc_o = {p["conc"]: p for p in optimized_points}

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


@dataclass
class _SweepRun:
    """One conc sweep's fixed inputs, and the progress both of its arms add to.

    ``results`` and ``budget`` are shared across arms on purpose: every incremental flush reports the whole sweep so
    far, and a budget stop inside one arm has to refuse the next. ``session_deadline_sec`` is the earlier monotonic
    sweep/session boundary, named by ``deadline_stop`` and distinct from the fixed per-spawn ``benchmark_timeout_sec``.
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

    @property
    def arms_order(self) -> list[_Arm]:
        """The order the two arms sweep in: the optimized side first, so a budget stop costs the baseline."""
        return [self.optimized, self.baseline]

    def arm_grid(self, arm: _Arm) -> list[GridVariant]:
        return _build_arm_grid(
            arm, self.concs_desc, isl=self.isl, osl=self.osl, num_prompts_factor=self.num_prompts_factor
        )

    def arm_results(self, arm_name: str) -> list[VariantResult]:
        prefix = f"{arm_name}_"
        return [result for result in self.results if result.name.startswith(prefix)]

    def record_rung(
        self,
        arm_name: str,
        result: VariantResult,
        *,
        stage: str,
        committed: bool = True,
        start_time: str = "",
        wall_duration_sec: float | None = None,
        budget_remaining_sec: float | None = None,
    ) -> None:
        """Record one rung on the sweep's event, if the sweep is being recorded.

        The result is flattened into the same point the report writes, so the
        recorded curve and the written one cannot differ. ``stage`` is a
        ``STAGE_*`` value naming how the rung came to run, ``committed`` says
        whether it is part of the published curve, and the seconds fields are
        wall clock.
        """
        if self.recorder is None:
            return
        point = _point_from_variant(result, arm=arm_name)
        self.recorder.record_variant(
            arm_name,
            SweepRung(
                stage=stage,
                conc=point.get("conc"),
                point=point,
                committed=committed,
                num_prompts=(result.extra_envs or {}).get("NUM_PROMPTS"),
                start_time=start_time,
                wall_duration_sec=wall_duration_sec,
                granted_cap_sec=self.benchmark_timeout_sec,
                budget_remaining_sec=budget_remaining_sec,
            ),
        )

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
        self,
        variant: GridVariant,
        *,
        serving_lease: Any,
        base_extra_envs: dict[str, str] | None = None,
        server_lifecycle: dict[str, Any] | None = None,
        server_already_ready: bool = False,
        warmup_before_measure: bool | None = None,
        lifecycle_boot_only: bool = False,
    ) -> list[VariantResult]:
        """Run one rung from the sweep's base config; a deadline stop inside it exhausts the sweep's budget."""
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
            server_lifecycle=server_lifecycle,
            base_extra_envs=base_extra_envs,
            warmup_before_measure=warmup_before_measure,
            server_already_ready=server_already_ready,
            serving_lease=serving_lease,
            session_deadline_sec=self.session_deadline_sec,
            variant_expected_sec=self.variant_expected_sec,
            deadline_stop=self.deadline_stop,
            lifecycle_boot_only=lifecycle_boot_only,
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


class _SweepRequest(NamedTuple):
    """What the caller asked this sweep to run."""

    concs: list[int]
    grid_source: str
    num_prompts_factor: int
    total_budget_sec: int | None

    @classmethod
    def of(cls, state: SharedState, concs: list[int] | None, num_prompts_factor: int, budget_sec: int | None):
        """Resolve the ladder the sweep runs, keeping whether the caller named it."""
        return cls(
            # ``None`` → default ladder; an explicit empty list is refused below.
            concs=list(concs) if concs is not None else default_concs_for_mode(getattr(state, "benchmark_mode", "")),
            # Whether the ladder was handed to the sweep or picked for the workload -- a distinction only this line
            # can still see, since the two are the same list once resolved.
            grid_source=GRID_REQUESTED if concs is not None else GRID_MODE_DEFAULT,
            num_prompts_factor=num_prompts_factor,
            total_budget_sec=budget_sec,
        )


class _SweepWorkload(NamedTuple):
    """The workload shape one sweep draws both its curves on, and the config they launch from."""

    isl: int
    osl: int
    baseline_tput: float
    optimized: _Arm | None
    base_config_path: str

    @classmethod
    def of(cls, state: SharedState) -> _SweepWorkload:
        """Read the session's workload shape, its retained arm, and the config the rungs launch from."""
        return cls(
            isl=int(getattr(state, "isl", 0) or 0),
            osl=int(getattr(state, "osl", 0) or 0),
            baseline_tput=float(getattr(state, "baseline_tput", 0.0) or 0.0),
            optimized=_optimized_arm(state),
            # Prefer the materialized baseline config; fall back to the shipped asset.
            base_config_path=str(getattr(state, "baseline_config_path", "") or "").strip()
            or str(default_baseline_config()),
        )


class _SweepRefusal(NamedTuple):
    """Why a sweep will not run, and the fields its decline envelope carries."""

    reason: str
    extras: dict[str, Any]


def _overlay_is_loadable(overlay: str) -> bool:
    """Whether a retained kernel overlay sits on a safe path and can still be loaded."""
    from hyperloom.inference_optimizer.runtime_environment import is_safe_path_entry
    from ..kernel.geak_config import geak_overlay_is_loadable

    return is_safe_path_entry(overlay) and geak_overlay_is_loadable(overlay)


def _sweep_refusal(workload: _SweepWorkload, request: _SweepRequest) -> _SweepRefusal | None:
    """Why this sweep cannot run, or ``None`` when every prerequisite is met."""
    optimized = workload.optimized
    total_budget_sec = request.total_budget_sec
    if workload.baseline_tput <= 0:
        return _SweepRefusal("no_baseline_tput", {})
    if workload.isl <= 0 or workload.osl <= 0:
        return _SweepRefusal("missing_workload_shape", {"isl": workload.isl, "osl": workload.osl})
    if optimized is None:
        return _SweepRefusal("no_optimization_to_compare", {})
    if optimized.overlay and not _overlay_is_loadable(optimized.overlay):
        return _SweepRefusal("optimized_overlay_unavailable", {"final_overlay": optimized.overlay})
    if not request.concs:
        return _SweepRefusal("empty_conc_list", {})
    # A non-positive budget is "no time left", not "budget gate off": running the
    # ladder here would spend wall-clock the caller already accounted as gone.
    # No variant started, so this is a decline (see conc_sweep_declined_to_run)
    # and must not stamp ``budget_exhausted``.
    if total_budget_sec is not None and int(total_budget_sec) <= 0:
        return _SweepRefusal("no_time_budget_remaining", {"total_budget_sec": int(total_budget_sec)})
    if not Path(workload.base_config_path).exists():
        return _SweepRefusal("baseline_config_missing", {"config_path": workload.base_config_path})
    return None


class _SweepLaunch(NamedTuple):
    """The resolved launch inputs every rung of the sweep runs with."""

    base_yaml_path: Path
    model_path: str
    gpu_type: str
    benchmark_script: str | None


def _sweep_workspace(session_dir: Path) -> Path:
    """Create this sweep's run directory; its name is the sweep's task id."""
    workspace = runs_root(session_dir) / "conc_sweep" / f"conc_sweep_{utc_now_compact()}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _materialize_sweep_config(state: SharedState, workspace: Path, base_config_path: str) -> _SweepLaunch:
    """Materialize the benchmark config into the sweep's workspace and resolve what it launches with.

    Idempotent, and re-run even when the session already named a materialized config, in case the sweep fell back to
    the shipped asset. The GPU type mirrors the main flow (baseline/sweep/...): prefer $GPU_TYPE (cli.py
    canonicalizes mi325x/mi308x -> mi300x), fall back to state.gpu_type, then canonicalize through _gpu_runner_type
    so the selected Magpie script is a shipped runner (sglang_mi300x.sh), never the unshipped sglang_mi325x.sh.
    """
    from hyperloom.inference_optimizer.gpu_types import _gpu_runner_type

    resolved_model = resolve_session_model_path(
        state_model_path=str(getattr(state, "model_path", "") or ""),
        for_serving=True,
    )
    resolved_gpu = _gpu_runner_type(
        os.environ.get("GPU_TYPE", "").strip().lower() or str(getattr(state, "gpu_type", "") or "").strip().lower()
    )
    benchmark_script = state.accepted_baseline_script()
    return _SweepLaunch(
        base_yaml_path=prepare_grid_benchmark_config(
            Path(base_config_path),
            workspace,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=benchmark_script,
            out_name="conc_sweep_base.with_envs.yaml",
            grading=getattr(state, "grading", None),
        ),
        model_path=resolved_model,
        gpu_type=resolved_gpu,
        benchmark_script=benchmark_script,
    )


class _ReportPaths(NamedTuple):
    """The report files this sweep rewrites after every rung."""

    json_path: Path
    csv_path: Path

    @classmethod
    def under(cls, session_dir: Path) -> _ReportPaths:
        """Pre-create the reports dir so incremental checkpoints can carry these paths."""
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        return cls(json_path=rdir / "conc_sweep_summary.json", csv_path=rdir / "conc_sweep_raw.csv")


@dataclass(frozen=True)
class _SweepSetup:
    """Everything one sweep resolved before its first rung: where it writes, what it launches, what it compares."""

    session_dir: Path
    workspace: Path
    workload: _SweepWorkload
    launch: _SweepLaunch
    reports: _ReportPaths

    @property
    def task_id(self) -> str:
        """The sweep's task id, which is the name of the run directory it owns."""
        return self.workspace.name


def _record_sweep_preflight(recorder: Any, state: SharedState, setup: _SweepSetup) -> None:
    """Open the sweep's event on the workload, the anchor it compares against, and where it runs."""
    if recorder is None:
        return
    workload = setup.workload
    optimized = workload.optimized
    cb = state.current_best
    recorder.record_workload(
        session_id=getattr(state, "session_id", "") or setup.session_dir.name,
        isl=workload.isl,
        osl=workload.osl,
        tp=getattr(state, "tp", 0),
        benchmark_mode=getattr(state, "benchmark_mode", ""),
    )
    recorder.record_anchor(
        SweepAnchor(
            baseline_tput=workload.baseline_tput,
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
            sweep_task_id=setup.task_id,
            workspace=setup.workspace.as_posix(),
            model_path=setup.launch.model_path,
            gpu_type=setup.launch.gpu_type,
            base_config_path=setup.launch.base_yaml_path.as_posix(),
            report_json_path=setup.reports.json_path.as_posix(),
            report_csv_path=setup.reports.csv_path.as_posix(),
        )
    )


class _SweepDeadline(NamedTuple):
    """When this sweep's rungs stop running, and whose stop names that boundary."""

    at_monotonic_sec: float | None
    stop: StoppedByTheRun

    @classmethod
    def earlier_of(cls, session_deadline_sec: float | None, total_budget_sec: int | None) -> _SweepDeadline:
        """The earlier of the session's deadline and this sweep's own budget."""
        session_stop = STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS]
        if total_budget_sec is None:
            return cls(session_deadline_sec, session_stop)
        sweep_deadline_sec = time.monotonic() + total_budget_sec
        if session_deadline_sec is None or sweep_deadline_sec < session_deadline_sec:
            return cls(sweep_deadline_sec, _SWEEP_BUDGET_STOP)
        return cls(session_deadline_sec, session_stop)


def _record_sweep_budget(recorder: Any, request: _SweepRequest, started_at: float, rung_cost_sec: float | None) -> None:
    """Declare the wall-clock this sweep was granted."""
    if recorder is None:
        return
    total_budget_sec = request.total_budget_sec
    recorder.record_budget(
        SweepBudget(
            declared_total_sec=total_budget_sec,
            granted_total_sec=total_budget_sec,
            rung_cost_sec=rung_cost_sec,
            raised=False,
            gate_active=total_budget_sec is not None,
            deadline=started_at + total_budget_sec if total_budget_sec is not None else None,
        )
    )


def _sweep_run(state: SharedState, setup: _SweepSetup, request: _SweepRequest, recorder: Any) -> _SweepRun:
    """Assemble the run both arms add their rungs to, and open its budget on the event."""
    _, benchmark_timeout_sec = resolve_benchmark_timeouts()
    session_deadline_sec, variant_expected_sec = session_grid_bounds(state)
    deadline = _SweepDeadline.earlier_of(session_deadline_sec, request.total_budget_sec)
    started_at = time.time()
    _record_sweep_budget(recorder, request, started_at, variant_expected_sec)
    workload = setup.workload
    assert workload.optimized is not None
    return _SweepRun(
        state=state,
        session_dir=setup.session_dir,
        workspace=setup.workspace,
        base_yaml_path=setup.launch.base_yaml_path,
        model_path=setup.launch.model_path,
        gpu_type=setup.launch.gpu_type,
        benchmark_script=setup.launch.benchmark_script,
        isl=workload.isl,
        osl=workload.osl,
        concs=request.concs,
        num_prompts_factor=request.num_prompts_factor,
        optimized=workload.optimized,
        baseline=_Arm(name="baseline"),
        benchmark_timeout_sec=benchmark_timeout_sec,
        session_deadline_sec=deadline.at_monotonic_sec,
        variant_expected_sec=variant_expected_sec,
        deadline_stop=deadline.stop,
        started_at=started_at,
        total_budget_sec=request.total_budget_sec,
        json_path=setup.reports.json_path,
        csv_path=setup.reports.csv_path,
        recorder=recorder,
    )


class _ArmRefusal(NamedTuple):
    """Why an arm never ran: what the event records, the time left, and how each of its rungs is filed."""

    reason: str
    remaining_sec: float
    skip_result: Callable[[GridVariant], VariantResult]


def _refuse_next_arm(run: _SweepRun) -> _ArmRefusal | None:
    """Refuse the next arm when the run is out of time, spending the session reserve if that is what stopped it."""
    if run.observe_cancel():
        return _ArmRefusal(ORCHESTRATOR_CANCELLED_CLASS, 0.0, lambda v: _deadline_skip_result(v, _CANCEL_STOP))
    if run.budget.refuses_next_arm():
        return _ArmRefusal(
            run.budget.skip_reason,
            run.budget.remaining_sec,
            lambda v: _deadline_skip_result(v, run.deadline_stop),
        )
    if run.session_closing():
        run.budget.spend_on_session_reserve()
        return _ArmRefusal(_SESSION_RESERVE, 0.0, _budget_skip_result)
    return None


async def _sweep_both_arms(run: _SweepRun, request: _SweepRequest) -> None:
    """Run the optimized arm's ladder and then the baseline's, filing the rungs of any arm there is no time for."""
    if run.recorder is not None:
        run.recorder.record_plan(
            SweepPlan(
                concs_requested=request.concs,
                concs_ordered=run.concs_desc,
                grid_source=request.grid_source,
                num_prompts_factor=request.num_prompts_factor,
                variant_timeout_sec=run.benchmark_timeout_sec,
                arms_order=[arm.name for arm in run.arms_order],
            )
        )
    for arm in run.arms_order:
        refusal = _refuse_next_arm(run)
        if refusal is not None:
            run.results.extend(refusal.skip_result(variant) for variant in run.arm_grid(arm))
            if run.recorder is not None:
                run.recorder.record_arm_refused(arm.name, reason=refusal.reason, remaining_sec=refusal.remaining_sec)
            continue
        if run.recorder is not None:
            run.recorder.open_arm(arm.name, extra_server_args=arm.args, extra_envs=arm.envs)
        await _sweep_one_arm_single_server(run, arm)


def _publish_sweep_report(run: _SweepRun, payload: dict[str, Any]) -> None:
    """Write the sweep's report and close its event on what the ladder measured."""
    report_error = run.write(payload)
    summary = payload["summary"]
    if run.recorder is not None:
        if report_error is not None:
            run.recorder.record_fault(stage="report_write", exc=report_error)
        run.recorder.record_progress(comparison=payload["comparison"], summary=summary)
        run.recorder.finish(payload, stop_reason=getattr(run.state, "stop_reason", ""))
    log.info(
        "conc_sweep: done — status=%s successful_pairs=%d failed_pairs=%d best_speedup=%s%s",
        payload["status"],
        summary["successful_pairs"],
        summary["failed_pairs"],
        summary["best_speedup"],
        f" cancelled_by={run.cancel_reason}" if run.cancelled else "",
    )


def _sweep_payload_with_ceiling(run: _SweepRun, state: SharedState) -> dict[str, Any]:
    """The sweep's final report body, with the decode roofline drawn beside the measured curves."""
    payload = run.payload(in_progress=False)
    ceiling = _build_roofline_ceiling(
        state,
        concs=run.concs,
        isl=run.isl,
        osl=run.osl,
        baseline_points=payload["baseline"]["points"],
        optimized_points=payload["optimized"]["points"],
    )
    if ceiling is not None:
        payload["roofline_ceiling"] = ceiling
    return payload


async def run_conc_sweep(
    state: SharedState,
    session_dir: Path,
    *,
    concs: list[int] | None = None,
    total_budget_sec: int | None = DEFAULT_TOTAL_BUDGET_SEC,
    num_prompts_factor: int = DEFAULT_NUM_PROMPTS_FACTOR,
    recorder: Any = None,
) -> dict[str, Any]:
    """Run the full conc-sweep SWEEP-phase action end-to-end (always returns a dict; never raises; no files written when skipped).

    ``concs`` of ``None`` uses the default ladder. ``total_budget_sec`` of
    ``None`` removes only the sweep budget, not the session deadline; ``<=0`` means the caller's clamp
    left no time and the sweep skips immediately. A ``None`` recorder records
    nothing, which is what a direct caller with no session bound wants.
    Returns a skip envelope when prerequisites are unmet.
    """
    session_dir = Path(session_dir)
    request = _SweepRequest.of(state, concs, num_prompts_factor, total_budget_sec)
    workload = _SweepWorkload.of(state)
    refusal = _sweep_refusal(workload, request)
    if refusal is not None:
        return _declined(recorder, refusal.reason, **refusal.extras)
    workspace = _sweep_workspace(session_dir)
    try:
        launch = _materialize_sweep_config(state, workspace, workload.base_config_path)
    except FrameworkScriptMismatchError as exc:
        return _declined(
            recorder,
            "framework_script_mismatch",
            error_class="framework_script_mismatch",
            error=str(exc),
            workspace=str(workspace),
        )
    setup = _SweepSetup(
        session_dir=session_dir,
        workspace=workspace,
        workload=workload,
        launch=launch,
        reports=_ReportPaths.under(session_dir),
    )
    _record_sweep_preflight(recorder, state, setup)
    run = _sweep_run(state, setup, request, recorder)
    log.info(
        "conc_sweep (single-server): arms=optimized,baseline concs=%s isl=%d osl=%d total_budget=%s",
        run.concs_desc,
        run.isl,
        run.osl,
        f"{total_budget_sec}s" if total_budget_sec is not None else "unbounded",
    )
    await _sweep_both_arms(run, request)
    payload = _sweep_payload_with_ceiling(run, state)
    _publish_sweep_report(run, payload)
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
