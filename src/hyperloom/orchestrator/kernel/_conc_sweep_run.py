# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One conc sweep's run: the two arms it compares, the rungs each one's ladder runs, and the curve they add to."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from hyperloom.common import io as _common_io
from hyperloom.common.gain_math import conc_pair_comparison
from hyperloom.common.perf_metric import GRADED_INTVTY, GRADED_OUTPUT
from hyperloom.inference_optimizer.breakdown.recorder.conc_sweep_event import STAGE_BOOT_ATTEMPT, SweepRung
from hyperloom.inference_optimizer.grading import resolved_grading
from hyperloom.inference_optimizer.session.session_paths import reports_dir
from ..actions.cancel_channel import current_cancel_scope
from ..actions.executors._grid_runner import GridVariant, VariantResult, run_grid, variant_conc
from ..actions.executors._proposal_identity import controls_of, is_executable, normalize_proposal
from ..actions.executors._subprocess_kill import session_deadline_to_remaining_sec
from ..actions.stop_attribution import (
    ORCHESTRATOR_CANCELLED_CLASS,
    SESSION_TIME_EXHAUSTED_CLASS,
    STOPPED_BY_THE_RUN,
    StoppedByTheRun,
)
from ..state.shared_state import SharedState


log = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class _RungServer:
    """The serving lease and server lifecycle a rung runs under."""

    serving_lease: Any
    base_extra_envs: dict[str, str] | None = None
    server_lifecycle: dict[str, Any] | None = None
    server_already_ready: bool = False
    warmup_before_measure: bool | None = None
    lifecycle_boot_only: bool = False


@dataclass(frozen=True)
class _ArmCurves:
    """The measured points of both arms of a sweep."""

    baseline: list[dict[str, Any]]
    optimized: list[dict[str, Any]]


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
                granted_cap_sec=self.benchmark_timeout_sec,
                budget_remaining_sec=note.budget_remaining_sec,
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

    async def run_rung(self, variant: GridVariant, server: _RungServer) -> list[VariantResult]:
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
            server_lifecycle=server.server_lifecycle,
            base_extra_envs=server.base_extra_envs,
            warmup_before_measure=server.warmup_before_measure,
            server_already_ready=server.server_already_ready,
            serving_lease=server.serving_lease,
            session_deadline_sec=self.session_deadline_sec,
            variant_expected_sec=self.variant_expected_sec,
            deadline_stop=self.deadline_stop,
            lifecycle_boot_only=server.lifecycle_boot_only,
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
    except (OSError, ValueError, csv.Error) as exc:
        log.warning("conc_sweep: _flush_conc_sweep_report failed", exc_info=True)
        return exc
    return None
