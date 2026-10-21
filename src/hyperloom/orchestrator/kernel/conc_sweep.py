# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Concurrency sweep over the CONC ladder."""

from __future__ import annotations
from hyperloom.orchestrator.actions.executors._workload_envs import prepare_grid_benchmark_config

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import is_agentx_mode
from hyperloom.common.timeutil import utc_now_compact
from hyperloom.inference_optimizer.breakdown.recorder.conc_sweep_event import (
    GRID_MODE_DEFAULT,
    GRID_REQUESTED,
    SweepAnchor,
    SweepBudget,
    SweepEnvironment,
    SweepPlan,
)
from hyperloom.inference_optimizer.session.session_paths import reports_dir, runs_root
from ._conc_sweep_ladder import _sweep_one_arm_single_server
from ._conc_sweep_run import (
    _Arm,
    _ArmCurves,
    _build_arm_grid,
    _budget_skip_result,
    _CANCEL_STOP,
    _deadline_skip_result,
    _flush_conc_sweep_report,
    _optimized_arm,
    _order_concs_desc,
    _SESSION_RESERVE,
    _SWEEP_BUDGET_STOP,
    _SweepRun,
    SCHEMA_VERSION,
)
from ..actions.executors._grid_runner import GridVariant, VariantResult, session_grid_bounds
from ..actions.executors._subprocess_kill import resolve_benchmark_timeouts
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
    benchmark_timeout_sec: float

    @classmethod
    def of(cls, state: SharedState, concs: list[int] | None, budget_sec: int | None) -> _SweepRequest:
        """Resolve the ladder the sweep runs, keeping whether the caller named it."""
        _, benchmark_timeout_sec = resolve_benchmark_timeouts()
        return cls(
            # ``None`` → default ladder; an explicit empty list is refused below.
            concs=list(concs) if concs is not None else default_concs_for_mode(getattr(state, "benchmark_mode", "")),
            # Whether the ladder was handed to the sweep or picked for the workload -- a distinction only this line
            # can still see, since the two are the same list once resolved.
            grid_source=GRID_REQUESTED if concs is not None else GRID_MODE_DEFAULT,
            num_prompts_factor=DEFAULT_NUM_PROMPTS_FACTOR,
            total_budget_sec=budget_sec,
            benchmark_timeout_sec=benchmark_timeout_sec,
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
    """Materialize the benchmark config into the sweep's workspace and resolve what it launches with."""
    from hyperloom.inference_optimizer.gpu_types import gpu_runner_type

    resolved_model = resolve_session_model_path(
        state_model_path=str(getattr(state, "model_path", "") or ""),
        for_serving=True,
    )
    resolved_gpu = gpu_runner_type(
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
    def create_reports_dir(cls, session_dir: Path) -> _ReportPaths:
        """Create the reports dir and name the two files in it, so a checkpoint written mid-sweep lands."""
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
        benchmark_timeout_sec=request.benchmark_timeout_sec,
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
        _ArmCurves(baseline=payload["baseline"]["points"], optimized=payload["optimized"]["points"]),
        concs=run.concs,
        isl=run.isl,
        osl=run.osl,
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
    request = _SweepRequest.of(state, concs, total_budget_sec)
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
        reports=_ReportPaths.create_reports_dir(session_dir),
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
