# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Primitive facts frozen for one phase decision, read off the live state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyperloom_common.coerce import optional_positive_float, to_float, to_unix
from hyperloom_common.env import env_float as _env_float, env_int as _env_int
from . import machine_state as _phase_state
from .phase_chain import (
    PHASE_CYCLE_END,
    PHASE_ENABLEMENT,
    PHASE_FRAMEWORK_AGENT,
    PHASE_KERNEL_AGENT,
    PHASE_PRELUDE,
)


def _cumulative_gain_validated(state: Any) -> float:
    """Return ``state.cumulative_gain_validated``, defensively coerced to float."""
    try:
        return float(state.cumulative_gain_validated or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _kernel_idle_max_ticks() -> int:
    """Consecutive no-work KERNEL_AGENT ticks before winding down to CYCLE_END."""
    val = _env_int("INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS", default=3)
    return val if val >= 1 else 3


def _kernel_idle_min_seconds() -> float:
    """Wall-clock seconds a KERNEL idle streak must last before winding down."""
    val = _env_float("INFERENCE_OPTIMIZER_KERNEL_IDLE_MIN_SECONDS", default=600.0)
    return val if val > 0.0 else 600.0


def _kernel_idle_since_unix(state: Any) -> float:
    """Return when the current KERNEL idle streak opened, defensively coerced."""
    raw = state.kernel_idle_since_unix
    try:
        return max(0.0, float(raw or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _budget_predicate_inputs(
    state: Any,
    *,
    now_unix: float,
) -> dict[str, Any]:
    """Normalize the clocks compared by phase budget predicates."""
    remaining_sec = _phase_state.phase_budget_remaining_seconds(state, now_unix=now_unix)
    budget: dict[str, Any] = {
        "remaining_sec": remaining_sec,
        "entry_elapsed_sec": _phase_state.phase_elapsed_seconds(state, now_unix=now_unix),
        "cumulative_elapsed_sec": _phase_state.phase_cumulative_seconds(state, now_unix=now_unix),
        "current_balance": remaining_sec,
    }
    return budget


@dataclass(frozen=True)
class PhaseFlags:
    """Which optional phases a run has switched on."""

    kernel_enabled: bool = True
    optimize_enabled: bool = True
    enablement_enabled: bool = False


@dataclass(frozen=True)
class WorkInFlight:
    """Which phases still have work of their own queued or running, as one decision sees it."""

    enablement: bool = False
    kernel: bool = False
    prelude: bool = False


def _base_workflow_predicate_inputs(
    state: Any,
    *,
    current: str,
    now_unix: float,
    flags: PhaseFlags,
    enablement_in_flight: bool,
) -> dict[str, Any]:
    global_inputs: dict[str, Any] = {
        "stop_reason": str(state.stop_reason or "").strip(),
        "closing_phase": bool(state.closing_phase),
        "target_reached_at": str(state.target_reached_at or ""),
    }
    return {
        "current_phase": current,
        "now_unix": now_unix,
        "global": global_inputs,
        "baseline": None,
        "pending_work": None,
        "budget": None,
        "plateau": None,
        "hint": state.pending_escalate_hint or None,
        "cycle_end_result": None,
        "macro_cycle": state.macro_cycle,
        "run_flags": {
            "kernel_enabled": bool(flags.kernel_enabled),
            "framework_agent_enabled": bool(flags.optimize_enabled),
            "enablement_enabled": bool(flags.enablement_enabled),
        },
        "enablement_in_flight": bool(enablement_in_flight),
    }


def _prelude_predicate_inputs(state: Any, *, now_unix: float) -> dict[str, Any]:
    outcome = state.warm_replay_outcome or {}
    warm_status = str(outcome.get("status") or "").strip() if isinstance(outcome, dict) else ""
    return {
        "failure_streak": int(state.baseline_failure_streak or 0),
        "failures_this_leg": int(state.baseline_failures_this_leg or 0),
        "warm_replay_status": warm_status,
        "measure_round_dropped": bool(state.baseline_measure_round_dropped),
        "tput": to_float(state.baseline_tput) or 0.0,
        **initial_analysis_inputs(state),
        "session_usable_sec": state.session_budget_usable_sec(),
        "phase_spent_sec": _phase_state.phase_cumulative_seconds(state, phase=PHASE_PRELUDE, now_unix=now_unix),
        "runtime_sec": optional_positive_float(state.baseline_runtime_sec),
        "post_ready_runtime_sec": optional_positive_float(state.baseline_post_ready_runtime_sec),
        "warm_runtime_sec": optional_positive_float(state.baseline_warm_runtime_sec),
        "eval_runtime_sec": optional_positive_float(state.baseline_eval_runtime_sec),
    }


def _rounds_since_phase_entry(state: Any) -> list[dict[str, Any]]:
    """This cycle's specialist rounds booked since the current phase was entered.

    PRELUDE shares macro-cycle 0 with FRAMEWORK_AGENT, and ``source_phase`` names a round's ownership lane rather
    than the phase it ran in, so only the entry time separates this phase's rounds from earlier ones.
    """
    rows = _phase_state._rows_for_current_cycle(state.specialist_rounds or [], state)
    entered = _phase_state.phase_started_unix(state)
    # No entry time: keep every row of the cycle. With one: a row with no parseable completed_at is not counted.
    if entered <= 0.0:
        return rows
    return [row for row in rows if (to_unix(row.get("completed_at")) or 0.0) >= entered]


def _framework_predicate_inputs(
    state: Any,
    *,
    now_unix: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _all_dry, evidence = _phase_state.per_lever_dryness(state)
    plateau = {
        key: value
        for key, value in evidence.items()
        if key not in {"config_arm_plateaued", "source_arm_plateaued", "switch_bottleneck"}
    }
    plateau["attempt_count"] = len(_phase_state._rows_for_current_cycle(state.attempts or [], state))
    plateau["specialist_round_count"] = len(_rounds_since_phase_entry(state))
    return plateau, _budget_predicate_inputs(state, now_unix=now_unix)


def _kernel_predicate_inputs(
    state: Any,
    *,
    now_unix: float,
    kernel_work_in_flight: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    controller = state.kernel_rewrite_controller_result or {}
    if not isinstance(controller, dict):
        controller = {}
    pending = {
        "kernel_work_pending": bool(_phase_state.kernel_work_pending(state)),
        "kernel_agent_in_flight": bool(kernel_work_in_flight),
        "optimizer": str(state.kernel_optimizer or "").strip().lower(),
        "controller_cycle": int(to_float(controller.get("macro_cycle"), -1)),
        "controller_status": str(controller.get("status") or "").strip().lower(),
        "controller_patch_count": int(controller.get("patch_count") or 0),
        "controller_task_count": int(controller.get("task_count") or 0),
        "controller_reason": str(controller.get("reason") or ""),
    }
    if pending["optimizer"] == "geak":
        geak = state.geak_result if isinstance(state.geak_result, dict) else {}
        pending["geak_status"] = str(geak.get("status") or "").strip().lower()
        pending["geak_error_class"] = str(geak.get("error_class") or "")
        pending["geak_error"] = str(geak.get("error") or "")[:500]
    plateau = {
        "idle_ticks": int(state.kernel_idle_ticks or 0),
        "idle_since_unix": _kernel_idle_since_unix(state),
        "idle_max_ticks": _kernel_idle_max_ticks(),
        "idle_min_seconds": _kernel_idle_min_seconds(),
        "rejected_kernel_count": len(state.rejected_kernel_ids or []),
    }
    return pending, plateau, _budget_predicate_inputs(state, now_unix=now_unix)


def cycle_end_predicate_inputs(
    state: Any,
    *,
    now_unix: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cycle = int(state.macro_cycle or 0)
    saturated = state.saturated_directions or {}
    saturated_facts = (
        {str(key): bool(value.get("saturated")) for key, value in saturated.items() if isinstance(value, dict)}
        if isinstance(saturated, dict)
        else {}
    )
    result = {
        "settled": int(state.cycle_end_settled_cycle) == cycle,
        "reloop": {
            "current_gain_pct": _cumulative_gain_validated(state),
            "gain_at_cycle_start_pct": to_float(state.gain_at_cycle_start) or 0.0,
            "min_gain_pct": _phase_state.decaying_keep_threshold_pct(cycle),
            "prior_no_gain_cycle_streak": int(state.no_gain_cycle_streak or 0),
            "no_gain_cycles": _phase_state.DEFAULT_GLOBAL_CONVERGENCE_NO_GAIN_CYCLES,
            "saturated_directions": saturated_facts,
            "session_remaining_sec": _phase_state.session_remaining_seconds(state, now_unix=now_unix),
            "min_remaining_sec": _phase_state.cycle_reloop_min_remaining_sec(state),
        },
    }
    return result, _budget_predicate_inputs(state, now_unix=now_unix)


def workflow_predicate_inputs(
    state: Any,
    *,
    flags: PhaseFlags = PhaseFlags(),
    now_unix: float | None = None,
    in_flight: WorkInFlight = WorkInFlight(),
    framework_round_fits: bool = True,
) -> dict[str, Any]:
    """Freeze primitive, normalized facts consumed by one phase decision."""
    current = (state.phase or "").strip().upper() or PHASE_PRELUDE
    frozen_now = float(now_unix if now_unix is not None else _phase_state.state_now_unix(state))
    inputs = _base_workflow_predicate_inputs(
        state,
        current=current,
        now_unix=frozen_now,
        flags=flags,
        enablement_in_flight=in_flight.enablement,
    )
    if current == PHASE_PRELUDE:
        inputs["baseline"] = _prelude_predicate_inputs(state, now_unix=frozen_now)
        inputs["pending_work"] = {"prelude_work_in_flight": bool(in_flight.prelude)}
    elif current == PHASE_ENABLEMENT:
        enablement = state.enablement
        inputs["baseline"] = {"tput": to_float(state.baseline_tput) or 0.0, **initial_analysis_inputs(state)}
        inputs["pending_work"] = {
            "validation_pending": bool(enablement.validation_pending),
            "enablement_in_flight": bool(in_flight.enablement),
        }
    elif current == PHASE_FRAMEWORK_AGENT:
        inputs["plateau"], inputs["budget"] = _framework_predicate_inputs(state, now_unix=frozen_now)
        inputs["budget"]["round_fits"] = framework_round_fits
    elif current == PHASE_KERNEL_AGENT:
        inputs["pending_work"], inputs["plateau"], inputs["budget"] = _kernel_predicate_inputs(
            state,
            now_unix=frozen_now,
            kernel_work_in_flight=in_flight.kernel,
        )
    elif current == PHASE_CYCLE_END:
        inputs["cycle_end_result"], inputs["budget"] = cycle_end_predicate_inputs(state, now_unix=frozen_now)
    return inputs


def initial_workflow_predicate_inputs(state: Any, *, current_phase: str, flags: PhaseFlags) -> dict[str, Any]:
    """Identity-transition facts for fresh start or explicit CLOSE resume."""
    return _base_workflow_predicate_inputs(
        state,
        current=current_phase,
        now_unix=float(_phase_state.state_now_unix(state)),
        flags=flags,
        enablement_in_flight=False,
    )


def initial_analysis_inputs(state: Any) -> dict[str, Any]:
    """The two facts :func:`_holds_for_initial_analysis` decides on, read once for the predicates and the block."""
    return {
        "initial_analysis_pending": bool((state.auto_roofline_pending_task_id or "").strip()),
        "analysis_affordable_sec": _phase_state.prelude_affordable_seconds(state)[0],
    }
