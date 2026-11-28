# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The next-phase decision replayed from frozen predicate inputs, and the per-tick phase block."""

from __future__ import annotations

import time
from typing import Any

from hyperloom_common.coerce import to_float
from hyperloom.inference_optimizer.breakdown.stop_reasons import is_valid_stop_reason
from . import machine_state as _phase_state
from .controller_settlement import CONTROLLER_TERMINAL_STATUSES
from .exit_predicates import budget_exit, framework_exit
from .phase_chain import (
    PHASE_CLOSE,
    PHASE_CYCLE_END,
    PHASE_ENABLEMENT,
    PHASE_FRAMEWORK_AGENT,
    PHASE_KERNEL_AGENT,
    PHASE_PRELUDE,
    allowed_actions_for,
    phase_index,
)
from .phase_predicates import (
    PhaseFlags,
    WorkInFlight,
    cycle_end_predicate_inputs,
    initial_analysis_inputs,
    workflow_predicate_inputs,
)
from .prelude_exits import _baseline_is_anchored, _cold_anchor_shortfall, _prelude_budget_line, _prelude_viability


def phase_status_summary(
    state: Any,
    *,
    now_unix: float | None = None,
) -> str:
    """Render the per-tick ``=== Phase ===`` block (≤7 lines). The ``entered`` line carries the reason of the transition into the phase. The mid-chain phases add a ``cycle_reloop`` line showing whether another macro-cycle is still affordable, and a phase held for its initial roofline a ``hold`` line."""
    phase = (state.phase or "").strip().upper() or "UNSET"
    elapsed = int(_phase_state.phase_elapsed_seconds(state, now_unix=now_unix))
    # ``remaining`` paces this entry; ``cumulative`` totals every entry.
    cumulative = int(_phase_state.phase_cumulative_seconds(state, now_unix=now_unix))
    budget_pct_for_phase = state.phase_budget_pct.get(phase, 0.0)
    remaining = _phase_state.phase_budget_remaining_seconds(state, now_unix=now_unix)
    budget_line: str
    if remaining is None:
        budget_line = f"budget    : pct={budget_pct_for_phase:.2f} (unlimited run; no per-phase cap)"
    else:
        budget_line = (
            f"budget    : pct={budget_pct_for_phase:.2f} elapsed_sec={elapsed} "
            f"cumulative_sec={cumulative} remaining_sec={int(remaining)}"
        )
    entered = (
        time.strftime("%FT%TZ", time.gmtime(t0)) if (t0 := _phase_state.phase_started_unix(state)) > 0 else "(unset)"
    )
    if entry_reason := _phase_state.phase_entry_reason(state):
        entered += f" reason={entry_reason}"
    actions_in_phase = allowed_actions_for(phase)
    allowed_line = f"allowed   : {', '.join(actions_in_phase) if actions_in_phase else '(none)'}"
    lines = [
        f"phase     : {phase}",
        f"cycle     : {state.macro_cycle}",
        f"entered   : {entered}",
        budget_line,
        allowed_line,
    ]
    # Whether deferring work to a later cycle is still a real option.
    if phase in (PHASE_ENABLEMENT, PHASE_FRAMEWORK_AGENT, PHASE_KERNEL_AGENT, PHASE_CYCLE_END):
        feasible, evidence = cycle_reloop_decision(state, now_unix=now_unix)
        reloop_line = f"reloop    : cycle_reloop_feasible={'true' if feasible else 'false'}"
        threshold = evidence.get("min_remaining_sec_effective")
        if threshold is not None:
            reloop_line += f" threshold_sec={int(threshold)}"
        session_remaining = _phase_state.session_remaining_seconds(state, now_unix=now_unix)
        if session_remaining is not None:
            reloop_line += f" session_remaining_sec={int(session_remaining)}"
        blocked = evidence.get("reloop_blocked")
        if blocked:
            reloop_line += f" blocked={blocked}"
        if phase != PHASE_CYCLE_END:
            reloop_line += " (projected)"
        lines.append(reloop_line)
    if hold_line := _initial_analysis_hold_line(state, phase):
        lines.append(hold_line)
    return "\n".join(lines)


_STOP_TIME_EXHAUSTED = "time_exhausted"


#: CYCLE_END's close reason per ``reloop_blocked`` value.
_RELOOP_BLOCK_TERMINALS: dict[str, str] = {
    "global_converged": "global_converged",
    "all_directions_saturated": "global_converged",
    "target_reached": "target_reached",
    "insufficient_remaining": _STOP_TIME_EXHAUSTED,
}


def _post_prelude_target(*, optimize_enabled: bool, kernel_enabled: bool) -> str:
    """First active work phase after PRELUDE or ENABLEMENT: FRAMEWORK_AGENT, else KERNEL_AGENT, else CYCLE_END."""
    if optimize_enabled:
        return PHASE_FRAMEWORK_AGENT
    if kernel_enabled:
        return PHASE_KERNEL_AGENT
    return PHASE_CYCLE_END


def _transition_result(
    target: str,
    reason: str,
    evidence: dict[str, Any],
    predicate_inputs: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    return target, reason, {**evidence, "predicate_inputs": predicate_inputs}


def _controller_round_settled(pending: dict[str, Any], macro_cycle: Any) -> bool:
    """Whether the Forge rewrite controller has finished the round this macro-cycle opened."""
    return (
        pending.get("optimizer") == "forge"
        and int(pending.get("controller_cycle", -1)) == int(macro_cycle or 0)
        and str(pending.get("controller_status") or "") in CONTROLLER_TERMINAL_STATUSES
    )


def _kernel_idle_exit(plateau: dict[str, Any], now_unix: Any) -> tuple[str, dict[str, Any]] | None:
    """The no-progress exit a KERNEL phase takes once it has been quiet for long enough."""
    idle_ticks = int(plateau.get("idle_ticks") or 0)
    idle_max_ticks = int(plateau.get("idle_max_ticks") or 0)
    idle_since = to_float(plateau.get("idle_since_unix")) or 0.0
    if idle_ticks < idle_max_ticks or idle_since <= 0.0:
        return None
    idle_seconds = max(0.0, (to_float(now_unix) or 0.0) - idle_since)
    idle_min = to_float(plateau.get("idle_min_seconds")) or 0.0
    if idle_seconds < idle_min:
        return None
    return "kernel_no_more_leverage", {
        "evidence": "kernel_idle_no_progress",
        "idle_ticks": idle_ticks,
        "idle_max_ticks": idle_max_ticks,
        "idle_seconds": round(idle_seconds, 3),
        "idle_min_seconds": idle_min,
    }


def _kernel_exit(inputs: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    pending = dict(inputs.get("pending_work") or {})
    plateau = dict(inputs.get("plateau") or {})
    work_pending = bool(pending.get("kernel_work_pending"))
    if not bool(pending.get("kernel_agent_in_flight")):
        controller_settled = _controller_round_settled(pending, inputs.get("macro_cycle"))
        if controller_settled and not work_pending:
            return "kernel_controller_done", {
                "controller_status": pending.get("controller_status"),
                "patch_count": int(pending.get("controller_patch_count") or 0),
                "task_count": int(pending.get("controller_task_count") or 0),
                "reason": str(pending.get("controller_reason") or ""),
            }
        if not work_pending:
            if pending.get("geak_status") == _phase_state.GEAK_STATUS_ERROR:
                return "kernel_geak_failed", {
                    "geak_status": _phase_state.GEAK_STATUS_ERROR,
                    "error_class": pending.get("geak_error_class"),
                    "error": pending.get("geak_error"),
                }
            return "kernel_no_more_leverage", {"evidence": "kernel_settled"}
        idle = _kernel_idle_exit(plateau, inputs.get("now_unix"))
        if idle is not None:
            return idle
    return budget_exit(
        dict(inputs.get("budget") or {}),
        exhausted_reason="kernel_phase_budget_exhausted",
        evidence={"rejected_kernel_count": int(plateau.get("rejected_kernel_count") or 0)},
    )


def _cycle_end_exit(inputs: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """CYCLE_END is left once its entry work settled the cycle; it has no budget of its own to run out."""
    if bool(dict(inputs.get("cycle_end_result") or {}).get("settled")):
        return "cycle_end_done", {}
    return None


def _reloop_block_evidence(
    facts: dict[str, Any],
    global_inputs: dict[str, Any],
    streak: int,
) -> dict[str, Any]:
    """The evidence naming the rule that refuses another macro-cycle; empty when none does."""
    if str(global_inputs.get("target_reached_at") or ""):
        return {"reloop_blocked": "target_reached"}
    saturated = dict(facts.get("saturated_directions") or {})
    if saturated and all(bool(value) for value in saturated.values()):
        return {"reloop_blocked": "all_directions_saturated", "saturated_directions": sorted(saturated)}
    if streak >= int(facts.get("no_gain_cycles") or 0):
        return {"reloop_blocked": "global_converged"}
    return {}


def _reloop_decision(inputs: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    facts = dict(dict(inputs.get("cycle_end_result") or {}).get("reloop") or {})
    cycle = int(inputs.get("macro_cycle") or 0)
    current_gain = to_float(facts.get("current_gain_pct")) or 0.0
    start_gain = to_float(facts.get("gain_at_cycle_start_pct")) or 0.0
    threshold = to_float(facts.get("min_gain_pct")) or 0.0
    gained = (current_gain - start_gain) > threshold
    streak = 0 if gained else int(facts.get("prior_no_gain_cycle_streak") or 0) + 1
    evidence: dict[str, Any] = {
        "macro_cycle": cycle,
        "min_gain_pct": round(threshold, 6),
        "cycle_gain_delta": round(current_gain - start_gain, 6),
        "cycle_gained": gained,
        "no_gain_cycle_streak_effective": streak,
    }
    blocked = _reloop_block_evidence(facts, dict(inputs.get("global") or {}), streak)
    if blocked:
        evidence.update(blocked)
        return False, evidence
    minimum = to_float(facts.get("min_remaining_sec")) or 0.0
    evidence["min_remaining_sec_effective"] = round(minimum, 2)
    remaining = to_float(facts.get("session_remaining_sec"))
    if remaining is not None and remaining < minimum:
        evidence["reloop_blocked"] = "insufficient_remaining"
        evidence["session_remaining_seconds"] = round(remaining, 2)
        return False, evidence
    evidence.update(reloop=True, next_cycle=cycle + 1)
    return True, evidence


def cycle_reloop_decision(state: Any, *, now_unix: float | None = None) -> tuple[bool, dict[str, Any]]:
    """Whether another macro-cycle can open from here, with the evidence the decision rests on."""
    frozen_now = float(now_unix if now_unix is not None else _phase_state.state_now_unix(state))
    cycle_end_result, _ = cycle_end_predicate_inputs(state, now_unix=frozen_now)
    reloop, evidence = _reloop_decision(
        {
            "macro_cycle": state.macro_cycle,
            "global": {"target_reached_at": str(state.target_reached_at or "")},
            "cycle_end_result": cycle_end_result,
        }
    )
    return reloop and state.framework_agent_phase_enabled, evidence


def compute_next_phase(
    state: Any,
    *,
    flags: PhaseFlags = PhaseFlags(),
    now_unix: float | None = None,
    in_flight: WorkInFlight = WorkInFlight(),
    framework_round_fits: bool = True,
) -> tuple[str, str, dict[str, Any]] | None:
    """Return the next transition with the primitive inputs it consumed."""
    inputs = workflow_predicate_inputs(
        state, flags=flags, now_unix=now_unix, in_flight=in_flight, framework_round_fits=framework_round_fits
    )
    return replay_next_phase(inputs)


def _holds_for_initial_analysis(baseline: dict[str, Any]) -> bool:
    """Whether the phase must stay open for an initial roofline it enqueued but has not booked yet.

    Leaving now would hand that analysis to the next phase instead of seeding this one with it.
    The hold is bounded by the optimization reserve, which it must never eat into.
    """
    if not bool(baseline.get("initial_analysis_pending")):
        return False
    affordable = to_float(baseline.get("analysis_affordable_sec"))
    return affordable is None or affordable > 0.0


def _initial_analysis_hold_line(state: Any, phase: str) -> str | None:
    """The ``hold`` line of a PRELUDE or ENABLEMENT kept open for its initial roofline, or ``None`` when none holds it."""
    if phase not in (PHASE_PRELUDE, PHASE_ENABLEMENT) or (to_float(state.baseline_tput) or 0.0) <= 0.0:
        return None
    facts = initial_analysis_inputs(state)
    if not _holds_for_initial_analysis(facts):
        return None
    affordable = facts["analysis_affordable_sec"]
    bound = "" if affordable is None else f", for at most {int(affordable)}s more"
    return (
        f"hold      : waiting for initial roofline {state.auto_roofline_pending_task_id.strip()}{bound}. "
        f"No hint advances {phase}, and remaining_sec above is this phase's share, not the session's; "
        "answer each tick with a send_message observation until it moves on."
    )


def _terminal_close_transition(inputs: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    """The CLOSE a stop reason or the closing deadline forces, if one does."""
    if str(inputs["current_phase"]) == PHASE_CLOSE:
        return None
    global_inputs = dict(inputs.get("global") or {})
    stop_reason = str(global_inputs.get("stop_reason") or "")
    if stop_reason:
        evidence = {"reason_origin": "shared_state.stop_reason"}
        if not is_valid_stop_reason(stop_reason):
            evidence["vocab"] = "unknown"
        return _transition_result(PHASE_CLOSE, stop_reason, {"terminal": True, **evidence}, inputs)
    if bool(global_inputs.get("closing_phase")):
        return _transition_result(
            PHASE_CLOSE,
            _STOP_TIME_EXHAUSTED,
            {"terminal": True, "reason_origin": "closing_phase"},
            inputs,
        )
    return None


def _anchored_baseline_transition(
    inputs: dict[str, Any],
    flags: dict[str, Any],
    baseline: dict[str, Any],
    viability: dict[str, Any],
    reason: str,
) -> tuple[str, str, dict[str, Any]]:
    """The phase an anchored baseline hands off to, and the evidence it hands over."""
    target = _post_prelude_target(
        optimize_enabled=bool(flags.get("framework_agent_enabled")),
        kernel_enabled=bool(flags.get("kernel_enabled")),
    )
    evidence = {"baseline_tput": float(baseline["tput"]), **viability}
    if target != PHASE_FRAMEWORK_AGENT:
        evidence["optimize_skipped"] = True
    return _transition_result(target, reason, evidence, inputs)


def _prelude_transition(
    inputs: dict[str, Any],
    flags: dict[str, Any],
) -> tuple[str, str, dict[str, Any]] | None:
    """The transition PRELUDE takes once its baseline round has settled."""
    baseline = dict(inputs.get("baseline") or {})
    pending = dict(inputs.get("pending_work") or {})
    streak = int(baseline.get("failure_streak") or 0)
    if streak >= 3:
        return _transition_result(
            PHASE_CLOSE,
            "prelude_baseline_failed",
            {"terminal": True, "baseline_failure_streak": streak},
            inputs,
        )
    shortfall = _cold_anchor_shortfall(baseline, pending)
    if shortfall is not None:
        return _transition_result(PHASE_CLOSE, "prelude_cold_anchor_low_budget", shortfall, inputs)
    if bool(flags.get("enablement_enabled")) and streak >= 1:
        return _transition_result(
            PHASE_ENABLEMENT,
            "enablement_entered",
            {"baseline_failure_streak": streak},
            inputs,
        )
    if _baseline_is_anchored(baseline):
        if _holds_for_initial_analysis(baseline):
            return None
        return _anchored_baseline_transition(inputs, flags, baseline, _prelude_viability(baseline), "prelude_done")
    usable = to_float(baseline.get("session_usable_sec"))
    if usable is not None and usable <= 0.0:
        return _transition_result(
            PHASE_CLOSE,
            "time_exhausted_during_prelude",
            {
                "terminal": True,
                "session_usable_sec": round(usable, 1),
                "prelude_spent_sec": round(to_float(baseline.get("phase_spent_sec")) or 0.0, 1),
            },
            inputs,
        )
    reserve_line = _prelude_budget_line(baseline, pending)
    if reserve_line is not None:
        return _transition_result(PHASE_CLOSE, "prelude_budget_exhausted", reserve_line, inputs)
    return None


def _enablement_transition(
    inputs: dict[str, Any],
    flags: dict[str, Any],
) -> tuple[str, str, dict[str, Any]] | None:
    """The transition ENABLEMENT takes once the anchor it promoted has settled."""
    baseline = dict(inputs.get("baseline") or {})
    pending = dict(inputs.get("pending_work") or {})
    if (
        (to_float(baseline.get("tput")) or 0.0) > 0.0
        and not bool(pending.get("validation_pending"))
        and not bool(pending.get("enablement_in_flight"))
    ):
        if _holds_for_initial_analysis(baseline):
            # A session whose first baseline failed promotes its first anchor here, and that is what
            # enqueues the initial roofline, so this exit faces the same race PRELUDE does.
            return None
        return _anchored_baseline_transition(inputs, flags, baseline, {}, "enablement_done")
    return None


def _framework_transition(
    inputs: dict[str, Any],
    flags: dict[str, Any],
) -> tuple[str, str, dict[str, Any]] | None:
    """The transition FRAMEWORK_AGENT takes when its leverage or its budget runs out."""
    normal = framework_exit(inputs)
    if normal is None:
        return None
    if bool(flags.get("kernel_enabled")):
        return _transition_result(PHASE_KERNEL_AGENT, normal[0], normal[1], inputs)
    return _transition_result(PHASE_CYCLE_END, normal[0], {**normal[1], "kernel_skipped": True}, inputs)


def _kernel_transition(inputs: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    """The transition KERNEL_AGENT takes when its leverage or its budget runs out."""
    normal = _kernel_exit(inputs)
    if normal is None:
        return None
    return _transition_result(PHASE_CYCLE_END, normal[0], normal[1], inputs)


def _cycle_end_transition(
    inputs: dict[str, Any],
    flags: dict[str, Any],
) -> tuple[str, str, dict[str, Any]] | None:
    """The transition CYCLE_END takes: another macro-cycle, or the close the run has earned."""
    normal = _cycle_end_exit(inputs)
    if normal is None:
        return None
    exit_reason, exit_evidence = normal
    reloop, reloop_evidence = _reloop_decision(inputs)
    cycles_enabled = bool(flags.get("framework_agent_enabled"))
    if reloop and cycles_enabled:
        return _transition_result(
            PHASE_FRAMEWORK_AGENT,
            "cycle_reloop",
            {**exit_evidence, **reloop_evidence, "loopback": True},
            inputs,
        )
    blocked = str(reloop_evidence.get("reloop_blocked") or "")
    # Without FRAMEWORK_AGENT no further cycle was possible, so only a reached target outranks the exit's own reason.
    terminal_reason = _RELOOP_BLOCK_TERMINALS.get(blocked) if cycles_enabled or blocked == "target_reached" else None
    if terminal_reason is not None:
        return _transition_result(
            PHASE_CLOSE,
            terminal_reason,
            {**exit_evidence, **reloop_evidence, "terminal": True},
            inputs,
        )
    return _transition_result(PHASE_CLOSE, exit_reason, {**exit_evidence, **reloop_evidence}, inputs)


def replay_next_phase(inputs: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    """Recompute a phase decision from persisted primitive facts only."""
    forced = _terminal_close_transition(inputs)
    if forced is not None:
        return forced
    current = str(inputs["current_phase"])
    target_at = str(dict(inputs.get("global") or {}).get("target_reached_at") or "")
    if target_at and phase_index(PHASE_ENABLEMENT) <= phase_index(current) < phase_index(PHASE_CYCLE_END):
        return _transition_result(PHASE_CYCLE_END, "target_reached", {"target_reached_at": target_at}, inputs)
    flags = dict(inputs.get("run_flags") or {})
    if current == PHASE_PRELUDE:
        return _prelude_transition(inputs, flags)
    if current == PHASE_ENABLEMENT:
        return _enablement_transition(inputs, flags)
    if current == PHASE_FRAMEWORK_AGENT:
        return _framework_transition(inputs, flags)
    if current == PHASE_KERNEL_AGENT:
        return _kernel_transition(inputs)
    if current == PHASE_CYCLE_END:
        return _cycle_end_transition(inputs, flags)
    return None
