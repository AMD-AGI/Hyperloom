# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase state machine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

from hyperloom.common.coerce import optional_positive_float, to_float, to_unix
from hyperloom.common.env import env_float as _env_float, env_int as _env_int
from hyperloom.common.timeutil import now_iso as _now_iso
from hyperloom.orchestrator.lever import (
    LEVER_CONFIG,
    LEVER_SOURCE_PATCH,
    LEVER_UPSTREAM_PR,
)
from hyperloom.inference_optimizer.breakdown.stop_reasons import is_valid_stop_reason
from ..kernel import kernel_decisions
from ..state.shared_state import (
    ESCALATE_HINT_SKIP_TO_KERNEL,
    ESCALATE_HINT_SKIP_TO_SWEEP,
    _LIFECYCLE_CAP,
    is_valid_escalate_hint,
)
from .phase_chain import (
    LIFECYCLE_STATUS_ENTER,
    LIFECYCLE_STEP_LABELS,
    PHASE_ALLOWED_ACTIONS,
    PHASE_CLOSE,
    PHASE_COORDINATOR_RESERVED,
    PHASE_ENABLEMENT,
    PHASE_FRAMEWORK_AGENT,
    PHASE_HUMAN_LABELS,
    PHASE_INDEX,
    PHASE_KERNEL_AGENT,
    PHASE_NAMES,
    PHASE_PRELUDE,
    PHASE_SWEEP,
    HistoryEntry,
    LifecycleMark,
    _PHASE_HISTORY_CAP,
    allowed_actions_for,
    lifecycle_label,
    make_history_row,
    make_lifecycle_event,
    phase_index,
    render_phase_action_bullets,
)


log = logging.getLogger(__name__)


# Consecutive FAILED rounds before the lane stops with enablement_attempts_exhausted;
# abandoned and expired rounds are neutral and do not count towards it.
ENABLEMENT_MAX_ATTEMPTS: int = 8


# Default phase budgets (% of wall-clock). ENABLEMENT is absent on purpose: a
# budget apportions optimisation effort, and a combo that cannot run has nothing
# to optimise. It carries no cap at all, and so has no override flag either.
DEFAULT_PHASE_BUDGET_PCT: dict[str, float] = {
    PHASE_PRELUDE: 0.03,
    # The optimisation phase carries both levers' share.
    PHASE_FRAMEWORK_AGENT: 0.38,
    PHASE_KERNEL_AGENT: 0.47,
    PHASE_SWEEP: 0.05,
    PHASE_CLOSE: 0.02,
}

# Share of the session held back for the phases that actually produce a result.
OPTIMIZATION_RESERVE_PCT: float = 0.50

# Wall-clock ceiling for an unbounded run (``max_minutes`` == 0): the container lifetime.
DEFAULT_LONGRUN_MAX_MINUTES: int = 14 * 24 * 60


# Plateau judgment defaults (CLI --plateau-* flags); kept here for pure callers + tests.
DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT: float = 0.5
DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK: int = 5
DEFAULT_PLATEAU_EXPLORE_LOOKBACK: int = 5


# FRAMEWORK per-candidate plateau: after this many consecutive resolved candidates without a KEEP (including
# non-benchmarked terminal outcomes), the source arm is dry.
DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK: int = 5


# R1 macro-cycle reloop: SWEEP loops back to FRAMEWORK_AGENT for a new macro-cycle while budget remains and the run
# hasn't globally converged.

# Share of a bounded session's total budget that must remain to open a cycle.
_CYCLE_RELOOP_BUDGET_RATIO: float = 0.15

# Ceiling on the floor once it is raised to cover one granted variant round, so a
# session too short to fund a round is not treated as exhausted from tick one.
_CYCLE_RELOOP_MAX_BUDGET_SHARE: float = 0.5


def _default_cycle_reloop_min_remaining_sec() -> float:
    """Absolute reloop floor in seconds; env-overridable via ``INFERENCE_OPTIMIZER_CYCLE_RELOOP_MIN_REMAINING_SEC``."""
    v = _env_float("INFERENCE_OPTIMIZER_CYCLE_RELOOP_MIN_REMAINING_SEC", default=10800.0)
    return v if v > 0 else 10800.0


# R7 global convergence: number of consecutive no-gain macro-cycles after which the run is considered converged (stop
# looping → CLOSE).
DEFAULT_GLOBAL_CONVERGENCE_NO_GAIN_CYCLES: int = 3

# Decaying acceptance curve: the marginal-gain bar shrinks each macro-cycle.
KEEP_THRESHOLD_FLOOR_PCT: float = 0.1
KEEP_THRESHOLD_SPAN_PCT: float = 0.9
# Multi-node baseline noise floor is ~2x single-node; scale the curve to match.
MULTI_NODE_KEEP_THRESHOLD_FACTOR: float = 2.0


def resolve_keep_threshold(state: Any) -> float:
    """Current-cycle KEEP threshold for every path that injects ``keep_threshold_pct``."""
    from ..actions.executors._multi_node_env import is_multi_node

    cycle = state.macro_cycle
    return decaying_keep_threshold_pct(cycle, multi_node=is_multi_node())


def decaying_keep_threshold_pct(macro_cycle: int, *, multi_node: bool = False) -> float:
    """KEEP / convergence gain threshold for cycle N = ``macro_cycle`` + 1."""
    n = max(1, int(macro_cycle) + 1)
    base = KEEP_THRESHOLD_FLOOR_PCT + KEEP_THRESHOLD_SPAN_PCT / n
    return base * MULTI_NODE_KEEP_THRESHOLD_FACTOR if multi_node else base


# Long-run budget threshold.
DEFAULT_LONGRUN_THRESHOLD_MINUTES: float = 24 * 60


def is_long_run(state: Any) -> bool:
    """True when the session budget should use long-run budget accounting."""
    mm = _max_minutes(state)
    if mm <= 0:
        return True
    return mm >= float(DEFAULT_LONGRUN_THRESHOLD_MINUTES)


def _cumulative_gain_validated(state: Any) -> float:
    """Return ``state.cumulative_gain_validated``, defensively coerced to float."""
    try:
        return float(state.cumulative_gain_validated or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _cycle_reloop_min_remaining_sec(state: Any) -> float:
    """Session-scaled floor on the seconds that must remain to justify a new cycle.

    The session-scaled share keeps a short run from being blocked by a threshold
    it can never satisfy, but that share can fall below the cost of the cheapest
    unit of work in a cycle. The floor is therefore raised back to one granted
    variant round, so a cycle is never opened with budget it cannot spend. That
    raise is itself capped at :data:`_CYCLE_RELOOP_MAX_BUDGET_SHARE` of the
    session so a run too short to fund a round does not read as exhausted from
    its first tick.

    Args:
        state (Any): Frozen SharedState view exposing ``max_minutes``.

    Returns:
        float: The effective floor in seconds.
    """
    from hyperloom.orchestrator.actions.executors._subprocess_kill import resolve_benchmark_timeouts

    effective = _default_cycle_reloop_min_remaining_sec()
    max_minutes = _max_minutes(state)
    if max_minutes > 0:
        budget_sec = max_minutes * 60.0
        effective = min(effective, budget_sec * _CYCLE_RELOOP_BUDGET_RATIO)
        grant = min(resolve_benchmark_timeouts()[1], budget_sec * _CYCLE_RELOOP_MAX_BUDGET_SHARE)
        effective = max(effective, grant)
    return effective


def _kernel_idle_max_ticks() -> int:
    """Consecutive no-work KERNEL_AGENT ticks before winding down to SWEEP."""
    val = _env_int("INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS", default=3)
    return val if val >= 1 else 3


def _kernel_idle_min_seconds() -> float:
    """Wall-clock seconds a KERNEL idle streak must last before winding down."""
    val = _env_float("INFERENCE_OPTIMIZER_KERNEL_IDLE_MIN_SECONDS", default=600.0)
    return val if val > 0.0 else 600.0


#: How often the intent router refreshes the inline-step liveness stamp.
KERNEL_HEARTBEAT_SEC: float = 150.0

#: How stale ``kernel_inline_step_seen_unix`` may be and still mean "running".
#: Three heartbeat intervals absorb a late beat under load; a stamp orphaned by a
#: process that died mid-step expires shortly after rather than muting the guard.
KERNEL_INLINE_STEP_STALE_SECONDS: float = 3.0 * KERNEL_HEARTBEAT_SEC


def kernel_inline_step_running(state: Any, *, now_unix: float | None = None) -> bool:
    """Report whether an inline kernel request is executing right now."""
    seen = state.kernel_inline_step_seen_unix
    try:
        seen = float(seen or 0.0)
    except (TypeError, ValueError):
        return False
    if seen <= 0.0:
        return False
    now = float(now_unix if now_unix is not None else _now_unix(state))
    return 0.0 <= (now - seen) <= KERNEL_INLINE_STEP_STALE_SECONDS


def normalize_budget_pct(
    budget_pct: dict[str, float] | None,
) -> dict[str, float]:
    """Return a sanitized ``phase -> pct`` mapping (budgets are upper bounds, not renormalized to 1.0)."""
    out = dict(DEFAULT_PHASE_BUDGET_PCT)
    if not budget_pct:
        return out
    for phase, val in budget_pct.items():
        canon = (phase or "").strip().upper()
        if canon not in PHASE_NAMES:
            # An unknown key silently reverts that share to its default, which reads downstream as a choice nobody
            # made.
            log.warning(
                "phase budget: dropping override for unknown phase %r (known: %s)",
                phase,
                ", ".join(PHASE_NAMES),
            )
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            log.warning("phase budget: dropping non-numeric override %r=%r", canon, val)
            continue
        if not (0.0 <= f <= 1.0):
            log.warning("phase budget: dropping out-of-range override %s=%r (want 0.0-1.0)", canon, f)
            continue
        out[canon] = f
    return out


def redistribute_budget_pct(
    base: dict[str, float],
    *,
    optimize_enabled: bool = True,
    kernel_enabled: bool = True,
    sweep_enabled: bool = True,
) -> dict[str, float]:
    """Move disabled work-phase shares to enabled work phases.

    FRAMEWORK_AGENT, KERNEL_AGENT, and SWEEP absorb proportionally, capped at
    1.0; PRELUDE, ENABLEMENT, and CLOSE never absorb. A SWEEP whose concurrency
    sweep is off only drains and validates on entry, which the session budget
    alone gates, so a share there would just shrink the phases before it.
    """
    out = dict(base)
    toggles = {PHASE_FRAMEWORK_AGENT: optimize_enabled, PHASE_KERNEL_AGENT: kernel_enabled, PHASE_SWEEP: sweep_enabled}
    disabled = [phase for phase, enabled in toggles.items() if not enabled]
    freed = sum(float(out.get(p, 0.0)) for p in disabled)
    for p in disabled:
        out[p] = 0.0
    if freed <= 0.0:
        return out
    absorbers = [p for p in (PHASE_FRAMEWORK_AGENT, PHASE_KERNEL_AGENT, PHASE_SWEEP) if p not in disabled]
    weight = sum(float(out.get(p, 0.0)) for p in absorbers)
    if weight > 0.0:
        for p in absorbers:
            out[p] = float(out.get(p, 0.0)) + freed * float(out.get(p, 0.0)) / weight
    elif sweep_enabled:
        # No weighted absorber left → park the freed share on SWEEP.
        out[PHASE_SWEEP] = float(out.get(PHASE_SWEEP, 0.0)) + freed
    # Own our output: a share above a full wall clock is unspendable, and
    # leaving it in place makes the downstream re-normalize drop it back to the
    # phase default (i.e. *less* budget than asked for). Discard the excess.
    for p in absorbers:
        if float(out.get(p, 0.0)) > 1.0:
            log.warning(
                "phase budget: capping %s at 1.0 (redistribution reached %.4f); "
                "lower its --*-pct override to reclaim the excess elsewhere",
                p,
                float(out[p]),
            )
            out[p] = 1.0
    return out


# Pure judgment helpers (used by Coordinator at each tick end)
def _now_unix(state: Any) -> float:
    """Resolve the \"now\" timestamp; tests can inject ``state._now_unix``."""
    if hasattr(state, "_now_unix") and callable(state._now_unix):
        return float(state._now_unix())  # type: ignore[attr-defined]
    import time as _time

    return _time.time()


def _phase_started_unix(state: Any) -> float:
    """Return the Unix timestamp the current phase started, defensively coerced."""
    raw = state.phase_started_unix
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _resume_boundary_unix(state: Any) -> float:
    """Return when the current run leg began, i.e. the most recent ``--resume``."""
    return max(0.0, to_unix(state.resumed_ts, 0.0) or 0.0)


def _kernel_idle_since_unix(state: Any) -> float:
    """Return when the current KERNEL idle streak opened, defensively coerced."""
    raw = state.kernel_idle_since_unix
    try:
        return max(0.0, float(raw or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _pending_escalate_hint(state: Any) -> str:
    """Return a pending escalate hint to act on this tick (unknown hints → empty)."""
    raw = str(state.pending_escalate_hint or "").strip()
    if not raw:
        return ""
    if is_valid_escalate_hint(raw):
        return raw
    return ""


def _max_minutes(state: Any) -> float:
    """Return the session's configured ``max_minutes`` budget, defensively coerced."""
    try:
        return float(state.max_minutes or 0)
    except (TypeError, ValueError):
        return 0.0


def _budget_minutes(state: Any) -> float:
    """Wall-clock minutes the PER-PHASE budget fractions apply to (R2)."""
    try:
        cm = float(state.cycle_minutes or 0)
    except (TypeError, ValueError):
        cm = 0.0
    if cm > 0 and is_long_run(state):
        return cm
    return _max_minutes(state)


def phase_elapsed_seconds(state: Any, *, now_unix: float | None = None) -> float:
    """Return wall-clock seconds spent in the current phase."""
    started = _phase_started_unix(state)
    if started <= 0:
        return 0.0
    started = max(started, _resume_boundary_unix(state))
    now = float(now_unix if now_unix is not None else _now_unix(state))
    return max(0.0, now - started)


def phase_cumulative_seconds(
    state: Any,
    *,
    phase: str | None = None,
    now_unix: float | None = None,
) -> float:
    """Return wall-clock seconds spent in ``phase``, summed over EVERY entry."""
    current = (state.phase or "").strip().upper()
    target = (phase or current or "").strip().upper()
    if not target:
        return 0.0
    accumulated = 0.0
    totals = state.phase_elapsed_totals
    if isinstance(totals, dict):
        try:
            accumulated = max(0.0, float(totals.get(target, 0.0) or 0.0))
        except (TypeError, ValueError):
            # A malformed banked total degrades to "nothing banked", i.e. the pre-fix per-entry behaviour for this
            # phase.
            accumulated = 0.0
    if target == current:
        accumulated += phase_elapsed_seconds(state, now_unix=now_unix)
    return accumulated


def _phase_budget_total_seconds(
    state: Any,
    *,
    now_unix: float | None = None,
) -> float | None:
    """Effective TOTAL budget (seconds) allotted to the current phase."""
    budget = state.phase_budget_pct
    phase = (state.phase or "").strip().upper()
    if phase not in budget:
        return None
    pct = float(budget[phase])
    if pct <= 0.0:
        # Zero fraction = no time. ``None`` would read as "unbounded" to callers.
        return 0.0

    session_remaining = session_remaining_seconds(state, now_unix=now_unix)
    if session_remaining is not None:
        # Charge-back. remaining_at_entry reconstructs the time left when the phase's live segment opened: within one
        # run leg session_remaining shrinks exactly as phase_elapsed grows, so their sum holds.
        remaining_at_entry = max(0.0, session_remaining + phase_elapsed_seconds(state, now_unix=now_unix))
        if is_long_run(state):
            # Long bounded run: the per-cycle window caps the base as a planning ceiling so one cycle never plans
            # beyond one macro-cycle window.
            cycle_window = _budget_minutes(state) * 60.0
            if cycle_window > 0.0:
                remaining_at_entry = min(cycle_window, remaining_at_entry)
        # Normalize ONLY over the current phase and the phases still to come: already-elapsed phases (notably PRELUDE)
        # are excluded — their spend is already reflected in the base — while CLOSE stays in so it keeps its reserved
        # share.
        denom = sum(
            float(budget.get(p, 0.0)) for p in PHASE_NAMES[phase_index(phase) :] if float(budget.get(p, 0.0)) > 0.0
        )
        if denom <= 0.0:
            return None
        return remaining_at_entry * pct / denom
    # No session clock (unbounded run, or ``start_ts`` unset): fall back to the flat per-window allotment —
    # charge-back needs a wall-clock reference.
    mm = _budget_minutes(state)
    if mm <= 0:
        return None
    return mm * 60.0 * pct


def phase_budget_remaining_seconds(
    state: Any,
    *,
    now_unix: float | None = None,
) -> float | None:
    """Return seconds left in the current phase ENTRY's budget (``None`` when budget window 0 = unlimited)."""
    total = _phase_budget_total_seconds(state, now_unix=now_unix)
    if total is None:
        return None
    return max(0.0, total - phase_elapsed_seconds(state, now_unix=now_unix))


def session_remaining_seconds(
    state: Any,
    *,
    now_unix: float | None = None,
) -> float | None:
    """Total wall-clock seconds remaining for the session (``None`` when unbounded).

    Derived from the same forward-summed elapsed total the Coordinator loop and
    admission read, so the three cannot disagree about what a multi-leg session
    has already spent. An unarmed leg anchor means no leg is charging through
    this view; the charged total answers for a state reloaded between legs, and
    wall time since ``start_ts`` for one that never charged.

    Args:
        state (Any): Frozen SharedState view exposing ``max_minutes``,
            ``elapsed_charged_sec``, ``leg_anchor_unix`` and ``start_ts``.
        now_unix (float | None): Override for the current time, kept in the same
            time source as ``phase_elapsed_seconds(now_unix=...)``.

    Returns:
        float | None: Non-negative seconds left in the session, ``None`` when
        unbounded (``max_minutes`` is 0), and ``None`` when nothing on the state
        dates the session -- no charge, no anchor, no parseable ``start_ts``.
    """
    mm = _max_minutes(state)
    if mm <= 0:
        return None
    try:
        charged = max(0.0, float(state.elapsed_charged_sec or 0.0))
        anchor = float(state.leg_anchor_unix or 0.0)
    except (TypeError, ValueError):
        return None
    now = float(now_unix) if now_unix is not None else time.time()
    if anchor > 0.0:
        return max(0.0, mm * 60.0 - (charged + max(0.0, now - anchor)))
    if charged > 0.0:
        return max(0.0, mm * 60.0 - charged)
    started = to_unix(str(state.start_ts or "").strip())
    if started is None:
        return None
    return max(0.0, mm * 60.0 - max(0.0, now - started))


def phase_status_summary(
    state: Any,
    *,
    now_unix: float | None = None,
) -> str:
    """Render the per-tick ``=== Phase ===`` block (≤7 lines). The mid-chain phases add a ``cycle_reloop`` line showing whether another macro-cycle is still affordable, and a phase held for its initial roofline a ``hold`` line."""
    phase = (state.phase or "").strip().upper() or "UNSET"
    elapsed = int(phase_elapsed_seconds(state, now_unix=now_unix))
    # ``remaining`` paces this entry; ``cumulative`` totals every entry.
    cumulative = int(phase_cumulative_seconds(state, now_unix=now_unix))
    budget_pct_for_phase = state.phase_budget_pct.get(phase, 0.0)
    remaining = phase_budget_remaining_seconds(state, now_unix=now_unix)
    budget_line: str
    if remaining is None:
        budget_line = f"budget    : pct={budget_pct_for_phase:.2f} (unlimited run; no per-phase cap)"
    else:
        budget_line = (
            f"budget    : pct={budget_pct_for_phase:.2f} elapsed_sec={elapsed} "
            f"cumulative_sec={cumulative} remaining_sec={int(remaining)}"
        )
    entered = time.strftime("%FT%TZ", time.gmtime(t0)) if (t0 := _phase_started_unix(state)) > 0 else "(unset)"
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
    if phase in (PHASE_ENABLEMENT, PHASE_FRAMEWORK_AGENT, PHASE_KERNEL_AGENT, PHASE_SWEEP):
        feasible, evidence = cycle_reloop_decision(state, now_unix=now_unix)
        reloop_line = f"reloop    : cycle_reloop_feasible={'true' if feasible else 'false'}"
        threshold = evidence.get("min_remaining_sec_effective")
        if threshold is not None:
            reloop_line += f" threshold_sec={int(threshold)}"
        session_remaining = session_remaining_seconds(state, now_unix=now_unix)
        if session_remaining is not None:
            reloop_line += f" session_remaining_sec={int(session_remaining)}"
        blocked = evidence.get("reloop_blocked")
        if blocked:
            reloop_line += f" blocked={blocked}"
        if phase != PHASE_SWEEP:
            reloop_line += " (projected)"
        lines.append(reloop_line)
    if hold_line := _initial_analysis_hold_line(state, phase):
        lines.append(hold_line)
    return "\n".join(lines)


# plateau pure functions
def _row_cycle(row: dict[str, Any]) -> int:
    """Return a row cycle, treating legacy unstamped rows as cycle zero."""
    try:
        return int(row.get("cycle", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _rows_for_current_cycle(rows: Any, state: Any) -> list[dict[str, Any]]:
    """Filter durable ledger rows to the current macro-cycle."""
    if not isinstance(rows, list):
        return []
    dict_rows = [row for row in rows if isinstance(row, dict)]
    if not any("cycle" in row for row in dict_rows):
        return dict_rows
    cycle = state.macro_cycle
    return [row for row in dict_rows if _row_cycle(row) == cycle]


# per-phase judgments
def warm_replay_in_flight(state: Any) -> bool:
    """True while the PRELUDE warm-recipe replay task has not finished (PRELUDE must not exit until False — GPU contention)."""
    outcome = state.warm_replay_outcome or {}
    if not isinstance(outcome, dict):
        return False
    return str(outcome.get("status") or "").strip() == "in_flight"


# The statuses a finished GEAK run writes to ``geak_result.status``.
GEAK_TERMINAL_STATUSES = frozenset(
    {
        "ok",
        "no_gain",
        "error",
        "timeout",
        "failed",
        "skipped",
        "baseline_reproduction_failed",
    }
)


def _geak_phase_terminal(state: Any) -> bool:
    """Return true once the GEAK-owned KERNEL phase has produced a terminal result."""
    if str(state.kernel_optimizer or "").strip().lower() != "geak":
        return False
    result = state.geak_result or {}
    if not isinstance(result, dict):
        return False
    return str(result.get("status") or "").strip().lower() in GEAK_TERMINAL_STATUSES


#: Every status the rewrite controller can end on. All of them are terminal for
#: the phase: the controller is not re-run inside one macro cycle, so a failure
#: is as final as a published patch.
CONTROLLER_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "completed",
        "failed",
        "no_opportunity",
        "no_result",
        "partial",
    }
)


def _controller_phase_terminal(state: Any) -> bool:
    """Return true when this macro cycle's rewrite controller has stopped."""
    if str(state.kernel_optimizer or "").strip().lower() != "forge":
        return False
    result = state.kernel_rewrite_controller_result or {}
    if not isinstance(result, dict):
        return False
    try:
        result_cycle = int(result.get("macro_cycle", -1))
        current_cycle = int(state.macro_cycle or 0)
    except (TypeError, ValueError):
        return False
    status = str(result.get("status") or "").strip().lower()
    return result_cycle == current_cycle and status in CONTROLLER_TERMINAL_STATUSES


def compute_kernel_progress_fingerprint(
    state: Any,
    *,
    inflight_task_ids: Any = (),
) -> str:
    """Digest the KERNEL signals that change if and only if something moved."""
    import hashlib
    import json

    stack = state.optimization_stack
    pending = state.pending_kernel_integrations
    controller = state.kernel_rewrite_controller_result
    controller = controller if isinstance(controller, dict) else {}
    payload = {
        "inflight": sorted(str(task_id) for task_id in (inflight_task_ids or ())),
        "integrate_verdicts": sorted(
            f"{key}:{entry['attempt_count']}" for key, entry in state.kernel_integrate_attempts.items()
        ),
        "pending_integrations": sorted(str(key) for key in pending) if isinstance(pending, dict) else [],
        "rejected": sorted(str(kid) for kid in (state.rejected_kernel_ids or [])),
        "stack_len": len(stack) if isinstance(stack, list) else 0,
        "rewrite_controller": [
            str(controller.get("macro_cycle", "")),
            str(controller.get("status", "")),
            str(controller.get("patch_count", "")),
            str(controller.get("finished_at", "")),
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def kernel_work_pending(state: Any) -> bool:
    """Return True while KERNEL has work that can still affect validated gain."""
    if kernel_decisions.has_keep_pending_integrate(state):
        return True
    if _controller_phase_terminal(state) or _geak_phase_terminal(state):
        return False

    return bool(kernel_decisions.untried_hot_reusable_kernels(state))


def measured_seconds(state: Any, field: str) -> float | None:
    """Read a duration an earlier round measured, or ``None`` when none did."""
    try:
        value = float(getattr(state, field, 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def boot_cost_sec(state: Any) -> float | None:
    """What bringing this workload's server up costs, or ``None`` when unmeasured."""
    total_sec = measured_seconds(state, "baseline_runtime_sec")
    post_ready_sec = measured_seconds(state, "baseline_post_ready_runtime_sec")
    if total_sec is None or post_ready_sec is None:
        return None
    return max(0.0, total_sec - post_ready_sec)


def benchmark_cost_sec(state: Any) -> float | None:
    """What one benchmark pass costs on a server already up, or ``None``."""
    hot_sec = measured_seconds(state, "baseline_warm_runtime_sec")
    if hot_sec is not None:
        return hot_sec
    return measured_seconds(state, "baseline_post_ready_runtime_sec")


def baseline_round_cost_sec(state: Any, *, double_run: bool) -> float | None:
    """What a baseline round costs, or ``None`` when unmeasured."""
    first_pass_sec = measured_seconds(state, "baseline_runtime_sec")
    if first_pass_sec is None or not double_run:
        return first_pass_sec
    second_pass_sec = benchmark_cost_sec(state)
    return first_pass_sec if second_pass_sec is None else first_pass_sec + second_pass_sec


def one_more_measurement_sec(state: Any) -> float | None:
    """What the next measured variant will cost, or ``None`` when unmeasured."""
    boot_sec = boot_cost_sec(state)
    benchmark_sec = benchmark_cost_sec(state)
    if boot_sec is None or benchmark_sec is None:
        return None
    return boot_sec + benchmark_sec


def append_phase_evidence_row(history: Any, *, key: str, row: dict[str, Any]) -> bool:
    """Append ``row`` to the current phase's ``evidence[key]`` list."""
    if not isinstance(history, list) or not history:
        return False
    current = history[-1]
    if not isinstance(current, dict):
        return False
    evidence = current.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
        current["evidence"] = evidence
    rows = evidence.get(key)
    if not isinstance(rows, list):
        rows = []
        evidence[key] = rows
    rows.append(row)
    return True


def session_usable_seconds(state: Any) -> float | None:
    """Seconds a unit of work may still claim, from the session's own accounting."""
    return state.session_budget_usable_sec()


def prelude_affordable_seconds(state: Any) -> tuple[float | None, dict[str, Any]]:
    """Seconds PRELUDE may still spend, and the numbers the figure is built from."""
    max_sec = _max_minutes(state) * 60.0
    usable = session_usable_seconds(state)
    if max_sec <= 0.0 or usable is None:
        return None, {"reason": "unbounded_budget"}
    reserve_sec = max_sec * OPTIMIZATION_RESERVE_PCT
    affordable_sec = usable - reserve_sec
    return affordable_sec, {
        "optimization_reserve_sec": round(reserve_sec, 1),
        "session_usable_sec": round(usable, 1),
        "affordable_sec": round(affordable_sec, 1),
        "bound": "optimization_reserve",
    }


def prelude_can_afford(
    state: Any,
    *,
    expected_cost_sec: float,
) -> tuple[bool, dict[str, Any]]:
    """Decide whether PRELUDE can still buy an optional arm costing ``expected_cost_sec``."""
    cost = max(0.0, float(expected_cost_sec or 0.0))
    affordable_sec, evidence = prelude_affordable_seconds(state)
    priced = {"expected_cost_sec": round(cost, 1), **evidence}
    if affordable_sec is None:
        return True, priced
    return affordable_sec >= cost, priced


def _resolve_plateau_overrides(state: Any) -> dict[str, Any]:
    """Pull operator-tuned plateau thresholds off :attr:`SharedState.plateau_overrides` (empty → library defaults)."""
    overrides = state.plateau_overrides or {}
    return dict(overrides) if isinstance(overrides, dict) else {}


def _lever_attempts(state: Any, *levers: str) -> list[dict[str, Any]]:
    """This cycle's attempts on the given levers, in the order they were recorded."""
    rows = _rows_for_current_cycle(state.attempts or [], state)
    return [r for r in rows if str(r.get("lever_kind") or "") in levers]


def _trailing_no_keep(attempts: list[dict[str, Any]]) -> int:
    """Count trailing attempts that did not adopt.

    Only resolved attempts reach the ledger: a candidate its lane will retry
    leaves nothing here to plateau on.
    """
    streak = 0
    for row in reversed(attempts):
        if row.get("adopted"):
            break
        streak += 1
    return streak


def _fold_config_rounds(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-variant config attempts into one row per benched round.

    A ``run_grid`` call benches a whole grid against one anchor, so the round is
    the unit the config lever succeeds or fails at: it adopted if any variant in
    it did, and its gain is what those variants banked. Counting variants instead
    would let a single grid of eight cross a five-deep streak floor, and would
    read a round as dry whenever its KEEP happened not to be the last variant.

    A row with no ``round_id`` is its own round — the local-exploration arm
    delivers server args one attempt at a time, outside any grid.
    """
    rounds: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for row in attempts:
        round_id = str(row.get("round_id") or "")
        pos = index.get(round_id) if round_id else None
        if pos is None:
            pos = len(rounds)
            rounds.append({"round_id": round_id, "adopted": False, "gain_pct": 0.0})
            if round_id:
                index[round_id] = pos
        if row.get("adopted"):
            rounds[pos]["adopted"] = True
            rounds[pos]["gain_pct"] = float(rounds[pos]["gain_pct"]) + float(row.get("gain_pct") or 0.0)
    return rounds


def _config_lever_dry(state: Any, overrides: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Whether the config lever has stopped paying.

    Judged per benched round, not per variant. Two conditions, both required:
    the recent adopted gain is below the floor, and a run of rounds has produced
    nothing. A grid that is still landing small wins has not plateaued.
    """
    lookback = int(overrides.get("explore_lookback", DEFAULT_PLATEAU_EXPLORE_LOOKBACK))
    gain_floor = float(overrides.get("explore_keep_gain_pct", DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT))
    streak_floor = int(overrides.get("explore_empty_streak", DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK))

    rounds = _fold_config_rounds(_lever_attempts(state, LEVER_CONFIG))
    recent_gain = sum(float(r.get("gain_pct") or 0.0) for r in rounds[-lookback:] if r.get("adopted"))
    streak = _trailing_no_keep(rounds)
    return (recent_gain < gain_floor and streak >= streak_floor), {
        "recent_keep_gain_pct": round(recent_gain, 4),
        "keep_gain_threshold_pct": gain_floor,
        "empty_streak": streak,
        "empty_streak_threshold": streak_floor,
        "lookback": lookback,
    }


def _patch_lever_dry(state: Any, overrides: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Whether the patch levers have stopped paying.

    Source patches and upstream PRs share a supply — the discovery and authoring
    ladder — so they are judged together: a trailing run of resolved candidates
    without an adoption, or a pump that has declared the supply exhausted. An
    authoring round that came back empty or never ran resolves its candidate
    too, so a lane that cannot run dries up like one that finds nothing.
    """
    streak_floor = int(
        overrides.get("framework_no_keep_streak", DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK),
    )
    streak = _trailing_no_keep(_lever_attempts(state, LEVER_SOURCE_PATCH, LEVER_UPSTREAM_PR))
    exhausted = bool(state.framework_agent_phase_done)
    return (streak >= streak_floor or exhausted), {
        "source_consecutive_no_keep": streak,
        "source_threshold": streak_floor,
        "source_candidates_exhausted": exhausted,
    }


def per_lever_dryness(state: Any) -> tuple[bool, dict[str, Any]]:
    """Whether every lever has run dry, over the unified attempts ledger.

    One lever going quiet raises ``switch_bottleneck`` so the next macro-cycle
    steers elsewhere; the phase advances only once none of them is paying.
    """
    overrides = _resolve_plateau_overrides(state)
    config_dry, config_ev = _config_lever_dry(state, overrides)
    patch_dry, patch_ev = _patch_lever_dry(state, overrides)
    return (config_dry and patch_dry), {
        **config_ev,
        **patch_ev,
        "config_arm_plateaued": config_dry,
        "source_arm_plateaued": patch_dry,
        "switch_bottleneck": bool(config_dry or patch_dry),
    }


_STOP_TIME_EXHAUSTED = "time_exhausted"

#: SWEEP's close reason per ``reloop_blocked`` value.
_RELOOP_BLOCK_TERMINALS: dict[str, str] = {
    "global_converged": "global_converged",
    "all_directions_saturated": "global_converged",
    "target_reached": "target_reached",
    "insufficient_remaining": _STOP_TIME_EXHAUSTED,
}


def _post_prelude_target(*, optimize_enabled: bool, kernel_enabled: bool) -> str:
    """First active work phase after PRELUDE or ENABLEMENT: FRAMEWORK_AGENT, else KERNEL_AGENT, else SWEEP."""
    if optimize_enabled:
        return PHASE_FRAMEWORK_AGENT
    if kernel_enabled:
        return PHASE_KERNEL_AGENT
    return PHASE_SWEEP


def _budget_predicate_inputs(
    state: Any,
    *,
    now_unix: float,
) -> dict[str, Any]:
    """Normalize the clocks compared by phase budget predicates."""
    remaining_sec = phase_budget_remaining_seconds(state, now_unix=now_unix)
    budget: dict[str, Any] = {
        "remaining_sec": remaining_sec,
        "entry_elapsed_sec": phase_elapsed_seconds(state, now_unix=now_unix),
        "cumulative_elapsed_sec": phase_cumulative_seconds(state, now_unix=now_unix),
        "current_balance": remaining_sec,
    }
    return budget


@dataclass(frozen=True)
class PhaseFlags:
    """Which optional phases a run has switched on."""

    kernel_enabled: bool = True
    optimize_enabled: bool = True
    enablement_enabled: bool = False


def _base_workflow_predicate_inputs(
    state: Any,
    *,
    current: str,
    now_unix: float,
    flags: PhaseFlags,
    enablement_in_flight: bool,
) -> dict[str, Any]:
    hint = _pending_escalate_hint(state) or None
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
        "hint": hint,
        "sweep_result": None,
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
        "warm_replay_status": warm_status,
        "measure_round_dropped": bool(state.baseline_measure_round_dropped),
        "tput": to_float(state.baseline_tput) or 0.0,
        **_initial_analysis_inputs(state),
        "session_usable_sec": session_usable_seconds(state),
        "phase_spent_sec": phase_cumulative_seconds(state, phase=PHASE_PRELUDE, now_unix=now_unix),
        "runtime_sec": optional_positive_float(state.baseline_runtime_sec),
        "post_ready_runtime_sec": optional_positive_float(state.baseline_post_ready_runtime_sec),
        "warm_runtime_sec": optional_positive_float(state.baseline_warm_runtime_sec),
    }


def _rounds_since_phase_entry(state: Any) -> list[dict[str, Any]]:
    """This cycle's specialist rounds booked since the current phase was entered.

    PRELUDE shares macro-cycle 0 with FRAMEWORK_AGENT, and ``source_phase`` names a round's ownership lane rather
    than the phase it ran in, so only the entry time separates this phase's rounds from earlier ones.
    """
    rows = _rows_for_current_cycle(state.specialist_rounds or [], state)
    entered = _phase_started_unix(state)
    # No entry time: keep every row of the cycle. With one: a row with no parseable completed_at is not counted.
    if entered <= 0.0:
        return rows
    return [row for row in rows if (to_unix(row.get("completed_at")) or 0.0) >= entered]


def _framework_predicate_inputs(
    state: Any,
    *,
    now_unix: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _all_dry, evidence = per_lever_dryness(state)
    plateau = {
        key: value
        for key, value in evidence.items()
        if key not in {"config_arm_plateaued", "source_arm_plateaued", "switch_bottleneck"}
    }
    plateau["attempt_count"] = len(_rows_for_current_cycle(state.attempts or [], state))
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
        "kernel_work_pending": bool(kernel_work_pending(state)),
        "kernel_agent_in_flight": bool(kernel_work_in_flight),
        "optimizer": str(state.kernel_optimizer or "").strip().lower(),
        "controller_cycle": int(to_float(controller.get("macro_cycle"), -1)),
        "controller_status": str(controller.get("status") or "").strip().lower(),
        "controller_patch_count": int(controller.get("patch_count") or 0),
        "controller_task_count": int(controller.get("task_count") or 0),
        "controller_reason": str(controller.get("reason") or ""),
    }
    plateau = {
        "idle_ticks": int(state.kernel_idle_ticks or 0),
        "idle_since_unix": _kernel_idle_since_unix(state),
        "idle_max_ticks": _kernel_idle_max_ticks(),
        "idle_min_seconds": _kernel_idle_min_seconds(),
        "rejected_kernel_count": len(state.rejected_kernel_ids or []),
    }
    return pending, plateau, _budget_predicate_inputs(state, now_unix=now_unix)


def _sweep_predicate_inputs(
    state: Any,
    *,
    now_unix: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_conc = state.last_conc_sweep or {}
    if not isinstance(last_conc, dict):
        last_conc = {}
    summary = last_conc.get("summary") if isinstance(last_conc.get("summary"), dict) else {}
    cycle = int(state.macro_cycle or 0)
    saturated = state.saturated_directions or {}
    saturated_facts = (
        {str(key): bool(value.get("saturated")) for key, value in saturated.items() if isinstance(value, dict)}
        if isinstance(saturated, dict)
        else {}
    )
    result = {
        "status": str(last_conc.get("status") or "").lower(),
        "was_skipped": bool(last_conc.get("was_skipped")),
        "budget_exhausted": bool(last_conc.get("budget_exhausted")),
        "skip_reason": str(last_conc.get("skip_reason") or ""),
        "successful_pairs": int(summary.get("successful_pairs") or 0),
        "summary_present": bool(summary),
        "reloop": {
            "current_gain_pct": _cumulative_gain_validated(state),
            "gain_at_cycle_start_pct": to_float(state.gain_at_cycle_start) or 0.0,
            "min_gain_pct": decaying_keep_threshold_pct(cycle),
            "prior_no_gain_cycle_streak": int(state.no_gain_cycle_streak or 0),
            "no_gain_cycles": DEFAULT_GLOBAL_CONVERGENCE_NO_GAIN_CYCLES,
            "saturated_directions": saturated_facts,
            "session_remaining_sec": session_remaining_seconds(state, now_unix=now_unix),
            "min_remaining_sec": _cycle_reloop_min_remaining_sec(state),
        },
    }
    budget = _budget_predicate_inputs(state, now_unix=now_unix)
    if not result["status"]:
        _extend_budget_to_sweep_grant(budget, state, now_unix=now_unix)
    return result, budget


def _extend_budget_to_sweep_grant(budget: dict[str, Any], state: Any, *, now_unix: float) -> None:
    """Widen SWEEP's budget to cover the conc_sweep it granted and is still waiting on.

    SWEEP's only work is the sweep it enqueued on entry, whose budget is already clamped to the session clock. The
    per-entry budget is a share of the remaining session (5% of three hours is nine minutes) and would otherwise cancel
    that sweep mid-ladder. The widened numbers are what the predicate compares, so a replay of the frozen inputs reaches
    the same decision.
    """
    grant_left = (to_float(state.conc_sweep_granted_until_unix) or 0.0) - float(now_unix)
    if grant_left <= 0.0:
        return
    remaining = to_float(budget.get("remaining_sec"))
    if remaining is not None and remaining < grant_left:
        budget["remaining_sec"] = grant_left
        if "current_balance" in budget:
            budget["current_balance"] = grant_left


def workflow_predicate_inputs(
    state: Any,
    *,
    flags: PhaseFlags = PhaseFlags(),
    now_unix: float | None = None,
    enablement_in_flight: bool = False,
    kernel_work_in_flight: bool = False,
) -> dict[str, Any]:
    """Freeze primitive, normalized facts consumed by one phase decision."""
    current = (state.phase or "").strip().upper() or PHASE_PRELUDE
    frozen_now = float(now_unix if now_unix is not None else _now_unix(state))
    inputs = _base_workflow_predicate_inputs(
        state,
        current=current,
        now_unix=frozen_now,
        flags=flags,
        enablement_in_flight=enablement_in_flight,
    )
    if current == PHASE_PRELUDE:
        inputs["baseline"] = _prelude_predicate_inputs(state, now_unix=frozen_now)
    elif current == PHASE_ENABLEMENT:
        enablement = state.enablement
        inputs["baseline"] = {"tput": to_float(state.baseline_tput) or 0.0, **_initial_analysis_inputs(state)}
        inputs["pending_work"] = {
            "validation_pending": bool(enablement.validation_pending),
            "enablement_in_flight": bool(enablement_in_flight),
        }
    elif current == PHASE_FRAMEWORK_AGENT:
        inputs["plateau"], inputs["budget"] = _framework_predicate_inputs(state, now_unix=frozen_now)
    elif current == PHASE_KERNEL_AGENT:
        inputs["pending_work"], inputs["plateau"], inputs["budget"] = _kernel_predicate_inputs(
            state,
            now_unix=frozen_now,
            kernel_work_in_flight=kernel_work_in_flight,
        )
    elif current == PHASE_SWEEP:
        inputs["sweep_result"], inputs["budget"] = _sweep_predicate_inputs(state, now_unix=frozen_now)
    return inputs


def initial_workflow_predicate_inputs(state: Any, *, current_phase: str, flags: PhaseFlags) -> dict[str, Any]:
    """Identity-transition facts for fresh start or explicit CLOSE resume."""
    return _base_workflow_predicate_inputs(
        state,
        current=current_phase,
        now_unix=float(_now_unix(state)),
        flags=flags,
        enablement_in_flight=False,
    )


def _transition_result(
    target: str,
    reason: str,
    evidence: dict[str, Any],
    predicate_inputs: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    return target, reason, {**evidence, "predicate_inputs": predicate_inputs}


def _prelude_viability(baseline: dict[str, Any]) -> dict[str, Any]:
    usable = to_float(baseline.get("session_usable_sec"))
    runtime = optional_positive_float(baseline.get("runtime_sec"))
    post_ready = optional_positive_float(baseline.get("post_ready_runtime_sec"))
    warm = optional_positive_float(baseline.get("warm_runtime_sec"))
    benchmark = warm or post_ready
    round_sec = None
    if runtime is not None and post_ready is not None and benchmark is not None:
        round_sec = max(0.0, runtime - post_ready) + benchmark
        priced_by = "boot_plus_benchmark"
    else:
        round_sec = runtime
        priced_by = "cold_round"
    if usable is None or round_sec is None:
        return {}
    return {
        "session_usable_sec": round(usable, 1),
        "measured_round_sec": round(round_sec, 1),
        "priced_by": priced_by,
        "affordable_rounds": round(usable / round_sec, 2),
        "fits_one_optimization_round": usable >= round_sec,
    }


def _budget_exit(
    budget: dict[str, Any],
    *,
    exhausted_reason: str,
    evidence: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    remaining = to_float(budget.get("remaining_sec"))
    elapsed = to_float(budget.get("entry_elapsed_sec")) or 0.0
    cumulative = to_float(budget.get("cumulative_elapsed_sec")) or 0.0
    timing = {
        **evidence,
        "entry_elapsed_seconds": elapsed,
        "cumulative_elapsed_seconds": cumulative,
    }
    if remaining is not None and remaining <= 0.0:
        return exhausted_reason, timing
    return None


class _FrameworkPlateau(NamedTuple):
    """Whether each FRAMEWORK_AGENT arm has stopped paying, and the arm evidence that says so."""

    config_dry: bool
    source_dry: bool
    arms: dict[str, Any]
    did_work: bool


def _framework_plateau(plateau: dict[str, Any]) -> _FrameworkPlateau:
    """Read the per-arm plateau verdicts off the FRAMEWORK_AGENT predicate inputs."""
    recent_gain = to_float(plateau.get("recent_keep_gain_pct")) or 0.0
    keep_threshold = to_float(plateau.get("keep_gain_threshold_pct")) or 0.0
    empty_streak = int(plateau.get("empty_streak") or 0)
    empty_threshold = int(plateau.get("empty_streak_threshold") or 0)
    source_streak = int(plateau.get("source_consecutive_no_keep") or 0)
    source_threshold = int(plateau.get("source_threshold") or 0)
    config_dry = recent_gain < keep_threshold and empty_streak >= empty_threshold
    source_dry = source_streak >= source_threshold or bool(plateau.get("source_candidates_exhausted"))
    return _FrameworkPlateau(
        config_dry=config_dry,
        source_dry=source_dry,
        arms={
            **{key: value for key, value in plateau.items() if key not in {"attempt_count", "specialist_round_count"}},
            "config_arm_plateaued": config_dry,
            "source_arm_plateaued": source_dry,
            "switch_bottleneck": bool(config_dry or source_dry),
        },
        did_work=int(plateau.get("attempt_count") or 0) > 0 or int(plateau.get("specialist_round_count") or 0) > 0,
    )


def _framework_exit(inputs: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    plateau = _framework_plateau(dict(inputs.get("plateau") or {}))
    arms = plateau.arms
    hint = str(inputs.get("hint") or "")
    if hint == ESCALATE_HINT_SKIP_TO_KERNEL and plateau.did_work:
        return "optimize_no_more_leverage", {**arms, "evidence": "llm_escalation", "hint": hint}
    if hint == ESCALATE_HINT_SKIP_TO_SWEEP:
        return "optimize_no_more_leverage", {**arms, "evidence": "skip_to_sweep", "hint": hint}
    if plateau.config_dry and plateau.source_dry:
        return "optimize_no_more_leverage", {**arms, "evidence": "both_arms_plateaued", "plateau": True}
    return _budget_exit(
        dict(inputs.get("budget") or {}),
        exhausted_reason="optimize_phase_budget_exhausted",
        evidence=arms,
    )


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
            return "kernel_no_more_leverage", {"evidence": "kernel_settled"}
        idle = _kernel_idle_exit(plateau, inputs.get("now_unix"))
        if idle is not None:
            return idle
    return _budget_exit(
        dict(inputs.get("budget") or {}),
        exhausted_reason="kernel_phase_budget_exhausted",
        evidence={"rejected_kernel_count": int(plateau.get("rejected_kernel_count") or 0)},
    )


def _sweep_exit(inputs: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    sweep = dict(inputs.get("sweep_result") or {})
    status = str(sweep.get("status") or "")
    if status == "failed":
        return "sweep_failed", {"sweep_status": status}
    if status in {"succeeded", "partial", "completed", "skipped"}:
        evidence: dict[str, Any] = {"sweep_status": status}
        if bool(sweep.get("was_skipped")):
            evidence.update(
                sweep_was_skipped=True,
                sweep_skip_budget_exhausted=bool(sweep.get("budget_exhausted")),
                sweep_skip_reason=str(sweep.get("skip_reason") or ""),
            )
            spent_without_pair = bool(sweep.get("budget_exhausted")) or (
                str(sweep.get("skip_reason") or "") == "budget_exhausted_no_successful_pairs"
            )
            no_pair = bool(sweep.get("summary_present")) and int(sweep.get("successful_pairs") or 0) <= 0
            if spent_without_pair or (no_pair and not evidence["sweep_skip_reason"]):
                return "sweep_failed", evidence
        return "sweep_done", evidence
    # Anything else -- nothing landed yet, or a sweep the orchestrator cancelled -- leaves the phase to its budget:
    # a cancelled sweep was stopped by a session-wide stop that already decided the run is over.
    return _budget_exit(
        dict(inputs.get("budget") or {}),
        exhausted_reason="sweep_budget_exhausted",
        evidence={"sweep_status": status} if status else {},
    )


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
    facts = dict(dict(inputs.get("sweep_result") or {}).get("reloop") or {})
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
    frozen_now = float(now_unix if now_unix is not None else _now_unix(state))
    inputs = workflow_predicate_inputs(state, now_unix=frozen_now)
    inputs["sweep_result"], _ = _sweep_predicate_inputs(state, now_unix=frozen_now)
    reloop, evidence = _reloop_decision(inputs)
    return reloop and state.framework_agent_phase_enabled, evidence


def compute_next_phase(
    state: Any,
    *,
    flags: PhaseFlags = PhaseFlags(),
    now_unix: float | None = None,
    enablement_in_flight: bool = False,
    kernel_work_in_flight: bool = False,
) -> tuple[str, str, dict[str, Any]] | None:
    """Return the next transition with the primitive inputs it consumed."""
    inputs = workflow_predicate_inputs(
        state,
        flags=flags,
        now_unix=now_unix,
        enablement_in_flight=enablement_in_flight,
        kernel_work_in_flight=kernel_work_in_flight,
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


def _initial_analysis_inputs(state: Any) -> dict[str, Any]:
    """The two facts :func:`_holds_for_initial_analysis` decides on, read once for the predicates and the block."""
    return {
        "initial_analysis_pending": bool((state.auto_roofline_pending_task_id or "").strip()),
        "analysis_affordable_sec": prelude_affordable_seconds(state)[0],
    }


def _initial_analysis_hold_line(state: Any, phase: str) -> str | None:
    """The ``hold`` line of a PRELUDE or ENABLEMENT kept open for its initial roofline, or ``None`` when none holds it."""
    if phase not in (PHASE_PRELUDE, PHASE_ENABLEMENT) or (to_float(state.baseline_tput) or 0.0) <= 0.0:
        return None
    facts = _initial_analysis_inputs(state)
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


def _cold_anchor_shortfall(baseline: dict[str, Any]) -> dict[str, Any] | None:
    """The evidence that retrying a dropped measure round no longer fits the session's budget."""
    if not bool(baseline.get("measure_round_dropped")):
        return None
    usable = to_float(baseline.get("session_usable_sec"))
    runtime = optional_positive_float(baseline.get("runtime_sec"))
    benchmark = optional_positive_float(baseline.get("warm_runtime_sec")) or optional_positive_float(
        baseline.get("post_ready_runtime_sec")
    )
    retry = runtime
    if runtime is not None and benchmark is not None:
        retry += benchmark
    viability = _prelude_viability(baseline)
    use_sec = to_float(viability.get("measured_round_sec")) or runtime
    if usable is None or retry is None or use_sec is None or usable >= retry + use_sec:
        return None
    return {"terminal": True, "baseline_anchor": "cold", "retry_round_sec": round(retry, 1), **viability}


def _baseline_is_anchored(baseline: dict[str, Any]) -> bool:
    """Whether a measured baseline the rest of the session can be compared against is in hand."""
    return (
        str(baseline.get("warm_replay_status") or "") != "in_flight"
        and not bool(baseline.get("measure_round_dropped"))
        and (to_float(baseline.get("tput")) or 0.0) > 0.0
    )


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
    streak = int(baseline.get("failure_streak") or 0)
    if streak >= 3:
        return _transition_result(
            PHASE_CLOSE,
            "prelude_baseline_failed",
            {"terminal": True, "baseline_failure_streak": streak},
            inputs,
        )
    shortfall = _cold_anchor_shortfall(baseline)
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
    normal = _framework_exit(inputs)
    if normal is None:
        return None
    if bool(flags.get("kernel_enabled")):
        return _transition_result(PHASE_KERNEL_AGENT, normal[0], normal[1], inputs)
    return _transition_result(PHASE_SWEEP, normal[0], {**normal[1], "kernel_skipped": True}, inputs)


def _kernel_transition(inputs: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    """The transition KERNEL_AGENT takes when its leverage or its budget runs out."""
    normal = _kernel_exit(inputs)
    if normal is None:
        return None
    return _transition_result(PHASE_SWEEP, normal[0], normal[1], inputs)


def _sweep_transition(
    inputs: dict[str, Any],
    flags: dict[str, Any],
) -> tuple[str, str, dict[str, Any]] | None:
    """The transition SWEEP takes: another macro-cycle, or the close its result earns."""
    normal = _sweep_exit(inputs)
    if normal is None:
        return None
    exit_reason, exit_evidence = normal
    if exit_reason == "sweep_failed":
        return _transition_result(PHASE_CLOSE, exit_reason, exit_evidence, inputs)
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
    # Without FRAMEWORK_AGENT no further cycle was possible, so only a reached target outranks the ladder's reason.
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
    if target_at and phase_index(PHASE_ENABLEMENT) <= phase_index(current) < phase_index(PHASE_SWEEP):
        return _transition_result(PHASE_SWEEP, "target_reached", {"target_reached_at": target_at}, inputs)
    flags = dict(inputs.get("run_flags") or {})
    if current == PHASE_PRELUDE:
        return _prelude_transition(inputs, flags)
    if current == PHASE_ENABLEMENT:
        return _enablement_transition(inputs, flags)
    if current == PHASE_FRAMEWORK_AGENT:
        return _framework_transition(inputs, flags)
    if current == PHASE_KERNEL_AGENT:
        return _kernel_transition(inputs)
    if current == PHASE_SWEEP:
        return _sweep_transition(inputs, flags)
    return None


# Phase-transition / lifecycle write-owner functions (take ``state`` first and own the phase_history / lifecycle
# bookkeeping).
def bank_phase_segment(state, *, until_unix: float) -> float:
    """Bank the current phase's live segment, ending at ``until_unix``, into the durable totals."""
    phase = (state.phase or "").strip().upper()
    if not phase:
        return 0.0
    segment = phase_elapsed_seconds(state, now_unix=until_unix)
    totals = state.phase_elapsed_totals
    totals = dict(totals) if isinstance(totals, dict) else {}
    try:
        banked = max(0.0, float(totals.get(phase, 0.0) or 0.0))
    except (TypeError, ValueError):
        banked = 0.0
    totals[phase] = banked + segment
    state.phase_elapsed_totals = totals
    return segment


@dataclass(frozen=True)
class PhaseClock:
    """The wall-clock stamp a transition is recorded at; the current time for whichever part is ``None``."""

    ts: str | None = None
    ts_unix: float | None = None


class _PhaseBoundary(NamedTuple):
    """One recorded phase move, normalized for the session timeline."""

    from_phase: str
    to_phase: str
    reason: str
    evidence: dict[str, Any]
    ts: str
    ts_unix: float
    prev_cycle: int
    entry_cycle: int
    sequence: int


def _publish_phase_boundary(boundary: _PhaseBoundary) -> None:
    """Mirror a recorded phase move into LLM attribution and the session timeline."""
    # Publish the phase for LLM attribution: the spawn sites that tag outbound calls sit in specialists and kernel
    # tools and cannot reach SharedState.
    from hyperloom.common.llm_attribution import set_current_phase

    set_current_phase(boundary.to_phase)
    from hyperloom.inference_optimizer.trace.trajectory_trace import EVENT_PHASE, record_event

    record_event(
        EVENT_PHASE,
        phase=boundary.to_phase or None,
        attributes={
            "name": boundary.to_phase,
            "from_phase": boundary.from_phase or None,
            "to_phase": boundary.to_phase,
            "reason": boundary.reason,
            "macro_cycle": int(boundary.prev_cycle or 0),
        },
    )
    from hyperloom.inference_optimizer.breakdown.recorder import phase_event, record_stage_reached
    from hyperloom.inference_optimizer.breakdown.recorder.outcome_stage import PHASE_STAGES
    from hyperloom.inference_optimizer.breakdown.recorder.phase_records import PhaseCycle, PhaseEntry, PhaseExit
    from hyperloom.inference_optimizer.session.session_binding import SessionNotBoundError, bound_session

    try:
        # The phase itself, as a timeline event: close the span being left on
        # the exit that ended it, and open the one being entered. Recorded here
        # because here is where the two facts exist -- export could only pair
        # phase_history rows off two at a time to guess them back, and had no
        # row at all to close the segment the session ended in.
        if boundary.from_phase and boundary.from_phase != boundary.to_phase:
            phase_event.record_exit(
                PhaseExit(
                    at=PhaseCycle(boundary.from_phase, boundary.prev_cycle),
                    to_phase=boundary.to_phase,
                    reason=boundary.reason,
                    evidence=dict(boundary.evidence),
                    exited_at=boundary.ts,
                    exited_unix=boundary.ts_unix,
                )
            )
        phase_event.record_entry(
            PhaseEntry(
                at=PhaseCycle(boundary.to_phase, boundary.entry_cycle),
                sequence=boundary.sequence,
                from_phase=boundary.from_phase,
                reason=boundary.reason,
                evidence=dict(boundary.evidence),
                entered_at=boundary.ts,
                entered_unix=boundary.ts_unix,
            )
        )
        stage = PHASE_STAGES.get(boundary.to_phase)
        if stage:
            record_stage_reached(bound_session(), stage)
    except (OSError, ValueError, SessionNotBoundError):
        pass  # phase-boundary telemetry must not block the phase change


def record_phase_transition(
    state,
    *,
    to_phase: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
    at: PhaseClock = PhaseClock(),
) -> dict[str, Any]:
    """Append a phase_history row and atomically update ``phase`` fields; Coordinator-only writer. Returns the inserted row."""
    from datetime import datetime as _dt, timezone as _tz
    import time as _time

    now_ts = at.ts or _dt.now(_tz.utc).isoformat(timespec="seconds")
    now_unix = float(at.ts_unix if at.ts_unix is not None else _time.time())
    from_phase = (state.phase or "").strip().upper()
    prev_cycle = state.macro_cycle
    # The loopback opens the next cycle right after this row is recorded; the entered phase already belongs to it.
    entry_cycle = prev_cycle + 1 if (evidence or {}).get("loopback") else prev_cycle
    # Bank the finished segment for EVERY phase so the budget guards can charge
    # a phase for the whole run instead of the current entry.
    bank_phase_segment(state, until_unix=now_unix)
    row = make_history_row(HistoryEntry(from_phase, to_phase, reason, evidence, now_ts, now_unix, entry_cycle))
    history = list(state.phase_history or [])
    history.append(row)
    if len(history) > _PHASE_HISTORY_CAP:
        history = history[-_PHASE_HISTORY_CAP:]
    state.phase_history = history
    state.phase = row["to_phase"]
    state.phase_started_unix = now_unix
    _publish_phase_boundary(
        _PhaseBoundary(
            from_phase=from_phase,
            to_phase=str(row["to_phase"] or ""),
            reason=str(row.get("reason") or ""),
            evidence=dict(row.get("evidence") or {}),
            ts=str(row.get("ts") or ""),
            ts_unix=now_unix,
            prev_cycle=prev_cycle,
            entry_cycle=entry_cycle,
            sequence=len(history),
        )
    )
    return row


def append_phase_history_event(
    state,
    *,
    reason: str,
    evidence: dict[str, Any] | None = None,
    ts: str | None = None,
    ts_unix: float | None = None,
) -> dict[str, Any]:
    """Append a non-transition marker row for the current phase."""
    from datetime import datetime as _dt, timezone as _tz
    import time as _time

    now_ts = ts or _dt.now(_tz.utc).isoformat(timespec="seconds")
    now_unix = float(ts_unix if ts_unix is not None else _time.time())
    phase = (state.phase or "").strip().upper()
    row = make_history_row(
        HistoryEntry(
            from_phase=phase,
            to_phase=phase,
            reason=(reason or "").strip(),
            evidence=evidence,
            ts=now_ts,
            ts_unix=now_unix,
            cycle=state.macro_cycle,
        )
    )
    history = list(state.phase_history or [])
    history.append(row)
    if len(history) > _PHASE_HISTORY_CAP:
        history = history[-_PHASE_HISTORY_CAP:]
    state.phase_history = history
    from hyperloom.inference_optimizer.breakdown.recorder import phase_event
    from hyperloom.inference_optimizer.breakdown.recorder.phase_records import PhaseCycle, PhaseMarker

    phase_event.record_marker(
        PhaseMarker(
            at=PhaseCycle(phase, state.macro_cycle),
            sequence=len(history),
            reason=str(row.get("reason") or ""),
            evidence=dict(row.get("evidence") or {}),
            ts=str(row.get("ts") or ""),
        )
    )
    return row


def record_lifecycle_event(state, mark: LifecycleMark, *, ts: str | None = None) -> dict[str, Any]:
    """Append a structured lifecycle event marking a phase/step boundary; an unset phase is the current one."""
    events = state.lifecycle
    if events is None:
        events = state.lifecycle = []
    next_seq = (int(events[-1].get("seq", -1)) + 1) if events else 0
    event = make_lifecycle_event(
        mark if mark.phase is not None else replace(mark, phase=state.phase or ""),
        seq=next_seq,
        ts=ts or _now_iso(),
    )
    # Append in place, trim only when over the cap (O(1) common path).
    events.append(event)
    if len(events) > _LIFECYCLE_CAP:
        del events[:-_LIFECYCLE_CAP]
    return event


__all__ = [
    "DEFAULT_PHASE_BUDGET_PCT",
    "OPTIMIZATION_RESERVE_PCT",
    "DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK",
    "DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT",
    "DEFAULT_PLATEAU_EXPLORE_LOOKBACK",
    "LIFECYCLE_STATUS_ENTER",
    "LIFECYCLE_STEP_LABELS",
    "ENABLEMENT_MAX_ATTEMPTS",
    "PHASE_ALLOWED_ACTIONS",
    "PHASE_CLOSE",
    "PHASE_COORDINATOR_RESERVED",
    "PHASE_ENABLEMENT",
    "PHASE_FRAMEWORK_AGENT",
    "PHASE_HUMAN_LABELS",
    "PHASE_INDEX",
    "PHASE_KERNEL_AGENT",
    "PHASE_NAMES",
    "PHASE_PRELUDE",
    "PHASE_SWEEP",
    "HistoryEntry",
    "LifecycleMark",
    "PhaseClock",
    "PhaseFlags",
    "lifecycle_label",
    "make_lifecycle_event",
    "DEFAULT_GLOBAL_CONVERGENCE_NO_GAIN_CYCLES",
    "DEFAULT_LONGRUN_THRESHOLD_MINUTES",
    "is_long_run",
    "resolve_keep_threshold",
    "allowed_actions_for",
    "bank_phase_segment",
    "compute_next_phase",
    "cycle_reloop_decision",
    "initial_workflow_predicate_inputs",
    "replay_next_phase",
    "workflow_predicate_inputs",
    "per_lever_dryness",
    "append_phase_evidence_row",
    "append_phase_history_event",
    "baseline_round_cost_sec",
    "benchmark_cost_sec",
    "boot_cost_sec",
    "measured_seconds",
    "one_more_measurement_sec",
    "prelude_affordable_seconds",
    "prelude_can_afford",
    "session_usable_seconds",
    "render_phase_action_bullets",
    "compute_kernel_progress_fingerprint",
    "kernel_work_pending",
    "make_history_row",
    "normalize_budget_pct",
    "phase_budget_remaining_seconds",
    "phase_cumulative_seconds",
    "phase_elapsed_seconds",
    "phase_index",
    "phase_status_summary",
    "session_remaining_seconds",
    "warm_replay_in_flight",
]
