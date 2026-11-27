# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase state machine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

from hyperloom_common.coerce import to_unix
from hyperloom_common.env import env_float as _env_float
from hyperloom_common.timeutil import now_iso as _now_iso
from hyperloom.orchestrator.lever import (
    LEVER_CONFIG,
    LEVER_SOURCE_PATCH,
    LEVER_UPSTREAM_PR,
)
from ..kernel import kernel_decisions
from ..state.shared_state import (
    _LIFECYCLE_CAP,
)
from .controller_settlement import kernel_controller_settled
from .exit_predicates import framework_plateau
from .phase_chain import (
    LIFECYCLE_STATUS_ENTER,
    PHASE_ALLOWED_ACTIONS,
    PHASE_CLOSE,
    PHASE_COORDINATOR_RESERVED,
    PHASE_CYCLE_END,
    PHASE_ENABLEMENT,
    PHASE_FRAMEWORK_AGENT,
    PHASE_KERNEL_AGENT,
    PHASE_NAMES,
    PHASE_PRELUDE,
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
from .prelude_exits import (
    hot_pass_cost_sec,
)


log = logging.getLogger(__name__)


# Consecutive FAILED rounds before the lane stops with enablement_attempts_exhausted;
# abandoned and expired rounds are neutral and do not count towards it.
ENABLEMENT_MAX_ATTEMPTS: int = 8


# Default phase budgets (% of wall-clock). ENABLEMENT is absent on purpose: a
# budget apportions optimisation effort, and a combo that cannot run has nothing
# to optimise. CYCLE_END is absent too: it only drains and validates on entry,
# which the session budget alone gates. Neither carries a cap or an override flag.
DEFAULT_PHASE_BUDGET_PCT: dict[str, float] = {
    PHASE_PRELUDE: 0.03,
    # The optimisation phase carries both levers' share.
    PHASE_FRAMEWORK_AGENT: 0.40,
    PHASE_KERNEL_AGENT: 0.50,
    PHASE_CLOSE: 0.02,
}

# Share of the session held back for the phases that actually produce a result, and the reason a round
# PRELUDE can no longer pay for out of what is left outside it is refused.
OPTIMIZATION_RESERVE_PCT: float = 0.50
RETRY_PAST_THE_RESERVE: str = "retry_past_the_optimization_reserve"

# Wall-clock ceiling for an unbounded run (``max_minutes`` == 0): the container lifetime.
DEFAULT_LONGRUN_MAX_MINUTES: int = 14 * 24 * 60


# Plateau judgment defaults (CLI --plateau-* flags); kept here for pure callers + tests.
DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT: float = 0.5
DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK: int = 5
DEFAULT_PLATEAU_EXPLORE_LOOKBACK: int = 5


# FRAMEWORK per-candidate plateau: after this many consecutive resolved candidates without a KEEP (including
# non-benchmarked terminal outcomes), the source arm is dry.
DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK: int = 5


# R1 macro-cycle reloop: CYCLE_END loops back to FRAMEWORK_AGENT for a new macro-cycle while budget remains and the run
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


def cycle_reloop_min_remaining_sec(state: Any) -> float:
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
    now = float(now_unix if now_unix is not None else state_now_unix(state))
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
) -> dict[str, float]:
    """Move disabled work-phase shares to the enabled work phase.

    FRAMEWORK_AGENT and KERNEL_AGENT absorb proportionally, capped at 1.0; PRELUDE, ENABLEMENT, CYCLE_END, and CLOSE
    never absorb.
    """
    out = dict(base)
    toggles = {PHASE_FRAMEWORK_AGENT: optimize_enabled, PHASE_KERNEL_AGENT: kernel_enabled}
    disabled = [phase for phase, enabled in toggles.items() if not enabled]
    freed = sum(float(out.get(p, 0.0)) for p in disabled)
    for p in disabled:
        out[p] = 0.0
    if freed <= 0.0:
        return out
    absorbers = [p for p in (PHASE_FRAMEWORK_AGENT, PHASE_KERNEL_AGENT) if p not in disabled]
    weight = sum(float(out.get(p, 0.0)) for p in absorbers)
    if weight > 0.0:
        for p in absorbers:
            out[p] = float(out.get(p, 0.0)) + freed * float(out.get(p, 0.0)) / weight
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
def state_now_unix(state: Any) -> float:
    """Resolve the \"now\" timestamp; tests can inject ``state._now_unix``."""
    if hasattr(state, "_now_unix") and callable(state._now_unix):
        return float(state._now_unix())
    import time as _time

    return _time.time()


def phase_started_unix(state: Any) -> float:
    """Return the Unix timestamp the current phase started, defensively coerced."""
    raw = state.phase_started_unix
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _resume_boundary_unix(state: Any) -> float:
    """Return when the current run leg began, i.e. the most recent ``--resume``."""
    return max(0.0, to_unix(state.resumed_ts, 0.0) or 0.0)


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
    started = phase_started_unix(state)
    if started <= 0:
        return 0.0
    started = max(started, _resume_boundary_unix(state))
    now = float(now_unix if now_unix is not None else state_now_unix(state))
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


GEAK_STATUS_ERROR = "error"
# The statuses a finished GEAK run writes to ``geak_result.status``.
GEAK_TERMINAL_STATUSES = frozenset(
    {
        "ok",
        "no_gain",
        GEAK_STATUS_ERROR,
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
    if kernel_controller_settled(state) or _geak_phase_terminal(state):
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
    post_ready_sec = measured_seconds(state, _POST_READY_FIELD)
    if total_sec is None or post_ready_sec is None:
        return None
    return max(0.0, total_sec - post_ready_sec)


def baseline_round_cost_sec(state: Any, *, double_run: bool) -> float | None:
    """What a baseline round costs, or ``None`` when unmeasured."""
    first_pass_sec = measured_seconds(state, "baseline_runtime_sec")
    if first_pass_sec is None or not double_run:
        return first_pass_sec
    fields = ("baseline_warm_runtime_sec", _POST_READY_FIELD, "baseline_eval_runtime_sec")
    second_pass_sec = hot_pass_cost_sec(*(measured_seconds(state, field) for field in fields))
    return first_pass_sec if second_pass_sec is None else first_pass_sec + second_pass_sec


def one_more_measurement_sec(state: Any) -> float | None:
    """A boot plus one benchmark pass (the warm runtime once measured, else the cold post-ready time), or ``None``."""
    boot_sec = boot_cost_sec(state)
    warm_sec = measured_seconds(state, "baseline_warm_runtime_sec")
    benchmark_sec = warm_sec or measured_seconds(state, _POST_READY_FIELD)
    return None if boot_sec is None or benchmark_sec is None else boot_sec + benchmark_sec


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


def prelude_affordable_seconds(state: Any) -> tuple[float | None, dict[str, Any]]:
    """Seconds PRELUDE may still spend, and the numbers the figure is built from.

    The reserve is an absolute share of the session, so once preparation has eaten past it the subtraction
    would run negative. A negative is not a quantity of seconds anyone can spend: the figure is floored at
    zero, the same shape :meth:`SharedState.session_budget_usable_sec` keeps, so every caller reads one
    meaning -- zero is "nothing left outside the reserve", not a debt that grows while the phase waits.
    Reaching zero refuses a baseline retry at admission (:func:`a_retry_past_the_reserve_line`) and is a
    terminal for an idle PRELUDE (:func:`.prelude_exits._prelude_budget_line`), not a number the phase sits on.
    """
    max_sec = _max_minutes(state) * 60.0
    usable = state.session_budget_usable_sec()
    if max_sec <= 0.0 or usable is None:
        return None, {"reason": "unbounded_budget"}
    reserve_sec = max_sec * OPTIMIZATION_RESERVE_PCT
    affordable_sec = max(0.0, usable - reserve_sec)
    return affordable_sec, {
        "optimization_reserve_sec": round(reserve_sec, 1),
        "session_usable_sec": round(usable, 1),
        "affordable_sec": round(affordable_sec, 1),
        "bound": "optimization_reserve",
    }


def prelude_can_afford(state: Any, *, expected_cost_sec: float) -> tuple[bool, dict[str, Any]]:
    """Decide whether PRELUDE can still buy an optional arm costing ``expected_cost_sec``."""
    cost = max(0.0, float(expected_cost_sec or 0.0))
    affordable_sec, evidence = prelude_affordable_seconds(state)
    priced = {"expected_cost_sec": round(cost, 1), **evidence}
    if affordable_sec is None:
        return True, priced
    return affordable_sec >= cost, priced


def round_headroom_sec(state: Any) -> tuple[float | None, dict[str, Any]]:
    """Seconds this round's budget may still spend, and the numbers behind it."""
    if state is None:
        return None, {"reason": "no_session_state"}
    usable_sec = state.session_budget_usable_sec()
    if usable_sec is None:
        return None, {"reason": "unbounded_session_budget"}
    return usable_sec, {"bound": "session_usable", "affordable_sec": round(usable_sec, 1)}


def a_use_must_follow_the_round(state: Any) -> bool:
    """Whether this round is only worth running if something can be measured after it."""
    return str(getattr(state, "phase", "") or "").strip().upper() == PHASE_PRELUDE


def a_retry_past_the_reserve_line(state: Any) -> tuple[bool, dict[str, Any]]:
    """Whether this is a PRELUDE baseline retry the phase can no longer pay for.

    Once PRELUDE's own allowance past the reserve (:func:`prelude_affordable_seconds`) is gone, a leg that already
    failed a baseline may not retry; only this leg's failures count, so a resumed leg still gets its first one.
    """
    if not a_use_must_follow_the_round(state):
        return False, {}
    leg_failures = int(getattr(state, "baseline_failures_this_leg", 0) or 0)
    if leg_failures < 1:
        return False, {}
    affordable_sec, evidence = prelude_affordable_seconds(state)
    if affordable_sec is None or affordable_sec > 0.0:
        return False, {}
    return True, {"reason": RETRY_PAST_THE_RESERVE, "baseline_failures_this_leg": leg_failures, **evidence}


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


def _config_lever_evidence(state: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """The facts the config lever's plateau verdict is judged on.

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
    return {
        "recent_keep_gain_pct": round(recent_gain, 4),
        "keep_gain_threshold_pct": gain_floor,
        "empty_streak": streak,
        "empty_streak_threshold": streak_floor,
        "lookback": lookback,
    }


def _patch_lever_evidence(state: Any) -> dict[str, Any]:
    """The facts the patch levers' plateau verdict is judged on.

    Source patches and upstream PRs share a supply — the discovery and authoring
    ladder — so they are judged together: a trailing run of resolved candidates
    without an adoption, or a pump that has declared the supply exhausted. An
    authoring round that came back empty or never ran resolves its candidate
    too, so a lane that cannot run dries up like one that finds nothing.
    """
    streak = _trailing_no_keep(_lever_attempts(state, LEVER_SOURCE_PATCH, LEVER_UPSTREAM_PR))
    exhausted = bool(state.framework_agent_phase_done)
    return {
        "source_consecutive_no_keep": streak,
        "source_threshold": DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK,
        "source_candidates_exhausted": exhausted,
    }


def per_lever_dryness(state: Any) -> tuple[bool, dict[str, Any]]:
    """Whether every lever has run dry, over the unified attempts ledger.

    One lever going quiet raises ``switch_bottleneck`` so the next macro-cycle
    steers elsewhere; the phase advances only once none of them is paying.
    """
    overrides = _resolve_plateau_overrides(state)
    plateau = framework_plateau(
        {**_config_lever_evidence(state, overrides), **_patch_lever_evidence(state)},
    )
    return (plateau.config_dry and plateau.source_dry), plateau.arms


_POST_READY_FIELD = "baseline_post_ready_runtime_sec"


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
    from hyperloom_common.llm.llm_attribution import set_current_phase

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
    # A hint speaks for the phase it was sent in; carried into the next FRAMEWORK_AGENT it would end that one.
    state.pending_escalate_hint = ""
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
) -> dict[str, Any]:
    """Append a non-transition marker row for the current phase."""
    phase = (state.phase or "").strip().upper()
    row = make_history_row(
        HistoryEntry(
            from_phase=phase,
            to_phase=phase,
            reason=(reason or "").strip(),
            evidence=evidence,
            ts=_now_iso(timespec="seconds"),
            ts_unix=time.time(),
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


def record_lifecycle_event(state, mark: LifecycleMark) -> dict[str, Any]:
    """Append a structured lifecycle event marking a step boundary of the current phase."""
    events = state.lifecycle
    if events is None:
        events = state.lifecycle = []
    next_seq = (int(events[-1].get("seq", -1)) + 1) if events else 0
    event = make_lifecycle_event(
        mark,
        phase=state.phase or "",
        seq=next_seq,
        ts=_now_iso(),
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
    "ENABLEMENT_MAX_ATTEMPTS",
    "PHASE_ALLOWED_ACTIONS",
    "PHASE_CLOSE",
    "PHASE_COORDINATOR_RESERVED",
    "PHASE_CYCLE_END",
    "PHASE_ENABLEMENT",
    "PHASE_FRAMEWORK_AGENT",
    "PHASE_KERNEL_AGENT",
    "PHASE_NAMES",
    "PHASE_PRELUDE",
    "HistoryEntry",
    "LifecycleMark",
    "PhaseClock",
    "lifecycle_label",
    "make_lifecycle_event",
    "DEFAULT_GLOBAL_CONVERGENCE_NO_GAIN_CYCLES",
    "DEFAULT_LONGRUN_THRESHOLD_MINUTES",
    "is_long_run",
    "resolve_keep_threshold",
    "allowed_actions_for",
    "bank_phase_segment",
    "per_lever_dryness",
    "append_phase_evidence_row",
    "append_phase_history_event",
    "baseline_round_cost_sec",
    "hot_pass_cost_sec",
    "boot_cost_sec",
    "measured_seconds",
    "one_more_measurement_sec",
    "prelude_affordable_seconds",
    "prelude_can_afford",
    "render_phase_action_bullets",
    "compute_kernel_progress_fingerprint",
    "kernel_work_pending",
    "make_history_row",
    "normalize_budget_pct",
    "phase_budget_remaining_seconds",
    "phase_cumulative_seconds",
    "phase_elapsed_seconds",
    "phase_index",
    "session_remaining_seconds",
    "warm_replay_in_flight",
]
