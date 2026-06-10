# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Phase state machine.

Pure functions over a frozen SharedState; Coordinator is the only writer.
Monotonic chain PRELUDE → FRAMEWORK_PR → EXPLORE → KERNEL → SWEEP → CLOSE
(any phase → CLOSE on terminal/abort); ``recover`` is phase-orthogonal.
"""

from __future__ import annotations

import os
from typing import Any

from ..protocol.action_surfaces import (
    COORDINATOR_INTERNAL_ACTIONS,
    KERNEL_OWNED_ACTIONS,
    ROBUSTNESS_DELEGATE_ONLY_ACTIONS,
)


# Phase identifiers + ordering (monotonic chain)
PHASE_PRELUDE      = "PRELUDE"
PHASE_FRAMEWORK_PR = "FRAMEWORK_PR"
PHASE_EXPLORE      = "EXPLORE"
PHASE_KERNEL       = "KERNEL"
PHASE_SWEEP        = "SWEEP"
PHASE_CLOSE        = "CLOSE"

PHASE_NAMES: tuple[str, ...] = (
    PHASE_PRELUDE,
    PHASE_FRAMEWORK_PR,
    PHASE_EXPLORE,
    PHASE_KERNEL,
    PHASE_SWEEP,
    PHASE_CLOSE,
)
PHASE_INDEX: dict[str, int] = {name: i for i, name in enumerate(PHASE_NAMES)}


def phase_index(phase: str) -> int:
    """Return monotonic index of ``phase`` (Inv-2.1 check); unknown → -1."""
    return PHASE_INDEX.get((phase or "").strip().upper(), -1)


# Phase ↔ allowed action set: ALLOWED passes R1 phase_incompatible but
# Coordinator-auto actions stay out of PROPOSABLE so LLM proposals are denied.
PHASE_ALLOWED_ACTIONS: dict[str, frozenset[str]] = {
    PHASE_PRELUDE: frozenset({
        "target_analysis", "baseline", "roofline", "profile", "recover",
    }),
    PHASE_FRAMEWORK_PR: frozenset({
        # Coordinator-internal; integrate_patch is the Critic-gated consume side.
        "framework_pr", "integrate_patch", "roofline", "profile", "recover",
    }),
    PHASE_EXPLORE: frozenset({
        # merged grid runner + LLM specialist dispatch.
        "explore", "specialist",
        # Specialist source patches apply only through integrate_patch.
        "integrate_patch",
        # roofline/profile auto-enqueued on cumulative_gain_validated watermark.
        "roofline", "profile",
        "recover",
    }),
    PHASE_KERNEL: frozenset({
        # KERNEL_OWNED_ACTIONS from policy.py.
        "kernel_opt", "integrate", "deep_kernel_analysis",
        "operator_tuning", "vendor_kernel_config", "gemm_tuning",
        "roofline", "profile",
        "recover",
    }),
    PHASE_SWEEP: frozenset({
        # conc_sweep: Coordinator-internal post-sweep CONC-ladder benchmark; discovery-only.
        "sweep", "conc_sweep", "recover",
    }),
    PHASE_CLOSE: frozenset({
        "report", "session_breakdown",
        "recover",
    }),
}


def is_action_allowed_in_phase(action_name: str, phase: str) -> bool:
    """Return True iff ``action_name`` is in the phase allowlist (R1; unknown phase → deny)."""
    allowed = PHASE_ALLOWED_ACTIONS.get((phase or "").strip().upper())
    if allowed is None:
        return False
    return (action_name or "").strip() in allowed


def allowed_actions_for(phase: str) -> tuple[str, ...]:
    """Return ``PHASE_ALLOWED_ACTIONS[phase]`` as a sorted tuple (deterministic)."""
    return tuple(sorted(PHASE_ALLOWED_ACTIONS.get((phase or "").strip().upper(), frozenset())))


# Phase ↔ LLM-proposable set: allowlist minus Coordinator-managed and
# robustness-delegate-only actions, matching what PolicyGate accepts for Orchestration.
PHASE_LLM_PROPOSABLE_ACTIONS: dict[str, frozenset[str]] = {
    phase: actions - COORDINATOR_INTERNAL_ACTIONS - ROBUSTNESS_DELEGATE_ONLY_ACTIONS
    for phase, actions in PHASE_ALLOWED_ACTIONS.items()
}


def is_action_llm_proposable_in_phase(action_name: str, phase: str) -> bool:
    """Return True iff ``action_name`` is LLM-proposable in ``phase`` (unknown → deny)."""
    proposable = PHASE_LLM_PROPOSABLE_ACTIONS.get((phase or "").strip().upper())
    if proposable is None:
        return False
    return (action_name or "").strip() in proposable


def llm_proposable_actions_for(phase: str) -> tuple[str, ...]:
    """Return ``PHASE_LLM_PROPOSABLE_ACTIONS[phase]`` sorted (deterministic)."""
    return tuple(sorted(
        PHASE_LLM_PROPOSABLE_ACTIONS.get((phase or "").strip().upper(), frozenset())
    ))


# Interleave mode (env-flagged): widen EXPLORE/KERNEL proposable sets; chain stays monotonic.
PHASE_INTERLEAVE_ENV: str = "INFERENCE_OPTIMIZER_PHASE_INTERLEAVE"

# EXPLORE interleave adds KERNEL_OWNED_ACTIONS so kernel REQUESTs pass R1.
_INTERLEAVE_EXPLORE_EXTRAS: frozenset[str] = KERNEL_OWNED_ACTIONS

# KERNEL interleave adds the explore-side proposable triple.
_INTERLEAVE_KERNEL_EXTRAS: frozenset[str] = frozenset({
    "explore", "specialist", "integrate_patch",
})


def is_phase_interleave_enabled() -> bool:
    """Return True when EXPLORE↔KERNEL interleave is enabled (default ON, P3_18; env is rollback knob)."""
    raw = (os.environ.get(PHASE_INTERLEAVE_ENV) or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def llm_proposable_actions_for_with_interleave(
    phase: str, *, interleave: bool | None = None,
) -> frozenset[str]:
    """Return the active LLM-proposable set for ``phase`` (when interleave on, EXPLORE adds kernel-owned names, KERNEL adds the explore triple)."""
    key = (phase or "").strip().upper()
    base = PHASE_LLM_PROPOSABLE_ACTIONS.get(key, frozenset())
    if interleave is None:
        interleave = is_phase_interleave_enabled()
    if not interleave:
        return base
    if key == PHASE_EXPLORE:
        return base | _INTERLEAVE_EXPLORE_EXTRAS
    if key == PHASE_KERNEL:
        return base | _INTERLEAVE_KERNEL_EXTRAS
    return base


def is_action_llm_proposable_in_phase_with_interleave(
    action_name: str, phase: str, *, interleave: bool | None = None,
) -> bool:
    """Mirror of :func:`is_action_llm_proposable_in_phase` honoring the
    interleave flag."""
    proposable = llm_proposable_actions_for_with_interleave(
        phase, interleave=interleave,
    )
    return (action_name or "").strip() in proposable


# phase_exit_reasons vocab
PHASE_EXIT_REASONS: frozenset[str] = frozenset({
    # Normal exits
    "prelude_done",
    "plateau_explore",
    "plateau_kernel",
    "explore_phase_budget_exhausted",
    "kernel_phase_budget_exhausted",
    "sweep_done",
    "conc_sweep_done",                  # SWEEP → CLOSE when conc_sweep settles
    "sweep_budget_exhausted",
    "no_kernel_skipped",                # EXPLORE → SWEEP when kernel disabled
    "kernel_phase_aborted_no_trace",    # KERNEL → SWEEP when profile fails
    "explore_force_exit_low_budget",    # EXPLORE → next phase below operator force-exit thresholds
    "no_more_leverage",                 # EXPLORE/KERNEL → SWEEP (non-terminal) via steward skip_to_sweep; winds down through SWEEP → CLOSE
    # FRAMEWORK_PR phase transitions.
    "framework_pr_phase_done",          # FRAMEWORK_PR → EXPLORE normal completion (no more candidates)
    "framework_pr_plateau",             # FRAMEWORK_PR → EXPLORE; 3 consecutive batches with no candidate ≥1% gain
    "framework_pr_force_exit_low_budget",  # FRAMEWORK_PR → EXPLORE; remaining wall-clock dropped below configured fraction of max_hours

    # Terminal exits (any phase → CLOSE)
    "robustness_escalated",
    "target_reached",
    "time_exhausted",
    "time_exhausted_during_prelude",
    "user_stop_requested",
    "cortex_t0_failed",
    "cortex_drain_failed",
    "cortex_commit_failed",
    "prelude_baseline_failed",
    "prelude_policy_loop",
    "policy_loop",
    "crash_threshold_exceeded",
    "baseline_failed",                  # live baseline-failure marker
    "emergency",
    "max_ticks",
    "signal",

    # Construction sentinel — first phase_history entry on fresh session.
    "phase_entered",
})


# stop_reason vocab
STOP_REASON_VOCAB: frozenset[str] = frozenset({
    # Legacy sentinels — kept for backward compat (resume from old sessions).
    "target_reached",
    "no_more_leverage",
    "time_exhausted",
    "max_ticks",
    "policy_loop",
    "baseline_failed",
    "emergency",
    "coordinator_exception",
    "signal",
    "unknown",
    "custom",

    # Newer reasons.
    "crash_threshold_exceeded",
    "robustness_escalated",
    "user_stop_requested",
    "prelude_baseline_failed",
    "prelude_policy_loop",
    "time_exhausted_during_prelude",
    "cortex_t0_failed",
    "cortex_drain_failed",
    "cortex_commit_failed",
    "plateau_explore",
    "plateau_kernel",
    "no_kernel_skipped",
    "sweep_done",
    "conc_sweep_done",
    "explore_force_exit_low_budget",
    "framework_pr_phase_done",
    "framework_pr_plateau",
    "framework_pr_force_exit_low_budget",
    # R7: cyclic phase machine exhausted leverage across macro-cycles.
    "global_converged",

    # Context-window preflight: max_position_embeddings can't hold ISL+OSL.
    "model_context_window_too_small",
    # Model-arch preflight: multimodal/vision model unsupported.
    "unsupported_model_arch",
})


def is_valid_stop_reason(value: str) -> bool:
    return (value or "").strip() in STOP_REASON_VOCAB


def is_valid_phase_exit_reason(value: str) -> bool:
    return (value or "").strip() in PHASE_EXIT_REASONS


# Default phase budgets (% of wall-clock); P3_22 rebalance. IR-6 force-exit is the hard EXPLORE backstop; FRAMEWORK_PR uses a time wall.
DEFAULT_PHASE_BUDGET_PCT: dict[str, float] = {
    PHASE_PRELUDE: 0.03,
    PHASE_EXPLORE: 0.45,
    PHASE_KERNEL:  0.38,
    PHASE_SWEEP:   0.12,
    PHASE_CLOSE:   0.02,
}


# Plateau judgment defaults (CLI --plateau-* flags); kept here for pure callers + tests.
DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT:    float = 0.5
DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK:     int   = 3
DEFAULT_PLATEAU_EXPLORE_LOOKBACK:         int   = 5
DEFAULT_PLATEAU_KERNEL_REVERT_STREAK:     int   = 3
DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT:     float = 0.5
DEFAULT_PLATEAU_KERNEL_LOOKBACK:          int   = 5

# EXPLORE hard force-exit thresholds (IR-6 HARD time gate; overrides plateau).
# Fires when remaining wall-clock < HOURS_REMAINING OR EXPLORE budget fraction < BUDGET_PCT.
DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING: float = 3.0
DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT:      float = 0.20

# IR-6 under phase interleave (fix B): when EXPLORE↔KERNEL interleave is on,
# the original "leave 3h so the *separate* KERNEL phase can run" rationale is
# obsolete — kernel work already runs *inside* EXPLORE. The time gate then only
# needs to guarantee SWEEP → CLOSE + report can finish, so it collapses to a
# small CLOSE-buffer instead of the full KERNEL reservation. The phase-budget
# fraction gate is disabled in this mode for the same reason (EXPLORE legitimately
# spends the bulk of the budget because it is also doing KERNEL work). Both are
# overridable via the explicit thresholds the caller passes.
DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING_INTERLEAVE: float = 1.0
DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT_INTERLEAVE:      float = 0.0

# FRAMEWORK_PR plateau/force-exit knobs: plateau when each LOOKBACK batch < KEEP_GAIN_PCT; force-exit when remaining < RATIO * max_hours.
DEFAULT_FRAMEWORK_PR_PLATEAU_LOOKBACK:                 int   = 3
DEFAULT_FRAMEWORK_PR_PLATEAU_KEEP_GAIN_PCT:            float = 1.0
DEFAULT_FRAMEWORK_PR_FORCE_EXIT_HOURS_REMAINING_RATIO: float = 0.6


# ---------------------------------------------------------------------------
# R1 cyclic phase machine (env-gated)
# ---------------------------------------------------------------------------
# When enabled, SWEEP loops back to EXPLORE (a new macro-cycle) while budget
# remains and the run hasn't globally converged, instead of always terminating
# at CLOSE. Off by default => behaviour identical to the monotonic chain.
PHASE_CYCLIC_ENV: str = "INFERENCE_OPTIMIZER_CYCLIC_PHASES"

# Safety ceiling on macro-cycles (defense against a pathological tight loop).
DEFAULT_MAX_MACRO_CYCLES: int = 1000

# Minimum session wall-clock (seconds) that must remain to justify opening a new
# macro-cycle; below this we wind down to CLOSE instead of starting a cycle we
# cannot meaningfully use.
DEFAULT_CYCLE_RELOOP_MIN_REMAINING_SEC: float = 1800.0  # 30 min

# R7 global convergence: number of consecutive no-gain macro-cycles after which
# the run is considered converged (stop looping → CLOSE).
DEFAULT_GLOBAL_CONVERGENCE_NO_GAIN_CYCLES: int = 3

# A macro-cycle "gained" when validated cumulative gain rose by more than this
# (percentage points); guards against float noise being read as progress.
DEFAULT_CYCLE_MIN_GAIN_PCT: float = 1e-6


def is_cyclic_phases_enabled() -> bool:
    """True iff the R1 cyclic phase machine is enabled via env."""
    return os.environ.get(PHASE_CYCLIC_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _cumulative_gain_validated(state: Any) -> float:
    try:
        return float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def should_reloop_to_explore(
    state: Any,
    *,
    now_unix: float | None = None,
    max_cycles: int = DEFAULT_MAX_MACRO_CYCLES,
    min_remaining_sec: float = DEFAULT_CYCLE_RELOOP_MIN_REMAINING_SEC,
    no_gain_cycles: int = DEFAULT_GLOBAL_CONVERGENCE_NO_GAIN_CYCLES,
    min_gain_pct: float = DEFAULT_CYCLE_MIN_GAIN_PCT,
) -> tuple[bool, dict[str, Any]]:
    """Decide whether SWEEP should open a new macro-cycle (R1) or wind to CLOSE.

    Pure: never mutates state. Returns ``(reloop, evidence)``. The evidence
    carries the *effective* no-gain streak for the cycle that just completed so
    the Coordinator can persist it on the loopback/close transition.

    Loops back iff cyclic mode is on AND below the macro-cycle safety cap AND
    the run has not globally converged (R7: ``no_gain_cycles`` consecutive
    no-gain cycles) AND enough session budget remains to use a fresh cycle.
    """
    cycle = int(getattr(state, "macro_cycle", 0) or 0)
    evidence: dict[str, Any] = {"cyclic": is_cyclic_phases_enabled(), "macro_cycle": cycle}
    if not is_cyclic_phases_enabled():
        return False, evidence

    # Per-cycle gain since this cycle started → effective no-gain streak.
    cur_gain = _cumulative_gain_validated(state)
    start_gain = float(getattr(state, "gain_at_cycle_start", 0.0) or 0.0)
    cycle_gained = (cur_gain - start_gain) > float(min_gain_pct)
    prior_streak = int(getattr(state, "no_gain_cycle_streak", 0) or 0)
    effective_streak = 0 if cycle_gained else prior_streak + 1
    evidence["cycle_gain_delta"] = round(cur_gain - start_gain, 6)
    evidence["cycle_gained"] = cycle_gained
    evidence["no_gain_cycle_streak_effective"] = effective_streak

    # Safety cap on macro-cycles.
    if (cycle + 1) >= int(max_cycles):
        evidence["reloop_blocked"] = "max_cycles"
        return False, evidence

    # R7 global convergence.
    if effective_streak >= int(no_gain_cycles):
        evidence["reloop_blocked"] = "global_converged"
        return False, evidence

    # Budget remaining must justify a fresh cycle.
    remaining = session_remaining_seconds(state, now_unix=now_unix)
    if remaining is not None and remaining < float(min_remaining_sec):
        evidence["reloop_blocked"] = "insufficient_remaining"
        evidence["session_remaining_seconds"] = round(remaining, 2)
        return False, evidence

    evidence["reloop"] = True
    evidence["next_cycle"] = cycle + 1
    return True, evidence


# escalate_strategy_change hint vocabulary. Closed enum; unknown hints logged, never change phase.
ESCALATE_HINT_SKIP_TO_KERNEL:      str = "skip_to_kernel"
ESCALATE_HINT_SKIP_TO_SWEEP:       str = "skip_to_sweep"
ESCALATE_HINT_SKIP_TO_CLOSE:       str = "skip_to_close"
ESCALATE_HINT_EXTEND_EXPLORE_BUDGET: str = "extend_explore_budget"
ESCALATE_HINT_EXTEND_KERNEL_BUDGET:  str = "extend_kernel_budget"
ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX: str = "pause_specialist_"

# ``skip_to_sweep`` is the non-terminal "no more leverage" signal (winds down
# to SWEEP → CLOSE), unlike terminal ``skip_to_close``.
ESCALATE_HINT_VOCAB: frozenset[str] = frozenset({
    ESCALATE_HINT_SKIP_TO_KERNEL,
    ESCALATE_HINT_SKIP_TO_SWEEP,
    ESCALATE_HINT_SKIP_TO_CLOSE,
    ESCALATE_HINT_EXTEND_EXPLORE_BUDGET,
    ESCALATE_HINT_EXTEND_KERNEL_BUDGET,
})

# ``extend_*_budget`` hints raise a phase budget by DELTA up to CAP.
ESCALATE_HINT_BUDGET_BUMP_DELTA: float = 0.05   # +5 percentage points per hint
ESCALATE_HINT_BUDGET_BUMP_CAP:   float = 0.80   # absolute ceiling

# True when a hint string is structurally a pause-specialist directive.
def is_pause_specialist_hint(hint: str) -> bool:
    h = (hint or "").strip()
    return h.startswith(ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX) and len(h) > len(
        ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX,
    )


def is_valid_escalate_hint(hint: str) -> bool:
    """Return True for any hint Coordinator should act on (closed vocab + ``pause_specialist_<domain>``)."""
    return (hint or "").strip() in ESCALATE_HINT_VOCAB or is_pause_specialist_hint(hint)


def apply_escalate_budget_bump(
    current_budget_pct: dict[str, float] | None,
    *,
    phase: str,
    delta: float = ESCALATE_HINT_BUDGET_BUMP_DELTA,
    cap: float = ESCALATE_HINT_BUDGET_BUMP_CAP,
) -> dict[str, float]:
    """Return a budget map with ``phase`` raised by ``delta`` (capped at 80%)."""
    phase_key = (phase or "").strip().upper()
    if phase_key not in PHASE_NAMES:
        return dict(current_budget_pct or {})
    out = normalize_budget_pct(current_budget_pct)
    new_val = float(out.get(phase_key, 0.0)) + float(delta or 0.0)
    new_val = min(float(cap), max(0.0, new_val))
    out[phase_key] = new_val
    return out


def normalize_budget_pct(
    budget: dict[str, float] | None,
) -> dict[str, float]:
    """Return a sanitized ``phase -> pct`` mapping (budgets are upper bounds, not renormalized to 1.0)."""
    out = dict(DEFAULT_PHASE_BUDGET_PCT)
    if not budget:
        return out
    for phase, val in budget.items():
        canon = (phase or "").strip().upper()
        if canon not in PHASE_NAMES:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if not (0.0 < f <= 1.0):
            continue
        out[canon] = f
    return out


# Pure judgment helpers (used by Coordinator at each tick end)
def _now_unix(state: Any) -> float:
    """Resolve the "now" timestamp; tests can inject ``state._now_unix``."""
    if hasattr(state, "_now_unix") and callable(state._now_unix):
        return float(state._now_unix())  # type: ignore[attr-defined]
    import time as _time
    return _time.time()


def _phase_started_unix(state: Any) -> float:
    raw = getattr(state, "phase_started_unix", 0.0)
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _pending_escalate_hint(state: Any) -> str:
    """Return a pending escalate hint to act on this tick (unknown hints → empty)."""
    raw = str(getattr(state, "pending_escalate_hint", "") or "").strip()
    if not raw:
        return ""
    if is_valid_escalate_hint(raw):
        return raw
    return ""


def _max_minutes(state: Any) -> float:
    try:
        return float(getattr(state, "max_minutes", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _budget_minutes(state: Any) -> float:
    """Wall-clock minutes the PER-PHASE budget fractions apply to (R2).

    In cyclic mode the Coordinator sets ``cycle_minutes`` > 0 so each phase's
    budget (``DEFAULT_PHASE_BUDGET_PCT``) is a fraction of ONE macro-cycle's
    window rather than the whole run. 0 (legacy/non-cyclic) falls back to the
    total ``max_minutes`` so behaviour is identical to the monotonic chain.
    Note: ``session_remaining_seconds`` deliberately keeps using ``max_minutes``
    — the global deadline is per-run, not per-cycle.
    """
    try:
        cm = float(getattr(state, "cycle_minutes", 0) or 0)
    except (TypeError, ValueError):
        cm = 0.0
    if cm > 0:
        return cm
    return _max_minutes(state)


def phase_elapsed_seconds(state: Any, *, now_unix: float | None = None) -> float:
    """Return wall-clock seconds spent in current phase. 0 if not started."""
    started = _phase_started_unix(state)
    if started <= 0:
        return 0.0
    now = float(now_unix if now_unix is not None else _now_unix(state))
    return max(0.0, now - started)


def phase_budget_remaining_seconds(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
) -> float | None:
    """Return seconds remaining in the current phase's budget (``None`` when budget window 0 = unlimited)."""
    mm = _budget_minutes(state)
    if mm <= 0:
        return None
    budget = normalize_budget_pct(budget_pct or getattr(state, "phase_budget_pct", None))
    pct = budget.get((getattr(state, "phase", "") or "").upper(), 0.0)
    if pct <= 0:
        return None
    budget_seconds = mm * 60.0 * pct
    return max(0.0, budget_seconds - phase_elapsed_seconds(state, now_unix=now_unix))


# EXPLORE hard force-exit (HARD time gate)
def session_remaining_seconds(
    state: Any, *, now_unix: float | None = None,
) -> float | None:
    """Total wall-clock seconds remaining for the session (``None`` when ``max_minutes`` 0 or ``start_ts`` unparseable)."""
    mm = _max_minutes(state)
    if mm <= 0:
        return None
    start_ts = str(getattr(state, "start_ts", "") or "").strip()
    if not start_ts:
        return None
    try:
        from datetime import datetime, timezone
        start = datetime.fromisoformat(start_ts)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        now_dt = datetime.now(timezone.utc)
        elapsed_sec = max(0.0, (now_dt - start).total_seconds())
    except (ValueError, TypeError):
        return None
    return max(0.0, mm * 60.0 - elapsed_sec)


def should_force_exit_explore(
    state: Any,
    *,
    hours_remaining_threshold: float = DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING,
    budget_pct_threshold: float = DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Return ``(True, evidence)`` when HARD EXPLORE force-exit fires (IR-6).

    Fires when session remaining ≤ hours_threshold*3600 OR phase remaining
    pct ≤ budget_pct_threshold; ``evidence`` records which fired.

    Fix B (interleave-aware IR-6): when EXPLORE↔KERNEL interleave is enabled
    and the caller is using the default thresholds, narrow them to a
    CLOSE-buffer (the "reserve 3h for a separate KERNEL phase" rationale no
    longer applies because kernel work runs inside EXPLORE). Explicit
    non-default thresholds from the caller always win.
    """
    if is_phase_interleave_enabled():
        if float(hours_remaining_threshold) == DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING:
            hours_remaining_threshold = DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING_INTERLEAVE
        if float(budget_pct_threshold) == DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT:
            budget_pct_threshold = DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT_INTERLEAVE
    evidence: dict[str, Any] = {
        "hours_remaining_threshold":  float(hours_remaining_threshold),
        "budget_pct_threshold":       float(budget_pct_threshold),
        "interleave_aware_ir6":       bool(is_phase_interleave_enabled()),
    }
    fired = False
    fired_reasons: list[str] = []

    # Non-positive threshold = disabled; both disabled turns force-exit off.
    hours_threshold_enabled = float(hours_remaining_threshold) > 0.0
    pct_threshold_enabled = float(budget_pct_threshold) > 0.0

    session_remaining = session_remaining_seconds(state, now_unix=now_unix)
    if session_remaining is not None and hours_threshold_enabled:
        evidence["session_remaining_seconds"] = round(session_remaining, 2)
        threshold_sec = float(hours_remaining_threshold) * 3600.0
        if session_remaining <= threshold_sec:
            fired = True
            fired_reasons.append("session_remaining")

    phase_remaining = phase_budget_remaining_seconds(
        state, budget_pct=budget_pct, now_unix=now_unix,
    )
    if phase_remaining is not None:
        # Express remaining as a fraction of the phase's total budget (per-cycle
        # window in cyclic mode, else the whole run).
        mm = _budget_minutes(state)
        budget = normalize_budget_pct(
            budget_pct or getattr(state, "phase_budget_pct", None)
        )
        pct_alloc = budget.get((getattr(state, "phase", "") or "").upper(), 0.0)
        if mm > 0 and pct_alloc > 0:
            phase_total_sec = mm * 60.0 * pct_alloc
            remaining_pct = (
                phase_remaining / phase_total_sec if phase_total_sec > 0 else 0.0
            )
            evidence["phase_remaining_pct"] = round(remaining_pct, 4)
            evidence["phase_remaining_seconds"] = round(phase_remaining, 2)
            if (
                pct_threshold_enabled
                and remaining_pct <= float(budget_pct_threshold)
            ):
                fired = True
                fired_reasons.append("phase_remaining_pct")

    evidence["fired_reasons"] = fired_reasons
    return fired, evidence


# plateau pure functions
def compute_plateau_explore(
    state: Any,
    *,
    lookback: int = DEFAULT_PLATEAU_EXPLORE_LOOKBACK,
    keep_gain_threshold_pct: float = DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
    empty_streak_threshold: int = DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
) -> tuple[bool, dict[str, Any]]:
    """Real plateau_explore → ``(triggered, evidence)``.

    Trigger (AND, KB_design §3.8 §5.1): recent_keep_gain < threshold AND
    recent_empty_streak >= empty_streak_threshold.
    """
    if lookback <= 0:
        return False, {"reason": "lookback_disabled"}
    keep_gain_threshold_pct = float(keep_gain_threshold_pct or 0.0)
    empty_streak_threshold = int(empty_streak_threshold or 0)

    explore_search = getattr(state, "explore_search", None) or {}
    if not isinstance(explore_search, dict):
        explore_search = {}
    winners_history = explore_search.get("winners_history") or []
    if not isinstance(winners_history, list):
        winners_history = []
    recent_winners = list(winners_history[-lookback:])
    recent_keep_gain = 0.0
    for w in recent_winners:
        if not isinstance(w, dict):
            continue
        gain = w.get("gain_pct")
        try:
            recent_keep_gain += float(gain or 0.0)
        except (TypeError, ValueError):
            continue

    specialist_rounds = getattr(state, "specialist_rounds", None) or []
    if not isinstance(specialist_rounds, list):
        specialist_rounds = []

    def _round_is_empty(row: Any) -> bool:
        if not isinstance(row, dict):
            return False
        # Fall back to proposal_count for older round summaries.
        try:
            proposals = int(
                row.get("proposals_total")
                if row.get("proposals_total") is not None
                else row.get("proposal_count") or 0,
            )
        except (TypeError, ValueError):
            proposals = 0
        try:
            kept = int(
                row.get("proposals_kept")
                if row.get("proposals_kept") is not None
                else row.get("kept_count") or 0,
            )
        except (TypeError, ValueError):
            kept = 0
        return proposals == 0 and kept == 0

    # Walk from newest to oldest counting the trailing-empty streak.
    streak = 0
    for row in reversed(specialist_rounds):
        if _round_is_empty(row):
            streak += 1
        else:
            break

    triggered = (
        recent_keep_gain < keep_gain_threshold_pct
        and streak >= empty_streak_threshold
    )
    return triggered, {
        "recent_keep_gain_pct":     round(recent_keep_gain, 4),
        "keep_gain_threshold_pct":  keep_gain_threshold_pct,
        "empty_streak":             int(streak),
        "empty_streak_threshold":   empty_streak_threshold,
        "lookback":                 int(lookback),
        "winners_seen":             len(recent_winners),
        "specialist_rounds_seen":   len(specialist_rounds),
    }


def compute_plateau_kernel(
    state: Any,
    *,
    lookback: int = DEFAULT_PLATEAU_KERNEL_LOOKBACK,
    revert_streak_threshold: int = DEFAULT_PLATEAU_KERNEL_REVERT_STREAK,
    keep_gain_threshold_pct: float = DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT,
) -> tuple[bool, dict[str, Any]]:
    """Real plateau_kernel → ``(triggered, evidence)``.

    Trigger (OR, KB_design §3.8 §5.2 — weaker than explore's AND): revert_streak
    >= threshold OR recent_keep_gain < keep_gain_threshold_pct.
    """
    lookback = int(lookback or 0)
    revert_streak_threshold = int(revert_streak_threshold or 0)
    keep_gain_threshold_pct = float(keep_gain_threshold_pct or 0.0)
    if lookback <= 0 or revert_streak_threshold <= 0:
        return False, {"reason": "thresholds_disabled"}

    integ_attempts = getattr(state, "kernel_integrate_attempts", None) or {}
    if not isinstance(integ_attempts, dict):
        integ_attempts = {}

    # Flatten the integrate attempt log into a time-ordered list, take the
    # last ``lookback`` rows.
    flat: list[tuple[str, str, float]] = []   # (decision, ts, gain_pct)
    for ent in integ_attempts.values():
        if not isinstance(ent, dict):
            continue
        for a in ent.get("attempts") or []:
            if not isinstance(a, dict):
                continue
            decision = str(a.get("decision") or "").upper().strip()
            if not decision:
                continue
            ts = str(a.get("ts") or "")
            try:
                gain = float(a.get("gain_pct") or a.get("validated_gain_pct") or 0.0)
            except (TypeError, ValueError):
                gain = 0.0
            flat.append((decision, ts, gain))
    # Sort by ts (lexicographic on ISO works); fall back to insertion order.
    flat.sort(key=lambda r: r[1])
    recent = flat[-lookback:]

    # Empty-data guard: empty ledger (KERNEL just entered) must NOT auto-trigger plateau (would skip kernel phase).
    if not recent:
        return False, {
            "reason":                   "no_kernel_attempts_yet",
            "revert_streak_threshold":  int(revert_streak_threshold),
            "keep_gain_threshold_pct":  keep_gain_threshold_pct,
            "lookback":                 int(lookback),
            "attempts_seen":            0,
        }

    # REVERT streak from the tail.
    revert_streak = 0
    for decision, _ts, _g in reversed(recent):
        if decision in ("REVERT", "NEEDS_REVIEW"):
            revert_streak += 1
        else:
            break
    # KEEP-gain sum across the same lookback window.
    recent_keep_gain = sum(g for d, _t, g in recent if d == "KEEP")

    triggered = (
        revert_streak >= revert_streak_threshold
        or recent_keep_gain < keep_gain_threshold_pct
    )
    return triggered, {
        "revert_streak":            int(revert_streak),
        "revert_streak_threshold":  int(revert_streak_threshold),
        "recent_keep_gain_pct":     round(recent_keep_gain, 4),
        "keep_gain_threshold_pct":  keep_gain_threshold_pct,
        "lookback":                 int(lookback),
        "attempts_seen":            len(recent),
    }


# terminal / abort (global)
def _global_terminal(state: Any) -> tuple[str, dict[str, Any]] | None:
    """Return ``(stop_reason, evidence)`` for a phase-orthogonal stop.

    Priority: 1. ``skip_to_close`` → ``robustness_escalated``; 2. Coordinator ``stop_reason``.
    """
    hint = _pending_escalate_hint(state)
    if hint == ESCALATE_HINT_SKIP_TO_CLOSE:
        return "robustness_escalated", {
            "evidence": "llm_escalation",
            "hint": hint,
        }
    sr = (getattr(state, "stop_reason", "") or "").strip()
    if sr:
        # Coordinator-set stop_reason takes precedence over phase exits.
        if not is_valid_stop_reason(sr):
            # Unknown values tolerated for resume parity.
            return sr, {"reason_origin": "shared_state.stop_reason", "vocab": "unknown"}
        return sr, {"reason_origin": "shared_state.stop_reason"}
    return None


# per-phase judgments
def warm_replay_in_flight(state: Any) -> bool:
    """True while the PRELUDE warm-recipe replay task has not finished (PRELUDE must not exit until False — GPU contention)."""
    outcome = getattr(state, "warm_replay_outcome", None) or {}
    if not isinstance(outcome, dict):
        return False
    return str(outcome.get("status") or "").strip() == "in_flight"


def exit_normal_prelude(state: Any) -> tuple[str, dict[str, Any]] | None:
    """``baseline_tput > 0`` and warm-replay settled → ``prelude_done`` (else ``None``)."""
    if warm_replay_in_flight(state):
        return None
    try:
        tput = float(getattr(state, "baseline_tput", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if tput > 0.0:
        return "prelude_done", {"baseline_tput": tput}
    return None


def exit_terminal_prelude(state: Any) -> tuple[str, dict[str, Any]] | None:
    streak = int(getattr(state, "baseline_failure_streak", 0) or 0)
    if streak >= 3:
        return "prelude_baseline_failed", {"baseline_failure_streak": streak}
    return None


def abort_prelude(state: Any) -> tuple[str, dict[str, Any]] | None:
    # Treat cortex_t0_failed / time_exhausted_during_prelude etc. as a PRELUDE
    # abort so phase_history captures the boundary.
    sr = (getattr(state, "stop_reason", "") or "").strip()
    if sr in ("cortex_t0_failed", "time_exhausted_during_prelude",
              "prelude_policy_loop", "user_stop_requested"):
        return sr, {"reason_origin": "shared_state.stop_reason"}
    return None


def exit_normal_explore(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
    force_exit_hours_remaining: float = DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING,
    force_exit_budget_pct: float = DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT,
) -> tuple[str, dict[str, Any]] | None:
    """EXPLORE normal exit.

    Priority: 0. HARD force-exit (IR-6, overrides plateau); 1. ``skip_to_kernel``
    → ``plateau_explore``; 2. ``skip_to_sweep`` → ``no_more_leverage`` (non-terminal);
    3. phase budget exhausted.
    """
    forced, force_ev = should_force_exit_explore(
        state,
        hours_remaining_threshold=force_exit_hours_remaining,
        budget_pct_threshold=force_exit_budget_pct,
        budget_pct=budget_pct,
        now_unix=now_unix,
    )
    if forced:
        return "explore_force_exit_low_budget", {
            "evidence": "force_exit",
            **force_ev,
        }

    hint = _pending_escalate_hint(state)
    if hint == ESCALATE_HINT_SKIP_TO_KERNEL:
        return "plateau_explore", {
            "evidence": "llm_escalation",
            "hint": hint,
        }
    if hint == ESCALATE_HINT_SKIP_TO_SWEEP:
        return "no_more_leverage", {
            "evidence": "no_more_leverage",
            "hint": hint,
        }
    # R3: in cyclic mode a detected EXPLORE plateau is *actionable* — wind the
    # cycle down (→ SWEEP → reloop) and flag that the next cycle should switch
    # bottleneck. Advisory-only off cyclic mode (legacy behaviour preserved).
    if is_cyclic_phases_enabled():
        plateaued, plateau_ev = compute_plateau_explore(state)
        if plateaued:
            return "no_more_leverage", {
                "evidence": "plateau_explore",
                "plateau": True,
                "switch_bottleneck": True,
                **plateau_ev,
            }
    remaining = phase_budget_remaining_seconds(
        state, budget_pct=budget_pct, now_unix=now_unix,
    )
    if remaining is not None and remaining <= 0:
        return "explore_phase_budget_exhausted", {
            "elapsed_seconds": phase_elapsed_seconds(state, now_unix=now_unix),
        }
    return None


def exit_normal_kernel(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """KERNEL normal exit.

    Priority: 1. ``skip_to_close`` defers to global terminal; 2. ``skip_to_sweep``
    → ``no_more_leverage``; 3. phase budget exhausted.
    """
    if _pending_escalate_hint(state) == ESCALATE_HINT_SKIP_TO_SWEEP:
        return "no_more_leverage", {
            "evidence": "no_more_leverage",
            "hint": ESCALATE_HINT_SKIP_TO_SWEEP,
        }
    rejected = getattr(state, "rejected_kernel_ids", None) or []
    rejected_count = len(rejected) if isinstance(rejected, list) else 0
    remaining = phase_budget_remaining_seconds(
        state, budget_pct=budget_pct, now_unix=now_unix,
    )
    if remaining is not None and remaining <= 0:
        return "kernel_phase_budget_exhausted", {
            "elapsed_seconds": phase_elapsed_seconds(state, now_unix=now_unix),
            "rejected_kernel_count": rejected_count,
        }
    return None


def exit_normal_sweep(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """SWEEP normal exit: sweep_done OR conc_sweep_done OR budget exhausted.

    Bug #12: conc_sweep completion emits an exit so a singleton-blocked sweep doesn't idle.
    """
    last_sweep = getattr(state, "last_sweep", None) or {}
    if isinstance(last_sweep, dict):
        status = str(last_sweep.get("status") or "").lower()
        if status in ("succeeded", "partial", "completed"):
            return "sweep_done", {"sweep_status": status}
    last_conc = getattr(state, "last_conc_sweep", None) or {}
    if isinstance(last_conc, dict):
        cs_status = str(last_conc.get("status") or "").lower()
        if cs_status in ("succeeded", "partial", "completed", "skipped"):
            return "conc_sweep_done", {"conc_sweep_status": cs_status}
    remaining = phase_budget_remaining_seconds(
        state, budget_pct=budget_pct, now_unix=now_unix,
    )
    if remaining is not None and remaining <= 0:
        return "sweep_budget_exhausted", {
            "elapsed_seconds": phase_elapsed_seconds(state, now_unix=now_unix),
        }
    return None


# Transition decision (the only function the Coordinator calls each tick)
def _resolve_plateau_overrides(state: Any) -> dict[str, Any]:
    """Pull operator-tuned plateau thresholds off
    :attr:`SharedState.plateau_overrides` (empty → library defaults)."""
    overrides = getattr(state, "plateau_overrides", None) or {}
    return dict(overrides) if isinstance(overrides, dict) else {}


def _framework_pr_batch_is_complete(
    batch: dict[str, Any],
    progress_by_batch: dict[str, int],
) -> bool:
    """A FRAMEWORK_PR batch is complete iff every candidate has a terminal-status row in ``framework_pr_phase_progress`` (guards the plateau judge)."""
    candidates = batch.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return True
    total = sum(1 for c in candidates if isinstance(c, dict))
    if total == 0:
        return True
    batch_id = str(batch.get("batch_id") or "")
    processed = int(progress_by_batch.get(batch_id, 0))
    return processed >= total


def compute_plateau_framework_pr(
    state: Any,
    *,
    lookback: int = DEFAULT_FRAMEWORK_PR_PLATEAU_LOOKBACK,
    keep_gain_threshold_pct: float = DEFAULT_FRAMEWORK_PR_PLATEAU_KEEP_GAIN_PCT,
) -> tuple[bool, dict[str, Any]]:
    """Pure plateau judgment for FRAMEWORK_PR → ``(triggered, evidence)``.

    Triggers when the last ``lookback`` fully-processed batches each carry
    ``max_gain_pct_observed_in_batch < keep_gain_threshold_pct``. Advisory-only.
    """
    batches = getattr(state, "framework_pr_batches", None) or []
    lookback_int = int(lookback or 0)
    base_evidence = {
        "lookback":                lookback_int,
        "keep_gain_pct_threshold": float(keep_gain_threshold_pct),
        "batch_max_gains":         [],
    }
    if (
        not isinstance(batches, list)
        or lookback_int <= 0
        or len(batches) < lookback_int
    ):
        return False, base_evidence
    progress = getattr(state, "framework_pr_phase_progress", None) or []
    progress_by_batch: dict[str, int] = {}
    for row in progress:
        if isinstance(row, dict):
            bid = str(row.get("batch_id") or "")
            progress_by_batch[bid] = progress_by_batch.get(bid, 0) + 1
    complete_tail: list[dict[str, Any]] = []
    for entry in reversed(batches):
        if not isinstance(entry, dict):
            continue
        if _framework_pr_batch_is_complete(entry, progress_by_batch):
            complete_tail.append(entry)
            if len(complete_tail) >= lookback_int:
                break
    if len(complete_tail) < lookback_int:
        return False, base_evidence
    max_gains: list[float] = []
    for entry in complete_tail:
        try:
            max_gains.append(
                float(entry.get("max_gain_pct_observed_in_batch") or 0.0)
            )
        except (TypeError, ValueError):
            max_gains.append(0.0)
    triggered = bool(max_gains) and all(
        g < float(keep_gain_threshold_pct) for g in max_gains
    )
    return triggered, {
        "lookback":                lookback_int,
        "keep_gain_pct_threshold": float(keep_gain_threshold_pct),
        "batch_max_gains":         list(reversed(max_gains)),
    }


def _framework_pr_pending_candidate_count(state: Any) -> int:
    """Count candidates discovered into a batch but missing a progress row."""
    batches = getattr(state, "framework_pr_batches", None) or []
    if not isinstance(batches, list) or not batches:
        return 0
    progress = getattr(state, "framework_pr_phase_progress", None) or []
    progress_by_batch: dict[str, int] = {}
    for row in progress:
        if isinstance(row, dict):
            bid = str(row.get("batch_id") or "")
            progress_by_batch[bid] = progress_by_batch.get(bid, 0) + 1
    pending = 0
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        candidates = batch.get("candidates") or []
        if not isinstance(candidates, list):
            continue
        total = sum(1 for c in candidates if isinstance(c, dict))
        bid = str(batch.get("batch_id") or "")
        done = int(progress_by_batch.get(bid, 0))
        pending += max(0, total - done)
    return pending


def exit_normal_framework_pr(
    state: Any,
    *,
    max_hours: float | None = None,
    now_unix: float | None = None,
    force_exit_hours_remaining_ratio: float = (
        DEFAULT_FRAMEWORK_PR_FORCE_EXIT_HOURS_REMAINING_RATIO
    ),
) -> tuple[str, dict[str, Any]] | None:
    """FRAMEWORK_PR normal exit.

    Priority: 0. HARD force-exit when remaining < ratio*max_hours →
    ``framework_pr_force_exit_low_budget``; 1. ``framework_pr_phase_done``; else ``None``.
    """
    if max_hours and max_hours > 0:
        remaining_min_fn = getattr(state, "remaining_minutes", None)
        if callable(remaining_min_fn):
            try:
                remaining_minutes = float(remaining_min_fn(now_unix=now_unix))
            except TypeError:
                remaining_minutes = float(remaining_min_fn())
            except Exception:  # noqa: BLE001
                remaining_minutes = float("inf")
        else:
            remaining_minutes = float("inf")
        threshold_minutes = float(force_exit_hours_remaining_ratio) * float(max_hours) * 60.0
        if remaining_minutes < threshold_minutes:
            return "framework_pr_force_exit_low_budget", {
                "evidence":               "force_exit",
                "remaining_minutes":      remaining_minutes,
                "threshold_minutes":      threshold_minutes,
                "hours_remaining_ratio":  float(force_exit_hours_remaining_ratio),
                "max_hours":              float(max_hours),
                "pending_candidate_count": _framework_pr_pending_candidate_count(state),
            }

    batches = getattr(state, "framework_pr_batches", None) or []
    if bool(getattr(state, "framework_pr_phase_done", False)):
        return "framework_pr_phase_done", {
            "evidence": "no_more_candidates",
            "batch_count": len(batches) if isinstance(batches, list) else 0,
        }

    return None


def _post_prelude_target(*, explore_enabled: bool, kernel_enabled: bool) -> str:
    """First active phase after PRELUDE / FRAMEWORK_PR: EXPLORE, else KERNEL,
    else SWEEP (``--no-explore`` / ``--no-kernel`` collapse the chain)."""
    if explore_enabled:
        return PHASE_EXPLORE
    if kernel_enabled:
        return PHASE_KERNEL
    return PHASE_SWEEP


def compute_next_phase(
    state: Any,
    *,
    kernel_enabled: bool = True,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
    framework_phase_enabled: bool = False,
    explore_enabled: bool = True,
    max_hours: float | None = None,
) -> tuple[str, str, dict[str, Any]] | None:
    """Return ``(next_phase, reason, evidence)`` or ``None``.

    Priority (Inv-8.2 + §3.8 §7.1): global terminal first, then abort > exit_terminal > exit_normal.
    """
    current = (getattr(state, "phase", "") or "").strip().upper() or PHASE_PRELUDE
    overrides = _resolve_plateau_overrides(state)

    # Global terminal stop_reason overrides phase-local judgments.
    terminal = _global_terminal(state)
    if terminal is not None and current != PHASE_CLOSE:
        reason, evidence = terminal
        return PHASE_CLOSE, reason, {"terminal": True, **evidence}

    if current == PHASE_PRELUDE:
        ab = abort_prelude(state)
        if ab is not None:
            return PHASE_CLOSE, ab[0], {"terminal": True, **ab[1]}
        term = exit_terminal_prelude(state)
        if term is not None:
            return PHASE_CLOSE, term[0], {"terminal": True, **term[1]}
        norm = exit_normal_prelude(state)
        if norm is not None:
            if framework_phase_enabled:
                return PHASE_FRAMEWORK_PR, norm[0], norm[1]
            # Framework off → first active phase; ``explore_skipped`` stamped
            # when EXPLORE is bypassed.
            target = _post_prelude_target(
                explore_enabled=explore_enabled, kernel_enabled=kernel_enabled,
            )
            evidence = dict(norm[1])
            if target != PHASE_EXPLORE:
                evidence["explore_skipped"] = True
            return target, norm[0], evidence
        return None

    if current == PHASE_FRAMEWORK_PR:
        norm = exit_normal_framework_pr(
            state,
            max_hours=max_hours,
            now_unix=now_unix,
            force_exit_hours_remaining_ratio=float(overrides.get(
                "framework_pr_force_exit_hours_ratio",
                DEFAULT_FRAMEWORK_PR_FORCE_EXIT_HOURS_REMAINING_RATIO,
            )),
        )
        if norm is not None:
            # FRAMEWORK_PR → EXPLORE (or KERNEL/SWEEP when collapsed).
            target = _post_prelude_target(
                explore_enabled=explore_enabled, kernel_enabled=kernel_enabled,
            )
            evidence = dict(norm[1])
            if target != PHASE_EXPLORE:
                evidence["explore_skipped"] = True
            return target, norm[0], evidence
        return None

    if current == PHASE_EXPLORE:
        norm = exit_normal_explore(
            state,
            budget_pct=budget_pct,
            now_unix=now_unix,
            force_exit_hours_remaining=float(overrides.get(
                "force_exit_hours_remaining",
                DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING,
            )),
            force_exit_budget_pct=float(overrides.get(
                "force_exit_budget_pct",
                DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT,
            )),
        )
        if norm is not None:
            # Non-terminal "no more leverage" → wind down to SWEEP,
            # skipping the KERNEL hop.
            if norm[0] == "no_more_leverage":
                return PHASE_SWEEP, norm[0], norm[1]
            if kernel_enabled:
                return PHASE_KERNEL, norm[0], norm[1]
            return PHASE_SWEEP, "no_kernel_skipped", {
                "passed_through_reason": norm[0], **norm[1],
            }
        return None

    if current == PHASE_KERNEL:
        norm = exit_normal_kernel(
            state,
            budget_pct=budget_pct,
            now_unix=now_unix,
        )
        if norm is not None:
            return PHASE_SWEEP, norm[0], norm[1]
        return None

    if current == PHASE_SWEEP:
        norm = exit_normal_sweep(state, budget_pct=budget_pct, now_unix=now_unix)
        if norm is not None:
            # R1: in cyclic mode, loop back to EXPLORE (a new macro-cycle)
            # while budget remains and the run hasn't globally converged (R7);
            # otherwise wind down to CLOSE (the monotonic-chain behaviour).
            reloop, reloop_ev = should_reloop_to_explore(state, now_unix=now_unix)
            if reloop and explore_enabled:
                return PHASE_EXPLORE, "cycle_reloop", {
                    **norm[1], **reloop_ev, "loopback": True,
                }
            # R7: if cyclic looping was blocked because leverage is exhausted
            # (global convergence) or the safety cap is hit, terminate the run
            # with a terminal stop_reason instead of idling in CLOSE until the
            # deadline. ``insufficient_remaining`` defers to the run-loop
            # deadline (non-terminal CLOSE), matching the monotonic chain.
            blocked = str(reloop_ev.get("reloop_blocked") or "")
            if blocked in ("global_converged", "max_cycles"):
                return PHASE_CLOSE, "global_converged", {
                    **norm[1], **reloop_ev, "terminal": True,
                }
            return PHASE_CLOSE, norm[0], {**norm[1], **reloop_ev}
        return None

    # PHASE_CLOSE — terminal, no further transitions.
    return None


# phase_history helper (shape used by SharedState.record_phase_transition)
def make_history_row(
    *,
    from_phase: str,
    to_phase: str,
    reason: str,
    evidence: dict[str, Any] | None,
    ts: str,
    ts_unix: float,
    cycle: int = 0,
) -> dict[str, Any]:
    """Construct a canonical phase_history row (Inv-2.2 + KB_design §3.2 §6); ``reason`` unvalidated for resume tools.

    ``cycle`` stamps the R1 macro-cycle this transition belongs to (0 for the
    first pass / legacy non-cyclic runs).
    """
    return {
        "from_phase": (from_phase or "").strip().upper(),
        "to_phase":   (to_phase or "").strip().upper(),
        "reason":     (reason or "").strip(),
        "evidence":   dict(evidence or {}),
        "ts":         ts,
        "ts_unix":    float(ts_unix or 0.0),
        "cycle":      int(cycle or 0),
    }


__all__ = [
    "DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT",
    "DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT_INTERLEAVE",
    "DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING",
    "DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING_INTERLEAVE",
    "DEFAULT_PHASE_BUDGET_PCT",
    "DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK",
    "DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT",
    "DEFAULT_PLATEAU_EXPLORE_LOOKBACK",
    "DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT",
    "DEFAULT_PLATEAU_KERNEL_LOOKBACK",
    "DEFAULT_PLATEAU_KERNEL_REVERT_STREAK",
    "ESCALATE_HINT_BUDGET_BUMP_CAP",
    "ESCALATE_HINT_BUDGET_BUMP_DELTA",
    "ESCALATE_HINT_EXTEND_EXPLORE_BUDGET",
    "ESCALATE_HINT_EXTEND_KERNEL_BUDGET",
    "ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX",
    "ESCALATE_HINT_SKIP_TO_CLOSE",
    "ESCALATE_HINT_SKIP_TO_KERNEL",
    "ESCALATE_HINT_SKIP_TO_SWEEP",
    "ESCALATE_HINT_VOCAB",
    "PHASE_ALLOWED_ACTIONS",
    "PHASE_INTERLEAVE_ENV",
    "PHASE_LLM_PROPOSABLE_ACTIONS",
    "PHASE_CLOSE",
    "PHASE_EXIT_REASONS",
    "PHASE_EXPLORE",
    "PHASE_FRAMEWORK_PR",
    "PHASE_INDEX",
    "PHASE_KERNEL",
    "PHASE_NAMES",
    "PHASE_PRELUDE",
    "PHASE_SWEEP",
    "STOP_REASON_VOCAB",
    "DEFAULT_FRAMEWORK_PR_PLATEAU_LOOKBACK",
    "DEFAULT_FRAMEWORK_PR_PLATEAU_KEEP_GAIN_PCT",
    "DEFAULT_FRAMEWORK_PR_FORCE_EXIT_HOURS_REMAINING_RATIO",
    "DEFAULT_MAX_MACRO_CYCLES",
    "DEFAULT_CYCLE_RELOOP_MIN_REMAINING_SEC",
    "DEFAULT_GLOBAL_CONVERGENCE_NO_GAIN_CYCLES",
    "DEFAULT_CYCLE_MIN_GAIN_PCT",
    "PHASE_CYCLIC_ENV",
    "is_cyclic_phases_enabled",
    "should_reloop_to_explore",
    "abort_prelude",
    "allowed_actions_for",
    "apply_escalate_budget_bump",
    "compute_next_phase",
    "compute_plateau_explore",
    "compute_plateau_framework_pr",
    "compute_plateau_kernel",
    "exit_normal_explore",
    "exit_normal_framework_pr",
    "exit_normal_kernel",
    "exit_normal_prelude",
    "exit_normal_sweep",
    "exit_terminal_prelude",
    "is_action_allowed_in_phase",
    "is_action_llm_proposable_in_phase",
    "is_action_llm_proposable_in_phase_with_interleave",
    "is_phase_interleave_enabled",
    "llm_proposable_actions_for",
    "llm_proposable_actions_for_with_interleave",
    "is_pause_specialist_hint",
    "is_valid_escalate_hint",
    "is_valid_phase_exit_reason",
    "is_valid_stop_reason",
    "make_history_row",
    "normalize_budget_pct",
    "phase_budget_remaining_seconds",
    "phase_elapsed_seconds",
    "phase_index",
    "session_remaining_seconds",
    "should_force_exit_explore",
    "warm_replay_in_flight",
]
