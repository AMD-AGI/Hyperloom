# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase state machine."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from hyperloom.common.coerce import to_unix
from hyperloom.inference_optimizer.protocol.action_surfaces import (
    COORDINATOR_INTERNAL_ACTIONS,
    ROBUSTNESS_DELEGATE_ONLY_ACTIONS,
)


log = logging.getLogger(__name__)


# Phase identifiers + ordering (monotonic chain)
PHASE_PRELUDE = "PRELUDE"
PHASE_FRAMEWORK_AGENT = "FRAMEWORK_AGENT"
PHASE_KERNEL_AGENT = "KERNEL_AGENT"
PHASE_SWEEP = "SWEEP"
PHASE_CLOSE = "CLOSE"

PHASE_NAMES: tuple[str, ...] = (
    PHASE_PRELUDE,
    PHASE_FRAMEWORK_AGENT,
    PHASE_KERNEL_AGENT,
    PHASE_SWEEP,
    PHASE_CLOSE,
)
PHASE_INDEX: dict[str, int] = {name: i for i, name in enumerate(PHASE_NAMES)}


def phase_index(phase: str) -> int:
    """Return monotonic index of ``phase`` (Inv-2.1 check); unknown → -1."""
    return PHASE_INDEX.get((phase or "").strip().upper(), -1)


# Phase ↔ allowed action set: ALLOWED passes R1; Coordinator-auto actions stay out of PROPOSABLE so LLM proposals are
# denied.
PHASE_ALLOWED_ACTIONS: dict[str, frozenset[str]] = {
    PHASE_PRELUDE: frozenset(
        {
            "target_analysis",
            "baseline",
            "roofline",
            "profile",
            "recover",
        }
    ),
    # Three levers: configuration grids (``explore``), investigation and authoring (``specialist``), and landing a
    # patch from any source (``integrate_patch``).
    PHASE_FRAMEWORK_AGENT: frozenset(
        {
            "explore",
            "specialist",
            "integrate_patch",
            # roofline/profile auto-enqueued on the cumulative-gain watermark.
            "roofline",
            "profile",
            "recover",
        }
    ),
    # No kernel_opt or gemm_tuning: the Coordinator dispatches both once at phase entry, so an LLM re-issuing them per
    # tick would bypass the lane budget they are derived from.
    PHASE_KERNEL_AGENT: frozenset(
        {
            "integrate",
            "specialist",
            "roofline",
            "profile",
            "recover",
        }
    ),
    # No specialist below: SWEEP is the validation window and CLOSE only reports.
    PHASE_SWEEP: frozenset(
        {
            # conc_sweep: Coordinator-internal CONC-ladder benchmark.
            "conc_sweep",
            "recover",
        }
    ),
    PHASE_CLOSE: frozenset(
        {
            "report",
            "session_breakdown",
            "recover",
        }
    ),
}


# Dispatched by the Coordinator or owned by the Robustness ladder.
_NOT_LLM_PROPOSABLE: frozenset[str] = COORDINATOR_INTERNAL_ACTIONS | ROBUSTNESS_DELEGATE_ONLY_ACTIONS


# Task kinds that mean the KERNEL lane is busy, which is a wider question than what a model may propose: a
# Coordinator-owned lane is dispatched without ever being proposable, and its task occupies the phase just the same.
KERNEL_LANE_TASK_KINDS: frozenset[str] = PHASE_ALLOWED_ACTIONS[PHASE_KERNEL_AGENT] | frozenset(
    {
        "kernel_opt",
        "gemm_tuning",
    }
)


def _action_in_phase_map(action_name: str, phase: str, mapping: dict[str, frozenset[str]]) -> bool:
    """Return True iff stripped ``action_name`` is a member of ``mapping[phase]`` (unknown phase → deny)."""
    actions = mapping.get((phase or "").strip().upper())
    if actions is None:
        return False
    return (action_name or "").strip() in actions


def is_action_allowed_in_phase(action_name: str, phase: str) -> bool:
    """Return True iff ``action_name`` is in the phase allowlist (R1; unknown phase → deny)."""
    return _action_in_phase_map(action_name, phase, PHASE_ALLOWED_ACTIONS)


def allowed_actions_for(phase: str) -> tuple[str, ...]:
    """Return the phase's LLM-proposable actions as a sorted tuple (deterministic)."""
    actions = PHASE_ALLOWED_ACTIONS.get((phase or "").strip().upper(), frozenset())
    return tuple(sorted(actions - _NOT_LLM_PROPOSABLE))


def render_phase_action_bullets(
    *,
    disabled_suffix: dict[str, str] | None = None,
) -> list[str]:
    """Render per-phase action bullets for the prompt (informational, not enforced)."""
    suffix = disabled_suffix or {}
    out: list[str] = []
    for phase in PHASE_NAMES:
        actions = allowed_actions_for(phase)
        flag = suffix.get(phase)
        if flag:
            out.append(f"- **{phase}**: {', '.join(actions)} (DISABLED: {flag} — phase skipped)")
        else:
            out.append(f"- **{phase}**: {', '.join(actions)}")
    return out


# phase_exit_reasons vocab
PHASE_EXIT_REASONS: frozenset[str] = frozenset(
    {
        # Normal exits
        "prelude_done",
        "plateau_explore",
        "plateau_kernel",
        "optimize_phase_budget_exhausted",
        "kernel_phase_budget_exhausted",
        "optimize_budget_cap",  # OPTIMIZE → next phase at the absolute per-phase wall-clock cap
        "kernel_budget_cap",  # KERNEL_AGENT → SWEEP at the absolute per-phase wall-clock cap
        "kernel_controller_done",  # KERNEL_AGENT → SWEEP after the phase-level rewrite controller
        "sweep_budget_cap",  # SWEEP → reloop/CLOSE at the absolute per-phase wall-clock cap
        "sweep_done",  # SWEEP → CLOSE when the concurrency ladder settles
        "sweep_failed",  # SWEEP → CLOSE when the ladder reaches a failed terminal result
        "sweep_budget_exhausted",
        "no_kernel_skipped",  # FRAMEWORK_AGENT → SWEEP when kernel disabled
        "kernel_phase_aborted_no_trace",  # KERNEL_AGENT → SWEEP when profile fails
        "optimize_no_more_leverage",  # OPTIMIZE → KERNEL_AGENT (non-terminal): both arms plateaued, or skip_to_sweep
        "kernel_no_more_leverage",  # KERNEL_AGENT → SWEEP (non-terminal) via skip_to_sweep
        # Cyclic phase machine back-edge reasons (transitions that reopen a macro-cycle).
        "cycle_reloop",  # SWEEP → FRAMEWORK_AGENT; opens a new macro-cycle while budget + leverage remain
        "global_converged",  # SWEEP → CLOSE; cyclic leverage exhausted across macro-cycles (also a terminal stop_reason)
        # Terminal exits (any phase → CLOSE)
        "robustness_escalated",
        # A phase after PRELUDE → SWEEP on the way in, SWEEP → CLOSE on the way out.
        "target_reached",
        "time_exhausted",
        "time_exhausted_during_prelude",
        "user_stop_requested",
        "recipe_kb_t0_failed",
        "recipe_kb_drain_failed",
        "recipe_kb_commit_failed",
        "prelude_baseline_failed",
        "prelude_cold_anchor_low_budget",  # PRELUDE → CLOSE; only a cold anchor, nothing comparable to it affordable
        "prelude_policy_loop",
        "policy_loop",
        "crash_threshold_exceeded",
        "baseline_failed",  # live baseline-failure marker
        "emergency",
        "max_ticks",
        "signal",
        # Construction sentinel — first phase_history entry on fresh session.
        "phase_entered",
        # Marker row: the source arm has nothing left to dispatch.
        "no_candidates_and_discovery_exhausted",
    }
)


#: Named rather than inlined below because the writeback gate that sets it lives
#: in another module, and the vocabulary is closed -- PolicyGate rejects any
#: stop_reason outside it, so a typo on either side would silently degrade into
#: "the run did not stop" rather than into an error anyone sees.
AGENTX_PREFLIGHT_STOP_REASON: str = "agentx_client_unavailable"


# stop_reason vocab
STOP_REASON_VOCAB: frozenset[str] = frozenset(
    {
        # Legacy sentinels — kept for backward compat (resume from old sessions).
        "target_reached",
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
        "prelude_cold_anchor_low_budget",
        "prelude_policy_loop",
        "time_exhausted_during_prelude",
        "recipe_kb_t0_failed",
        "recipe_kb_drain_failed",
        "recipe_kb_commit_failed",
        "warm_replay_rollback_failed",
        "active_inferencex_checkout_missing",
        "plateau_explore",
        "plateau_kernel",
        "no_kernel_skipped",
        "sweep_done",
        "sweep_failed",
        "framework_agent_phase_done",
        "framework_agent_plateau",
        # R7: cyclic phase machine exhausted leverage across macro-cycles.
        "global_converged",
        # Context-window preflight: max_position_embeddings can't hold ISL+OSL.
        "model_context_window_too_small",
        # Model-arch preflight: multimodal/vision model unsupported.
        "unsupported_model_arch",
        # Pre-run model-config compatibility preflight: config.json is corrupt or declares RoPE scaling without a
        # max-position field (both crash at load).
        "model_config_incompatible",
        # Baseline arg-validation fast-exit: >=2 consecutive baseline attempts exited <30s on a bad CLI arg.
        "baseline_arg_error",
        # Enablement gave up without a booting baseline: a revalidation the
        # round depended on never promoted.
        "enablement_stalled",
        # Enablement attempt cap: too many consecutive rounds bought no ground.
        # A bring-up that is still advancing is bounded by the run's wall clock.
        "enablement_attempts_exhausted",
        # The baseline could not produce an accuracy result even though the
        # accuracy test was expected to run (broken eval / missing quality
        # gate). Optimizing against an unvalidated baseline is unsafe, so the
        # run halts. Post-baseline accuracy failures REVERT the offending
        # change instead of stopping.
        "baseline_accuracy_failed",
        # Bring-up terminals: the host cannot run the combo, or the harness
        # composed an argument the installed parser does not have. Classified as
        # infrastructure by ``INFRASTRUCTURE_STOP_REASONS``.
        "environment_fault",
        "server_argv_invalid",
        # A bring-up round expired with nothing confirming its holder dead, so
        # it keeps excluding the machine.
        # The out-of-band supervisor found the coordinator's process gone; it
        # reaches a report through the terminal artifact the supervisor writes.
        "supervisor_coordinator_died",
        # The out-of-band supervisor found the tick not advancing and the
        # coordinator did not answer the stop it was sent; it reaches a report
        # through the terminal artifact the supervisor writes.
        "supervisor_tick_stalled",
        # AgentX is on but its benchmark client (aiperf) is missing or is not
        # the pinned build, and the runtime install could not supply it. An
        # environment/supply gap, not a code gap: nothing downstream can author
        # its way out of it, so the run halts on the FIRST occurrence instead of
        # spending the budget in the enablement lane.
        AGENTX_PREFLIGHT_STOP_REASON,
    }
)


def is_valid_stop_reason(value: str) -> bool:
    """Return True when ``value`` is a member of :data:`STOP_REASON_VOCAB`."""
    return (value or "").strip() in STOP_REASON_VOCAB


def is_valid_phase_exit_reason(value: str) -> bool:
    """Return True when ``value`` is a member of :data:`PHASE_EXIT_REASONS`."""
    return (value or "").strip() in PHASE_EXIT_REASONS


# Default phase budgets (% of wall-clock).
DEFAULT_PHASE_BUDGET_PCT: dict[str, float] = {
    PHASE_PRELUDE: 0.03,
    # The optimisation phase carries both levers' share.
    PHASE_FRAMEWORK_AGENT: 0.40,
    PHASE_KERNEL_AGENT: 0.50,
    PHASE_SWEEP: 0.05,
    PHASE_CLOSE: 0.02,
}

# Share of the session held back for the phases that actually produce a result.
OPTIMIZATION_RESERVE_PCT: float = 0.50

# Wall-clock ceiling for an unbounded run (``max_minutes`` == 0): the container lifetime.
DEFAULT_LONGRUN_MAX_MINUTES: int = 14 * 24 * 60
# Reference window the absolute per-phase cap applies its budget fraction to.
PHASE_ABSOLUTE_CAP_REFERENCE_MINUTES: int = 24 * 60


# Plateau judgment defaults (CLI --plateau-* flags); kept here for pure callers + tests.
DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT: float = 0.5
DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK: int = 5
DEFAULT_PLATEAU_EXPLORE_LOOKBACK: int = 5
DEFAULT_PLATEAU_KERNEL_REVERT_STREAK: int = 3
DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT: float = 0.5
DEFAULT_PLATEAU_KERNEL_LOOKBACK: int = 5


import os as _os_env  # noqa: E402

# FRAMEWORK per-candidate plateau: after this many consecutive resolved candidates without a KEEP (including
# non-benchmarked terminal outcomes), the source arm is dry.
DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK: int = 5


# R1 macro-cycle reloop: SWEEP loops back to FRAMEWORK_AGENT for a new macro-cycle while budget remains and the run
# hasn't globally converged.

# Safety ceiling on macro-cycles (defense against a pathological tight loop).
DEFAULT_MAX_MACRO_CYCLES: int = 1000

# Share of a bounded session's total budget that must remain to open a cycle.
_CYCLE_RELOOP_BUDGET_RATIO: float = 0.15

# Ceiling on the floor once it is raised to cover one granted variant round, so a
# session too short to fund a round is not treated as exhausted from tick one.
_CYCLE_RELOOP_MAX_BUDGET_SHARE: float = 0.5


def _default_cycle_reloop_min_remaining_sec() -> float:
    """Absolute reloop floor in seconds; env-overridable via ``INFERENCE_OPTIMIZER_CYCLE_RELOOP_MIN_REMAINING_SEC``."""
    raw = (_os_env.environ.get("INFERENCE_OPTIMIZER_CYCLE_RELOOP_MIN_REMAINING_SEC", "") or "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass  # malformed env override; fall through to the 3 h default
    return 10800.0


# Minimum session wall-clock (seconds) that must remain to justify opening a new macro-cycle; below this we wind down
# to CLOSE instead of starting a cycle we cannot meaningfully use.
DEFAULT_CYCLE_RELOOP_MIN_REMAINING_SEC: float = _default_cycle_reloop_min_remaining_sec()

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

    cycle = int(getattr(state, "macro_cycle", 0) or 0)
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
        return float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def target_was_reached(state: Any) -> bool:
    """Whether the run objective has been met."""
    return bool(str(getattr(state, "target_reached_at", "") or "").strip())


def _one_variant_grant_sec(state: Any) -> float:
    """Seconds a single variant round is actually granted, for budget arithmetic.

    Prices the round the way the sweep's admission check does rather than at the
    declared timeout, so both sides agree on what a cycle costs.

    Args:
        state (Any): Frozen SharedState view exposing the declared variant timeout.

    Returns:
        float: The granted per-variant cap in seconds, or ``0.0`` when unknown.
    """
    declared = getattr(state, "conc_sweep_variant_timeout_sec", 0) or 0
    try:
        declared_sec = int(declared)
    except (TypeError, ValueError):
        return 0.0
    if declared_sec <= 0:
        return 0.0
    try:
        from hyperloom.orchestrator.actions.executors._grid_runner import agentx_variant_timeout_sec
    except ImportError:  # grid runner unavailable; price at the declared timeout
        return float(declared_sec)
    return float(agentx_variant_timeout_sec(declared_sec, shared_state=state))


def _cycle_reloop_min_remaining_sec(
    state: Any,
    min_remaining_sec: float = DEFAULT_CYCLE_RELOOP_MIN_REMAINING_SEC,
) -> float:
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
        min_remaining_sec (float): Absolute floor before session scaling.

    Returns:
        float: The effective floor in seconds.
    """
    effective = float(min_remaining_sec)
    max_minutes = _max_minutes(state)
    if max_minutes > 0:
        budget_sec = max_minutes * 60.0
        effective = min(effective, budget_sec * _CYCLE_RELOOP_BUDGET_RATIO)
        grant = min(_one_variant_grant_sec(state), budget_sec * _CYCLE_RELOOP_MAX_BUDGET_SHARE)
        effective = max(effective, grant)
    return effective


def should_reloop_to_explore(
    state: Any,
    *,
    now_unix: float | None = None,
    max_cycles: int = DEFAULT_MAX_MACRO_CYCLES,
    min_remaining_sec: float = DEFAULT_CYCLE_RELOOP_MIN_REMAINING_SEC,
    no_gain_cycles: int = DEFAULT_GLOBAL_CONVERGENCE_NO_GAIN_CYCLES,
    min_gain_pct: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Decide whether SWEEP should open a new macro-cycle (R1) or wind to CLOSE."""
    cycle = int(getattr(state, "macro_cycle", 0) or 0)
    evidence: dict[str, Any] = {"macro_cycle": cycle}

    # Per-cycle gain since this cycle started → effective no-gain streak.
    effective_min_gain = decaying_keep_threshold_pct(cycle) if min_gain_pct is None else float(min_gain_pct)
    cur_gain = _cumulative_gain_validated(state)
    start_gain = float(getattr(state, "gain_at_cycle_start", 0.0) or 0.0)
    cycle_gained = (cur_gain - start_gain) > effective_min_gain
    evidence["min_gain_pct"] = round(effective_min_gain, 6)
    prior_streak = int(getattr(state, "no_gain_cycle_streak", 0) or 0)
    effective_streak = 0 if cycle_gained else prior_streak + 1
    evidence["cycle_gain_delta"] = round(cur_gain - start_gain, 6)
    evidence["cycle_gained"] = cycle_gained
    evidence["no_gain_cycle_streak_effective"] = effective_streak

    if target_was_reached(state):
        evidence["reloop_blocked"] = "target_reached"
        return False, evidence

    # Safety cap on macro-cycles.
    if (cycle + 1) >= int(max_cycles):
        evidence["reloop_blocked"] = "max_cycles"
        return False, evidence

    # Physical ceiling convergence: if every roofline family that dominated is now within its saturation threshold,
    # stop cleanly.
    sat = getattr(state, "saturated_directions", {}) or {}
    if isinstance(sat, dict) and sat:
        rows = [v for v in sat.values() if isinstance(v, dict)]
        if rows and all(bool(v.get("saturated")) for v in rows):
            evidence["reloop_blocked"] = "all_directions_saturated"
            evidence["saturated_directions"] = sorted(str(k) for k in sat.keys())
            return False, evidence

    # R7 global convergence.
    if effective_streak >= int(no_gain_cycles):
        evidence["reloop_blocked"] = "global_converged"
        return False, evidence

    # Require a session-scaled floor that can still fund one variant round.
    effective_min_remaining = _cycle_reloop_min_remaining_sec(state, min_remaining_sec)
    evidence["min_remaining_sec_effective"] = round(effective_min_remaining, 2)
    remaining = session_remaining_seconds(state, now_unix=now_unix)
    if remaining is not None and remaining < effective_min_remaining:
        evidence["reloop_blocked"] = "insufficient_remaining"
        evidence["session_remaining_seconds"] = round(remaining, 2)
        return False, evidence

    evidence["reloop"] = True
    evidence["next_cycle"] = cycle + 1
    return True, evidence


# escalate_strategy_change hint vocabulary (closed enum; unknown hints ignored).
ESCALATE_HINT_SKIP_TO_KERNEL: str = "skip_to_kernel"
ESCALATE_HINT_SKIP_TO_SWEEP: str = "skip_to_sweep"
ESCALATE_HINT_SKIP_TO_CLOSE: str = "skip_to_close"


def _kernel_idle_max_ticks() -> int:
    """Consecutive no-work KERNEL_AGENT ticks before winding down to SWEEP."""
    raw = (_os_env.environ.get("INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS", "") or "").strip()
    try:
        val = int(raw)
        return val if val >= 1 else 3
    except (TypeError, ValueError):
        return 3


KERNEL_IDLE_MAX_TICKS: int = _kernel_idle_max_ticks()


def _kernel_idle_min_seconds() -> float:
    """Wall-clock seconds a KERNEL idle streak must last before winding down."""
    raw = (_os_env.environ.get("INFERENCE_OPTIMIZER_KERNEL_IDLE_MIN_SECONDS", "") or "").strip()
    try:
        val = float(raw)
        return val if val > 0.0 else 600.0
    except (TypeError, ValueError):
        return 600.0


KERNEL_IDLE_MIN_SECONDS: float = _kernel_idle_min_seconds()

#: How often the intent router refreshes the inline-step liveness stamp.
KERNEL_HEARTBEAT_SEC: float = 150.0

#: How stale ``kernel_inline_step_seen_unix`` may be and still mean "running".
#: Three heartbeat intervals absorb a late beat under load; a stamp orphaned by a
#: process that died mid-step expires shortly after rather than muting the guard.
KERNEL_INLINE_STEP_STALE_SECONDS: float = 3.0 * KERNEL_HEARTBEAT_SEC


def kernel_inline_step_running(state: Any, *, now_unix: float | None = None) -> bool:
    """Report whether an inline kernel request is executing right now."""
    seen = getattr(state, "kernel_inline_step_seen_unix", 0.0)
    try:
        seen = float(seen or 0.0)
    except (TypeError, ValueError):
        return False
    if seen <= 0.0:
        return False
    now = float(now_unix if now_unix is not None else _now_unix(state))
    return 0.0 <= (now - seen) <= KERNEL_INLINE_STEP_STALE_SECONDS


ESCALATE_HINT_EXTEND_EXPLORE_BUDGET: str = "extend_explore_budget"
ESCALATE_HINT_EXTEND_KERNEL_BUDGET: str = "extend_kernel_budget"

# ``skip_to_sweep`` is the non-terminal "exhausted the current lever" signal: from FRAMEWORK_AGENT it advances to
# KERNEL, from KERNEL it winds down to SWEEP → CLOSE.
ESCALATE_HINT_VOCAB: frozenset[str] = frozenset(
    {
        ESCALATE_HINT_SKIP_TO_KERNEL,
        ESCALATE_HINT_SKIP_TO_SWEEP,
        ESCALATE_HINT_SKIP_TO_CLOSE,
        ESCALATE_HINT_EXTEND_EXPLORE_BUDGET,
        ESCALATE_HINT_EXTEND_KERNEL_BUDGET,
    }
)

# ``extend_*_budget`` hints raise a phase budget by DELTA up to CAP.
ESCALATE_HINT_BUDGET_BUMP_DELTA: float = 0.05  # +5 percentage points per hint
ESCALATE_HINT_BUDGET_BUMP_CAP: float = 0.80  # absolute ceiling


def is_valid_escalate_hint(hint: str) -> bool:
    """Return True for any hint Coordinator should act on (closed vocab)."""
    return (hint or "").strip() in ESCALATE_HINT_VOCAB


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
    """Move disabled work-phase shares to enabled work phases.

    FRAMEWORK_AGENT, KERNEL_AGENT, and SWEEP absorb proportionally, capped at
    1.0; PRELUDE and CLOSE never absorb.
    """
    out = dict(base)
    disabled: list[str] = []
    if not optimize_enabled:
        disabled.append(PHASE_FRAMEWORK_AGENT)
    if not kernel_enabled:
        disabled.append(PHASE_KERNEL_AGENT)
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
    else:
        # No weighted absorber left → park the freed share on SWEEP (always on).
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
    raw = getattr(state, "phase_started_unix", 0.0)
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _resume_boundary_unix(state: Any) -> float:
    """Return when the current run leg began, i.e. the most recent ``--resume``."""
    return max(0.0, to_unix(getattr(state, "resumed_ts", ""), 0.0) or 0.0)


def _kernel_idle_since_unix(state: Any) -> float:
    """Return when the current KERNEL idle streak opened, defensively coerced."""
    raw = getattr(state, "kernel_idle_since_unix", 0.0)
    try:
        return max(0.0, float(raw or 0.0))
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
    """Return the session's configured ``max_minutes`` budget, defensively coerced."""
    try:
        return float(getattr(state, "max_minutes", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _budget_minutes(state: Any) -> float:
    """Wall-clock minutes the PER-PHASE budget fractions apply to (R2)."""
    try:
        cm = float(getattr(state, "cycle_minutes", 0) or 0)
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


def is_phase_transition_row(row: Any) -> bool:
    """True when ``row`` records an actual phase change, not an in-phase marker."""
    if not isinstance(row, dict):
        return False
    to_phase = str(row.get("to_phase") or "").strip().upper()
    from_phase = str(row.get("from_phase") or "").strip().upper()
    return bool(to_phase) and to_phase != from_phase


def phase_history_event_name(row: Any) -> str:
    """Return a marker event name from either legacy or canonical history rows."""
    if not isinstance(row, dict):
        return ""
    legacy = str(row.get("event") or "").strip()
    if legacy:
        return legacy
    evidence = row.get("evidence")
    if isinstance(evidence, dict):
        nested = str(evidence.get("event") or "").strip()
        if nested:
            return nested
    return str(row.get("reason") or "").strip()


def phase_elapsed_totals_from_history(history: Any) -> dict[str, float]:
    """Rebuild per-phase completed-segment totals from a ``phase_history`` log."""
    if not isinstance(history, list):
        return {}
    rows = [row for row in history if isinstance(row, dict)]
    totals: dict[str, float] = {}
    transition_rows = [row for row in rows if is_phase_transition_row(row)]
    for idx in range(len(transition_rows) - 1):
        phase = str(transition_rows[idx].get("to_phase") or "").strip().upper()
        try:
            entered = float(transition_rows[idx].get("ts_unix") or 0.0)
            exited = float(transition_rows[idx + 1].get("ts_unix") or 0.0)
        except (TypeError, ValueError):
            continue
        if not phase or entered <= 0.0 or exited <= entered:
            continue
        totals[phase] = totals.get(phase, 0.0) + (exited - entered)
    return totals


def phase_cumulative_seconds(
    state: Any,
    *,
    phase: str | None = None,
    now_unix: float | None = None,
) -> float:
    """Return wall-clock seconds spent in ``phase``, summed over EVERY entry."""
    current = (getattr(state, "phase", "") or "").strip().upper()
    target = (phase or current or "").strip().upper()
    if not target:
        return 0.0
    accumulated = 0.0
    totals = getattr(state, "phase_elapsed_totals", None)
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


def explore_elapsed_seconds(state: Any, *, now_unix: float | None = None) -> float | None:
    """Return total optimisation-phase wall-clock seconds across all macro cycles."""
    raw_accumulated = getattr(state, "explore_elapsed_accum_s", 0.0)
    if raw_accumulated is None:
        return None
    try:
        accumulated = float(raw_accumulated or 0.0)
    except (TypeError, ValueError):
        return None
    if (getattr(state, "phase", "") or "").strip().upper() == PHASE_FRAMEWORK_AGENT:
        accumulated += phase_elapsed_seconds(state, now_unix=now_unix)
    return max(0.0, accumulated)


def _phase_budget_total_seconds(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
) -> float | None:
    """Effective TOTAL budget (seconds) allotted to the current phase."""
    budget = normalize_budget_pct(budget_pct or getattr(state, "phase_budget_pct", None))
    phase = (getattr(state, "phase", "") or "").strip().upper()
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
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
) -> float | None:
    """Return seconds left in the current phase ENTRY's budget (``None`` when budget window 0 = unlimited)."""
    total = _phase_budget_total_seconds(state, budget_pct=budget_pct, now_unix=now_unix)
    if total is None:
        return None
    return max(0.0, total - phase_elapsed_seconds(state, now_unix=now_unix))


def effective_max_minutes(state: Any) -> float:
    """Session minutes for deadline/cap math; unbounded runs use the 14-day ceiling."""
    mm = _max_minutes(state)
    return mm if mm > 0 else float(DEFAULT_LONGRUN_MAX_MINUTES)


def phase_cap_seconds(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
) -> float | None:
    """Absolute wall-clock ceiling (seconds) for the current phase."""
    budget = normalize_budget_pct(budget_pct or getattr(state, "phase_budget_pct", None))
    phase = (getattr(state, "phase", "") or "").upper()
    if phase not in budget:
        return None
    pct = float(budget[phase])
    if pct <= 0.0:
        # Zero fraction = no wall-clock allowed, as opposed to no cap at all.
        return 0.0
    proportional = effective_max_minutes(state) * 60.0 * pct
    abs_cap = math.ceil(PHASE_ABSOLUTE_CAP_REFERENCE_MINUTES * pct) * 60.0
    return float(min(proportional, abs_cap))


def phase_cap_exceeded(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
) -> bool:
    """True when time spent in the current phase has reached its absolute cap."""
    cap = phase_cap_seconds(state, budget_pct=budget_pct)
    if cap is None:
        return False
    return phase_cumulative_seconds(state, now_unix=now_unix) >= cap


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
        charged = max(0.0, float(getattr(state, "elapsed_charged_sec", 0.0) or 0.0))
        anchor = float(getattr(state, "leg_anchor_unix", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    now = float(now_unix) if now_unix is not None else time.time()
    if anchor > 0.0:
        return max(0.0, mm * 60.0 - (charged + max(0.0, now - anchor)))
    if charged > 0.0:
        return max(0.0, mm * 60.0 - charged)
    started = to_unix(str(getattr(state, "start_ts", "") or "").strip())
    if started is None:
        return None
    return max(0.0, mm * 60.0 - max(0.0, now - started))


# plateau pure functions
def _current_macro_cycle(state: Any) -> int:
    """Return the current macro-cycle index."""
    try:
        return int(getattr(state, "macro_cycle", 0) or 0)
    except (TypeError, ValueError):
        return 0


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
    cycle = _current_macro_cycle(state)
    return [row for row in dict_rows if _row_cycle(row) == cycle]


def compute_plateau_explore(
    state: Any,
    *,
    lookback: int = DEFAULT_PLATEAU_EXPLORE_LOOKBACK,
    keep_gain_threshold_pct: float = DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
    empty_streak_threshold: int = DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
) -> tuple[bool, dict[str, Any]]:
    """Real plateau_explore → ``(triggered, evidence)``."""
    if lookback <= 0:
        return False, {"reason": "lookback_disabled"}
    keep_gain_threshold_pct = float(keep_gain_threshold_pct or 0.0)
    empty_streak_threshold = int(empty_streak_threshold or 0)

    explore_search = getattr(state, "explore_search", None) or {}
    if not isinstance(explore_search, dict):
        explore_search = {}
    winners_history = _rows_for_current_cycle(explore_search.get("winners_history") or [], state)
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

    specialist_rounds = _rows_for_current_cycle(getattr(state, "specialist_rounds", None) or [], state)

    def _round_is_empty(row: Any) -> bool:
        """Return True when a specialist-round summary produced no work."""
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
                row.get("proposals_kept") if row.get("proposals_kept") is not None else row.get("kept_count") or 0,
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

    triggered = recent_keep_gain < keep_gain_threshold_pct and streak >= empty_streak_threshold
    return triggered, {
        "recent_keep_gain_pct": round(recent_keep_gain, 4),
        "keep_gain_threshold_pct": keep_gain_threshold_pct,
        "empty_streak": int(streak),
        "empty_streak_threshold": empty_streak_threshold,
        "lookback": int(lookback),
        "winners_seen": len(recent_winners),
        "specialist_rounds_seen": len(specialist_rounds),
    }


def compute_plateau_kernel(
    state: Any,
    *,
    lookback: int = DEFAULT_PLATEAU_KERNEL_LOOKBACK,
    revert_streak_threshold: int = DEFAULT_PLATEAU_KERNEL_REVERT_STREAK,
    keep_gain_threshold_pct: float = DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT,
) -> tuple[bool, dict[str, Any]]:
    """Real plateau_kernel → ``(triggered, evidence)``."""
    lookback = int(lookback or 0)
    revert_streak_threshold = int(revert_streak_threshold or 0)
    keep_gain_threshold_pct = float(keep_gain_threshold_pct or 0.0)
    if lookback <= 0 or revert_streak_threshold <= 0:
        return False, {"reason": "thresholds_disabled"}

    integ_attempts = getattr(state, "kernel_integrate_attempts", None) or {}
    if not isinstance(integ_attempts, dict):
        integ_attempts = {}

    # Flatten the integrate attempt log into a time-ordered list, take the last ``lookback`` rows.
    has_cycle = any(
        isinstance(attempt, dict) and "cycle" in attempt
        for entry in integ_attempts.values()
        if isinstance(entry, dict)
        for attempt in (entry.get("attempts") or [])
    )
    flat: list[tuple[str, str, float]] = []  # (decision, ts, gain_pct)
    for ent in integ_attempts.values():
        if not isinstance(ent, dict):
            continue
        for a in ent.get("attempts") or []:
            if not isinstance(a, dict):
                continue
            if has_cycle and _row_cycle(a) != _current_macro_cycle(state):
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
            "reason": "no_kernel_attempts_yet",
            "revert_streak_threshold": int(revert_streak_threshold),
            "keep_gain_threshold_pct": keep_gain_threshold_pct,
            "lookback": int(lookback),
            "attempts_seen": 0,
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

    triggered = revert_streak >= revert_streak_threshold or recent_keep_gain < keep_gain_threshold_pct
    return triggered, {
        "revert_streak": int(revert_streak),
        "revert_streak_threshold": int(revert_streak_threshold),
        "recent_keep_gain_pct": round(recent_keep_gain, 4),
        "keep_gain_threshold_pct": keep_gain_threshold_pct,
        "lookback": int(lookback),
        "attempts_seen": len(recent),
    }


# Let SWEEP's recorded closeout outrank an LLM skip_to_close hint.
_SWEEP_CLOSEOUT_STATUSES: frozenset[str] = frozenset({"succeeded", "partial", "completed", "skipped", "failed"})


def _sweep_has_recorded_closeout(state: Any) -> bool:
    """Whether SWEEP already recorded a result the phase machine can close on."""
    last_conc = getattr(state, "last_conc_sweep", None) or {}
    if isinstance(last_conc, dict):
        return str(last_conc.get("status") or "").lower() in _SWEEP_CLOSEOUT_STATUSES
    return False


# terminal / abort (global)
def _global_terminal(state: Any) -> tuple[str, dict[str, Any]] | None:
    """Return ``(stop_reason, evidence)`` for a phase-orthogonal stop.

    A recorded SWEEP closeout wins; otherwise skip_to_close maps to
    time_exhausted or robustness_escalated before the coordinator stop reason.
    """
    hint = _pending_escalate_hint(state)
    if hint == ESCALATE_HINT_SKIP_TO_CLOSE:
        current = (getattr(state, "phase", "") or "").strip().upper()
        if current == PHASE_SWEEP and _sweep_has_recorded_closeout(state):
            return None
        evidence: dict[str, Any] = {"evidence": "llm_escalation", "hint": hint}
        # The robustness label is only justified by a robustness signal; record the
        # crash count alongside the budget so the two can be told apart after the run.
        evidence["crash_count"] = int(getattr(state, "crash_count", 0) or 0)
        floor = _cycle_reloop_min_remaining_sec(state)
        evidence["min_remaining_sec_effective"] = round(floor, 2)
        remaining = session_remaining_seconds(state)
        if remaining is not None:
            evidence["session_remaining_seconds"] = round(remaining, 2)
            # Too little left for another cycle means the budget ran out; that is
            # the honest terminal, not a robustness abort.
            if remaining < floor:
                return "time_exhausted", evidence
        return "robustness_escalated", evidence
    sr = (getattr(state, "stop_reason", "") or "").strip()
    if sr:
        # Coordinator-set stop_reason takes precedence over phase exits.
        if not is_valid_stop_reason(sr):
            # Unknown values tolerated for resume parity.
            return sr, {"reason_origin": "shared_state.stop_reason", "vocab": "unknown"}
        return sr, {"reason_origin": "shared_state.stop_reason"}
    return None


def _closing_phase_terminal(state: Any) -> tuple[str, dict[str, Any]] | None:
    """Return a CLOSE stop when the wall-clock path has entered the closing phase."""
    if not bool(getattr(state, "closing_phase", False)):
        return None
    return "time_exhausted", {"reason_origin": "closing_phase"}


# per-phase judgments
def warm_replay_in_flight(state: Any) -> bool:
    """True while the PRELUDE warm-recipe replay task has not finished (PRELUDE must not exit until False — GPU contention)."""
    outcome = getattr(state, "warm_replay_outcome", None) or {}
    if not isinstance(outcome, dict):
        return False
    return str(outcome.get("status") or "").strip() == "in_flight"


def _kernel_opt_max_failures() -> int:
    """Resolve the kernel infra-failure retry budget (lazy import)."""
    from ..state.shared_state import resolve_kernel_opt_max_failures

    return resolve_kernel_opt_max_failures()


# The statuses a finished GEAK run writes to ``geak_result.status``.
GEAK_TERMINAL_STATUSES = frozenset(
    {
        "ok",
        "no_gain",
        "error",
        "failed",
        "skipped",
        "baseline_reproduction_failed",
    }
)


def _geak_phase_terminal(state: Any) -> bool:
    """Return true once the GEAK-owned KERNEL phase has produced a terminal result."""
    if str(getattr(state, "kernel_optimizer", "") or "").strip().lower() != "geak":
        return False
    result = getattr(state, "geak_result", None) or {}
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
    if str(getattr(state, "kernel_optimizer", "") or "").strip().lower() != "forge":
        return False
    result = getattr(state, "kernel_rewrite_controller_result", None) or {}
    if not isinstance(result, dict):
        return False
    try:
        result_cycle = int(result.get("macro_cycle", -1))
        current_cycle = int(getattr(state, "macro_cycle", 0) or 0)
    except (TypeError, ValueError):
        return False
    status = str(result.get("status") or "").strip().lower()
    return result_cycle == current_cycle and status in CONTROLLER_TERMINAL_STATUSES


# Ledger subfields that change when a kernel attempt actually advances.
_KERNEL_ATTEMPT_PROGRESS_FIELDS: tuple[str, ...] = (
    "current_kernel_id",
    "failure_count",
    "integration_status",
    "last_decision",
    "last_source_file",
    "last_status",
    "rejected_reason",
    "task_group_key",
)

# ``last_kernel_opt`` subfields that identify WHICH result is the latest one; a new result always changes at least one
# of them.
_LAST_KERNEL_OPT_PROGRESS_FIELDS: tuple[str, ...] = (
    "best_artifact_path",
    "decision",
    "kernel_id",
    "task_group_key",
    "ts",
)


def compute_kernel_progress_fingerprint(
    state: Any,
    *,
    inflight_task_ids: Any = (),
) -> str:
    """Digest the KERNEL signals that change if and only if something moved."""
    import hashlib
    import json

    attempts: list[list[str]] = []
    ledger = getattr(state, "kernel_opt_task_attempts", None)
    if isinstance(ledger, dict):
        for ledger_id, attempt in ledger.items():
            if not isinstance(attempt, dict):
                continue
            attempts.append(
                [str(ledger_id)] + [str(attempt.get(field, "")) for field in _KERNEL_ATTEMPT_PROGRESS_FIELDS]
            )
    attempts.sort()

    last_opt = getattr(state, "last_kernel_opt", None)
    last_opt = last_opt if isinstance(last_opt, dict) else {}
    stack = getattr(state, "optimization_stack", None)
    pending = getattr(state, "pending_kernel_integrations", None)
    controller = getattr(state, "kernel_rewrite_controller_result", None)
    controller = controller if isinstance(controller, dict) else {}
    payload = {
        "attempts": attempts,
        "inflight": sorted(str(task_id) for task_id in (inflight_task_ids or ())),
        "last_kernel_opt": [str(last_opt.get(field, "")) for field in _LAST_KERNEL_OPT_PROGRESS_FIELDS],
        "pending_integrations": sorted(str(key) for key in pending) if isinstance(pending, dict) else [],
        "rejected": sorted(str(kid) for kid in (getattr(state, "rejected_kernel_ids", None) or [])),
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
    try:
        if bool(getattr(state, "has_keep_pending_integrate", False)):
            return True
    except Exception:
        # Optional capability probe; treat a failure as 'not available'.
        pass

    if _controller_phase_terminal(state):
        return False
    if _geak_phase_terminal(state):
        result = getattr(state, "geak_result", None) or {}
        pending = getattr(state, "geak_pending", None) or {}
        if (
            isinstance(result, dict)
            and str(result.get("status") or "").strip().lower() == "ok"
            and isinstance(pending, dict)
            and str(pending.get("status") or "").strip().lower() == "awaiting_rebench"
            and bool(str(pending.get("revalidation_task_id") or "").strip())
        ):
            return True
        return False

    try:
        untried_hot = getattr(state, "untried_hot_reusable_kernels", None)
        if callable(untried_hot) and bool(untried_hot()):
            return True
    except Exception:
        # Optional capability probe; treat a failure as 'not available'.
        pass

    rejected = {str(x) for x in (getattr(state, "rejected_kernel_ids", None) or [])}
    integrated_entries: list[dict[str, Any]] = []
    integrated_sources: set[str] = set()
    for entry in getattr(state, "optimization_stack", None) or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("action") or "") == "integrate":
            integrated_entries.append(entry)
            source_file = str(entry.get("target_file") or entry.get("source_file") or "")
            if source_file:
                integrated_sources.add(source_file)

    attempts = getattr(state, "kernel_opt_task_attempts", None) or {}
    if not isinstance(attempts, dict):
        return False
    for ledger_id, attempt in attempts.items():
        if not isinstance(attempt, dict):
            continue
        kernel_id = str(attempt.get("current_kernel_id") or attempt.get("kernel_id") or ledger_id)
        source_file = str(attempt.get("last_source_file") or "")
        task_group_key = str(attempt.get("task_group_key") or "")
        integrated = False
        for integrated_entry in integrated_entries:
            integrated_key = str(integrated_entry.get("task_group_key") or "")
            if task_group_key and integrated_key:
                integrated = task_group_key == integrated_key
            else:
                if str(integrated_entry.get("kernel_id") or "") != kernel_id:
                    continue
                integrated_source = str(
                    integrated_entry.get("target_file") or integrated_entry.get("source_file") or ""
                )
                integrated = not source_file or not integrated_source or source_file == integrated_source
            if integrated:
                break
        if integrated:
            continue
        if source_file and source_file in integrated_sources:
            continue
        decision = str(attempt.get("last_decision") or "").strip().upper()
        status = str(attempt.get("last_status") or "").strip().lower()
        rejected_reason = str(attempt.get("rejected_reason") or "").strip()
        integration_status = str(attempt.get("integration_status") or "").strip().lower()
        if integration_status in {"integrated", "rejected"}:
            continue
        if kernel_id in rejected and (not task_group_key or rejected_reason):
            continue
        if decision == "KEEP":
            return True
        if decision == "REVERT" or rejected_reason:
            continue
        if status == "failed":
            try:
                failure_count = int(attempt.get("failure_count") or 0)
            except (TypeError, ValueError):
                failure_count = 0
            if 0 < failure_count < _kernel_opt_max_failures():
                return True
            continue
        if decision in ("", "PARTIAL", "NEEDS_REVIEW"):
            return True
    return False


def exit_normal_prelude(state: Any) -> tuple[str, dict[str, Any]] | None:
    """``baseline_tput > 0`` and warm-replay settled → ``prelude_done`` (else ``None``)."""
    if warm_replay_in_flight(state):
        return None
    if bool(getattr(state, "baseline_measure_round_dropped", False)):
        return None
    try:
        tput = float(getattr(state, "baseline_tput", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if tput > 0.0:
        return "prelude_done", {"baseline_tput": tput, **prelude_exit_viability(state)}
    return None


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


def prelude_exit_viability(state: Any) -> dict[str, Any]:
    """Report whether the budget PRELUDE leaves behind can still fund one optimization round."""
    usable = session_usable_seconds(state)
    round_sec = one_more_measurement_sec(state)
    priced_by = "boot_plus_benchmark"
    if round_sec is None:
        round_sec = measured_seconds(state, "baseline_runtime_sec")
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
    getter = getattr(state, "session_budget_usable_sec", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001 — fall back to the attribute path
            pass
    return session_remaining_seconds(state)


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


def exit_time_exhausted_prelude(
    state: Any,
    *,
    now_unix: float | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Route to CLOSE when the session clock runs out before PRELUDE lands a baseline."""
    usable = session_usable_seconds(state)
    if usable is None or usable > 0.0:
        return None
    return "time_exhausted_during_prelude", {
        "session_usable_sec": round(usable, 1),
        "prelude_spent_sec": round(
            phase_cumulative_seconds(state, phase=PHASE_PRELUDE, now_unix=now_unix),
            1,
        ),
    }


def exit_cold_anchor_prelude(state: Any) -> tuple[str, dict[str, Any]] | None:
    """Route to CLOSE when PRELUDE could only produce a cold anchor."""
    if not bool(getattr(state, "baseline_measure_round_dropped", False)):
        return None
    usable = session_usable_seconds(state)
    if usable is None:
        return None
    # Without the boot/benchmark split -- a scriptable workload runs no server, so it has no ready boundary to split
    # on -- the cold round's whole wall-clock stands in for each half, the same upper bound the round's own gate falls
    # back to.
    cold_sec = measured_seconds(state, "baseline_runtime_sec")
    round_sec = (
        baseline_round_cost_sec(
            state,
            double_run=bool(getattr(state, "baseline_double_run", False)),
        )
        or cold_sec
    )
    use_sec = one_more_measurement_sec(state) or cold_sec
    if round_sec is None or use_sec is None:
        return None
    if usable >= round_sec + use_sec:
        return None
    return "prelude_cold_anchor_low_budget", {
        "baseline_anchor": "cold",
        "retry_round_sec": round(round_sec, 1),
        **prelude_exit_viability(state),
    }


def exit_terminal_prelude(state: Any) -> tuple[str, dict[str, Any]] | None:
    """Decide the PRELUDE terminal exit on repeated baseline failures."""
    streak = int(getattr(state, "baseline_failure_streak", 0) or 0)
    if streak >= 3:
        return "prelude_baseline_failed", {"baseline_failure_streak": streak}
    return None


def exit_normal_kernel(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """KERNEL normal exit."""
    # ``kernel_work_pending`` answers for outstanding integrations before it short-circuits on a terminal Controller,
    # so asking it here keeps this exit from stepping over an unintegrated KEEP.
    if _controller_phase_terminal(state) and not kernel_work_pending(state):
        result = getattr(state, "kernel_rewrite_controller_result", None) or {}
        return "kernel_controller_done", {
            "controller_status": result.get("status"),
            "patch_count": int(result.get("patch_count") or 0),
            "task_count": int(result.get("task_count") or 0),
            "reason": str(result.get("reason") or ""),
        }
    if _pending_escalate_hint(state) == ESCALATE_HINT_SKIP_TO_SWEEP:
        if not kernel_work_pending(state):
            return "kernel_no_more_leverage", {
                "evidence": "kernel_no_more_leverage",
                "hint": ESCALATE_HINT_SKIP_TO_SWEEP,
            }
    # Idle-spin guard: the escalate-hint handoff above needs the kernel_agent to emit ``escalate_strategy_change``,
    # but PolicyGate denies that intent for the kernel_agent role — so when the phase stops moving it can otherwise
    # spin (hallucinated kernel-id requests / no-intent turns) until the wall-clock cap.
    idle_ticks = int(getattr(state, "kernel_idle_ticks", 0) or 0)
    idle_since = _kernel_idle_since_unix(state)
    if idle_ticks >= KERNEL_IDLE_MAX_TICKS and idle_since > 0.0:
        now = float(now_unix if now_unix is not None else _now_unix(state))
        idle_seconds = max(0.0, now - idle_since)
        if idle_seconds >= KERNEL_IDLE_MIN_SECONDS:
            return "kernel_no_more_leverage", {
                "evidence": "kernel_idle_no_progress",
                "idle_ticks": idle_ticks,
                "idle_max_ticks": KERNEL_IDLE_MAX_TICKS,
                "idle_seconds": round(idle_seconds, 3),
                "idle_min_seconds": KERNEL_IDLE_MIN_SECONDS,
            }
    rejected = getattr(state, "rejected_kernel_ids", None) or []
    rejected_count = len(rejected) if isinstance(rejected, list) else 0
    remaining = phase_budget_remaining_seconds(
        state,
        budget_pct=budget_pct,
        now_unix=now_unix,
    )
    if remaining is not None and remaining <= 0:
        return "kernel_phase_budget_exhausted", {
            "entry_elapsed_seconds": phase_elapsed_seconds(state, now_unix=now_unix),
            "cumulative_elapsed_seconds": phase_cumulative_seconds(state, now_unix=now_unix),
            "rejected_kernel_count": rejected_count,
        }
    if phase_cap_exceeded(state, budget_pct=budget_pct, now_unix=now_unix):
        return "kernel_budget_cap", {
            "entry_elapsed_seconds": phase_elapsed_seconds(state, now_unix=now_unix),
            "cumulative_elapsed_seconds": phase_cumulative_seconds(state, now_unix=now_unix),
            "rejected_kernel_count": rejected_count,
        }
    return None


#: ``reloop_blocked`` values that name the terminal SWEEP closes on. A block for
#: any other reason keeps the ladder's own exit reason.
_RELOOP_BLOCK_TERMINALS: dict[str, str] = {
    "global_converged": "global_converged",
    "max_cycles": "global_converged",
    "target_reached": "target_reached",
}


def exit_normal_sweep(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """SWEEP normal exit: the concurrency ladder's terminal state, or budget exhausted."""
    last_conc = getattr(state, "last_conc_sweep", None) or {}
    if isinstance(last_conc, dict):
        status = str(last_conc.get("status") or "").lower()
        if status == "failed":
            return "sweep_failed", {"sweep_status": status}
        if status in ("succeeded", "partial", "completed", "skipped"):
            evidence: dict[str, Any] = {"sweep_status": status}
            # A sweep that declined to run is also terminal, and the exit reason alone cannot tell the two apart
            # afterwards. was_skipped covers both declining and spending the whole budget without a comparable pair,
            # so it is only carried with the flag that separates them (see
            # kernel.conc_sweep.conc_sweep_declined_to_run).
            if last_conc.get("was_skipped"):
                evidence["sweep_was_skipped"] = True
                evidence["sweep_skip_budget_exhausted"] = bool(last_conc.get("budget_exhausted"))
                evidence["sweep_skip_reason"] = str(last_conc.get("skip_reason") or "")
            return "sweep_done", evidence
    remaining = phase_budget_remaining_seconds(
        state,
        budget_pct=budget_pct,
        now_unix=now_unix,
    )
    if remaining is not None and remaining <= 0:
        return "sweep_budget_exhausted", {
            "entry_elapsed_seconds": phase_elapsed_seconds(state, now_unix=now_unix),
            "cumulative_elapsed_seconds": phase_cumulative_seconds(state, now_unix=now_unix),
        }
    if phase_cap_exceeded(state, budget_pct=budget_pct, now_unix=now_unix):
        return "sweep_budget_cap", {
            "entry_elapsed_seconds": phase_elapsed_seconds(state, now_unix=now_unix),
            "cumulative_elapsed_seconds": phase_cumulative_seconds(state, now_unix=now_unix),
        }
    return None


# Transition decision (the only function the Coordinator calls each tick)
def _resolve_plateau_overrides(state: Any) -> dict[str, Any]:
    """Pull operator-tuned plateau thresholds off :attr:`SharedState.plateau_overrides` (empty → library defaults)."""
    overrides = getattr(state, "plateau_overrides", None) or {}
    return dict(overrides) if isinstance(overrides, dict) else {}


# stops re-selecting the candidate, and is skipped by the plateau streak because an infrastructure failure is not
# evidence that the search is exhausted.
_FRAMEWORK_DISPATCH_FAILED_STATUS = "dispatch_failed"


def framework_agent_consecutive_no_keep(state: Any) -> int:
    """Count trailing consecutive resolved candidates that did not KEEP."""
    progress = getattr(state, "framework_agent_phase_progress", None) or []
    if not isinstance(progress, list):
        return 0
    count = 0
    for row in reversed(progress):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip().lower()
        # Macro-cycle boundary marker stops the streak walk so a prior cycle's trailing no-KEEP rows cannot instantly
        # re-plateau the next cycle.
        if status == "cycle_boundary":
            break
        # A specialist that never ran produced no search result to plateau on.
        if status == _FRAMEWORK_DISPATCH_FAILED_STATUS:
            continue
        is_keep = bool(row.get("kept")) or status == "kept"
        if is_keep:
            break
        count += 1
    return count


def framework_agent_plateau_streak_threshold() -> int:
    """Resolve the consecutive-no-keep plateau threshold."""
    return DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK


def source_arm_plateaued(state: Any) -> tuple[bool, dict[str, Any]]:
    """Whether the source arm (candidates, authored patches) has run dry."""
    streak = framework_agent_consecutive_no_keep(state)
    threshold = framework_agent_plateau_streak_threshold()
    exhausted = bool(getattr(state, "framework_agent_phase_done", False))
    evidence = {
        "source_consecutive_no_keep": streak,
        "source_threshold": threshold,
        "source_candidates_exhausted": exhausted,
    }
    return (streak >= threshold or exhausted), evidence


def _optimize_did_work_this_cycle(state: Any) -> bool:
    """Whether either arm has dispatched or benched anything this macro-cycle."""
    if _rows_for_current_cycle(getattr(state, "specialist_rounds", None) or [], state):
        return True
    explore_search = getattr(state, "explore_search", None) or {}
    tested = explore_search.get("tested") if isinstance(explore_search, dict) else None
    if isinstance(tested, dict) and _rows_for_current_cycle(list(tested.values()), state):
        return True
    return bool(_rows_for_current_cycle(getattr(state, "framework_agent_phase_progress", None) or [], state))


def exit_normal_optimize(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
    plateau_lookback: int = DEFAULT_PLATEAU_EXPLORE_LOOKBACK,
    plateau_keep_gain_threshold_pct: float = DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
    plateau_empty_streak_threshold: int = DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
) -> tuple[str, dict[str, Any]] | None:
    """OPTIMIZE normal exit."""
    source_dry, source_ev = source_arm_plateaued(state)
    config_dry, config_ev = compute_plateau_explore(
        state,
        lookback=plateau_lookback,
        keep_gain_threshold_pct=plateau_keep_gain_threshold_pct,
        empty_streak_threshold=plateau_empty_streak_threshold,
    )
    arms = {
        **source_ev,
        **config_ev,
        "source_arm_plateaued": source_dry,
        "config_arm_plateaued": config_dry,
        # Either arm running dry is enough to redirect the next cycle.
        "switch_bottleneck": bool(source_dry or config_dry),
    }

    hint = str(getattr(state, "pending_escalate_hint", "") or "").strip()
    if hint == ESCALATE_HINT_SKIP_TO_KERNEL:
        # Honoured only once the phase has actually run something this cycle: a phase that dispatched nothing must not
        # end with zero validated work.
        if _optimize_did_work_this_cycle(state):
            return "optimize_no_more_leverage", {**arms, "evidence": "llm_escalation", "hint": hint}
    if hint == ESCALATE_HINT_SKIP_TO_SWEEP:
        return "optimize_no_more_leverage", {**arms, "evidence": "skip_to_sweep", "hint": hint}

    if source_dry and config_dry:
        return "optimize_no_more_leverage", {**arms, "evidence": "both_arms_plateaued", "plateau": True}

    remaining = phase_budget_remaining_seconds(state, budget_pct=budget_pct, now_unix=now_unix)
    if remaining is not None and remaining <= 0:
        return "optimize_phase_budget_exhausted", {
            **arms,
            "entry_elapsed_seconds": phase_elapsed_seconds(state, now_unix=now_unix),
            "cumulative_elapsed_seconds": phase_cumulative_seconds(state, now_unix=now_unix),
        }
    if phase_cap_exceeded(state, budget_pct=budget_pct, now_unix=now_unix):
        return "optimize_budget_cap", {
            **arms,
            "entry_elapsed_seconds": phase_elapsed_seconds(state, now_unix=now_unix),
            "cumulative_elapsed_seconds": phase_cumulative_seconds(state, now_unix=now_unix),
        }
    return None


def _post_prelude_target(*, optimize_enabled: bool, kernel_enabled: bool) -> str:
    """First active phase after PRELUDE: OPTIMIZE, else KERNEL, else SWEEP (``--no-framework-agent`` / ``--no-kernel`` collapse the chain)."""
    if optimize_enabled:
        return PHASE_FRAMEWORK_AGENT
    if kernel_enabled:
        return PHASE_KERNEL_AGENT
    return PHASE_SWEEP


def compute_next_phase(
    state: Any,
    *,
    kernel_enabled: bool = True,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
    optimize_enabled: bool = True,
) -> tuple[str, str, dict[str, Any]] | None:
    """Return ``(next_phase, reason, evidence)`` or ``None``."""
    current = (getattr(state, "phase", "") or "").strip().upper() or PHASE_PRELUDE
    overrides = _resolve_plateau_overrides(state)

    # Global terminal stop_reason overrides phase-local judgments.
    terminal = _global_terminal(state)
    if terminal is not None and current != PHASE_CLOSE:
        reason, evidence = terminal
        return PHASE_CLOSE, reason, {"terminal": True, **evidence}

    closing = _closing_phase_terminal(state)
    if closing is not None and current != PHASE_CLOSE:
        reason, evidence = closing
        return PHASE_CLOSE, reason, {"terminal": True, **evidence}

    # A met target ends the optimizing phases early; SWEEP is their normal next station, and the curve then measures
    # the configuration it was met on.
    if target_was_reached(state) and phase_index(PHASE_PRELUDE) < phase_index(current) < phase_index(PHASE_SWEEP):
        return PHASE_SWEEP, "target_reached", {"target_reached_at": str(getattr(state, "target_reached_at", "") or "")}

    if current == PHASE_PRELUDE:
        term = exit_terminal_prelude(state)
        if term is not None:
            return PHASE_CLOSE, term[0], {"terminal": True, **term[1]}
        # Asked before the normal exit, which sees only that a figure exists: a cold anchor is a figure the later
        # phases cannot honestly compare to.
        cold = exit_cold_anchor_prelude(state)
        if cold is not None:
            return PHASE_CLOSE, cold[0], {"terminal": True, **cold[1]}
        norm = exit_normal_prelude(state)
        if norm is None:
            # No baseline and no clock left: name the failure instead of letting the run read as an ordinary exit.
            exhausted = exit_time_exhausted_prelude(state, now_unix=now_unix)
            if exhausted is not None:
                return PHASE_CLOSE, exhausted[0], {"terminal": True, **exhausted[1]}
        if norm is not None:
            target = _post_prelude_target(
                optimize_enabled=optimize_enabled,
                kernel_enabled=kernel_enabled,
            )
            evidence = dict(norm[1])
            if target != PHASE_FRAMEWORK_AGENT:
                evidence["optimize_skipped"] = True
            return target, norm[0], evidence
        return None

    if current == PHASE_FRAMEWORK_AGENT:
        norm = exit_normal_optimize(
            state,
            budget_pct=budget_pct,
            now_unix=now_unix,
            plateau_lookback=int(
                overrides.get("explore_lookback", DEFAULT_PLATEAU_EXPLORE_LOOKBACK),
            ),
            plateau_keep_gain_threshold_pct=float(
                overrides.get("explore_keep_gain_pct", DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT),
            ),
            plateau_empty_streak_threshold=int(
                overrides.get("explore_empty_streak", DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK),
            ),
        )
        if norm is not None:
            # Exhausted optimisation leverage is not terminal: switch lever and advance to KERNEL; only with KERNEL
            # disabled does it wind down.
            if kernel_enabled:
                return PHASE_KERNEL_AGENT, norm[0], norm[1]
            return (
                PHASE_SWEEP,
                "no_kernel_skipped",
                {"passed_through_reason": norm[0], **norm[1]},
            )
        return None

    if current == PHASE_KERNEL_AGENT:
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
            exit_reason, exit_evidence = norm
            # Failed conc_sweep closeout is terminal: preserve the honest stop_reason instead of opening another
            # macro-cycle.
            if exit_reason == "sweep_failed":
                return PHASE_CLOSE, exit_reason, exit_evidence
            # R1: open a new macro-cycle while budget remains and the run hasn't globally converged (R7); wind down to
            # CLOSE only when reloop is blocked (budget, convergence, or max_cycles).
            reloop, reloop_ev = should_reloop_to_explore(state, now_unix=now_unix)
            if reloop and optimize_enabled:
                reloop_target = PHASE_FRAMEWORK_AGENT
                return (
                    reloop_target,
                    "cycle_reloop",
                    {
                        **exit_evidence,
                        **reloop_ev,
                        "loopback": True,
                    },
                )
            # R7: if looping was blocked by global convergence or the safety cap, terminate with a terminal
            # stop_reason instead of idling in CLOSE.
            blocked = str(reloop_ev.get("reloop_blocked") or "")
            terminal_reason = _RELOOP_BLOCK_TERMINALS.get(blocked)
            if terminal_reason is not None:
                return (
                    PHASE_CLOSE,
                    terminal_reason,
                    {
                        **exit_evidence,
                        **reloop_ev,
                        "terminal": True,
                    },
                )
            return PHASE_CLOSE, exit_reason, {**exit_evidence, **reloop_ev}
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
    """Construct a canonical phase_history row; ``reason`` unvalidated for resume tools."""
    return {
        "from_phase": (from_phase or "").strip().upper(),
        "to_phase": (to_phase or "").strip().upper(),
        "reason": (reason or "").strip(),
        "evidence": dict(evidence or {}),
        "ts": ts,
        "ts_unix": float(ts_unix or 0.0),
        "cycle": int(cycle or 0),
    }


# Lifecycle events — operator-facing phase/step boundary log.
LIFECYCLE_STATUS_START = "START"
LIFECYCLE_STATUS_END = "END"
LIFECYCLE_STATUS_ERROR = "ERROR"
# Phase-boundary marker: a point-in-time "entered <phase>" mark with no matching END (unlike START, which pairs with a
# later END for the same step).
LIFECYCLE_STATUS_ENTER = "ENTER"
LIFECYCLE_STATUSES: frozenset[str] = frozenset(
    {
        LIFECYCLE_STATUS_START,
        LIFECYCLE_STATUS_END,
        LIFECYCLE_STATUS_ERROR,
        LIFECYCLE_STATUS_ENTER,
    }
)

# Human-friendly labels for the coordinator phases.
PHASE_HUMAN_LABELS: dict[str, str] = {
    PHASE_PRELUDE: "Prelude (baseline + roofline)",
    PHASE_FRAMEWORK_AGENT: "Optimize (config / source / upstream)",
    PHASE_KERNEL_AGENT: "Kernel optimization",
    PHASE_SWEEP: "Concurrency sweep",
    PHASE_CLOSE: "Close (report)",
}

# Human-friendly labels for the lifecycle steps surfaced to operators.
LIFECYCLE_STEP_LABELS: dict[str, str] = {
    "roofline": "TraceLens",
    "trace_analyze": "TraceLens",
    "run_gemm_tuning": "GEMM tuning",
    "run_optimization": "GEAK",
    "integrate": "Integrate",
    "apply_patch": "Integrate",
    "explore": "Validate (bench on the stack)",
    "sweep": "Concurrency sweep",
    "report": "Report",
    "session_breakdown": "Report (session breakdown)",
}


def lifecycle_label(name: str) -> str:
    """Resolve a human-friendly label for a step or phase name."""
    key = (name or "").strip()
    if key in LIFECYCLE_STEP_LABELS:
        return LIFECYCLE_STEP_LABELS[key]
    upper = key.upper()
    if upper in PHASE_HUMAN_LABELS:
        return PHASE_HUMAN_LABELS[upper]
    return key


def make_lifecycle_event(
    *,
    step: str,
    status: str,
    phase: str,
    label: str | None,
    artifacts: dict[str, str] | None,
    detail: str,
    duration_s: float | None,
    seq: int,
    ts: str,
) -> dict[str, Any]:
    """Construct a canonical lifecycle event row."""
    event: dict[str, Any] = {
        "seq": int(seq),
        "ts": ts,
        "phase": (phase or "").strip().upper(),
        "step": (step or "").strip(),
        "label": (label or lifecycle_label(step)),
        "status": (status or "").strip().upper(),
        "detail": (detail or "").strip(),
        "artifacts": {str(k): str(v) for k, v in (artifacts or {}).items() if v not in (None, "")},
    }
    if duration_s is not None:
        try:
            event["duration_s"] = round(float(duration_s), 3)
        except (TypeError, ValueError):
            # A malformed duration_s is omitted rather than failing creation.
            pass
    return event


# Phase-transition / lifecycle write-owner functions (take ``state`` first and own the phase_history / lifecycle
# bookkeeping).
def bank_phase_segment(state, *, until_unix: float) -> float:
    """Bank the current phase's live segment, ending at ``until_unix``, into the durable totals."""
    phase = (getattr(state, "phase", "") or "").strip().upper()
    if not phase:
        return 0.0
    segment = phase_elapsed_seconds(state, now_unix=until_unix)
    totals = getattr(state, "phase_elapsed_totals", None)
    totals = dict(totals) if isinstance(totals, dict) else {}
    try:
        banked = max(0.0, float(totals.get(phase, 0.0) or 0.0))
    except (TypeError, ValueError):
        banked = 0.0
    totals[phase] = banked + segment
    state.phase_elapsed_totals = totals
    # The optimisation phase keeps its own accumulator: it carries a tri-state "unknown" for legacy resumes that
    # status telemetry reports as absent, whereas ``phase_elapsed_totals`` must never report "unknown" — a budget
    # guard would read that as "no cap".
    if phase == PHASE_FRAMEWORK_AGENT:
        raw_accumulated = getattr(state, "explore_elapsed_accum_s", 0.0)
        if raw_accumulated is not None:
            try:
                accumulated = float(raw_accumulated or 0.0)
            except (TypeError, ValueError):
                state.explore_elapsed_accum_s = None
            else:
                state.explore_elapsed_accum_s = accumulated + segment
    return segment


def record_phase_transition(
    state,
    *,
    to_phase: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
    ts: str | None = None,
    ts_unix: float | None = None,
) -> dict[str, Any]:
    """Append a phase_history row and atomically update ``phase`` fields; ``phase``/``phase_history`` are CORE_STATE_FIELDS so LLM update_state is rejected. Returns the inserted row."""
    from datetime import datetime as _dt, timezone as _tz
    import time as _time
    from ..state.shared_state import _PHASE_HISTORY_CAP

    now_ts = ts or _dt.now(_tz.utc).isoformat(timespec="seconds")
    now_unix = float(ts_unix if ts_unix is not None else _time.time())
    from_phase = (state.phase or "").strip().upper()
    # Read before the loopback's bump can be observed here: it increments
    # ``macro_cycle`` on the way out of a phase, so the cycle in scope at the
    # transition is not always the one the outgoing phase ran in.
    prev_cycle = int(getattr(state, "macro_cycle", 0) or 0)
    # Bank the finished segment for EVERY phase so the budget guards can charge
    # a phase for the whole run instead of the current entry.
    bank_phase_segment(state, until_unix=now_unix)
    row = make_history_row(
        from_phase=from_phase,
        to_phase=to_phase,
        reason=reason,
        evidence=evidence,
        ts=now_ts,
        ts_unix=now_unix,
        cycle=int(getattr(state, "macro_cycle", 0) or 0),
    )
    history = list(state.phase_history or [])
    history.append(row)
    if len(history) > _PHASE_HISTORY_CAP:
        history = history[-_PHASE_HISTORY_CAP:]
    state.phase_history = history
    state.phase = row["to_phase"]
    state.phase_started_ts = now_ts
    state.phase_started_unix = now_unix
    # Publish the phase for LLM attribution: the spawn sites that tag outbound calls sit in specialists and kernel
    # tools and cannot reach SharedState.
    from hyperloom.common.llm_attribution import set_current_phase

    set_current_phase(str(row["to_phase"] or ""))
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import phase_event

        # The phase itself, as a timeline event: close the span being left on
        # the exit that ended it, and open the one being entered. Recorded here
        # because here is where the two facts exist -- export could only pair
        # phase_history rows off two at a time to guess them back, and had no
        # row at all to close the segment the session ended in.
        if from_phase and from_phase != str(row.get("to_phase") or ""):
            phase_event.record_exit(
                phase=from_phase,
                macro_cycle=prev_cycle,
                to_phase=str(row.get("to_phase") or ""),
                reason=str(row.get("reason") or ""),
                evidence=dict(row.get("evidence") or {}),
                exited_at=str(row.get("ts") or ""),
                exited_unix=now_unix,
            )
        phase_event.record_entry(
            phase=str(row.get("to_phase") or ""),
            macro_cycle=int(getattr(state, "macro_cycle", 0) or 0),
            sequence=len(history),
            from_phase=from_phase,
            reason=str(row.get("reason") or ""),
            evidence=dict(row.get("evidence") or {}),
            entered_at=str(row.get("ts") or ""),
            entered_unix=now_unix,
        )
    except Exception:  # noqa: BLE001 -- telemetry must never block phase changes
        pass
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
    from ..state.shared_state import _PHASE_HISTORY_CAP

    now_ts = ts or _dt.now(_tz.utc).isoformat(timespec="seconds")
    now_unix = float(ts_unix if ts_unix is not None else _time.time())
    phase = (state.phase or "").strip().upper()
    row = make_history_row(
        from_phase=phase,
        to_phase=phase,
        reason=(reason or "").strip(),
        evidence=evidence,
        ts=now_ts,
        ts_unix=now_unix,
        cycle=int(getattr(state, "macro_cycle", 0) or 0),
    )
    history = list(state.phase_history or [])
    history.append(row)
    if len(history) > _PHASE_HISTORY_CAP:
        history = history[-_PHASE_HISTORY_CAP:]
    state.phase_history = history
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import phase_event

        phase_event.record_marker(
            phase=phase,
            macro_cycle=int(getattr(state, "macro_cycle", 0) or 0),
            sequence=len(history),
            reason=str(row.get("reason") or ""),
            evidence=dict(row.get("evidence") or {}),
            ts=str(row.get("ts") or ""),
        )
    except Exception:  # noqa: BLE001 -- telemetry must never block the marker
        pass
    return row


def record_lifecycle_event(
    state,
    *,
    step: str,
    status: str,
    phase: str | None = None,
    label: str | None = None,
    artifacts: dict[str, str] | None = None,
    detail: str = "",
    duration_s: float | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    """Append a structured lifecycle event marking a phase/step boundary."""
    from ..state.shared_state import _LIFECYCLE_CAP, _now_iso

    events = state.lifecycle
    if events is None:
        events = state.lifecycle = []
    next_seq = (int(events[-1].get("seq", -1)) + 1) if events else 0
    event = make_lifecycle_event(
        step=step,
        status=status,
        phase=(phase if phase is not None else (state.phase or "")),
        label=label,
        artifacts=artifacts,
        detail=detail,
        duration_s=duration_s,
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
    "DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT",
    "DEFAULT_PLATEAU_KERNEL_LOOKBACK",
    "DEFAULT_PLATEAU_KERNEL_REVERT_STREAK",
    "ESCALATE_HINT_BUDGET_BUMP_CAP",
    "ESCALATE_HINT_BUDGET_BUMP_DELTA",
    "ESCALATE_HINT_EXTEND_EXPLORE_BUDGET",
    "ESCALATE_HINT_EXTEND_KERNEL_BUDGET",
    "ESCALATE_HINT_SKIP_TO_CLOSE",
    "ESCALATE_HINT_SKIP_TO_KERNEL",
    "ESCALATE_HINT_SKIP_TO_SWEEP",
    "ESCALATE_HINT_VOCAB",
    "LIFECYCLE_STATUSES",
    "LIFECYCLE_STATUS_END",
    "LIFECYCLE_STATUS_ENTER",
    "LIFECYCLE_STATUS_ERROR",
    "LIFECYCLE_STATUS_START",
    "LIFECYCLE_STEP_LABELS",
    "PHASE_ALLOWED_ACTIONS",
    "PHASE_CLOSE",
    "PHASE_EXIT_REASONS",
    "PHASE_FRAMEWORK_AGENT",
    "PHASE_HUMAN_LABELS",
    "PHASE_INDEX",
    "PHASE_KERNEL_AGENT",
    "PHASE_NAMES",
    "PHASE_PRELUDE",
    "PHASE_SWEEP",
    "STOP_REASON_VOCAB",
    "lifecycle_label",
    "make_lifecycle_event",
    "DEFAULT_MAX_MACRO_CYCLES",
    "DEFAULT_CYCLE_RELOOP_MIN_REMAINING_SEC",
    "DEFAULT_GLOBAL_CONVERGENCE_NO_GAIN_CYCLES",
    "DEFAULT_LONGRUN_THRESHOLD_MINUTES",
    "is_long_run",
    "resolve_keep_threshold",
    "should_reloop_to_explore",
    "target_was_reached",
    "allowed_actions_for",
    "apply_escalate_budget_bump",
    "bank_phase_segment",
    "compute_next_phase",
    "compute_plateau_explore",
    "framework_agent_consecutive_no_keep",
    "framework_agent_plateau_streak_threshold",
    "compute_plateau_kernel",
    "exit_normal_optimize",
    "source_arm_plateaued",
    "exit_normal_kernel",
    "exit_cold_anchor_prelude",
    "exit_normal_prelude",
    "exit_normal_sweep",
    "exit_terminal_prelude",
    "exit_time_exhausted_prelude",
    "append_phase_evidence_row",
    "append_phase_history_event",
    "is_phase_transition_row",
    "phase_history_event_name",
    "baseline_round_cost_sec",
    "benchmark_cost_sec",
    "boot_cost_sec",
    "measured_seconds",
    "one_more_measurement_sec",
    "prelude_affordable_seconds",
    "prelude_can_afford",
    "prelude_exit_viability",
    "session_usable_seconds",
    "render_phase_action_bullets",
    "is_valid_escalate_hint",
    "is_valid_phase_exit_reason",
    "is_valid_stop_reason",
    "compute_kernel_progress_fingerprint",
    "kernel_work_pending",
    "make_history_row",
    "explore_elapsed_seconds",
    "normalize_budget_pct",
    "phase_budget_remaining_seconds",
    "phase_cumulative_seconds",
    "phase_elapsed_seconds",
    "phase_elapsed_totals_from_history",
    "phase_index",
    "session_remaining_seconds",
    "warm_replay_in_flight",
]
