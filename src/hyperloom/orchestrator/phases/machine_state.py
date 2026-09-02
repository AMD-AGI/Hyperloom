# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase state machine.

Pure functions over a frozen SharedState; Coordinator is the only writer.
Chain PRELUDE → FRAMEWORK_AGENT → KERNEL_AGENT → SWEEP → CLOSE (monotonic
within a macro-cycle; SWEEP reloops back to FRAMEWORK_AGENT across macro-cycles
until convergence, budget, or the cycle cap forces CLOSE).
Any phase → CLOSE on terminal/abort; ``recover`` is phase-orthogonal.
"""

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
    """Return monotonic index of ``phase`` (Inv-2.1 check); unknown → -1.

    Args:
        phase (str): Phase name; stripped and upper-cased before lookup.

    Returns:
        int: The phase's position in :data:`PHASE_NAMES`, or ``-1`` when unknown.
    """
    return PHASE_INDEX.get((phase or "").strip().upper(), -1)


# Phase ↔ allowed action set: ALLOWED passes R1; Coordinator-auto actions stay
# out of PROPOSABLE so LLM proposals are denied.
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
    # Three levers: configuration grids (``explore``), investigation and
    # authoring (``specialist``), and landing a patch from any source
    # (``integrate_patch``).
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
    # No kernel_opt or gemm_tuning: the Coordinator dispatches both once at phase
    # entry, so an LLM re-issuing them per tick would bypass the lane budget and
    # the nomination it is derived from. ``integrate`` stays -- draining the KEEP
    # queue is still the model's job. This set is about what may be *proposed*;
    # what counts as kernel-lane work in flight is ``KERNEL_LANE_TASK_KINDS``.
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


# Task kinds that mean the KERNEL lane is busy, which is a wider question than
# what a model may propose: a Coordinator-owned lane is dispatched without ever
# being proposable, and its task occupies the phase just the same. Kept separate
# from PHASE_ALLOWED_ACTIONS because the idle guard once shared that set and a
# running kernel_opt went invisible the moment the action stopped being
# model-requestable.
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
    """Return ``PHASE_ALLOWED_ACTIONS[phase]`` as a sorted tuple (deterministic).

    Args:
        phase (str): Phase name; stripped and upper-cased before lookup.

    Returns:
        tuple[str, ...]: The phase's allowed actions sorted ascending, or an
        empty tuple for an unknown phase.
    """
    return tuple(sorted(PHASE_ALLOWED_ACTIONS.get((phase or "").strip().upper(), frozenset())))


# Phase ↔ LLM-proposable set: allowlist minus Coordinator-managed and
# robustness-delegate-only actions (what PolicyGate accepts for Orchestration).
PHASE_LLM_PROPOSABLE_ACTIONS: dict[str, frozenset[str]] = {
    phase: actions - COORDINATOR_INTERNAL_ACTIONS - ROBUSTNESS_DELEGATE_ONLY_ACTIONS
    for phase, actions in PHASE_ALLOWED_ACTIONS.items()
}


def is_action_llm_proposable_in_phase(action_name: str, phase: str) -> bool:
    """Return True iff ``action_name`` is LLM-proposable in ``phase`` (unknown → deny)."""
    return _action_in_phase_map(action_name, phase, PHASE_LLM_PROPOSABLE_ACTIONS)


def llm_proposable_actions_for(phase: str) -> tuple[str, ...]:
    """Return ``PHASE_LLM_PROPOSABLE_ACTIONS[phase]`` sorted (deterministic).

    Args:
        phase (str): Phase name; stripped and upper-cased before lookup.

    Returns:
        tuple[str, ...]: The phase's LLM-proposable actions sorted ascending,
        or an empty tuple for an unknown phase.
    """
    return tuple(sorted(PHASE_LLM_PROPOSABLE_ACTIONS.get((phase or "").strip().upper(), frozenset())))


def render_phase_proposable_bullets(
    *,
    disabled_suffix: dict[str, str] | None = None,
) -> list[str]:
    """Render per-phase LLM-proposable action bullets (shared by the prompt builders).

    Args:
        disabled_suffix (dict[str, str] | None): Optional ``phase -> flag`` map;
            a present flag annotates that phase as ``(DISABLED: <flag> — phase
            skipped)``.

    Returns:
        list[str]: One markdown bullet per phase in :data:`PHASE_NAMES`.
    """
    suffix = disabled_suffix or {}
    out: list[str] = []
    for phase in PHASE_NAMES:
        proposable = llm_proposable_actions_for(phase)
        flag = suffix.get(phase)
        if flag:
            out.append(f"- **{phase}**: {', '.join(proposable)} (DISABLED: {flag} — phase skipped)")
        else:
            out.append(f"- **{phase}**: {', '.join(proposable)}")
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
        # Marker row: the source arm has nothing left to dispatch. Not a
        # transition -- the config arm may still be paying.
        "no_candidates_and_discovery_exhausted",
    }
)


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
        # Pre-run model-config compatibility preflight: config.json is corrupt or
        # declares RoPE scaling without a max-position field (both crash at load).
        "model_config_incompatible",
        # Baseline arg-validation fast-exit: >=2 consecutive baseline attempts
        # exited <30s on a bad CLI arg.
        "baseline_arg_error",
        # Enablement loop stall: >= _ENABLEMENT_MAX_STALL consecutive rounds made
        # no forward progress. A progressing round resets the streak.
        "enablement_stalled",
        # The baseline could not produce an accuracy result even though the
        # accuracy test was expected to run (broken eval / missing quality
        # gate). Optimizing against an unvalidated baseline is unsafe, so the
        # run halts. Post-baseline accuracy failures REVERT the offending
        # change instead of stopping.
        "baseline_accuracy_failed",
    }
)


def is_valid_stop_reason(value: str) -> bool:
    """Return True when ``value`` is a member of :data:`STOP_REASON_VOCAB`.

    PolicyGate uses this to reject any write of ``stop_reason`` that is not
    in the closed vocabulary. The value is stripped before comparison.

    Args:
        value (str): Candidate stop-reason string.

    Returns:
        bool: True if the stripped value is a recognized stop reason.
    """
    return (value or "").strip() in STOP_REASON_VOCAB


def is_valid_phase_exit_reason(value: str) -> bool:
    """Return True when ``value`` is a member of :data:`PHASE_EXIT_REASONS`.

    Closed by test, not at runtime: a reason outside it is a bug in the
    writer. The value is stripped before comparison.

    Args:
        value (str): Candidate phase-exit reason string.

    Returns:
        bool: True if the stripped value is a recognized phase-exit reason.
    """
    return (value or "").strip() in PHASE_EXIT_REASONS


# Default phase budgets (% of wall-clock).
DEFAULT_PHASE_BUDGET_PCT: dict[str, float] = {
    PHASE_PRELUDE: 0.03,
    # The optimisation phase carries both levers' share. Rotation between them
    # is the arms' plateau judgement, not a wall-clock cap.
    PHASE_FRAMEWORK_AGENT: 0.40,
    PHASE_KERNEL_AGENT: 0.50,
    PHASE_SWEEP: 0.05,
    PHASE_CLOSE: 0.02,
}

# Share of the session held back for the phases that actually produce a result.
#
# PRELUDE's ``DEFAULT_PHASE_BUDGET_PCT`` entry is 3%, but nothing enforced it:
# :func:`exit_normal_prelude` tests only whether a baseline landed, so the phase
# ran to whatever its contents cost. Two sessions on unrelated models (different
# quantization, different PRELUDE composition -- one baseline-dominated, one
# split with a TraceLens roofline) spent 73.8% and 72.8% of a three-hour budget
# in it, and both reached FRAMEWORK_AGENT with ~47 minutes left against its
# 108-minute entry threshold. Both budgets were honoured; both sessions produced
# nothing. Stopping the run on time is necessary and not sufficient -- the
# phases that spend the budget have to be the ones that produce the result.
#
# This bounds the preparation rather than the session, and it is deliberately
# looser than 3%: a baseline that legitimately takes an hour should still run,
# it just cannot also buy an 80-minute roofline out of the optimization phases'
# time.
#
# A second bound used to sit beside it -- a ceiling on PRELUDE's own banked
# spend -- and it was removed because the two clocks it straddled can disagree.
# Banked phase spend accumulates against the phase ledger; the reserve is read
# off the session clock. A session resumed with a reanchored budget restarts the
# session clock and carries the ledger over, so preparation was born over its
# ceiling with hours genuinely left, and every resumed session was refused the
# measured half of its baseline. The reserve alone answers the question that
# decides the matter: after this work, is enough left for the phases that
# produce the result?
OPTIMIZATION_RESERVE_PCT: float = 0.50

# Wall-clock ceiling for an unbounded run (``max_minutes`` == 0): the container
# lifetime. Used both as the global deadline and as the basis for the absolute
# per-phase cap so an unbounded run still forces phase rotation.
DEFAULT_LONGRUN_MAX_MINUTES: int = 14 * 24 * 60
# Reference window the absolute per-phase cap applies its budget fraction to.
# Short bounded runs bind on the (smaller) session-derived term; long/unbounded
# runs bind on this 24h reference.
PHASE_ABSOLUTE_CAP_REFERENCE_MINUTES: int = 24 * 60


# Plateau judgment defaults (CLI --plateau-* flags); kept here for pure callers + tests.
DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT: float = 0.5
DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK: int = 5
DEFAULT_PLATEAU_EXPLORE_LOOKBACK: int = 5
DEFAULT_PLATEAU_KERNEL_REVERT_STREAK: int = 3
DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT: float = 0.5
DEFAULT_PLATEAU_KERNEL_LOOKBACK: int = 5


import os as _os_env  # noqa: E402

# FRAMEWORK per-candidate plateau: after this many consecutive resolved
# candidates without a KEEP (including non-benchmarked terminal outcomes), the
# source arm is dry. A KEEP — or a macro-cycle boundary — resets it.
DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK: int = 5


# R1 macro-cycle reloop: SWEEP loops back to FRAMEWORK_AGENT for a new
# macro-cycle while budget remains and the run hasn't globally converged.

# Safety ceiling on macro-cycles (defense against a pathological tight loop).
DEFAULT_MAX_MACRO_CYCLES: int = 1000

# Share of a bounded session's total budget that must remain to open a cycle.
_CYCLE_RELOOP_BUDGET_RATIO: float = 0.15


def _default_cycle_reloop_min_remaining_sec() -> float:
    """Absolute reloop floor in seconds; env-overridable via
    ``INFERENCE_OPTIMIZER_CYCLE_RELOOP_MIN_REMAINING_SEC``.

    Default 3 h. It is the only floor for unbounded runs; bounded runs take the
    smaller of it and :data:`_CYCLE_RELOOP_BUDGET_RATIO` of their total budget,
    so a short session is not blocked by a threshold it can never satisfy.
    """
    raw = (_os_env.environ.get("INFERENCE_OPTIMIZER_CYCLE_RELOOP_MIN_REMAINING_SEC", "") or "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass  # malformed env override; fall through to the 3 h default
    return 10800.0


# Minimum session wall-clock (seconds) that must remain to justify opening a new
# macro-cycle; below this we wind down to CLOSE instead of starting a cycle we
# cannot meaningfully use.
DEFAULT_CYCLE_RELOOP_MIN_REMAINING_SEC: float = _default_cycle_reloop_min_remaining_sec()

# R7 global convergence: number of consecutive no-gain macro-cycles after which
# the run is considered converged (stop looping → CLOSE).
DEFAULT_GLOBAL_CONVERGENCE_NO_GAIN_CYCLES: int = 3

# Decaying acceptance curve: the marginal-gain bar shrinks each macro-cycle. It
# is injected at dispatch for explore and integrate_patch, and
# also sets the stack-stable threshold (=keep/2) and the convergence gain bar.
# The kernel-owned families hold their own fixed thresholds instead.
KEEP_THRESHOLD_FLOOR_PCT: float = 0.1
KEEP_THRESHOLD_SPAN_PCT: float = 0.9
# Multi-node baseline noise floor is ~2x single-node; scale the curve to match.
MULTI_NODE_KEEP_THRESHOLD_FACTOR: float = 2.0


def resolve_keep_threshold(state: Any) -> float:
    """Current-cycle KEEP threshold for every path that injects ``keep_threshold_pct``.

    Reads ``macro_cycle`` off ``state`` and the multi-node flag off the
    environment so callers pass nothing but the state.

    Args:
        state: The SharedState (or any object carrying ``macro_cycle``).

    Returns:
        The gain percentage a variant must clear to be KEPT this cycle.
    """
    from ..actions.executors._multi_node_env import is_multi_node

    cycle = int(getattr(state, "macro_cycle", 0) or 0)
    return decaying_keep_threshold_pct(cycle, multi_node=is_multi_node())


def decaying_keep_threshold_pct(macro_cycle: int, *, multi_node: bool = False) -> float:
    """KEEP / convergence gain threshold for cycle N = ``macro_cycle`` + 1.

    ``0.1 + 0.9 / N`` (percentage points): N=1 → 1.0% (identical to the legacy
    fixed threshold), decaying toward the 0.1% floor. Multi-node scales the
    whole curve by 2 so N=1 → 2.0% (legacy multi-node baseline).

    Args:
        macro_cycle (int): Zero-based macro-cycle counter (N = macro_cycle + 1).
        multi_node (bool): Scale the curve for the multi-node noise floor.

    Returns:
        float: Threshold in percentage points.
    """
    n = max(1, int(macro_cycle) + 1)
    base = KEEP_THRESHOLD_FLOOR_PCT + KEEP_THRESHOLD_SPAN_PCT / n
    return base * MULTI_NODE_KEEP_THRESHOLD_FACTOR if multi_node else base


# Long-run budget threshold. Long/unbounded runs use the per-cycle budget window;
# short bounded runs keep charge-back phase budgeting against the remaining
# session time even though they can now open new macro-cycles.
DEFAULT_LONGRUN_THRESHOLD_MINUTES: float = 24 * 60


def is_long_run(state: Any) -> bool:
    """True when the session budget should use long-run budget accounting.

    Unbounded runs (``max_minutes`` == 0, i.e. the 14-day ceiling) and bounded
    runs at least as long as :data:`DEFAULT_LONGRUN_THRESHOLD_MINUTES` are
    "long". Everything ``< 24h`` is a short bounded run: it may still open
    macro-cycles, but keeps charge-back phase budgeting instead of the fixed
    per-cycle window.

    Args:
        state (Any): Frozen SharedState view exposing ``max_minutes``.

    Returns:
        bool: True for unbounded runs (``max_minutes`` == 0) or bounded runs
        at least as long as :data:`DEFAULT_LONGRUN_THRESHOLD_MINUTES`.
    """
    mm = _max_minutes(state)
    if mm <= 0:
        return True
    return mm >= float(DEFAULT_LONGRUN_THRESHOLD_MINUTES)


def _cumulative_gain_validated(state: Any) -> float:
    """Return ``state.cumulative_gain_validated``, defensively coerced to float.

    Args:
        state (Any): Frozen SharedState view exposing
            ``cumulative_gain_validated``.

    Returns:
        float: The validated cumulative gain, or ``0.0`` when missing or
        non-numeric.
    """
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
    min_gain_pct: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Decide whether SWEEP should open a new macro-cycle (R1) or wind to CLOSE.

    Pure: never mutates state. Returns ``(reloop, evidence)``. The evidence
    carries the *effective* no-gain streak for the cycle that just completed so
    the Coordinator can persist it on the loopback/close transition.

    Loops back iff below the macro-cycle safety cap AND the run has not globally
    converged (R7: ``no_gain_cycles`` consecutive no-gain cycles) AND no
    roofline direction is saturated AND enough session budget remains to use a
    fresh cycle.

    Args:
        state (Any): Frozen SharedState view.
        now_unix (float | None): Override for the current time; defaults to
            wall-clock resolution when None.
        max_cycles (int): Safety ceiling on macro-cycles.
        min_remaining_sec (float): Absolute floor on the session seconds that
            must remain to justify a new cycle. A bounded session instead uses
            the smaller of this and :data:`_CYCLE_RELOOP_BUDGET_RATIO` of its
            total budget; the applied value is reported as
            ``evidence["min_remaining_sec_effective"]``.
        no_gain_cycles (int): Consecutive no-gain cycles that mark global
            convergence (R7).
        min_gain_pct (float | None): Per-cycle gain bar; ``None`` uses the
            decaying KEEP threshold for the cycle.

    Returns:
        tuple[bool, dict[str, Any]]: ``(reloop, evidence)`` — whether to open a
        new macro-cycle, and the evidence map (including the effective no-gain
        streak and any ``reloop_blocked`` reason).
    """
    cycle = int(getattr(state, "macro_cycle", 0) or 0)
    evidence: dict[str, Any] = {"macro_cycle": cycle}

    # Per-cycle gain since this cycle started → effective no-gain streak. A cycle
    # "gained" only when its validated gain rose by at least the decaying KEEP bar.
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

    # Safety cap on macro-cycles.
    if (cycle + 1) >= int(max_cycles):
        evidence["reloop_blocked"] = "max_cycles"
        return False, evidence

    # Physical ceiling convergence: if every roofline family that dominated is
    # now within its saturation threshold, stop cleanly.
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

    # Budget remaining must justify a fresh cycle. A bounded session scales the
    # floor to its own length so it is never blocked by an unreachable bar.
    effective_min_remaining = float(min_remaining_sec)
    max_minutes = _max_minutes(state)
    if max_minutes > 0:
        effective_min_remaining = min(
            effective_min_remaining,
            max_minutes * 60.0 * _CYCLE_RELOOP_BUDGET_RATIO,
        )
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
    """Consecutive no-work KERNEL_AGENT ticks before winding down to SWEEP.

    Env-overridable via ``INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS``. Default 3
    tolerates transient no-work windows (e.g. a candidate being set up) while
    still winding down promptly once candidates are genuinely exhausted, instead
    of spinning until the KERNEL wall-clock cap.
    """
    raw = (_os_env.environ.get("INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS", "") or "").strip()
    try:
        val = int(raw)
        return val if val >= 1 else 3
    except (TypeError, ValueError):
        return 3


KERNEL_IDLE_MAX_TICKS: int = _kernel_idle_max_ticks()


def _kernel_idle_min_seconds() -> float:
    """Wall-clock seconds a KERNEL idle streak must last before winding down.

    Env-overridable via ``INFERENCE_OPTIMIZER_KERNEL_IDLE_MIN_SECONDS``. Ticks
    are the wrong unit on their own: a coordinator tick is a handful of seconds
    and the phase machine advances more than once per tick, so
    :data:`KERNEL_IDLE_MAX_TICKS` alone would wind KERNEL down after roughly
    twenty seconds of quiet — shorter than the gap between a kernel result
    landing and the next dispatch being reasoned out. The default 600s is far
    longer than any dispatch gap a healthy phase produces, while still cutting
    hours off a phase that has genuinely stopped moving.
    """
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
    """Report whether an inline kernel request is executing right now.

    Reads the stamp ``SharedState.kernel_inline_step_seen_unix`` carries, which
    the idle guard has no other way to see. One older than
    :data:`KERNEL_INLINE_STEP_STALE_SECONDS` is a leftover, not a live step.

    Args:
        state: Frozen SharedState view.
        now_unix: Override for the current time.

    Returns:
        ``True`` when an inline kernel step reported itself recently enough.
    """
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

# ``skip_to_sweep`` is the non-terminal "exhausted the current lever" signal:
# from FRAMEWORK_AGENT it advances to KERNEL, from KERNEL it winds down to
# SWEEP → CLOSE.
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
    """Return True for any hint Coordinator should act on (closed vocab).

    Args:
        hint (str): Candidate escalate hint string; stripped before comparison.

    Returns:
        bool: True when ``hint`` is in :data:`ESCALATE_HINT_VOCAB`.
    """
    return (hint or "").strip() in ESCALATE_HINT_VOCAB


def apply_escalate_budget_bump(
    current_budget_pct: dict[str, float] | None,
    *,
    phase: str,
    delta: float = ESCALATE_HINT_BUDGET_BUMP_DELTA,
    cap: float = ESCALATE_HINT_BUDGET_BUMP_CAP,
) -> dict[str, float]:
    """Return a budget map with ``phase`` raised by ``delta`` (capped at 80%).

    Args:
        current_budget_pct (dict[str, float] | None): Existing ``phase -> pct``
            map; ``None`` starts from the defaults.
        phase (str): Phase to raise; stripped and upper-cased. An unknown phase
            returns the input map unchanged.
        delta (float): Percentage-point increment to apply.
        cap (float): Absolute ceiling for the resulting fraction.

    Returns:
        dict[str, float]: A normalized budget map with ``phase`` bumped by
        ``delta`` and clamped to ``[0.0, cap]``.
    """
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
    """Return a sanitized ``phase -> pct`` mapping (budgets are upper bounds, not renormalized to 1.0).

    ``0.0`` is kept, not dropped: it is the sentinel
    :func:`redistribute_budget_pct` writes for a phase the run turned off, and
    overlaying the default back on it would leak that phase's share into the
    charge-back denominator of every earlier phase.

    Args:
        budget (dict[str, float] | None): Raw ``phase -> pct`` overrides;
            unknown phases and out-of-range / unparseable values are dropped.

    Returns:
        dict[str, float]: The defaults overlaid with the valid overrides (each
        in the ``[0.0, 1.0]`` range).
    """
    out = dict(DEFAULT_PHASE_BUDGET_PCT)
    if not budget:
        return out
    for phase, val in budget.items():
        canon = (phase or "").strip().upper()
        if canon not in PHASE_NAMES:
            # An unknown key silently reverts that share to its default,
            # which reads downstream as a choice nobody made.
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
    """Reallocate disabled phases' budget shares to the enabled work phases.

    When a work phase is turned off (``--no-framework-agent`` →
    FRAMEWORK_AGENT, ``--no-kernel`` → KERNEL_AGENT), its ``pct`` is zeroed and
    its freed share is spread across the still-enabled work phases, weighted by
    their base ``pct``. PRELUDE and CLOSE are fixed overhead and never absorb.
    Idempotent: once a phase is 0 its freed share is 0, so re-running per tick
    is a no-op.

    Args:
        base (dict[str, float]): A ``phase -> pct`` map, already sanitized by
            :func:`normalize_budget_pct`.
        optimize_enabled (bool): Whether the merged optimisation phase runs.
        kernel_enabled (bool): Whether the KERNEL_AGENT phase runs.

    Returns:
        dict[str, float]: A new map with disabled phases at 0 and their share
        redistributed to the enabled work phases.
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
    return out


# Pure judgment helpers (used by Coordinator at each tick end)
def _now_unix(state: Any) -> float:
    """Resolve the "now" timestamp; tests can inject ``state._now_unix``.

    Args:
        state (Any): Frozen SharedState view; may expose a callable
            ``_now_unix`` override for tests.

    Returns:
        float: Current time in seconds since the epoch.
    """
    if hasattr(state, "_now_unix") and callable(state._now_unix):
        return float(state._now_unix())  # type: ignore[attr-defined]
    import time as _time

    return _time.time()


def _phase_started_unix(state: Any) -> float:
    """Return the Unix timestamp the current phase started, defensively coerced.

    Reads ``state.phase_started_unix`` and returns ``0.0`` when the field is
    missing or non-numeric (e.g. legacy / partially-initialized state).

    Args:
        state (Any): Frozen SharedState view.

    Returns:
        float: Phase start time in seconds since the epoch, or ``0.0``.
    """
    raw = getattr(state, "phase_started_unix", 0.0)
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _resume_boundary_unix(state: Any) -> float:
    """Return when the current run leg began, i.e. the most recent ``--resume``.

    Args:
        state (Any): Frozen SharedState view exposing ``resumed_ts``.

    Returns:
        float: Leg start in seconds since the epoch, or ``0.0`` for a session
        that has only ever run once (or an unparseable stamp).
    """
    return max(0.0, to_unix(getattr(state, "resumed_ts", ""), 0.0) or 0.0)


def _kernel_idle_since_unix(state: Any) -> float:
    """Return when the current KERNEL idle streak opened, defensively coerced.

    Returns ``0.0`` when the field is missing or non-numeric, which the idle
    guard reads as "no measured idle window" and refuses to act on — a resumed
    or partially-initialised state must not be able to wind the phase down
    before the phase machine has observed a single streak of its own.

    Args:
        state (Any): Frozen SharedState view.

    Returns:
        float: Streak start in seconds since the epoch, or ``0.0``.
    """
    raw = getattr(state, "kernel_idle_since_unix", 0.0)
    try:
        return max(0.0, float(raw or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _pending_escalate_hint(state: Any) -> str:
    """Return a pending escalate hint to act on this tick (unknown hints → empty).

    Args:
        state (Any): Frozen SharedState view exposing ``pending_escalate_hint``.

    Returns:
        str: The pending hint when it is a recognized escalate hint, else ``""``.
    """
    raw = str(getattr(state, "pending_escalate_hint", "") or "").strip()
    if not raw:
        return ""
    if is_valid_escalate_hint(raw):
        return raw
    return ""


def _max_minutes(state: Any) -> float:
    """Return the session's configured ``max_minutes`` budget, defensively coerced.

    A value of ``0.0`` is the conventional "unlimited run" sentinel and is
    also returned when the field is missing or non-numeric.

    Args:
        state (Any): Frozen SharedState view.

    Returns:
        float: Maximum wall-clock minutes for the session, or ``0.0`` for
        unlimited / unparseable.
    """
    try:
        return float(getattr(state, "max_minutes", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _budget_minutes(state: Any) -> float:
    """Wall-clock minutes the PER-PHASE budget fractions apply to (R2).

    The Coordinator sets ``cycle_minutes`` > 0 so each phase's budget
    (``DEFAULT_PHASE_BUDGET_PCT``) is a fraction of ONE macro-cycle's window
    rather than the whole run. When ``cycle_minutes`` is 0 this falls back to
    the total ``max_minutes`` (the whole-session anchor).

    Used by :func:`_phase_budget_total_seconds` as the charge-back *base* (and
    planning cap) for long/unbounded runs (:func:`is_long_run`). A short bounded
    run (``--max-hours < 24``) does not use this — it charges back against the
    remaining session time — so its phases are never silently compressed to the
    cycle window (DEFAULT_CYCLE_HOURS).
    Note: ``session_remaining_seconds`` deliberately keeps using ``max_minutes``
    — the global deadline is per-run, not per-cycle.
    """
    try:
        cm = float(getattr(state, "cycle_minutes", 0) or 0)
    except (TypeError, ValueError):
        cm = 0.0
    if cm > 0 and is_long_run(state):
        return cm
    return _max_minutes(state)


def phase_elapsed_seconds(state: Any, *, now_unix: float | None = None) -> float:
    """Return wall-clock seconds spent in the current phase.

    Returns ``0.0`` when the phase start timestamp is unset (phase not yet
    entered) so callers can treat "not started" as zero elapsed.

    Exiting the process is not a phase transition, so ``phase_started_unix``
    survives a ``--resume`` and the entry it stamps spans both run legs. The
    current leg's boundary (:func:`_resume_boundary_unix`) therefore floors the
    segment: the phase was not executing while nothing was, and a session
    resumed days later would otherwise read as having overspent every phase
    ceiling before it did any work. The floor only applies to the entry the
    stop interrupted — a later entry stamps a newer ``phase_started_unix``.

    The previous leg's own share of that entry is not measured here; a resume
    banks it into ``phase_elapsed_totals`` when the stopped leg recorded when
    it ended.

    Args:
        state (Any): Frozen SharedState view exposing ``phase_started_unix``.
        now_unix (float | None): Override for the current time; defaults to
            :func:`_now_unix` resolution when None.

    Returns:
        float: Non-negative seconds elapsed in the current phase.
    """
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
    """Rebuild per-phase completed-segment totals from a ``phase_history`` log.

    Used when resuming a state written before ``phase_elapsed_totals`` existed.
    Every history row records the phase it entered and when, so consecutive rows
    bound one completed segment. The trailing row is the still-active segment and
    is deliberately excluded: :func:`phase_cumulative_seconds` adds the live
    segment itself, and double-counting it would over-charge the phase.

    The rebuild is an estimate, not a bound in either direction. ``phase_history``
    is capped, so a very long session loses its oldest segments and is
    under-charged. And no row marks the process exiting, so two rows either side
    of a resume bound one "segment" spanning the idle gap between the legs, and
    charge it to the phase the earlier row named — the over-charge direction
    :func:`phase_elapsed_seconds` floors off the live segment. That floor reads
    ``resumed_ts``, which dates the newest leg only; ``phase_history`` dates none
    of them, so the rebuild cannot repeat it.

    Args:
        history (Any): The ``phase_history`` list; any other type yields ``{}``.

    Returns:
        dict[str, float]: Phase name -> completed seconds. Rows with no phase,
        no entry timestamp, or a non-advancing timestamp are skipped.
    """
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
    """Return wall-clock seconds spent in ``phase``, summed over EVERY entry.

    :func:`phase_elapsed_seconds` measures the CURRENT entry only, because
    ``phase_started_unix`` is reset on every phase entry. The absolute cap
    (:func:`phase_cap_exceeded`) bounds the whole run, so it needs this total
    rather than one entry's clock. This reads the durable
    ``phase_elapsed_totals`` banked at each transition out of the phase and adds
    the live segment when ``phase`` is the phase currently running.

    Args:
        state (Any): Frozen SharedState view exposing ``phase_elapsed_totals``.
        phase (str | None): Phase to total; defaults to the current phase.
        now_unix (float | None): Override for the current time.

    Returns:
        float: Non-negative seconds spent in ``phase`` across the whole run.
    """
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
            # A malformed banked total degrades to "nothing banked", i.e. the
            # pre-fix per-entry behaviour for this phase. Under-charging is the
            # only tolerable direction: over-charging would end a phase early.
            accumulated = 0.0
    if target == current:
        accumulated += phase_elapsed_seconds(state, now_unix=now_unix)
    return accumulated


def explore_elapsed_seconds(state: Any, *, now_unix: float | None = None) -> float | None:
    """Return total optimisation-phase wall-clock seconds across all macro cycles.

    Completed segments are accumulated at every transition out of the
    optimisation phase. If it is still current, append the live segment at
    read time so runtime telemetry remains current between transitions. Returns
    ``None`` when a legacy resumed state has no trustworthy historical total.
    """
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
    """Effective TOTAL budget (seconds) allotted to the current phase.

    Bounded runs (a wall-clock ``start_ts`` + ``max_minutes`` are set) use
    "charge-back": the phase gets its share of the time still available,
    renormalized over the current phase and the phases yet to come, so an earlier
    phase's overrun (e.g. a slow PRELUDE) transparently reduces every later
    phase's budget instead of a fixed ``base*pct`` allotment. The charge-back
    *base* differs by session length (:func:`is_long_run`):

    - Short bounded runs charge back against the remaining SESSION time.
    - Long bounded runs charge back against the remaining session time too, but
      the per-cycle window (``cycle_minutes`` via :func:`_budget_minutes`) caps
      the base as a planning ceiling — so one cycle never plans beyond one
      macro-cycle window.

    Unbounded runs (or a state with no parseable ``start_ts``) have no clock to
    charge back against, so they fall back to the flat per-window allotment
    (``_budget_minutes*60*pct``).

    Args:
        state (Any): Frozen SharedState view.
        budget_pct (dict[str, float] | None): Phase-budget overrides; defaults
            to ``state.phase_budget_pct`` when None.
        now_unix (float | None): Override for the current time (shared with
            ``session_remaining_seconds`` / ``phase_elapsed_seconds``).

    Returns:
        float | None: Effective total budget in seconds, ``0.0`` when the phase
        is explicitly allocated no budget, or ``None`` when no finite budget
        applies (unbounded window, or an unset/unrecognized phase).
    """
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
        # Charge-back. remaining_at_entry reconstructs the time left when the
        # phase's live segment opened: within one run leg session_remaining
        # shrinks exactly as phase_elapsed grows, so their sum holds. A resume
        # that keeps ``start_ts`` drops that sum by the idle gap between the
        # legs — the session is charged for it, the phase is not — so a resumed
        # phase charges back against a smaller, honest base.
        remaining_at_entry = max(0.0, session_remaining + phase_elapsed_seconds(state, now_unix=now_unix))
        if is_long_run(state):
            # Long bounded run: the per-cycle window caps the base as a planning
            # ceiling so one cycle never plans beyond one macro-cycle window.
            cycle_window = _budget_minutes(state) * 60.0
            if cycle_window > 0.0:
                remaining_at_entry = min(cycle_window, remaining_at_entry)
        # Normalize ONLY over the current phase and the phases still to come:
        # already-elapsed phases (notably PRELUDE) are excluded — their spend is
        # already reflected in the base — while CLOSE stays in so it keeps its
        # reserved share. Do NOT normalize over all six phases.
        denom = sum(
            float(budget.get(p, 0.0)) for p in PHASE_NAMES[phase_index(phase) :] if float(budget.get(p, 0.0)) > 0.0
        )
        if denom <= 0.0:
            return None
        return remaining_at_entry * pct / denom
    # No session clock (unbounded run, or ``start_ts`` unset): fall back to the
    # flat per-window allotment — charge-back needs a wall-clock reference.
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
    """Return seconds left in the current phase ENTRY's budget (``None`` when budget window 0 = unlimited).

    Charges :func:`phase_elapsed_seconds`, matching the allotment's basis:
    :func:`_phase_budget_total_seconds` charges back against the clock at this
    entry's start, so earlier entries are already priced into it. Subtracting
    the cumulative total as well bills them twice, which pins a re-entered phase
    at ``0`` for the rest of the run however much session is left.

    Lifetime spend is bounded by :func:`phase_cap_exceeded` instead; every exit
    predicate that reads this checks that cap alongside it.

    Args:
        state (Any): Frozen SharedState view.
        budget_pct (dict[str, float] | None): Phase-budget overrides; defaults
            to ``state.phase_budget_pct`` when None.
        now_unix (float | None): Override for the current time.

    Returns:
        float | None: Non-negative seconds left in this entry's budget, or
        ``None`` when the budget window is unlimited or the phase has no
        allocated fraction.
    """
    total = _phase_budget_total_seconds(state, budget_pct=budget_pct, now_unix=now_unix)
    if total is None:
        return None
    return max(0.0, total - phase_elapsed_seconds(state, now_unix=now_unix))


def effective_max_minutes(state: Any) -> float:
    """Session minutes for deadline/cap math; unbounded runs use the 14-day ceiling.

    Args:
        state (Any): Frozen SharedState view exposing ``max_minutes``.

    Returns:
        float: ``max_minutes`` when positive, else
        :data:`DEFAULT_LONGRUN_MAX_MINUTES` (the unbounded-run ceiling).
    """
    mm = _max_minutes(state)
    return mm if mm > 0 else float(DEFAULT_LONGRUN_MAX_MINUTES)


def phase_cap_seconds(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
) -> float | None:
    """Absolute wall-clock ceiling (seconds) for the current phase.

    Independent of the per-cycle budget window so it still fires when
    ``max_minutes`` is 0 (unbounded), where ``phase_budget_remaining_seconds``
    returns ``None``. Equals the smaller of the session-derived term and a
    fixed 24h reference, each scaled by the phase budget fraction: short bounded
    runs bind on the (smaller) session term, long/unbounded runs bind on the
    24h reference so no single phase can monopolise the run. (Emergent from
    ``min(proportional, abs_cap)`` — this cap does not branch on
    :func:`is_long_run`.)

    Returns:
        float | None: Cap in seconds, ``0.0`` when the phase is explicitly
        allocated no budget, or ``None`` when the phase is unset/unrecognized.
    """
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
    """True when time spent in the current phase has reached its absolute cap.

    Measures CUMULATIVE time across every entry into the phase
    (:func:`phase_cumulative_seconds`). :func:`phase_cap_seconds` exists so "no
    single phase can monopolise the run", which is a statement about the run —
    comparing it against the current entry alone made it re-arm on each re-entry
    and left the cap unenforceable in a cyclic session.

    Args:
        state (Any): Frozen SharedState view.
        budget_pct (dict[str, float] | None): Phase-budget overrides; defaults
            to ``state.phase_budget_pct`` when None.
        now_unix (float | None): Override for the current time.

    Returns:
        bool: True when cumulative phase time has reached the absolute cap;
        False when no cap applies.
    """
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

    Prefers a stamped ``deadline_unix`` when present so this agrees with
    admission and the Coordinator loop, even when persisted ``max_minutes``
    was truncated to 0. Falls back to ``start_ts + max_minutes`` for sessions
    that predate the stamp.

    Args:
        state (Any): Frozen SharedState view exposing ``max_minutes`` and
            ``start_ts``.
        now_unix (float | None): Override for the current time; the deadline is
            subtracted from this when stamped, otherwise compared against
            ``start_ts``.

    Returns:
        float | None: Non-negative seconds left in the session, or ``None``
        when unbounded (no stamp and ``max_minutes`` is 0) or ``start_ts`` is
        missing/unparseable.
    """
    try:
        deadline = float(getattr(state, "deadline_unix", 0.0) or 0.0)
    except (TypeError, ValueError):
        deadline = 0.0
    if deadline > 0.0:
        now = float(now_unix) if now_unix is not None else time.time()
        return max(0.0, deadline - now)
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
        # Honor an injected now_unix so this stays in the same time source as
        # phase_elapsed_seconds(now_unix=...) for pure/testable budget math.
        if now_unix is not None:
            now_dt = datetime.fromtimestamp(float(now_unix), tz=timezone.utc)
        else:
            now_dt = datetime.now(timezone.utc)
        elapsed_sec = max(0.0, (now_dt - start).total_seconds())
    except (ValueError, TypeError):
        return None
    return max(0.0, mm * 60.0 - elapsed_sec)


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
    """Real plateau_explore → ``(triggered, evidence)``.

    Trigger (AND): recent_keep_gain < threshold, and the arm is out of new
    things to try. The second half reads the trailing empty specialist-round
    streak, or — for a run with no research lane, which records no specialist
    round at all — the count of variants benched this cycle.

    Args:
        state (Any): Frozen SharedState view exposing ``explore_search`` and
            ``specialist_rounds``.
        lookback (int): Window of recent winners to sum for keep-gain;
            non-positive disables the judgment.
        keep_gain_threshold_pct (float): Keep-gain floor below which the gain
            arm trips.
        empty_streak_threshold (int): Trailing empty specialist-round count
            that trips the streak arm.

    Returns:
        tuple[bool, dict[str, Any]]: ``(triggered, evidence)`` — whether the
        plateau fired, and the supporting evidence map.
    """
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
        """Return True when a specialist-round summary produced no work.

        Args:
            row (Any): A specialist-round summary; non-dicts count as
                non-empty (False) so malformed rows break the streak.

        Returns:
            bool: True when both the proposal and kept counts are zero.
        """
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
    """Real plateau_kernel → ``(triggered, evidence)``.

    Trigger (OR, weaker than the config arm's AND): revert_streak
    >= threshold OR recent_keep_gain < keep_gain_threshold_pct.

    Args:
        state (Any): Frozen SharedState view exposing
            ``kernel_integrate_attempts``.
        lookback (int): Window of recent integrate attempts to inspect;
            non-positive disables the judgment.
        revert_streak_threshold (int): Trailing REVERT/NEEDS_REVIEW streak that
            trips the streak arm; non-positive disables the judgment.
        keep_gain_threshold_pct (float): Keep-gain floor below which the gain
            arm trips.

    Returns:
        tuple[bool, dict[str, Any]]: ``(triggered, evidence)`` — whether the
        plateau fired, and the supporting evidence map.
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


# Statuses on last_conc_sweep that exit_normal_sweep already treats as SWEEP
# closeout. skip_to_close must not override those: the LLM emits it when the
# sweep was refused, and mapping that to robustness_escalated turns a
# successful run into a CI failure.
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

    Priority: 1. ``skip_to_close`` → ``robustness_escalated``, except in SWEEP
    when a sweep/conc_sweep closeout is already recorded (the honest SWEEP
    exit wins); 2. Coordinator ``stop_reason``.

    Args:
        state (Any): Frozen SharedState view exposing ``stop_reason`` and any
            pending escalate hint.

    Returns:
        tuple[str, dict[str, Any]] | None: ``(stop_reason, evidence)`` for a
        phase-orthogonal stop, or ``None`` when none applies.
    """
    hint = _pending_escalate_hint(state)
    if hint == ESCALATE_HINT_SKIP_TO_CLOSE:
        current = (getattr(state, "phase", "") or "").strip().upper()
        if current == PHASE_SWEEP and _sweep_has_recorded_closeout(state):
            return None
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


def _closing_phase_terminal(state: Any) -> tuple[str, dict[str, Any]] | None:
    """Return a CLOSE stop when the wall-clock path has entered the closing phase.

    PRELUDE with a baseline otherwise advances to FRAMEWORK_AGENT, whose
    allowlist does not include ``report``. That transition would cancel the
    closing report ``_enter_closing_phase`` just enqueued.

    Args:
        state (Any): Frozen SharedState view exposing ``closing_phase``.

    Returns:
        tuple[str, dict[str, Any]] | None: ``("time_exhausted", evidence)``
        when the closing phase is active, else ``None``.
    """
    if not bool(getattr(state, "closing_phase", False)):
        return None
    return "time_exhausted", {"reason_origin": "closing_phase"}


# per-phase judgments
def warm_replay_in_flight(state: Any) -> bool:
    """True while the PRELUDE warm-recipe replay task has not finished (PRELUDE must not exit until False — GPU contention).

    Args:
        state (Any): Frozen SharedState view exposing ``warm_replay_outcome``.

    Returns:
        bool: True when the warm-replay outcome status is ``in_flight``.
    """
    outcome = getattr(state, "warm_replay_outcome", None) or {}
    if not isinstance(outcome, dict):
        return False
    return str(outcome.get("status") or "").strip() == "in_flight"


def _kernel_opt_max_failures() -> int:
    """Resolve the kernel infra-failure retry budget (lazy import)."""
    from ..state.shared_state import resolve_kernel_opt_max_failures

    return resolve_kernel_opt_max_failures()


# The statuses a finished GEAK run writes to ``geak_result.status``. Named
# rather than inlined because the breakdown layer has to agree with it: its
# ``geak_runs`` projection maps each of these onto a closed enum and warns about
# anything it does not know, so a status added here and nowhere else surfaces as
# vocabulary drift on a run that was working exactly as intended. See
# ``breakdown/collectors/v6_stages._GEAK_STATUS_ALIASES``, which is pinned
# against this set by test.
#
# Not exhaustive of what can appear in the field: the collector adds its own
# words for a result it could not read back (``missing``,
# ``no_result_recovered_from_disk``), and ``timeout`` reaches it from the runner
# without being terminal here.
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


# Ledger subfields that change when a kernel attempt actually advances. Listed
# explicitly rather than digesting whole entries so incidental churn (a
# re-rendered field, a refreshed timestamp) cannot masquerade as forward
# progress and keep restarting the idle streak.
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

# ``last_kernel_opt`` subfields that identify WHICH result is the latest one; a
# new result always changes at least one of them.
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
    """Digest the KERNEL signals that change if and only if something moved.

    The idle-spin guard needs to know whether the phase is making forward
    progress. :func:`kernel_work_pending` cannot answer that — it reports whether
    the ledger still holds anything unresolved, so attempts that can never be
    advanced keep answering "yes" indefinitely. A digest over the observable
    outcome fields answers the right question: an unchanged digest means nothing
    happened between two ticks.

    In-flight task ids are part of the digest so a dispatch starting or a task
    finishing both count as progress in their own right, even before the ledger
    records an outcome.

    Args:
        state (Any): Frozen SharedState view exposing the kernel ledgers.
        inflight_task_ids (Any): Task ids of kernel-lane work currently queued or
            running; any iterable of strings.

    Returns:
        str: A stable hex digest. Equal digests on consecutive ticks mean no
        observable kernel progress happened in between.
    """
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
    last_collective = getattr(state, "last_collective", None)
    if last_collective is None:
        last_collective = {}
    if not isinstance(last_collective, dict):
        raise ValueError("last_collective must be a mapping")
    payload = {
        "attempts": attempts,
        "inflight": sorted(str(task_id) for task_id in (inflight_task_ids or ())),
        "last_kernel_opt": [str(last_opt.get(field, "")) for field in _LAST_KERNEL_OPT_PROGRESS_FIELDS],
        "pending_integrations": sorted(str(key) for key in pending) if isinstance(pending, dict) else [],
        "rejected": sorted(str(kid) for kid in (getattr(state, "rejected_kernel_ids", None) or [])),
        "stack_len": len(stack) if isinstance(stack, list) else 0,
        "last_collective": [
            str(last_collective.get(field, ""))
            for field in (
                "collective_attempt_id",
                "status",
                "decision",
                "patch_cleanup_status",
                "integration_decision",
                "patch_cleanup_action",
                "integration_revert_status",
                "integration_finalize_status",
            )
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def collective_integration_pending(state: Any) -> bool:
    """Return whether a kept collective still requires terminal E2E handling."""
    last = getattr(state, "last_collective", None)
    if last in (None, {}):
        return False
    if not isinstance(last, dict):
        raise ValueError("last_collective must be a mapping")
    kept = last.get("kept", False)
    requires_e2e = last.get("requires_e2e_validation", False)
    if not isinstance(kept, bool) or not isinstance(requires_e2e, bool):
        raise ValueError("collective E2E flags must be boolean")
    if kept != requires_e2e:
        raise ValueError("collective E2E flags are inconsistent")
    # Fall back to legacy field name for --resume compat.
    cleanup = str(last.get("patch_cleanup_status") or last.get("integration_status") or "")
    return kept and cleanup != "complete"


def kernel_auto_pass_complete(state: Any) -> bool:
    """Whether a nomination pass already ran to completion this macro cycle.

    Stored with the cycle it belongs to rather than as a bare flag, so entering
    the next cycle retires it without anyone having to remember to clear it.

    Args:
        state: Shared state carrying the marker and the current cycle.

    Returns:
        True when this cycle's pass is done.
    """
    marker = getattr(state, "kernel_auto_pass_cycle", None)
    if marker is None:
        return False
    try:
        return int(marker) == int(getattr(state, "macro_cycle", 0) or 0)
    except (TypeError, ValueError):
        return False


def mark_kernel_auto_pass_complete(state: Any) -> None:
    """Record that this cycle's nomination pass finished, whatever it selected.

    Called for an empty selection too: "nobody wanted anything" is a completed
    pass, and it is exactly the case the phase used to hang on.
    """
    state.kernel_auto_pass_cycle = int(getattr(state, "macro_cycle", 0) or 0)


def kernel_work_pending(state: Any) -> bool:
    """Return True while KERNEL has work that can still affect validated gain.

    This guards the non-terminal ``skip_to_sweep`` handoff: a plateau hint should
    not end KERNEL while a KEEP still needs integrate, or while a kernel-agent
    attempt is only partially recorded, or while trace analysis still exposes
    hot reusable kernels that have not received a kernel_opt attempt. Hard
    time/budget exits are still handled by :func:`exit_normal_kernel`.

    Short-circuits in order: a pending collective integration keeps the phase
    open; ``collective_only_mode`` then answers False because no other lane may
    run; a terminal GEAK phase answers on its own (True only while an ``ok``
    result has an ``awaiting_rebench`` pending with a revalidation task, else
    False); then the optional ``has_keep_pending_integrate`` and
    ``untried_hot_reusable_kernels`` capability probes, whose failures are
    treated as 'not available'; then the kernel_opt attempt ledger, filtered by
    task group, source file, integration status and rejected kernel ids.
    """
    if collective_integration_pending(state):
        return True
    if bool(getattr(state, "collective_only_mode", False)):
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
        if bool(getattr(state, "has_keep_pending_integrate", False)):
            return True
    except Exception:
        # Optional capability probe; treat a failure as 'not available'.
        pass

    # A kernel a nominator looked at and passed over leaves no ledger row, is
    # not rejected, and shares no source with anything integrated -- so it stays
    # "untried" forever and this predicate never goes quiet. Once a pass has run
    # to completion for the cycle, its verdict is what counts: a still-untried
    # kernel means nobody wants it, not that work is outstanding.
    if not kernel_auto_pass_complete(state):
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
        if str(entry.get("action") or "") in {"integrate", "collective"}:
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


def enablement_engaged(state: Any) -> bool:
    """Whether an enablement round has started and is still making progress.

    While engaged, repeated baseline boot failures are forward progress (each
    round clears a deeper gap), so the ``baseline_failed`` fast-fail must stand
    down and let the ``enablement_stalled`` cap terminate instead. Always false
    when the session did not admit either enablement lane.

    Args:
        state (Any): Frozen SharedState view exposing ``enablement_mode`` and the
            ``enablement_*`` progress fields.

    Returns:
        bool: ``True`` when an enablement round is stacked, dispatched or tried.
    """
    from ..actions.executors._accuracy_gate import ENABLEMENT_MODE_OFF, resolve_enablement_mode

    if resolve_enablement_mode(state) == ENABLEMENT_MODE_OFF:
        return False
    return bool(
        (getattr(state.enablement, "kept_patches", None) or [])
        or getattr(state.enablement, "inflight_task_id", "")
        or int(getattr(state.enablement, "attempts", 0) or 0) > 0
    )


def exit_normal_prelude(state: Any) -> tuple[str, dict[str, Any]] | None:
    """``baseline_tput > 0`` and warm-replay settled → ``prelude_done`` (else ``None``).

    A figure whose hot pass was dropped for budget does not finish preparation,
    even though it is a figure. It is depressed by the boot, the compile and the
    graph capture it could not discard, so declaring PRELUDE done on it would
    hand the optimization phases a denominator that was never the baseline. The
    phase stays open instead, and what happens next is decided by the budget:
    :func:`exit_cold_anchor_prelude` closes a session that still cannot afford a
    comparable baseline, and one resumed with a fresh clock measures another.

    Args:
        state (Any): Frozen SharedState view exposing ``baseline_tput`` and the
            warm-replay outcome.

    Returns:
        tuple[str, dict[str, Any]] | None: ``("prelude_done", evidence)`` when
        the baseline succeeded and warm-replay has settled, else ``None``.
    """
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
    """Read a duration an earlier round measured, or ``None`` when none did.

    Every budget decision that prices work off a measurement has to tell "no
    round has run" apart from "a round ran and took no time", because the first
    means the decision cannot be made and the second would make everything look
    free.

    Args:
        state (Any): Frozen SharedState view.
        field (str): The ``SharedState`` attribute holding the duration.

    Returns:
        float | None: The duration when one was measured, else ``None``.
    """
    try:
        value = float(getattr(state, field, 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def boot_cost_sec(state: Any) -> float | None:
    """What bringing this workload's server up costs, or ``None`` when unmeasured.

    The baseline's cold round is the one round that pays this in the open, so it
    is where the figure comes from: its wall-clock less the part of it that ran
    after the server reported ready.

    Args:
        state (Any): Frozen SharedState view.

    Returns:
        float | None: Seconds spent before the server was ready, or ``None`` when
        no round has reported the split (no baseline yet, or a scriptable
        workload, which runs no server and so has no boot to separate).
    """
    total_sec = measured_seconds(state, "baseline_runtime_sec")
    post_ready_sec = measured_seconds(state, "baseline_post_ready_runtime_sec")
    if total_sec is None or post_ready_sec is None:
        return None
    return max(0.0, total_sec - post_ready_sec)


def benchmark_cost_sec(state: Any) -> float | None:
    """What one benchmark pass costs on a server already up, or ``None``.

    Two figures can answer this and they are not equally good, so the better one
    wins when it exists:

    * The measured hot pass (``baseline_warm_runtime_sec``) is the answer, being
      exactly a benchmark against a server someone else booted.
    * The cold round's post-ready segment is the fallback, and it over-predicts:
      that segment also pays the first request's kernel compile, which a pass on
      a now-populated JIT cache does not. It is what a session has before its hot
      pass runs, and over-predicting a benchmark is a smaller error than pricing
      one at a whole cold round.

    Args:
        state (Any): Frozen SharedState view.

    Returns:
        float | None: Seconds one benchmark pass costs, or ``None`` when neither
        figure has been measured.
    """
    hot_sec = measured_seconds(state, "baseline_warm_runtime_sec")
    if hot_sec is not None:
        return hot_sec
    return measured_seconds(state, "baseline_post_ready_runtime_sec")


def baseline_round_cost_sec(state: Any, *, double_run: bool) -> float | None:
    """What a baseline round costs, or ``None`` when unmeasured.

    A round is one or two passes and they are priced apart, because they buy
    different things. The first brings a server up and benchmarks it cold, paying
    the first request's kernel compile on the way; the session has measured that
    whole pass directly, so it is read rather than reconstructed. The second
    re-attaches to the server the first left running, so it costs a benchmark and
    no second boot.

    Reconstructing the first pass as boot-plus-hot-benchmark would drop the
    compile and under-price the round, which matters because this figure guards
    ignition while the post-warmup gate that follows prices the same work from
    the pass it just watched. A round admitted here and then certainly refused
    there costs a whole cold pass to learn nothing.

    Reading the measured total also gives a price to rounds with no boot/benchmark
    split to reconstruct from -- multi-node and scriptable workloads, which never
    report one -- so those are gated rather than waved through.

    Args:
        state (Any): Frozen SharedState view.
        double_run (bool): Whether the round runs a warmup pass and a measured
            one, or a single pass.

    Returns:
        float | None: Seconds the round costs, or ``None`` when the session has
        measured nothing to price it from.
    """
    first_pass_sec = measured_seconds(state, "baseline_runtime_sec")
    if first_pass_sec is None or not double_run:
        return first_pass_sec
    second_pass_sec = benchmark_cost_sec(state)
    return first_pass_sec if second_pass_sec is None else first_pass_sec + second_pass_sec


def one_more_measurement_sec(state: Any) -> float | None:
    """What the next measured variant will cost, or ``None`` when unmeasured.

    A variant is not a benchmark; it is a boot and then a benchmark. Its config
    differs from the baseline's in the very knobs that decide how a server comes
    up -- parallelism, quantization, kernel backends -- so it cannot re-attach to
    a server already running and has to bring up its own.

    This is the unit every "is there time left to use a result" question is asked
    in, because a result nothing gets measured against is a result the session
    could not have used.

    Args:
        state (Any): Frozen SharedState view.

    Returns:
        float | None: Seconds one further measured variant costs, or ``None``
        when either half of it is unmeasured.
    """
    boot_sec = boot_cost_sec(state)
    benchmark_sec = benchmark_cost_sec(state)
    if boot_sec is None or benchmark_sec is None:
        return None
    return boot_sec + benchmark_sec


def prelude_exit_viability(state: Any) -> dict[str, Any]:
    """Report whether the budget PRELUDE leaves behind can still fund one optimization round.

    A session can honour its wall clock and still be over: both field sessions
    left FRAMEWORK_AGENT ~47 minutes against a 108-minute threshold, and every
    later phase then declined in turn, each for its own local reason. Nothing
    said the plain thing — preparation had spent the run.

    Stated here, on the exit that caused it, because this is the last moment
    the answer is actionable and the first moment it is knowable: the baseline
    is measured, so one benchmark round has a price rather than an estimate.

    Priced as what an optimization round actually is -- one boot and one
    benchmark (:func:`one_more_measurement_sec`) -- rather than as the baseline's
    whole cold wall-clock, which also carries the first request's compile and so
    over-reports a prepared session as a spent one. The cold figure remains the
    fallback for a session whose baseline reported no split, with ``priced_by``
    naming which ruler answered so two runs' evidence cannot be compared as
    though they used the same one.

    Args:
        state (Any): Frozen SharedState view.

    Returns:
        dict[str, Any]: Evidence for the phase record; empty when the budget is
        unbounded or no round has been measured.
    """
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
    """Append ``row`` to the current phase's ``evidence[key]`` list.

    Phases record what happened inside them on the newest ``phase_history``
    row, which the session breakdown exports verbatim. Creates the ``evidence``
    dict and the list under ``key`` when absent, and replaces a non-list value
    rather than raising — a malformed row must not take a phase down.

    Args:
        history (Any): The ``phase_history`` list.
        key (str): Evidence key holding the list of rows.
        row (dict[str, Any]): The row to append.

    Returns:
        bool: ``True`` when the row landed; ``False`` when there was no usable
        history row to attach it to.
    """
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
    """Seconds a unit of work may still claim, from the session's own accounting.

    Prefers ``SharedState.session_budget_usable_sec`` — the single number
    admission control and the grid deadline both read — so this policy cannot
    disagree with them about how much budget is left. Falls back to the raw
    remaining time for the frozen views and test doubles that expose attributes
    only. Called rather than imported because ``shared_state`` imports this
    module.

    Args:
        state (Any): Frozen SharedState view.

    Returns:
        float | None: Usable seconds, or ``None`` when the budget is unbounded.
    """
    getter = getattr(state, "session_budget_usable_sec", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001 — fall back to the attribute path
            pass
    return session_remaining_seconds(state)


def prelude_affordable_seconds(state: Any) -> tuple[float | None, dict[str, Any]]:
    """Seconds PRELUDE may still spend, and the numbers the figure is built from.

    What is left on the session clock once the optimization phases' reserve
    (:data:`OPTIMIZATION_RESERVE_PCT`) is held back, measured against the same
    usable remainder every other budget decision reads, so the figure survives
    a resume that reanchors the budget. Work that does not fit is not work the
    session needed — it is work the session could not have used the result of.

    Read directly by callers that have to *size* a unit of work rather than
    judge one they can already price, which is the position a first baseline is
    in: it has no measured runtime to judge against, but the share it may spend
    is known before anything runs.

    Args:
        state (Any): Frozen SharedState view.

    Returns:
        tuple[float | None, dict[str, Any]]: The affordable seconds — which may
        be negative once the reserve is eaten into — or ``None`` on an
        unbounded budget, plus the evidence behind it.
    """
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
    """Decide whether PRELUDE can still buy an optional arm costing ``expected_cost_sec``.

    The share itself is :func:`prelude_affordable_seconds`; this judges one cost
    against it.

    Args:
        state (Any): Frozen SharedState view.
        expected_cost_sec (float): What the arm is expected to cost. Callers
            should anchor this on something this session measured; the static
            per-action estimates are calibrated on small models and understate
            a large one by an order of magnitude.

    Returns:
        tuple[bool, dict[str, Any]]: ``(affordable, evidence)``. Evidence
        carries the numbers behind the decision so a skip is legible in the log
        and the phase record.
    """
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
    """Route to CLOSE when the session clock runs out before PRELUDE lands a baseline.

    ``time_exhausted_during_prelude`` was in the terminal-reason vocabulary and
    in the report's reason glossary, but no code ever assigned it: the state
    machine had a word for this failure and no way to reach it. A session that
    burns its whole budget preparing then read as an ordinary exit.

    Only fires while PRELUDE is still incomplete. Once a baseline exists the
    later phases carry their own budgets, and reporting the run as "never
    began optimizing" would be false.

    Args:
        state (Any): Frozen SharedState view.
        now_unix (float | None): Override for the current time.

    Returns:
        tuple[str, dict[str, Any]] | None: ``("time_exhausted_during_prelude",
        evidence)`` when the budget is gone, else ``None``.
    """
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
    """Route to CLOSE when PRELUDE could only produce a cold anchor.

    A baseline's hot pass is dropped when the budget cannot cover it together with
    one variant to read against it. What survives is the cold pass's figure, and
    it is depressed: it carries the server boot, the first request's kernel
    compile and the graph capture in its throughput denominator.

    Continuing on it is worse than stopping. Every variant measured against a
    depressed denominator reads as an improvement over a baseline that was never
    the baseline, so the session would spend the rest of its clock producing
    findings that a later run cannot reproduce. Stopping keeps the number and the
    marker that says what it is.

    Fires only on the dropped-pass marker, not on a cold figure as such: a session
    configured for a single-round baseline reports a cold figure by design, and
    its comparisons are consistent because everything downstream is measured the
    same way.

    And only while the budget still cannot buy a comparable baseline. A session
    resumed with a fresh clock carries the earlier leg's marker but not its
    shortfall, and it can now do the thing it was stopped for: measure a hot
    baseline. Firing on the marker alone would send it straight back to CLOSE on
    the strength of a constraint that no longer holds, and no later baseline could
    ever clear the marker because none would run.

    Args:
        state (Any): Frozen SharedState view.

    Returns:
        tuple[str, dict[str, Any]] | None: ``("prelude_cold_anchor_low_budget",
        evidence)`` when the hot pass was dropped and still cannot be afforded,
        else ``None``.
    """
    if not bool(getattr(state, "baseline_measure_round_dropped", False)):
        return None
    usable = session_usable_seconds(state)
    if usable is None:
        return None
    # Without the boot/benchmark split -- a scriptable workload runs no server, so
    # it has no ready boundary to split on -- the cold round's whole wall-clock
    # stands in for each half, the same upper bound the round's own gate falls
    # back to. Undecidable rather than assumed only when nothing was measured at
    # all, which cannot happen with the marker set but is not worth closing on.
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
    """Decide the PRELUDE terminal exit on repeated baseline failures.

    Fires once the consecutive baseline-failure streak reaches 3, routing
    the session straight to CLOSE with ``prelude_baseline_failed``.

    Enablement-aware: suppressed while enablement is actively engaged (a
    progressing patch stacked, a specialist dispatched, or ≥1 attempt made),
    because serial-enablement baseline crashes are forward progress and the
    ``enablement_stalled`` cap is the correct fast-fail there.

    Args:
        state (Any): Frozen SharedState view exposing ``baseline_failure_streak``
            and the ``enablement_*`` progress fields.

    Returns:
        tuple[str, dict[str, Any]] | None: ``("prelude_baseline_failed",
        evidence)`` when the streak threshold is met and enablement is not
        engaged, else ``None``.
    """
    streak = int(getattr(state, "baseline_failure_streak", 0) or 0)
    if streak >= 3 and not enablement_engaged(state):
        return "prelude_baseline_failed", {"baseline_failure_streak": streak}
    return None


def exit_normal_kernel(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """KERNEL normal exit.

    Priority: 1. ``skip_to_close`` defers to global terminal; 2. ``skip_to_sweep``
    → ``kernel_no_more_leverage`` (non-terminal); 3. phase budget exhausted.

    Args:
        state (Any): Frozen SharedState view.
        budget_pct (dict[str, float] | None): Phase-budget overrides; defaults
            to ``state.phase_budget_pct`` when None.
        now_unix (float | None): Override for the current time.

    Returns:
        tuple[str, dict[str, Any]] | None: ``(reason, evidence)`` for the KERNEL
        exit, or ``None`` when KERNEL should continue.
    """
    if _pending_escalate_hint(state) == ESCALATE_HINT_SKIP_TO_SWEEP:
        if not kernel_work_pending(state):
            return "kernel_no_more_leverage", {
                "evidence": "kernel_no_more_leverage",
                "hint": ESCALATE_HINT_SKIP_TO_SWEEP,
            }
    # Idle-spin guard: the escalate-hint handoff above needs the kernel_agent to
    # emit ``escalate_strategy_change``, but PolicyGate denies that intent for the
    # kernel_agent role — so when the phase stops moving it can otherwise spin
    # (hallucinated kernel-id requests / no-intent turns) until the wall-clock cap.
    #
    # The streak is deliberately NOT gated on ``kernel_work_pending``. That
    # predicate reports whether the ledger still holds anything unresolved, so a
    # session carrying attempts that can never be advanced answers "yes" forever
    # and pins the streak at zero — which is exactly how a real run spun for
    # 10.4h with every GPU idle. ``kernel_idle_ticks`` / ``kernel_idle_since_unix``
    # are maintained per-tick by the phase machine off an observable progress
    # fingerprint, and only advance when nothing changed AND no kernel-lane task
    # is in flight.
    #
    # Both a tick count and a wall-clock floor must be satisfied: ticks prove we
    # sampled repeatedly, the floor proves enough real time passed that a healthy
    # dispatch gap cannot be mistaken for a stall.
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


def exit_normal_sweep(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """SWEEP normal exit: the concurrency ladder's terminal state, or budget exhausted.

    The ladder is the only sweep, so its status is the phase's.

    Args:
        state (Any): Frozen SharedState view exposing ``last_conc_sweep``.
        budget_pct (dict[str, float] | None): Phase-budget overrides; defaults
            to ``state.phase_budget_pct`` when None.
        now_unix (float | None): Override for the current time.

    Returns:
        tuple[str, dict[str, Any]] | None: ``(reason, evidence)`` for the SWEEP
        exit, or ``None`` when SWEEP should continue.
    """
    last_conc = getattr(state, "last_conc_sweep", None) or {}
    if isinstance(last_conc, dict):
        status = str(last_conc.get("status") or "").lower()
        if status == "failed":
            return "sweep_failed", {"sweep_status": status}
        if status in ("succeeded", "partial", "completed", "skipped"):
            evidence: dict[str, Any] = {"sweep_status": status}
            # A sweep that declined to run is also terminal, and the exit
            # reason alone cannot tell the two apart afterwards. was_skipped
            # covers both declining and spending the whole budget without a
            # comparable pair, so it is only carried with the flag that
            # separates them (see kernel.conc_sweep.conc_sweep_declined_to_run).
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
    """Pull operator-tuned plateau thresholds off
    :attr:`SharedState.plateau_overrides` (empty → library defaults).

    Args:
        state (Any): Frozen SharedState view exposing ``plateau_overrides``.

    Returns:
        dict[str, Any]: A copy of the overrides mapping, or an empty dict when
        unset or non-dict.
    """
    overrides = getattr(state, "plateau_overrides", None) or {}
    return dict(overrides) if isinstance(overrides, dict) else {}


# stops re-selecting the candidate, and is skipped by the plateau streak because
# an infrastructure failure is not evidence that the search is exhausted.
_FRAMEWORK_DISPATCH_FAILED_STATUS = "dispatch_failed"


def framework_agent_consecutive_no_keep(state: Any) -> int:
    """Count trailing consecutive resolved candidates that did not KEEP.

    Walks ``framework_agent_phase_progress`` newest-first: a KEEP row breaks the
    streak; any other terminal row increments it — both ``reverted`` (applied +
    benchmarked, below floor) and the non-benchmarked terminal outcomes
    (``not_applicable`` / ``apply_failed`` / ``authored_empty`` / ``no_patches``
    / ``already_present`` / audit-skip / ``critic_denied``). Counting the latter
    lets a batch of dead candidates trip the plateau instead of grinding on.

    ``dispatch_failed`` is the exception: the specialist never ran, so the row
    carries no information about whether the search still has leverage.

    Args:
        state (Any): Frozen SharedState view exposing
            ``framework_agent_phase_progress``.

    Returns:
        int: Number of trailing consecutive resolved candidates without a KEEP.
    """
    progress = getattr(state, "framework_agent_phase_progress", None) or []
    if not isinstance(progress, list):
        return 0
    count = 0
    for row in reversed(progress):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip().lower()
        # Macro-cycle boundary marker stops the streak walk so a prior cycle's
        # trailing no-KEEP rows cannot instantly re-plateau the next cycle.
        if status == "cycle_boundary":
            break
        # A specialist that never ran produced no search result to plateau on.
        # Transparent to the walk: rows on either side still add up, and a KEEP
        # behind one still breaks the streak.
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
    """Whether the source arm (candidates, authored patches) has run dry.

    Args:
        state (Any): Frozen SharedState view.

    Returns:
        tuple[bool, dict[str, Any]]: ``(plateaued, evidence)``.
    """
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
    """OPTIMIZE normal exit.

    The phase carries two arms — configuration search and source/upstream
    landing — and leaves only when **both** have run dry. A single arm going
    quiet is a reason to switch levers inside the phase, not to abandon the
    other one: that is what the two separate phases got wrong, because a
    wall-clock cap rotated the lever whether or not the arm was still paying.

    ``switch_bottleneck`` rides out whenever either arm plateaued, so the next
    macro-cycle steers off the bottleneck this one exhausted.

    Priority: 1. an explicit escalate hint; 2. both arms plateaued; 3. phase
    budget exhausted; 4. absolute phase cap.

    Args:
        state (Any): Frozen SharedState view.
        budget_pct (dict[str, float] | None): Phase-budget overrides.
        now_unix (float | None): Override for the current time.
        plateau_lookback (int): Config-arm trailing rounds inspected.
        plateau_keep_gain_threshold_pct (float): Config-arm KEEP-gain floor.
        plateau_empty_streak_threshold (int): Config-arm empty-round streak.

    Returns:
        tuple[str, dict[str, Any]] | None: ``(reason, evidence)``, or ``None``
        when the phase should continue.
    """
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
        # Honoured only once the phase has actually run something this cycle: a
        # phase that dispatched nothing must not end with zero validated work.
        # Either arm's work counts, so a grid-only run can still be skipped.
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
    """First active phase after PRELUDE: OPTIMIZE, else KERNEL, else SWEEP
    (``--no-framework-agent`` / ``--no-kernel`` collapse the chain).

    Args:
        optimize_enabled (bool): Whether the OPTIMIZE phase is enabled.
        kernel_enabled (bool): Whether the KERNEL_AGENT phase is enabled.

    Returns:
        str: ``PHASE_FRAMEWORK_AGENT``, ``PHASE_KERNEL_AGENT``, or ``PHASE_SWEEP`` depending on
        which phases are enabled.
    """
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
    """Return ``(next_phase, reason, evidence)`` or ``None``.

    Priority (Inv-8.2): global terminal first, then the wall-clock closing
    phase, then exit_terminal > exit_normal.

    Args:
        state (Any): Frozen SharedState view exposing the current ``phase``.
        kernel_enabled (bool): Whether the KERNEL_AGENT phase is enabled.
        budget_pct (dict[str, float] | None): Phase-budget overrides; defaults
            to ``state.phase_budget_pct`` when None.
        now_unix (float | None): Override for the current time.
            after PRELUDE.
        optimize_enabled (bool): Whether the OPTIMIZE phase is enabled.

    Returns:
        tuple[str, str, dict[str, Any]] | None: ``(next_phase, reason,
        evidence)`` when a transition fires, else ``None``.
    """
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

    if current == PHASE_PRELUDE:
        term = exit_terminal_prelude(state)
        if term is not None:
            return PHASE_CLOSE, term[0], {"terminal": True, **term[1]}
        # Asked before the normal exit, which sees only that a figure exists: a
        # cold anchor is a figure the later phases cannot honestly compare to.
        cold = exit_cold_anchor_prelude(state)
        if cold is not None:
            return PHASE_CLOSE, cold[0], {"terminal": True, **cold[1]}
        norm = exit_normal_prelude(state)
        if norm is None:
            # No baseline and no clock left: name the failure instead of
            # letting the run read as an ordinary exit.
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
            # Exhausted optimisation leverage is not terminal: switch lever and
            # advance to KERNEL; only with KERNEL disabled does it wind down.
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
            # Failed conc_sweep closeout is terminal: preserve the honest
            # stop_reason instead of opening another macro-cycle.
            if exit_reason == "sweep_failed":
                return PHASE_CLOSE, exit_reason, exit_evidence
            # R1: open a new macro-cycle while budget remains and the run
            # hasn't globally converged (R7); wind down to CLOSE only when
            # reloop is blocked (budget, convergence, or max_cycles).
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
            # R7: if looping was blocked by global convergence or the safety cap,
            # terminate with a terminal stop_reason instead of idling in CLOSE.
            blocked = str(reloop_ev.get("reloop_blocked") or "")
            if blocked in ("global_converged", "max_cycles"):
                return (
                    PHASE_CLOSE,
                    "global_converged",
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
    """Construct a canonical phase_history row; ``reason`` unvalidated for resume tools.

    ``cycle`` stamps the R1 macro-cycle this transition belongs to (0 for the
    first macro-cycle, or resume from a pre-cyclic session).

    Args:
        from_phase (str): Source phase name; normalized to upper-case.
        to_phase (str): Destination phase name; normalized to upper-case.
        reason (str): Transition reason; stripped, left unvalidated for resume
            tools.
        evidence (dict[str, Any] | None): Supporting evidence; copied (``None``
            → empty dict).
        ts (str): ISO timestamp string for the transition.
        ts_unix (float): Unix timestamp for the transition.
        cycle (int): R1 macro-cycle index this transition belongs to.

    Returns:
        dict[str, Any]: The canonical phase_history row.
    """
    return {
        "from_phase": (from_phase or "").strip().upper(),
        "to_phase": (to_phase or "").strip().upper(),
        "reason": (reason or "").strip(),
        "evidence": dict(evidence or {}),
        "ts": ts,
        "ts_unix": float(ts_unix or 0.0),
        "cycle": int(cycle or 0),
    }


# Lifecycle events — operator-facing phase/step boundary log. Each event carries
# ``phase`` (the coordinator phase active when it fired), ``step`` (the machine
# step/handler name), and ``label`` (the human-friendly name). ``make_lifecycle_event``
# is a pure builder; ``SharedState.record_lifecycle_event`` is the single writer
# (``policy.CORE_STATE_FIELDS`` guards the ``lifecycle`` field).
LIFECYCLE_STATUS_START = "START"
LIFECYCLE_STATUS_END = "END"
LIFECYCLE_STATUS_ERROR = "ERROR"
# Phase-boundary marker: a point-in-time "entered <phase>" mark with no
# matching END (unlike START, which pairs with a later END for the same step).
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

# Human-friendly labels for the lifecycle steps surfaced to operators. Keys are
# the coordinator's machine step/handler names; several map to the same label.
LIFECYCLE_STEP_LABELS: dict[str, str] = {
    "roofline": "TraceLens",
    "trace_analyze": "TraceLens",
    "run_gemm_tuning": "GEMM tuning",
    "run_optimization": "GEAK",
    "run_collective": "Collective optimization",
    "integrate": "Integrate",
    "apply_patch": "Integrate",
    "explore": "Validate (bench on the stack)",
    "sweep": "Concurrency sweep",
    "report": "Report",
    "session_breakdown": "Report (session breakdown)",
}


def lifecycle_label(name: str) -> str:
    """Resolve a human-friendly label for a step or phase name.

    Falls back to the phase-label table, then to the verbatim name.

    Args:
        name (str): A coordinator step or phase name; stripped before lookup.

    Returns:
        str: The mapped step label, else the mapped phase label, else the
        verbatim ``name``.
    """
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
    """Construct a canonical lifecycle event row.

    ``status`` is not hard-validated here so recovery/resume tools can emit
    synthetic rows. Empty / ``None`` artifact values are dropped.

    Args:
        step (str): Machine step/handler name.
        status (str): Event status (e.g. START/END/ENTER); not hard-validated.
        phase (str): Active coordinator phase; normalized to upper-case.
        label (str | None): Human-friendly label; resolved via
            :func:`lifecycle_label` when None.
        artifacts (dict[str, str] | None): Artifact paths; empty/None values are
            dropped.
        detail (str): Free-form detail string; stripped.
        duration_s (float | None): Optional duration in seconds; omitted when
            unparseable.
        seq (int): Monotonic sequence number for the event.
        ts (str): ISO timestamp string for the event.

    Returns:
        dict[str, Any]: The canonical lifecycle event row.
    """
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


# Phase-transition / lifecycle write-owner functions (take ``state`` first and
# own the phase_history / lifecycle bookkeeping). ``SharedState`` keeps
# forwarding shims for the two that were once its methods, so existing
# ``state.record_*(...)`` call sites still reach them; ``bank_phase_segment``
# never was one and is called by name, from here and from the resume path.
def bank_phase_segment(state, *, until_unix: float) -> float:
    """Bank the current phase's live segment, ending at ``until_unix``, into the durable totals.

    ``phase_started_unix`` holds the live segment and is overwritten by the next
    phase entry, so a phase's spend only survives once it is banked here. Called
    at every transition out of a phase, and by a resume for the segment the
    stopped leg never transitioned out of.

    Args:
        state: The live SharedState; ``phase_elapsed_totals`` (and the optimisation-phase
            accumulator) are mutated in place.
        until_unix (float): When the segment ended, in seconds since the epoch.

    Returns:
        float: Seconds banked. ``0.0`` when no phase is set, which is the very
        first transition of a fresh session — it has no segment to bank.
    """
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
    # The optimisation phase keeps its own accumulator: it carries a
    # tri-state "unknown" for
    # legacy resumes that status telemetry reports as absent, whereas
    # ``phase_elapsed_totals`` must never report "unknown" — a budget guard
    # would read that as "no cap". The two answer different questions.
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
    """Append a phase_history row and atomically update ``phase`` fields; ``phase``/``phase_history`` are CORE_STATE_FIELDS so LLM update_state is rejected. Returns the inserted row.

    Args:
        to_phase (str): The phase being entered.
        reason (str): The transition reason (from ``PHASE_EXIT_REASONS``).
        evidence (dict[str, Any] | None): Optional structured evidence
            attached to the history row.
        ts (str | None): Optional ISO timestamp; defaults to now (UTC).
        ts_unix (float | None): Optional Unix epoch matching ``ts``;
            defaults to the current time.

    Returns:
        dict[str, Any]: The inserted phase_history row.
    """
    from datetime import datetime as _dt, timezone as _tz
    import time as _time
    from ..state.shared_state import _PHASE_HISTORY_CAP

    now_ts = ts or _dt.now(_tz.utc).isoformat(timespec="seconds")
    now_unix = float(ts_unix if ts_unix is not None else _time.time())
    from_phase = (state.phase or "").strip().upper()
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
    # Publish the phase for LLM attribution: the spawn sites that tag outbound
    # calls sit in specialists and kernel tools and cannot reach SharedState.
    from hyperloom.common.llm_attribution import set_current_phase

    set_current_phase(str(row["to_phase"] or ""))
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        transition_id = (
            f"phase:{int(getattr(state, 'macro_cycle', 0) or 0)}:"
            f"tick:{int(getattr(state, 'tick', 0) or 0)}:"
            f"event:{len(history)}:"
            f"{row.get('from_phase') or 'START'}:{row.get('to_phase') or ''}:"
            f"{now_unix:.9f}"
        )
        instrument.record_phase_transition(
            getattr(state, "_session_dir", None),
            transition_id=transition_id,
            from_phase=str(row.get("from_phase") or ""),
            phase=str(row.get("to_phase") or ""),
            reason=str(row.get("reason") or ""),
            evidence=dict(row.get("evidence") or {}),
            macro_cycle=int(getattr(state, "macro_cycle", 0) or 0),
            tick=int(getattr(state, "tick", 0) or 0),
            event_sequence=len(history),
            ts=str(row.get("ts") or ""),
        )
        instrument.record_trace_event(
            getattr(state, "_session_dir", None),
            trace_event_id=f"trace:{transition_id}",
            kind="phase_transition",
            from_phase=str(row.get("from_phase") or ""),
            phase=str(row.get("to_phase") or ""),
            reason=str(row.get("reason") or ""),
            ts=str(row.get("ts") or ""),
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
    """Append a non-transition marker row for the current phase.

    Uses the same schema and cap as :func:`record_phase_transition` without
    changing ``state.phase`` or banking elapsed segments.

    Args:
        reason (str): Human-readable reason for the marker.
        evidence (dict[str, Any] | None): Structured payload for the event.
        ts (str | None): Optional ISO timestamp; defaults to now (UTC).
        ts_unix (float | None): Optional Unix epoch matching ``ts``.

    Returns:
        dict[str, Any]: The inserted phase_history row.
    """
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
    """Append a structured lifecycle event marking a phase/step boundary.

    ``step`` is the machine step/handler name; ``label`` defaults to the
    human-friendly name from :data:`LIFECYCLE_STEP_LABELS`. ``phase`` defaults to
    the current coordinator phase; ``seq`` is monotonic across the cap.
    Coordinator-only writer (``policy.CORE_STATE_FIELDS`` guards ``lifecycle``).
    Returns the inserted row.

    Args:
        step (str): The machine step/handler name (e.g. ``trace_analyze``).
        status (str): The boundary status (e.g. ``START`` / ``END`` /
            ``ERROR``).
        phase (str | None): The owning phase; defaults to the current
            coordinator phase.
        label (str | None): Human-friendly step name; defaults to the
            mapping in ``phase_state.LIFECYCLE_STEP_LABELS``.
        artifacts (dict[str, str] | None): Optional produced-artifact
            path map recorded on the event.
        detail (str): Optional free-text detail.
        duration_s (float | None): Optional step duration in seconds.
        ts (str | None): Optional ISO timestamp; defaults to now (UTC).

    Returns:
        dict[str, Any]: The inserted lifecycle event row.
    """
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
    "PHASE_LLM_PROPOSABLE_ACTIONS",
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
    "is_action_allowed_in_phase",
    "is_action_llm_proposable_in_phase",
    "llm_proposable_actions_for",
    "is_valid_escalate_hint",
    "is_valid_phase_exit_reason",
    "is_valid_stop_reason",
    "compute_kernel_progress_fingerprint",
    "collective_integration_pending",
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
