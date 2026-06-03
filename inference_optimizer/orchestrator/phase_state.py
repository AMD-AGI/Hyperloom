"""Phase state machine — v0.8 M2.

The Coordinator owns the run-level phase ("where are we in the optimization
lifecycle") so that LLM-side decisions stay scoped to one phase at a time
and Cortex KB entries carry phase provenance.

This module is **pure**: every public function takes a frozen SharedState
view (a ``Any`` typed shim — we deliberately avoid importing SharedState
to keep this module side-effect free and trivially testable) and returns
either a string sentinel / dict, never mutating its inputs. The
Coordinator is the only writer of SharedState; this module just decides
*what* to write.

Design intent reference: ``KB_design/3.2_pipeline_phases/README.md``,
``KB_design/3.8_phase_state_machine/README.md``, ``KB_design/3.10_shared_state_evolution/README.md``.

The six phases form a strictly monotonic chain (Inv-2.1 phase
monotonicity):

::

    PRELUDE → FRAMEWORK_PR → EXPLORE → KERNEL → SWEEP → CLOSE
                ↘ (--no-framework) ↗
                              ↘ (no_kernel) ↗
                              ↘──── any phase ────→ CLOSE  (terminal / abort)

``recover`` is *phase-orthogonal* and not modeled as a transition.

Vocabularies are closed enums; PolicyGate cross-checks any write of
``stop_reason`` or any ``phase_history.reason`` against them.

v0.8 transition (M3 complete)
-----------------------------
The §3.2 design talks about ``explore`` (single merged action) and
``specialist`` (LLM sub-agent). Both have shipped; the EXPLORE
allowed-action set is ``explore`` / ``specialist`` / ``recover``.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Phase identifiers + ordering (Inv-2.1: monotonic chain)
# ---------------------------------------------------------------------------
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
    """Return monotonic index of ``phase`` (used by Inv-2.1 check).

    Unknown phases return ``-1`` so legacy data fails the
    monotonicity check cleanly instead of pretending to advance.
    """
    return PHASE_INDEX.get((phase or "").strip().upper(), -1)


# ---------------------------------------------------------------------------
# Phase ↔ allowed action set
# ---------------------------------------------------------------------------
# v0.8 view:
#
# * EXPLORE allowlist contains the merged ``explore`` action and the
#   ``specialist`` LLM sub-agent.
# * ``recover`` stays in every phase — phase-orthogonal per §3.2.
# * ``session_breakdown`` is a CLOSE action (it materializes the
#   report bundle). The per-KEEP stack rebench is inlined into
#   ``explore``.
PHASE_ALLOWED_ACTIONS: dict[str, frozenset[str]] = {
    PHASE_PRELUDE: frozenset({
        # ``roofline`` / ``profile`` are Coordinator-auto-enqueued at
        # PRELUDE after baseline lands so the EXPLORE-phase first
        # specialist sees a populated trace (and, when roofline is on,
        # an ``analysis.md`` snapshot). The Coordinator picks the kind
        # via ``shared_state.enable_roofline`` — both names sit in the
        # allowlist so the internal-enqueue passes R1
        # ``phase_incompatible``. LLM-side propose_action /
        # delegate is denied by PolicyGate's
        # ``analysis_action_not_llm_proposable`` rule for both names.
        "target_analysis", "baseline", "roofline", "profile", "recover",
    }),
    PHASE_FRAMEWORK_PR: frozenset({
        # Coordinator-internal phase between PRELUDE and EXPLORE.
        # ``framework_pr`` is enqueued per candidate; the executor uses
        # the same single-variant ``run_grid`` path as ``integrate_patch``
        # so the latter stays in the allowlist as the consume side of
        # the Critic-gated patch flow. ``roofline`` / ``profile`` are
        # auto-enqueued on watermark crossings here too (KEEP path
        # writes ``cumulative_gain_validated`` via the same single-
        # writer hook). LLM-side ``framework_pr`` proposes are denied
        # by ``framework_pr_action_not_llm_proposable``.
        "framework_pr", "integrate_patch", "roofline", "profile", "recover",
    }),
    PHASE_EXPLORE: frozenset({
        # v0.8 canonical: merged grid runner + LLM specialist dispatch.
        "explore", "specialist",
        # PR-A1 (Arbor-into-Hyperloom): specialists in EXPLORE may write
        # source patches into ``runs/specialist/<task_id>/worktree/``;
        # ``integrate_patch`` is the orchestrator-side serving-lane-locked
        # apply+restart+gate step that consumes those patches.
        "integrate_patch",
        # IR-7 (Honest self-stop): thin wrapper that dispatches the
        # session_steward_specialist domain. Coordinator also enqueues
        # this internally on plateau (bypasses PolicyGate); LLM-side
        # proposes are throttled by ``assess_remaining_gaps_throttle``.
        "assess_remaining_gaps",
        # dynamic_action.MD P1 — supplementary cross-domain ReAct
        # sub-agent channel; orchestration-only dispatch, capped at
        # MAX_DYNAMIC_PER_ROUND per EXPLORE round. See PolicyGate
        # ``_validate_dynamic_action_dispatch``.
        "dynamic_action",
        # ``roofline`` / ``profile`` are Coordinator-auto-enqueued mid-
        # EXPLORE whenever the watermark check at the
        # cumulative_gain_validated writer fires (10% step compound vs
        # ``last_roofline_tput``). The Coordinator picks the kind via
        # ``enable_roofline``; the auto_roofline_pending_task_id
        # blocker holds dispatches until either kind lands.
        "roofline", "profile",
        "recover",
    }),
    PHASE_KERNEL: frozenset({
        # KERNEL_OWNED_ACTIONS from policy.py.
        "kernel_opt", "integrate", "deep_kernel_analysis",
        "operator_tuning", "vendor_kernel_config", "gemm_tuning",
        # ``roofline`` / ``profile`` are auto-enqueued on watermark
        # crossing here too — kernel integrate KEEPs flow through the
        # same single-writer hook as explore/specialist KEEPs; mode is
        # picked via ``enable_roofline``.
        "roofline", "profile",
        "recover",
    }),
    PHASE_SWEEP: frozenset({
        # ``conc_sweep`` is a Coordinator-internal post-sweep action
        # (PHASE_SWEEP only, on by default; disable via
        # ``--no-enable-conc-sweep``) that benchmarks both baseline
        # and ``current_best`` across a CONC ladder. Like ``sweep``
        # it is discovery-only and never
        # promotes; PolicyGate denies LLM-proposed
        # ``delegate{action_name='conc_sweep'}`` because the auto-enqueue
        # at sweep-task completion is the sole legitimate entry point.
        "sweep", "conc_sweep", "recover",
    }),
    PHASE_CLOSE: frozenset({
        "report", "session_breakdown",
        "recover",
    }),
}


def is_action_allowed_in_phase(action_name: str, phase: str) -> bool:
    """Return True iff ``action_name`` is in the phase allowlist.

    Used by PolicyGate R1 phase_incompatible. Unknown phase defaults to
    *deny* so a corrupted state.json doesn't silently let anything pass.
    """
    allowed = PHASE_ALLOWED_ACTIONS.get((phase or "").strip().upper())
    if allowed is None:
        return False
    return (action_name or "").strip() in allowed


def allowed_actions_for(phase: str) -> tuple[str, ...]:
    """Return ``PHASE_ALLOWED_ACTIONS[phase]`` as a sorted tuple.

    Stable order so prompt rendering / R1 hints are deterministic.
    """
    return tuple(sorted(PHASE_ALLOWED_ACTIONS.get((phase or "").strip().upper(), frozenset())))


# ---------------------------------------------------------------------------
# phase_exit_reasons vocab
# ---------------------------------------------------------------------------
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
    "explore_force_exit_low_budget",    # EXPLORE → next phase when total remaining or phase remaining drops below operator-configured thresholds
    "no_more_leverage",                 # EXPLORE/KERNEL → SWEEP (non-terminal): steward stop_session via the skip_to_sweep hint. (The Coordinator's automatic no-more-leverage safety net was removed for long-run continuity.) Reclassified from a terminal stop_reason — it now winds the session down through SWEEP → CLOSE instead of aborting.
    # FRAMEWORK_PR phase transitions (PR-A1 / FRAMEWORK_PR phase).
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
    "baseline_failed",                  # v0.6 sentinel; left for resume parity
    "emergency",
    "max_ticks",
    "signal",

    # Construction sentinel — the first entry written when a fresh
    # session enters PRELUDE.  Not a transition reason proper, but
    # surfacing it in the same vocab keeps phase_history homogeneous
    # for breakdown collection.
    "phase_entered",

    # legacy resume inference (lenient mapping).
    "resumed_from_v06_inferred",
})


# ---------------------------------------------------------------------------
# stop_reason vocab
# ---------------------------------------------------------------------------
STOP_REASON_VOCAB: frozenset[str] = frozenset({
    # v0.6 sentinels — kept for backward compat (resume from old sessions).
    "target_reached",
    "no_more_leverage",
    "time_exhausted",
    "max_ticks",
    "policy_loop",
    "baseline_failed",
    "emergency",
    "signal",
    "unknown",
    "custom",

    # v0.8 additions.
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
})


def is_valid_stop_reason(value: str) -> bool:
    return (value or "").strip() in STOP_REASON_VOCAB


def is_valid_phase_exit_reason(value: str) -> bool:
    return (value or "").strip() in PHASE_EXIT_REASONS


# ---------------------------------------------------------------------------
# Default phase budgets (% of total wall-clock) — KB_design §3.8 §5.3
# ---------------------------------------------------------------------------
DEFAULT_PHASE_BUDGET_PCT: dict[str, float] = {
    # Rebalanced 2026-06 based on field telemetry from the
    # Qwen3-30B-A3B / MI355x runs once conc_sweep was on by default:
    # SWEEP was running ~2.5x over its old 8% allocation (sweep action
    # ~37min + conc_sweep ~31min, total ~68min on a ~5.7h budget) so we
    # shift PRELUDE -3pp / EXPLORE -7pp into SWEEP +10pp while keeping
    # KERNEL at its historical 35% (GEAK quick-mode needs full cycles).
    # PRELUDE only ever spent ~5min of its old 27min slice; EXPLORE
    # force-exit at phase_remaining_pct=0.176 confirmed the old 47%
    # was over-provisioned by ~7pp.
    #
    # FRAMEWORK_PR is *not* given a phase budget pct — the time wall is
    # ``force_exit_hours_remaining_ratio * max_hours`` instead (default
    # 0.6), matching the design's "leave at least 60% for the rest of
    # the session" intent.
    PHASE_PRELUDE: 0.05,
    PHASE_EXPLORE: 0.40,
    PHASE_KERNEL:  0.35,
    PHASE_SWEEP:   0.18,
    PHASE_CLOSE:   0.02,
}


# ---------------------------------------------------------------------------
# Plateau judgment defaults
# ---------------------------------------------------------------------------
# Default thresholds the CLI exposes via --plateau-* flags. Kept here
# (not in cli.py) so pure-function callers + tests can introspect the
# canonical default set without importing argparse.
DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT:    float = 0.5
DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK:     int   = 3
DEFAULT_PLATEAU_EXPLORE_LOOKBACK:         int   = 5
DEFAULT_PLATEAU_KERNEL_REVERT_STREAK:     int   = 3
DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT:     float = 0.5
DEFAULT_PLATEAU_KERNEL_LOOKBACK:          int   = 5

# v0.8 — EXPLORE hard force-exit thresholds (HARD time gate; overrides
# plateau + steward). The Coordinator may exit EXPLORE the moment EITHER
# of the following holds:
#
# * total wall-clock remaining (``state.remaining_minutes()``) is below
#   ``DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING`` hours, OR
# * the EXPLORE phase's own remaining budget fraction is below
#   ``DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT`` of its allocated slice.
#
# The defaults match the report iter 19 lesson: leave at least 3h or
# 20% of EXPLORE budget for the downstream KERNEL/SWEEP/CLOSE phases so
# the session can produce a clean report + recipe write-back instead of
# burning the whole budget on diminishing-returns explore variants.
# Operators tune via ``--explore-force-exit-hours-remaining`` /
# ``--explore-force-exit-budget-pct`` (locked at session start into
# ``SharedState.plateau_overrides`` under
# ``force_exit_hours_remaining`` / ``force_exit_budget_pct``).
DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING: float = 3.0
DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT:      float = 0.20

# FRAMEWORK_PR plateau / force-exit knobs.
#
# * ``DEFAULT_FRAMEWORK_PR_PLATEAU_LOOKBACK`` — number of consecutive
#   ``fa phase-discover`` batches we look back at when deciding the
#   plateau condition (default 3).
# * ``DEFAULT_FRAMEWORK_PR_PLATEAU_KEEP_GAIN_PCT`` — minimum per-batch
#   ``max_gain_pct_observed_in_batch`` that counts as "non-flat" (default
#   1.0%). The plateau fires when every one of the last
#   ``lookback`` batches sits below this threshold.
# * ``DEFAULT_FRAMEWORK_PR_FORCE_EXIT_HOURS_REMAINING_RATIO`` — when
#   ``remaining_minutes()/60 < ratio * max_hours`` we hard-exit
#   FRAMEWORK_PR so EXPLORE/KERNEL/SWEEP/CLOSE still have room to run
#   (default 0.6 → leave 60% of the original budget for downstream).
DEFAULT_FRAMEWORK_PR_PLATEAU_LOOKBACK:                 int   = 3
DEFAULT_FRAMEWORK_PR_PLATEAU_KEEP_GAIN_PCT:            float = 1.0
DEFAULT_FRAMEWORK_PR_FORCE_EXIT_HOURS_REMAINING_RATIO: float = 0.6


# ---------------------------------------------------------------------------
# Exploration-depth gate thresholds.
#
# The depth gate only evaluates once the explore loop has shown
# instability (``consecutive_reverts`` >= the activation threshold).
# Each dimension is a minimum the session must reach before a stop /
# advance verdict is allowed through. Dimensions whose evidence is not
# being supplied this session (no specialist reports PR / diff / nvidia
# refs) are dropped from the decision (recorded as N/A) so they never
# become an unsatisfiable stall.
# ---------------------------------------------------------------------------
DEFAULT_DEPTH_GATE_SCOUT_RUNS:          int = 2
DEFAULT_DEPTH_GATE_PRS_FETCHED:         int = 5
DEFAULT_DEPTH_GATE_PR_DIFFS_READ:       int = 3
DEFAULT_DEPTH_GATE_NVIDIA_REFS:         int = 2
DEFAULT_DEPTH_GATE_CODE_PATCHES:        int = 1
DEFAULT_DEPTH_GATE_REVERTS_TO_EVALUATE: int = 3


# ---------------------------------------------------------------------------
# escalate_strategy_change hint vocabulary
# ---------------------------------------------------------------------------
# Closed enum so the Coordinator + phase_state agree on which hints can
# influence phase transitions / budgets. Unknown hints are logged but
# do NOT change phase (defensive — prevents an arbitrary
# robustness-emitted string from steering the state machine).
ESCALATE_HINT_SKIP_TO_KERNEL:      str = "skip_to_kernel"
ESCALATE_HINT_SKIP_TO_SWEEP:       str = "skip_to_sweep"
ESCALATE_HINT_SKIP_TO_CLOSE:       str = "skip_to_close"
ESCALATE_HINT_EXTEND_EXPLORE_BUDGET: str = "extend_explore_budget"
ESCALATE_HINT_EXTEND_KERNEL_BUDGET:  str = "extend_kernel_budget"
ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX: str = "pause_specialist_"

# ``skip_to_sweep`` is the non-terminal "no more leverage" signal: it
# winds EXPLORE (or KERNEL) down to SWEEP → CLOSE *without* terminating
# the optimization. Unlike ``skip_to_close`` (which is terminal via
# ``_global_terminal`` → ``robustness_escalated``), ``skip_to_sweep``
# still runs the SWEEP validation pass and produces a clean report.
# It is set by the IR-7 steward's ``stop_session`` verdict, which used
# to set a terminal ``stop_reason='no_more_leverage'``. (The Coordinator
# also used to drive this via an automatic no-more-leverage safety net;
# that net was removed for long-run continuity.)
ESCALATE_HINT_VOCAB: frozenset[str] = frozenset({
    ESCALATE_HINT_SKIP_TO_KERNEL,
    ESCALATE_HINT_SKIP_TO_SWEEP,
    ESCALATE_HINT_SKIP_TO_CLOSE,
    ESCALATE_HINT_EXTEND_EXPLORE_BUDGET,
    ESCALATE_HINT_EXTEND_KERNEL_BUDGET,
})

# Cap that ``extend_*_budget`` hints can lift the per-phase budget to
# (relative ratio, not absolute). 0.80 means: EXPLORE budget can grow to
# at most 80% of total wall-clock budget, no matter how many
# ``extend_explore_budget`` hints fire. Mirrors KB_design §3.13 M7 §5.3
# "上限 80%".
ESCALATE_HINT_BUDGET_BUMP_DELTA: float = 0.05   # +5 percentage points per hint
ESCALATE_HINT_BUDGET_BUMP_CAP:   float = 0.80   # absolute ceiling

# Whether a hint string is structurally a pause-specialist directive.
# ``pause_specialist_kernel_switch_specialist`` etc.; the suffix is the domain
# key (specialist_domains.SPECIALIST_DOMAIN_KEYS membership is checked
# by the Coordinator handler, not here, so this module stays pure).
def is_pause_specialist_hint(hint: str) -> bool:
    h = (hint or "").strip()
    return h.startswith(ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX) and len(h) > len(
        ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX,
    )


def is_valid_escalate_hint(hint: str) -> bool:
    """Return True for any hint Coordinator should act on.

    The closed vocab + the ``pause_specialist_<domain>`` family form
    the full surface. Anything else is dropped to log.
    """
    return (hint or "").strip() in ESCALATE_HINT_VOCAB or is_pause_specialist_hint(hint)


def apply_escalate_budget_bump(
    current_budget_pct: dict[str, float] | None,
    *,
    phase: str,
    delta: float = ESCALATE_HINT_BUDGET_BUMP_DELTA,
    cap: float = ESCALATE_HINT_BUDGET_BUMP_CAP,
) -> dict[str, float]:
    """Return a budget map with ``phase`` raised by ``delta`` (capped).

    Pure helper that the Coordinator can call when it pops an
    ``extend_explore_budget`` / ``extend_kernel_budget`` hint. The
    resulting dict is suitable for assigning back into
    :attr:`SharedState.phase_budget_pct`.

    "上限 80% (绝对值)".
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
    """Return a sanitized ``phase -> pct`` mapping.

    Missing phases fall back to defaults; obviously bad values (negative
    or > 1.0) get clamped to the default. The total is *not* renormalized
    to 1.0 — phase budgets are *upper bounds*, not a probability dist.
    """
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


# ---------------------------------------------------------------------------
# Pure judgment helpers (used by Coordinator at each tick end)
# ---------------------------------------------------------------------------
def _now_unix(state: Any) -> float:
    """Resolve the "now" timestamp; tests can inject a stub via
    ``state._now_unix`` for determinism."""
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
    """Return a pending escalate hint to act on this tick.

    Coordinator's ``_handle_escalate_strategy_change`` writes the
    incoming ``next_action_hint`` into ``SharedState.pending_escalate_hint``;
    phase_state checks the field here when computing the exit
    decision. The Coordinator clears it once the phase write lands.

    Unknown hints are surfaced as empty so the state machine ignores
    them (defensive — prevents arbitrary robustness payloads from
    steering phases).
    """
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
    """Return seconds remaining in the current phase's budget.

    Returns ``None`` when ``max_minutes`` is 0 (unlimited run) — caller
    interprets that as "budget never exhausted".
    """
    mm = _max_minutes(state)
    if mm <= 0:
        return None
    budget = normalize_budget_pct(budget_pct or getattr(state, "phase_budget_pct", None))
    pct = budget.get((getattr(state, "phase", "") or "").upper(), 0.0)
    if pct <= 0:
        return None
    budget_seconds = mm * 60.0 * pct
    return max(0.0, budget_seconds - phase_elapsed_seconds(state, now_unix=now_unix))


# ----------------------- EXPLORE hard force-exit (HARD time gate) -----------
def session_remaining_seconds(
    state: Any, *, now_unix: float | None = None,
) -> float | None:
    """Total wall-clock seconds remaining for the session.

    Returns ``None`` when ``max_minutes`` is 0 (unlimited run) or
    ``start_ts`` is unparseable. Mirrors
    :meth:`SharedState.remaining_minutes` without taking a datetime
    dependency so phase_state stays pure.
    """
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
    """Return ``(True, evidence)`` when HARD EXPLORE force-exit fires.

    The gate fires when EITHER of:

    * ``session_remaining_seconds(state) <= hours_remaining_threshold * 3600``
      — total wall-clock budget about to run out; downstream phases need
      buffer for report + recipe write-back (report iter 19 lesson).
    * ``phase_budget_remaining_pct <= budget_pct_threshold`` — the EXPLORE
      slice is almost exhausted; don't squeeze a final variant in only
      to lose it to a half-finished benchmark.

    Either gate alone is sufficient. Both fields land in ``evidence``
    so the audit trail captures which condition fired. The Coordinator
    routes EXPLORE→KERNEL (or →SWEEP when ``kernel_enabled=False``)
    via the existing ``compute_next_phase`` plumbing — the new exit
    reason ``explore_force_exit_low_budget`` reuses the same target
    selection logic.

    Returns ``(False, evidence)`` when neither condition holds; evidence
    is still populated for diagnostics so callers can render
    ``force_exit_remaining_sec`` into the Orchestration prompt.
    """
    evidence: dict[str, Any] = {
        "hours_remaining_threshold":  float(hours_remaining_threshold),
        "budget_pct_threshold":       float(budget_pct_threshold),
    }
    fired = False
    fired_reasons: list[str] = []

    # Non-positive threshold = disabled. Lets callers opt out of either
    # sub-gate (e.g. tests that want to isolate budget_exhausted, or an
    # operator who wants only the phase-pct backstop). Both thresholds
    # disabled effectively turns IR-6 off for this call.
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
        # Compute the phase's *total* allotted budget so we can express
        # remaining as a fraction.
        mm = _max_minutes(state)
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


def depth_gate(
    state: Any,
    *,
    scout_runs_min: int = DEFAULT_DEPTH_GATE_SCOUT_RUNS,
    prs_fetched_min: int = DEFAULT_DEPTH_GATE_PRS_FETCHED,
    pr_diffs_read_min: int = DEFAULT_DEPTH_GATE_PR_DIFFS_READ,
    nvidia_refs_min: int = DEFAULT_DEPTH_GATE_NVIDIA_REFS,
    code_patches_min: int = DEFAULT_DEPTH_GATE_CODE_PATCHES,
    reverts_to_evaluate: int = DEFAULT_DEPTH_GATE_REVERTS_TO_EVALUATE,
) -> tuple[bool, list[str], str]:
    """Deterministic exploration-depth check.

    Returns ``(satisfied, blockers, next_action)``:

    * ``satisfied`` — whether every *supplied* depth dimension has met
      its minimum. The gate only activates once the explore loop has
      shown instability (``consecutive_reverts >= reverts_to_evaluate``);
      before that it reports ``satisfied=True`` (nothing to enforce).
    * ``blockers`` — human-readable list of unmet dimensions, most
      important first.
    * ``next_action`` — a concrete deepening instruction for the most
      important blocker (empty when satisfied).

    Dimensions whose evidence is not being supplied this session (no
    specialist reports PR / diff / nvidia refs) are dropped — their bar
    cannot be satisfied, so enforcing them would stall the session. The
    deterministic dimensions (scout runs, code patches) are always
    enforced. Pure + fail-soft: a missing / malformed tracker yields
    ``satisfied=True``.
    """
    snap_fn = getattr(state, "depth_snapshot", None)
    try:
        snap = snap_fn() if callable(snap_fn) else {}
    except Exception:  # noqa: BLE001 — fail-soft on any malformed tracker
        snap = {}
    if not isinstance(snap, dict) or not snap:
        return True, [], ""
    if not bool(snap.get("enabled", True)):
        return True, [], ""

    reverts = int(snap.get("consecutive_reverts") or 0)
    if reverts < int(reverts_to_evaluate):
        return True, [], ""

    scout_runs = int(snap.get("research_scout_runs") or 0)
    code_patches = int(snap.get("code_patches_attempted") or 0)
    prs = len(snap.get("prs_fetched") or [])
    diffs = len(snap.get("pr_diffs_read") or [])
    nvidia = len(snap.get("nvidia_refs_compared") or [])

    blockers: list[str] = []
    # (label, current, minimum, supplied, next_action) — order encodes
    # priority: cheap / always-supplied deepening first.
    dims = [
        (
            "research_scout_runs", scout_runs, int(scout_runs_min), True,
            "dispatch another research scout to gather proven priors",
        ),
        (
            "code_patches_attempted", code_patches, int(code_patches_min), True,
            "try at least one source-level code patch, not just config tweaks",
        ),
        (
            "prs_fetched", prs, int(prs_fetched_min), prs > 0,
            "fetch more candidate PRs across frameworks before stopping",
        ),
        (
            "pr_diffs_read", diffs, int(pr_diffs_read_min), diffs > 0,
            "read more PR diffs to extract concrete optimizations",
        ),
        (
            "nvidia_refs_compared", nvidia, int(nvidia_refs_min), nvidia > 0,
            "compare against more NVIDIA / cross-vendor reference numbers",
        ),
    ]
    next_action = ""
    for label, current, minimum, supplied, action in dims:
        if minimum <= 0 or not supplied:
            continue
        if current < minimum:
            blockers.append(f"{label}={current}<{minimum}")
            if not next_action:
                next_action = action
    return (len(blockers) == 0), blockers, next_action


# ----------------------- plateau pure functions --------
def compute_plateau_explore(
    state: Any,
    *,
    lookback: int = DEFAULT_PLATEAU_EXPLORE_LOOKBACK,
    keep_gain_threshold_pct: float = DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
    empty_streak_threshold: int = DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
) -> tuple[bool, dict[str, Any]]:
    """Real plateau_explore.

    Returns ``(triggered, evidence)``. ``evidence`` always contains the
    decision inputs so phase_history can audit the call (even when
    ``triggered=False``).

    Inputs (all read from :class:`SharedState`, pure):

    * ``explore_search.winners_history``: last K KEEP'd variants
; ``gain_pct`` per row drives the
      "recent_keep_gain" sum.
    * ``specialist_rounds``: last K specialist rounds;
      ``proposals_total`` + ``proposals_kept`` per row drive the
      empty-round detection.

    Trigger (AND of two signals, KB_design §3.8 §5.1):

        recent_keep_gain    < ``keep_gain_threshold_pct``  AND
        recent_empty_streak >= ``empty_streak_threshold``

    The AND semantics intentionally avoid false positives — "we tried
    but it didn't help" alone or "specialist returned empty" alone is
    not enough; both must hold.
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
        # Designed shape: ``proposals_total`` /
        # ``proposals_kept`` (M5+); fall back to ``proposal_count``
        # for forward-compat with older round summaries.
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
    """Real plateau_kernel.

    Returns ``(triggered, evidence)``. Inputs (all SharedState):

    * ``kernel_opt_attempts``: per-kernel_id history dicts (KB_design
      §3.10 §4.2). Each entry typically carries
      ``{attempts, partial_count, last_decision, last_ts, history}``.
    * ``kernel_integrate_attempts``: per-(kernel_id+patch) integrate
      attempts; each entry carries an ``attempts`` list of
      ``{decision, ts}``.
    * ``rejected_kernel_ids``: cross-session retired kernels.

    Trigger (OR — KB_design §3.8 §5.2 deliberately weaker than the
    explore AND because kernel attempts are expensive):

        recent_revert_streak >= ``revert_streak_threshold``
        OR recent_keep_gain  <  ``keep_gain_threshold_pct``
    """
    lookback = int(lookback or 0)
    revert_streak_threshold = int(revert_streak_threshold or 0)
    keep_gain_threshold_pct = float(keep_gain_threshold_pct or 0.0)
    if lookback <= 0 or revert_streak_threshold <= 0:
        return False, {"reason": "thresholds_disabled"}

    integ_attempts = getattr(state, "kernel_integrate_attempts", None) or {}
    if not isinstance(integ_attempts, dict):
        integ_attempts = {}

    # Flatten the integrate attempt log into a single time-ordered list
    # (relying on the embedded ts), then take the last ``lookback`` rows.
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

    # Empty-data guard: an empty ledger (KERNEL just entered, or every
    # prior session-kernel attempt was dropped during resume cleanup)
    # must NOT auto-trigger plateau via the
    # ``recent_keep_gain (=0.0) < keep_gain_threshold_pct (=0.5)`` arm.
    # That degenerate trigger used to flip plateau true the moment KERNEL
    # started with zero attempts on record — coupled with EXPLORE that
    # produced no KEEPs (e.g. force-exit on low budget), the session went
    # EXPLORE→KERNEL→SWEEP without ever spawning a single
    # ``trace_analyze`` / ``run_optimization`` request. Returning False
    # here lets the LLM (and the in-loop scheduling-police) actually
    # exercise the kernel phase before plateau is reconsidered.
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


# ----------------------- terminal / abort (global) -------------------------
def _global_terminal(state: Any) -> tuple[str, dict[str, Any]] | None:
    """Return ``(stop_reason, evidence)`` if a phase-orthogonal stop has
    fired.  Caller routes this to CLOSE regardless of current phase.

    Priority order:

    1. ``escalate_strategy_change`` hint of ``skip_to_close`` →
       ``robustness_escalated``.
    2. Coordinator-set ``stop_reason`` (any phase, any reason).

    Note: ``time_exhausted`` is signalled by the existing Coordinator
    closing-phase mechanism (`run()` writes ``stop_reason`` on
    deadline); we read SharedState to detect it here so phase_history
    has a final row.
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
            # PolicyGate enforces vocab; unknown values reaching here
            # are tolerated for resume parity but flagged via evidence.
            return sr, {"reason_origin": "shared_state.stop_reason", "vocab": "unknown"}
        return sr, {"reason_origin": "shared_state.stop_reason"}
    return None


# ----------------------- per-phase judgments -------------------------------
def warm_replay_in_flight(state: Any) -> bool:
    """True while the PRELUDE warm-recipe replay task has not finished.

    The Coordinator stamps ``warm_replay_outcome.status='in_flight'`` at
    enqueue time and clears it in ``_promote_warm_replay``. PRELUDE must
    not exit (and the initial roofline must not enqueue) until this
    returns False — both paths launch Magpie/sglang on the same GPU.
    """
    outcome = getattr(state, "warm_replay_outcome", None) or {}
    if not isinstance(outcome, dict):
        return False
    return str(outcome.get("status") or "").strip() == "in_flight"


def exit_normal_prelude(state: Any) -> tuple[str, dict[str, Any]] | None:
    """``baseline_tput > 0`` and warm-replay settled → ``prelude_done``.

    Returns ``(reason, evidence)`` when ready to transition, ``None``
    otherwise. Evidence is later spliced into ``phase_history``.

    Warm-replay blocks the exit while ``warm_replay_in_flight`` so
    FRAMEWORK_PR / auto-roofline cannot start a second Magpie job on
    the GPU before the KB config replay finishes.
    """
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
    # Cortex T0 failure already wrote stop_reason; treat
    # cortex_t0_failed / time_exhausted_during_prelude as PRELUDE abort
    # so phase_history captures the boundary.
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
    plateau_keep_gain_pct: float = DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
    plateau_empty_streak: int = DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
    plateau_lookback: int = DEFAULT_PLATEAU_EXPLORE_LOOKBACK,
    disable_legacy_proxy: bool = False,
    force_exit_hours_remaining: float = DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING,
    force_exit_budget_pct: float = DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT,
) -> tuple[str, dict[str, Any]] | None:
    """EXPLORE normal exit.

    Priority order:

    0. HARD force-exit (:func:`should_force_exit_explore`) — overrides
       every other gate. Fires when total session remaining wall-clock
       drops below ``force_exit_hours_remaining`` hours OR when this
       phase's remaining budget pct drops below ``force_exit_budget_pct``.
       Iron Rule IR-6: the steward / plateau / LLM proposals do not get
       to argue with this gate.
    1. ``escalate_strategy_change`` hint ``skip_to_kernel`` →
       ``plateau_explore`` (``evidence='llm_escalation'``).
    2. Real ``plateau_explore`` (:func:`compute_plateau_explore`)
       when signals are present (``explore_search.winners_history``
       or ``specialist_rounds``).
    3. KB_gaps/Gap-15 R-09 transitional proxy
       (``params_no_promote_streak``) when signals are absent
       AND ``disable_legacy_proxy=False``. Evidence carries
       ``r09_provisional=True`` so the breakdown collector surfaces
       a ``plateau_proxy_provisional`` warning. Operators can set
       ``INFERENCE_OPTIMIZER_DISABLE_PLATEAU_PROXY=1`` (Coordinator
       reads + passes through) once their fleet is fully v0.8 to
       fail closed.
    5. Phase budget exhausted.
    """
    # Priority 0 — HARD force-exit (IR-6).
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
    # Non-terminal "no more leverage" signal (steward stop_session). Winds
    # EXPLORE down to SWEEP (skipping KERNEL) → CLOSE rather than aborting
    # the session.
    # compute_next_phase routes the ``no_more_leverage`` reason to SWEEP.
    if hint == ESCALATE_HINT_SKIP_TO_SWEEP:
        return "no_more_leverage", {
            "evidence": "no_more_leverage",
            "hint": hint,
        }
    explore_search = getattr(state, "explore_search", None) or {}
    has_v08_signals = (
        isinstance(explore_search, dict)
        and (explore_search.get("winners_history") or [])
    ) or bool(getattr(state, "specialist_rounds", None) or [])
    if has_v08_signals:
        triggered, evidence = compute_plateau_explore(
            state,
            lookback=plateau_lookback,
            keep_gain_threshold_pct=plateau_keep_gain_pct,
            empty_streak_threshold=plateau_empty_streak,
        )
        if triggered:
            # IR-7 — steward gate. The plateau judge fired; consult the
            # session_steward verdict before actually exiting. Three
            # routes:
            #   * steward_disabled override → exit immediately (plateau).
            #   * no assessment yet → return None with a sentinel
            #     evidence the Coordinator picks up (it enqueues the
            #     steward internally; we stay in EXPLORE one more tick).
            #   * recommendation == 'continue_explore' AND continuation
            #     not yet used → stay in EXPLORE; Coordinator already
            #     reset the plateau counters when it routed the verdict.
            #   * otherwise → exit plateau normally; evidence carries
            #     the recommendation so the audit trail reflects it.
            overrides = _resolve_plateau_overrides(state)
            steward_disabled = bool(overrides.get("steward_disabled", False))
            if steward_disabled:
                return "plateau_explore", {
                    "evidence":           "plateau_judgment",
                    "steward_disabled":   True,
                    **evidence,
                }
            assessment = getattr(
                state, "last_remaining_gaps_assessment", None,
            ) or {}
            rec = str(
                assessment.get("recommendation") or ""
            ).strip().lower() if isinstance(assessment, dict) else ""
            steward_used = bool(getattr(
                state, "steward_continuation_used", False,
            ))
            # When the depth gate is active it owns continuation
            # bounding (rewrites stop/advance to continue as often as
            # needed, IR-6 being the only backstop), so the legacy
            # single-continuation cap is lifted. With the gate off the
            # original "one continuation per session" semantics apply.
            depth_gate_on = True
            depth_enabled_fn = getattr(state, "depth_gate_enabled", None)
            if callable(depth_enabled_fn):
                try:
                    depth_gate_on = bool(depth_enabled_fn())
                except Exception:  # noqa: BLE001 — fail-soft
                    depth_gate_on = True
            if not rec:
                # No verdict yet — stay in EXPLORE one more tick and
                # let the Coordinator enqueue an internal steward run
                # (it polls ``wants_steward_assessment`` after
                # ``compute_next_phase`` returns None).
                return None
            if rec == "continue_explore" and (depth_gate_on or not steward_used):
                # Continuation granted; the Coordinator already
                # processed the verdict (reset plateau counters,
                # injected next_gap). Stay in EXPLORE.
                return None
            # advance_to_kernel / stop_session / continuation exhausted.
            return "plateau_explore", {
                "evidence":              "plateau_judgment",
                "steward_recommendation": rec,
                **evidence,
            }
    elif not disable_legacy_proxy:
        params_streak = int(getattr(state, "params_no_promote_streak", 0) or 0)
        explore_search = getattr(state, "explore_search", None) or {}
        explore_accepted = 0
        if isinstance(explore_search, dict):
            accepted = explore_search.get("accepted") or []
            explore_accepted = len(accepted) if isinstance(accepted, list) else 0
        optimization_stack = getattr(state, "optimization_stack", None) or []
        has_results = bool(
            isinstance(optimization_stack, list) and optimization_stack
        )
        if params_streak >= 5 and explore_accepted == 0 and has_results:
            return "plateau_explore", {
                "evidence": "m2_proxy",
                "r09_provisional": True,
                "params_no_promote_streak": params_streak,
                "explore_accepted": explore_accepted,
                "note": (
                    "KB_design §3.14 R-09 — legacy params_no_promote_streak "
                    "proxy fired (signals empty); set "
                    "INFERENCE_OPTIMIZER_DISABLE_PLATEAU_PROXY=1 to forbid"
                ),
            }
    remaining = phase_budget_remaining_seconds(
        state, budget_pct=budget_pct, now_unix=now_unix,
    )
    if remaining is not None and remaining <= 0:
        return "explore_phase_budget_exhausted", {
            "elapsed_seconds": phase_elapsed_seconds(state, now_unix=now_unix),
        }
    return None


def wants_steward_assessment(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
    plateau_keep_gain_pct: float = DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
    plateau_empty_streak: int = DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
    plateau_lookback: int = DEFAULT_PLATEAU_EXPLORE_LOOKBACK,
) -> bool:
    """Return True when Coordinator should enqueue a steward task NOW.

    Pre-conditions (all must hold):

    * current phase is EXPLORE,
    * plateau has triggered (real plateau judgment, not the legacy
      m2_proxy — the proxy is too noisy to drive a steward dispatch),
    * no fresh ``last_remaining_gaps_assessment`` exists (otherwise we
      already routed on the existing verdict),
    * steward is not disabled by operator,
    * HARD force-exit has not already fired (in which case there's no
      time to dispatch an LLM run — IR-6 wins outright).

    The Coordinator calls this on every tick after
    :func:`compute_next_phase` returns ``None``; ``True`` means enqueue
    a ``session_steward_specialist`` task (bypassing PolicyGate; the
    same idempotency pattern as ``closing_report_task_id``).
    """
    phase = (getattr(state, "phase", "") or "").strip().upper()
    if phase != PHASE_EXPLORE:
        return False
    overrides = _resolve_plateau_overrides(state)
    if bool(overrides.get("steward_disabled", False)):
        return False
    # HARD force-exit takes precedence — if it fires, IR-6 routes us
    # straight to KERNEL and the steward never runs.
    forced, _ = should_force_exit_explore(
        state,
        hours_remaining_threshold=float(overrides.get(
            "force_exit_hours_remaining",
            DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING,
        )),
        budget_pct_threshold=float(overrides.get(
            "force_exit_budget_pct",
            DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT,
        )),
        budget_pct=budget_pct,
        now_unix=now_unix,
    )
    if forced:
        return False
    explore_search = getattr(state, "explore_search", None) or {}
    has_v08_signals = (
        isinstance(explore_search, dict)
        and (explore_search.get("winners_history") or [])
    ) or bool(getattr(state, "specialist_rounds", None) or [])
    if not has_v08_signals:
        return False
    triggered, _ = compute_plateau_explore(
        state,
        lookback=plateau_lookback,
        keep_gain_threshold_pct=plateau_keep_gain_pct,
        empty_streak_threshold=plateau_empty_streak,
    )
    if not triggered:
        return False
    assessment = getattr(state, "last_remaining_gaps_assessment", None) or {}
    if isinstance(assessment, dict):
        rec = str(assessment.get("recommendation") or "").strip().lower()
        if rec in ("continue_explore", "advance_to_kernel", "stop_session"):
            # Already have a routable verdict.
            return False
    return True


def exit_normal_kernel(
    state: Any,
    *,
    budget_pct: dict[str, float] | None = None,
    now_unix: float | None = None,
    plateau_revert_streak: int = DEFAULT_PLATEAU_KERNEL_REVERT_STREAK,
    plateau_keep_gain_pct: float = DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT,
    plateau_lookback: int = DEFAULT_PLATEAU_KERNEL_LOOKBACK,
) -> tuple[str, dict[str, Any]] | None:
    """KERNEL normal exit.

    Priority:

    1. ``escalate_strategy_change`` hint of ``skip_to_close`` →
       defer to the global terminal handler (caller writes
       ``stop_reason=robustness_escalated``). Returns ``None`` here so
       the global path wins.
    2. ``skip_to_sweep`` hint (non-terminal "no more leverage" from the
       steward's ``stop_session`` verdict) → ``no_more_leverage``; KERNEL
       already exits to SWEEP so this just forces the wind-down now.
    3. FP8 GEMM tuning completed and no source-level kernel attempts are
       queued/recorded → move on to SWEEP. GEMM tuning is the deterministic
       KERNEL-entry operator-level lever; if it succeeded and the LLM has
       not produced kernel_opt work, do not burn the entire KERNEL budget
       on heartbeat turns.
    4. Real :func:`compute_plateau_kernel` (KB_design §3.8 §5.2
       OR-of clauses).
    5. Phase budget exhausted.
    """
    if _pending_escalate_hint(state) == ESCALATE_HINT_SKIP_TO_SWEEP:
        return "no_more_leverage", {
            "evidence": "no_more_leverage",
            "hint": ESCALATE_HINT_SKIP_TO_SWEEP,
        }
    rejected = getattr(state, "rejected_kernel_ids", None) or []
    rejected_count = len(rejected) if isinstance(rejected, list) else 0
    last_gemm = getattr(state, "last_gemm_tuning", None) or {}
    kernel_attempts = getattr(state, "kernel_opt_attempts", None) or {}
    if (
        isinstance(last_gemm, dict)
        and str(last_gemm.get("status") or "").lower() in {"complete", "completed", "ok", "succeeded", "success"}
        and str(last_gemm.get("decision") or "").upper() == "KEEP"
        and not kernel_attempts
        and not bool(getattr(state, "continue_kernel_after_gemm", True))
    ):
        return "plateau_kernel", {
            "evidence": "gemm_tuning_complete_no_kernel_opt",
            "rejected_kernel_count": rejected_count,
            "gemm_speedup": last_gemm.get("best_speedup"),
            "tuned_file": last_gemm.get("tuned_file"),
        }
    triggered, evidence = compute_plateau_kernel(
        state,
        lookback=plateau_lookback,
        revert_streak_threshold=plateau_revert_streak,
        keep_gain_threshold_pct=plateau_keep_gain_pct,
    )
    if triggered:
        return "plateau_kernel", {
            "evidence": "plateau_judgment",
            "rejected_kernel_count": rejected_count,
            **evidence,
        }
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
    """SWEEP normal exit. sweep_done OR conc_sweep_done OR budget exhausted.

    Bug #12 fix: previously only sweep_done / budget_exhausted were
    honoured. When sweep was singleton-blocked but conc_sweep ran to
    completion, the phase had no exit signal and idled until budget
    exhaustion (wasting hours of GPU on repeated conc_sweep proposals).
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


# ---------------------------------------------------------------------------
# Transition decision (the only function the Coordinator calls each tick)
# ---------------------------------------------------------------------------
def _resolve_plateau_overrides(state: Any) -> dict[str, Any]:
    """Pull operator-tuned plateau thresholds off SharedState.

    CLI flags lock plateau thresholds at
    session start; phase_state reads them on every tick via
    :attr:`SharedState.plateau_overrides`. Empty / missing → library
    defaults.
    """
    overrides = getattr(state, "plateau_overrides", None) or {}
    return dict(overrides) if isinstance(overrides, dict) else {}


def _framework_pr_batch_is_complete(
    batch: dict[str, Any],
    progress_by_batch: dict[str, int],
) -> bool:
    """A FRAMEWORK_PR batch is "complete" iff every candidate it carries
    has a matching row in ``framework_pr_phase_progress`` (any terminal
    status — KEEP / REVERT / apply_failed / enqueue_failed / critic_denied).

    Used by the plateau judge so that a freshly-discovered batch whose
    first candidate just got enqueued cannot cause an early exit. Without
    this guard, ``max_gain_pct_observed_in_batch`` defaults to 0.0 on
    creation, making the brand-new batch look like another "no gain"
    data point and tripping plateau the moment ``lookback`` such 0.0
    entries appear in the tail.
    """
    candidates = batch.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return True
    total = sum(1 for c in candidates if isinstance(c, dict))
    if total == 0:
        return True
    batch_id = str(batch.get("batch_id") or "")
    processed = int(progress_by_batch.get(batch_id, 0))
    return processed >= total


def _framework_pr_pending_candidate_count(state: Any) -> int:
    """Count candidates discovered into a batch but missing a progress
    row. Surfaced in force-exit evidence so operators see how many
    candidates were skipped by the wall-clock guard."""
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
    lookback: int = DEFAULT_FRAMEWORK_PR_PLATEAU_LOOKBACK,
    plateau_keep_gain_pct: float = DEFAULT_FRAMEWORK_PR_PLATEAU_KEEP_GAIN_PCT,
    force_exit_hours_remaining_ratio: float = (
        DEFAULT_FRAMEWORK_PR_FORCE_EXIT_HOURS_REMAINING_RATIO
    ),
) -> tuple[str, dict[str, Any]] | None:
    """FRAMEWORK_PR normal exit.

    Priority order:

    0. HARD force-exit — fires when the session's remaining wall-clock
       drops below ``force_exit_hours_remaining_ratio * max_hours``.
       Reason: ``framework_pr_force_exit_low_budget``. Matches the
       design's "leave the rest of the session breathing room" intent.
       Evidence carries ``pending_candidate_count`` so operators see how
       much was skipped.
    1. Plateau — fires when the last ``lookback`` **fully-processed**
       batches in ``state.framework_pr_batches`` each have
       ``max_gain_pct_observed_in_batch < plateau_keep_gain_pct``. A
       batch is fully-processed when every candidate it carries has a
       matching row in ``framework_pr_phase_progress``; in-flight or
       newly-discovered batches do NOT count toward the lookback. This
       prevents the case where the pump enqueues the first candidate of
       a fresh batch, ``max_gain_pct_observed_in_batch`` is still its
       initial 0.0, and the plateau judge mistakes that for a real "no
       gain" data point and exits early.
    2. Normal completion — fires when the Coordinator marked the phase
       done by setting ``state.framework_pr_phase_done = True`` (no
       more candidates to enqueue / ``fa phase-discover`` returned 0).
       Reason: ``framework_pr_phase_done``.

    Returns ``None`` when none of the above apply (the Coordinator
    stays in FRAMEWORK_PR and enqueues the next batch).
    """
    # Priority 0 — force-exit on remaining wall-clock.
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

    # Priority 1 — plateau over the last ``lookback`` fully-processed batches.
    batches = getattr(state, "framework_pr_batches", None) or []
    lookback_int = int(lookback)
    if isinstance(batches, list) and lookback_int > 0 and len(batches) >= lookback_int:
        progress = getattr(state, "framework_pr_phase_progress", None) or []
        progress_by_batch: dict[str, int] = {}
        for row in progress:
            if isinstance(row, dict):
                bid = str(row.get("batch_id") or "")
                progress_by_batch[bid] = progress_by_batch.get(bid, 0) + 1
        complete_tail: list[dict[str, Any]] = []
        # Walk newest-to-oldest, take the most recent ``lookback`` *complete*
        # batches. Skipping incomplete batches means a still-pumping batch
        # cannot count toward (or block) plateau.
        for entry in reversed(batches):
            if not isinstance(entry, dict):
                continue
            if _framework_pr_batch_is_complete(entry, progress_by_batch):
                complete_tail.append(entry)
                if len(complete_tail) >= lookback_int:
                    break
        if len(complete_tail) >= lookback_int:
            max_gains: list[float] = []
            for entry in complete_tail:
                try:
                    max_gains.append(
                        float(entry.get("max_gain_pct_observed_in_batch") or 0.0)
                    )
                except (TypeError, ValueError):
                    max_gains.append(0.0)
            if max_gains and all(g < float(plateau_keep_gain_pct) for g in max_gains):
                return "framework_pr_plateau", {
                    "evidence":              "plateau_judgment",
                    "lookback":              lookback_int,
                    "keep_gain_pct_threshold": float(plateau_keep_gain_pct),
                    "batch_max_gains":       list(reversed(max_gains)),
                }

    # Priority 2 — Coordinator-signalled normal completion.
    if bool(getattr(state, "framework_pr_phase_done", False)):
        return "framework_pr_phase_done", {
            "evidence": "no_more_candidates",
            "batch_count": len(batches) if isinstance(batches, list) else 0,
        }

    return None


def _post_prelude_target(*, explore_enabled: bool, kernel_enabled: bool) -> str:
    """First active phase after PRELUDE / FRAMEWORK_PR.

    EXPLORE when enabled; otherwise KERNEL when enabled; otherwise SWEEP.
    Mirrors the EXPLORE-exit fallthrough (kernel disabled → SWEEP) so the
    ``--no-explore`` opt-out collapses the chain the same way
    ``--no-kernel`` does at the EXPLORE→ boundary.
    """
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
    disable_legacy_proxy: bool = False,
    framework_phase_enabled: bool = False,
    explore_enabled: bool = True,
    max_hours: float | None = None,
) -> tuple[str, str, dict[str, Any]] | None:
    """Return ``(next_phase, reason, evidence)`` or ``None``.

    The Coordinator calls this at the end of each tick. When it returns
    non-None, the Coordinator writes the transition to phase_history,
    updates ``state.phase`` + ``phase_started_*`` and persists. Priority
    order (Inv-8.2 + §3.8 §7.1): ``abort > exit_terminal > exit_normal``.
    Any global terminal stop_reason fires first regardless of phase.
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
            # Framework phase off → straight through to the first active
            # phase with the historical ``prelude_done`` reason. Neither
            # FRAMEWORK_PR nor EXPLORE has a dedicated "skipped" reason:
            # --no-framework / --no-explore simply collapse the chain and
            # the routing record stays backward-compatible with
            # pre-FRAMEWORK_PR sessions. ``explore_skipped`` evidence is
            # stamped when EXPLORE is bypassed (--no-explore).
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
            lookback=int(overrides.get(
                "framework_pr_lookback",
                DEFAULT_FRAMEWORK_PR_PLATEAU_LOOKBACK,
            )),
            plateau_keep_gain_pct=float(overrides.get(
                "framework_pr_keep_gain_pct",
                DEFAULT_FRAMEWORK_PR_PLATEAU_KEEP_GAIN_PCT,
            )),
            force_exit_hours_remaining_ratio=float(overrides.get(
                "framework_pr_force_exit_hours_ratio",
                DEFAULT_FRAMEWORK_PR_FORCE_EXIT_HOURS_REMAINING_RATIO,
            )),
        )
        if norm is not None:
            # FRAMEWORK_PR → EXPLORE normally; --no-explore collapses
            # straight to KERNEL (or SWEEP when --no-kernel too).
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
            plateau_keep_gain_pct=float(overrides.get(
                "explore_keep_gain_pct",
                DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
            )),
            plateau_empty_streak=int(overrides.get(
                "explore_empty_streak",
                DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
            )),
            plateau_lookback=int(overrides.get(
                "explore_lookback",
                DEFAULT_PLATEAU_EXPLORE_LOOKBACK,
            )),
            disable_legacy_proxy=disable_legacy_proxy,
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
            # skipping the KERNEL hop (the steward / safety net judged
            # there is no leverage left to chase). SWEEP → CLOSE follows.
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
            plateau_revert_streak=int(overrides.get(
                "kernel_revert_streak",
                DEFAULT_PLATEAU_KERNEL_REVERT_STREAK,
            )),
            plateau_keep_gain_pct=float(overrides.get(
                "kernel_keep_gain_pct",
                DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT,
            )),
            plateau_lookback=int(overrides.get(
                "kernel_lookback",
                DEFAULT_PLATEAU_KERNEL_LOOKBACK,
            )),
        )
        if norm is not None:
            return PHASE_SWEEP, norm[0], norm[1]
        return None

    if current == PHASE_SWEEP:
        norm = exit_normal_sweep(state, budget_pct=budget_pct, now_unix=now_unix)
        if norm is not None:
            return PHASE_CLOSE, norm[0], norm[1]
        return None

    # PHASE_CLOSE — terminal, no further transitions.
    return None


# ---------------------------------------------------------------------------
# v0.6 → legacy resume inference
# ---------------------------------------------------------------------------
def infer_phase_from_state(state: Any) -> tuple[str, dict[str, Any]]:
    """Best-effort inference for sessions that lack a ``phase`` field.

    Returns ``(phase, evidence)`` where evidence captures the inference
    inputs so phase_history surfaces an audit row.

    Inference rules (precedence top-down):

    1. ``stop_reason`` is set → CLOSE
    2. ``baseline_tput <= 0`` → PRELUDE
    3. ``last_sweep`` present + non-empty → SWEEP
    4. ``last_kernel_opt`` present + ``kernel_enabled`` → KERNEL
    5. ``optimization_stack`` non-empty → EXPLORE
    6. Fallback → EXPLORE
    """
    sr = (getattr(state, "stop_reason", "") or "").strip()
    if sr:
        return PHASE_CLOSE, {"inferred_from": "stop_reason", "stop_reason": sr}
    try:
        tput = float(getattr(state, "baseline_tput", 0.0) or 0.0)
    except (TypeError, ValueError):
        tput = 0.0
    if tput <= 0:
        return PHASE_PRELUDE, {"inferred_from": "baseline_tput=0"}
    last_sweep = getattr(state, "last_sweep", None) or {}
    if isinstance(last_sweep, dict) and last_sweep:
        return PHASE_SWEEP, {"inferred_from": "last_sweep_present"}
    last_kernel_opt = getattr(state, "last_kernel_opt", None) or {}
    kernel_enabled = bool(getattr(state, "kernel_enabled", True))
    if isinstance(last_kernel_opt, dict) and last_kernel_opt and kernel_enabled:
        return PHASE_KERNEL, {"inferred_from": "last_kernel_opt_present"}
    opt_stack = getattr(state, "optimization_stack", None) or []
    if isinstance(opt_stack, list) and opt_stack:
        return PHASE_EXPLORE, {"inferred_from": "optimization_stack_nonempty"}
    return PHASE_EXPLORE, {"inferred_from": "fallback"}


# ---------------------------------------------------------------------------
# phase_history helper (shape used by SharedState.record_phase_transition)
# ---------------------------------------------------------------------------
def make_history_row(
    *,
    from_phase: str,
    to_phase: str,
    reason: str,
    evidence: dict[str, Any] | None,
    ts: str,
    ts_unix: float,
) -> dict[str, Any]:
    """Construct a canonical phase_history row.

    PolicyGate / breakdown collectors share this exact shape (Inv-2.2 +
    KB_design §3.2 §6). ``reason`` is *not* validated here — callers
    that want the strict check go through
    :func:`is_valid_phase_exit_reason` first; we keep the constructor
    lenient so resume tools can emit synthetic rows during recovery.
    """
    return {
        "from_phase": (from_phase or "").strip().upper(),
        "to_phase":   (to_phase or "").strip().upper(),
        "reason":     (reason or "").strip(),
        "evidence":   dict(evidence or {}),
        "ts":         ts,
        "ts_unix":    float(ts_unix or 0.0),
    }


__all__ = [
    "DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT",
    "DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING",
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
    "abort_prelude",
    "allowed_actions_for",
    "apply_escalate_budget_bump",
    "compute_next_phase",
    "compute_plateau_explore",
    "compute_plateau_kernel",
    "exit_normal_explore",
    "exit_normal_framework_pr",
    "exit_normal_kernel",
    "exit_normal_prelude",
    "exit_normal_sweep",
    "exit_terminal_prelude",
    "infer_phase_from_state",
    "is_action_allowed_in_phase",
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
    "wants_steward_assessment",
]
