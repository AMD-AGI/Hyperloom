# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""v0.8 M7 — plateau pure functions + escalate hints + stop_reason ENUM.

Covers KB_design §3.8 + §3.13 M7:

* ``compute_plateau_explore`` AND-of (recent_keep_gain < ε AND
  recent_empty_streak ≥ ε).
* ``compute_plateau_kernel`` OR-of (revert_streak OR keep_gain).
* ``exit_normal_explore`` / ``exit_normal_kernel`` plug the real
  plateau pure functions, with M2 proxy fallback when explore_search
  is empty.
* ``escalate_strategy_change`` hint plumbing:
  ``skip_to_kernel`` / ``skip_to_close`` / ``extend_*_budget`` /
  ``pause_specialist_<domain>``.
* ``apply_escalate_budget_bump`` clamps to ESCALATE_HINT_BUDGET_BUMP_CAP.
* ``SharedState.set_stop_reason`` lenient + strict ENUM enforcement.
* ``SharedState.set_pending_escalate_hint`` /
  ``consume_pending_escalate_hint``.
* ``collect_attribution.phase_breakdown`` per-phase bucketing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator.phase_state import (
    DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
    ESCALATE_HINT_BUDGET_BUMP_CAP,
    ESCALATE_HINT_BUDGET_BUMP_DELTA,
    ESCALATE_HINT_SKIP_TO_CLOSE,
    ESCALATE_HINT_SKIP_TO_KERNEL,
    ESCALATE_HINT_SKIP_TO_SWEEP,
    ESCALATE_HINT_VOCAB,
    PHASE_CLOSE,
    PHASE_KERNEL,
    PHASE_SWEEP,
    STOP_REASON_VOCAB,
    apply_escalate_budget_bump,
    compute_next_phase,
    compute_plateau_explore,
    compute_plateau_kernel,
    exit_normal_explore,
    exit_normal_kernel,
    is_pause_specialist_hint,
    is_valid_escalate_hint,
    is_valid_stop_reason,
)
from inference_optimizer.orchestrator.shared_state import SharedState


# ===========================================================================
# 1. ESCALATE_HINT vocab
# ===========================================================================
def test_escalate_hint_vocab_closed():
    assert ESCALATE_HINT_VOCAB == frozenset({
        "skip_to_kernel", "skip_to_sweep", "skip_to_close",
        "extend_explore_budget", "extend_kernel_budget",
    })


def test_is_valid_escalate_hint_accepts_vocab_and_pause_specialist():
    assert is_valid_escalate_hint("skip_to_kernel")
    assert is_valid_escalate_hint("pause_specialist_serving_specialist")
    assert is_valid_escalate_hint("pause_specialist_anything")
    assert not is_valid_escalate_hint("garbage")
    assert not is_valid_escalate_hint("")
    # The bare prefix is invalid — must have a suffix.
    assert not is_valid_escalate_hint("pause_specialist_")


def test_is_pause_specialist_hint_requires_suffix():
    assert is_pause_specialist_hint("pause_specialist_x")
    assert not is_pause_specialist_hint("pause_specialist_")
    assert not is_pause_specialist_hint("skip_to_kernel")


# ===========================================================================
# 2. compute_plateau_explore (KB_design §3.8 §5.1 AND-of)
# ===========================================================================
def test_plateau_explore_empty_state_returns_false():
    state = SimpleNamespace()
    triggered, ev = compute_plateau_explore(state)
    assert triggered is False
    assert ev["empty_streak"] == 0
    assert ev["recent_keep_gain_pct"] == 0.0


def test_plateau_explore_AND_low_gain_AND_streak_triggers():
    state = SimpleNamespace(
        explore_search={"winners_history": [
            {"gain_pct": 0.1},
            {"gain_pct": 0.05},
        ]},
        specialist_rounds=[
            {"proposals_total": 1, "proposals_kept": 1},
            {"proposals_total": 0, "proposals_kept": 0},
            {"proposals_total": 0, "proposals_kept": 0},
            {"proposals_total": 0, "proposals_kept": 0},
        ],
    )
    triggered, ev = compute_plateau_explore(state)
    assert triggered is True
    assert ev["empty_streak"] == 3
    assert ev["recent_keep_gain_pct"] < DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT


def test_plateau_explore_high_gain_blocks_trigger():
    """Even with empty streak, large recent KEEP gain blocks plateau."""
    state = SimpleNamespace(
        explore_search={"winners_history": [
            {"gain_pct": 3.0},
            {"gain_pct": 2.0},
        ]},
        specialist_rounds=[
            {"proposals_total": 0, "proposals_kept": 0},
            {"proposals_total": 0, "proposals_kept": 0},
            {"proposals_total": 0, "proposals_kept": 0},
        ],
    )
    triggered, _ev = compute_plateau_explore(state)
    assert triggered is False


def test_plateau_explore_short_empty_streak_blocks_trigger():
    """Low gain alone (without empty streak) does not trigger plateau."""
    state = SimpleNamespace(
        explore_search={"winners_history": [{"gain_pct": 0.1}]},
        specialist_rounds=[
            {"proposals_total": 0, "proposals_kept": 0},
            # newest round produced something → streak resets to 0
            {"proposals_total": 5, "proposals_kept": 2},
        ],
    )
    triggered, ev = compute_plateau_explore(state)
    assert triggered is False
    assert ev["empty_streak"] == 0


def test_plateau_explore_supports_threshold_overrides():
    state = SimpleNamespace(
        explore_search={"winners_history": [{"gain_pct": 1.5}]},
        specialist_rounds=[
            {"proposals_total": 0, "proposals_kept": 0},
        ],
    )
    # Defaults: 0.5% threshold, 3-streak → not triggered (gain too high).
    triggered, _ = compute_plateau_explore(state)
    assert triggered is False
    # Raise threshold to 3.0 (gain 1.5 < 3.0) and drop streak to 1.
    triggered, _ = compute_plateau_explore(
        state,
        keep_gain_threshold_pct=3.0,
        empty_streak_threshold=1,
    )
    assert triggered is True


# ===========================================================================
# 3. compute_plateau_kernel (KB_design §3.8 §5.2 OR-of)
# ===========================================================================
def test_plateau_kernel_revert_streak_triggers():
    """3 consecutive REVERTs → triggered."""
    state = SimpleNamespace(
        kernel_integrate_attempts={
            "k1": {"attempts": [
                {"decision": "REVERT", "ts": "2026-05-19T18:00:00"},
            ]},
            "k2": {"attempts": [
                {"decision": "REVERT", "ts": "2026-05-19T18:01:00"},
            ]},
            "k3": {"attempts": [
                {"decision": "REVERT", "ts": "2026-05-19T18:02:00"},
            ]},
        },
    )
    triggered, ev = compute_plateau_kernel(state)
    assert triggered is True
    assert ev["revert_streak"] == 3


def test_plateau_kernel_low_gain_triggers():
    """Low cumulative KEEP gain alone triggers (OR semantics)."""
    state = SimpleNamespace(
        kernel_integrate_attempts={
            "k1": {"attempts": [
                {"decision": "KEEP", "ts": "2026-05-19T18:00:00", "gain_pct": 0.1},
            ]},
        },
    )
    triggered, ev = compute_plateau_kernel(state)
    assert triggered is True
    assert ev["recent_keep_gain_pct"] == 0.1


def test_plateau_kernel_high_gain_blocks_revert_streak():
    """When the REVERT streak is below threshold and gain is large,
    plateau doesn't fire."""
    state = SimpleNamespace(
        kernel_integrate_attempts={
            "k1": {"attempts": [
                {"decision": "KEEP", "ts": "2026-05-19T18:00:00", "gain_pct": 5.0},
            ]},
            "k2": {"attempts": [
                {"decision": "REVERT", "ts": "2026-05-19T18:01:00"},
            ]},
        },
    )
    triggered, _ev = compute_plateau_kernel(state)
    assert triggered is False


def test_plateau_kernel_zero_lookback_returns_false():
    state = SimpleNamespace(kernel_integrate_attempts={})
    triggered, ev = compute_plateau_kernel(state, lookback=0)
    assert triggered is False
    assert "thresholds_disabled" in ev.get("reason", "")


def test_plateau_kernel_empty_attempts_does_not_trigger():
    """Fix A: zero kernel attempts must NOT flip plateau via the
    ``recent_keep_gain == 0.0 < 0.5`` arm.

    Pre-fix this returned ``(True, ...)`` the moment KERNEL entered
    with an empty ``kernel_integrate_attempts`` ledger — coupled with
    EXPLORE that produced no KEEPs (e.g. force-exit on low budget),
    the session went EXPLORE → KERNEL → SWEEP without ever spawning a
    single ``trace_analyze`` / ``run_optimization`` request.
    """
    state = SimpleNamespace(kernel_integrate_attempts={})
    triggered, ev = compute_plateau_kernel(state)
    assert triggered is False
    assert ev.get("reason") == "no_kernel_attempts_yet"
    assert ev.get("attempts_seen") == 0


def test_plateau_kernel_empty_attempts_dict_with_no_entries_does_not_trigger():
    """Same Fix-A invariant when the ledger has keys but every entry
    is structurally empty (resume cleanup edge case)."""
    state = SimpleNamespace(
        kernel_integrate_attempts={
            "k_pruned": {"attempts": []},
            "k_corrupt": {},
        },
    )
    triggered, ev = compute_plateau_kernel(state)
    assert triggered is False
    assert ev.get("reason") == "no_kernel_attempts_yet"


# ===========================================================================
# 4. exit_normal_explore / exit_normal_kernel — wired to real plateau
# ===========================================================================
def test_exit_normal_explore_does_not_exit_on_plateau():
    """Loosen P3_17: plateau is advisory only; the EXPLORE exit is now
    driven exclusively by IR-6 force-exit, an explicit skip hint, or
    phase budget exhaustion. A bare plateau signal must NOT exit."""
    state = SimpleNamespace(
        phase="EXPLORE",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        explore_search={"winners_history": [{"gain_pct": 0.1}]},
        specialist_rounds=[
            {"proposals_total": 0, "proposals_kept": 0},
            {"proposals_total": 0, "proposals_kept": 0},
            {"proposals_total": 0, "proposals_kept": 0},
        ],
        pending_escalate_hint="",
        stop_reason="",
        plateau_overrides={},
    )
    assert exit_normal_explore(state) is None


def test_exit_normal_explore_skip_to_kernel_hint_short_circuits():
    """``skip_to_kernel`` hint forces ``plateau_explore`` even when the
    real signals don't agree."""
    state = SimpleNamespace(
        phase="EXPLORE",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        explore_search={"winners_history": [{"gain_pct": 5.0}]},
        specialist_rounds=[{"proposals_total": 10, "proposals_kept": 8}],
        pending_escalate_hint=ESCALATE_HINT_SKIP_TO_KERNEL,
        stop_reason="",
    )
    out = exit_normal_explore(state)
    assert out is not None and out[0] == "plateau_explore"
    assert out[1]["evidence"] == "llm_escalation"


def test_exit_normal_kernel_does_not_exit_on_plateau():
    """Loosen P3_17: KERNEL plateau is advisory only; only the
    skip_to_sweep hint or phase-budget exhaustion may exit KERNEL."""
    state = SimpleNamespace(
        phase="KERNEL",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        kernel_integrate_attempts={
            f"k{i}": {"attempts": [{"decision": "REVERT", "ts": f"2026-05-19T18:0{i}:00"}]}
            for i in range(3)
        },
        rejected_kernel_ids=[],
        pending_escalate_hint="",
        stop_reason="",
    )
    assert exit_normal_kernel(state) is None


def test_exit_normal_kernel_after_gemm_does_not_exit():
    """Loosen P3_17: the GEMM-completed shortcut is removed.
    GEMM completion alone never advances KERNEL → SWEEP."""
    state = SimpleNamespace(
        phase="KERNEL",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        kernel_integrate_attempts={},
        kernel_opt_attempts={},
        continue_kernel_after_gemm=False,
        rejected_kernel_ids=[],
        last_gemm_tuning={
            "status": "complete",
            "decision": "KEEP",
            "best_speedup": 1.48,
            "tuned_file": "/tmp/tuned.csv",
        },
        stop_reason="",
    )
    assert exit_normal_kernel(state) is None


def test_compute_next_phase_skip_to_close_routes_to_close():
    state = SimpleNamespace(
        phase="EXPLORE",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        explore_search={},
        specialist_rounds=[],
        params_no_promote_streak=0,
        backends_search={},
        optimization_stack=[],
        pending_escalate_hint=ESCALATE_HINT_SKIP_TO_CLOSE,
        stop_reason="",
        plateau_overrides={},
    )
    out = compute_next_phase(state, kernel_enabled=True)
    assert out is not None
    target, reason, evidence = out
    assert target == PHASE_CLOSE
    assert reason == "robustness_escalated"
    assert evidence.get("terminal") is True
    assert evidence.get("hint") == ESCALATE_HINT_SKIP_TO_CLOSE


def _skip_to_sweep_state(phase: str) -> SimpleNamespace:
    return SimpleNamespace(
        phase=phase,
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        explore_search={},
        specialist_rounds=[],
        params_no_promote_streak=0,
        backends_search={},
        rejected_kernel_ids=[],
        optimization_stack=[],
        pending_escalate_hint=ESCALATE_HINT_SKIP_TO_SWEEP,
        stop_reason="",
        plateau_overrides={},
    )


def test_exit_normal_explore_skip_to_sweep_is_non_terminal():
    # skip_to_sweep (steward stop_session / safety net) is the new
    # non-terminal "no more leverage" signal.
    out = exit_normal_explore(_skip_to_sweep_state("EXPLORE"))
    assert out is not None
    reason, evidence = out
    assert reason == "no_more_leverage"
    assert evidence.get("hint") == ESCALATE_HINT_SKIP_TO_SWEEP


def test_compute_next_phase_skip_to_sweep_from_explore_skips_kernel():
    # Even with kernel enabled, no_more_leverage routes EXPLORE -> SWEEP
    # (no KERNEL hop) and is NOT terminal.
    out = compute_next_phase(_skip_to_sweep_state("EXPLORE"), kernel_enabled=True)
    assert out is not None
    target, reason, evidence = out
    assert target == PHASE_SWEEP
    assert reason == "no_more_leverage"
    assert evidence.get("terminal") is not True


def test_compute_next_phase_skip_to_sweep_from_kernel_routes_to_sweep():
    out = compute_next_phase(_skip_to_sweep_state("KERNEL"), kernel_enabled=True)
    assert out is not None
    target, reason, _ = out
    assert target == PHASE_SWEEP
    assert reason == "no_more_leverage"


# ===========================================================================
# 5. apply_escalate_budget_bump
# ===========================================================================
def test_apply_escalate_budget_bump_lifts_phase_within_cap():
    out = apply_escalate_budget_bump(
        {"EXPLORE": 0.60}, phase="EXPLORE",
    )
    assert out["EXPLORE"] == pytest.approx(
        0.60 + ESCALATE_HINT_BUDGET_BUMP_DELTA,
    )


def test_apply_escalate_budget_bump_clamps_to_cap():
    out = apply_escalate_budget_bump(
        {"EXPLORE": 0.95}, phase="EXPLORE",
    )
    assert out["EXPLORE"] == ESCALATE_HINT_BUDGET_BUMP_CAP


def test_apply_escalate_budget_bump_ignores_unknown_phase():
    inp = {"EXPLORE": 0.60}
    out = apply_escalate_budget_bump(inp, phase="NOT_A_PHASE")
    # No bump, but the helper still returns a normalised copy with
    # all known phases populated.
    assert out["EXPLORE"] == 0.60


# ===========================================================================
# 6. SharedState — escalate hint plumbing + stop_reason ENUM
# ===========================================================================
def test_set_pending_escalate_hint_accepts_vocab():
    s = SharedState()
    assert s.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_KERNEL) == "skip_to_kernel"
    assert s.pending_escalate_hint == "skip_to_kernel"


def test_set_pending_escalate_hint_drops_unknown():
    s = SharedState()
    assert s.set_pending_escalate_hint("garbage") == ""
    assert s.pending_escalate_hint == ""


def test_consume_pending_escalate_hint_clears_and_audits():
    s = SharedState()
    s.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_KERNEL)
    consumed = s.consume_pending_escalate_hint()
    assert consumed == "skip_to_kernel"
    assert s.pending_escalate_hint == ""
    assert s.last_consumed_escalate_hint == "skip_to_kernel"
    assert s.last_consumed_escalate_hint_ts != ""


def test_consume_pending_escalate_hint_noop_when_empty():
    s = SharedState()
    assert s.consume_pending_escalate_hint() == ""
    assert s.last_consumed_escalate_hint == ""


def test_set_stop_reason_accepts_vocab():
    s = SharedState()
    assert s.set_stop_reason("target_reached") == "target_reached"
    assert s.stop_reason == "target_reached"


def test_set_stop_reason_lenient_maps_unknown_to_unknown(caplog):
    s = SharedState()
    with caplog.at_level("WARNING"):
        v = s.set_stop_reason("not_a_real_reason")
    assert v == "unknown"
    assert s.stop_reason == "unknown"


def test_set_stop_reason_strict_raises():
    s = SharedState()
    with pytest.raises(ValueError, match="not in STOP_REASON_VOCAB"):
        s.set_stop_reason("not_a_real_reason", strict=True)


def test_set_stop_reason_empty_string_clears():
    s = SharedState()
    s.set_stop_reason("target_reached")
    assert s.stop_reason == "target_reached"
    s.set_stop_reason("")
    assert s.stop_reason == ""


def test_stop_reason_vocab_has_v08_additions():
    for new in (
        "plateau_explore", "plateau_kernel", "no_kernel_skipped",
        "sweep_done", "robustness_escalated", "user_stop_requested",
        "cortex_drain_failed", "cortex_t0_failed", "cortex_commit_failed",
        "prelude_baseline_failed", "prelude_policy_loop",
        "time_exhausted_during_prelude", "crash_threshold_exceeded",
    ):
        assert new in STOP_REASON_VOCAB
        assert is_valid_stop_reason(new)


# ===========================================================================
# 7. plateau is advisory only — pure compute_plateau_* still works
# ===========================================================================
def test_compute_next_phase_does_not_advance_on_plateau():
    """Loosen P3_17: even when the EXPLORE plateau judge fires,
    compute_next_phase returns None unless an explicit hint or hard
    budget gate is active."""
    state = SimpleNamespace(
        phase="EXPLORE",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        explore_search={"winners_history": [{"gain_pct": 0.1}]},
        specialist_rounds=[
            {"proposals_total": 0, "proposals_kept": 0},
            {"proposals_total": 0, "proposals_kept": 0},
            {"proposals_total": 0, "proposals_kept": 0},
        ],
        params_no_promote_streak=0,
        backends_search={},
        optimization_stack=[],
        pending_escalate_hint="",
        stop_reason="",
        plateau_overrides={},
    )
    assert compute_next_phase(state, kernel_enabled=True) is None
    triggered, _ = compute_plateau_explore(state)
    assert triggered is True


# ===========================================================================
# 8. breakdown.attribution.phase_breakdown
# ===========================================================================
def test_collect_phase_breakdown_buckets_by_phase():
    from inference_optimizer.breakdown.collectors import collect_attribution

    state = {
        "cumulative_gain_validated": 12.5,
        "optimization_stack": [
            {"action": "explore", "variant_name": "v1"},
            {"action": "integrate", "kernel_id": "fmoe_fp8"},
        ],
        "gain_per_stack_entry": [
            {
                "action": "explore",
                "variant_name": "v1",
                "fingerprint": "fpfp",
                "delta_pct": 5.0,
                "ts_unix": 100.0,
            },
            {
                "action": "integrate",
                "kernel_id": "fmoe_fp8",
                "delta_pct": 7.5,
                "ts_unix": 300.0,
            },
        ],
        "phase_history": [
            {"to_phase": "PRELUDE", "ts_unix": 0.0, "reason": "phase_entered"},
            {"to_phase": "EXPLORE", "ts_unix": 50.0, "reason": "prelude_done"},
            {"to_phase": "KERNEL",  "ts_unix": 200.0, "reason": "plateau_explore"},
            {"to_phase": "SWEEP",   "ts_unix": 400.0, "reason": "plateau_kernel"},
        ],
        "explore_search": {
            "winners_history": [
                {"fingerprint": "fpfp", "provenance": "serving_specialist"},
            ],
        },
    }
    out = collect_attribution(state, [], [], [], [])
    pb = out["phase_breakdown"]
    assert pb["explore"]["total_gain_pct"] == 5.0
    assert pb["explore"]["by_domain"]["serving_specialist"] == 5.0
    assert pb["kernel"]["total_gain_pct"] == 7.5
    assert pb["kernel"]["by_kernel_id"]["fmoe_fp8"] == 7.5
    # No prelude/sweep contributions.
    assert pb["prelude"]["total_gain_pct"] == 0.0
    assert pb["sweep"]["total_gain_pct"] == 0.0


def test_collect_phase_breakdown_falls_back_to_action_family_when_history_empty():
    """Legacy resume — no phase_history → action-family fallback."""
    from inference_optimizer.breakdown.collectors import collect_attribution

    state = {
        "cumulative_gain_validated": 2.5,
        "optimization_stack": [{"action": "explore"}],
        "gain_per_stack_entry": [
            {
                "action": "explore",
                "fingerprint": "fp1",
                "delta_pct": 2.5,
                "ts_unix": 100.0,
            },
        ],
        "phase_history": [],
    }
    warnings: list[str] = []
    out = collect_attribution(state, [], [], [], warnings)
    pb = out["phase_breakdown"]
    assert pb["explore"]["total_gain_pct"] == 2.5
    # Default provenance bucket when winners_history doesn't supply one.
    assert pb["explore"]["by_domain"].get("default_grid", 0.0) == 2.5
    # Warning logged for the fallback path.
    assert any("phase_history empty" in w for w in warnings)


def test_collect_phase_breakdown_skips_zero_or_negative_deltas():
    """Negative / None deltas (e.g. validate_stack measurement) don't
    enter the per-phase bucket."""
    from inference_optimizer.breakdown.collectors import collect_attribution

    state = {
        "optimization_stack": [{"action": "explore"}],
        "gain_per_stack_entry": [
            {"action": "explore", "delta_pct": -0.5, "ts_unix": 100.0},
            {"action": "explore", "delta_pct": None, "ts_unix": 110.0},
        ],
        "phase_history": [
            {"to_phase": "EXPLORE", "ts_unix": 0.0, "reason": "prelude_done"},
        ],
    }
    out = collect_attribution(state, [], [], [], [])
    assert out["phase_breakdown"]["explore"]["total_gain_pct"] == 0.0
