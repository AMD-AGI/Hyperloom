# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""plateau pure functions + escalate hints + stop_reason ENUM."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.phases.machine_state import (
    DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
    DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
    ESCALATE_HINT_BUDGET_BUMP_CAP,
    ESCALATE_HINT_BUDGET_BUMP_DELTA,
    ESCALATE_HINT_SKIP_TO_CLOSE,
    ESCALATE_HINT_SKIP_TO_KERNEL,
    ESCALATE_HINT_SKIP_TO_SWEEP,
    ESCALATE_HINT_VOCAB,
    PHASE_CLOSE,
    PHASE_KERNEL_AGENT,
    PHASE_SWEEP,
    STOP_REASON_VOCAB,
    apply_escalate_budget_bump,
    compute_next_phase,
    compute_plateau_explore,
    compute_plateau_kernel,
    exit_normal_optimize,
    exit_normal_kernel,
    is_valid_escalate_hint,
    is_valid_stop_reason,
    kernel_work_pending,
)
from hyperloom.orchestrator.state import shared_state
from hyperloom.orchestrator.state.shared_state import SharedState


def test_escalate_hint_vocab_closed():
    assert ESCALATE_HINT_VOCAB == frozenset(
        {
            "skip_to_kernel",
            "skip_to_sweep",
            "skip_to_close",
            "extend_explore_budget",
            "extend_kernel_budget",
        }
    )


def test_is_valid_escalate_hint_accepts_vocab():
    assert is_valid_escalate_hint("skip_to_kernel")
    assert not is_valid_escalate_hint("garbage")
    assert not is_valid_escalate_hint("")


def test_plateau_explore_empty_state_returns_false():
    state = SimpleNamespace()
    triggered, ev = compute_plateau_explore(state)
    assert triggered is False
    assert ev["empty_streak"] == 0
    assert ev["recent_keep_gain_pct"] == 0.0


def test_plateau_explore_AND_low_gain_AND_streak_triggers():
    state = SimpleNamespace(
        explore_search={
            "winners_history": [
                {"gain_pct": 0.1},
                {"gain_pct": 0.05},
            ]
        },
        specialist_rounds=(
            [{"proposals_total": 1, "proposals_kept": 1}]
            + [{"proposals_total": 0, "proposals_kept": 0} for _ in range(DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK)]
        ),
    )
    triggered, ev = compute_plateau_explore(state)
    assert triggered is True
    assert ev["empty_streak"] == DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK
    assert ev["recent_keep_gain_pct"] < DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT


def test_plateau_explore_high_gain_blocks_trigger():
    """Even with empty streak, large recent KEEP gain blocks plateau."""
    state = SimpleNamespace(
        explore_search={
            "winners_history": [
                {"gain_pct": 3.0},
                {"gain_pct": 2.0},
            ]
        },
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
            # newest round produced something → streak resets to 0.
            {"proposals_total": 5, "proposals_kept": 2},
        ],
    )
    triggered, ev = compute_plateau_explore(state)
    assert triggered is False
    assert ev["empty_streak"] == 0


def test_plateau_explore_ignores_prior_macro_cycle_rows():
    state = SimpleNamespace(
        macro_cycle=1,
        explore_search={"winners_history": [{"gain_pct": 0.0, "cycle": 0}]},
        specialist_rounds=[
            {"proposals_total": 0, "proposals_kept": 0, "cycle": 0} for _ in range(DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK)
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
    # Defaults → not triggered (gain too high).
    triggered, _ = compute_plateau_explore(state)
    assert triggered is False
    # Raise threshold above the gain and drop streak to 1 → triggers.
    triggered, _ = compute_plateau_explore(
        state,
        keep_gain_threshold_pct=3.0,
        empty_streak_threshold=1,
    )
    assert triggered is True


def test_plateau_kernel_revert_streak_triggers():
    """3 consecutive REVERTs → triggered."""
    state = SimpleNamespace(
        kernel_integrate_attempts={
            "k1": {
                "attempts": [
                    {"decision": "REVERT", "ts": "2026-05-19T18:00:00"},
                ]
            },
            "k2": {
                "attempts": [
                    {"decision": "REVERT", "ts": "2026-05-19T18:01:00"},
                ]
            },
            "k3": {
                "attempts": [
                    {"decision": "REVERT", "ts": "2026-05-19T18:02:00"},
                ]
            },
        },
    )
    triggered, ev = compute_plateau_kernel(state)
    assert triggered is True
    assert ev["revert_streak"] == 3


def test_plateau_kernel_low_gain_triggers():
    """Low cumulative KEEP gain alone triggers (OR semantics)."""
    state = SimpleNamespace(
        kernel_integrate_attempts={
            "k1": {
                "attempts": [
                    {"decision": "KEEP", "ts": "2026-05-19T18:00:00", "gain_pct": 0.1},
                ]
            },
        },
    )
    triggered, ev = compute_plateau_kernel(state)
    assert triggered is True
    assert ev["recent_keep_gain_pct"] == 0.1


def test_plateau_kernel_ignores_prior_macro_cycle_attempts():
    state = SimpleNamespace(
        macro_cycle=1,
        kernel_integrate_attempts={
            "k1": {
                "attempts": [
                    {"decision": "REVERT", "ts": "2026-05-19T18:00:00", "cycle": 0},
                    {"decision": "REVERT", "ts": "2026-05-19T18:01:00", "cycle": 0},
                    {"decision": "REVERT", "ts": "2026-05-19T18:02:00", "cycle": 0},
                ]
            }
        },
    )
    triggered, ev = compute_plateau_kernel(state)
    assert triggered is False
    assert ev["reason"] == "no_kernel_attempts_yet"


def test_plateau_kernel_high_gain_blocks_revert_streak():
    """When the REVERT streak is below threshold and gain is large, plateau doesn't fire."""
    state = SimpleNamespace(
        kernel_integrate_attempts={
            "k1": {
                "attempts": [
                    {"decision": "KEEP", "ts": "2026-05-19T18:00:00", "gain_pct": 5.0},
                ]
            },
            "k2": {
                "attempts": [
                    {"decision": "REVERT", "ts": "2026-05-19T18:01:00"},
                ]
            },
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
    """Zero kernel attempts must NOT flip plateau via the ``recent_keep_gain == 0.0 < 0.5`` arm."""
    state = SimpleNamespace(kernel_integrate_attempts={})
    triggered, ev = compute_plateau_kernel(state)
    assert triggered is False
    assert ev.get("reason") == "no_kernel_attempts_yet"
    assert ev.get("attempts_seen") == 0


def test_plateau_kernel_empty_attempts_dict_with_no_entries_does_not_trigger():
    """Same invariant when the ledger has keys but every entry is structurally empty."""
    state = SimpleNamespace(
        kernel_integrate_attempts={
            "k_pruned": {"attempts": []},
            "k_corrupt": {},
        },
    )
    triggered, ev = compute_plateau_kernel(state)
    assert triggered is False
    assert ev.get("reason") == "no_kernel_attempts_yet"


def test_reset_per_cycle_plateau_state_preserves_durable_ledgers():
    state = SharedState(session_id="t")
    state.params_no_promote_streak = 4
    state.framework_agent_phase_done = True
    state.framework_agent_discover_failures = 2
    state.framework_agent_empty_discoveries = 2
    state.specialist_domain_empty_streak = {"serving_specialist": 3}
    state.rounds_since_last_specialist = {"serving_specialist": 4}
    state.rounds_since_last_keep = {"serving_specialist": 5}
    state.last_sweep = {"status": "succeeded"}
    state.last_conc_sweep = {"status": "succeeded"}
    state.explore_search = {"tested": {"stable": {"cycle": 0}}}
    state.kernel_integrate_attempts = {"stable": {"attempts": [{"cycle": 0}]}}

    state.reset_per_cycle_plateau_state()

    assert state.params_no_promote_streak == 0
    assert state.framework_agent_phase_done is False
    assert state.framework_agent_discover_failures == 0
    assert state.framework_agent_empty_discoveries == 0
    assert state.specialist_domain_empty_streak == {}
    assert state.rounds_since_last_specialist == {}
    assert state.rounds_since_last_keep == {}
    assert state.last_sweep == {}
    assert state.last_conc_sweep == {}
    assert state.explore_search["tested"]["stable"]["cycle"] == 0
    assert state.kernel_integrate_attempts["stable"]["attempts"][0]["cycle"] == 0


def test_exit_normal_optimize_exits_on_plateau():
    """Both arms dry advances to the next lever."""
    state = SimpleNamespace(
        phase="FRAMEWORK_AGENT",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        explore_search={"winners_history": [{"gain_pct": 0.1}]},
        specialist_rounds=[
            {"proposals_total": 0, "proposals_kept": 0} for _ in range(DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK)
        ],
        pending_escalate_hint="",
        stop_reason="",
        plateau_overrides={},
        # Both arms must be dry before the merged phase may leave.
        framework_agent_phase_done=True,
    )
    out = exit_normal_optimize(state)
    assert out is not None
    assert out[0] == "optimize_no_more_leverage"


def test_exit_normal_optimize_skip_to_kernel_hint_short_circuits():
    """A ``skip_to_kernel`` hint exits even when the arms' own signals disagree."""
    state = SimpleNamespace(
        phase="FRAMEWORK_AGENT",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        explore_search={"winners_history": [{"gain_pct": 5.0}]},
        specialist_rounds=[{"proposals_total": 10, "proposals_kept": 8}],
        pending_escalate_hint=ESCALATE_HINT_SKIP_TO_KERNEL,
        stop_reason="",
    )
    out = exit_normal_optimize(state)
    assert out is not None and out[0] == "optimize_no_more_leverage"
    assert out[1]["evidence"] == "llm_escalation"


def test_exit_normal_kernel_does_not_exit_on_plateau():
    """KERNEL_AGENT plateau is advisory only; only the skip_to_sweep hint or budget exhaustion may exit KERNEL."""
    state = SimpleNamespace(
        phase="KERNEL_AGENT",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        kernel_integrate_attempts={
            f"k{i}": {"attempts": [{"decision": "REVERT", "ts": f"2026-05-19T18:0{i}:00"}]} for i in range(3)
        },
        rejected_kernel_ids=[],
        pending_escalate_hint="",
        stop_reason="",
    )
    assert exit_normal_kernel(state) is None


def test_exit_normal_kernel_after_gemm_does_not_exit():
    """The GEMM-completed shortcut is removed; GEMM completion alone never advances KERNEL_AGENT → SWEEP."""
    state = SimpleNamespace(
        phase="KERNEL_AGENT",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        kernel_integrate_attempts={},
        kernel_opt_task_attempts={},
        auto_kernel_opt_enabled=False,
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
        phase="FRAMEWORK_AGENT",
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


def test_exit_normal_optimize_skip_to_sweep_is_non_terminal():
    # skip_to_sweep exhausts the explore lever, non-terminal.
    out = exit_normal_optimize(_skip_to_sweep_state("FRAMEWORK_AGENT"))
    assert out is not None
    reason, evidence = out
    assert reason == "optimize_no_more_leverage"
    assert evidence.get("hint") == ESCALATE_HINT_SKIP_TO_SWEEP


def test_compute_next_phase_skip_to_sweep_from_explore_routes_to_kernel():
    # Exhausted explore leverage switches lever EXPLORE -> KERNEL, non-terminal.
    out = compute_next_phase(_skip_to_sweep_state("FRAMEWORK_AGENT"), kernel_enabled=True)
    assert out is not None
    target, reason, evidence = out
    assert target == PHASE_KERNEL_AGENT
    assert reason == "optimize_no_more_leverage"
    assert evidence.get("terminal") is not True


def test_compute_next_phase_skip_to_sweep_from_kernel_routes_to_sweep():
    out = compute_next_phase(_skip_to_sweep_state("KERNEL_AGENT"), kernel_enabled=True)
    assert out is not None
    target, reason, _ = out
    assert target == PHASE_SWEEP
    assert reason == "kernel_no_more_leverage"


def test_kernel_skip_to_sweep_waits_for_pending_keep():
    state = _skip_to_sweep_state("KERNEL_AGENT")
    state.has_keep_pending_integrate = True

    assert kernel_work_pending(state) is True
    assert exit_normal_kernel(state) is None
    assert compute_next_phase(state, kernel_enabled=True) is None


def test_kernel_skip_to_sweep_waits_for_partial_kernel_attempt():
    state = _skip_to_sweep_state("KERNEL_AGENT")
    state.kernel_opt_task_attempts = {
        "k009": {
            "last_decision": "PARTIAL",
            "last_status": "ok",
            "rejected_reason": "",
        },
    }

    assert kernel_work_pending(state) is True
    assert exit_normal_kernel(state) is None
    assert compute_next_phase(state, kernel_enabled=True) is None


def test_kernel_skip_to_sweep_waits_for_untried_hot_kernel():
    state = _skip_to_sweep_state("KERNEL_AGENT")
    state.untried_hot_reusable_kernels = lambda: ["k017"]

    assert kernel_work_pending(state) is True
    assert exit_normal_kernel(state) is None
    assert compute_next_phase(state, kernel_enabled=True) is None


def test_geak_terminal_skip_to_sweep_ignores_per_kernel_pending_work():
    state = _skip_to_sweep_state("KERNEL_AGENT")
    state.kernel_optimizer = "geak"
    state.geak_result = {"status": "no_gain"}
    state.untried_hot_reusable_kernels = lambda: ["k017"]

    assert kernel_work_pending(state) is False
    out = compute_next_phase(state, kernel_enabled=True)
    assert out is not None
    target, reason, _ = out
    assert target == PHASE_SWEEP
    assert reason == "kernel_no_more_leverage"


def test_collective_only_terminal_ignores_untried_regular_kernels():
    """A completed directed Collective lane must advance directly to SWEEP."""
    state = _skip_to_sweep_state("KERNEL_AGENT")
    state.collective_only_mode = True
    state.last_collective = {
        "status": "ok",
        "kept": True,
        "requires_e2e_validation": True,
        "integration_status": "complete",
    }
    state.untried_hot_reusable_kernels = lambda: ["k017"]

    assert kernel_work_pending(state) is False
    out = compute_next_phase(state, kernel_enabled=True)
    assert out is not None
    target, reason, _ = out
    assert target == PHASE_SWEEP
    assert reason == "kernel_no_more_leverage"


def test_collective_only_waits_for_pending_integration():
    """A crash checkpoint must keep KERNEL open until Collective E2E finishes."""
    state = _skip_to_sweep_state("KERNEL_AGENT")
    state.collective_only_mode = True
    state.last_collective = {
        "status": "ok",
        "kept": True,
        "requires_e2e_validation": True,
        "integration_status": "pending",
    }

    assert kernel_work_pending(state) is True
    assert compute_next_phase(state, kernel_enabled=True) is None


def test_kernel_skip_to_sweep_waits_for_retryable_failed_kernel():
    state = _skip_to_sweep_state("KERNEL_AGENT")
    state.kernel_opt_task_attempts = {
        "k018": {
            "attempts": 1,
            "failure_count": 1,
            "last_decision": "",
            "last_status": "failed",
            "rejected_reason": "",
        },
    }

    assert kernel_work_pending(state) is True
    assert exit_normal_kernel(state) is None
    assert compute_next_phase(state, kernel_enabled=True) is None


def test_kernel_skip_to_sweep_ignores_rejected_or_integrated_attempts():
    state = _skip_to_sweep_state("KERNEL_AGENT")
    state.rejected_kernel_ids = ["k001"]
    state.optimization_stack = [{"action": "integrate", "kernel_id": "k002"}]
    state.kernel_opt_task_attempts = {
        "k001": {
            "last_decision": "REVERT",
            "last_status": "ok",
            "rejected_reason": "revert_decision",
        },
        "k002": {
            "last_decision": "KEEP",
            "last_status": "ok",
            "rejected_reason": "",
        },
    }

    assert kernel_work_pending(state) is False
    out = compute_next_phase(state, kernel_enabled=True)
    assert out is not None
    target, reason, _ = out
    assert target == PHASE_SWEEP
    assert reason == "kernel_no_more_leverage"


def test_apply_escalate_budget_bump_lifts_phase_within_cap():
    out = apply_escalate_budget_bump(
        {"FRAMEWORK_AGENT": 0.60},
        phase="FRAMEWORK_AGENT",
    )
    assert out["FRAMEWORK_AGENT"] == pytest.approx(
        0.60 + ESCALATE_HINT_BUDGET_BUMP_DELTA,
    )


def test_apply_escalate_budget_bump_clamps_to_cap():
    out = apply_escalate_budget_bump(
        {"FRAMEWORK_AGENT": 0.95},
        phase="FRAMEWORK_AGENT",
    )
    assert out["FRAMEWORK_AGENT"] == ESCALATE_HINT_BUDGET_BUMP_CAP


def test_apply_escalate_budget_bump_ignores_unknown_phase():
    inp = {"FRAMEWORK_AGENT": 0.60}
    out = apply_escalate_budget_bump(inp, phase="NOT_A_PHASE")
    # No bump; returns a normalised copy with all known phases populated.
    assert out["FRAMEWORK_AGENT"] == 0.60


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
    assert s.stop_ts != ""


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
    assert s.stop_ts == ""


def test_a_later_stop_reason_does_not_move_the_stop_time(monkeypatch):
    """CLOSE stops the session on entry and ships the breakdown; a later write must not re-date it."""
    s = SharedState()
    monkeypatch.setattr(shared_state, "_now_iso", lambda: "2026-08-08T00:00:00.000000+00:00")
    s.set_stop_reason("time_exhausted")
    monkeypatch.setattr(shared_state, "_now_iso", lambda: "2026-08-08T02:00:00.000000+00:00")
    s.set_stop_reason("target_reached")
    assert s.stop_reason == "target_reached"
    assert s.stop_ts == "2026-08-08T00:00:00.000000+00:00"


def test_rewriting_the_same_stop_reason_does_not_move_the_stop_time(monkeypatch):
    """The Coordinator's ``finally`` re-asserts the reason CLOSE already wrote."""
    s = SharedState()
    monkeypatch.setattr(shared_state, "_now_iso", lambda: "2026-08-08T00:00:00.000000+00:00")
    s.set_stop_reason("time_exhausted")
    monkeypatch.setattr(shared_state, "_now_iso", lambda: "2026-08-08T00:04:00.000000+00:00")
    s.set_stop_reason(s.stop_reason)
    assert s.stop_ts == "2026-08-08T00:00:00.000000+00:00"


def test_saving_a_stopped_session_again_does_not_move_its_stop_time(tmp_path):
    s = SharedState()
    s.set_stop_reason("target_reached")
    pinned = s.stop_ts
    s.save(tmp_path)
    s.save(tmp_path)
    assert s.stop_ts == pinned
    assert SharedState.load_or_init(tmp_path).stop_ts == pinned


def test_stop_reason_vocab_has_v08_additions():
    for new in (
        "plateau_explore",
        "plateau_kernel",
        "no_kernel_skipped",
        "sweep_done",
        "robustness_escalated",
        "user_stop_requested",
        "recipe_kb_drain_failed",
        "recipe_kb_t0_failed",
        "recipe_kb_commit_failed",
        "prelude_baseline_failed",
        "prelude_policy_loop",
        "time_exhausted_during_prelude",
        "crash_threshold_exceeded",
    ):
        assert new in STOP_REASON_VOCAB
        assert is_valid_stop_reason(new)


def test_compute_next_phase_advances_on_plateau():
    """When both arms report dry, compute_next_phase routes to KERNEL_AGENT."""
    state = SimpleNamespace(
        phase="FRAMEWORK_AGENT",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        explore_search={"winners_history": [{"gain_pct": 0.1}]},
        specialist_rounds=[
            {"proposals_total": 0, "proposals_kept": 0} for _ in range(DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK)
        ],
        params_no_promote_streak=0,
        backends_search={},
        optimization_stack=[],
        pending_escalate_hint="",
        stop_reason="",
        plateau_overrides={},
        framework_agent_phase_done=True,
    )
    target, reason, _ = compute_next_phase(state, kernel_enabled=True)
    assert target == "KERNEL_AGENT"
    assert reason == "optimize_no_more_leverage"
    triggered, _ = compute_plateau_explore(state)
    assert triggered is True


def test_compute_next_phase_honors_explore_plateau_overrides():
    """CLI plateau overrides control the actual phase transition."""
    state = SimpleNamespace(
        phase="FRAMEWORK_AGENT",
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
        plateau_overrides={"explore_empty_streak": 3},
        framework_agent_phase_done=True,
    )

    target, reason, evidence = compute_next_phase(state, kernel_enabled=True)

    assert target == "KERNEL_AGENT"
    assert reason == "optimize_no_more_leverage"
    assert evidence["empty_streak_threshold"] == 3


def test_collect_phase_breakdown_buckets_by_phase():
    from hyperloom.inference_optimizer.breakdown.collectors import collect_attribution

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
            {"to_phase": "FRAMEWORK_AGENT", "ts_unix": 50.0, "reason": "prelude_done"},
            {"to_phase": "KERNEL_AGENT", "ts_unix": 200.0, "reason": "plateau_explore"},
            {"to_phase": "SWEEP", "ts_unix": 400.0, "reason": "plateau_kernel"},
        ],
        "explore_search": {
            "winners_history": [
                {"fingerprint": "fpfp", "provenance": "serving_specialist"},
            ],
        },
    }
    out = collect_attribution(state, [], [], [])
    pb = out["phase_breakdown"]
    # The KEEP landed inside FRAMEWORK_AGENT, but it moved a config lever, so
    # it belongs to the config bucket rather than the upstream-PR one.
    assert pb["explore"]["total_gain_pct"] == 5.0
    assert pb["explore"]["by_domain"]["serving_specialist"] == 5.0
    assert pb["framework"]["total_gain_pct"] == 0.0
    assert pb["kernel_agent"]["total_gain_pct"] == 7.5
    assert pb["kernel_agent"]["by_kernel_id"]["fmoe_fp8"] == 7.5
    assert pb["prelude"]["total_gain_pct"] == 0.0
    assert pb["sweep"]["total_gain_pct"] == 0.0


def test_collect_phase_breakdown_falls_back_to_action_family_when_history_empty():
    """Legacy resume — no phase_history → action-family fallback."""
    from hyperloom.inference_optimizer.breakdown.collectors import collect_attribution

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
    out = collect_attribution(state, [], [], warnings)
    pb = out["phase_breakdown"]
    assert pb["explore"]["total_gain_pct"] == 2.5
    # Default provenance bucket when winners_history doesn't supply one.
    assert pb["explore"]["by_domain"].get("default_grid", 0.0) == 2.5
    assert any("phase_history empty" in w for w in warnings)


def test_collect_phase_breakdown_skips_zero_or_negative_deltas():
    """Negative / None deltas don't enter the per-phase bucket."""
    from hyperloom.inference_optimizer.breakdown.collectors import collect_attribution

    state = {
        "optimization_stack": [{"action": "explore"}],
        "gain_per_stack_entry": [
            {"action": "explore", "delta_pct": -0.5, "ts_unix": 100.0},
            {"action": "explore", "delta_pct": None, "ts_unix": 110.0},
        ],
        "phase_history": [
            {"to_phase": "FRAMEWORK_AGENT", "ts_unix": 0.0, "reason": "prelude_done"},
        ],
    }
    out = collect_attribution(state, [], [], [])
    assert out["phase_breakdown"]["explore"]["total_gain_pct"] == 0.0
