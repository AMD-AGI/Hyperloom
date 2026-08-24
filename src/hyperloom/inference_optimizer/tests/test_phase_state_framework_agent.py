# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure-function tests for FRAMEWORK_AGENT phase routing and exit conditions."""

from __future__ import annotations

from typing import Any


from hyperloom.orchestrator.phases import machine_state as phase_state


class _State:
    """Minimal stand-in for SharedState."""

    def __init__(
        self,
        *,
        phase: str = phase_state.PHASE_FRAMEWORK_AGENT,
        baseline_tput: float | None = 1500.0,
        framework_agent_batches: list[dict[str, Any]] | None = None,
        framework_agent_phase_done: bool = False,
        framework_agent_phase_progress: list[dict[str, Any]] | None = None,
        remaining_minutes_value: float = 9999.0,
        phase_history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.phase = phase
        self.baseline_tput = baseline_tput
        self.framework_agent_batches = framework_agent_batches or []
        self.framework_agent_phase_done = framework_agent_phase_done
        self.framework_agent_phase_progress = framework_agent_phase_progress or []
        self._rem_min = remaining_minutes_value
        self.phase_history = phase_history or []
        self.optimization_stack: list[dict[str, Any]] = []
        self.plateau_overrides: dict[str, Any] = {}
        self.explore_search: dict[str, Any] = {}
        self.specialist_rounds: list[dict[str, Any]] = []
        self.stop_reason = ""
        self.escalate_history: list[dict[str, Any]] = []

    def remaining_minutes(self, *, now_unix: float | None = None) -> float:
        return self._rem_min


def test_framework_is_in_phase_names_between_prelude_and_explore():
    names = phase_state.PHASE_NAMES
    i = names.index("FRAMEWORK_AGENT")
    assert i > 0
    assert names[i - 1] == "PRELUDE"
    assert names[i + 1] == "EXPLORE"


def test_framework_exit_reasons_registered():
    reasons = {
        "framework_agent_phase_done",
        "framework_agent_plateau",
    }
    assert reasons <= phase_state.PHASE_EXIT_REASONS
    assert reasons <= phase_state.STOP_REASON_VOCAB


def test_framework_agent_force_exit_low_budget_is_retired_vocab():
    """``framework_agent_budget_cap`` replaced the session-remaining force-exit."""
    assert "framework_agent_force_exit_low_budget" not in phase_state.PHASE_EXIT_REASONS
    assert "framework_agent_force_exit_low_budget" not in phase_state.STOP_REASON_VOCAB
    assert "framework_agent_budget_cap" in phase_state.PHASE_EXIT_REASONS


def test_framework_agent_skipped_is_not_a_registered_reason():
    """``framework_agent_skipped`` is dead vocab — never emitted, must not be registered."""
    assert "framework_agent_skipped" not in phase_state.PHASE_EXIT_REASONS
    assert "framework_agent_skipped" not in phase_state.STOP_REASON_VOCAB


def test_framework_action_allowlist():
    allowed = phase_state.PHASE_ALLOWED_ACTIONS[phase_state.PHASE_FRAMEWORK_AGENT]
    assert "framework_agent" in allowed
    assert "integrate_patch" in allowed
    assert "roofline" in allowed
    assert "profile" in allowed
    assert "recover" in allowed
    assert "explore" not in allowed
    assert "specialist" in allowed


def test_phase_action_helpers_return_empty_for_unknown_phase():
    assert phase_state.allowed_actions_for("unknown") == ()
    assert phase_state.llm_proposable_actions_for("unknown") == ()
    assert not phase_state.is_action_llm_proposable_in_phase("explore", "unknown")


def test_long_run_and_gain_helpers_handle_edge_values():
    unbounded = _State()
    unbounded.max_minutes = 0
    assert phase_state.is_long_run(unbounded)

    invalid_gain = _State()
    invalid_gain.cumulative_gain_validated = "not-a-number"
    assert phase_state._cumulative_gain_validated(invalid_gain) == 0.0  # noqa: SLF001


def test_exit_normal_framework_agent_returns_none_when_nothing_to_do():
    state = _State()
    assert phase_state.exit_normal_framework_agent(state) is None


def test_exit_normal_framework_agent_ignores_session_remaining():
    """Low session-remaining alone must not evict a phase that has done nothing.

    FRAMEWORK now leaves only on its own charge-back cap, a plateau, or an
    exhausted candidate list — never on the session clock.
    """
    assert phase_state.exit_normal_framework_agent(_State(remaining_minutes_value=30.0)) is None
    below_streak = [{"candidate_id": f"c{i}", "status": "reverted", "kept": False} for i in range(3)]
    state = _State(framework_agent_phase_progress=below_streak, remaining_minutes_value=10.0)
    assert phase_state.exit_normal_framework_agent(state) is None


def test_exit_normal_framework_agent_exits_on_consecutive_reject_plateau():
    """A full streak of resolved-no-keep candidates (here 'reject') trips the
    plateau exit."""
    batches = [
        {
            "batch_id": "b1",
            "max_gain_pct_observed_in_batch": 0.2,
            "candidates": [{"id": "c1a"}],
        },
        {
            "batch_id": "b2",
            "max_gain_pct_observed_in_batch": 0.0,
            "candidates": [{"id": "c2a"}],
        },
        {
            "batch_id": "b3",
            "max_gain_pct_observed_in_batch": 0.5,
            "candidates": [{"id": "c3a"}],
        },
    ]
    progress = [
        {"batch_id": f"b{i}", "candidate_id": f"c{i}a", "status": "reject"}
        for i in range(phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK)
    ]
    state = _State(
        framework_agent_batches=batches,
        framework_agent_phase_progress=progress,
    )
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    assert out[0] == "framework_agent_plateau"
    assert out[1]["consecutive_no_keep"] == phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK


def test_compute_plateau_framework_agent_returns_signal():
    """compute_plateau_framework_agent remains as a pure advisory."""
    batches = [
        {
            "batch_id": f"b{i}",
            "max_gain_pct_observed_in_batch": 0.0,
            "candidates": [{"id": f"c{i}a"}],
        }
        for i in range(phase_state.DEFAULT_FRAMEWORK_PLATEAU_LOOKBACK)
    ]
    progress = [
        {"batch_id": f"b{i}", "candidate_id": f"c{i}a", "status": "reject"}
        for i in range(phase_state.DEFAULT_FRAMEWORK_PLATEAU_LOOKBACK)
    ]
    state = _State(
        framework_agent_batches=batches,
        framework_agent_phase_progress=progress,
    )
    triggered, ev = phase_state.compute_plateau_framework_agent(state)
    assert triggered is True
    assert ev["lookback"] == phase_state.DEFAULT_FRAMEWORK_PLATEAU_LOOKBACK


def test_framework_batch_plateau_ignores_prior_macro_cycle():
    batches = [
        {
            "batch_id": f"b{i}",
            "cycle": 0,
            "max_gain_pct_observed_in_batch": 0.0,
            "candidates": [{"id": f"c{i}"}],
        }
        for i in range(3)
    ]
    progress = [{"batch_id": f"b{i}", "candidate_id": f"c{i}", "status": "reverted", "cycle": 0} for i in range(3)]
    state = _State(framework_agent_batches=batches, framework_agent_phase_progress=progress)
    state.macro_cycle = 1
    triggered, evidence = phase_state.compute_plateau_framework_agent(state)
    assert triggered is False
    assert evidence["batch_max_gains"] == []


def test_framework_batch_plateau_counts_current_cycle_audit_skips():
    n = phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK
    batches = [
        {
            "batch_id": f"b{i}",
            "cycle": 1,
            "max_gain_pct_observed_in_batch": 0.0,
            "candidates": [{"id": f"c{i}"}],
        }
        for i in range(n)
    ]
    progress = [
        {
            "batch_id": f"b{i}",
            "candidate_id": f"c{i}",
            "status": "not_applicable",
            "cycle": 1,
        }
        for i in range(n)
    ]
    state = _State(framework_agent_batches=batches, framework_agent_phase_progress=progress)
    state.macro_cycle = 1
    triggered, _ = phase_state.compute_plateau_framework_agent(state)
    assert triggered is True


def test_exit_normal_framework_agent_exit_evidence_carries_pending_count():
    """Regression: an early exit surfaces how many candidates it left behind."""
    n = phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK
    batches = [
        {
            "batch_id": "b1",
            "max_gain_pct_observed_in_batch": 0.0,
            "candidates": [{"id": f"c{i}"} for i in range(n + 2)],
        },
    ]
    progress = [{"batch_id": "b1", "candidate_id": f"c{i}", "status": "reverted"} for i in range(n)]
    state = _State(
        framework_agent_batches=batches,
        framework_agent_phase_progress=progress,
    )
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    reason, ev = out
    assert reason == "framework_agent_plateau"
    assert ev["pending_candidate_count"] == 2


def test_exit_normal_framework_agent_phase_done_when_signalled():
    state = _State(framework_agent_phase_done=True)
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    reason, ev = out
    assert reason == "framework_agent_phase_done"
    assert ev["evidence"] == "no_more_candidates"


def test_exit_normal_framework_agent_plateau_at_threshold_consecutive_no_keep():
    """N consecutive benchmarked tests with no KEEP → framework_agent_plateau."""
    n = phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK
    progress = [{"candidate_id": f"c{i}", "status": "reverted", "kept": False} for i in range(n)]
    state = _State(framework_agent_phase_progress=progress)
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    reason, ev = out
    assert reason == "framework_agent_plateau"
    assert ev["consecutive_no_keep"] == n
    assert ev["threshold"] == n


def test_exit_normal_framework_agent_no_plateau_below_threshold():
    """Two reverts is below the streak threshold → no exit."""
    progress = [
        {"candidate_id": "c1", "status": "reverted", "kept": False},
        {"candidate_id": "c2", "status": "reverted", "kept": False},
    ]
    state = _State(framework_agent_phase_progress=progress)
    assert phase_state.exit_normal_framework_agent(state) is None


def test_local_explore_author_empty_uses_official_plateau_threshold():
    n = phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK
    progress = [{"candidate_id": f"local_explore:{i}", "status": "author_empty", "kept": False} for i in range(n - 1)]
    assert phase_state.exit_normal_framework_agent(_State(framework_agent_phase_progress=progress)) is None
    progress.append({"candidate_id": f"local_explore:{n - 1}", "status": "author_empty", "kept": False})
    out = phase_state.exit_normal_framework_agent(_State(framework_agent_phase_progress=progress))
    assert out is not None
    assert out[0] == "framework_agent_plateau"
    assert out[1]["consecutive_no_keep"] == n


def test_cycle_boundary_resets_local_explore_plateau_streak():
    n = phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK
    progress = [{"candidate_id": f"local_explore:{i}", "status": "author_empty", "kept": False} for i in range(n)]
    progress.append({"candidate_id": "", "status": "cycle_boundary", "kept": False})
    progress.append({"candidate_id": "local_explore:new", "status": "author_empty", "kept": False})
    assert phase_state.exit_normal_framework_agent(_State(framework_agent_phase_progress=progress)) is None


def test_exit_normal_framework_agent_keep_resets_no_keep_streak():
    """A KEEP breaks the streak; only reverts after it count."""
    progress = [
        {"candidate_id": "c1", "status": "reverted", "kept": False},
        {"candidate_id": "c2", "status": "kept", "kept": True},
        {"candidate_id": "c3", "status": "reverted", "kept": False},
        {"candidate_id": "c4", "status": "reverted", "kept": False},
    ]
    state = _State(framework_agent_phase_progress=progress)
    # Trailing run after the KEEP is only 2 reverts → no plateau.
    assert phase_state.exit_normal_framework_agent(state) is None


def test_framework_agent_consecutive_no_keep_handles_malformed_progress():
    state = _State(
        framework_agent_phase_progress=[
            {"candidate_id": "c0", "status": "kept", "kept": True},
            "not-a-row",
            {"candidate_id": "c1", "status": "reverted", "kept": False},
            object(),
            {"candidate_id": "c2", "status": "apply_failed", "kept": False},
        ],
    )

    assert phase_state._framework_agent_consecutive_no_keep(state) == 2  # noqa: SLF001


def test_exit_normal_framework_agent_plateau_counts_non_benchmarked_no_keep_rows():
    """not_applicable / apply_failed / authored_empty rows count toward the
    no-keep streak (they are resolved candidates that did not KEEP), so a
    batch of dead candidates still trips the plateau gate.
    """
    progress = [
        {"candidate_id": "c1", "status": "not_applicable", "kept": False},
        {"candidate_id": "c2", "status": "apply_failed", "kept": False},
        {"candidate_id": "c3", "status": "authored_empty", "kept": False},
        {"candidate_id": "c4", "status": "no_patches", "kept": False},
        {"candidate_id": "c5", "status": "already_present", "kept": False},
    ]
    state = _State(framework_agent_phase_progress=progress)
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    assert out[0] == "framework_agent_plateau"
    assert out[1]["consecutive_no_keep"] == phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK


def test_exit_normal_framework_agent_plateau_mixed_terminal_no_keep_rows():
    """A mix of reverted + non-benchmarked terminal rows all count; a KEEP
    still breaks the streak so only the trailing run is counted."""
    progress = [
        {"candidate_id": "c0", "status": "kept", "kept": True},
        {"candidate_id": "c1", "status": "reverted", "kept": False},
        {"candidate_id": "c2", "status": "not_applicable", "kept": False},
        {"candidate_id": "c3", "status": "apply_failed", "kept": False},
        {"candidate_id": "c4", "status": "authored_empty", "kept": False},
        {"candidate_id": "c5", "status": "reverted", "kept": False},
    ]
    state = _State(framework_agent_phase_progress=progress)
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    assert out[0] == "framework_agent_plateau"
    assert out[1]["consecutive_no_keep"] == phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK


def test_exit_normal_framework_agent_plateau_beats_phase_done():
    """Priority order: plateau > phase_done."""
    progress = [
        {"candidate_id": f"c{i}", "status": "reverted", "kept": False}
        for i in range(phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK)
    ]
    state = _State(
        framework_agent_phase_progress=progress,
        framework_agent_phase_done=True,
    )
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    assert out[0] == "framework_agent_plateau"


def test_exit_normal_framework_agent_plateau_routes_to_explore():
    """A plateau exit routes FRAMEWORK_AGENT → EXPLORE via compute_next_phase."""
    progress = [
        {"candidate_id": f"c{i}", "status": "reverted", "kept": False}
        for i in range(phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK)
    ]
    state = _State(framework_agent_phase_progress=progress)
    out = phase_state.compute_next_phase(state, framework_agent_phase_enabled=True)
    assert out is not None
    next_phase, reason, _ev = out
    assert next_phase == phase_state.PHASE_EXPLORE
    assert reason == "framework_agent_plateau"


def test_exit_normal_framework_agent_plateau_uses_default_threshold():
    """The framework plateau threshold is fixed at the default."""
    progress = [
        {"candidate_id": f"c{i}", "status": "reverted", "kept": False}
        for i in range(phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK)
    ]
    state = _State(framework_agent_phase_progress=progress)
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    assert out[0] == "framework_agent_plateau"
    assert out[1]["threshold"] == phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK


def test_framework_agent_plateau_streak_threshold_is_default():
    assert (
        phase_state._framework_agent_plateau_streak_threshold()  # noqa: SLF001
        == phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK
    )


def test_exit_normal_framework_agent_phase_done_wins_when_low_on_session_time():
    """A nearly spent session no longer masks the honest ``phase_done`` reason."""
    batches = [
        {"max_gain_pct_observed_in_batch": 0.1},
        {"max_gain_pct_observed_in_batch": 0.1},
        {"max_gain_pct_observed_in_batch": 0.1},
    ]
    state = _State(
        framework_agent_batches=batches,
        framework_agent_phase_done=True,
        remaining_minutes_value=10.0,
    )
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    assert out[0] == "framework_agent_phase_done"


def test_compute_next_phase_prelude_to_framework_when_enabled():
    state = _State(phase=phase_state.PHASE_PRELUDE, baseline_tput=1500.0)
    out = phase_state.compute_next_phase(state, framework_agent_phase_enabled=True)
    assert out is not None
    next_phase, reason, _ev = out
    assert next_phase == phase_state.PHASE_FRAMEWORK_AGENT
    assert reason == "prelude_done"


def test_compute_next_phase_closing_phase_goes_to_close_not_framework():
    """A spent wall-clock must not start FRAMEWORK; that allowlist drops ``report``."""
    state = _State(phase=phase_state.PHASE_PRELUDE, baseline_tput=1500.0)
    state.closing_phase = True
    out = phase_state.compute_next_phase(state, framework_agent_phase_enabled=True)
    assert out is not None
    next_phase, reason, ev = out
    assert next_phase == phase_state.PHASE_CLOSE
    assert reason == "time_exhausted"
    assert ev.get("terminal") is True
    assert ev.get("reason_origin") == "closing_phase"


def test_compute_next_phase_prelude_to_explore_when_disabled_keeps_prelude_done_reason():
    """``framework_agent_phase_enabled=False`` preserves the historical ``prelude_done`` reason."""
    state = _State(phase=phase_state.PHASE_PRELUDE, baseline_tput=1500.0)
    out = phase_state.compute_next_phase(state, framework_agent_phase_enabled=False)
    assert out is not None
    next_phase, reason, _ev = out
    assert next_phase == phase_state.PHASE_EXPLORE
    assert reason == "prelude_done"


def test_compute_next_phase_framework_does_not_advance_on_plateau():
    """Plateau no longer drives FRAMEWORK exit."""
    state = _State(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        framework_agent_batches=[
            {"max_gain_pct_observed_in_batch": 0.0},
            {"max_gain_pct_observed_in_batch": 0.5},
            {"max_gain_pct_observed_in_batch": 0.7},
        ],
    )
    assert phase_state.compute_next_phase(state, framework_agent_phase_enabled=True) is None


def test_compute_next_phase_framework_holds_on_a_nearly_spent_session():
    """A reloop into FRAMEWORK late in the run must not bounce straight out.

    FRAMEWORK is the preferred macro-cycle reloop target, and every reloop
    happens once most of the session is gone.
    """
    state = _State(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        remaining_minutes_value=30.0,
    )
    assert phase_state.compute_next_phase(state, framework_agent_phase_enabled=True) is None


def test_compute_next_phase_framework_stays_when_no_signal():
    state = _State(phase=phase_state.PHASE_FRAMEWORK_AGENT)
    out = phase_state.compute_next_phase(
        state,
        framework_agent_phase_enabled=True,
    )
    assert out is None


def test_compute_next_phase_prelude_skips_explore_to_kernel():
    state = _State(phase=phase_state.PHASE_PRELUDE, baseline_tput=1500.0)
    out = phase_state.compute_next_phase(
        state,
        framework_agent_phase_enabled=False,
        explore_enabled=False,
        kernel_enabled=True,
    )
    assert out is not None
    next_phase, reason, ev = out
    assert next_phase == phase_state.PHASE_KERNEL_AGENT
    assert reason == "prelude_done"
    assert ev.get("explore_skipped") is True


def test_compute_next_phase_prelude_skips_to_sweep_when_no_explore_no_kernel():
    state = _State(phase=phase_state.PHASE_PRELUDE, baseline_tput=1500.0)
    out = phase_state.compute_next_phase(
        state,
        framework_agent_phase_enabled=False,
        explore_enabled=False,
        kernel_enabled=False,
    )
    assert out is not None
    next_phase, reason, ev = out
    assert next_phase == phase_state.PHASE_SWEEP
    assert reason == "prelude_done"
    assert ev.get("explore_skipped") is True


def test_compute_next_phase_framework_skips_explore_to_kernel():
    """With explore disabled, FRAMEWORK_AGENT phase_done routes straight to KERNEL."""
    state = _State(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        framework_agent_phase_done=True,
    )
    out = phase_state.compute_next_phase(
        state,
        framework_agent_phase_enabled=True,
        explore_enabled=False,
        kernel_enabled=True,
    )
    assert out is not None
    next_phase, reason, ev = out
    assert next_phase == phase_state.PHASE_KERNEL_AGENT
    assert reason == "framework_agent_phase_done"
    assert ev.get("explore_skipped") is True


def test_compute_next_phase_explore_enabled_default_routes_to_explore():
    state = _State(phase=phase_state.PHASE_PRELUDE, baseline_tput=1500.0)
    out = phase_state.compute_next_phase(state, framework_agent_phase_enabled=False)
    assert out is not None
    next_phase, _reason, ev = out
    assert next_phase == phase_state.PHASE_EXPLORE
    assert "explore_skipped" not in ev
