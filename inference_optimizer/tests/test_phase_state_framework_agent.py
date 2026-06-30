# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pure-function tests for FRAMEWORK_AGENT phase routing and exit conditions."""

from __future__ import annotations

from typing import Any


from inference_optimizer.orchestrator import phase_state


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


# PHASE_NAMES + exit reason vocab presence
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
        "framework_agent_force_exit_low_budget",
    }
    assert reasons <= phase_state.PHASE_EXIT_REASONS
    assert reasons <= phase_state.STOP_REASON_VOCAB


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
    assert "specialist" not in allowed


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


# exit_normal_framework_agent
def test_exit_normal_framework_agent_returns_none_when_nothing_to_do():
    state = _State()
    assert phase_state.exit_normal_framework_agent(state) is None


def test_exit_normal_framework_agent_force_exit_when_remaining_below_ratio():
    # remaining 30min < 0.6 × 2h × 60 = 72min → fires.
    state = _State(remaining_minutes_value=30.0)
    out = phase_state.exit_normal_framework_agent(state, max_hours=2.0)
    assert out is not None
    reason, ev = out
    assert reason == "framework_agent_force_exit_low_budget"
    assert ev["evidence"] == "force_exit"
    assert ev["remaining_minutes"] == 30.0


def test_exit_normal_framework_agent_no_force_exit_when_remaining_above_ratio():
    # remaining 80min > 0.6 × 2h × 60 = 72min → no force exit.
    state = _State(remaining_minutes_value=80.0)
    assert phase_state.exit_normal_framework_agent(state, max_hours=2.0) is None


def test_exit_normal_framework_agent_accepts_positional_remaining_minutes():
    class PositionalRemainingState(_State):
        def remaining_minutes(self) -> float:  # type: ignore[override]
            return 30.0

    out = phase_state.exit_normal_framework_agent(PositionalRemainingState(), max_hours=2.0)

    assert out is not None
    assert out[0] == "framework_agent_force_exit_low_budget"
    assert out[1]["remaining_minutes"] == 30.0


def test_exit_normal_framework_agent_exits_on_consecutive_reject_plateau():
    """3 consecutive resolved-no-keep candidates (here 'reject') trip the
    plateau exit. (Updated from the pre-plateau-feature behaviour where the
    streak was advisory-only and reject rows were ignored.)"""
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
        {"batch_id": "b1", "candidate_id": "c1a", "status": "reject"},
        {"batch_id": "b2", "candidate_id": "c2a", "status": "reject"},
        {"batch_id": "b3", "candidate_id": "c3a", "status": "reject"},
    ]
    state = _State(
        framework_agent_batches=batches,
        framework_agent_phase_progress=progress,
    )
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    assert out[0] == "framework_agent_plateau"
    assert out[1]["consecutive_no_keep"] == 3


def test_compute_plateau_framework_agent_returns_signal():
    """compute_plateau_framework_agent remains as a pure advisory."""
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
        {"batch_id": "b1", "candidate_id": "c1a", "status": "reject"},
        {"batch_id": "b2", "candidate_id": "c2a", "status": "reject"},
        {"batch_id": "b3", "candidate_id": "c3a", "status": "reject"},
    ]
    state = _State(
        framework_agent_batches=batches,
        framework_agent_phase_progress=progress,
    )
    triggered, ev = phase_state.compute_plateau_framework_agent(state)
    assert triggered is True
    assert ev["lookback"] == 3


def test_exit_normal_framework_agent_force_exit_evidence_carries_pending_count():
    """Regression for P1.a — force-exit evidence surfaces ``pending_candidate_count``."""
    batches = [
        {
            "batch_id": "b1",
            "max_gain_pct_observed_in_batch": 0.0,
            "candidates": [{"id": "c1a"}, {"id": "c1b"}, {"id": "c1c"}],
        },
    ]
    progress = [
        {"batch_id": "b1", "candidate_id": "c1a", "status": "reject"},
    ]
    state = _State(
        framework_agent_batches=batches,
        framework_agent_phase_progress=progress,
        remaining_minutes_value=10.0,
    )
    out = phase_state.exit_normal_framework_agent(state, max_hours=2.0)
    assert out is not None
    reason, ev = out
    assert reason == "framework_agent_force_exit_low_budget"
    assert ev["pending_candidate_count"] == 2


def test_exit_normal_framework_agent_phase_done_when_signalled():
    state = _State(framework_agent_phase_done=True)
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    reason, ev = out
    assert reason == "framework_agent_phase_done"
    assert ev["evidence"] == "no_more_candidates"


def test_exit_normal_framework_agent_plateau_after_three_consecutive_no_keep():
    """3 consecutive benchmarked tests with no KEEP → framework_agent_plateau."""
    progress = [
        {"candidate_id": "c1", "status": "reverted", "kept": False},
        {"candidate_id": "c2", "status": "reverted", "kept": False},
        {"candidate_id": "c3", "status": "reverted", "kept": False},
    ]
    state = _State(framework_agent_phase_progress=progress)
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    reason, ev = out
    assert reason == "framework_agent_plateau"
    assert ev["consecutive_no_keep"] == 3
    assert ev["threshold"] == 3


def test_exit_normal_framework_agent_no_plateau_below_threshold():
    """Two reverts is below the streak threshold → no exit."""
    progress = [
        {"candidate_id": "c1", "status": "reverted", "kept": False},
        {"candidate_id": "c2", "status": "reverted", "kept": False},
    ]
    state = _State(framework_agent_phase_progress=progress)
    assert phase_state.exit_normal_framework_agent(state) is None


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
    no-keep streak (they are resolved candidates that did not KEEP).

    In a wheel-based framework env the direct_apply path cannot run and
    authoring overwhelmingly returns not_applicable / authored_empty, so a batch
    of dead candidates must still trip the plateau gate — otherwise FRAMEWORK
    (which has no wall-clock budget cap) grinds for hours without leverage.
    """
    progress = [
        {"candidate_id": "c1", "status": "not_applicable", "kept": False},
        {"candidate_id": "c2", "status": "apply_failed", "kept": False},
        {"candidate_id": "c3", "status": "authored_empty", "kept": False},
    ]
    state = _State(framework_agent_phase_progress=progress)
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    assert out[0] == "framework_agent_plateau"
    assert out[1]["consecutive_no_keep"] == 3


def test_exit_normal_framework_agent_plateau_mixed_terminal_no_keep_rows():
    """A mix of reverted + non-benchmarked terminal rows all count; a KEEP
    still breaks the streak so only the trailing run is counted."""
    progress = [
        {"candidate_id": "c0", "status": "kept", "kept": True},
        {"candidate_id": "c1", "status": "reverted", "kept": False},
        {"candidate_id": "c2", "status": "not_applicable", "kept": False},
        {"candidate_id": "c3", "status": "apply_failed", "kept": False},
    ]
    state = _State(framework_agent_phase_progress=progress)
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    assert out[0] == "framework_agent_plateau"
    assert out[1]["consecutive_no_keep"] == 3


def test_exit_normal_framework_agent_force_exit_beats_plateau():
    """Priority order: force-exit > plateau."""
    progress = [
        {"candidate_id": "c1", "status": "reverted", "kept": False},
        {"candidate_id": "c2", "status": "reverted", "kept": False},
        {"candidate_id": "c3", "status": "reverted", "kept": False},
    ]
    state = _State(
        framework_agent_phase_progress=progress,
        remaining_minutes_value=10.0,
    )
    out = phase_state.exit_normal_framework_agent(state, max_hours=2.0)
    assert out is not None
    assert out[0] == "framework_agent_force_exit_low_budget"


def test_exit_normal_framework_agent_plateau_beats_phase_done():
    """Priority order: plateau > phase_done."""
    progress = [
        {"candidate_id": "c1", "status": "reverted", "kept": False},
        {"candidate_id": "c2", "status": "reverted", "kept": False},
        {"candidate_id": "c3", "status": "reverted", "kept": False},
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
        {"candidate_id": "c1", "status": "reverted", "kept": False},
        {"candidate_id": "c2", "status": "reverted", "kept": False},
        {"candidate_id": "c3", "status": "reverted", "kept": False},
    ]
    state = _State(framework_agent_phase_progress=progress)
    out = phase_state.compute_next_phase(state, framework_agent_phase_enabled=True)
    assert out is not None
    next_phase, reason, _ev = out
    assert next_phase == phase_state.PHASE_EXPLORE
    assert reason == "framework_agent_plateau"


def test_exit_normal_framework_agent_plateau_streak_env_override(monkeypatch):
    """INFERENCE_OPTIMIZER_FRAMEWORK_PLATEAU_STREAK overrides the threshold."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FRAMEWORK_PLATEAU_STREAK", "2")
    progress = [
        {"candidate_id": "c1", "status": "reverted", "kept": False},
        {"candidate_id": "c2", "status": "reverted", "kept": False},
    ]
    state = _State(framework_agent_phase_progress=progress)
    out = phase_state.exit_normal_framework_agent(state)
    assert out is not None
    assert out[0] == "framework_agent_plateau"
    assert out[1]["threshold"] == 2


def test_framework_agent_plateau_streak_env_invalid_uses_default(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FRAMEWORK_PLATEAU_STREAK", "invalid")
    assert phase_state._framework_agent_plateau_streak_threshold() == 3  # noqa: SLF001

    monkeypatch.setenv("INFERENCE_OPTIMIZER_FRAMEWORK_PLATEAU_STREAK", "0")
    assert phase_state._framework_agent_plateau_streak_threshold() == 3  # noqa: SLF001


def test_exit_normal_framework_agent_force_exit_beats_phase_done():
    """Priority order: force-exit > phase_done."""
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
    out = phase_state.exit_normal_framework_agent(state, max_hours=2.0)
    assert out is not None
    assert out[0] == "framework_agent_force_exit_low_budget"


# compute_next_phase routing (with explicit framework_agent_phase_enabled)
def test_compute_next_phase_prelude_to_framework_when_enabled():
    state = _State(phase=phase_state.PHASE_PRELUDE, baseline_tput=1500.0)
    out = phase_state.compute_next_phase(state, framework_agent_phase_enabled=True)
    assert out is not None
    next_phase, reason, _ev = out
    assert next_phase == phase_state.PHASE_FRAMEWORK_AGENT
    assert reason == "prelude_done"


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


def test_compute_next_phase_framework_agent_force_exit_passes_max_hours_through():
    state = _State(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        remaining_minutes_value=30.0,
    )
    out = phase_state.compute_next_phase(
        state,
        framework_agent_phase_enabled=True,
        max_hours=2.0,
    )
    assert out is not None
    next_phase, reason, ev = out
    assert next_phase == phase_state.PHASE_EXPLORE
    assert reason == "framework_agent_force_exit_low_budget"
    assert ev["max_hours"] == 2.0


def test_compute_next_phase_framework_stays_when_no_signal():
    state = _State(phase=phase_state.PHASE_FRAMEWORK_AGENT)
    out = phase_state.compute_next_phase(
        state,
        framework_agent_phase_enabled=True,
        max_hours=10.0,
    )
    assert out is None


# compute_next_phase routing with explore_enabled=False (--no-explore)
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
