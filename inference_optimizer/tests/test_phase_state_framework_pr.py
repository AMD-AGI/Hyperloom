# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pure-function tests for the FRAMEWORK_PR phase routing and exit
conditions (Stage 2a coverage).
"""

from __future__ import annotations

from typing import Any


from inference_optimizer.orchestrator import phase_state


class _State:
    """Minimal stand-in for SharedState. Only carries the fields
    ``exit_normal_framework_pr`` + ``compute_next_phase`` read."""

    def __init__(
        self,
        *,
        phase: str = phase_state.PHASE_FRAMEWORK_PR,
        baseline_tput: float | None = 1500.0,
        framework_pr_batches: list[dict[str, Any]] | None = None,
        framework_pr_phase_done: bool = False,
        framework_pr_phase_progress: list[dict[str, Any]] | None = None,
        remaining_minutes_value: float = 9999.0,
        phase_history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.phase = phase
        self.baseline_tput = baseline_tput
        self.framework_pr_batches = framework_pr_batches or []
        self.framework_pr_phase_done = framework_pr_phase_done
        self.framework_pr_phase_progress = framework_pr_phase_progress or []
        self._rem_min = remaining_minutes_value
        self.phase_history = phase_history or []
        # Fields read by other branches we don't exercise here.
        self.optimization_stack: list[dict[str, Any]] = []
        self.plateau_overrides: dict[str, Any] = {}
        self.explore_search: dict[str, Any] = {}
        self.specialist_rounds: list[dict[str, Any]] = []
        self.stop_reason = ""
        self.escalate_history: list[dict[str, Any]] = []

    def remaining_minutes(self, *, now_unix: float | None = None) -> float:
        return self._rem_min


# ---------------------------------------------------------------------------
# PHASE_NAMES + exit reason vocab presence
# ---------------------------------------------------------------------------
def test_framework_pr_is_in_phase_names_between_prelude_and_explore():
    names = phase_state.PHASE_NAMES
    i = names.index("FRAMEWORK_PR")
    assert i > 0
    assert names[i - 1] == "PRELUDE"
    assert names[i + 1] == "EXPLORE"


def test_framework_pr_exit_reasons_registered():
    reasons = {
        "framework_pr_phase_done",
        "framework_pr_plateau",
        "framework_pr_force_exit_low_budget",
    }
    assert reasons <= phase_state.PHASE_EXIT_REASONS
    assert reasons <= phase_state.STOP_REASON_VOCAB


def test_framework_pr_skipped_is_not_a_registered_reason():
    """``framework_pr_skipped`` was registered but never emitted by any
    code path — when --no-framework is set, PRELUDE → EXPLORE reuses
    the ``prelude_done`` reason for back-compat with pre-FRAMEWORK_PR
    sessions. Remove the dead vocab entry."""
    assert "framework_pr_skipped" not in phase_state.PHASE_EXIT_REASONS
    assert "framework_pr_skipped" not in phase_state.STOP_REASON_VOCAB


def test_framework_pr_action_allowlist():
    allowed = phase_state.PHASE_ALLOWED_ACTIONS[phase_state.PHASE_FRAMEWORK_PR]
    assert "framework_pr" in allowed
    assert "integrate_patch" in allowed
    assert "roofline" in allowed
    assert "profile" in allowed
    assert "recover" in allowed
    # Defensive: must not leak EXPLORE-only actions.
    assert "explore" not in allowed
    assert "specialist" not in allowed


# ---------------------------------------------------------------------------
# exit_normal_framework_pr
# ---------------------------------------------------------------------------
def test_exit_normal_framework_pr_returns_none_when_nothing_to_do():
    state = _State()
    assert phase_state.exit_normal_framework_pr(state) is None


def test_exit_normal_framework_pr_force_exit_when_remaining_below_ratio():
    # remaining 30min < 0.6 × 2h × 60 = 72min → fires.
    state = _State(remaining_minutes_value=30.0)
    out = phase_state.exit_normal_framework_pr(state, max_hours=2.0)
    assert out is not None
    reason, ev = out
    assert reason == "framework_pr_force_exit_low_budget"
    assert ev["evidence"] == "force_exit"
    assert ev["remaining_minutes"] == 30.0


def test_exit_normal_framework_pr_no_force_exit_when_remaining_above_ratio():
    # remaining 80min > 0.6 × 2h × 60 = 72min → no force exit.
    state = _State(remaining_minutes_value=80.0)
    assert phase_state.exit_normal_framework_pr(state, max_hours=2.0) is None


def test_exit_normal_framework_pr_plateau_after_three_flat_batches():
    batches = [
        {"batch_id": "b1", "max_gain_pct_observed_in_batch": 0.4},
        {"batch_id": "b2", "max_gain_pct_observed_in_batch": 0.0},
        {"batch_id": "b3", "max_gain_pct_observed_in_batch": 0.7},
    ]
    state = _State(framework_pr_batches=batches)
    out = phase_state.exit_normal_framework_pr(state)
    assert out is not None
    reason, ev = out
    assert reason == "framework_pr_plateau"
    assert ev["lookback"] == 3
    assert ev["keep_gain_pct_threshold"] == 1.0


def test_exit_normal_framework_pr_no_plateau_when_recent_batch_above_threshold():
    batches = [
        {"max_gain_pct_observed_in_batch": 0.4},
        {"max_gain_pct_observed_in_batch": 0.0},
        {"max_gain_pct_observed_in_batch": 3.7},  # recent recovery
    ]
    state = _State(framework_pr_batches=batches)
    assert phase_state.exit_normal_framework_pr(state) is None


def test_exit_normal_framework_pr_no_plateau_when_latest_batch_undrained():
    """Regression for P1.a — a freshly-discovered batch whose first
    candidate has not finished must NOT count toward plateau lookback,
    even though its ``max_gain_pct_observed_in_batch`` defaults to 0.0
    on creation. Without the drain check, three such 0.0 entries would
    trip plateau the moment the pump enqueues the first candidate of
    the third batch."""
    batches = [
        {
            "batch_id": "b1",
            "max_gain_pct_observed_in_batch": 0.0,
            "candidates": [{"id": "c1a"}, {"id": "c1b"}],
        },
        {
            "batch_id": "b2",
            "max_gain_pct_observed_in_batch": 0.0,
            "candidates": [{"id": "c2a"}, {"id": "c2b"}],
        },
        {
            "batch_id": "b3",
            "max_gain_pct_observed_in_batch": 0.0,
            "candidates": [{"id": "c3a"}, {"id": "c3b"}, {"id": "c3c"}],
        },
    ]
    # Only b1 and b2 are fully drained; b3 has 1 of 3 processed.
    progress = [
        {"batch_id": "b1", "candidate_id": "c1a", "status": "reject"},
        {"batch_id": "b1", "candidate_id": "c1b", "status": "reject"},
        {"batch_id": "b2", "candidate_id": "c2a", "status": "reject"},
        {"batch_id": "b2", "candidate_id": "c2b", "status": "reject"},
        {"batch_id": "b3", "candidate_id": "c3a", "status": "reject"},
    ]
    state = _State(
        framework_pr_batches=batches,
        framework_pr_phase_progress=progress,
    )
    assert phase_state.exit_normal_framework_pr(state) is None


def test_exit_normal_framework_pr_plateau_fires_when_all_three_batches_complete():
    """Regression for P1.a — once every batch in the lookback window is
    fully drained AND below the keep-gain threshold, plateau still fires
    as it did before the drain check was added."""
    batches = [
        {
            "batch_id": "b1",
            "max_gain_pct_observed_in_batch": 0.2,
            "candidates": [{"id": "c1a"}, {"id": "c1b"}],
        },
        {
            "batch_id": "b2",
            "max_gain_pct_observed_in_batch": 0.5,
            "candidates": [{"id": "c2a"}, {"id": "c2b"}],
        },
        {
            "batch_id": "b3",
            "max_gain_pct_observed_in_batch": 0.7,
            "candidates": [{"id": "c3a"}, {"id": "c3b"}],
        },
    ]
    progress = [
        {"batch_id": "b1", "candidate_id": "c1a", "status": "reject"},
        {"batch_id": "b1", "candidate_id": "c1b", "status": "reject"},
        {"batch_id": "b2", "candidate_id": "c2a", "status": "reject"},
        {"batch_id": "b2", "candidate_id": "c2b", "status": "reject"},
        {"batch_id": "b3", "candidate_id": "c3a", "status": "reject"},
        {"batch_id": "b3", "candidate_id": "c3b", "status": "reject"},
    ]
    state = _State(
        framework_pr_batches=batches,
        framework_pr_phase_progress=progress,
    )
    out = phase_state.exit_normal_framework_pr(state)
    assert out is not None
    reason, ev = out
    assert reason == "framework_pr_plateau"
    assert ev["lookback"] == 3


def test_exit_normal_framework_pr_force_exit_evidence_carries_pending_count():
    """Regression for P1.a — force-exit evidence must surface
    ``pending_candidate_count`` so operators can see how many candidates
    were skipped by the wall-clock guard."""
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
        framework_pr_batches=batches,
        framework_pr_phase_progress=progress,
        remaining_minutes_value=10.0,
    )
    out = phase_state.exit_normal_framework_pr(state, max_hours=2.0)
    assert out is not None
    reason, ev = out
    assert reason == "framework_pr_force_exit_low_budget"
    assert ev["pending_candidate_count"] == 2


def test_exit_normal_framework_pr_phase_done_when_signalled():
    state = _State(framework_pr_phase_done=True)
    out = phase_state.exit_normal_framework_pr(state)
    assert out is not None
    reason, ev = out
    assert reason == "framework_pr_phase_done"
    assert ev["evidence"] == "no_more_candidates"


def test_exit_normal_framework_pr_force_exit_beats_plateau():
    """Priority order: force-exit > plateau > phase_done."""
    batches = [
        {"max_gain_pct_observed_in_batch": 0.1},
        {"max_gain_pct_observed_in_batch": 0.1},
        {"max_gain_pct_observed_in_batch": 0.1},
    ]
    state = _State(
        framework_pr_batches=batches,
        framework_pr_phase_done=True,
        remaining_minutes_value=10.0,
    )
    out = phase_state.exit_normal_framework_pr(state, max_hours=2.0)
    assert out is not None
    assert out[0] == "framework_pr_force_exit_low_budget"


# ---------------------------------------------------------------------------
# compute_next_phase routing (with explicit framework_phase_enabled)
# ---------------------------------------------------------------------------
def test_compute_next_phase_prelude_to_framework_pr_when_enabled():
    state = _State(phase=phase_state.PHASE_PRELUDE, baseline_tput=1500.0)
    out = phase_state.compute_next_phase(state, framework_phase_enabled=True)
    assert out is not None
    next_phase, reason, _ev = out
    assert next_phase == phase_state.PHASE_FRAMEWORK_PR
    assert reason == "prelude_done"


def test_compute_next_phase_prelude_to_explore_when_disabled_keeps_prelude_done_reason():
    """When ``framework_phase_enabled=False`` the routing must preserve
    the historical ``prelude_done`` reason so phase_history stays
    compatible with pre-FRAMEWORK_PR sessions. The FRAMEWORK_PR phase
    has no dedicated "skipped" reason — see
    :func:`test_framework_pr_skipped_is_not_a_registered_reason`."""
    state = _State(phase=phase_state.PHASE_PRELUDE, baseline_tput=1500.0)
    out = phase_state.compute_next_phase(state, framework_phase_enabled=False)
    assert out is not None
    next_phase, reason, _ev = out
    assert next_phase == phase_state.PHASE_EXPLORE
    assert reason == "prelude_done"


def test_compute_next_phase_framework_pr_to_explore_on_plateau():
    state = _State(
        phase=phase_state.PHASE_FRAMEWORK_PR,
        framework_pr_batches=[
            {"max_gain_pct_observed_in_batch": 0.0},
            {"max_gain_pct_observed_in_batch": 0.5},
            {"max_gain_pct_observed_in_batch": 0.7},
        ],
    )
    out = phase_state.compute_next_phase(state, framework_phase_enabled=True)
    assert out is not None
    next_phase, reason, _ = out
    assert next_phase == phase_state.PHASE_EXPLORE
    assert reason == "framework_pr_plateau"


def test_compute_next_phase_framework_pr_force_exit_passes_max_hours_through():
    state = _State(
        phase=phase_state.PHASE_FRAMEWORK_PR, remaining_minutes_value=30.0,
    )
    out = phase_state.compute_next_phase(
        state, framework_phase_enabled=True, max_hours=2.0,
    )
    assert out is not None
    next_phase, reason, ev = out
    assert next_phase == phase_state.PHASE_EXPLORE
    assert reason == "framework_pr_force_exit_low_budget"
    assert ev["max_hours"] == 2.0


def test_compute_next_phase_framework_pr_stays_when_no_signal():
    state = _State(phase=phase_state.PHASE_FRAMEWORK_PR)
    out = phase_state.compute_next_phase(
        state, framework_phase_enabled=True, max_hours=10.0,
    )
    assert out is None


# ---------------------------------------------------------------------------
# compute_next_phase routing with explore_enabled=False (--no-explore)
# ---------------------------------------------------------------------------
def test_compute_next_phase_prelude_skips_explore_to_kernel():
    state = _State(phase=phase_state.PHASE_PRELUDE, baseline_tput=1500.0)
    out = phase_state.compute_next_phase(
        state,
        framework_phase_enabled=False,
        explore_enabled=False,
        kernel_enabled=True,
    )
    assert out is not None
    next_phase, reason, ev = out
    assert next_phase == phase_state.PHASE_KERNEL
    assert reason == "prelude_done"
    assert ev.get("explore_skipped") is True


def test_compute_next_phase_prelude_skips_to_sweep_when_no_explore_no_kernel():
    state = _State(phase=phase_state.PHASE_PRELUDE, baseline_tput=1500.0)
    out = phase_state.compute_next_phase(
        state,
        framework_phase_enabled=False,
        explore_enabled=False,
        kernel_enabled=False,
    )
    assert out is not None
    next_phase, reason, ev = out
    assert next_phase == phase_state.PHASE_SWEEP
    assert reason == "prelude_done"
    assert ev.get("explore_skipped") is True


def test_compute_next_phase_framework_pr_skips_explore_to_kernel():
    state = _State(
        phase=phase_state.PHASE_FRAMEWORK_PR,
        framework_pr_batches=[
            {"max_gain_pct_observed_in_batch": 0.1},
            {"max_gain_pct_observed_in_batch": 0.0},
            {"max_gain_pct_observed_in_batch": 0.2},
        ],
    )
    out = phase_state.compute_next_phase(
        state,
        framework_phase_enabled=True,
        explore_enabled=False,
        kernel_enabled=True,
    )
    assert out is not None
    next_phase, reason, ev = out
    assert next_phase == phase_state.PHASE_KERNEL
    assert reason == "framework_pr_plateau"
    assert ev.get("explore_skipped") is True


def test_compute_next_phase_explore_enabled_default_routes_to_explore():
    # Regression: the default explore_enabled=True keeps PRELUDE -> EXPLORE
    # with no explore_skipped marker.
    state = _State(phase=phase_state.PHASE_PRELUDE, baseline_tput=1500.0)
    out = phase_state.compute_next_phase(state, framework_phase_enabled=False)
    assert out is not None
    next_phase, _reason, ev = out
    assert next_phase == phase_state.PHASE_EXPLORE
    assert "explore_skipped" not in ev
