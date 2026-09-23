# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``exit_normal_optimize``: when the merged optimisation phase gives up."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.phases import machine_state as ps

from ._optimize_fixtures import optimize_state

_DRY_SOURCE = {"source_no_keep": 99}
_DRY_CONFIG = {"config_keep_gain_pct": 0.0, "config_empty_rounds": 9}


# --------------------------------------------------------------------------- # The defining rule: both arms, or the
# phase stays. --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "source_dry,config_dry,leaves",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ],
)
def test_the_phase_leaves_only_when_both_arms_are_dry(source_dry, config_dry, leaves):
    state = optimize_state(**({**_DRY_SOURCE} if source_dry else {}), **({**_DRY_CONFIG} if config_dry else {}))
    verdict = ps.exit_normal_optimize(state)
    assert (verdict is not None) is leaves
    if leaves:
        assert verdict[0] == "optimize_no_more_leverage"
        assert verdict[1]["evidence"] == "both_arms_plateaued"


def test_candidate_exhaustion_dries_the_source_arm_without_a_no_keep_streak():
    """Discovery reporting itself done is the other way the source arm ends."""
    state = optimize_state(source_exhausted=True, **_DRY_CONFIG)
    verdict = ps.exit_normal_optimize(state)
    assert verdict is not None
    assert verdict[1]["source_candidates_exhausted"] is True


# --------------------------------------------------------------------------- # Either arm going quiet redirects the
# next cycle, whether or not it exits. --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", [_DRY_SOURCE, _DRY_CONFIG])
def test_one_dry_arm_still_flags_a_bottleneck_switch(arm):
    state = optimize_state(**arm)
    patch_dry, _ = ps._patch_lever_dry(state, {})
    config_dry, _ = ps._config_lever_dry(state, {})
    assert patch_dry or config_dry
    # No exit, but the signal is available to whatever does exit later.
    assert ps.exit_normal_optimize(state) is None


def test_switch_bottleneck_rides_every_exit_path():
    """It is dropped most easily on the paths that leave for another reason."""
    paths = {
        "both arms": optimize_state(**_DRY_SOURCE, **_DRY_CONFIG),
        "skip_to_sweep hint": optimize_state(pending_escalate_hint=ps.ESCALATE_HINT_SKIP_TO_SWEEP),
    }
    for label, state in paths.items():
        verdict = ps.exit_normal_optimize(state)
        assert verdict is not None, label
        assert "switch_bottleneck" in verdict[1], label


# --------------------------------------------------------------------------- # Priority ladder.
# --------------------------------------------------------------------------- #
def test_a_skip_to_sweep_hint_leaves_with_an_arm_still_paying():
    """An explicit hint outranks the two-arm rule; that is the point of it."""
    state = optimize_state(pending_escalate_hint=ps.ESCALATE_HINT_SKIP_TO_SWEEP)
    verdict = ps.exit_normal_optimize(state)
    assert verdict is not None
    assert verdict[0] == "optimize_no_more_leverage"
    assert verdict[1]["evidence"] == "skip_to_sweep"


# --------------------------------------------------------------------------- # Where the phase goes next.
# --------------------------------------------------------------------------- #
def test_exhausted_leverage_switches_lever_rather_than_ending_the_run():
    state = optimize_state(**_DRY_SOURCE, **_DRY_CONFIG)
    target, reason, _ = ps.compute_next_phase(state, kernel_enabled=True)
    assert target == ps.PHASE_KERNEL_AGENT
    assert reason == "optimize_no_more_leverage"


def test_with_kernel_disabled_it_winds_down_to_sweep_carrying_its_reason():
    state = optimize_state(**_DRY_SOURCE, **_DRY_CONFIG)
    target, reason, evidence = ps.compute_next_phase(state, kernel_enabled=False)
    assert target == ps.PHASE_SWEEP
    assert reason == "no_kernel_skipped"
    assert evidence["passed_through_reason"] == "optimize_no_more_leverage"


def test_the_config_arm_needs_a_trailing_no_keep_streak_to_report_dry():
    """Low gain alone (streak == 0) does not trigger config lever dryness."""
    state = SimpleNamespace(
        macro_cycle=0,
        # A single adopted row: gain is below floor but streak is 0.
        attempts=[
            {"lever_kind": "config", "outcome": "KEEP", "adopted": True, "gain_pct": 0.01, "cycle": 0},
        ],
    )
    triggered, evidence = ps._config_lever_dry(state, {})
    assert triggered is False
    assert evidence["empty_streak"] == 0


def test_skip_to_kernel_leaves_on_an_optimize_reason():
    """Every exit from the merged phase names it; ``plateau_explore`` named a phase that is gone."""
    state = optimize_state(source_no_keep=0, config_keep_gain_pct=9.0)
    state.pending_escalate_hint = ps.ESCALATE_HINT_SKIP_TO_KERNEL
    state.explore_search = {"tested": {"fp0": {"cycle": 0}}, "winners_history": []}

    out = ps.exit_normal_optimize(state)

    assert out is not None
    assert out[0] == "optimize_no_more_leverage"
    assert out[1]["evidence"] == "llm_escalation"


def test_skip_to_kernel_is_refused_before_either_arm_has_run():
    """A phase that dispatched nothing must not end with zero validated work."""
    state = optimize_state(config_keep_gain_pct=9.0)
    state.pending_escalate_hint = ps.ESCALATE_HINT_SKIP_TO_KERNEL
    state.specialist_rounds = []
    state.explore_search = {"tested": {}, "winners_history": []}
    state.framework_agent_phase_progress = []
    # Clear attempts so the phase has recorded no work this cycle.
    state.attempts = []

    assert ps.exit_normal_optimize(state) is None
