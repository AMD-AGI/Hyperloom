# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``exit_normal_optimize``: when the merged optimisation phase gives up.

Replaces ``test_phase_state_framework_agent``, which asserted the single-arm
rule against a hand-rolled ``SharedState`` stand-in. Both halves of that were
wrong: the rule grew a second arm, and a partial stub tests whatever fields
the assertions happen to touch rather than the state machine.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.phases import machine_state as ps

from ._optimize_fixtures import optimize_state

_DRY_SOURCE = {"source_no_keep": 99}
_DRY_CONFIG = {"config_keep_gain_pct": 0.0, "config_empty_rounds": 9}


# --------------------------------------------------------------------------- #
# The defining rule: both arms, or the phase stays.
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Either arm going quiet redirects the next cycle, whether or not it exits.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", [_DRY_SOURCE, _DRY_CONFIG])
def test_one_dry_arm_still_flags_a_bottleneck_switch(arm):
    state = optimize_state(**arm)
    source_dry, _ = ps.source_arm_plateaued(state)
    config_dry, _ = ps.compute_plateau_explore(state)
    assert source_dry or config_dry
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


# --------------------------------------------------------------------------- #
# Priority ladder.
# --------------------------------------------------------------------------- #
def test_a_skip_to_sweep_hint_leaves_with_an_arm_still_paying():
    """An explicit hint outranks the two-arm rule; that is the point of it."""
    state = optimize_state(pending_escalate_hint=ps.ESCALATE_HINT_SKIP_TO_SWEEP)
    verdict = ps.exit_normal_optimize(state)
    assert verdict is not None
    assert verdict[0] == "optimize_no_more_leverage"
    assert verdict[1]["evidence"] == "skip_to_sweep"


def test_a_skip_to_kernel_hint_is_ignored_until_a_specialist_round_ran():
    """A phase that dispatched nothing must not end with zero validated work."""
    state = optimize_state(pending_escalate_hint=ps.ESCALATE_HINT_SKIP_TO_KERNEL)
    state.specialist_rounds = []
    assert ps.exit_normal_optimize(state) is None

    state.specialist_rounds = [{"proposals_total": 1, "proposals_kept": 0, "cycle": 0}]
    verdict = ps.exit_normal_optimize(state)
    assert verdict is not None
    assert verdict[0] == "plateau_explore"


# --------------------------------------------------------------------------- #
# Where the phase goes next.
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


def test_every_reason_it_emits_is_in_the_exit_vocabulary():
    """A reason absent from the vocab is one the report cannot explain."""
    emitted = {
        "optimize_no_more_leverage",
        "optimize_phase_budget_exhausted",
        "optimize_budget_cap",
        "optimize_force_exit_low_budget",
        "plateau_explore",
    }
    assert emitted <= ps.PHASE_EXIT_REASONS
