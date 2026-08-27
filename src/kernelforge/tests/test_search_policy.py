"""Tests for deterministic EXPLOIT and DIVERSIFY policy decisions."""

from __future__ import annotations

import pytest

from kernelforge.loop.run_state import RunState, SCHEMA_VERSION
from kernelforge.loop.search_policy import (
    MARGINAL_GAIN_FLOOR,
    NO_CHANGES_ESCALATION_THRESHOLD,
    OBJECTIVE_DISCOVER_NEW_MECHANISM,
    OBJECTIVE_IMMEDIATE_CANONICAL_GAIN,
    SEARCH_MODE_DIVERSIFY,
    SEARCH_MODE_EXPLOIT,
    SearchPolicyDecision,
    SearchPolicyEngine,
)


def test_warm_start_exploits_until_stalled():
    decision = SearchPolicyEngine().decide(
        best_source="warm_start",
        no_improvement_iters=0,
        stall_threshold=3,
    )

    assert decision.mode == SEARCH_MODE_EXPLOIT
    assert decision.reason_codes == ("KB_WARM_START_EXPLOIT",)


def test_stalled_warm_start_enters_diversify():
    decision = SearchPolicyEngine().decide(
        best_source="warm_start",
        no_improvement_iters=3,
        stall_threshold=3,
    )

    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("NO_IMPROVEMENT_STALL",)


def test_fresh_productive_search_exploits():
    decision = SearchPolicyEngine().decide(
        best_source="iteration",
        no_improvement_iters=0,
        stall_threshold=3,
    )

    assert decision.mode == SEARCH_MODE_EXPLOIT


def test_stall_enters_diversify():
    stalled = SearchPolicyEngine().decide(
        best_source="iteration",
        no_improvement_iters=3,
        stall_threshold=3,
    )

    assert stalled.mode == SEARCH_MODE_DIVERSIFY
    assert stalled.reason_codes == ("NO_IMPROVEMENT_STALL",)


def test_completed_diversify_cycle_opens_bounded_exploit_window():
    engine = SearchPolicyEngine()

    first = engine.decide(
        best_source="warm_start",
        no_improvement_iters=1,
        stall_threshold=3,
        current_mode=SEARCH_MODE_DIVERSIFY,
        diversification_cycle_completed=True,
    )
    second = engine.decide(
        best_source="warm_start",
        no_improvement_iters=2,
        stall_threshold=3,
        current_mode=first.mode,
        residence_iterations_remaining=(first.residence_iterations_remaining),
    )

    assert first.mode == SEARCH_MODE_EXPLOIT
    assert first.reason_codes == ("DIVERSIFY_PLAN_CREATED",)
    assert first.residence_iterations_remaining == 2
    assert second.mode == SEARCH_MODE_EXPLOIT
    assert second.reason_codes == ("MODE_RESIDENCE",)
    assert second.residence_iterations_remaining == 1


def test_incomplete_diversify_cycle_stays_in_diversify():
    decision = SearchPolicyEngine().decide(
        best_source="iteration",
        no_improvement_iters=5,
        stall_threshold=3,
        current_mode=SEARCH_MODE_DIVERSIFY,
    )

    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("NO_IMPROVEMENT_STALL",)


def test_repeated_no_changes_diversifies_below_the_stall_threshold():
    decision = SearchPolicyEngine().decide(
        best_source="iteration",
        no_improvement_iters=NO_CHANGES_ESCALATION_THRESHOLD,
        stall_threshold=NO_CHANGES_ESCALATION_THRESHOLD + 1,
        consecutive_no_changes=NO_CHANGES_ESCALATION_THRESHOLD,
    )

    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("REPEATED_NO_CHANGES",)
    assert decision.objective_kind == OBJECTIVE_DISCOVER_NEW_MECHANISM
    assert decision.residence_iterations_remaining == 0


def test_first_no_changes_does_not_escalate():
    decision = SearchPolicyEngine().decide(
        best_source="iteration",
        no_improvement_iters=1,
        stall_threshold=3,
        consecutive_no_changes=NO_CHANGES_ESCALATION_THRESHOLD - 1,
    )

    assert decision.mode == SEARCH_MODE_EXPLOIT
    assert decision.reason_codes == ("CANONICAL_GAIN_AVAILABLE",)


def test_repeated_no_changes_outranks_mode_residence():
    """Escalate on empty diffs even while the mode is held in EXPLOIT.

    Residence weighs how promising the current direction is; repeated empty
    diffs are evidence it cannot be turned into a candidate at all, so holding
    EXPLOIT would spend another session on a direction that produces no edit.
    """
    engine = SearchPolicyEngine()

    held = engine.decide(
        best_source="iteration",
        no_improvement_iters=0,
        stall_threshold=3,
        current_mode=SEARCH_MODE_EXPLOIT,
        residence_iterations_remaining=2,
        consecutive_no_changes=NO_CHANGES_ESCALATION_THRESHOLD,
    )
    # Same residence, one empty diff short of the threshold: the only difference
    # is the streak, so residence must still win here or the test above proves
    # nothing about which signal outranks which.
    not_yet = engine.decide(
        best_source="iteration",
        no_improvement_iters=0,
        stall_threshold=3,
        current_mode=SEARCH_MODE_EXPLOIT,
        residence_iterations_remaining=2,
        consecutive_no_changes=NO_CHANGES_ESCALATION_THRESHOLD - 1,
    )

    assert held.mode == SEARCH_MODE_DIVERSIFY
    assert held.reason_codes == ("REPEATED_NO_CHANGES",)
    assert not_yet.mode == SEARCH_MODE_EXPLOIT
    assert not_yet.reason_codes == ("MODE_RESIDENCE",)


def test_diminishing_returns_diversify_while_the_last_iteration_still_kept():
    """A ladder can flatten without ever stopping, and that is the case here.

    ``no_improvement_iters`` is 0: every recent iteration was a KEEP, so no
    stall signal exists and the campaign would refine the same direction for as
    long as it keeps producing gains too small to reach a different mechanism.
    """
    decision = SearchPolicyEngine().decide(
        best_source="iteration",
        no_improvement_iters=0,
        stall_threshold=3,
        window_gain_ratio=MARGINAL_GAIN_FLOOR / 5,
    )

    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("DIMINISHING_RETURNS",)
    assert decision.objective_kind == OBJECTIVE_DISCOVER_NEW_MECHANISM
    assert decision.residence_iterations_remaining == 0


def test_a_ladder_still_climbing_keeps_its_direction():
    decision = SearchPolicyEngine().decide(
        best_source="iteration",
        no_improvement_iters=0,
        stall_threshold=3,
        window_gain_ratio=MARGINAL_GAIN_FLOOR * 2,
    )

    assert decision.mode == SEARCH_MODE_EXPLOIT
    assert decision.reason_codes == ("CANONICAL_GAIN_AVAILABLE",)


def test_an_unmeasured_window_is_not_a_flat_one():
    """No window yet must not read as a window of zero gain.

    The two are one ``0.0`` apart at the call site, and conflating them would
    diversify every campaign on its first iterations -- before it has spent a
    single one on the direction it was given.
    """
    decision = SearchPolicyEngine().decide(
        best_source="iteration",
        no_improvement_iters=0,
        stall_threshold=3,
        window_gain_ratio=None,
    )

    assert decision.mode == SEARCH_MODE_EXPLOIT
    assert decision.reason_codes == ("CANONICAL_GAIN_AVAILABLE",)


def test_diminishing_returns_outrank_a_warm_started_incumbent():
    """A warm start earns exploitation, but not a whole flat window of it."""
    decision = SearchPolicyEngine().decide(
        best_source="warm_start",
        no_improvement_iters=0,
        stall_threshold=3,
        window_gain_ratio=0.0,
    )

    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("DIMINISHING_RETURNS",)


def test_mode_residence_outranks_diminishing_returns():
    """The round after a diversification is protected from the new trigger.

    Its window still holds the flat outcomes that forced that diversification,
    so without this the campaign would diversify again on the same evidence
    instead of exploiting what the diversification found.
    """
    engine = SearchPolicyEngine()

    completed = engine.decide(
        best_source="iteration",
        no_improvement_iters=0,
        stall_threshold=3,
        current_mode=SEARCH_MODE_DIVERSIFY,
        diversification_cycle_completed=True,
        window_gain_ratio=0.0,
    )
    held = engine.decide(
        best_source="iteration",
        no_improvement_iters=0,
        stall_threshold=3,
        current_mode=completed.mode,
        residence_iterations_remaining=(completed.residence_iterations_remaining),
        window_gain_ratio=0.0,
    )

    assert completed.mode == SEARCH_MODE_EXPLOIT
    assert completed.reason_codes == ("DIVERSIFY_PLAN_CREATED",)
    assert held.mode == SEARCH_MODE_EXPLOIT
    assert held.reason_codes == ("MODE_RESIDENCE",)


def test_a_stall_is_still_reported_as_a_stall():
    """A flat window and a stalled one are the same campaign; codes must not swap.

    A stalled campaign necessarily has a flat window, so the older code would
    disappear from the audit trail if the new branch were placed above it.
    """
    decision = SearchPolicyEngine().decide(
        best_source="iteration",
        no_improvement_iters=3,
        stall_threshold=3,
        window_gain_ratio=0.0,
    )

    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("NO_IMPROVEMENT_STALL",)


@pytest.mark.parametrize("ratio", [float("nan"), float("inf")])
def test_a_gain_ratio_that_is_not_a_number_is_refused(ratio):
    """A non-finite ratio compares false against the floor and reads as healthy.

    That is the silent failure this rejects: the trigger would be switched off
    for the rest of the campaign and nothing downstream would say so.
    """
    with pytest.raises(ValueError, match="window_gain_ratio"):
        SearchPolicyEngine().decide(
            best_source="iteration",
            no_improvement_iters=0,
            stall_threshold=3,
            window_gain_ratio=ratio,
        )


def test_a_decision_cannot_carry_a_mode_the_loop_cannot_run():
    """A mode outside the pair is not a third strategy, it is a typo."""
    with pytest.raises(ValueError, match="unsupported search mode"):
        SearchPolicyDecision(
            mode="EXPLORE",
            reason_codes=("NO_IMPROVEMENT_STALL",),
            objective_kind=OBJECTIVE_DISCOVER_NEW_MECHANISM,
        )


def test_a_decision_must_state_why_it_was_taken():
    """The reason codes are the audit trail; an unexplained mode switch is a bug."""
    with pytest.raises(ValueError, match="reason_codes"):
        SearchPolicyDecision(
            mode=SEARCH_MODE_EXPLOIT,
            reason_codes=(),
            objective_kind=OBJECTIVE_IMMEDIATE_CANONICAL_GAIN,
        )


def test_run_state_persists_plan_search_policy():
    state = RunState(
        search_mode=SEARCH_MODE_DIVERSIFY,
        search_reason_codes=["NO_IMPROVEMENT_STALL"],
        search_objective=OBJECTIVE_DISCOVER_NEW_MECHANISM,
        search_mode_residence_remaining=2,
        diversification_cycle_completed=True,
    )

    restored = RunState.from_dict(state.to_dict())

    assert restored.search_mode == SEARCH_MODE_DIVERSIFY
    assert restored.search_reason_codes == ["NO_IMPROVEMENT_STALL"]
    assert restored.search_objective == OBJECTIVE_DISCOVER_NEW_MECHANISM
    assert restored.search_mode_residence_remaining == 2
    assert restored.diversification_cycle_completed is True
    assert restored.schema_version == SCHEMA_VERSION
