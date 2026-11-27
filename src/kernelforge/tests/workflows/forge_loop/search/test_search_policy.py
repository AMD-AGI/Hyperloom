"""Tests for deterministic EXPLOIT and DIVERSIFY policy decisions."""

from __future__ import annotations

import pytest

from kernelforge.workflows.forge_loop.campaign.run_state import RunState
from kernelforge.workflows.forge_loop.search.search_policy import (
    DIVERSIFY_AFTER_ROUNDS,
    MARGINAL_GAIN_FLOOR,
    NO_CHANGES_ESCALATION_THRESHOLD,
    OBJECTIVE_DISCOVER_NEW_MECHANISM,
    OBJECTIVE_CANONICAL_GAIN,
    SEARCH_MODE_DIVERSIFY,
    SEARCH_MODE_EXPLOIT,
    SUPERVISE_AFTER_ROUNDS,
    RecentProgress,
    SearchPolicyDecision,
    SearchPolicyEngine,
    SearchState,
)


def test_warm_start_exploits_until_stalled():
    decision = SearchPolicyEngine().decide(
        SearchState(best_source="warm_start", no_improvement_iters=0), diversify_after=3
    )

    assert decision.mode == SEARCH_MODE_EXPLOIT
    assert decision.reason_codes == ("KB_WARM_START_EXPLOIT",)


def test_stalled_warm_start_enters_diversify():
    decision = SearchPolicyEngine().decide(
        SearchState(best_source="warm_start", no_improvement_iters=3), diversify_after=3
    )

    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("NO_IMPROVEMENT_STALL",)


def test_fresh_productive_search_exploits():
    decision = SearchPolicyEngine().decide(
        SearchState(best_source="iteration", no_improvement_iters=0), diversify_after=3
    )

    assert decision.mode == SEARCH_MODE_EXPLOIT


def test_stall_enters_diversify():
    stalled = SearchPolicyEngine().decide(
        SearchState(best_source="iteration", no_improvement_iters=3), diversify_after=3
    )

    assert stalled.mode == SEARCH_MODE_DIVERSIFY
    assert stalled.reason_codes == ("NO_IMPROVEMENT_STALL",)


def test_a_stalled_search_stays_diversified_until_a_new_best():
    """Nothing but a new best ends the stall: every round in it is sent to a different mechanism."""
    engine = SearchPolicyEngine()

    def decide(stalled: int) -> SearchPolicyDecision:
        return engine.decide(
            SearchState(best_source="iteration", no_improvement_iters=stalled), diversify_after=DIVERSIFY_AFTER_ROUNDS
        )

    assert [decide(stalled).mode for stalled in (1, 2, 3, 4)] == [SEARCH_MODE_DIVERSIFY] * 4
    assert decide(0).mode == SEARCH_MODE_EXPLOIT


def test_the_search_turns_one_round_before_the_supervisor_reviews_it():
    assert (DIVERSIFY_AFTER_ROUNDS, SUPERVISE_AFTER_ROUNDS) == (1, 2)


def test_repeated_no_changes_diversifies_below_the_stall_threshold():
    decision = SearchPolicyEngine().decide(
        SearchState(best_source="iteration", no_improvement_iters=NO_CHANGES_ESCALATION_THRESHOLD),
        diversify_after=NO_CHANGES_ESCALATION_THRESHOLD + 1,
        recent=RecentProgress(consecutive_no_changes=NO_CHANGES_ESCALATION_THRESHOLD),
    )

    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("REPEATED_NO_CHANGES",)
    assert decision.objective_kind == OBJECTIVE_DISCOVER_NEW_MECHANISM


def test_first_no_changes_does_not_escalate():
    decision = SearchPolicyEngine().decide(
        SearchState(best_source="iteration", no_improvement_iters=1),
        diversify_after=3,
        recent=RecentProgress(consecutive_no_changes=NO_CHANGES_ESCALATION_THRESHOLD - 1),
    )

    assert decision.mode == SEARCH_MODE_EXPLOIT
    assert decision.reason_codes == ("CANONICAL_GAIN_AVAILABLE",)


def test_diminishing_returns_diversify_while_the_last_iteration_still_kept():
    """A ladder can flatten without ever stopping, and that is the case here."""
    decision = SearchPolicyEngine().decide(
        SearchState(best_source="iteration", no_improvement_iters=0),
        diversify_after=3,
        recent=RecentProgress(window_gain_ratio=MARGINAL_GAIN_FLOOR / 5),
    )

    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("DIMINISHING_RETURNS",)
    assert decision.objective_kind == OBJECTIVE_DISCOVER_NEW_MECHANISM


def test_a_ladder_still_climbing_keeps_its_direction():
    decision = SearchPolicyEngine().decide(
        SearchState(best_source="iteration", no_improvement_iters=0),
        diversify_after=3,
        recent=RecentProgress(window_gain_ratio=MARGINAL_GAIN_FLOOR * 2),
    )

    assert decision.mode == SEARCH_MODE_EXPLOIT
    assert decision.reason_codes == ("CANONICAL_GAIN_AVAILABLE",)


def test_an_unmeasured_window_is_not_a_flat_one():
    """No window yet must not read as a window of zero gain."""
    decision = SearchPolicyEngine().decide(
        SearchState(best_source="iteration", no_improvement_iters=0),
        diversify_after=3,
        recent=RecentProgress(window_gain_ratio=None),
    )

    assert decision.mode == SEARCH_MODE_EXPLOIT
    assert decision.reason_codes == ("CANONICAL_GAIN_AVAILABLE",)


def test_diminishing_returns_outrank_a_warm_started_incumbent():
    """A warm start earns exploitation, but not a whole flat window of it."""
    decision = SearchPolicyEngine().decide(
        SearchState(best_source="warm_start", no_improvement_iters=0),
        diversify_after=3,
        recent=RecentProgress(window_gain_ratio=0.0),
    )

    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("DIMINISHING_RETURNS",)


def test_a_stall_is_still_reported_as_a_stall():
    """A flat window and a stalled one are the same campaign; codes must not swap."""
    decision = SearchPolicyEngine().decide(
        SearchState(best_source="iteration", no_improvement_iters=3),
        diversify_after=3,
        recent=RecentProgress(window_gain_ratio=0.0),
    )

    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("NO_IMPROVEMENT_STALL",)


@pytest.mark.parametrize("ratio", [float("nan"), float("inf")])
def test_a_gain_ratio_that_is_not_a_number_is_refused(ratio):
    """A non-finite ratio compares false against the floor and reads as healthy."""
    with pytest.raises(ValueError, match="window_gain_ratio"):
        SearchPolicyEngine().decide(
            SearchState(best_source="iteration", no_improvement_iters=0),
            diversify_after=3,
            recent=RecentProgress(window_gain_ratio=ratio),
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
            objective_kind=OBJECTIVE_CANONICAL_GAIN,
        )


def test_run_state_persists_the_search_policy():
    state = RunState(
        search_mode=SEARCH_MODE_DIVERSIFY,
        search_reason_codes=["NO_IMPROVEMENT_STALL"],
        search_objective=OBJECTIVE_DISCOVER_NEW_MECHANISM,
    )

    restored = RunState.from_dict(state.to_dict())

    assert restored.search_mode == SEARCH_MODE_DIVERSIFY
    assert restored.search_reason_codes == ["NO_IMPROVEMENT_STALL"]
    assert restored.search_objective == OBJECTIVE_DISCOVER_NEW_MECHANISM
