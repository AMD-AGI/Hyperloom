# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for admitting a round only when the remaining budget can finish it."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

from kernelforge.loop.round_budget import (
    ADMISSION_SESSION_SEC,
    DISPATCH_FLOOR_SEC,
    DISPATCH_SESSION_SEC,
    FIRST_ROUND_MEASUREMENT_SEC,
    PLANNING_FLOOR_SEC,
    admit_dispatch,
    admit_round,
    estimate_measurement_sec,
    estimate_planning_sec,
)
from kernelforge.loop.run_state import (
    ROUND_COST_WINDOW,
    RoundCost,
    RoundCostState,
    RunState,
    apply_round_cost,
)
from kernelforge.orchestrator.plan_critic import PLAN_CRITIC_TIMEOUT_SEC


def _admit(remaining_sec, *, lanes=3, history=None, measurement_sec=None):
    history = list(history or [])
    return admit_round(
        remaining_sec=remaining_sec,
        requested_lanes=lanes,
        history=history,
        measurement_sec=(estimate_measurement_sec(history) if measurement_sec is None else measurement_sec),
    )


def _fan_out_history(planning_sec, *, lanes=3, rounds=1, measurement_sec=0.0):
    return [
        RoundCost(
            iteration=index + 1,
            lanes=lanes,
            planning_sec=planning_sec,
            total_sec=planning_sec + 3000.0,
            measurement_sec=measurement_sec,
        )
        for index in range(rounds)
    ]


# What a round with no observed history of its own has to cover once its plans
# exist, as the check BEFORE planning prices it: the least a session can be
# given and the canonical measurement that judges it. Dispatch prices the same
# execution higher, which is the asymmetry between a bound that only refuses
# what cannot run and one that commits the loop to a session it cannot
# interrupt.
_EXECUTION_SEC = ADMISSION_SESSION_SEC + FIRST_ROUND_MEASUREMENT_SEC


# ---------------------------------------------------------------------------
# Before planning: a lower bound that refuses only what cannot run at all.
# ---------------------------------------------------------------------------


def test_first_round_is_admitted_at_full_width_without_history():
    decision = _admit(3600.0)

    assert decision.admitted is True
    assert decision.lanes == 3
    assert decision.narrowed is False
    assert decision.required_sec == pytest.approx(PLANNING_FLOOR_SEC + _EXECUTION_SEC)


def test_a_round_that_cannot_buy_the_cheapest_planning_is_refused():
    """The one thing this check is for: not paying for a plan nothing can run."""
    decision = _admit(PLANNING_FLOOR_SEC + _EXECUTION_SEC - 1.0)

    assert decision.admitted is False
    assert decision.lanes == 1


def test_round_that_fits_is_admitted_unchanged():
    history = _fan_out_history(1500.0)
    required = 1500.0 + _EXECUTION_SEC

    decision = _admit(required + 1.0, history=history)

    assert decision.admitted is True
    assert decision.lanes == 3
    assert decision.narrowed is False
    assert decision.required_sec == pytest.approx(required)


def test_a_round_that_does_not_fit_narrows_one_width_at_a_time():
    """Two lanes search twice as widely as one, so two is tried before one."""
    history = _fan_out_history(2400.0)
    fan_out_required = 2400.0 + _EXECUTION_SEC

    decision = _admit(fan_out_required - 60.0, history=history)

    assert decision.admitted is True
    assert decision.lanes == 2
    assert decision.narrowed is True
    # One lane plan the Critic no longer has to read is the whole saving.
    assert decision.planning_sec == pytest.approx(2400.0 - PLAN_CRITIC_TIMEOUT_SEC)
    assert decision.required_sec < fan_out_required


def test_a_round_narrows_to_one_lane_when_two_still_do_not_fit():
    history = _fan_out_history(2400.0)

    decision = _admit(2400.0 - PLAN_CRITIC_TIMEOUT_SEC + _EXECUTION_SEC - 60.0, history=history)

    assert decision.admitted is True
    assert decision.lanes == 1
    assert decision.narrowed is True


def test_round_is_refused_when_even_one_lane_does_not_fit():
    history = _fan_out_history(2400.0)

    decision = _admit(600.0, history=history)

    assert decision.admitted is False
    assert decision.lanes == 1
    assert decision.remaining_sec == pytest.approx(600.0)
    assert decision.required_sec > 600.0


def test_refusal_reports_the_narrowest_round_it_could_not_afford():
    history = _fan_out_history(2400.0)

    decision = _admit(0.0, history=history)

    assert decision.planning_sec == pytest.approx(2400.0 - 2 * PLAN_CRITIC_TIMEOUT_SEC)


def test_single_lane_campaign_is_never_narrowed():
    history = _fan_out_history(2400.0, lanes=1)

    decision = _admit(0.0, lanes=1, history=history)

    assert decision.admitted is False
    assert decision.narrowed is False
    assert decision.lanes == 1


# ---------------------------------------------------------------------------
# What the two halves of a round are priced from.
# ---------------------------------------------------------------------------


def test_the_planning_bound_is_the_cheapest_round_observed_not_the_worst():
    """A bound, not an expectation: the worst round would refuse the rest."""
    history = [
        *_fan_out_history(2700.0),
        *_fan_out_history(1900.0),
        *_fan_out_history(2500.0),
    ]

    assert estimate_planning_sec(history, lanes=3) == pytest.approx(1900.0)
    assert estimate_planning_sec([], lanes=3) == pytest.approx(PLANNING_FLOOR_SEC)


def test_a_narrower_round_is_priced_from_its_own_observation_first():
    history = [
        *_fan_out_history(2400.0),
        RoundCost(iteration=9, lanes=1, planning_sec=1100.0, total_sec=4000.0),
    ]

    assert estimate_planning_sec(history, lanes=1) == pytest.approx(1100.0)


def test_a_narrower_round_is_otherwise_priced_by_the_plans_left_unread():
    history = _fan_out_history(2400.0, lanes=4)

    assert estimate_planning_sec(history, lanes=2) == pytest.approx(2400.0 - 2 * PLAN_CRITIC_TIMEOUT_SEC)


def test_no_bound_claims_to_plan_faster_than_anything_observed():
    history = _fan_out_history(900.0, lanes=8)

    assert estimate_planning_sec(history, lanes=1) == pytest.approx(PLANNING_FLOOR_SEC)


def test_a_wider_round_is_never_priced_below_a_narrower_observed_one():
    history = [
        RoundCost(iteration=1, lanes=1, planning_sec=1100.0, total_sec=4000.0),
    ]

    assert estimate_planning_sec(history, lanes=3) == pytest.approx(1100.0)


def test_the_measurement_estimate_starts_at_a_constant_and_then_observes():
    """The old estimate was the timeout ceilings: 19x the observed p90."""
    assert estimate_measurement_sec([]) == pytest.approx(FIRST_ROUND_MEASUREMENT_SEC)
    assert FIRST_ROUND_MEASUREMENT_SEC < 1800.0 + 300.0

    observed = [
        *_fan_out_history(1500.0, measurement_sec=40.0),
        *_fan_out_history(1500.0, measurement_sec=150.0),
        *_fan_out_history(1500.0, measurement_sec=90.0),
    ]

    assert estimate_measurement_sec(observed) == pytest.approx(150.0)


def test_a_round_that_never_measured_is_not_an_observation_of_a_free_cycle():
    history = _fan_out_history(1500.0, measurement_sec=0.0, rounds=3)

    assert estimate_measurement_sec(history) == pytest.approx(FIRST_ROUND_MEASUREMENT_SEC)


# ---------------------------------------------------------------------------
# After planning: the decisive check.
# ---------------------------------------------------------------------------


def _dispatch(remaining_sec, *, measurement_sec=FIRST_ROUND_MEASUREMENT_SEC):
    return admit_dispatch(
        remaining_sec=remaining_sec,
        measurement_sec=measurement_sec,
    )


# The worst of the 171 production validate-and-benchmark cycles, and a quarter
# of what a campaign assumes before it has one of its own. The estimate is a
# high-water over what this campaign has measured, so a campaign converges to
# its own worst cycle: this is the LARGEST any of the production campaigns
# would have ended up with, and most of them would have gone lower.
_FAST_MEASUREMENT_SEC = 150.0

# The production distribution each constant is read from, kept here so that
# re-calibrating one has to argue with the measurement it came from rather than
# slip past a test that only re-derives the formula.
_SESSION_P25_SEC = 8.0 * 60.0
_SESSION_MEDIAN_SEC = 12.3 * 60.0
_SESSION_P90_SEC = 34.6 * 60.0
# Production ran a 10.75-hour budget against an 11-hour external kill.
_EXTERNAL_GRACE_SEC = (11.0 - 10.75) * 3600.0


def test_each_constant_is_the_production_number_it_claims_to_be():
    """The values themselves, not just the arithmetic over them.

    Both session prices and the floor are calibrated numbers: every other test
    here recomputes the same formula the module does and would stay green if
    one of them were quietly moved.
    """
    assert ADMISSION_SESSION_SEC == pytest.approx(_SESSION_P25_SEC)
    assert DISPATCH_SESSION_SEC == pytest.approx(_SESSION_MEDIAN_SEC)
    # The floor is the p90 session less the grace the external kill allows:
    # at exactly this much time in hand, a p90 session ends as the kill lands.
    assert DISPATCH_FLOOR_SEC == pytest.approx(_SESSION_P90_SEC - _EXTERNAL_GRACE_SEC)


def test_dispatch_needs_a_session_and_the_measurement_that_judges_it():
    """A campaign with nothing observed pays for a median session and a cycle."""
    required = DISPATCH_SESSION_SEC + FIRST_ROUND_MEASUREMENT_SEC
    assert required > DISPATCH_FLOOR_SEC

    decision = _dispatch(required)

    assert decision.admitted is True
    assert decision.floored is False
    assert decision.required_sec == pytest.approx(required)
    assert _dispatch(required - 1.0).admitted is False


def test_dispatch_prices_a_session_above_the_check_taken_before_planning():
    """The same session, the opposite asymmetry.

    Before planning, a bound that is too generous refuses a round that would
    have worked, so a session is priced at the p25 of what sessions cost. After
    planning the loop cannot take the session back once it starts, and too
    small a bound starts one the external timeout kills -- so the same session
    is priced at the median. What passes the first check therefore does not
    automatically pass the second.
    """
    assert DISPATCH_SESSION_SEC > ADMISSION_SESSION_SEC
    assert _dispatch(_EXECUTION_SEC).admitted is False


def test_an_expensive_measurement_cycle_still_raises_the_requirement():
    """The floor is a lower bound on the estimate, not a replacement for it."""
    history = _fan_out_history(1500.0, measurement_sec=900.0)
    required = DISPATCH_SESSION_SEC + 900.0
    assert required > DISPATCH_FLOOR_SEC

    decision = admit_dispatch(
        remaining_sec=required - 1.0,
        measurement_sec=estimate_measurement_sec(history),
    )

    assert decision.admitted is False
    assert decision.floored is False
    assert decision.required_sec == pytest.approx(required)


def test_the_dispatch_requirement_never_falls_below_its_floor():
    """No history buys a lower bar, because history does not price the kill.

    What the dispatch check guards is the external timeout: the loop cannot
    interrupt the session it starts and does not size that session from what
    remains, so a round dispatched too late runs past the deadline whatever
    this campaign has measured. A campaign that measures quickly has observed
    its own validation cycle, not that deadline.
    """
    histories = [
        [],
        _fan_out_history(1500.0, measurement_sec=5.0, rounds=4),
        _fan_out_history(1500.0, measurement_sec=36.0),
        _fan_out_history(1500.0, measurement_sec=_FAST_MEASUREMENT_SEC, rounds=3),
        _fan_out_history(1500.0, measurement_sec=0.0, rounds=5),
        _fan_out_history(1500.0, measurement_sec=900.0),
    ]

    for history in histories:
        decision = admit_dispatch(
            remaining_sec=0.0,
            measurement_sec=estimate_measurement_sec(history),
        )
        assert decision.required_sec >= DISPATCH_FLOOR_SEC

    # Nor does any cycle a campaign could observe, from the cheapest
    # production ever ran to one an order of magnitude past its worst. (A
    # measurement of zero is not an observation of a free cycle at all -- it
    # falls back to the constant, which is tested separately.)
    for measured in range(5, 2000, 25):
        decision = admit_dispatch(
            remaining_sec=0.0,
            measurement_sec=estimate_measurement_sec(_fan_out_history(1500.0, measurement_sec=float(measured))),
        )
        assert decision.required_sec >= DISPATCH_FLOOR_SEC

    # Nor an estimate no campaign could produce at all.
    assert _dispatch(0.0, measurement_sec=-1e6).required_sec >= DISPATCH_FLOOR_SEC


def test_a_fast_campaign_lowers_the_first_check_but_not_the_second():
    """Observation is right for one question and not for the other."""
    history = _fan_out_history(1500.0, measurement_sec=_FAST_MEASUREMENT_SEC, rounds=3)
    measurement = estimate_measurement_sec(history)
    assert measurement == pytest.approx(_FAST_MEASUREMENT_SEC)

    # Before planning, the campaign's own speed is exactly what should count:
    # a campaign that validates quickly may keep starting rounds later.
    admission = _admit(3600.0, history=history, measurement_sec=measurement)
    assert admission.execution_sec == pytest.approx(ADMISSION_SESSION_SEC + _FAST_MEASUREMENT_SEC)
    assert admission.execution_sec < _EXECUTION_SEC

    # After planning it is not: the requirement stops at the floor.
    decision = _dispatch(0.0, measurement_sec=measurement)
    assert decision.floored is True
    assert decision.required_sec == pytest.approx(DISPATCH_FLOOR_SEC)
    assert decision.required_sec > DISPATCH_SESSION_SEC + _FAST_MEASUREMENT_SEC


def test_a_refusal_names_what_it_was_priced_from():
    """A floored line says so; an estimated one still shows its parts."""
    estimated = _dispatch(60.0, measurement_sec=1200.0).summary()

    assert estimated.startswith(f"{(DISPATCH_SESSION_SEC + 1200.0) / 60:.0f} min needed after planning")
    assert "session 12, measurement 20" in estimated
    assert "floor" not in estimated

    summary = _dispatch(60.0, measurement_sec=_FAST_MEASUREMENT_SEC).summary()

    assert summary.startswith(f"{DISPATCH_FLOOR_SEC / 60:.0f} min needed after planning")
    assert "external-timeout floor" in summary
    assert "1 min remain" in summary


# ---------------------------------------------------------------------------
# The acceptance criterion: the rounds this policy was re-calibrated against.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProductionRound:
    """One round of the ten 11-hour production campaigns (2026-08-17).

    ``planning_min`` is None for the round whose planning never returned, which
    can only be judged by the check taken before it.
    """

    name: str
    remaining_min: float
    planning_min: float | None
    survived: bool


# Round start, measured planning, and what became of the round. The two that
# did not survive were killed by the external timeout with their lane sessions
# still running and no report written; three of the five that survived produced
# a KEEP, one of them the largest single gain any of the ten campaigns found.
PRODUCTION_ROUNDS = (
    _ProductionRound("gemma4-fused-moe iter 21", 30.0, 22.7, False),
    _ProductionRound("paged-attn-decode iter 13", 32.0, 23.7, False),
    _ProductionRound("gemma4-unified-attn iter 19", 33.0, None, True),
    _ProductionRound("verified-unified-attn iter 18", 42.0, 17.2, True),
    _ProductionRound("sparse-attn iter 18", 49.0, 19.8, True),
    _ProductionRound("mhc-fused iter 18", 50.0, 20.9, True),
    _ProductionRound("gemma4-unified-attn iter 17", 61.0, 21.9, True),
)


def _replay(round_: _ProductionRound, *, history) -> bool:
    """Whether the policy would have let this round run to a measurement."""
    admission = _admit(
        round_.remaining_min * 60.0,
        history=history,
        measurement_sec=estimate_measurement_sec(history),
    )
    if not admission.admitted:
        return False
    if round_.planning_min is None:
        # Planning never returned, so the round was never asked the second
        # question. Admitting it to planning is the whole decision here.
        return True
    return admit_dispatch(
        remaining_sec=(round_.remaining_min - round_.planning_min) * 60.0,
        measurement_sec=estimate_measurement_sec(history),
    ).admitted


@pytest.mark.parametrize(
    "round_",
    PRODUCTION_ROUNDS,
    ids=[entry.name for entry in PRODUCTION_ROUNDS],
)
def test_the_policy_matches_what_production_did_with_these_rounds(round_):
    """A campaign with no history of its own, priced from the constants."""
    assert _replay(round_, history=[]) is round_.survived


@pytest.mark.parametrize(
    "round_",
    PRODUCTION_ROUNDS,
    ids=[entry.name for entry in PRODUCTION_ROUNDS],
)
def test_the_verdict_does_not_change_once_the_campaign_has_observed_itself(
    round_,
):
    """The same rounds, on a campaign that has already run one like them.

    The one whose planning never returned is priced from the earlier round of
    its own campaign (iteration 17, 21.9 minutes), which is what that campaign
    would in fact have had in hand.
    """
    planning_min = round_.planning_min if round_.planning_min else 21.9
    history = _fan_out_history(planning_min * 60.0, lanes=3)

    assert _replay(round_, history=history) is round_.survived


@pytest.mark.parametrize(
    "round_",
    PRODUCTION_ROUNDS,
    ids=[entry.name for entry in PRODUCTION_ROUNDS],
)
def test_the_verdict_does_not_change_on_a_campaign_that_measures_quickly(
    round_,
):
    """The same rounds on a campaign whose measurement cycle is observed.

    Every campaign becomes this one after its first round. The constant a
    campaign assumes before it has measured anything is four times the worst
    cycle production ever ran, so the first observation replaces 600 seconds
    with something far smaller and every estimate built on it falls. These
    verdicts must not fall with them -- what dispatch is guarding against is an
    external deadline, and a campaign that validates quickly has learned
    nothing about that.
    """
    planning_min = round_.planning_min if round_.planning_min else 21.9
    history = _fan_out_history(planning_min * 60.0, lanes=3, measurement_sec=_FAST_MEASUREMENT_SEC)

    assert _replay(round_, history=history) is round_.survived


def _post_planning_sec(entries) -> list[float]:
    """What each of these rounds had left when its planning returned."""
    return [(entry.remaining_min - entry.planning_min) * 60.0 for entry in entries if entry.planning_min is not None]


def test_the_rounds_that_died_are_separated_from_the_survivors_after_planning():
    """The gap the whole re-calibration rests on: 8.3 minutes against 24.8."""
    killed = _post_planning_sec([entry for entry in PRODUCTION_ROUNDS if not entry.survived])
    survived = _post_planning_sec([entry for entry in PRODUCTION_ROUNDS if entry.survived])

    # Whatever this campaign has observed, the dispatch bar lands in the gap.
    for measurement_sec in (
        FIRST_ROUND_MEASUREMENT_SEC,
        _FAST_MEASUREMENT_SEC,
        36.0,
        0.0,
    ):
        required = _dispatch(0.0, measurement_sec=measurement_sec).required_sec
        assert max(killed) < required <= min(survived)

    # The floor alone -- the part observation cannot lower -- already clears
    # the deaths by more than twice their margin, and still refuses none of the
    # rounds that went on to a measured candidate.
    assert 2 * max(killed) < DISPATCH_FLOOR_SEC <= min(survived)


# ---------------------------------------------------------------------------
# The history the estimates are read from.
# ---------------------------------------------------------------------------


def test_recorded_round_costs_drive_the_next_admission():
    state = RunState()
    apply_round_cost(
        state,
        iteration=1,
        lanes=3,
        planning_sec=2400.0,
        total_sec=5000.0,
        measurement_sec=90.0,
        campaign_sec=6000.0,
    )

    decision = _admit(1500.0, history=state.round_costs.recent)

    assert decision.admitted is False
    # Both halves come from the round just recorded: the planning bound falls
    # with each unread plan, and the measurement is the one observed, not the
    # no-history constant.
    assert decision.planning_sec == pytest.approx(2400.0 - 2 * PLAN_CRITIC_TIMEOUT_SEC)
    assert decision.execution_sec == pytest.approx(ADMISSION_SESSION_SEC + 90.0)
    assert state.round_costs.rounds == 1
    assert state.round_costs.planning_total_sec == pytest.approx(2400.0)
    assert state.round_costs.total_sec == pytest.approx(5000.0)
    assert state.round_costs.recent[0].measurement_sec == pytest.approx(90.0)


def test_round_history_keeps_only_the_recent_window():
    state = RunState()
    for iteration in range(1, ROUND_COST_WINDOW + 4):
        apply_round_cost(
            state,
            iteration=iteration,
            lanes=2,
            planning_sec=100.0 * iteration,
            total_sec=1000.0 * iteration,
            campaign_sec=2000.0 * iteration,
        )

    assert state.round_costs.rounds == ROUND_COST_WINDOW + 3
    assert [cost.iteration for cost in state.round_costs.recent] == list(range(4, ROUND_COST_WINDOW + 4))


def test_a_round_that_did_not_plan_records_nothing():
    state = RunState()

    with pytest.raises(ValueError):
        apply_round_cost(
            state,
            iteration=1,
            lanes=1,
            planning_sec=0.0,
            total_sec=900.0,
            campaign_sec=900.0,
        )

    assert state.round_costs.rounds == 0


# The planning share and the span it is a share OF. These pin the structure
# that keeps them describing the same thing: the numerator cannot grow without
# the denominator, no caller supplies a denominator of its own, and a state
# that violates it does not load.


def test_planning_cannot_be_charged_without_advancing_the_span_it_is_in():
    """``campaign_sec`` is required, so the two halves advance together.

    The defect was a cumulative numerator paired with whatever span the caller
    happened to have. Making the span an argument the caller must pass, on the
    same call that charges the planning, is what makes that pairing impossible
    rather than merely unlikely.
    """
    state = RunState()

    with pytest.raises(TypeError):
        apply_round_cost(
            state,
            iteration=1,
            lanes=1,
            planning_sec=600.0,
            total_sec=900.0,
        )

    assert state.round_costs.rounds == 0


def test_the_campaign_span_is_never_left_behind_the_planning_inside_it():
    """A clock passed short of the planning charged to it is raised, not kept.

    A caller can hand over a span it measured badly -- a resumed session whose
    own process clock is seconds old is exactly that. It cannot make the share
    exceed 100 by doing so.
    """
    state = RunState()

    apply_round_cost(
        state,
        iteration=1,
        lanes=1,
        planning_sec=2400.0,
        total_sec=3000.0,
        campaign_sec=5.0,
    )

    assert state.round_costs.planning_total_sec == pytest.approx(2400.0)
    assert state.round_costs.campaign_sec == pytest.approx(2400.0)
    assert state.round_costs.planning_share_pct() == pytest.approx(100.0)


def test_the_campaign_span_only_moves_forward():
    """Sessions resume; the clock they inherit is not restarted by a later one."""
    state = RunState()
    for iteration, campaign_sec in ((1, 6000.0), (2, 100.0)):
        apply_round_cost(
            state,
            iteration=iteration,
            lanes=1,
            planning_sec=1200.0,
            total_sec=1500.0,
            campaign_sec=campaign_sec,
        )

    assert state.round_costs.campaign_sec == pytest.approx(6000.0)
    assert state.round_costs.planning_share_pct() == pytest.approx(40.0)


def test_a_campaign_with_no_clock_reports_no_share_rather_than_zero():
    """A share of nothing is not zero percent, and saying so would be a lie."""
    assert RunState().round_costs.planning_share_pct() is None


def test_planning_share_takes_no_denominator():
    """The single way these totals become a share, by construction.

    ``planning_share_pct`` reads the campaign clock stored beside the planning
    it divides. There is no parameter for a caller to pass a different span
    through, which is the property this test exists to keep.
    """
    assert inspect.signature(RoundCostState.planning_share_pct).parameters.keys() == {"self"}


def test_a_state_whose_span_cannot_contain_its_planning_does_not_load():
    """The invariant is checked at load, not only at the call that maintains it.

    A checkpoint edited by hand, or written by some future path that skips
    ``apply_round_cost``, would otherwise resurrect the share above 100.
    """
    state = RunState()
    state.round_costs = RoundCostState(
        rounds=1,
        planning_total_sec=2700.0,
        total_sec=3000.0,
        campaign_sec=600.0,
    )

    with pytest.raises(ValueError, match="campaign_sec must cover"):
        RunState.from_dict(state.to_dict())
