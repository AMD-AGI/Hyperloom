# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dividing the phase between lanes, and how many targets each share pays for."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.orchestrator.kernel import lane_budget as lb


def test_remaining_time_becomes_a_phase_budget_less_the_reserve() -> None:
    assert lb.phase_budget_sec(100.0, reserve_sec=300) == 100 * 60 - 300


def test_an_unbounded_session_yields_no_allocation() -> None:
    """None means nothing to divide, which callers read as "do not allocate"."""
    assert lb.phase_budget_sec(None) == 0


def test_a_phase_shorter_than_the_reserve_yields_zero_not_a_negative() -> None:
    assert lb.phase_budget_sec(1.0, reserve_sec=300) == 0


@pytest.mark.parametrize("value", [True, "20", object()])
def test_a_non_numeric_remaining_time_is_refused(value: Any) -> None:
    with pytest.raises(lb.LaneBudgetError, match="must be numeric"):
        lb.phase_budget_sec(value)


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_a_nonsensical_remaining_time_is_refused(value: float) -> None:
    with pytest.raises(lb.LaneBudgetError, match="finite and non-negative"):
        lb.phase_budget_sec(value)


def test_the_split_never_exceeds_the_whole() -> None:
    """Floored per lane, so rounding can only leave time unspent."""
    shares = lb.split_lanes(10_000)
    assert sum(shares.values()) <= 10_000


def test_the_split_follows_the_weights() -> None:
    shares = lb.split_lanes(1000, weights={lb.LANE_REWRITE: 3.0, lb.LANE_GEMM: 1.0})
    assert shares == {lb.LANE_REWRITE: 750, lb.LANE_GEMM: 250}


def test_weights_need_not_sum_to_one() -> None:
    shares = lb.split_lanes(100, weights={lb.LANE_REWRITE: 30.0, lb.LANE_GEMM: 10.0})
    assert shares == {lb.LANE_REWRITE: 75, lb.LANE_GEMM: 25}


def test_a_zero_phase_budget_splits_into_zeros() -> None:
    assert set(lb.split_lanes(0).values()) == {0}


def test_an_unknown_lane_is_refused() -> None:
    with pytest.raises(lb.LaneBudgetError, match="unknown lane"):
        lb.split_lanes(100, weights={"collective": 1.0})


@pytest.mark.parametrize("weight", [0.0, -1.0, float("inf"), True, "1"])
def test_an_unusable_weight_is_refused(weight: Any) -> None:
    with pytest.raises(lb.LaneBudgetError):
        lb.split_lanes(100, weights={lb.LANE_REWRITE: weight})


def test_no_weights_at_all_is_refused() -> None:
    with pytest.raises(lb.LaneBudgetError, match="at least one lane weight"):
        lb.split_lanes(100, weights={})


def test_rewrite_divides_by_its_admission_floor() -> None:
    """Overshooting here costs the whole share, not a slice of it."""
    assert lb.max_targets(lb.LANE_REWRITE, lb.REWRITE_MIN_TARGET_SEC * 2) == 2
    assert lb.max_targets(lb.LANE_REWRITE, lb.REWRITE_MIN_TARGET_SEC * 2 - 1) == 1


def test_rewrite_below_one_floor_funds_nothing() -> None:
    assert lb.max_targets(lb.LANE_REWRITE, lb.REWRITE_MIN_TARGET_SEC - 1) == 0


def test_gemm_consumes_router_estimates_in_order() -> None:
    """A tuner that does not fit is skipped, so the greedy fit stops there."""
    assert lb.max_targets(lb.LANE_GEMM, 1800, target_costs_sec=(900, 900, 900)) == 2


def test_gemm_stops_rather_than_skipping_to_fit_a_cheaper_later_tuner() -> None:
    """Order is the router's priority, so a costly early tuner is not passed over."""
    assert lb.max_targets(lb.LANE_GEMM, 1000, target_costs_sec=(900, 200, 50)) == 1


def test_gemm_fits_every_estimate_when_they_all_fit() -> None:
    assert lb.max_targets(lb.LANE_GEMM, 1000, target_costs_sec=(900, 50, 50)) == 3


def test_gemm_without_estimates_uses_its_default_cost() -> None:
    assert lb.max_targets(lb.LANE_GEMM, lb.GEMM_DEFAULT_TARGET_SEC * 3) == 3


@pytest.mark.parametrize("cost", [0, -5, None, "x", True])
def test_gemm_treats_an_unusable_estimate_as_the_default(cost: Any) -> None:
    assert lb.max_targets(lb.LANE_GEMM, lb.GEMM_DEFAULT_TARGET_SEC, target_costs_sec=(cost,)) == 1


def test_fusion_is_capped_by_count_not_by_division() -> None:
    """It has neither an admission floor nor a per-recipe estimate."""
    assert lb.max_targets(lb.LANE_FUSION, 10) == lb.FUSION_MAX_TARGETS
    assert lb.max_targets(lb.LANE_FUSION, 10_000) == lb.FUSION_MAX_TARGETS


def test_fusion_with_no_budget_funds_nothing() -> None:
    assert lb.max_targets(lb.LANE_FUSION, 0) == 0


@pytest.mark.parametrize("budget", [-1, True, 1.5, "100"])
def test_an_unusable_lane_budget_is_refused(budget: Any) -> None:
    with pytest.raises(lb.LaneBudgetError, match="non-negative int"):
        lb.max_targets(lb.LANE_REWRITE, budget)


def test_max_targets_refuses_an_unknown_lane() -> None:
    with pytest.raises(lb.LaneBudgetError, match="unknown lane"):
        lb.max_targets("collective", 100)


def test_allocate_covers_every_lane_in_one_pass() -> None:
    allocations = lb.allocate(180.0)
    assert set(allocations) == set(lb.DEFAULT_LANE_WEIGHTS)
    for lane, allocation in allocations.items():
        assert allocation.lane == lane
        assert allocation.budget_sec > 0


def test_allocate_on_a_three_hour_phase_funds_rewrite_but_not_generously() -> None:
    """The documented shape: a 3h session leaves rewrite one or two kernels."""
    allocations = lb.allocate(158.0)
    rewrite = allocations[lb.LANE_REWRITE]
    assert rewrite.max_targets in {0, 1}
    assert rewrite.is_fundable == (rewrite.max_targets > 0)


def test_allocate_on_an_unbounded_session_funds_nothing() -> None:
    allocations = lb.allocate(None)
    assert all(allocation.max_targets == 0 for allocation in allocations.values())
    assert all(not allocation.is_fundable for allocation in allocations.values())


def test_allocate_passes_router_estimates_to_gemm() -> None:
    allocations = lb.allocate(180.0, weights={lb.LANE_GEMM: 1.0}, gemm_target_costs_sec=(900, 900))
    assert allocations[lb.LANE_GEMM].max_targets == 2


def test_a_gemm_tuner_is_capped_strictly_below_the_session() -> None:
    """Passing the same value to both made the producer's own min() an identity."""
    per_tuner = lb.gemm_per_tuner_timeout_sec(10_000)
    assert 0 < per_tuner < 10_000


@pytest.mark.parametrize("total", [2, 3, 10, 100, 10_000])
def test_the_cap_stays_below_any_usable_session(total: int) -> None:
    assert 0 < lb.gemm_per_tuner_timeout_sec(total) < total


def test_a_one_second_session_is_degenerate_and_yields_one_second() -> None:
    """No integer is both below one and usable, so the invariant cannot hold."""
    assert lb.gemm_per_tuner_timeout_sec(1) == 1


@pytest.mark.parametrize("value", [0, -1, None, "x", True])
def test_an_unbounded_gemm_session_passes_through(value: Any) -> None:
    assert lb.gemm_per_tuner_timeout_sec(value) == 0


def test_the_gemm_cap_follows_its_share() -> None:
    assert lb.gemm_per_tuner_timeout_sec(1000) == int(1000 * lb.GEMM_PER_TUNER_SHARE)
