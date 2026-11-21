# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A campaign's score is the arithmetic mean of its per-case speedups unless it declares the geometric one."""

from __future__ import annotations

import pytest

from kernelforge.evaluation.benchmark.runner import (
    calculate_mean_case_speedup,
    calculate_measurement_case_speedups,
)
from kernelforge.evaluation.scoring import CaseAggregate, attribute_sigma

# One case four times faster, one four times slower.
_BASELINE = {"won": 1.0, "lost": 1.0}
_CANDIDATE = {"won": 0.25, "lost": 4.0}


def test_the_score_is_the_arithmetic_mean_unless_an_aggregate_is_named() -> None:
    assert calculate_mean_case_speedup(_CANDIDATE, _BASELINE) == pytest.approx((4.0 + 0.25) / 2)
    assert calculate_mean_case_speedup(
        _CANDIDATE, _BASELINE, aggregate=CaseAggregate.ARITHMETIC
    ) == calculate_mean_case_speedup(_CANDIDATE, _BASELINE)


def test_a_geometric_score_counts_a_gain_by_its_ratio() -> None:
    """A 4x on one case and a 0.25x on another cancel, where the arithmetic mean reports a 2.125x win."""
    score = calculate_mean_case_speedup(_CANDIDATE, _BASELINE, aggregate=CaseAggregate.GEOMETRIC)

    assert score == pytest.approx(1.0)


def test_an_unscored_case_is_left_out_of_either_aggregate() -> None:
    baseline = {**_BASELINE, "unscored": 1.0}
    candidate = {**_CANDIDATE, "unscored": 0.001}

    for aggregate in CaseAggregate:
        assert calculate_mean_case_speedup(candidate, baseline, ["unscored"], aggregate) == pytest.approx(
            calculate_mean_case_speedup(_CANDIDATE, _BASELINE, aggregate=aggregate)
        )


def test_every_independent_measurement_is_scored_under_the_named_aggregate() -> None:
    benchmark = {
        "success": True,
        "measurements": [
            {"success": True, "case_times": {"won": 0.25, "lost": 1.0}, "unscored_cases": []},
            {"success": True, "case_times": {"won": 0.25, "lost": 4.0}, "unscored_cases": []},
        ],
    }

    arithmetic = calculate_measurement_case_speedups(benchmark, _BASELINE, expected_measurements=2)
    geometric = calculate_measurement_case_speedups(
        benchmark, _BASELINE, expected_measurements=2, aggregate=CaseAggregate.GEOMETRIC
    )

    assert arithmetic == pytest.approx([2.5, 2.125])
    assert geometric == pytest.approx([2.0, 1.0])


def test_the_formula_an_agent_is_shown_names_the_aggregate() -> None:
    assert CaseAggregate.ARITHMETIC.formula == "mean(pristine_case_ms / candidate_case_ms)"
    assert CaseAggregate.GEOMETRIC.formula == "geomean(pristine_case_ms / candidate_case_ms)"


def test_a_geometric_score_blames_its_spread_on_the_case_with_the_larger_relative_noise() -> None:
    """A 10x case wobbling by 1% moves an arithmetic mean more than a 1x case wobbling by 5%; a geometric mean less."""
    baseline = {"fast": 1.0, "steady": 1.0}
    series = {
        "fast": [0.1 / factor for factor in (0.99, 1.0, 1.01)],
        "steady": [1.0 / factor for factor in (0.95, 1.0, 1.05)],
    }

    arithmetic = attribute_sigma(series, baseline)
    geometric = attribute_sigma(series, baseline, CaseAggregate.GEOMETRIC)

    assert arithmetic.variance_shares["fast"] == pytest.approx(0.8, abs=0.01)
    assert geometric.variance_shares["steady"] == pytest.approx(25 / 26, abs=0.01)
