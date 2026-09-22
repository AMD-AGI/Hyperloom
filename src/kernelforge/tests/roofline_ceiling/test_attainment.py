# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Attainment: the ratio, the aggregate, and every case it refuses to score."""

from __future__ import annotations

import pytest

from kernelforge.roofline_ceiling.attainment import measure_attainment
from kernelforge.roofline_ceiling.contract import Hardware, build_report
from kernelforge.roofline_ceiling.specs import PEAK_SOURCE_REFERENCE


def _report(cases):
    payload = {
        "cases": [{"case_id": case_id, "t_ideal_ms": ideal, "bound": "memory"} for case_id, ideal in cases],
        "confidence": "high",
        "analysis_md": "\n".join(f"Case `{case_id}` at {ideal} ms." for case_id, ideal in cases),
    }
    return build_report(
        payload,
        canonical_id="roofline-ceiling:op:gfx950",
        hardware=Hardware(
            arch="gfx950",
            peak_flops={"bf16_mfma": 1.2e15},
            bandwidth={"hbm": 6.24e12},
            peak_source=PEAK_SOURCE_REFERENCE,
            dispatch_floor_s=1.5e-6,
        ),
        expected_case_ids=[case_id for case_id, _ in cases],
    )


def test_attainment_is_the_ceiling_over_the_measured_latency():
    standing = measure_attainment(_report([("a", 5.0)]), {"a": 10.0})

    assert standing.mean == pytest.approx(0.5)
    assert standing.cases[0].attainment == pytest.approx(0.5)
    assert standing.cases[0].remaining_speedup == pytest.approx(2.0)


def test_a_kernel_at_its_ceiling_reads_exactly_one():
    standing = measure_attainment(_report([("a", 5.0)]), {"a": 5.0})

    assert standing.mean == pytest.approx(1.0)
    assert standing.excluded == {}


def test_the_aggregate_weights_every_case_the_same():
    """Matching the KEEP objective, so a stop rule and a KEEP rule agree."""
    standing = measure_attainment(_report([("small", 1.0), ("large", 90.0)]), {"small": 2.0, "large": 100.0})

    # 50% and 90%; a latency-weighted mean would read close to 90%.
    assert standing.mean == pytest.approx(0.7)


def test_a_ceiling_above_the_measured_latency_is_refused_with_its_reason():
    standing = measure_attainment(_report([("a", 20.0)]), {"a": 10.0})

    assert standing.mean is None
    assert not standing.usable
    assert "understates the minimum legal work" in standing.excluded["a"]


def test_a_broken_case_does_not_drag_the_others_up():
    standing = measure_attainment(_report([("good", 9.0), ("broken", 40.0)]), {"good": 10.0, "broken": 20.0})

    assert standing.mean == pytest.approx(0.9)
    assert set(standing.excluded) == {"broken"}


def test_an_unmeasured_case_is_named_rather_than_assumed():
    standing = measure_attainment(_report([("a", 5.0), ("b", 5.0)]), {"a": 10.0})

    assert standing.mean == pytest.approx(0.5)
    assert "no measured latency" in standing.excluded["b"]


def test_a_case_the_objective_does_not_score_is_left_out():
    standing = measure_attainment(
        _report([("a", 5.0), ("check", 5.0)]),
        {"a": 10.0, "check": 500.0},
        unscored_cases=["check"],
    )

    assert standing.mean == pytest.approx(0.5)
    assert "does not score it" in standing.excluded["check"]


def test_a_non_positive_or_absurd_latency_is_refused():
    for measured in (0.0, -1.0, float("nan"), float("inf"), "slow", None):
        standing = measure_attainment(_report([("a", 5.0)]), {"a": measured})
        assert standing.mean is None, measured


def test_coverage_is_what_lets_a_caller_act_on_the_mean():
    standing = measure_attainment(_report([("a", 5.0), ("b", 5.0)]), {"a": 10.0})

    assert standing.usable
    # Usable and yet not an answer about this case set: `b` is missing.
    assert not standing.covers(["a", "b"])
    assert standing.covers(["a"])


def test_coverage_of_nothing_is_never_satisfied():
    """An empty case set must not read as a covered one."""
    standing = measure_attainment(_report([("a", 5.0)]), {"a": 10.0})

    assert not standing.covers([])


def test_the_cases_short_of_a_target_come_back_worst_first():
    standing = measure_attainment(
        _report([("a", 9.0), ("b", 4.0), ("c", 9.5)]),
        {"a": 10.0, "b": 10.0, "c": 10.0},
    )

    short = standing.below(0.95)

    assert [entry.case_id for entry in short] == ["b", "a"]


def test_no_cases_means_no_figure_and_no_crash():
    standing = measure_attainment(_report([("a", 5.0)]), {})

    assert standing.mean is None
    assert standing.cases == ()
