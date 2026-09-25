# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Reading the one file a ceiling run produces, and refusing one that cannot be read."""

from __future__ import annotations

import pytest

from kernelforge.roofline_ceiling.contract import CeilingContractError, load_report


def test_a_well_formed_file_is_taken_as_given():
    report = load_report({"cases": {"m_1": 0.0016, "m_4096": 0.0097}, "mean_ideal_ms": 0.00565})

    assert report.ideal_ms() == {"m_1": 0.0016, "m_4096": 0.0097}


def test_the_mean_is_recomputed_so_the_file_cannot_disagree_with_itself():
    report = load_report({"cases": {"a": 1.0, "b": 9.0}, "mean_ideal_ms": 999.0})

    assert report.mean_ideal_ms() == pytest.approx(5.0)


def test_the_mean_is_equal_weight_because_that_is_how_the_suite_is_scored():
    report = load_report({"cases": {"small": 0.001, "large": 10.0}})

    assert report.mean_ideal_ms() == pytest.approx(5.0005)


def test_a_report_with_no_cases_has_no_mean_rather_than_a_zero():
    report = load_report({"cases": {"a": 1.0}})
    empty = type(report)(cases=())

    assert empty.mean_ideal_ms() is None


def test_one_case_can_be_looked_up_by_id():
    report = load_report({"cases": {"a": 1.0, "b": 2.0}})

    assert report.case("b").t_ideal_ms == 2.0
    assert report.case("missing") is None


# --- what is refused -----------------------------------------------------------


def test_a_file_that_is_not_an_object_is_refused():
    with pytest.raises(CeilingContractError, match="JSON object"):
        load_report(["not", "an", "object"])


def test_a_file_with_no_cases_is_refused():
    with pytest.raises(CeilingContractError, match="non-empty 'cases'"):
        load_report({"mean_ideal_ms": 1.0})


def test_an_empty_cases_object_is_refused():
    with pytest.raises(CeilingContractError, match="non-empty 'cases'"):
        load_report({"cases": {}})


def test_cases_as_an_array_is_refused_because_it_names_no_shapes():
    with pytest.raises(CeilingContractError, match="non-empty 'cases'"):
        load_report({"cases": [0.1, 0.2]})


def test_a_latency_that_is_not_a_number_is_refused():
    with pytest.raises(CeilingContractError, match="finite positive number"):
        load_report({"cases": {"a": "fast"}})


def test_a_non_positive_latency_is_refused():
    for latency in (0.0, -1.0):
        with pytest.raises(CeilingContractError, match="finite positive number"):
            load_report({"cases": {"a": latency}})


def test_a_latency_that_is_not_finite_is_refused():
    """A NaN would poison every attainment it is divided into."""
    for latency in (float("nan"), float("inf")):
        with pytest.raises(CeilingContractError, match="finite positive number"):
            load_report({"cases": {"a": latency}})


def test_the_refusal_names_the_case_so_it_can_be_rewritten():
    with pytest.raises(CeilingContractError, match="'broken'"):
        load_report({"cases": {"fine": 1.0, "broken": None}})
