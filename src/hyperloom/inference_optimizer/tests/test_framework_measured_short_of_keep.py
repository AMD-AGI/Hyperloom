# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A config already measured short of KEEP should not be measured again -- but a round that never measured is not that."""

from __future__ import annotations

from hyperloom.orchestrator.phases.framework import _measured_short_of_keep


def _rec(outcome: str, tps: float) -> dict[str, object]:
    return {"outcome": outcome, "tps_delta_pct": tps, "applicability": "fp"}


def test_a_measured_result_under_the_bar_is_remembered_as_settled():
    assert _measured_short_of_keep(_rec("integrated", 0.4), 1.0) is True
    assert _measured_short_of_keep(_rec("integrated", -3.0), 1.0) is True


def test_a_measured_result_at_or_over_the_bar_is_not_blocked():
    assert _measured_short_of_keep(_rec("integrated", 1.0), 1.0) is False
    assert _measured_short_of_keep(_rec("integrated", 9.5), 1.0) is False


def test_a_round_that_never_reached_a_measurement_settles_nothing():
    """The GLM-5.3 prefix-cache rounds died before the server served; the idea was worth 94% of the run's prefill.

    Their `tps_delta_pct` is the zero the record defaults to. Read as a result,
    a correct direction would be retired permanently on evidence that does not
    exist.
    """
    for outcome in (
        "rejected_apply_fail",
        "reverted_smoke_fail",
        "reverted_parity_inconclusive",
        "reverted_switch_off_parity",
        "already_present",
        "",
    ):
        assert _measured_short_of_keep(_rec(outcome, 0.0), 1.0) is False, outcome


def test_a_decayed_bar_lets_a_small_gain_back_in():
    """The KEEP bar decays per macro cycle, so a result short of it in cycle 1 can clear it later."""
    record = _rec("integrated", 0.8)

    assert _measured_short_of_keep(record, 2.0) is True
    assert _measured_short_of_keep(record, 0.5) is False


def test_the_outcome_decides_before_the_number_does():
    """The writer always stores a number, so an integrated record without one measured zero."""
    assert _measured_short_of_keep({"outcome": "integrated"}, 1.0) is True
    # No outcome at all is not a measurement, whatever else the record carries.
    assert _measured_short_of_keep({"tps_delta_pct": -50.0}, 1.0) is False
