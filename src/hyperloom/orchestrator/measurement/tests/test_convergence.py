# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the throughput convergence judge.

The numbers below are the real ones. ``_RCA_ROUNDS`` is the three-round series
that started this (58% spread, monotonically rising). ``_T6_ROUNDS`` is the
five-pass controlled repeat that showed the spread is cold start, not noise:
117.6% across all five, 3.9% once the first is dropped.
"""

from __future__ import annotations

from hyperloom.orchestrator.measurement.convergence import (
    assess_convergence,
)

# warmup / measure / accuracy from a real session.
_RCA_ROUNDS = [14202.70, 19373.98, 22424.80]

# Five identical benchmark passes against one resident server (req/s).
_T6_ROUNDS = [63.90, 133.85, 139.04, 137.29, 117.13]


class TestTheSeriesThatStartedThis:
    def test_rca_series_is_not_converged(self):
        verdict = assess_convergence(_RCA_ROUNDS)
        assert verdict.converged is False
        assert verdict.value is None

    def test_rca_series_is_rejected_on_spread(self):
        # Dropping the warm-up round leaves 19373.98 -> 22424.80: a 15.8% gap,
        # five times the tolerance. Taking the last round here is what turned a
        # warm-up climb into a reported "gain".
        verdict = assess_convergence(_RCA_ROUNDS)
        assert verdict.reason == "spread_exceeds_tolerance"
        assert verdict.spread_pct > 15.0

    def test_two_rounds_are_not_called_a_trend(self):
        # A rising pair is not evidence of warm-up: half of all steady pairs
        # rise. The trend verdict needs three points.
        assert assess_convergence(_RCA_ROUNDS).monotonic is False

    def test_warmup_round_is_discarded_not_averaged(self):
        verdict = assess_convergence(_RCA_ROUNDS)
        assert verdict.discarded == [14202.70]
        assert 14202.70 not in verdict.used


class TestT6ControlledRepeat:
    def test_all_five_rounds_would_look_catastrophic(self):
        # Keeping the cold round: >100% spread.
        verdict = assess_convergence(_T6_ROUNDS, warmup_rounds=0)
        assert verdict.converged is False
        assert verdict.spread_pct > 100.0

    def test_dropping_the_cold_round_still_fails_on_the_shared_box(self):
        # Rounds 2-5 include the round-5 dip caused by another workload landing
        # on the shared machine, so this is correctly NOT converged -- the fix
        # for that is paired measurement, not a looser threshold.
        verdict = assess_convergence(_T6_ROUNDS)
        assert verdict.converged is False
        assert verdict.reason == "spread_exceeds_tolerance"

    def test_rounds_two_to_four_are_converged(self):
        # The steady window: 3.9% spread, inside the 3%-order tolerance once the
        # dip is excluded.
        verdict = assess_convergence(_T6_ROUNDS[1:4], warmup_rounds=0, tolerance_pct=5.0)
        assert verdict.converged is True
        assert verdict.spread_pct < 4.0
        assert verdict.value == sum(_T6_ROUNDS[1:4]) / 3


class TestVerdicts:
    def test_steady_series_converges_to_the_mean_of_used_rounds(self):
        verdict = assess_convergence([50.0, 100.0, 101.0, 100.5])
        assert verdict.converged is True
        assert verdict.discarded == [50.0]
        assert verdict.value == (100.0 + 101.0 + 100.5) / 3

    def test_single_usable_round_cannot_be_judged(self):
        verdict = assess_convergence([50.0, 100.0])
        assert verdict.converged is False
        assert verdict.reason == "insufficient_rounds"

    def test_empty_series(self):
        verdict = assess_convergence([])
        assert verdict.converged is False and verdict.reason == "no_measurements"

    def test_non_positive_rounds_are_ignored(self):
        verdict = assess_convergence([0.0, -1.0, 100.0, 100.5, 100.2])
        assert verdict.converged is True
        assert 0.0 not in verdict.used and -1.0 not in verdict.used

    def test_tight_but_still_climbing_series_is_rejected(self):
        # Where the trend rule earns its keep: the spread is inside tolerance,
        # so only the monotonic check catches that this is still warming up.
        verdict = assess_convergence([50.0, 100.0, 101.0, 102.0])
        assert verdict.converged is False
        assert verdict.reason == "monotonic_increasing"
        assert verdict.spread_pct < 3.0

    def test_decreasing_series_is_allowed_when_tight(self):
        # Only *increasing* series indicate an unfinished warm-up; a tight
        # decreasing one is just noise around steady state.
        verdict = assess_convergence([50.0, 100.5, 100.2, 100.0])
        assert verdict.converged is True

    def test_verdict_reports_every_round_for_audit(self):
        d = assess_convergence(_RCA_ROUNDS).to_dict()
        assert d["rounds_discarded"] == [14202.70]
        assert d["rounds_used"] == [19373.98, 22424.80]
        assert d["converged"] is False
        assert d["spread_pct"] > 15.0


