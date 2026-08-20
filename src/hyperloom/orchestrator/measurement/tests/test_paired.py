# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the paired A/B judge."""

from __future__ import annotations

from hyperloom.orchestrator.measurement.paired import (
    assess_paired,
    interleaved_plan,
)


class TestPlan:
    def test_order_alternates(self):
        assert interleaved_plan(3) == ["A", "B", "A", "B", "A", "B"]

    def test_zero_and_negative_are_empty(self):
        assert interleaved_plan(0) == [] and interleaved_plan(-1) == []


class TestVerdicts:
    def test_consistent_win_is_decisive(self):
        v = assess_paired([(100.0, 110.0), (102.0, 112.0), (99.0, 109.0)])
        assert v.decisive and v.candidate_wins
        assert v.reason == "candidate_faster"
        assert v.median_delta_pct > 3.0

    def test_consistent_loss_is_decisive(self):
        v = assess_paired([(100.0, 90.0), (101.0, 91.0)])
        assert v.decisive and v.reason == "candidate_slower"
        assert not v.candidate_wins

    def test_small_consistent_difference_is_within_noise(self):
        v = assess_paired([(100.0, 101.0), (100.0, 101.5)])
        assert v.decisive and v.reason == "within_noise"
        assert not v.candidate_wins

    def test_sign_disagreement_is_inconclusive(self):
        # One pair says the candidate is faster, another says slower. The
        # machine moved more than the change did; averaging would invent a
        # winner out of that.
        v = assess_paired([(100.0, 115.0), (100.0, 88.0)])
        assert not v.decisive and v.reason == "sign_disagreement"
        assert not v.candidate_wins

    def test_single_pair_is_not_a_comparison(self):
        v = assess_paired([(100.0, 130.0)])
        assert not v.decisive and v.reason == "insufficient_pairs"

    def test_empty(self):
        assert assess_paired([]).reason == "insufficient_pairs"


class TestMedianNotMean:
    def test_one_disturbed_pair_does_not_carry_the_result(self):
        # Three pairs agree on ~+4%; a fourth is wrecked by a neighbour landing
        # on the box. The mean would be dragged far off; the median holds.
        pairs = [(100.0, 104.0), (100.0, 104.5), (100.0, 103.5), (100.0, 160.0)]
        v = assess_paired(pairs)
        assert v.decisive and v.candidate_wins
        assert 3.5 < v.median_delta_pct < 6.0

    def test_deltas_are_reported_for_audit(self):
        v = assess_paired([(100.0, 110.0), (100.0, 120.0)])
        assert v.deltas_pct == [10.0, 20.0]
        assert v.to_dict()["pairs"] == [[100.0, 110.0], [100.0, 120.0]]


class TestBadInput:
    def test_non_positive_values_are_dropped(self):
        v = assess_paired([(0.0, 110.0), (100.0, 110.0), (100.0, 111.0)])
        assert len(v.pairs) == 2 and v.decisive

    def test_dropping_leaves_too_few(self):
        v = assess_paired([(0.0, 110.0), (100.0, -1.0)])
        assert v.reason == "insufficient_pairs"

    def test_threshold_is_configurable(self):
        pairs = [(100.0, 104.0), (100.0, 104.5)]
        assert assess_paired(pairs).reason == "candidate_faster"
        assert assess_paired(pairs, threshold_pct=10.0).reason == "within_noise"
