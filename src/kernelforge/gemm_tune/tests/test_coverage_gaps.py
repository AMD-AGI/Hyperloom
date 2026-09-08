# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for tuning-coverage gaps.

``coverage_gaps`` reports what the runtime asked for that no tuner wrote, and
why. The distinction it draws -- an absent capability versus a tuner that
existed and declined or was not routed to -- is the whole point: only the first
says anything about coverage, and lumping them together manufactures demand
that is not there.
"""

from __future__ import annotations

from kernelforge.gemm_tune.coverage import CoverageGap, coverage_gaps
from kernelforge.gemm_tune.router import TunerSpec


def _demand(table="odd_tuned_gemm.csv", tuner=None, misses=40, keys=7):
    return {
        "demands": [
            {
                "table": table,
                "tuner": tuner,
                "env_var": "AITER_CONFIG_ODD",
                "key_schema": ["M", "N", "K"],
                "logged_fields": ["M", "N", "K"],
                "miss_count": misses,
                "distinct_keys": keys,
            }
        ]
    }


class TestCoverageGaps:
    def test_a_table_no_tuner_owns_is_a_gap(self):
        (gap,) = coverage_gaps(_demand(tuner=None), [TunerSpec("a8w8")])
        assert gap.table == "odd_tuned_gemm.csv"
        assert "no tuner is registered" in gap.reason
        assert gap.miss_count == 40
        assert gap.kind == "no_tuner"

    def test_a_tuner_that_exists_but_was_not_selected_is_a_routing_gap(self):
        # A real vLLM log missed 122 bf16 keys while sglang_dense_bf16 -- the
        # tuner that owns that exact table -- simply was not selected by the
        # framework branch. Filing that as "no tuner owns this" would send the
        # reader looking for a tuner that is already written.
        (gap,) = coverage_gaps(_demand(tuner="sglang_dense_bf16"), [TunerSpec("a8w8")])
        assert gap.kind == "not_selected"

    def test_a_tuner_that_declined_for_a_reason_of_its_own_is_not_a_coverage_gap(self):
        specs = [TunerSpec("fmoe_ck", skip_reason="the tuner script is missing")]
        (gap,) = coverage_gaps(_demand(tuner="fmoe_ck"), specs)
        assert gap.kind == "skipped"

    def test_the_three_kinds_are_reported_apart(self):
        report = {
            "demands": [
                {"table": "none.csv", "tuner": None, "miss_count": 5},
                {"table": "unrouted.csv", "tuner": "a8w8", "miss_count": 9},
                {"table": "declined.csv", "tuner": "fmoe_ck", "miss_count": 7},
            ]
        }
        specs = [TunerSpec("fmoe_ck", skip_reason="script is missing")]
        gaps = coverage_gaps(report, specs)
        assert [g.table for g in gaps if g.kind == "no_tuner"] == ["none.csv"]

    def test_a_covered_table_is_not_a_gap(self):
        specs = [TunerSpec("sglang_dense_bf16")]
        assert coverage_gaps(_demand(tuner="sglang_dense_bf16"), specs) == []

    def test_a_skip_that_is_an_answer_is_not_a_gap(self):
        # The capability exists and said no. A generated tuner would not change
        # any of these, so calling them coverage gaps would manufacture demand.
        for reason in (
            "FP4 GEMM is not supported on gfx942",
            "No GEMM shapes available: needs --untuned-csv",
            "Model is not MoE; fmoe_ck tuner not applicable",
            "1-stage ASM kernels are already at peak performance",
            "moe_intermediate_size not set in model config",
        ):
            specs = [TunerSpec("fmoe_ck", skip_reason=reason)]
            assert coverage_gaps(_demand(tuner="fmoe_ck"), specs) == [], reason

    def test_a_skip_with_no_such_explanation_is_still_recorded(self):
        specs = [TunerSpec("fmoe_ck", skip_reason="the tuner script is missing")]
        (gap,) = coverage_gaps(_demand(tuner="fmoe_ck"), specs)
        assert "script is missing" in gap.reason

    def test_no_demand_means_nothing_is_missing(self):
        assert coverage_gaps(None, [TunerSpec("a8w8")]) == []
        assert coverage_gaps({"demands": []}, []) == []

    def test_gaps_are_ordered_by_how_much_was_asked_for(self):
        report = {
            "demands": [
                {"table": "small.csv", "tuner": None, "miss_count": 3},
                {"table": "big.csv", "tuner": None, "miss_count": 900},
            ]
        }
        assert [g.table for g in coverage_gaps(report, [])] == ["big.csv", "small.csv"]

