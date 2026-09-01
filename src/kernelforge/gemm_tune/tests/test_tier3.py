# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the generated-tuner tier.

Three pieces, each guarding something that produced a wrong answer on real
hardware before it existed:

* ``coverage`` decides whether a generated tuner has a target at all, which was
  previously settled by argument rather than by the fleet;
* ``contract`` rejects output whose own numbers contradict each other;
* ``referee`` is what makes the whole tier safe -- the generated tuner proposes,
  and only these timings decide.
"""

from __future__ import annotations

import json

import pytest

from kernelforge.gemm_tune.router import TunerSpec
from kernelforge.gemm_tune.tier3 import (
    build_mandate,
    contract,
    coverage_gaps,
    judge_candidates,
    time_paired,
    validate_output_csv,
)
from kernelforge.gemm_tune.tier3.coverage import CoverageGap


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
        assert gap.warrants_generated_tuner

    def test_a_tuner_that_exists_but_was_not_selected_is_a_routing_gap(self):
        # A real vLLM log missed 122 bf16 keys while sglang_dense_bf16 -- the
        # tuner that owns that exact table -- simply was not selected by the
        # framework branch. The answer is to route better, not to write a tuner,
        # and conflating the two manufactures demand for the generated tier.
        (gap,) = coverage_gaps(_demand(tuner="sglang_dense_bf16"), [TunerSpec("a8w8")])
        assert gap.kind == "not_selected"
        assert not gap.warrants_generated_tuner

    def test_a_tuner_that_declined_for_a_reason_of_its_own_is_not_tier3_work(self):
        specs = [TunerSpec("fmoe_ck", skip_reason="the tuner script is missing")]
        (gap,) = coverage_gaps(_demand(tuner="fmoe_ck"), specs)
        assert gap.kind == "skipped"
        assert not gap.warrants_generated_tuner

    def test_only_a_missing_capability_reaches_the_generated_tier(self):
        report = {
            "demands": [
                {"table": "none.csv", "tuner": None, "miss_count": 5},
                {"table": "unrouted.csv", "tuner": "a8w8", "miss_count": 9},
                {"table": "declined.csv", "tuner": "fmoe_ck", "miss_count": 7},
            ]
        }
        specs = [TunerSpec("fmoe_ck", skip_reason="script is missing")]
        gaps = coverage_gaps(report, specs)
        assert [g.table for g in gaps if g.warrants_generated_tuner] == ["none.csv"]

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


class TestMandate:
    def _mandate(self):
        gap = CoverageGap(
            table="odd_tuned_gemm.csv",
            tuner=None,
            env_var="AITER_CONFIG_ODD",
            key_schema=["M", "N", "K"],
            miss_count=40,
            reason="no tuner is registered for odd_tuned_gemm.csv",
        )
        return build_mandate(
            gap,
            [{"M": 16, "N": 1536, "K": 7168}, {"M": 1024, "N": 1536, "K": 7168}],
            gpu="MI355X (gfx950)",
            framework="sglang",
        )

    def test_columns_are_keys_then_search_then_timings(self):
        assert self._mandate().output_columns == [
            "M",
            "N",
            "K",
            "backend",
            "config",
            "default_us",
            "tuned_us",
            "improved",
        ]

    def test_the_brief_carries_the_constraints_that_were_learned_the_hard_way(self):
        text = self._mandate().render()
        # A single correctness check passes an intermittently wrong kernel at
        # random; a Python-loop timer cannot rank kernels this small.
        assert "8 times" in text or "{} times".format(8) in text
        assert "fresh inputs" in text
        assert "captured graph" in text
        assert "12us" in text
        # And that its own numbers do not decide anything.
        assert "informational" in text

    def test_the_brief_names_the_shapes_and_the_reason(self):
        text = self._mandate().render()
        assert "M=16, N=1536, K=7168" in text
        assert "no tuner is registered" in text

    def test_round_trips_as_json(self):
        d = self._mandate().to_dict()
        assert json.loads(json.dumps(d))["table"] == "odd_tuned_gemm.csv"
        assert d["correctness_trials"] == 8

    def test_the_brief_says_how_the_error_is_measured(self):
        # Left to interpretation, the obvious element-wise ratio makes any
        # output element near zero dominate -- and by that measure the
        # unmodified torch.matmul scores 1.375, so the gate rejects the
        # default path. The rule is unusable without its definition.
        text = self._mandate().render()
        assert "mean|ref|" in text
        assert "1.375" in text, "the reason has to travel with the rule"

    def test_the_definition_reaches_the_machine_readable_form(self):
        assert "mean|ref|" in self._mandate().to_dict()["max_relative_error_definition"]


class TestContract:
    def _mandate(self):
        gap = CoverageGap(table="t.csv", tuner=None, key_schema=["M", "N", "K"])
        return build_mandate(gap, [{"M": 16, "N": 1536, "K": 7168}])

    def _write(self, tmp_path, header, rows):
        p = tmp_path / "out.csv"
        p.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
        return p

    _HDR = "M,N,K,backend,config,default_us,tuned_us,improved"

    def test_a_good_file_passes(self, tmp_path):
        p = self._write(tmp_path, self._HDR, ["16,1536,7168,hipblaslt,solidx=1,11.1,8.2,True"])
        assert validate_output_csv(p, self._mandate()) == []

    def test_a_wrong_header_is_named(self, tmp_path):
        p = self._write(tmp_path, "M,N,K,us", ["16,1536,7168,8.2"])
        (v, *_) = validate_output_csv(p, self._mandate())
        assert v.where == "header"

    def test_improved_must_agree_with_its_own_numbers(self, tmp_path):
        # The cheapest possible tell that a script is not measuring what it
        # reports.
        p = self._write(tmp_path, self._HDR, ["16,1536,7168,x,c=1,8.0,11.0,True"])
        problems = [str(v) for v in validate_output_csv(p, self._mandate())]
        assert any("contradicts" in s for s in problems)

    def test_a_missing_demanded_shape_is_reported(self, tmp_path):
        p = self._write(tmp_path, self._HDR, ["32,1536,7168,x,c=1,11.1,8.2,True"])
        problems = [str(v) for v in validate_output_csv(p, self._mandate())]
        assert any("have no row" in s for s in problems)

    def test_non_positive_and_non_numeric_times_are_rejected(self, tmp_path):
        p = self._write(
            tmp_path,
            self._HDR,
            [
                "16,1536,7168,x,c=1,0,8.2,True",
                "17,1536,7168,x,c=1,abc,8.2,True",
            ],
        )
        problems = [str(v) for v in validate_output_csv(p, self._mandate())]
        assert any("not a positive time" in s for s in problems)
        assert any("not a number" in s for s in problems)

    def test_a_comma_in_config_would_break_the_csv(self, tmp_path):
        p = self._write(tmp_path, self._HDR, ['16,1536,7168,x,"a,b",11.1,8.2,True'])
        problems = [str(v) for v in validate_output_csv(p, self._mandate())]
        assert any("comma" in s for s in problems)

    def test_duplicate_shapes_are_reported(self, tmp_path):
        row = "16,1536,7168,x,c=1,11.1,8.2,True"
        p = self._write(tmp_path, self._HDR, [row, row])
        problems = [str(v) for v in validate_output_csv(p, self._mandate())]
        assert any("duplicate" in s for s in problems)

    def test_missing_and_empty_files(self, tmp_path):
        assert validate_output_csv(tmp_path / "nope.csv", self._mandate())
        p = self._write(tmp_path, self._HDR, [])
        assert any("no rows" in str(v) for v in validate_output_csv(p, self._mandate()))

    def test_candidates_are_capped_and_sanitised(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(
            json.dumps(
                {
                    "16x1536x7168": [{"backend": "a"}] * 9,
                    "bad": "not a list",
                    "one": {"backend": "solo"},
                }
            ),
            encoding="utf-8",
        )
        out = contract.load_candidates(p, self._mandate())
        assert len(out["16x1536x7168"]) == 5
        assert out["one"] == [{"backend": "solo"}]
        assert "bad" not in out

    def test_unreadable_candidates_yield_nothing(self, tmp_path):
        assert contract.load_candidates(tmp_path / "nope.json", self._mandate()) == {}


class _Clock:
    """A deterministic stand-in for a device, so the protocol itself is testable."""

    def __init__(self, costs):
        self.costs = list(costs)
        self.now = 0.0
        self.i = 0

    def call(self, cost):
        def _fn():
            self.now += cost() if callable(cost) else cost

        return _fn

    def perf_counter(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock([])
    monkeypatch.setattr("kernelforge.gemm_tune.tier3.referee.time.perf_counter", c.perf_counter)
    return c


class TestReferee:
    def test_a_faster_candidate_is_reported_as_faster(self, clock):
        t = time_paired(clock.call(2e-6), clock.call(1e-6), repeats=3)
        assert t.usable and t.speedup == pytest.approx(2.0)

    def test_a_slower_candidate_is_reported_as_slower(self, clock):
        t = time_paired(clock.call(1e-6), clock.call(2e-6), repeats=3)
        assert t.usable and t.speedup == pytest.approx(0.5)

    def test_interference_on_one_side_only_is_refused(self, clock):
        # One clean baseline window and four disturbed ones. The best case then
        # says the baseline is faster and the typical case says the candidate
        # is; the two sides were not measured under one machine state, so there
        # is no number to report.
        from kernelforge.gemm_tune.tier3 import referee

        calls = {"n": 0}

        def noisy_baseline():
            calls["n"] += 1
            # Warmup runs first and is not measured; the first *sampled* window
            # is the quiet one.
            if calls["n"] <= referee.WARMUP_CALLS:
                return 1e-6
            sample = (calls["n"] - referee.WARMUP_CALLS - 1) // referee.CALLS_PER_SAMPLE
            return 1e-6 if sample == 0 else 9e-6

        t = time_paired(clock.call(noisy_baseline), clock.call(2e-6), repeats=5)
        assert not t.usable
        assert "unstable" in t.reason and "contradicts" in t.reason

    def test_a_candidate_that_raises_is_data_not_a_crash(self, clock):
        def boom():
            raise RuntimeError("kernel refused this shape")

        t = time_paired(clock.call(1e-6), boom, repeats=3)
        assert not t.usable and "RuntimeError" in t.reason

    def test_the_generated_tuners_own_numbers_are_never_consulted(self, clock):
        # Its claim is 100x; ours is what the clock says.
        cands = [{"config": "a", "tuned_us": 0.01, "self_reported_speedup": 100.0}]
        j = judge_candidates(
            "16x1536x7168",
            cands,
            baseline=clock.call(2e-6),
            dispatch=lambda c: clock.call(1e-6),
        )
        assert j.improved
        assert j.best_timing.speedup == pytest.approx(2.0)

    def test_an_incorrect_candidate_is_rejected_before_being_timed(self, clock):
        cands = [{"config": "wrong"}, {"config": "right"}]
        j = judge_candidates(
            "s",
            cands,
            baseline=clock.call(2e-6),
            dispatch=lambda c: clock.call(1e-6),
            is_correct=lambda call: True,
        )
        assert j.rejected_incorrect == 0

        j2 = judge_candidates(
            "s",
            cands,
            baseline=clock.call(2e-6),
            dispatch=lambda c: clock.call(1e-6),
            is_correct=lambda call: False,
        )
        assert j2.rejected_incorrect == 2
        assert j2.best is None and not j2.improved

    def test_an_undispatchable_candidate_is_recorded_not_dropped(self, clock):
        j = judge_candidates(
            "s",
            [{"backend": "unknown"}],
            baseline=clock.call(1e-6),
            dispatch=lambda c: None,
        )
        assert j.best is None
        assert j.timings[0][1].reason == "not dispatchable"

    def test_the_fastest_of_several_wins(self, clock):
        costs = {"a": 4e-6, "b": 1e-6, "c": 2e-6}
        j = judge_candidates(
            "s",
            [{"n": k} for k in costs],
            baseline=clock.call(8e-6),
            dispatch=lambda c: clock.call(costs[c["n"]]),
        )
        assert j.best == {"n": "b"}
        assert j.improved

    def test_no_improvement_is_said_plainly(self, clock):
        j = judge_candidates(
            "s",
            [{"n": "a"}],
            baseline=clock.call(1e-6),
            dispatch=lambda c: clock.call(4e-6),
        )
        assert not j.improved
        assert j.best_timing.speedup == pytest.approx(0.25)

    def test_the_judgement_serialises(self, clock):
        j = judge_candidates(
            "s",
            [{"n": "a"}],
            baseline=clock.call(2e-6),
            dispatch=lambda c: clock.call(1e-6),
        )
        assert json.loads(json.dumps(j.to_dict()))["improved"] is True


@pytest.mark.parametrize("bad", ["", "16x1536", "16x1536x7168x4", "16xNx7168", "not-a-shape"])
def test_shape_key_rejects_a_shape_it_cannot_turn_into_m_n_k(bad: str) -> None:
    """It used to answer ``()`` here, which helped nobody.

    Every caller unpacks the result into three names, so the empty tuple only
    moved the failure a few frames out and stripped the shape from the message
    -- and ``"16x1536"`` did not even fail here, it returned a 2-tuple that blew
    up the same way. The adapter's caller already treats a raised error as
    "tier3 attempt failed; tuning continues", so this loses no tolerance.
    """
    from kernelforge.gemm_tune.tier3.dispatch import shape_key

    with pytest.raises(ValueError, match="MxNxK"):
        shape_key(bad)


def test_shape_key_parses_the_well_formed_case() -> None:
    from kernelforge.gemm_tune.tier3.dispatch import shape_key

    assert shape_key("16x1536x7168") == (16, 1536, 7168)
