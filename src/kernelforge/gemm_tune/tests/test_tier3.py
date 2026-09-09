# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the generated-tuner tier."""

from __future__ import annotations

import json
from pathlib import Path

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
from kernelforge.gemm_tune.tuners.base import TuneResult


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
        # A real vLLM log missed 122 bf16 keys while sglang_dense_bf16 -- the tuner that owns that exact table --
        # simply was not selected by the framework branch.
        (gap,) = coverage_gaps(_demand(tuner="sglang_dense_bf16"), [TunerSpec("a8w8")])
        assert gap.kind == "not_selected"
        assert not gap.warrants_generated_tuner

    def test_a_tuner_that_declined_for_no_reason_of_substance_is_tier3_work(self):
        # "The script is missing" is not a property of the table, the hardware
        # or the checkpoint -- it is our own capability being absent on this
        # run, which is the same position a table with no owner is in. The
        # reasons that really are answers are excluded before a gap is built;
        # see test_a_skip_that_is_an_answer_is_not_a_gap.
        specs = [TunerSpec("fmoe_ck", skip_reason="the tuner script is missing")]
        (gap,) = coverage_gaps(_demand(tuner="fmoe_ck"), specs)
        assert gap.kind == "skipped"
        assert gap.warrants_generated_tuner

    def test_an_owner_that_ran_and_landed_nothing_is_tier3_work(self):
        # The distinction that matters is not whether a tuner exists but
        # whether the table got tuned. Selecting an owner that then hands back
        # nothing leaves the runtime missing every key it asked about.
        specs = [TunerSpec("sglang_dense_bf16")]
        demand = _demand(tuner="sglang_dense_bf16")
        assert coverage_gaps(demand, specs, results=None) == []
        empty = TuneResult(tuner_name="sglang_dense_bf16", status="no_improvement")
        (gap,) = coverage_gaps(demand, specs, results=[empty])
        assert gap.kind == "empty"
        assert gap.warrants_generated_tuner
        assert "produced nothing landable" in gap.reason

    def test_an_owner_that_landed_something_is_not_a_gap(self):
        specs = [TunerSpec("sglang_dense_bf16")]
        landed = TuneResult(
            tuner_name="sglang_dense_bf16",
            status="ok",
            artifact_path="/tmp/tuned.csv",
            improved_shapes=3,
            best_micro_speedup=1.4,
        )
        assert coverage_gaps(_demand(tuner="sglang_dense_bf16"), specs, results=[landed]) == []

    def test_routing_is_the_one_kind_that_stays_out(self):
        report = {
            "demands": [
                {"table": "none.csv", "tuner": None, "miss_count": 5},
                {"table": "unrouted.csv", "tuner": "a8w8", "miss_count": 9},
                {"table": "declined.csv", "tuner": "fmoe_ck", "miss_count": 7},
            ]
        }
        specs = [TunerSpec("fmoe_ck", skip_reason="script is missing")]
        gaps = coverage_gaps(report, specs)
        assert [g.table for g in gaps if not g.warrants_generated_tuner] == ["unrouted.csv"]
        assert {g.table for g in gaps if g.warrants_generated_tuner} == {"none.csv", "declined.csv"}

    def test_a_covered_table_is_not_a_gap(self):
        specs = [TunerSpec("sglang_dense_bf16")]
        assert coverage_gaps(_demand(tuner="sglang_dense_bf16"), specs) == []

    def test_a_skip_that_is_an_answer_is_not_a_gap(self):
        # The capability exists and said no.
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
        # A single correctness check passes an intermittently wrong kernel at random; a Python-loop timer cannot rank
        # kernels this small.
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
        # Left to interpretation, the obvious element-wise ratio makes any output element near zero dominate -- and by
        # that measure the unmodified torch.matmul scores 1.375, so the gate rejects the default path.
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
        # The cheapest possible tell that a script is not measuring what it reports.
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
        # One clean baseline window and four disturbed ones.
        from kernelforge.gemm_tune.tier3 import referee

        calls = {"n": 0}

        def noisy_baseline():
            calls["n"] += 1
            # Warmup runs first and is not measured; the first *sampled* window is the quiet one.
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
    """It used to answer ``()`` here, which helped nobody."""
    from kernelforge.gemm_tune.tier3.dispatch import shape_key

    with pytest.raises(ValueError, match="MxNxK"):
        shape_key(bad)


def test_shape_key_parses_the_well_formed_case() -> None:
    from kernelforge.gemm_tune.tier3.dispatch import shape_key

    assert shape_key("16x1536x7168") == (16, 1536, 7168)


class TestAWinHasToClearTheNoise:
    """The baseline used to beat itself, so "faster" needed a floor.

    Measured on an MI355X: ``time_paired`` run 25 times per shape with the same
    callable on both sides, over four fleet-demanded shapes. 43 of the 100
    trials were refused as unstable; of the 57 that produced a number, 28 read
    above 1.0 and the largest was 1.00925x. None reached 1.01x.
    """

    def test_the_floor_sits_above_every_null_reading_we_measured(self):
        from kernelforge.gemm_tune.tier3 import referee

        assert referee.MIN_SPEEDUP > 1.00925

    def test_a_speedup_inside_the_noise_is_not_an_improvement(self, clock):
        # 1.0076x is verbatim what torch measured against itself on hardware.
        j = judge_candidates(
            "8192x3456x1152",
            [{"backend": "torch", "config": ""}],
            baseline=clock.call(1.0076e-6),
            dispatch=lambda c: clock.call(1e-6),
        )
        assert j.best_timing.usable
        assert j.best_timing.speedup == pytest.approx(1.0076)
        assert not j.improved, "the unmodified path is not an improvement over itself"

    def test_a_win_that_clears_the_floor_still_counts(self, clock):
        from kernelforge.gemm_tune.tier3 import referee

        clearly_over = (referee.MIN_SPEEDUP + 0.01) * 1e-6
        j = judge_candidates(
            "s",
            [{"n": "a"}],
            baseline=clock.call(clearly_over),
            dispatch=lambda c: clock.call(1e-6),
        )
        assert j.improved

    def test_the_timing_is_still_reported_when_it_falls_short(self, clock):
        """Refusing to call it a win is not the same as hiding the number."""
        j = judge_candidates(
            "s",
            [{"n": "a"}],
            baseline=clock.call(1.002e-6),
            dispatch=lambda c: clock.call(1e-6),
        )
        assert not j.improved
        assert j.to_dict()["best_timing"]["speedup"] == pytest.approx(1.002)


class TestHipblasltCannotTakeTheRunDownWithIt:
    """Two ways to die on this backend, both confirmed on an MI355X.

    Neither is catchable: the C++ error handler ends the process. Since the
    referee runs in-process at the tail of a tuning session, after every other
    tuner has finished, either one costs the whole run its report.

    * ``hipb_mm`` with no extension created -> SIGSEGV (rc=-11).
    * ``hipb_mm`` with an invented ``solidx`` -> ``INVALID_VALUE`` at
      ``hipbsolgemm.cu:945``, rc=1. A generated tuner writes ``solidx`` as a
      free integer, so this is the expected case for a first draft.
    """

    @pytest.fixture
    def aiter(self, monkeypatch):
        import sys
        import types

        mod = types.ModuleType("aiter")
        mod.calls = []
        mod.created = False
        mod.sols = [17, 42, 99]

        def create_extension():
            mod.created = True

        def findallsols(*a, **k):
            # The real one dies here when the extension was never created.
            assert mod.created, "hipb_findallsols before hipb_create_extension"
            mod.calls.append("findallsols")
            return list(mod.sols)

        def hipb_mm(a, bt, sol, *rest, **k):
            assert mod.created, "hipb_mm before hipb_create_extension"
            assert sol in mod.sols, f"invented solidx {sol} would abort the process"
            return "out"

        mod.hipb_create_extension = create_extension
        mod.hipb_findallsols = findallsols
        mod.hipb_mm = hipb_mm
        monkeypatch.setitem(sys.modules, "aiter", mod)
        return mod

    @pytest.fixture
    def adapter(self, monkeypatch):
        import types

        from kernelforge.gemm_tune.tier3.dispatch import _Bf16DenseAdapter

        a = _Bf16DenseAdapter()
        operand = types.SimpleNamespace(t=lambda: "bt")
        monkeypatch.setattr(a, "_ops", lambda key: (operand, operand))
        monkeypatch.setattr(a, "_torch", lambda: types.SimpleNamespace(bfloat16="bf16"))
        return a

    def test_the_extension_is_created_before_anything_touches_the_handle(self, adapter, aiter):
        call = adapter._build((512, 512, 512), {"backend": "hipblaslt", "config": "solidx=42"})
        assert aiter.created, "the guard used to warm up with findallsols, which needs the same handle"
        assert call is not None and call() == "out"

    def test_a_solidx_hipblaslt_never_offered_is_refused_not_run(self, adapter, aiter):
        call = adapter._build((512, 512, 512), {"backend": "hipblaslt", "config": "solidx=999999999"})
        assert call is None, "dispatching this would end the process, not raise"

    def test_a_solidx_that_is_not_even_a_number_is_refused(self, adapter, aiter):
        assert adapter._build((512, 512, 512), {"backend": "hipblaslt", "config": "solidx=fastest"}) is None

    def test_the_solution_set_is_fetched_once_per_shape(self, adapter, aiter):
        for cfg in ("solidx=17", "solidx=42", "solidx=99"):
            assert adapter._build((512, 512, 512), {"backend": "hipblaslt", "config": cfg}) is not None
        assert aiter.calls == ["findallsols"]

    def test_a_different_shape_asks_again(self, adapter, aiter):
        """Solutions are per-operand; reusing one shape's set would invent indices."""
        adapter._build((512, 512, 512), {"backend": "hipblaslt", "config": "solidx=17"})
        adapter._build((1024, 512, 512), {"backend": "hipblaslt", "config": "solidx=17"})
        assert aiter.calls == ["findallsols", "findallsols"]


class TestTheAuthorIsToldWhatACandidateHasToLookLike:
    """The brief used to ask for re-dispatchable candidates without saying how.

    ``dispatch._build`` reads five backend names and a ``k=v;k=v`` config, and
    files candidates by an ``MxNxK`` key. None of that appeared anywhere the
    author could see it, so a script written from the mandate alone -- the whole
    premise of this tier -- proposes candidates in some other vocabulary and
    every one of them is recorded as "not dispatchable". The gate opens, the
    search runs, and nothing can be promoted.
    """

    def _gap(self, table="bf16_tuned_gemm.csv"):
        return CoverageGap(
            table=table,
            tuner="sglang_dense_bf16",
            env_var="AITER_CONFIG_GEMM_BF16",
            key_schema=["M", "N", "K"],
            miss_count=40,
            reason="sglang_dense_bf16 produced nothing landable",
        )

    def _brief(self, table="bf16_tuned_gemm.csv"):
        return build_mandate(self._gap(table), [{"M": 16, "N": 1536, "K": 7168}]).render()

    def test_every_backend_the_referee_can_run_is_named_in_the_brief(self):
        from kernelforge.gemm_tune.tier3.dispatch import DENSE_BF16_BACKENDS

        text = self._brief()
        for backend in DENSE_BF16_BACKENDS:
            assert backend in text, f"{backend} is dispatchable but the author is never told"

    def test_the_config_grammar_and_the_shape_key_are_spelled_out(self):
        text = self._brief()
        assert "MxNxK" in text, "the keys of candidates.json are parsed, not free text"
        assert "8192x3456x1152" in text, "an example beats a description of one"
        assert "key=value" in text and "`;`" in text

    def test_the_required_key_of_each_backend_is_stated(self):
        text = self._brief()
        # A candidate missing these is refused, so leaving them implicit costs
        # the whole shape.
        assert "solidx" in text
        assert "kernelName" in text
        assert "kernelId" in text

    def test_a_table_with_no_adapter_is_told_so_rather_than_told_nothing(self):
        text = self._brief("odd_tuned_gemm.csv")
        assert "re-dispatch" in text
        assert "hipblaslt" not in text, "naming backends that cannot re-time this table misleads"

    def test_the_protocol_reaches_the_machine_readable_form(self):
        d = build_mandate(self._gap(), [{"M": 16, "N": 1536, "K": 7168}]).to_dict()
        assert "hipblaslt" in d["candidate_protocol"]

    def test_a_caller_can_still_supply_its_own(self):
        m = build_mandate(self._gap(), [], candidate_protocol="say it however you like")
        assert "say it however you like" in m.render()
        assert "hipblaslt" not in m.render()


class TestTheVocabularyIsOneObject:
    """Advertised and dispatchable have to be the same set, or the gap is silent.

    ``_build`` checks the candidate's backend against the very table the mandate
    is rendered from, so a name can neither be runnable-but-unadvertised (the
    author never proposes it) nor advertised-but-unrunnable (every proposal of
    it is refused) without a test failing here.
    """

    def test_a_backend_nobody_advertised_is_refused_before_anything_is_touched(self, monkeypatch):
        from kernelforge.gemm_tune.tier3.dispatch import _Bf16DenseAdapter

        adapter = _Bf16DenseAdapter()

        def explode(*_a, **_k):
            raise AssertionError("an unknown backend must not reach the operands")

        monkeypatch.setattr(adapter, "_ops", explode)
        assert adapter._build((16, 16, 16), {"backend": "cutlass", "config": "tile=128"}) is None

    def test_the_advertised_names_are_the_ones_build_implements(self, monkeypatch):
        """Each advertised backend reaches its own branch, not the fallthrough."""
        import sys
        import types

        import kernelforge.gemm_tune.tier3.dispatch as d

        # Enough of aiter to be imported; the branches are reached and then
        # bail on their own missing config, which is the point.
        monkeypatch.setitem(sys.modules, "aiter", types.ModuleType("aiter"))
        reached = []
        monkeypatch.setattr(d.log, "warning", lambda msg, *a: reached.append(a[0] if a else msg))
        adapter = d._Bf16DenseAdapter()
        operand = types.SimpleNamespace(t=lambda: "bt")
        monkeypatch.setattr(adapter, "_ops", lambda key: (operand, operand))
        monkeypatch.setattr(adapter, "_torch", lambda: types.SimpleNamespace(bfloat16="bf16"))
        for backend in d.DENSE_BF16_BACKENDS:
            # Deliberately configless: every branch then returns None early or
            # raises inside its own try, and neither is the fallthrough.
            adapter._build((16, 16, 16), {"backend": backend, "config": ""})
        assert reached == [], f"advertised but not implemented: {reached}"


class TestTheAuthoringSessionHasSomewhereItIsAllowedToWrite:
    """The generate stage never reached a model on a real box.

    A writable session runs under the workspace guard, and the guard starts by
    resolving ``git rev-parse --show-toplevel`` from the session's cwd. That cwd
    is ``<tuning output>/tier3/<table>/`` -- an output directory, never a
    repository -- so every attempt died at ``WorkspaceSafetyError('not a git
    repository')``. Measured on an MI355X with a working provider: the gate
    opened, the adapter resolved, and the run stopped at ``stage='generate'``
    with that message, having asked nothing.
    """

    def _work_dir(self, tmp_path):
        d = tmp_path / "tier3" / "bf16_tuned_gemm_csv"
        d.mkdir(parents=True)
        return d

    def test_an_output_directory_becomes_a_worktree_of_its_own(self, tmp_path):
        from kernelforge.gemm_tune.tier3.generate import _isolate

        work = self._work_dir(tmp_path)
        assert _isolate(work) is None
        assert (work / ".git").exists()

    def test_the_guard_can_resolve_a_baseline_afterwards(self, tmp_path):
        """Exactly what the guard asks for, asked the same way it asks."""
        import subprocess

        from kernelforge.gemm_tune.tier3.generate import _isolate

        work = self._work_dir(tmp_path)
        _isolate(work)
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=work, capture_output=True, text=True, check=True
        )
        assert top.stdout.strip(), "the guard refuses an empty toplevel"
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=work, capture_output=True, text=True, check=False)
        assert head.returncode == 0, "rollback needs a commit to roll back to"

    def test_the_sandbox_is_its_own_root_even_inside_a_checkout(self, tmp_path):
        """Otherwise the guard judges the operator's real tree, not the sandbox."""
        import subprocess

        from kernelforge.gemm_tune.tier3.generate import _isolate

        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        work = self._work_dir(tmp_path)
        _isolate(work)
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=work, capture_output=True, text=True, check=True
        ).stdout.strip()
        assert Path(top).resolve() == work.resolve()

    def test_an_existing_worktree_is_left_alone(self, tmp_path):
        """The retry loop calls this twice; the second must not wipe the first."""
        from kernelforge.gemm_tune.tier3.generate import _isolate

        work = self._work_dir(tmp_path)
        _isolate(work)
        (work / "tuner.py").write_text("# attempt 1", encoding="utf-8")
        assert _isolate(work) is None
        assert (work / "tuner.py").read_text(encoding="utf-8") == "# attempt 1"

    def test_git_missing_is_a_reason_not_a_traceback(self, tmp_path, monkeypatch):
        import subprocess

        from kernelforge.gemm_tune.tier3 import generate

        def no_git(*_a, **_k):
            raise OSError("No such file or directory: 'git'")

        monkeypatch.setattr(subprocess, "run", no_git)
        out = generate._isolate(self._work_dir(tmp_path))
        assert out is not None and not out.ok
        assert "sandbox worktree" in out.reason

    def test_the_session_is_not_started_when_the_sandbox_cannot_be_made(self, tmp_path, monkeypatch):
        """A failure here is reported, not walked past into a doomed session."""
        from kernelforge.gemm_tune.tier3 import generate
        from kernelforge.gemm_tune.tier3.mandate import TunerMandate

        monkeypatch.setattr(
            generate,
            "_isolate",
            lambda _w: generate.GeneratedTuner(False, None, "could not prepare a sandbox worktree"),
        )

        def must_not_run(*_a, **_k):
            raise AssertionError("the authoring session was started without a sandbox")

        monkeypatch.setattr(generate, "_run", must_not_run)
        out = generate.generate_tuner(
            TunerMandate(table="t.csv", key_schema=["M"], demand_shapes=[], why_existing_tiers_failed=""),
            tmp_path / "w",
        )
        assert not out.ok and "sandbox worktree" in out.reason


class TestTheAuthoringSessionIsAllowedToDoTheJob:
    """It asked for permission and there was nobody there.

    Measured on an MI355X with a working provider and a sandbox worktree: the
    session could reach the model, and then stopped to request Write access and
    permission to run python3, explaining that it could not enumerate hipBLASLt
    solutions without them. `end_reason=agent_stopped`, no script, a GPU run
    spent. `tool_policy` was None, so the backend set no allowed_tools at all
    and nothing was pre-approved.
    """

    def _spec(self, monkeypatch, tmp_path):
        from kernelforge.gemm_tune.tier3 import generate
        from kernelforge.gemm_tune.tier3.mandate import TunerMandate

        seen = {}

        class Backend:
            name = "fake"

            def run(self, spec):
                seen["spec"] = spec
                Path(spec.cwd, "tuner.py").write_text("# authored", encoding="utf-8")
                return type("R", (), {"text": "done", "end_reason": "agent_stopped", "session_id": "s"})()

        monkeypatch.setattr(generate, "_isolate", lambda _w: None)
        import kernelforge.agent_backends.registry as reg

        monkeypatch.setattr(reg, "select_default_agent_provider", lambda _m: type("P", (), {"name": "fake"})())
        monkeypatch.setattr(reg, "resolve_agent_runtime", lambda *a, **k: object())
        monkeypatch.setattr(reg, "create_registered_backend", lambda _r: Backend())
        out = generate.generate_tuner(
            TunerMandate(table="t.csv", key_schema=["M"], demand_shapes=[], why_existing_tiers_failed=""),
            tmp_path,
        )
        assert out.ok, out.reason
        return seen["spec"]

    def test_the_tools_it_needs_are_pre_approved(self, monkeypatch, tmp_path):
        policy = self._spec(monkeypatch, tmp_path).tool_policy
        assert policy is not None, "a None policy leaves allowed_tools unset and the session asks"
        assert policy.write, "it cannot deliver tuner.py without Write"
        # solidx, ASM kernel names and opus kernel ids only exist in the
        # installed library; the mandate says inventing one kills the process.
        assert policy.shell, "it cannot enumerate what is callable without running python"

    def test_it_gets_more_than_one_turn(self, monkeypatch, tmp_path):
        policy = self._spec(monkeypatch, tmp_path).tool_policy
        assert policy.max_turns is None or policy.max_turns > 1, (
            "the mandate tells it to explore before searching, which is several turns"
        )

    def test_what_the_agent_said_survives_into_the_reason(self, monkeypatch, tmp_path):
        """Otherwise every failure reads the same and explains nothing."""
        from kernelforge.gemm_tune.tier3 import generate
        from kernelforge.gemm_tune.tier3.mandate import TunerMandate

        class Silent:
            name = "fake"

            def run(self, _spec):
                return type(
                    "R",
                    (),
                    {
                        "text": "Please grant Bash python3 execution and write access.",
                        "end_reason": "agent_stopped",
                        "session_id": "s",
                    },
                )()

        monkeypatch.setattr(generate, "_isolate", lambda _w: None)
        import kernelforge.agent_backends.registry as reg

        monkeypatch.setattr(reg, "select_default_agent_provider", lambda _m: type("P", (), {"name": "fake"})())
        monkeypatch.setattr(reg, "resolve_agent_runtime", lambda *a, **k: object())
        monkeypatch.setattr(reg, "create_registered_backend", lambda _r: Silent())
        out = generate.generate_tuner(
            TunerMandate(table="t.csv", key_schema=["M"], demand_shapes=[], why_existing_tiers_failed=""),
            tmp_path / "w",
        )
        assert not out.ok
        assert "write access" in out.reason, "the agent named its own blocker and we dropped it"
