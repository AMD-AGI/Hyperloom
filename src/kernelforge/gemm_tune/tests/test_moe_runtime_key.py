# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MoE dispatch key, read off the log instead of guessed from the config.

``fmoe_ck`` refuses to tune a key it inferred from ``config.json``, for good
reasons it documents: the quantisation pair, the per-partition ``inter_dim`` and
the EP path's extra masked expert slot are all chosen by the serving framework.
The refusal was correct and the tuner still never ran -- across 33 models on a
real box it skipped 33 times, because the only accepted source was a
hand-prepared ``moe_untuned_csv`` that nothing produced. The key was in the
serving log the whole time; these tests pin down reading it.

Every tuple literal below is copied verbatim from a production sglang log
(MiniMax-M3-MXFP4, TP8, gfx950). The layout matters more than it looks: the
previous parser documented it as starting at ``cu_num``, read the token out of
the CU-count slot, and reported every model's token set as the constant [256].
"""

from __future__ import annotations

import json

import pytest

from kernelforge.gemm_tune import evidence as ev

# Verbatim from the production log, for token counts 1 and 512.
_TUPLE = (
    "('gfx950', 256, {tok}, 6144, 384, 128, 4, <ActivationType.Swiglu: 2>, "
    "'torch.bfloat16', 'torch.float4_e2m1fn_x2', 'torch.float4_e2m1fn_x2', "
    "'QuantType.per_1x32', True, False)"
)
_DISPATCH = "[aiter] [fused_moe] using 2stage default for " + _TUPLE
_MISS = (
    "[aiter] [fused_moe] no tuned FlyDSL config for "
    + _TUPLE
    + ", using heuristic FlyDSL fallback (kn1='flydsl_moe1_afp4_wfp4_bf16', "
    "kn2='flydsl_moe2_afp4_wfp4_bf16')"
)


def _log(*lines: str) -> dict:
    return ev.parse_log("\n".join(lines) + "\n")


class TestTupleLayout:
    def test_the_token_is_read_from_the_token_slot_not_the_cu_count(self):
        rep = _log(_DISPATCH.format(tok=1), _DISPATCH.format(tok=512))
        (key,) = ev.moe_dispatch_keys(rep)
        assert key["tokens"] == [1, 512]
        assert key["cu_num"] == "256"

    def test_the_shape_fields_land_in_the_right_columns(self):
        rep = _log(_DISPATCH.format(tok=1))
        (key,) = ev.moe_dispatch_keys(rep)
        assert key["model_dim"] == "6144"
        # 384, not the config's moe_intermediate_size of 3072: this log is TP8,
        # and the per-partition width is precisely what cannot be derived.
        assert key["inter_dim"] == "384"
        assert key["expert"] == "128"
        assert key["topk"] == "4"
        assert key["q_dtype_a"] == "torch.float4_e2m1fn_x2"
        assert key["q_type"] == "QuantType.per_1x32"

    def test_reprs_are_normalised_to_the_csv_spelling(self):
        rep = _log(_DISPATCH.format(tok=1))
        (key,) = ev.moe_dispatch_keys(rep)
        assert key["act_type"] == "ActivationType.Swiglu"  # was <...: 2>
        assert key["use_g1u1"] == "1" and key["doweight_stage1"] == "0"
        assert key["dtype"] == "torch.bfloat16"  # quotes stripped

    def test_a_tuple_without_the_arch_prefix_still_yields_its_token(self):
        # Builds that print no arch start the tuple at cu_num. The token is
        # still the field after the box properties.
        rep = _log("[aiter] [fused_moe] using 2stage default for (304, 32, 4096, 1536, 8, 2)")
        stages = rep["dispatch"]["moe"]["by_stage"]
        assert stages["2stage/default"]["tokens"] == [32]

    def test_short_tuples_record_no_key_and_warn_once_per_log(self, caplog):
        short_dispatch = "[aiter] [fused_moe] using 2stage default for (304, 32, 4096, 1536, 8, 2)"
        short_miss = (
            "[aiter] [fused_moe] no tuned FlyDSL config for "
            "(304, 32, 4096, 1536, 8, 2), using heuristic FlyDSL fallback"
        )
        with caplog.at_level("WARNING"):
            rep = _log(short_dispatch, short_dispatch, short_miss)

        assert ev.moe_dispatch_keys(rep) == []
        moe = rep["dispatch"]["moe"]
        assert moe["unkeyed_tuple_count"] == 3
        assert moe["unkeyed_miss_count"] == 1
        warnings = [r for r in caplog.records if "recording tokens only" in r.message]
        assert len(warnings) == 1


class TestMissSignal:
    def test_the_miss_line_is_what_marks_a_token_as_needing_tuning(self):
        # The dispatch line prints identically whether or not a tuned row was
        # found, so it cannot be the miss signal on its own.
        rep = _log(_DISPATCH.format(tok=1), _DISPATCH.format(tok=512), _MISS.format(tok=512))
        (key,) = ev.moe_dispatch_keys(rep)
        assert key["tokens"] == [1, 512]
        assert key["untuned_tokens"] == [512]
        assert key["miss_count"] == 1

    def test_the_two_lines_fold_into_one_key_not_two(self):
        rep = _log(_DISPATCH.format(tok=1), _MISS.format(tok=1))
        assert len(ev.moe_dispatch_keys(rep)) == 1

    def test_the_fallback_flavour_is_recorded(self):
        rep = _log(_MISS.format(tok=1))
        assert rep["dispatch"]["moe"]["fallback_flavour"] == "FlyDSL"

    def test_keys_are_ordered_most_missed_first(self):
        wide = _TUPLE.replace("6144", "8192")
        rep = _log(
            _DISPATCH.format(tok=1),
            ("[aiter] [fused_moe] no tuned FlyDSL config for " + wide).format(tok=8),
            ("[aiter] [fused_moe] no tuned FlyDSL config for " + wide).format(tok=16),
        )
        keys = ev.moe_dispatch_keys(rep)
        assert [k["model_dim"] for k in keys] == ["8192", "6144"]


class TestUntunedCsv:
    def test_the_header_is_exactly_the_tuple_minus_the_box_properties(self):
        from kernelforge.gemm_tune.tuners.fmoe_ck import _FMOE_CSV_HEADER

        assert ",".join(ev.MOE_KEY_FIELDS) == _FMOE_CSV_HEADER

    def test_rows_carry_the_missed_tokens(self):
        rep = _log(_DISPATCH.format(tok=1), _MISS.format(tok=512), _MISS.format(tok=1024))
        (key,) = ev.moe_dispatch_keys(rep)
        text = ev.moe_untuned_csv_text(key)
        lines = text.strip().splitlines()
        assert lines[0].split(",") == list(ev.MOE_KEY_FIELDS)
        assert [ln.split(",")[0] for ln in lines[1:]] == ["512", "1024"]
        assert lines[1].split(",")[1:5] == ["6144", "384", "128", "4"]

    def test_it_falls_back_to_every_token_seen_when_none_missed(self):
        rep = _log(_DISPATCH.format(tok=1), _DISPATCH.format(tok=32))
        (key,) = ev.moe_dispatch_keys(rep)
        rows = ev.moe_untuned_csv_text(key).strip().splitlines()[1:]
        assert [r.split(",")[0] for r in rows] == ["1", "32"]

    def test_the_rendered_csv_passes_the_tuners_own_validator(self, tmp_path):
        from kernelforge.gemm_tune.tuners.fmoe_ck import _validate_fmoe_csv

        rep = _log(_DISPATCH.format(tok=1), _MISS.format(tok=512))
        (key,) = ev.moe_dispatch_keys(rep)
        path = tmp_path / "untuned_fmoe.csv"
        path.write_text(ev.moe_untuned_csv_text(key), encoding="utf-8")
        assert _validate_fmoe_csv(path) is None


class TestReportSurvivesDisk:
    def test_keys_round_trip_through_demand_json(self, tmp_path):
        # The tuner reads the report back off disk, so the key has to be JSON
        # -- the sets it is accumulated in are not.
        rep = _log(_DISPATCH.format(tok=1), _MISS.format(tok=512))
        out = ev.write_demand(rep, tmp_path / "demand.json")
        loaded = json.loads(out.read_text(encoding="utf-8"))
        (key,) = ev.moe_dispatch_keys(loaded)
        assert key["untuned_tokens"] == [512]


class TestFmoeCkAcceptsIt:
    """The guard must open for a logged key and stay shut for a guessed one."""

    def _tuner(self, tmp_path, demand_json):
        pytest.importorskip("kernelforge.gemm_tune.tuners.fmoe_ck")
        from kernelforge.gemm_tune.tuners.fmoe_ck import FmoeCKTuner

        tuner = FmoeCKTuner.__new__(FmoeCKTuner)
        tuner.ctx = type(
            "Ctx",
            (),
            {
                "demand_json": demand_json,
                "moe_untuned_csv": None,
                "tokens": [],
                "token_hint": None,
            },
        )()
        tuner.work_dir = tmp_path
        return tuner

    def _demand(self, tmp_path):
        rep = _log(
            _DISPATCH.format(tok=1),
            _DISPATCH.format(tok=512),
            _MISS.format(tok=512),
        )
        return ev.write_demand(rep, tmp_path / "demand.json")

    def test_a_logged_key_is_found(self, tmp_path):
        tuner = self._tuner(tmp_path, self._demand(tmp_path))
        key = tuner._demand_key()
        assert key is not None and key["inter_dim"] == "384"

    def test_no_demand_file_means_no_key(self, tmp_path):
        assert self._tuner(tmp_path, None)._demand_key() is None

    def test_a_log_with_no_moe_dispatch_means_no_key(self, tmp_path):
        out = ev.write_demand(_log("[aiter] nothing to see"), tmp_path / "demand.json")
        assert self._tuner(tmp_path, out)._demand_key() is None

    def test_dispatches_with_zero_misses_do_not_become_tuning_demand(self, tmp_path):
        rep = _log(*[_DISPATCH.format(tok=32) for _ in range(5)])
        out = ev.write_demand(rep, tmp_path / "demand.json")
        (observed,) = ev.moe_dispatch_keys(rep)
        assert observed["miss_count"] == 0
        assert observed["untuned_tokens"] == []
        assert observed["tokens"] == [32]

        assert self._tuner(tmp_path, out)._demand_key() is None

    def test_misses_outside_ck_two_stage_tokens_are_not_ck_demand(self, tmp_path):
        ck_dispatch = _DISPATCH.replace("default", "ck").format(tok=16)
        asm_dispatch = _DISPATCH.replace("2stage default", "1stage asm").format(tok=4096)
        rep = _log(ck_dispatch, asm_dispatch, _MISS.format(tok=4096))
        out = ev.write_demand(rep, tmp_path / "demand.json")

        assert ev.moe_ck_missed_keys(rep) == []
        assert self._tuner(tmp_path, out)._demand_key() is None

    def test_ck_demand_contains_only_misses_in_the_two_stage_token_range(self, tmp_path):
        ck_dispatch = _DISPATCH.replace("default", "ck").format(tok=16)
        asm_dispatch = _DISPATCH.replace("2stage default", "1stage asm").format(tok=4096)
        rep = _log(
            ck_dispatch,
            asm_dispatch,
            _MISS.format(tok=16),
            _MISS.format(tok=4096),
        )
        out = ev.write_demand(rep, tmp_path / "demand.json")

        key = self._tuner(tmp_path, out)._demand_key()
        assert key is not None
        assert key["untuned_tokens"] == [16]

    def test_demand_json_is_parsed_once_across_validate_and_run(self, tmp_path, monkeypatch):
        out = self._demand(tmp_path)
        tuner = self._tuner(tmp_path, out)
        original = ev.load_demand
        calls = 0

        def _load(path):
            nonlocal calls
            calls += 1
            return original(path)

        monkeypatch.setattr(ev, "load_demand", _load)
        assert tuner._demand_key() is not None
        assert tuner._demand_key() is not None
        assert calls == 1

    def test_the_untuned_csv_is_written_from_the_logged_key(self, tmp_path):
        tuner = self._tuner(tmp_path, self._demand(tmp_path))
        path = tuner._untuned_csv_from_demand(tuner._demand_key())
        rows = path.read_text(encoding="utf-8").strip().splitlines()
        assert rows[1].startswith("512,6144,384,128,4,ActivationType.Swiglu,")

    def test_the_token_budget_keeps_both_ends_of_the_range(self, tmp_path):
        # Keeping the largest N would tune prefill only and leave decode on the
        # untuned fallback, which is the opposite of where serving time goes.
        rep = _log(*[_MISS.format(tok=t) for t in (1, 8, 64, 512)])
        out = ev.write_demand(rep, tmp_path / "demand.json")
        tuner = self._tuner(tmp_path, out)
        tuner.ctx.tokens = [0, 0]  # a budget of two, whatever its values
        path = tuner._untuned_csv_from_demand(tuner._demand_key())
        rows = path.read_text(encoding="utf-8").strip().splitlines()[1:]
        assert [r.split(",")[0] for r in rows] == ["1", "512"]

    def test_the_budget_thins_evenly_rather_than_clipping(self, tmp_path):
        rep = _log(*[_MISS.format(tok=t) for t in (1, 2, 4, 8, 16, 32, 64)])
        out = ev.write_demand(rep, tmp_path / "demand.json")
        tuner = self._tuner(tmp_path, out)
        tuner.ctx.tokens = [0, 0, 0]
        path = tuner._untuned_csv_from_demand(tuner._demand_key())
        rows = path.read_text(encoding="utf-8").strip().splitlines()[1:]
        assert [r.split(",")[0] for r in rows] == ["1", "8", "64"]

    def test_a_budget_of_one_keeps_the_largest(self, tmp_path):
        rep = _log(*[_MISS.format(tok=t) for t in (1, 8, 64)])
        out = ev.write_demand(rep, tmp_path / "demand.json")
        tuner = self._tuner(tmp_path, out)
        tuner.ctx.tokens = [0]
        path = tuner._untuned_csv_from_demand(tuner._demand_key())
        rows = path.read_text(encoding="utf-8").strip().splitlines()[1:]
        assert [r.split(",")[0] for r in rows] == ["64"]

    def test_every_observed_token_is_tuned_when_the_budget_allows(self, tmp_path):
        rep = _log(*[_MISS.format(tok=t) for t in (1, 8, 64)])
        out = ev.write_demand(rep, tmp_path / "demand.json")
        tuner = self._tuner(tmp_path, out)
        tuner.ctx.tokens = [0] * 10
        path = tuner._untuned_csv_from_demand(tuner._demand_key())
        rows = path.read_text(encoding="utf-8").strip().splitlines()[1:]
        assert [r.split(",")[0] for r in rows] == ["1", "8", "64"]

    def test_the_token_hint_restricts_the_set_it_does_not_just_size_it(self, tmp_path):
        # A run where CK 2-stage serves token 16 and the 1-stage path serves
        # 4096. The router hands this tuner token_hint=[16] for exactly that
        # reason. Reading the hint as a budget of one spent the single slot on
        # 4096 -- a token CK never dispatches -- and dropped the one it does.
        rep = _log(*[_MISS.format(tok=t) for t in (16, 4096)])
        out = ev.write_demand(rep, tmp_path / "demand.json")
        tuner = self._tuner(tmp_path, out)
        tuner.ctx.tokens = [16]
        tuner.ctx.token_hint = [16]
        path = tuner._untuned_csv_from_demand(tuner._demand_key())
        rows = path.read_text(encoding="utf-8").strip().splitlines()[1:]
        assert [r.split(",")[0] for r in rows] == ["16"]

    def test_a_hint_disjoint_from_the_misses_is_rejected(self, tmp_path):
        # Hint and misses come from the same log, so a disjoint pair says these
        # tokens belong to another stage. A CK row for them is unreachable.
        rep = _log(*[_MISS.format(tok=t) for t in (8, 64)])
        out = ev.write_demand(rep, tmp_path / "demand.json")
        tuner = self._tuner(tmp_path, out)
        tuner.ctx.tokens = [128, 256]  # the budget still applies, and allows both
        tuner.ctx.token_hint = [128, 256]
        with pytest.raises(ValueError, match="CK 2-stage token hint"):
            tuner._untuned_csv_from_demand(tuner._demand_key())

    def test_no_hint_leaves_the_budget_behaviour_untouched(self, tmp_path):
        # ctx.tokens without a hint is the run's coverage sweep, not a
        # restriction: intersecting against it would drop every observed token
        # that the sweep happens not to list.
        rep = _log(*[_MISS.format(tok=t) for t in (1, 8, 64)])
        out = ev.write_demand(rep, tmp_path / "demand.json")
        tuner = self._tuner(tmp_path, out)
        tuner.ctx.tokens = [128, 256, 512]  # disjoint from the observed set
        tuner.ctx.token_hint = None
        path = tuner._untuned_csv_from_demand(tuner._demand_key())
        rows = path.read_text(encoding="utf-8").strip().splitlines()[1:]
        assert [r.split(",")[0] for r in rows] == ["1", "8", "64"]

    def test_provenance_is_reported_as_runtime_observed(self, tmp_path):
        tuner = self._tuner(tmp_path, self._demand(tmp_path))
        _, source = tuner._resolve_untuned_csv()
        assert source == "runtime_observed"
