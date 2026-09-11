# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test mapping fused-MoE dispatch misses into eligible tuner demand."""

from __future__ import annotations

import pytest

from kernelforge.gemm_tune import evidence as ev

_TUPLE = (
    "('gfx950', 256, {tok}, 6144, 384, 128, 4, <ActivationType.Swiglu: 2>, "
    "'torch.bfloat16', 'torch.float4_e2m1fn_x2', 'torch.float4_e2m1fn_x2', "
    "'QuantType.per_1x32', True, False)"
)
_MISS = (
    "[aiter] [fused_moe] no tuned FlyDSL config for "
    + _TUPLE
    + ", using heuristic FlyDSL fallback (kn1='flydsl_moe1_afp4_wfp4_bf16', "
    "kn2='flydsl_moe2_afp4_wfp4_bf16')"
)
_2STAGE = "[aiter] [fused_moe] using 2stage ck for " + _TUPLE
_1STAGE = "[aiter] [fused_moe] using 1stage asm for " + _TUPLE
_TRITON = "Using configuration from /x/E=8,N=14336.json for MoE layer"


def _log(*lines: str) -> dict:
    return ev.parse_log("\n".join(lines) + "\n")


def _fmoe(report: dict):
    entries = [d for d in report["demands"] if d["table"] == "tuned_fmoe.csv"]
    return entries[0] if entries else None


class TestTheMissesBecomeADemand:
    def test_the_demand_exists_and_names_its_owner(self):
        entry = _fmoe(_log(_MISS.format(tok=16)))
        assert entry is not None, "a run that missed fmoe demanded fmoe"
        assert entry["tuner"] == "fmoe_ck"
        assert entry["env_var"] == "AITER_CONFIG_FMOE"

    def test_the_key_schema_is_the_fmoe_csv_header(self):
        # Not (M, N, K): a fused-MoE row is keyed on the dispatch tuple, and the
        # header aiter reads untuned_fmoe.csv with is what a tuner must produce.
        entry = _fmoe(_log(_MISS.format(tok=16)))
        assert entry["key_schema"] == list(ev.MOE_KEY_FIELDS)
        assert "token" in entry["key_schema"]

    def test_one_row_per_token_the_runtime_asked_for(self):
        # Keys collapse across token counts so the shape stays one row; a demand
        # does not, because aiter looks a row up at the exact token count.
        entry = _fmoe(_log(_MISS.format(tok=16), _MISS.format(tok=64)))
        assert sorted(k["token"] for k in entry["keys"]) == ["16", "64"]
        assert all(k["inter_dim"] == "384" for k in entry["keys"])

    def test_requests_are_counted_per_token_not_split_evenly(self):
        # The shape-level miss_count cannot say which token count the runtime
        # spent its misses on, and dividing it would invent the answer.
        entry = _fmoe(_log(_MISS.format(tok=16), _MISS.format(tok=16), _MISS.format(tok=64)))
        by_token = {k["token"]: k["requests"] for k in entry["keys"]}
        assert by_token == {"16": 2, "64": 1}

    def test_the_most_requested_token_comes_first(self):
        entry = _fmoe(_log(_MISS.format(tok=64), _MISS.format(tok=16), _MISS.format(tok=16)))
        assert entry["keys"][0]["token"] == "16"

    def test_miss_count_is_the_sum_of_what_was_counted(self):
        entry = _fmoe(_log(_MISS.format(tok=16), _MISS.format(tok=16), _MISS.format(tok=64)))
        assert entry["miss_count"] == 3
        assert entry["distinct_keys"] == 2

    def test_a_run_that_never_missed_demands_nothing(self):
        # A dispatch line is not a demand: it says the runtime called fused_moe,
        # not that it went looking for a tuned row and failed.
        assert _fmoe(_log(_2STAGE.format(tok=16))) is None


class TestItRefusesWhatCannotBeServed:
    def test_a_token_only_1stage_ever_served_is_not_demanded(self):
        # fmoe_ck tunes the CK 2-stage path. Demanding a row for a token the log
        # only ever saw on the asm 1-stage path asks for a row nothing reads.
        report = _log(
            _2STAGE.format(tok=16),
            _1STAGE.format(tok=4096),
            _MISS.format(tok=16),
            _MISS.format(tok=4096),
        )
        entry = _fmoe(report)
        assert [k["token"] for k in entry["keys"]] == ["16"]

    def test_a_triton_moe_run_demands_nothing_from_the_ck_tuner(self):
        # Both kinds of line in one log means the runtime is not simply CK, and
        # these keys no longer describe what it will dispatch.
        assert _fmoe(_log(_MISS.format(tok=16), _TRITON)) is None

    def test_the_dispatch_record_is_left_alone_either_way(self):
        # The demand is a restatement, not a move: consumers of the dispatch
        # facts must not lose them because a demand was added beside them.
        report = _log(_MISS.format(tok=16), _TRITON)
        (key,) = ev.moe_dispatch_keys(report)
        assert key["untuned_tokens"] == [16]


class TestWhatItUnblocks:
    def test_the_tuner_can_now_find_its_own_demand(self):
        report = _log(_MISS.format(tok=16))
        assert ev.demand_for_tuner(report, "fmoe_ck") is not None

    def test_the_demand_yields_shapes(self):
        # demand_shapes reads M/N/K and buckets on padded M. A MoE key has
        # neither, so the dense path returned [] for it -- a demand that
        # produces no shapes is a mandate with nothing in it to cover.
        entry = _fmoe(_log(_MISS.format(tok=16), _MISS.format(tok=64)))
        shapes = ev.demand_shapes(entry)
        assert len(shapes) == 2
        assert {s["token"] for s in shapes} == {"16", "64"}
        assert "observed_M" not in shapes[0], "padded-M bucketing does not apply to fmoe rows"

    def test_the_budget_takes_the_most_requested_first(self):
        entry = _fmoe(_log(_MISS.format(tok=64), _MISS.format(tok=16), _MISS.format(tok=16)))
        (shape,) = ev.demand_shapes(entry, limit=1)
        assert shape["token"] == "16"

    def test_the_router_selects_the_ck_tuner_off_the_demand(self):
        # The demand branch adds a tuner the framework branch left out. Before
        # this, no demand ever named fmoe_ck, so that branch could not reach it
        # however many times the runtime had missed its table.
        from kernelforge.gemm_tune.model_analyzer import ModelProfile
        from kernelforge.gemm_tune.router import select_tuners

        specs = select_tuners(
            ModelProfile(model_path="/fake", architecture="LlamaForCausalLM"),
            framework="sglang",
            precision="bf16",
            quant_type="none",
            gpu_type="mi355x",
            demand_report=_log(_MISS.format(tok=16)),
        )
        assert "fmoe_ck" in [s.name for s in specs if not s.skip_reason]

    def test_a_dense_tuner_does_not_borrow_moe_shapes(self):
        # vllm_dense_tunableop borrows shapes from every table in
        # TABLE_KEY_SCHEMA when it has no demand of its own. A fused-MoE key has
        # no (M, N, K) to borrow, so tuned_fmoe.csv must stay out of that map.
        assert "tuned_fmoe.csv" not in ev.TABLE_KEY_SCHEMA


class TestTheKillSwitch:
    def test_it_can_be_turned_off_without_a_revert(self, monkeypatch):
        # Emitting this demand changes two behaviours outside evidence.py, so
        # there has to be a way to put them back that is not a code change.
        monkeypatch.setenv(ev.MOE_DEMAND_DISABLE_ENV, "1")
        report = _log(_MISS.format(tok=16))
        assert _fmoe(report) is None
        # And the records it was built from are still reported.
        assert ev.moe_dispatch_keys(report)

    @pytest.mark.parametrize("value", ["", "0", "no"])
    def test_it_is_on_unless_the_switch_says_otherwise(self, monkeypatch, value):
        monkeypatch.setenv(ev.MOE_DEMAND_DISABLE_ENV, value)
        assert _fmoe(_log(_MISS.format(tok=16))) is not None
