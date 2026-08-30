# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for serving-log evidence parsing.

Two parsing rules carry most of the weight:

* the wide key group (dtype/otype/bias/scaleAB/bpreshuffle) is **optional** --
  the bf16 op prints it, the a8w8_blockscale op prints M/N/K only. Requiring it
  dropped 252 of 440 misses in the first version;
* zero hit lines means *unknown*, not zero hits -- hit logging is gated behind
  ``AITER_LOG_TUNED_CONFIG=1`` while miss logging is unconditional. Reading it
  as zero would REVERT every arm that did not set the flag.
"""

from __future__ import annotations

from kernelforge.gemm_tune import evidence as ev

_BF16_MISS = (
    "[aiter] shape is M:65536, N:3456, K:1152 dtype='torch.bfloat16' "
    "otype='torch.bfloat16' bias=True, scaleAB=False, bpreshuffle=False, "
    "not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv, will use default config!"
)
_NARROW_MISS = (
    "[aiter] shape is M:512, N:1536, K:7168, "
    "not found tuned config in /tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv"
)
_HIT = (
    "[aiter] shape is M:15, N:4096, K:4096 dtype='torch.bfloat16' "
    "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, "
    "found padded_M: 16"
)
_MERGE = "[aiter] merge tuned file under model_configs/ and configs/ /tmp/a.csv:/tmp/bf16_tuned_gemm.csv"


class TestKeyGroupIsOptional:
    def test_wide_form_captures_every_field(self):
        rep = ev.parse_log(_BF16_MISS)
        key = rep["demands"][0]["keys"][0]
        assert (key["M"], key["N"], key["K"]) == ("65536", "3456", "1152")
        assert key["bias"] == "True" and key["bpreshuffle"] == "False"

    def test_narrow_form_is_not_dropped(self):
        # The regression that lost 252/440 misses.
        rep = ev.parse_log(_NARROW_MISS)
        assert rep["apply_verdict"]["miss"] == 1
        assert rep["demands"][0]["table"] == "a8w8_blockscale_tuned_gemm.csv"

    def test_both_forms_in_one_log(self):
        rep = ev.parse_log("\n".join([_BF16_MISS, _NARROW_MISS]))
        assert {d["table"] for d in rep["demands"]} == {
            "bf16_tuned_gemm.csv",
            "a8w8_blockscale_tuned_gemm.csv",
        }

    def test_key_schema_follows_the_table_not_the_line(self):
        rep = ev.parse_log(_NARROW_MISS)
        assert rep["demands"][0]["key_schema"] == ["M", "N", "K"]
        rep = ev.parse_log(_BF16_MISS)
        assert "bpreshuffle" in rep["demands"][0]["key_schema"]


class TestApplyVerdict:
    def test_misses_without_hit_logging_is_inconclusive(self):
        # NOT "zero hits": hit lines need AITER_LOG_TUNED_CONFIG=1.
        rep = ev.parse_log(_BF16_MISS)
        assert rep["apply_verdict"]["verdict"] == "inconclusive_no_hit_logging"

    def test_any_hit_means_served(self):
        rep = ev.parse_log("\n".join([_HIT, _BF16_MISS]))
        av = rep["apply_verdict"]
        assert av["hit"] == 1 and av["miss"] == 1 and av["verdict"] == "served"

    def test_no_lookups_at_all(self):
        assert ev.parse_log("nothing here")["apply_verdict"]["verdict"] == "no_lookups"

    def test_merged_tables_are_collected(self):
        rep = ev.parse_log(_MERGE)
        assert "/tmp/bf16_tuned_gemm.csv" in rep["merged_tables"]


class TestDemandAggregation:
    def test_repeated_key_is_counted_not_duplicated(self):
        rep = ev.parse_log("\n".join([_BF16_MISS] * 3))
        d = rep["demands"][0]
        assert d["miss_count"] == 3 and d["distinct_keys"] == 1
        assert d["keys"][0]["requests"] == 3

    def test_keys_are_ordered_by_request_count(self):
        other = _BF16_MISS.replace("M:65536", "M:8")
        rep = ev.parse_log("\n".join([other] + [_BF16_MISS] * 2))
        assert rep["demands"][0]["keys"][0]["M"] == "65536"

    def test_table_maps_to_its_tuner_and_env(self):
        d = ev.parse_log(_BF16_MISS)["demands"][0]
        assert d["tuner"] == "sglang_dense_bf16"
        assert d["env_var"] == "AITER_CONFIG_GEMM_BF16"


class TestMoEDispatch:
    def test_stage_tokens_are_kept_per_stage(self):
        # A model dispatches different stages at different token counts; a single
        # "saw 1stage" boolean collapses that away and suppresses tuning for the
        # range 2stage actually serves.
        log = "\n".join(
            [
                "[aiter] [fused_moe] using 1stage default for (304, 1, 4096, 1536, 256, 6)",
                "[aiter] [fused_moe] using 2stage default for (304, 64, 4096, 1536, 256, 6)",
            ]
        )
        moe = ev.parse_log(log)["dispatch"]["moe"]
        assert moe["impl"] == "aiter_ck"
        assert moe["stages_seen"] == ["1stage", "2stage"]
        assert moe["tunable_ck_2stage"] is True

    def test_vllm_triton_hits_and_misses(self):
        log = "\n".join(
            [
                "Using configuration from /cfg/E=256,N=2048.json for MoE layer",
                "Config file not found at /cfg/E=256,N=4096.json",
            ]
        )
        moe = ev.parse_log(log)["dispatch"]["moe"]
        assert moe["impl"] == "vllm_triton"
        assert moe["vllm_config_hit"] == 1 and moe["vllm_config_miss"] == 1


class TestDemandConsumption:
    def _report(self):
        return ev.parse_log("\n".join([_BF16_MISS] * 2 + [_BF16_MISS.replace("M:65536", "M:8")]))

    def test_demand_for_tuner_selects_the_right_table(self):
        rep = ev.parse_log("\n".join([_BF16_MISS, _NARROW_MISS]))
        assert ev.demand_for_tuner(rep, "sglang_dense_bf16")["table"] == "bf16_tuned_gemm.csv"
        assert ev.demand_for_tuner(rep, "not_a_tuner") is None

    def test_shapes_are_typed_and_ordered_by_requests(self):
        entry = ev.demand_for_tuner(self._report(), "sglang_dense_bf16")
        shapes = ev.demand_shapes(entry, bucket=False)
        assert shapes[0]["M"] == 65536 and isinstance(shapes[0]["M"], int)
        assert shapes[0]["requests"] == 2

    def test_limit_truncates_by_request_order(self):
        entry = ev.demand_for_tuner(self._report(), "sglang_dense_bf16")
        got = ev.demand_shapes(entry, limit=1, bucket=False)
        assert [s["M"] for s in got] == [65536]

    def test_shapes_default_to_the_padded_M_a_row_must_be_written_at(self):
        # aiter reaches a tuned row at the exact M, else at the padded M. 65536
        # is past the 8192 clamp, so a row for it lives at 8192; writing it at
        # 65536 produces a table no lookup can reach.
        entry = ev.demand_for_tuner(self._report(), "sglang_dense_bf16")
        shapes = ev.demand_shapes(entry)
        assert shapes[0]["M"] == 8192
        assert shapes[0]["observed_M"] == [65536]

    def test_keys_sharing_a_bucket_cost_one_slot_not_several(self):
        # The whole point of bucketing: three raw keys in one padded bucket are
        # served by a single tuned row, so they must not eat three of the budget.
        # Listed most-requested first, as parse_log emits them.
        entry = {
            "keys": [
                {"M": "64", "N": "4096", "K": "4096", "requests": 11},
                {"M": "300", "N": "4096", "K": "4096", "requests": 5},
                {"M": "400", "N": "4096", "K": "4096", "requests": 4},
                {"M": "512", "N": "4096", "K": "4096", "requests": 3},
            ]
        }
        shapes = ev.demand_shapes(entry)
        assert [(s["M"], s["requests"]) for s in shapes] == [(512, 12), (64, 11)]
        assert shapes[0]["observed_M"] == [300, 400, 512]
        # ...and with one slot, the bucket worth 12 requests wins over the raw
        # key worth 11, which the raw ordering would have picked first.
        assert [s["M"] for s in ev.demand_shapes(entry, limit=1)] == [512]
        assert [s["M"] for s in ev.demand_shapes(entry, limit=1, bucket=False)] == [64]

    def test_padding_matches_aiter_next_power_of_two_capped(self):
        assert [ev.padded_m(m) for m in (1, 2, 3, 15, 16, 17, 64, 100, 513)] == [1, 2, 4, 16, 16, 32, 64, 128, 1024]
        assert ev.padded_m(8192) == 8192 and ev.padded_m(20000) == 8192

    def test_buckets_do_not_merge_across_differing_extended_keys(self):
        entry = {
            "keys": [
                {"M": "300", "N": "4096", "K": "4096", "bias": "True", "requests": 5},
                {"M": "400", "N": "4096", "K": "4096", "bias": "False", "requests": 4},
            ]
        }
        shapes = ev.demand_shapes(entry)
        assert len(shapes) == 2
        assert {s["bias"] for s in shapes} == {"True", "False"}

    def test_extended_fields_survive_into_shapes(self):
        entry = ev.demand_for_tuner(self._report(), "sglang_dense_bf16")
        assert ev.demand_shapes(entry)[0]["bias"] == "True"

    def test_roundtrip_through_disk(self, tmp_path):
        out = ev.write_demand(self._report(), tmp_path / "demand.json")
        loaded = ev.load_demand(out)
        assert loaded["schema"] == ev.SCHEMA_VERSION
        assert ev.demand_for_tuner(loaded, "sglang_dense_bf16") is not None

    def test_bad_demand_file_is_none_not_a_crash(self, tmp_path):
        assert ev.load_demand(tmp_path / "missing.json") is None
        bad = tmp_path / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        assert ev.load_demand(bad) is None


class TestRobustness:
    def test_unreadable_log_yields_empty_report(self, tmp_path):
        rep = ev.parse_log_file(tmp_path / "nope.log")
        assert rep["demands"] == [] and rep["apply_verdict"]["verdict"] == "no_lookups"

    def test_unknown_table_still_recorded_with_default_schema(self):
        line = "[aiter] shape is M:1, N:2, K:3, not found tuned config in /tmp/brand_new.csv"
        d = ev.parse_log(line)["demands"][0]
        assert d["table"] == "brand_new.csv"
        assert d["tuner"] is None and d["key_schema"] == ["M", "N", "K"]
