# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test fused-MoE dispatch decisions that do not require a GPU."""

from __future__ import annotations

import logging

import pytest

from kernelforge.gemm_tune.evidence import MOE_KEY_FIELDS, MOE_TABLE
from kernelforge.gemm_tune.tier3 import dispatch as dp

_KEY = {
    "token": "512",
    "model_dim": "6144",
    "inter_dim": "384",
    "expert": "128",
    "topk": "4",
    "act_type": "ActivationType.Swiglu",
    "dtype": "torch.bfloat16",
    "q_dtype_a": "torch.float4_e2m1fn_x2",
    "q_dtype_w": "torch.float4_e2m1fn_x2",
    "q_type": "QuantType.per_1x32",
    "use_g1u1": "1",
    "doweight_stage1": "0",
}
_SHAPE = "|".join(f"{k}={v}" for k, v in _KEY.items())
_KN1 = "flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_kw2_fp4"
# Tile K=128, not 256: stage 2 walks K over inter_dim, and 384 % 256 sends
# aiter to a smaller tile than the name says. This fixture read 256 until the
# check for it existed, which is the case for the check.
_KN2 = "flydsl_moe2_afp4_wfp4_bf16_t32x128x128_atomic"
_CAND = {"backend": "aiter_fmoe", "config": f"block_m=32;ksplit=0;kernelName1={_KN1};kernelName2={_KN2}"}


class TestTheShapeKey:
    def test_it_round_trips_the_whole_dispatch_key(self):
        assert dp.moe_shape_key(_SHAPE) == _KEY

    def test_it_keeps_schema_order_whatever_order_it_was_written_in(self):
        # Downstream writes these into a CSV column by column, so the order the
        # author happened to use must not reach it.
        shuffled = "|".join(f"{k}={_KEY[k]}" for k in reversed(list(_KEY)))
        assert list(dp.moe_shape_key(shuffled)) == list(MOE_KEY_FIELDS)

    @pytest.mark.parametrize("value", ["torch.float4_e2m1fn_x2", "QuantType.per_1x32"])
    def test_values_containing_dots_and_x_survive(self, value):
        # The reason this is not MxNxK: two of the twelve values contain an "x"
        # and four contain a ".", so every obvious positional encoding is
        # ambiguous against the data it has to carry.
        assert value in dp.moe_shape_key(_SHAPE).values()

    def test_a_missing_field_says_which_one(self):
        partial = "|".join(f"{k}={v}" for k, v in _KEY.items() if k != "topk")
        with pytest.raises(ValueError, match="topk"):
            dp.moe_shape_key(partial)

    def test_a_dense_shape_is_refused_rather_than_half_read(self):
        with pytest.raises(ValueError):
            dp.moe_shape_key("8192x3456x1152")


class TestTheAdapterIsReachable:
    def test_the_table_has_an_adapter_now(self):
        assert MOE_TABLE in dp.SUPPORTED_TABLES
        assert dp.adapters_for(MOE_TABLE) is not None

    def test_a_table_we_still_cannot_serve_says_so(self):
        assert dp.adapters_for("a4w4_blockscale_tuned_gemm.csv") is None

    def test_the_mandate_describes_the_moe_protocol(self):
        # An author working from the mandate alone has to learn the shape
        # encoding and the config keys from it; nothing else tells them.
        text = dp.describe_candidate_protocol(MOE_TABLE)
        assert "kernelName1" in text and "kernelName2" in text
        assert "aiter_fmoe" in text
        for field in MOE_KEY_FIELDS:
            assert field in text

    def test_the_dense_protocol_is_untouched(self):
        assert "MxNxK" in dp.describe_candidate_protocol("bf16_tuned_gemm.csv")


class TestTheCandidateBecomesARow:
    def _adapter(self, monkeypatch):
        adapter = dp._FusedMoeAdapter()
        monkeypatch.setattr(adapter, "_gfx", lambda: "gfx950")
        monkeypatch.setattr(adapter, "_cu_num", lambda: 256)
        return adapter

    def test_the_row_is_the_key_plus_the_kernels(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        path, kn1, kn2 = adapter._candidate_csv(_SHAPE, _CAND)
        header, row = path.read_text(encoding="utf-8").splitlines()
        cells = dict(zip(header.split(","), row.split(","), strict=True))
        assert (kn1, kn2) == (_KN1, _KN2)
        assert cells["kernelName1"] == _KN1
        assert cells["gfx"] == "gfx950" and cells["cu_num"] == "256"
        for field, value in _KEY.items():
            assert cells[field] == value

    def test_the_header_is_the_one_aiter_reads(self, monkeypatch):
        # Transcribed from a file aiter's own tuner wrote on gfx950. A column
        # out of place here is a config aiter silently declines to match.
        adapter = self._adapter(monkeypatch)
        path, _, _ = adapter._candidate_csv(_SHAPE, _CAND)
        assert path.read_text(encoding="utf-8").splitlines()[0] == (
            "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,"
            "q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,"
            "err2,us,run_1stage,xbf16,flat,tflops,bw"
        )

    def test_the_baseline_table_is_empty_not_absent(self, monkeypatch):
        # Unsetting AITER_CONFIG_FMOE would let aiter fall back to whatever
        # tuned file is installed -- on a fleet box, one we deployed. Timing a
        # candidate against our own previous answer is not a baseline.
        adapter = self._adapter(monkeypatch)
        assert adapter._baseline_csv.read_text(encoding="utf-8").strip().startswith("gfx,cu_num,token")
        assert len(adapter._baseline_csv.read_text(encoding="utf-8").splitlines()) == 1

    @pytest.mark.parametrize(
        "cand",
        [
            {"backend": "aiter_flydsl", "config": f"kernelName1={_KN1};kernelName2={_KN2}"},
            {"backend": "aiter_fmoe", "config": f"kernelName1={_KN1}"},
            {"backend": "aiter_fmoe", "config": "tile_m=32;tile_n=64"},
            {"backend": "aiter_fmoe", "config": "kernelName1=nope1;kernelName2=nope2"},
        ],
    )
    def test_a_candidate_it_cannot_dispatch_is_refused_on_paper(self, monkeypatch, cand):
        # Refused before any GPU work: an unknown backend or a missing kernel
        # name cannot become a row, and guessing what was meant would time
        # something nobody proposed.
        assert self._adapter(monkeypatch)._candidate_csv(_SHAPE, cand) is None


class TestThePairAiterWillActuallyRun:
    """Reject pairs aiter logs but silently declines to dispatch."""

    def test_a_flydsl_stage_one_carries_a_foreign_partner(self):
        # The partner is dispatched, but only from inside the branch the FlyDSL
        # name opened, so one of the two is enough.
        assert dp.aiter_honours_kernel_pair(_KN1, "cktile_moe2_something")

    def test_a_flydsl_stage_two_is_enough_on_its_own(self):
        assert dp.aiter_honours_kernel_pair("ck_moe_stage1_something", _KN2)

    def test_neither_flydsl_means_the_row_is_read_and_dropped(self):
        assert not dp.aiter_honours_kernel_pair("nope1", "nope2")

    def test_a_plausible_looking_ck_pair_is_still_refused(self):
        # Not a typo-catcher: these are names aiter uses elsewhere. They are
        # refused because on this path nothing would consult them.
        assert not dp.aiter_honours_kernel_pair("cktile_moe1_afp4", "cktile_moe2_afp4")

    def test_a_fake_flydsl_name_is_left_to_aiter_to_reject(self):
        # Let aiter own kernel-name validation so a duplicate list cannot go stale.
        assert dp.aiter_honours_kernel_pair("flydsl_moe1_no_such_tile", "flydsl_moe2_no_such_tile")


class TestAPairThatNamesOneKernelAndRunsAnother:
    """Reject silent tile substitution and incompatible ``block_m`` values."""

    def test_a_tile_that_divides_is_fine(self):
        assert (
            dp.flydsl_pair_misconfigured(
                "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2",
                "flydsl_moe2_afp4_wfp4_bf16_t32x128x128_atomic",
                32,
                384,
            )
            == ""
        )

    def test_stage_one_tile_n_that_does_not_divide_inter_dim(self):
        # 384 % 256 -- aiter's resolve_flydsl_stage1_tile_n would run 128.
        assert "stage-1 tile N=256" in dp.flydsl_pair_misconfigured(
            "flydsl_moe1_afp4_wfp4_bf16_t32x256x256",
            "flydsl_moe2_afp4_wfp4_bf16_t32x128x128_atomic",
            32,
            384,
        )

    def test_stage_two_tile_k_that_does_not_divide_inter_dim(self):
        assert "stage-2 tile K=256" in dp.flydsl_pair_misconfigured(
            "flydsl_moe1_afp4_wfp4_bf16_t32x128x256",
            "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic",
            32,
            384,
        )

    def test_the_same_tile_is_legal_when_it_does_divide(self):
        # The constraint is on the shape, not on the number: inter_dim=512
        # makes both of the above legal.
        assert (
            dp.flydsl_pair_misconfigured(
                "flydsl_moe1_afp4_wfp4_bf16_t32x256x256",
                "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic",
                32,
                512,
            )
            == ""
        )

    def test_block_m_below_the_tile(self):
        # The measured case: t128 stage 1 with the default block_m of 32 ran at
        # 71.70us against a 111.67us baseline and was wrong by 1.404.
        assert "block_m=32 is not the stage-1 tile M=128" in dp.flydsl_pair_misconfigured(
            "flydsl_moe1_afp4_wfp4_bf16_t128x128x256",
            "flydsl_moe2_afp4_wfp4_bf16_t128x128x128_atomic",
            32,
            384,
        )

    def test_block_m_matching_only_one_of_the_two(self):
        assert "stage-2 tile M=32" in dp.flydsl_pair_misconfigured(
            "flydsl_moe1_afp4_wfp4_bf16_t128x128x256",
            "flydsl_moe2_afp4_wfp4_bf16_t32x128x128_atomic",
            128,
            384,
        )

    def test_a_foreign_partner_is_not_second_guessed(self):
        # A CKTile name carries no t<M>x<N>x<K>, so there is nothing to check.
        # Skipping it beats inventing a reading of a spelling we do not own.
        assert (
            dp.flydsl_pair_misconfigured(
                "flydsl_moe1_afp4_wfp4_bf16_t32x128x256",
                "cktile_moe2_something",
                32,
                384,
            )
            == ""
        )

    def test_a_flydsl_name_with_no_tile_at_all_is_left_alone(self):
        assert dp.flydsl_tile("flydsl_moe1_no_such_tile") is None


class TestReadingBackWhatAiterServed:
    """The check that separates "ignored" from "tied"."""

    @staticmethod
    def _emitting(*messages: str):
        def run():
            for message in messages:
                logging.getLogger("aiter").info(message)

        return run

    def test_it_finds_the_kernel_pair_aiter_named(self, monkeypatch):
        adapter = dp._FusedMoeAdapter()
        monkeypatch.setattr(adapter, "_torch", staticmethod(lambda: _FakeTorch()))
        served = adapter._resolved_kernels(
            self._emitting(f"[fused_moe] using 2stage (kernelName1='{_KN1}', kernelName2='{_KN2}') for (...)")
        )
        assert served == (_KN1, _KN2)

    def test_the_heuristic_fallback_names_no_pair(self, monkeypatch):
        # This is the line the baseline produces. Reading it as a match would
        # make every unreachable key look like a candidate that ran.
        adapter = dp._FusedMoeAdapter()
        monkeypatch.setattr(adapter, "_torch", staticmethod(lambda: _FakeTorch()))
        assert (
            adapter._resolved_kernels(
                self._emitting("[fused_moe] no tuned FlyDSL config for (...), using heuristic FlyDSL fallback")
            )
            is None
        )

    def test_the_last_dispatch_wins_over_an_earlier_one(self, monkeypatch):
        adapter = dp._FusedMoeAdapter()
        monkeypatch.setattr(adapter, "_torch", staticmethod(lambda: _FakeTorch()))
        served = adapter._resolved_kernels(
            self._emitting(
                "[fused_moe] using 2stage (kernelName1='stale1', kernelName2='stale2') for (...)",
                f"[fused_moe] using 2stage (kernelName1='{_KN1}', kernelName2='{_KN2}') for (...)",
            )
        )
        assert served == (_KN1, _KN2)

    def test_it_leaves_the_aiter_logger_as_it_found_it(self, monkeypatch):
        adapter = dp._FusedMoeAdapter()
        monkeypatch.setattr(adapter, "_torch", staticmethod(lambda: _FakeTorch()))
        aiter_log = logging.getLogger("aiter")
        before = (aiter_log.level, list(aiter_log.handlers))
        adapter._resolved_kernels(self._emitting("nothing to see"))
        assert (aiter_log.level, list(aiter_log.handlers)) == before


class TestTheErrorMeasure:
    """Why fused MoE does not use the measure the dense adapter uses."""

    def test_one_ulp_at_the_peak_swamps_the_peak_measure(self):
        # A single peak perturbation reproduces the measured 0.04701 peak error
        # versus 0.00021 mean error that motivates the MoE metric.
        torch = pytest.importorskip("torch")
        ref = torch.full((512, 6144), 1.0)
        ref[0, 0] = 1248.0
        got = ref.clone()
        got[0, 0] += 8.0
        assert dp.relative_error(got, ref) > dp.MAX_RELATIVE_ERROR * 0.8
        assert dp.mean_error(got, ref) < dp.MAX_MOE_MEAN_ERROR / 100

    def test_a_broadly_wrong_answer_still_fails_the_mean_measure(self):
        # The measure has to stay able to reject. On hardware, deliberately
        # holed scales scored 0.5697 and a genuinely divergent kernel 0.1445.
        torch = pytest.importorskip("torch")
        ref = torch.full((64, 64), 1.0)
        assert dp.mean_error(ref * 1.2, ref) > dp.MAX_MOE_MEAN_ERROR


class _FakeTorch:
    """Just enough torch for the log read-back, which only needs a sync."""

    class cuda:  # noqa: N801 - mirrors the attribute path being stood in for
        @staticmethod
        def synchronize():
            return None
