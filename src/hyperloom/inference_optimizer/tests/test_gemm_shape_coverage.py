# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for aiter tuned-GEMM shape alignment and coverage reporting."""

from __future__ import annotations

import json

from hyperloom.orchestrator.kernel.gemm_shape_coverage import (
    aiter_lookup_keys,
    aiter_padded_m_coarse,
    aiter_padded_m_fine,
    align_shapes_to_aiter_keys,
    fmoe_dispatch_lookup_key,
    fmoe_tuned_config_coverage,
    load_shapes_json,
    parse_aiter_consulted_tables,
    parse_aiter_fused_moe_dispatches,
    parse_aiter_shape_lookups,
    tuned_config_coverage,
    tuned_csv_shapes,
    tuned_fmoe_csv_keys,
    tuned_fmoe_csv_rows,
    write_shapes_json,
)


class TestAiterPadding:
    """The padding ladder must mirror ``csrc/py_itfs_cu/gemm_common.cu``."""

    def test_fine_padding_ladder(self):
        assert aiter_padded_m_fine(1) == 16
        assert aiter_padded_m_fine(16) == 16
        assert aiter_padded_m_fine(17) == 32
        assert aiter_padded_m_fine(256) == 256
        assert aiter_padded_m_fine(257) == 288
        assert aiter_padded_m_fine(1024) == 1024
        assert aiter_padded_m_fine(1025) == 1088
        assert aiter_padded_m_fine(1076) == 1088
        assert aiter_padded_m_fine(4096) == 4096
        assert aiter_padded_m_fine(4097) == 4224
        assert aiter_padded_m_fine(7211) == 7296

    def test_coarse_padding_is_next_pow2(self):
        assert aiter_padded_m_coarse(1, 5120) == 1
        assert aiter_padded_m_coarse(1076, 5120) == 2048
        assert aiter_padded_m_coarse(8192, 5120) == 8192
        # M > 8192 with a wide N clamps to 8192 rather than 16384.
        assert aiter_padded_m_coarse(9000, 5120) == 8192
        assert aiter_padded_m_coarse(9000, 1024) == 16384

    def test_lookup_key_order_matches_runtime(self):
        assert aiter_lookup_keys((1076, 5120, 17408)) == (
            (1076, 5120, 17408),
            (1088, 5120, 17408),
            (2048, 5120, 17408),
        )


class TestAlignShapes:
    def test_raw_prefill_m_is_replaced_by_padded_keys(self):
        aligned, report = align_shapes_to_aiter_keys([(1076, 5120, 17408)])
        assert (1076, 5120, 17408) not in aligned
        assert (1088, 5120, 17408) in aligned
        assert (2048, 5120, 17408) in aligned
        assert report["observed"] == 1
        assert report["unchanged"] is False

    def test_neighbouring_runtime_m_becomes_covered(self):
        """The regression this fix targets: M drifts a few tokens between runs."""
        observed = [(1076, 5120, 17408)]
        raw_coverage = tuned_config_coverage(observed, [(1082, 5120, 17408)])
        assert raw_coverage["covered"] == 0

        aligned, _ = align_shapes_to_aiter_keys(observed)
        aligned_coverage = tuned_config_coverage(aligned, [(1082, 5120, 17408)])
        assert aligned_coverage["covered"] == 1
        assert aligned_coverage["coverage_pct"] == 100.0

    def test_every_nk_pair_keeps_a_row_under_a_tight_budget(self):
        observed = [
            (1076, 5120, 5120),
            (4142, 5120, 17408),
            (7211, 7168, 5120),
            (2087, 34816, 5120),
        ]
        aligned, report = align_shapes_to_aiter_keys(observed, max_shapes=1)
        nk_pairs = {(n, k) for _m, n, k in aligned}
        assert nk_pairs == {(5120, 5120), (5120, 17408), (7168, 5120), (34816, 5120)}
        assert report["aligned"] == len(aligned)

    def test_already_aligned_shapes_are_idempotent(self):
        aligned, _ = align_shapes_to_aiter_keys([(1088, 5120, 17408)])
        twice, _ = align_shapes_to_aiter_keys(aligned)
        assert set(twice) == set(aligned)

    def test_empty_and_invalid_input(self):
        assert align_shapes_to_aiter_keys([])[0] == []
        assert align_shapes_to_aiter_keys([(0, 5120, 5120)])[0] == []


class TestShapesJsonRoundTrip:
    def test_load_accepts_list_and_wrapped_forms(self, tmp_path):
        flat = tmp_path / "flat.json"
        flat.write_text(json.dumps([{"M": 8, "N": 16, "K": 32}]), encoding="utf-8")
        assert load_shapes_json(flat) == [(8, 16, 32)]

        wrapped = tmp_path / "wrapped.json"
        wrapped.write_text(json.dumps({"shapes": [{"m": 8, "n": 16, "k": 32}]}), encoding="utf-8")
        assert load_shapes_json(wrapped) == [(8, 16, 32)]

    def test_load_missing_file_is_empty(self, tmp_path):
        assert load_shapes_json(tmp_path / "nope.json") == []

    def test_write_then_load(self, tmp_path):
        out = write_shapes_json([(2048, 5120, 17408)], tmp_path / "sub" / "shapes.json")
        assert load_shapes_json(out) == [(2048, 5120, 17408)]


class TestServerLogParsing:
    MISS = (
        "(EngineCore pid=1) [aiter] shape is M:8192, N:7168, K:5120, not found tuned "
        "config in /x/candidate.csv, will use default config!"
    )
    MISS_QUANT = (
        "[aiter] shape is M:512, N:5120, K:17408, q_dtype_w:torch.float8_e4m3fn, "
        "not found tuned config in /x/candidate.csv, will use default config!"
    )
    HIT = (
        "[aiter] shape is M:1082, N:5120, K:17408, found padded_M: 1088, N:5120, "
        "K:17408 is tuned on cu_num = 256 in /x/candidate.csv , kernel name is k!"
    )

    def test_parses_misses_and_hits(self):
        missed, hit = parse_aiter_shape_lookups("\n".join([self.MISS, self.MISS_QUANT, self.HIT]))
        assert missed == {(8192, 7168, 5120), (512, 5120, 17408)}
        assert hit == {(1082, 5120, 17408)}

    def test_empty_log(self):
        assert parse_aiter_shape_lookups("") == (set(), set())

    def test_reports_which_table_the_runtime_consulted(self):
        """Observed on SGLang: the server resolves the bpreshuffle variant."""
        line = (
            "[aiter] shape is M:64, N:7168, K:5120, not found tuned config in "
            "/tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv, "
            "will use default config!"
        )
        assert parse_aiter_consulted_tables(line) == {
            "/tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
        }

    def test_consulted_tables_of_empty_log(self):
        assert parse_aiter_consulted_tables("") == set()


class TestFmoeDispatchParsing:
    _DISPATCH = (
        "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage "
        "(kernelName1='ck_moe_stage1_tuned', kernelName2='ck_moe_stage2_tuned') "
        "for ('gfx950', 256, 64, 7168, 2048, 128, 8, 'ActivationType.Swiglu', "
        "'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float8_e4m3fn', "
        "'QuantType.per_1x128', True, False)"
    )
    _DEFAULT = (
        "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage default for "
        "('gfx950', 256, 512, 6144, 512, 128, 4, 'ActivationType.Swiglu', "
        "'torch.bfloat16', 'torch.float4_e2m1fn_x2', 'torch.float4_e2m1fn_x2', "
        "'QuantType.per_1x32', True, False)"
    )

    def test_parses_fourteen_column_tuple_with_gfx_first(self):
        (record,) = parse_aiter_fused_moe_dispatches(self._DISPATCH)
        assert record["gfx"] == "gfx950"
        assert record["cu_num"] == "256"
        assert record["token"] == "64"
        assert record["kernelName1"] == "ck_moe_stage1_tuned"
        assert record["kernelName2"] == "ck_moe_stage2_tuned"
        assert record["descriptor"] != "default"

    def test_parses_default_descriptor(self):
        (record,) = parse_aiter_fused_moe_dispatches(self._DEFAULT)
        assert record["descriptor"] == "default"
        assert record["kernelName1"] == ""
        assert record["gfx"] == "gfx950"

    def test_exact_candidate_kernel_hit(self, tmp_path):
        path = tmp_path / "candidate_fmoe.csv"
        path.write_text(
            "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
            "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,kernelId,us,"
            "kernelName1,kernelName2\n"
            "gfx950,256,64,7168,2048,128,8,Swiglu,bf16,"
            "torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_1x128,1,0,1,10.0,"
            "ck_moe_stage1_tuned,ck_moe_stage2_tuned\n",
            encoding="utf-8",
        )
        (record,) = parse_aiter_fused_moe_dispatches(self._DISPATCH)
        rows = tuned_fmoe_csv_rows(path)
        report = fmoe_tuned_config_coverage(rows, [record])
        assert report["coverage_pct"] == 100.0

    def test_shape_overlap_without_kernel_match_is_not_served(self, tmp_path):
        path = tmp_path / "candidate_fmoe.csv"
        path.write_text(
            "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
            "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,kernelId,us,"
            "kernelName1,kernelName2\n"
            "gfx950,256,64,7168,2048,128,8,Swiglu,bf16,"
            "torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_1x128,1,0,1,10.0,"
            "other_stage1,other_stage2\n",
            encoding="utf-8",
        )
        (record,) = parse_aiter_fused_moe_dispatches(self._DISPATCH)
        report = fmoe_tuned_config_coverage(tuned_fmoe_csv_rows(path), [record])
        assert report["covered"] == 0
        assert report["kernel_name_mismatch"] == 1

    def test_csv_key_rejects_different_gfx(self, tmp_path):
        path = tmp_path / "tuned_fmoe.csv"
        path.write_text(
            "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
            "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,kernelId,us,"
            "kernelName1,kernelName2\n"
            "gfx942,256,64,7168,2048,128,8,Swiglu,bf16,"
            "torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_1x128,1,0,1,10.0,"
            "ck_moe_stage1_tuned,ck_moe_stage2_tuned\n",
            encoding="utf-8",
        )
        (record,) = parse_aiter_fused_moe_dispatches(self._DISPATCH)
        report = fmoe_tuned_config_coverage(tuned_fmoe_csv_rows(path), [record])
        assert report["covered"] == 0


class TestTunedCsvCoverage:
    HEADER = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"

    def _csv(self, tmp_path, rows):
        path = tmp_path / "tuned.csv"
        body = "\n".join(f"gfx950,256,{m},{n},{k},ck,0,0,1.0,name,1,1,0" for m, n, k in rows)
        path.write_text(f"{self.HEADER}\n{body}\n", encoding="utf-8")
        return path

    def test_reads_shape_keys(self, tmp_path):
        path = self._csv(tmp_path, [(1088, 5120, 17408)])
        assert tuned_csv_shapes(path) == {(1088, 5120, 17408)}

    def test_missing_file_is_empty(self, tmp_path):
        assert tuned_csv_shapes(tmp_path / "nope.csv") == set()

    def test_coverage_flags_an_unreachable_artifact(self, tmp_path):
        """Reproduces the observed failure: raw-M rows, drifted runtime M."""
        path = self._csv(tmp_path, [(1076, 5120, 17408), (4142, 5120, 5120)])
        report = tuned_config_coverage(
            tuned_csv_shapes(path),
            [(1082, 5120, 17408), (4112, 5120, 5120), (8192, 5120, 17408)],
        )
        assert report["covered"] == 0
        assert report["coverage_pct"] == 0.0
        assert report["tuned_rows"] == 2
        assert len(report["uncovered_sample"]) == 3

    def test_coverage_after_alignment(self, tmp_path):
        aligned, _ = align_shapes_to_aiter_keys([(1076, 5120, 17408), (4142, 5120, 5120)])
        path = self._csv(tmp_path, aligned)
        report = tuned_config_coverage(
            tuned_csv_shapes(path),
            [(1082, 5120, 17408), (4112, 5120, 5120)],
        )
        assert report["coverage_pct"] == 100.0

    def test_no_requested_shapes(self):
        report = tuned_config_coverage([(1, 2, 3)], [])
        assert report["requested"] == 0
        assert report["coverage_pct"] is None
