# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for aiter tuned-GEMM shape alignment and coverage reporting.

Also covers the fail-open guards around that reporting: its verdict can block a
KEEP, so every way it can fail to reach one has to degrade to "undetermined"
rather than to "the artifact did not apply".
"""

from __future__ import annotations

import json
import os

import pytest

from hyperloom.orchestrator.kernel.gemm_shape_coverage import (
    aiter_lookup_keys,
    aiter_padded_m_coarse,
    aiter_padded_m_fine,
    align_shapes_to_aiter_keys,
    fmoe_tuned_config_coverage,
    load_shapes_json,
    parse_aiter_consulted_tables,
    parse_aiter_fused_moe_dispatches,
    parse_aiter_shape_lookups,
    resolve_fmoe_candidate_csv,
    tuned_config_coverage,
    tuned_csv_shapes,
    tuned_fmoe_csv_rows,
    write_shapes_json,
    _normalize_fmoe_q_dtype,
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
    # Verbatim from a production Qwen3 vLLM server.log. The dispatch kwargs sit
    # between ``K:`` and ``found padded_M:``, which an earlier pattern did not
    # allow -- it matched none of that log's 5024 hit lines.
    HIT_WITH_KWARGS = (
        "(Worker_TP7 pid=380239) [aiter] shape is M:16384, N:4608, K:8192 "
        "dtype='torch.bfloat16' otype='torch.bfloat16' bias=False, scaleAB=False, "
        "bpreshuffle=False found padded_M: 8192, N:4608, K:8192 is tuned on "
        "cu_num = 256 in /shared_nfs/x/merged_tuned_dense_bf16.csv,"
    )

    def test_parses_misses_and_hits(self):
        missed, hit = parse_aiter_shape_lookups("\n".join([self.MISS, self.MISS_QUANT, self.HIT]))
        assert missed == {(8192, 7168, 5120), (512, 5120, 17408)}
        assert hit == {(1082, 5120, 17408)}

    def test_parses_hit_with_dispatch_kwargs_between_k_and_padded_m(self):
        missed, hit = parse_aiter_shape_lookups(self.HIT_WITH_KWARGS)
        assert missed == set()
        assert hit == {(16384, 4608, 8192)}

    def test_hit_pattern_does_not_span_lines(self):
        """A miss on one line must not pair with a hit on the next."""
        missed, hit = parse_aiter_shape_lookups("\n".join([self.MISS, self.HIT_WITH_KWARGS]))
        assert missed == {(8192, 7168, 5120)}
        assert hit == {(16384, 4608, 8192)}

    def test_empty_log(self):
        assert parse_aiter_shape_lookups("") == (set(), set())

    def test_reports_which_table_the_runtime_consulted(self):
        """Observed on SGLang: the server resolves the bpreshuffle variant."""
        line = (
            "[aiter] shape is M:64, N:7168, K:5120, not found tuned config in "
            "/tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv, "
            "will use default config!"
        )
        assert parse_aiter_consulted_tables(line) == {"/tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"}

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

    def test_resolve_merged_candidate_fmoe_csv(self, tmp_path):
        bare = tmp_path / "candidate_fmoe.csv"
        bare.write_text("gfx\n", encoding="utf-8")
        merged = tmp_path / "merged_candidate_fmoe.csv"
        merged.write_text("gfx\n", encoding="utf-8")
        assert resolve_fmoe_candidate_csv(merged) == bare

    def test_resolve_merged_tuned_fmoe_csv(self, tmp_path):
        bare = tmp_path / "tuned_fmoe.csv"
        bare.write_text("gfx\n", encoding="utf-8")
        merged = tmp_path / "merged_tuned_fmoe.csv"
        merged.write_text("gfx\n", encoding="utf-8")
        assert resolve_fmoe_candidate_csv(merged) == bare

    def test_q_dtype_aliases_normalize_known_forms_only(self, tmp_path):
        assert _normalize_fmoe_q_dtype("torch.float8_e4m3fn") == "torch.float8_e4m3fn"
        assert _normalize_fmoe_q_dtype("torch.float8_e5m2") == "torch.float8_e5m2"
        assert _normalize_fmoe_q_dtype("torch.float8_e4m3fnuz") == "torch.float8_e4m3fnuz"
        assert _normalize_fmoe_q_dtype("torch.float4_e2m1fn_x2") == "torch.float4_e2m1fn_x2"
        assert _normalize_fmoe_q_dtype("fp4") == "torch.float4_e2m1fn_x2"
        assert _normalize_fmoe_q_dtype("torch.float8_e5m2fn") == "torch.float8_e5m2fn"

        path = tmp_path / "candidate_fmoe.csv"
        path.write_text(
            "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
            "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,kernelId,us,"
            "kernelName1,kernelName2\n"
            "gfx950,256,64,7168,2048,128,8,Swiglu,bf16,"
            "fp4,fp4,QuantType.per_1x128,1,0,1,10.0,"
            "ck_moe_stage1_tuned,ck_moe_stage2_tuned\n",
            encoding="utf-8",
        )
        rows = tuned_fmoe_csv_rows(path)
        assert rows[0]["q_dtype_a"] == "torch.float4_e2m1fn_x2"
        assert rows[0]["q_dtype_w"] == "torch.float4_e2m1fn_x2"


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

    def test_an_fmoe_csv_yields_no_dense_shapes(self, tmp_path):
        """An MoE table has no M,N,K columns; reading one as dense would invent
        shapes and report coverage against a schema it never described."""
        path = tmp_path / "tuned_fmoe.csv"
        path.write_text(
            "token,model_dim,inter_dim,expert,topk,act_type,dtype,"
            "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,kernelName\n"
            "256,4096,512,256,6,ActivationType.Silu,torch.bfloat16,"
            "torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,kernel_a\n",
            encoding="utf-8",
        )
        assert tuned_csv_shapes(path) == set()

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

    def test_retry_integrate_dir_is_scanned(self, tmp_path):
        """A ``-2`` retry replaces the first attempt; it must not be ignored."""
        from hyperloom.orchestrator.kernel.gemm_shape_coverage import read_latest_integrate_server_log

        integrate = tmp_path / "runs" / "integrate"
        first = integrate / "integrate-gemm_tune_fmoe_ck" / "r"
        retry = integrate / "integrate-gemm_tune_fmoe_ck-2" / "r"
        for d in (first, retry):
            d.mkdir(parents=True)
        (first / "server.log").write_text("first attempt", encoding="utf-8")
        (retry / "server.log").write_text("retry attempt", encoding="utf-8")
        os.utime(retry / "server.log", (2_000_000_000, 2_000_000_000))
        os.utime(first / "server.log", (1_000_000_000, 1_000_000_000))

        found = read_latest_integrate_server_log(tmp_path, "integrate-gemm_tune_fmoe_ck")
        assert found is not None
        assert found[1] == "retry attempt"

    def test_runtime_confirmed_hit_counts_as_covered(self):
        """aiter saying it used a row beats our replay of its padding rules."""
        requested = [(16384, 4608, 8192)]
        # Deliberately empty: the ladder can prove nothing here.
        assert tuned_config_coverage([], requested)["covered"] == 0
        report = tuned_config_coverage([], requested, known_covered=requested)
        assert report["covered"] == 1
        assert report["coverage_pct"] == 100.0
        assert report["uncovered_sample"] == []


class TestCoverageGateDoesNotBlockOnMissingEvidence:
    """The coverage report can block a KEEP, so it must never guess.

    A report of 0% is a claim the runtime could not reach the tuned rows. When
    the CSV yields no keys at all, we have not established that -- we have
    failed to read our own artifact. Reporting it as 0% lets an unreadable file
    revert a candidate whose throughput genuinely improved, which is the exact
    conflation this change set exists to remove.
    """

    ENVS = {"AITER_CONFIG_GEMM": ""}
    LOOKUP_LINE = (
        "[aiter] shape is M:1082, N:5120, K:17408, not found tuned config in /x/candidate.csv, will use default config!"
    )
    HEADER = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"

    def _phase(self, tmp_path):
        from types import SimpleNamespace

        run_dir = tmp_path / "runs" / "integrate" / "integrate-gemm_tune_aiter_dense"
        run_dir.mkdir(parents=True)
        (run_dir / "server.log").write_text(self.LOOKUP_LINE + "\n", encoding="utf-8")
        return SimpleNamespace(session_dir=tmp_path)

    def _call(self, phase, csv_path):
        """Exercise the body directly, so a bound-method slip cannot fake a pass."""
        from hyperloom.orchestrator.phases.kernel import KernelPhase

        return KernelPhase._gemm_tuned_config_coverage_impl(phase, "aiter_dense", {"AITER_CONFIG_GEMM": str(csv_path)})

    def test_unreadable_csv_is_undetermined_not_zero_coverage(self, tmp_path):
        phase = self._phase(tmp_path)
        empty = tmp_path / "candidate.csv"
        empty.write_text("", encoding="utf-8")

        assert self._call(phase, empty) is None

    def test_missing_csv_is_undetermined(self, tmp_path):
        phase = self._phase(tmp_path)

        assert self._call(phase, tmp_path / "absent.csv") is None

    def test_csv_without_shape_columns_is_undetermined(self, tmp_path):
        phase = self._phase(tmp_path)
        odd = tmp_path / "candidate.csv"
        odd.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

        assert self._call(phase, odd) is None

    def test_readable_csv_with_wrong_keys_still_reports_zero(self, tmp_path):
        """Fail-open on unreadable input must not weaken the real check."""
        phase = self._phase(tmp_path)
        wrong = tmp_path / "candidate.csv"
        wrong.write_text(
            f"{self.HEADER}\ngfx950,256,4096,5120,5120,ck,0,0,1.0,name,1,1,0\n",
            encoding="utf-8",
        )

        report = self._call(phase, wrong)
        assert report is not None
        assert report["artifact_applied"] is False
        assert report["coverage_pct"] == 0.0

    def test_matching_csv_reports_applied(self, tmp_path):
        phase = self._phase(tmp_path)
        good = tmp_path / "candidate.csv"
        good.write_text(
            f"{self.HEADER}\ngfx950,256,1088,5120,17408,ck,0,0,1.0,name,1,1,0\n",
            encoding="utf-8",
        )

        report = self._call(phase, good)
        assert report is not None
        assert report["artifact_applied"] is True
        assert report["coverage_pct"] == 100.0

    def test_unexpected_failure_is_undetermined(self, tmp_path):
        """The wrapper swallows anything the body throws (it can block a KEEP)."""
        from types import SimpleNamespace

        from hyperloom.orchestrator.phases.kernel import KernelPhase

        def _boom(*_args, **_kwargs):
            raise RuntimeError("coverage exploded")

        phase = SimpleNamespace(
            session_dir=tmp_path,
            _gemm_tuned_config_coverage_impl=_boom,
        )

        assert KernelPhase._gemm_tuned_config_coverage(phase, "fmoe_ck", self.ENVS) is None


class TestSafeMtime:
    def test_missing_path_sorts_last_instead_of_raising(self, tmp_path):
        from hyperloom.orchestrator.phases.kernel import _safe_mtime

        assert _safe_mtime(tmp_path / "gone.log") == 0.0

    def test_existing_path_returns_its_mtime(self, tmp_path):
        from hyperloom.orchestrator.phases.kernel import _safe_mtime

        path = tmp_path / "server.log"
        path.write_text("x", encoding="utf-8")
        assert _safe_mtime(path) == path.stat().st_mtime


class TestE2EValidationFailsOpen:
    """E2E validation owns the coverage check, so its own failure cannot escape.

    Both entrypoints into gemm tuning guard only the tuning call, not the
    validation that follows it. An exception escaping here takes the KERNEL
    phase down over a candidate that simply went unmeasured.
    """

    def _phase(self, tmp_path, validate):
        from types import SimpleNamespace

        recorded: list[dict] = []
        saved: list[object] = []
        state = SimpleNamespace(
            record_gemm_tuning=recorded.append,
            save=saved.append,
            macro_cycle=0,
        )
        return SimpleNamespace(
            session_dir=tmp_path,
            shared_state=state,
            _sync_profile_state_after_gemm_roofline=lambda _r: None,
            _validate_gemm_tuning_e2e=validate,
        ), recorded

    @pytest.mark.asyncio
    async def test_exception_is_recorded_as_a_fault_not_raised(self, tmp_path):
        from hyperloom.orchestrator.phases.kernel import KernelPhase

        async def _boom(_result):
            raise RuntimeError("e2e exploded")

        phase, recorded = self._phase(tmp_path, _boom)
        result: dict = {"backend": "forge"}

        await KernelPhase._handle_gemm_tuning_result(phase, result)

        assert recorded == [result]
        fault = result["e2e_results"]["faults"][0]
        assert fault["error_class"] == "e2e_validation_exception"
        assert "RuntimeError: e2e exploded" in fault["error"]

    @pytest.mark.asyncio
    async def test_the_unmeasured_envelope_is_neutralised(self, tmp_path):
        """An arm that raised was never measured, so it must not read as a KEEP.

        Recording the fault while leaving the bridge's KEEP envelope in place
        would let Orchestration bundle an integrate against it.
        """
        from hyperloom.orchestrator.phases.kernel import KernelPhase

        async def _boom(_result):
            raise RuntimeError("e2e exploded")

        phase, _ = self._phase(tmp_path, _boom)
        result: dict = {
            "backend": "forge",
            "decision": "KEEP",
            "requires_e2e_validation": True,
            "recommended_env": {"AITER_CONFIG_FMOE": "/ws/tuned_fmoe.csv"},
            "extra_envs": {"AITER_CONFIG_FMOE": "/ws/tuned_fmoe.csv"},
        }

        await KernelPhase._handle_gemm_tuning_result(phase, result)

        assert result["decision"] == "REVERT"
        assert result["requires_e2e_validation"] is False
        assert result["e2e_validated"] is False
        assert not result["recommended_env"]
        assert not result["extra_envs"]
        # The reason still has to be legible, not just absent.
        assert result["e2e_results"]["faults"][0]["error_class"] == "e2e_validation_exception"

    @pytest.mark.asyncio
    async def test_existing_faults_are_preserved(self, tmp_path):
        from hyperloom.orchestrator.phases.kernel import KernelPhase

        async def _boom(_result):
            raise ValueError("second failure")

        phase, _ = self._phase(tmp_path, _boom)
        result: dict = {
            "backend": "forge",
            "e2e_results": {"faults": [{"tuner": "fmoe_ck", "error_class": "server_died"}]},
        }

        await KernelPhase._handle_gemm_tuning_result(phase, result)

        faults = result["e2e_results"]["faults"]
        assert [f["error_class"] for f in faults] == [
            "server_died",
            "e2e_validation_exception",
        ]

    @pytest.mark.asyncio
    async def test_success_path_adds_no_fault(self, tmp_path):
        from hyperloom.orchestrator.phases.kernel import KernelPhase

        async def _ok(_result):
            return None

        phase, recorded = self._phase(tmp_path, _ok)
        result: dict = {"backend": "forge"}

        await KernelPhase._handle_gemm_tuning_result(phase, result)

        assert recorded == [result]
        assert "e2e_results" not in result
