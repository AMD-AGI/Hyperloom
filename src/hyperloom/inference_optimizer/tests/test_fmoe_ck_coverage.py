# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for fmoe_ck E2E coverage and apply verification (#101821/#101810).

Dense BF16 GEMM lookups must not be treated as fmoe_ck evidence. Runtime
attribution requires a non-default descriptor whose kernelName1/kernelName2
pair matches the bare ``candidate_fmoe.csv`` row for the full fourteen-column
``get_2stage_cfgs`` lookup key.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.kernel import KernelPhase
from hyperloom.orchestrator.state.shared_state import SharedState

# Observed on gpt-oss-120b-style MoE runs: dense lm_head misses while MoE dispatches.
AITER_BF16_MISS = (
    "(EngineCore pid=1) [aiter] shape is M:1024, N:151936, K:5120 dtype='torch.bfloat16' "
    "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, not found tuned "
    "config in /tmp/aiter_configs/bf16_tuned_gemm.csv, will use default config! "
    "using torch solution:0"
)

# MiniMax-M3-MXFP4 baseline: runtime never selected a tuned row (#101821).
MINIMAX_DEFAULT = (
    "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage default for "
    "('gfx950', 256, 512, 6144, 512, 128, 4, 'ActivationType.Swiglu', "
    "'torch.bfloat16', 'torch.float4_e2m1fn_x2', 'torch.float4_e2m1fn_x2', "
    "'QuantType.per_1x32', True, False)"
)

# GLM-style parenthesised descriptor: parses, but keys/kernel names must match candidate.
GLM_KERNEL_DESCRIPTOR = (
    "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage "
    "(kernelName1='flydsl_moe1_afp4_wfp4_bf16_t128x128x256_fp4', "
    "kernelName2='flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce_bnt2_persist_sbm128') "
    "for ('gfx950', 256, 4096, 8192, 256, 512, 10, 'ActivationType.Silu', "
    "'torch.bfloat16', 'torch.float4_e2m1fn_x2', 'torch.float4_e2m1fn_x2', "
    "'QuantType.per_1x32', True, False)"
)

SYNTHETIC_SERVED = (
    "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage "
    "(kernelName1='ck_moe_stage1_tuned', kernelName2='ck_moe_stage2_tuned') "
    "for ('gfx950', 256, 64, 7168, 2048, 128, 8, 'ActivationType.Swiglu', "
    "'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float8_e4m3fn', "
    "'QuantType.per_1x128', True, False)"
)

SYNTHETIC_WRONG_KERNEL = (
    "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage "
    "(kernelName1='ck_moe_stage1_other', kernelName2='ck_moe_stage2_other') "
    "for ('gfx950', 256, 64, 7168, 2048, 128, 8, 'ActivationType.Swiglu', "
    "'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float8_e4m3fn', "
    "'QuantType.per_1x128', True, False)"
)

SYNTHETIC_WRONG_TOKEN = (
    "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage "
    "(kernelName1='ck_moe_stage1_tuned', kernelName2='ck_moe_stage2_tuned') "
    "for ('gfx950', 256, 128, 7168, 2048, 128, 8, 'ActivationType.Swiglu', "
    "'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float8_e4m3fn', "
    "'QuantType.per_1x128', True, False)"
)

_FMOE_HEADER = (
    "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
    "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,kernelId,us,"
    "kernelName1,kernelName2\n"
)
_CANDIDATE_ROW = (
    "gfx950,256,64,7168,2048,128,8,Swiglu,bf16,"
    "torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_1x128,1,0,1,10.0,"
    "ck_moe_stage1_tuned,ck_moe_stage2_tuned\n"
)
_GLM_SHAPE_ROW = (
    "gfx950,256,4096,8192,256,512,10,Silu,bf16,"
    "torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,1,10.0,"
    "bundled_kernel1,bundled_kernel2\n"
)
_MINIMAX_SHAPE_ROW = (
    "gfx950,256,512,6144,512,128,4,Swiglu,bf16,"
    "torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,1,10.0,"
    "bundled_kernel1,bundled_kernel2\n"
)

# Malformed fused-MoE line: marker present but tuple does not parse.
UNPARSEABLE_FMOE_MARKER = (
    "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage (kernelName1='broken') for (incomplete tuple"
)


def _phase(tmp_path: Path) -> KernelPhase:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        model_path=str(tmp_path / "model"),
        tp=1,
        framework="sglang",
        baseline_tput=100.0,
    )
    return KernelPhase(coord)


def _integrate_log(tmp_path: Path, text: str) -> Path:
    run_dir = tmp_path / "runs" / "integrate" / "integrate-gemm_tune_fmoe_ck" / "attempt-1"
    run_dir.mkdir(parents=True)
    log_path = run_dir / "server.log"
    log_path.write_text(text, encoding="utf-8")
    return log_path


def _candidate_csv(tmp_path: Path, row: str = _CANDIDATE_ROW) -> Path:
    path = tmp_path / "candidate_fmoe.csv"
    path.write_text(_FMOE_HEADER + row, encoding="utf-8")
    return path


def _merged_env(tmp_path: Path, *, candidate_row: str, merged_row: str) -> dict[str, str]:
    candidate = _candidate_csv(tmp_path, candidate_row)
    merged = tmp_path / "merged_candidate_fmoe.csv"
    merged.write_text(_FMOE_HEADER + merged_row, encoding="utf-8")
    return {"AITER_CONFIG_FMOE": str(merged), "AITER_LOG_TUNED_CONFIG": "1"}


def _envs(csv_path: Path) -> dict[str, str]:
    return {"AITER_CONFIG_FMOE": str(csv_path), "AITER_LOG_TUNED_CONFIG": "1"}


@pytest.fixture
def stub_forge_parser(monkeypatch):
    """Minimal forge parser so apply verification runs without forge installed."""

    def _fake_parse_log_file(path):
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        hits = misses = 0
        merged: list[str] = []
        consulted: set[str] = set()
        for line in text.splitlines():
            if "merge tuned file" in line:
                merged.extend(p for p in line.split()[-1].split(":") if p)
            elif "not found tuned config in" in line:
                misses += 1
                consulted.add(line.split("not found tuned config in")[1].split(",")[0].strip())
            elif "found padded_M" in line:
                hits += 1
        return {
            "apply_verdict": {"hit": hits, "miss": misses},
            "merged_tables": sorted(set(merged)),
            "consulted_tables": sorted(consulted),
        }

    fake = types.ModuleType("kernelforge.gemm_tune")
    fake_ev = types.ModuleType("kernelforge.gemm_tune.evidence")
    fake_ev.parse_log_file = _fake_parse_log_file  # type: ignore[attr-defined]
    fake.evidence = fake_ev  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kernelforge.gemm_tune", fake)
    monkeypatch.setitem(sys.modules, "kernelforge.gemm_tune.evidence", fake_ev)


class TestFmoeCoverageGate:
    def test_dense_bf16_miss_does_not_trigger_table_not_consulted(self, tmp_path):
        _integrate_log(tmp_path, "\n".join([AITER_BF16_MISS, SYNTHETIC_SERVED]))
        csv_path = _candidate_csv(tmp_path)
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage("fmoe_ck", _envs(csv_path))

        assert report is not None
        assert report["artifact_applied"] is True
        assert report.get("not_applied_reason") != "artifact_table_not_consulted"

    def test_minimax_default_blocks_even_when_merged_csv_has_shape(self, tmp_path):
        _integrate_log(tmp_path, MINIMAX_DEFAULT)
        env = _merged_env(tmp_path, candidate_row=_CANDIDATE_ROW, merged_row=_MINIMAX_SHAPE_ROW)
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage("fmoe_ck", env)

        assert report is not None
        assert report["artifact_applied"] is False
        assert report["not_applied_reason"] == "runtime_default_config"

    def test_glm_descriptor_parses_but_bare_candidate_mismatch_blocks(self, tmp_path):
        _integrate_log(tmp_path, GLM_KERNEL_DESCRIPTOR)
        env = _merged_env(tmp_path, candidate_row=_CANDIDATE_ROW, merged_row=_GLM_SHAPE_ROW)
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage("fmoe_ck", env)

        assert report is not None
        assert report["artifact_applied"] is False
        assert report["not_applied_reason"] in {
            "no_shape_key_matched",
            "kernel_name_mismatch",
        }

    def test_exact_candidate_kernel_hit_is_served(self, tmp_path):
        _integrate_log(tmp_path, SYNTHETIC_SERVED)
        csv_path = _candidate_csv(tmp_path)
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage("fmoe_ck", _envs(csv_path))

        assert report is not None
        assert report["artifact_applied"] is True
        assert report["covered"] == 1

    def test_fourteen_col_key_or_kernel_mismatch_blocks(self, tmp_path):
        _integrate_log(
            tmp_path,
            "\n".join([SYNTHETIC_WRONG_KERNEL, SYNTHETIC_WRONG_TOKEN]),
        )
        csv_path = _candidate_csv(tmp_path)
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage("fmoe_ck", _envs(csv_path))

        assert report is not None
        assert report["artifact_applied"] is False
        assert report["covered"] == 0

    def test_log_without_fused_moe_dispatch_blocks(self, tmp_path):
        _integrate_log(tmp_path, AITER_BF16_MISS)
        csv_path = _candidate_csv(tmp_path)
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage("fmoe_ck", _envs(csv_path))

        assert report is not None
        assert report["artifact_applied"] is False
        assert report.get("not_applied_reason") == "no_fused_moe_dispatch"

    def test_missing_server_log_stays_inconclusive(self, tmp_path):
        csv_path = _candidate_csv(tmp_path)
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage("fmoe_ck", _envs(csv_path))

        assert report is None

    def test_merged_env_without_bare_candidate_is_inconclusive(self, tmp_path):
        _integrate_log(tmp_path, SYNTHETIC_SERVED)
        merged = tmp_path / "merged_candidate_fmoe.csv"
        merged.write_text(_FMOE_HEADER + _CANDIDATE_ROW, encoding="utf-8")
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage(
            "fmoe_ck",
            {"AITER_CONFIG_FMOE": str(merged), "AITER_LOG_TUNED_CONFIG": "1"},
        )

        assert report is not None
        assert report["artifact_applied"] is False
        assert report["not_applied_reason"] == "candidate_csv_missing"
        assert report.get("conclusive") is False

    def test_merged_tuned_fmoe_resolves_bare_sibling(self, tmp_path):
        _integrate_log(tmp_path, SYNTHETIC_SERVED)
        bare = tmp_path / "tuned_fmoe.csv"
        bare.write_text(_FMOE_HEADER + _CANDIDATE_ROW, encoding="utf-8")
        merged = tmp_path / "merged_tuned_fmoe.csv"
        merged.write_text(_FMOE_HEADER + _CANDIDATE_ROW, encoding="utf-8")
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage(
            "fmoe_ck",
            {"AITER_CONFIG_FMOE": str(merged), "AITER_LOG_TUNED_CONFIG": "1"},
        )

        assert report is not None
        assert report["artifact_applied"] is True
        assert report["candidate_csv"] == str(bare)

    def test_fused_moe_marker_without_parse_is_inconclusive(self, tmp_path):
        _integrate_log(tmp_path, UNPARSEABLE_FMOE_MARKER)
        csv_path = _candidate_csv(tmp_path)
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage("fmoe_ck", _envs(csv_path))

        assert report is not None
        assert report["artifact_applied"] is False
        assert report["not_applied_reason"] == "fused_moe_parse_inconclusive"
        assert report.get("conclusive") is False

    def test_minimax_default_embedded_replay_blocks(self, tmp_path):
        from hyperloom.orchestrator.kernel.gemm_shape_coverage import (
            fmoe_tuned_config_coverage,
            parse_aiter_fused_moe_dispatches,
            tuned_fmoe_csv_rows,
        )

        candidate = tmp_path / "candidate_fmoe.csv"
        candidate.write_text(_FMOE_HEADER + _MINIMAX_SHAPE_ROW, encoding="utf-8")
        dispatches = parse_aiter_fused_moe_dispatches(MINIMAX_DEFAULT)
        assert dispatches
        assert all(d["descriptor"] == "default" for d in dispatches)

        report = fmoe_tuned_config_coverage(tuned_fmoe_csv_rows(candidate), dispatches)
        assert report["covered"] == 0
        assert report["runtime_default"] >= 1


class TestFmoeApplyVerdict:
    def test_dense_consulted_tables_do_not_block_fmoe(self, tmp_path, stub_forge_parser):
        _integrate_log(tmp_path, "\n".join([AITER_BF16_MISS, SYNTHETIC_SERVED]))
        csv_path = _candidate_csv(tmp_path)
        phase = _phase(tmp_path)

        verdict = phase._gemm_apply_verdict("fmoe_ck", _envs(csv_path))

        assert verdict is not None
        assert verdict.get("blocks_keep") is False
        assert verdict.get("verdict") == "served"

    def test_minimax_default_blocks_keep(self, tmp_path, stub_forge_parser):
        _integrate_log(tmp_path, MINIMAX_DEFAULT)
        env = _merged_env(tmp_path, candidate_row=_CANDIDATE_ROW, merged_row=_MINIMAX_SHAPE_ROW)
        phase = _phase(tmp_path)

        verdict = phase._gemm_apply_verdict("fmoe_ck", env)

        assert verdict is not None
        assert verdict.get("blocks_keep") is True
        assert verdict.get("verdict") == "runtime_default_config"

    def test_no_fused_moe_dispatch_blocks_apply(self, tmp_path, stub_forge_parser):
        _integrate_log(tmp_path, AITER_BF16_MISS)
        csv_path = _candidate_csv(tmp_path)
        phase = _phase(tmp_path)

        verdict = phase._gemm_apply_verdict("fmoe_ck", _envs(csv_path))

        assert verdict is not None
        assert verdict.get("blocks_keep") is True
        assert verdict.get("verdict") == "no_fused_moe_dispatch"

    def test_merged_without_bare_candidate_does_not_block_keep(self, tmp_path):
        _integrate_log(tmp_path, SYNTHETIC_SERVED)
        merged = tmp_path / "merged_candidate_fmoe.csv"
        merged.write_text(_FMOE_HEADER + _CANDIDATE_ROW, encoding="utf-8")
        phase = _phase(tmp_path)

        verdict = phase._gemm_apply_verdict(
            "fmoe_ck",
            {"AITER_CONFIG_FMOE": str(merged), "AITER_LOG_TUNED_CONFIG": "1"},
        )

        assert verdict is not None
        assert verdict.get("blocks_keep") is False
        assert verdict.get("conclusive") is False
        assert verdict.get("verdict") == "candidate_csv_missing"

    def test_fused_moe_marker_without_parse_does_not_block_keep(self, tmp_path):
        _integrate_log(tmp_path, UNPARSEABLE_FMOE_MARKER)
        csv_path = _candidate_csv(tmp_path)
        phase = _phase(tmp_path)

        verdict = phase._gemm_apply_verdict("fmoe_ck", _envs(csv_path))

        assert verdict is not None
        assert verdict.get("blocks_keep") is False
        assert verdict.get("conclusive") is False
        assert verdict.get("verdict") == "fused_moe_parse_inconclusive"

    def test_logging_disabled_without_dispatch_is_inconclusive(self, tmp_path):
        _integrate_log(tmp_path, AITER_BF16_MISS)
        csv_path = _candidate_csv(tmp_path)
        phase = _phase(tmp_path)

        verdict = phase._gemm_apply_verdict(
            "fmoe_ck",
            {"AITER_CONFIG_FMOE": str(csv_path), "AITER_LOG_TUNED_CONFIG": "0"},
        )

        assert verdict is not None
        assert verdict.get("blocks_keep") is False
        assert verdict.get("conclusive") is False
        assert verdict.get("verdict") == "fused_moe_logging_disabled"


class TestFmoeEmbeddedReplay:
    def test_glm_descriptor_embedded_replay_does_not_hit_candidate(self, tmp_path):
        from hyperloom.orchestrator.kernel.gemm_shape_coverage import (
            fmoe_tuned_config_coverage,
            parse_aiter_fused_moe_dispatches,
            tuned_fmoe_csv_rows,
        )

        candidate = tmp_path / "candidate_fmoe.csv"
        candidate.write_text(_FMOE_HEADER + _CANDIDATE_ROW, encoding="utf-8")
        dispatches = parse_aiter_fused_moe_dispatches(GLM_KERNEL_DESCRIPTOR)
        assert dispatches
        assert all(d["descriptor"] != "default" for d in dispatches)

        report = fmoe_tuned_config_coverage(tuned_fmoe_csv_rows(candidate), dispatches)
        assert report["covered"] == 0
