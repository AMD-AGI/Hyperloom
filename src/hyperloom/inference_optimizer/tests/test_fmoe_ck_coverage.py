# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for fmoe_ck E2E coverage and apply verification (#101821/#101810).

Dense BF16 GEMM lookups must not be treated as fmoe_ck evidence. When a MoE
model also runs dense linears, the server log carries bf16_tuned_gemm.csv misses
alongside fused-MoE dispatch lines; mis-reading the former produced
``artifact_table_not_consulted`` and ``not_merged`` false positives.
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

FMOE_DISPATCH = (
    "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage ck for "
    "(256, 64, 7168, 2048, 128, 8, 'ActivationType.Swiglu', 'torch.bfloat16', "
    "'torch.float8_e4m3fn', 'torch.float8_e4m3fn', 'QuantType.per_1x128', True, False)"
)

FMOE_DISPATCH_UNCOVERED = (
    "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage ck for "
    "(256, 128, 7168, 2048, 128, 8, 'ActivationType.Swiglu', 'torch.bfloat16', "
    "'torch.float8_e4m3fn', 'torch.float8_e4m3fn', 'QuantType.per_1x128', True, False)"
)

_FMOE_TUNED_HEADER = (
    "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
    "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,kernelId,us,kernelName\n"
)
_FMOE_TUNED_ROW = (
    "gfx950,256,64,7168,2048,128,8,Silu,bf16,"
    "torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_1x128,1,0,1,10.0,tuned\n"
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
    run_dir = (
        tmp_path
        / "runs"
        / "integrate"
        / "integrate-gemm_tune_fmoe_ck"
        / "attempt-1"
    )
    run_dir.mkdir(parents=True)
    log_path = run_dir / "server.log"
    log_path.write_text(text, encoding="utf-8")
    return log_path


def _fmoe_csv(tmp_path: Path) -> Path:
    path = tmp_path / "candidate_fmoe.csv"
    path.write_text(_FMOE_TUNED_HEADER + _FMOE_TUNED_ROW, encoding="utf-8")
    return path


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

    fake = types.ModuleType("forge_gemm_tune")
    fake_ev = types.ModuleType("forge_gemm_tune.evidence")
    fake_ev.parse_log_file = _fake_parse_log_file  # type: ignore[attr-defined]
    fake.evidence = fake_ev  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "forge_gemm_tune", fake)
    monkeypatch.setitem(sys.modules, "forge_gemm_tune.evidence", fake_ev)


class TestFmoeCoverageGate:
    def test_dense_bf16_miss_plus_matching_fmoe_dispatch_is_not_table_not_consulted(
        self, tmp_path
    ):
        _integrate_log(tmp_path, "\n".join([AITER_BF16_MISS, FMOE_DISPATCH]))
        csv_path = _fmoe_csv(tmp_path)
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage("fmoe_ck", _envs(csv_path))

        assert report is not None
        assert report["artifact_applied"] is True
        assert report.get("not_applied_reason") != "artifact_table_not_consulted"

    def test_log_without_fused_moe_dispatch_blocks(self, tmp_path):
        _integrate_log(tmp_path, AITER_BF16_MISS)
        csv_path = _fmoe_csv(tmp_path)
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage("fmoe_ck", _envs(csv_path))

        assert report is not None
        assert report["artifact_applied"] is False
        assert report.get("not_applied_reason") == "no_fused_moe_dispatch"

    def test_uncovered_fmoe_dispatch_blocks(self, tmp_path):
        _integrate_log(tmp_path, FMOE_DISPATCH_UNCOVERED)
        csv_path = _fmoe_csv(tmp_path)
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage("fmoe_ck", _envs(csv_path))

        assert report is not None
        assert report["artifact_applied"] is False
        assert report.get("not_applied_reason") == "no_shape_key_matched"

    def test_missing_server_log_stays_inconclusive(self, tmp_path):
        csv_path = _fmoe_csv(tmp_path)
        phase = _phase(tmp_path)

        report = phase._gemm_tuned_config_coverage("fmoe_ck", _envs(csv_path))

        assert report is None


class TestFmoeApplyVerdict:
    def test_dense_consulted_tables_do_not_block_fmoe(self, tmp_path, stub_forge_parser):
        _integrate_log(tmp_path, "\n".join([AITER_BF16_MISS, FMOE_DISPATCH]))
        csv_path = _fmoe_csv(tmp_path)
        phase = _phase(tmp_path)

        verdict = phase._gemm_apply_verdict("fmoe_ck", _envs(csv_path))

        assert verdict is not None
        assert verdict.get("blocks_keep") is False
        assert verdict.get("verdict") != "not_merged"

    def test_no_fused_moe_dispatch_blocks_apply(self, tmp_path, stub_forge_parser):
        _integrate_log(tmp_path, AITER_BF16_MISS)
        csv_path = _fmoe_csv(tmp_path)
        phase = _phase(tmp_path)

        verdict = phase._gemm_apply_verdict("fmoe_ck", _envs(csv_path))

        assert verdict is not None
        assert verdict.get("blocks_keep") is True
        assert verdict.get("verdict") == "no_fused_moe_dispatch"
