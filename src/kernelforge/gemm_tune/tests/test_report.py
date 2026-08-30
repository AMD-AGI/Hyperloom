# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for report module."""

from kernelforge.gemm_tune.model_analyzer import ModelProfile
from kernelforge.gemm_tune.tuners.base import TuneResult
from kernelforge.gemm_tune.report import build_report


def _make_profile():
    return ModelProfile(model_path="/fake/model", hidden_size=4096, intermediate_size=11008)


class TestBuildReport:
    def test_all_skipped(self):
        report = build_report(
            results=[],
            skipped=[("fmoe_ck", "not MoE")],
            profile=_make_profile(),
            framework="sglang",
            precision="bf16",
            quant_type="none",
            gpu_type="mi300x",
            tp=1,
            conc=64,
            tokens=[64, 128],
            started_at="2026-01-01T00:00:00Z",
            total_elapsed_s=1.0,
        )
        assert report.status == "skipped"
        assert report.micro_decision == "skipped"

    def test_candidate(self):
        result = TuneResult(
            tuner_name="fmoe_ck",
            status="ok",
            artifact_path="/path/to/csv",
            env_var="AITER_CONFIG_FMOE",
            env_value="/path/to/csv",
            improved_shapes=2,
            total_shapes=5,
            best_micro_speedup=1.15,
            avg_micro_speedup=1.08,
        )
        report = build_report(
            results=[result],
            skipped=[],
            profile=_make_profile(),
            framework="sglang",
            precision="bf16",
            quant_type="none",
            gpu_type="mi300x",
            tp=1,
            conc=256,
            tokens=[64, 128, 256],
            started_at="2026-01-01T00:00:00Z",
            total_elapsed_s=70.0,
        )
        assert report.status == "ok"
        assert report.micro_decision == "candidate"
        assert report.requires_e2e_validation is True
        assert "AITER_CONFIG_FMOE" in report.recommended_env

    def test_candidate_with_env_vars(self):
        result = TuneResult(
            tuner_name="vllm_dense_tunableop",
            status="ok",
            artifact_path="/path/to/tunableop_results.csv",
            env_vars={
                "PYTHONPATH": "/path/to/runtime_sitecustomize",
                "HL_TUNABLEOP_MODE": "candidate",
                "HL_TUNABLEOP_FILE": "/path/to/tunableop_results.csv",
            },
            improved_shapes=3,
            total_shapes=3,
            candidate=True,
        )
        report = build_report(
            results=[result],
            skipped=[],
            profile=_make_profile(),
            framework="vllm",
            precision="bf16",
            quant_type="none",
            gpu_type="mi300x",
            tp=1,
            conc=64,
            tokens=[64],
            started_at="2026-01-01T00:00:00Z",
            total_elapsed_s=80.0,
        )
        assert report.micro_decision == "candidate"
        assert report.recommended_env["HL_TUNABLEOP_MODE"] == "candidate"
        assert report.recommended_env["PYTHONPATH"] == "/path/to/runtime_sitecustomize"

    def test_tune_result_serializes_env_vars_and_candidate(self):
        result = TuneResult(
            tuner_name="vllm_dense_tunableop",
            status="ok",
            env_vars={"HL_TUNABLEOP_MODE": "candidate"},
            candidate=True,
        )
        data = result.to_dict()
        assert data["env_vars"] == {"HL_TUNABLEOP_MODE": "candidate"}
        assert data["candidate"] is True

    def test_no_improvement(self):
        result = TuneResult(
            tuner_name="a8w8_blockscale",
            status="ok",
            improved_shapes=0,
            total_shapes=10,
            best_micro_speedup=1.0,
            avg_micro_speedup=1.0,
        )
        report = build_report(
            results=[result],
            skipped=[],
            profile=_make_profile(),
            framework="sglang",
            precision="fp8",
            quant_type="blockscale",
            gpu_type="mi300x",
            tp=1,
            conc=64,
            tokens=[64],
            started_at="2026-01-01T00:00:00Z",
            total_elapsed_s=120.0,
        )
        assert report.status == "ok"
        assert report.micro_decision == "no_improvement"
        assert report.requires_e2e_validation is False

    def test_forced_candidate_promoted_despite_no_improvement(self):
        # split-K CSV: microbench shows no improvement (best=1.0) but candidate=True
        # forces e2e validation + deployment (recommended_env populated). Guards the
        # promotion gate against silently dropping a real e2e-only gain.
        result = TuneResult(
            tuner_name="a8w8_blockscale",
            status="no_improvement",
            artifact_path="/path/to/splitk.csv",
            env_var="AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
            env_value="/path/to/splitk.csv",
            improved_shapes=0,
            total_shapes=48,
            best_micro_speedup=1.0,
            avg_micro_speedup=1.0,
            candidate=True,
        )
        report = build_report(
            results=[result],
            skipped=[],
            profile=_make_profile(),
            framework="vllm-aiter",
            precision="fp8",
            quant_type="blockscale",
            gpu_type="mi355x",
            tp=1,
            conc=64,
            tokens=[64],
            started_at="2026-01-01T00:00:00Z",
            total_elapsed_s=120.0,
        )
        assert report.requires_e2e_validation is True
        assert "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE" in report.recommended_env

    def test_partial_failure_not_fatal(self):
        """One tuner fails, another succeeds with no improvement.

        The batch is not fatal -- status stays "ok" -- but the crash must not be
        rounded down to "no_improvement" either. That rounding is what let 14
        hard failures read as "this model has no headroom" for a week.
        """
        failed = TuneResult(tuner_name="vllm_moe_triton", status="failed", error="unsupported", error_class="api_error")
        ok = TuneResult(
            tuner_name="fmoe_ck",
            status="ok",
            improved_shapes=0,
            total_shapes=3,
            best_micro_speedup=1.0,
            avg_micro_speedup=1.0,
        )
        report = build_report(
            results=[failed, ok],
            skipped=[],
            profile=_make_profile(),
            framework="sglang",
            precision="bf16",
            quant_type="none",
            gpu_type="mi300x",
            tp=1,
            conc=64,
            tokens=[64],
            started_at="2026-01-01T00:00:00Z",
            total_elapsed_s=10.0,
        )
        assert report.status == "ok"
        assert report.micro_decision == "partial_failure"
        assert [f["tuner"] for f in report.failed_tuners] == ["vllm_moe_triton"]
        assert report.failed_tuners[0]["error_class"] == "api_error"

    def test_all_failed(self):
        failed1 = TuneResult(tuner_name="t1", status="failed", error="err1")
        failed2 = TuneResult(tuner_name="t2", status="failed", error="err2")
        report = build_report(
            results=[failed1, failed2],
            skipped=[],
            profile=_make_profile(),
            framework="sglang",
            precision="bf16",
            quant_type="none",
            gpu_type="mi300x",
            tp=1,
            conc=64,
            tokens=[64],
            started_at="2026-01-01T00:00:00Z",
            total_elapsed_s=5.0,
        )
        assert report.status == "failed"
        assert report.micro_decision == "failed"

    def test_candidate_wins_over_failure(self):
        """If one tuner has a candidate, overall is 'candidate' even if another failed."""
        failed = TuneResult(tuner_name="t1", status="failed", error="err")
        candidate = TuneResult(
            tuner_name="fmoe_ck",
            status="ok",
            artifact_path="/csv",
            env_var="AITER_CONFIG_FMOE",
            env_value="/csv",
            improved_shapes=1,
            total_shapes=3,
            best_micro_speedup=1.1,
        )
        report = build_report(
            results=[failed, candidate],
            skipped=[],
            profile=_make_profile(),
            framework="sglang",
            precision="bf16",
            quant_type="none",
            gpu_type="mi300x",
            tp=1,
            conc=64,
            tokens=[64],
            started_at="2026-01-01T00:00:00Z",
            total_elapsed_s=80.0,
        )
        assert report.status == "ok"
        assert report.micro_decision == "candidate"
