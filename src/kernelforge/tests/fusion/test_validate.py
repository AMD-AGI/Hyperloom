# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for kernel-level validation (no GPU, no LLM — fake runners only)."""

from __future__ import annotations

import json

from kernelforge.fusion.models import Recipe
from kernelforge.fusion.validate import (
    BenchOutcome,
    CompileOutcome,
    HarnessKernelRunner,
    ParitySample,
    classify_bench_skip,
    classify_compile_error,
    max_abs_err,
    snr_db,
    validate_recipe,
)


def _recipe(**over) -> Recipe:
    base = dict(
        pattern_id="residual_add_rmsnorm",
        description="Fold residual-add into RMSNorm.",
        env_flag="LFM2_FUSED_RESIDUAL",
        source_file="/sgl/models/lfm2.py",
        source_hints=["+ residual", "RMSNorm("],
        fusion_math="y, residual = norm(x + residual)",
        eager_reference_hint="Import the framework RMSNorm; compare rmsnorm(x+residual).",
        shapes={"hidden_size": 2048, "T": 16},
        matched_categories=["rmsnorm"],
        trigger_share=0.3,
    )
    base.update(over)
    return Recipe(**base)


class _FakeRunner:
    """Injectable fake: hands back canned compile/parity/microbench outcomes."""

    def __init__(self, compile_out=None, parity=None, bench=None):
        self._compile = compile_out if compile_out is not None else CompileOutcome(ok=True)
        self._parity = parity if parity is not None else [ParitySample(snr_db=45.0)]
        self._bench = bench if bench is not None else BenchOutcome(eager_us=100.0, fused_us=80.0)

    def compile_check(self, recipe):
        return self._compile

    def parity_samples(self, recipe):
        return list(self._parity)

    def microbench(self, recipe):
        return self._bench


# ── metric helpers ──────────────────────────────────────────────────────────
class TestMetrics:
    def test_snr_bit_exact_is_inf(self):
        assert snr_db([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == float("inf")

    def test_snr_higher_for_closer_signals(self):
        close = snr_db([1.0, 2.0, 3.0], [1.001, 2.001, 3.001])
        far = snr_db([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        assert close > far > 0

    def test_snr_none_on_length_mismatch(self):
        assert snr_db([1.0, 2.0], [1.0]) is None
        assert snr_db([], []) is None

    def test_max_abs_err(self):
        assert max_abs_err([1.0, 2.0], [1.0, 2.5]) == 0.5
        assert max_abs_err([1.0], [1.0, 2.0]) is None


# ── ROCm failure-mode classifiers ────────────────────────────────────────────
class TestClassifiers:
    def test_cuda_only_lesson(self):
        msg = classify_compile_error("fatal error: cuda_bf16.h: No such file or directory")
        assert "CUDA-only" in msg and "Triton" in msg

    def test_triton_build_lesson(self):
        msg = classify_compile_error("triton compilation failed: out of resource: shared memory")
        assert "gfx942" in msg or "shared-memory" in msg

    def test_generic_compile_lesson(self):
        msg = classify_compile_error("ImportError: cannot import name 'foo'")
        assert "ROCm-native" in msg

    def test_bench_skip_mamba(self):
        msg = classify_bench_skip("could not init mamba causal_conv1d backend")
        assert "Mamba" in msg and "unverified" in msg

    def test_bench_skip_generic(self):
        assert "parity" in classify_bench_skip("some other reason").lower()


# ── validate_recipe orchestration ────────────────────────────────────────────
class TestValidateRecipe:
    def test_compile_failure_fails_loudly_with_cuda_lesson(self):
        runner = _FakeRunner(
            compile_out=CompileOutcome(
                ok=False, is_triton=False, error="fatal error: cuda_bf16.h not found (fused_qk_norm_rope)"
            )
        )
        vr = validate_recipe(_recipe(), runner)
        assert vr.correctness_passed is False
        assert vr.kept is False
        assert vr.kernel_speedup is None
        assert "COMPILE FAILED" in vr.note
        assert "CUDA-only" in vr.note  # first-class ROCm failure-mode lesson

    def test_parity_failure_on_low_snr(self):
        runner = _FakeRunner(parity=[ParitySample(snr_db=12.0, max_abs_err=0.5)])
        vr = validate_recipe(_recipe(), runner)
        assert vr.correctness_passed is False
        assert vr.kept is False
        assert "PARITY FAILED" in vr.note

    def test_a_timing_floor_speedup_is_refused_rather_than_kept(self):
        """One self-reported timing has no repeat to average, so the ceiling is the gate."""
        runner = _FakeRunner(bench=BenchOutcome(eager_us=100.0, fused_us=0.001))
        vr = validate_recipe(_recipe(), runner)
        assert vr.kept is False
        assert vr.correctness_passed is True
        assert "not believable" in vr.note
        assert "plausibility ceiling" in vr.note

    def test_kept_when_parity_and_speedup_pass(self):
        runner = _FakeRunner(
            parity=[ParitySample(snr_db=42.0), ParitySample(snr_db=38.0)],
            bench=BenchOutcome(eager_us=120.0, fused_us=90.0),
        )
        vr = validate_recipe(_recipe(), runner, target_speedup=1.03)
        assert vr.correctness_passed is True
        assert vr.kept is True
        assert vr.kernel_speedup and vr.kernel_speedup > 1.03
        assert vr.eager_us == 120.0 and vr.fused_us == 90.0

    def test_correct_but_too_slow_is_not_kept(self):
        runner = _FakeRunner(bench=BenchOutcome(eager_us=100.0, fused_us=99.0))
        vr = validate_recipe(_recipe(), runner, target_speedup=1.03)
        assert vr.correctness_passed is True
        assert vr.kept is False
        assert vr.kernel_speedup is not None

    def test_microbench_skipped_for_mamba_hybrid(self):
        runner = _FakeRunner(bench=BenchOutcome(skipped=True, skip_reason="mamba backend cannot init on ROCm"))
        vr = validate_recipe(_recipe(), runner)
        assert vr.correctness_passed is True  # parity still counts
        assert vr.kept is False
        assert vr.kernel_speedup is None
        assert "SKIPPED" in vr.note and "Mamba" in vr.note

    def test_rtol_fallback_when_snr_unavailable(self):
        runner = _FakeRunner(parity=[ParitySample(snr_db=None, max_abs_err=1e-3)])
        vr = validate_recipe(_recipe(), runner, rtol=2e-2)
        assert vr.correctness_passed is True  # within rtol
        # And a too-large abs error under the rtol fallback fails:
        runner2 = _FakeRunner(parity=[ParitySample(snr_db=None, max_abs_err=0.5)])
        assert validate_recipe(_recipe(), runner2, rtol=2e-2).correctness_passed is False

    def test_empty_parity_samples_fail(self):
        runner = _FakeRunner(parity=[])
        vr = validate_recipe(_recipe(), runner)
        assert vr.correctness_passed is False
        assert "PARITY UNAVAILABLE" in vr.note


# ── HarnessKernelRunner (subprocess boundary) ─────────────────────────────────
class TestHarnessKernelRunner:
    def test_missing_harness_degrades_to_compile_failure(self):
        runner = HarnessKernelRunner("/nonexistent/harness.py", workdir=".")
        comp = runner.compile_check(_recipe())
        assert comp.ok is False and "not found" in comp.error
        # And that flows into a loud validation failure (never raises):
        vr = validate_recipe(_recipe(), runner)
        assert vr.correctness_passed is False and vr.kept is False

    def test_parses_harness_json(self, tmp_path):
        harness = tmp_path / "kernel_harness.py"
        payload = {
            "compiled": True,
            "is_triton": True,
            "error": "",
            "parity": [{"snr_db": 41.0, "max_abs_err": 1e-3, "label": "T16"}],
            "eager_us": 100.0,
            "fused_us": 70.0,
            "skipped": False,
            "skip_reason": "",
        }
        # Emit the JSON payload verbatim on stdout (avoid embedding JSON true/false
        # literals in Python source, which are not valid Python identifiers).
        harness.write_text("print(%r)\n" % json.dumps(payload), encoding="utf-8")
        runner = HarnessKernelRunner(str(harness), workdir=str(tmp_path))
        vr = validate_recipe(_recipe(), runner, target_speedup=1.03)
        assert vr.correctness_passed is True
        assert vr.kept is True
        assert vr.kernel_speedup and round(vr.kernel_speedup, 2) == round(100.0 / 70.0, 2)
