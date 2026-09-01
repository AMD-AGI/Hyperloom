# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for author-prompt assembly (no LLM, no GPU)."""

from __future__ import annotations

from kernelforge.fusion.author import build_author_prompt


def _recipe():
    return {
        "pattern": "residual_add_rmsnorm",
        "description": "Fold residual-add into RMSNorm.",
        "env_flag": "LFM2_FUSED_RESIDUAL",
        "source_file": "/sgl/python/sglang/srt/models/lfm2.py",
        "source_hints": ["+ residual", "RMSNorm("],
        "fusion_math": "y, residual = norm(x, residual)",
        "eager_reference_hint": "Import the framework RMSNorm; compare rmsnorm(x+residual).",
        "shapes": {"hidden_size": 2048, "T": 16},
        "rocm_native": True,
    }


def test_prompt_contains_recipe_fields():
    p = build_author_prompt(_recipe(), framework="sglang", ab_hint="run the A/B")
    assert "residual_add_rmsnorm" in p
    assert "LFM2_FUSED_RESIDUAL" in p  # env-gated by the recipe flag
    assert "y, residual = norm(x, residual)" in p  # fusion math
    assert "+ residual" in p and "RMSNorm(" in p  # source anchors
    assert "Import the framework RMSNorm" in p  # eager reference (no re-derive)
    assert "run the A/B" in p  # validation hint


def test_rocm_guard_present_when_native():
    p = build_author_prompt(_recipe(), framework="sglang", ab_hint="x")
    assert "ROCm" in p and "Triton" in p
    assert "Do NOT reuse a framework CUDA-only fused op" in p


def test_rocm_guard_absent_when_not_native():
    r = _recipe()
    r["rocm_native"] = False
    p = build_author_prompt(r, framework="sglang", ab_hint="x")
    assert "TARGET IS ROCm" not in p


def test_integration_candidate_benchmarks_existing_operator_before_authoring():
    r = _recipe()
    r["candidate_kind"] = "integration"
    r["existing_operator"] = "gemm_a16w16_gated"

    p = build_author_prompt(r, framework="sglang", ab_hint="x")

    assert "gemm_a16w16_gated" in p
    assert "benchmark and wire the existing operator first" in p
    assert "Do not author a replacement kernel unless" in p


def test_integration_candidate_demands_recorded_parity_evidence():
    """An integrated operator is third-party code whose numerics were never
    checked against this model's eager path. The prompt must require recorded
    parity evidence against the framework's own tolerance, otherwise a kernel
    can be wired in on a microbenchmark win alone."""
    r = _recipe()
    r["candidate_kind"] = "integration"
    r["existing_operator"] = "gemm_a16w16_gated"

    p = build_author_prompt(r, framework="sglang", ab_hint="x")

    assert "parity" in p.lower()
    assert "rtol" in p.lower()
    assert "record" in p.lower()
