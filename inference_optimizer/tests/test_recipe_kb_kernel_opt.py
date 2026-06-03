"""Regression tests: KEEP'd kernel optimizations (incl. E2E-verified-but-
no-gain) must be persisted into the recipe row.

Bug context (overnight 12h Qwen3-32B session 20260529T132829Z):
``k006`` rmsnorm_quant kernel optimization reached micro_speedup=1.32x,
compiled + passed correctness, was KEEP'd, AND went through E2E integrate
verification (new_tput=2477.82, gain_pct=-0.094%). Yet recipe.json had
``what_worked=[]`` and no trace of k006 at all — the entire kernel-opt +
E2E-verification outcome was dropped because ``what_worked`` is built
ONLY from ``optimization_stack`` (which a micro-only / no-E2E-gain kernel
never enters).

These tests pin the fix: a ``kernel_optimizations`` array on the recipe
that captures BOTH the micro result and the E2E verification outcome, so
a future warm-start can see "k006 tried: micro 1.32x but E2E flat" and
skip re-doing it.
"""

from __future__ import annotations

from inference_optimizer.recipe_kb.schema import KernelOptimization, Recipe


def test_kernel_optimization_roundtrip() -> None:
    """KernelOptimization survives to_dict -> from_dict byte-for-byte."""
    ko = KernelOptimization(
        kernel_id="k006",
        source_file="/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu",
        artifact_path="/x/optimized_versions/v1_n128_launch_tuning.cu",
        micro_speedup=1.3202,
        decision="KEEP",
        e2e_gain_pct=-0.094,
        e2e_tput=2477.82,
        integrated=True,
        ts="2026-05-29T19:17:27+00:00",
    )
    d = ko.to_dict()
    assert d["kernel_id"] == "k006"
    assert d["micro_speedup"] == 1.3202
    assert d["e2e_gain_pct"] == -0.094
    assert d["e2e_tput"] == 2477.82
    assert d["integrated"] is True
    back = KernelOptimization.from_dict(d)
    assert back == ko


def test_recipe_carries_kernel_optimizations_roundtrip() -> None:
    """Recipe.kernel_optimizations is a first-class field that round-trips
    through to_dict/from_dict and lands at the top level on disk."""
    r = Recipe(
        canonical_id="inference:qwen-qwen3-32b:mi355x:sglang:0.5.11:bf16",
        kernel_optimizations=[
            KernelOptimization(
                kernel_id="k006",
                source_file="rmsnorm_quant_kernels.cu",
                artifact_path="v1_n128_launch_tuning.cu",
                micro_speedup=1.3202,
                decision="KEEP",
                e2e_gain_pct=-0.094,
                e2e_tput=2477.82,
                integrated=True,
            ),
        ],
    )
    out = r.to_dict()
    assert "kernel_optimizations" in out, out.keys()
    assert out["kernel_optimizations"][0]["kernel_id"] == "k006"
    assert out["kernel_optimizations"][0]["micro_speedup"] == 1.3202
    # Re-parse: the field must NOT get bucketed into extras.
    back = Recipe.from_dict(out)
    assert len(back.kernel_optimizations) == 1
    assert back.kernel_optimizations[0].kernel_id == "k006"
    assert back.kernel_optimizations[0].e2e_gain_pct == -0.094
    assert "kernel_optimizations" not in back.extras


def test_recipe_from_dict_tolerates_missing_field() -> None:
    """Legacy recipe.json (no kernel_optimizations key) parses to []."""
    legacy = {
        "canonical_id": "inference:m:h:f:1.0:bf16",
        "version": 3,
        "what_worked": [],
    }
    r = Recipe.from_dict(legacy)
    assert r.kernel_optimizations == []
    # And re-serializing always emits the key (empty list).
    assert r.to_dict()["kernel_optimizations"] == []
