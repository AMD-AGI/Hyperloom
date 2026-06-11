# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression: KEEP'd kernel optimizations must be persisted into the recipe row.

Bug context (session 20260529T132829Z): k006 reached micro 1.32x but flat
E2E and was dropped from recipe.json; the ``kernel_optimizations`` array fixes it.
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
    """Recipe.kernel_optimizations round-trips and lands at the top level on disk."""
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
    assert r.to_dict()["kernel_optimizations"] == []
