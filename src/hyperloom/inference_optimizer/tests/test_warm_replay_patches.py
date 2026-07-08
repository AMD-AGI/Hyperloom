"""Tests for warm-replay code patch extraction + blocklist."""
from __future__ import annotations

from hyperloom.orchestrator.knowledge.cortex_t0 import (
    _build_warm_start_context,
    _extract_patches_from_prs_tested,
)


def _recipe_with_prs(prs_tested):
    return {
        "canonical_id": "inference:test:mi300x:sglang:llama:llamaforcausallm:0.5.11:fp8",
        "best_config": {},
        "prs_tested": prs_tested,
    }


def test_extract_keep_patch_with_arch_match():
    recipe = _recipe_with_prs([{
        "outcome": "KEEP",
        "patch_file": "vllm/fp8.py",
        "patch_content": "diff --git ...",
        "measured_gain_pct": 24.9,
        "applicable_arch": ["LlamaForCausalLM"],
    }])
    ctx = {"recommended_replay": {}}
    _extract_patches_from_prs_tested(ctx, recipe, ["LlamaForCausalLM"])
    patches = ctx["recommended_replay"]["patches"]
    assert len(patches) == 1
    assert patches[0]["measured_gain_pct"] == 24.9
    assert patches[0]["patch_file"] == "vllm/fp8.py"


def test_extract_keep_patch_arch_mismatch_skipped():
    recipe = _recipe_with_prs([{
        "outcome": "KEEP",
        "patch_file": "vllm/fp8.py",
        "patch_content": "diff ...",
        "measured_gain_pct": 24.9,
        "applicable_arch": ["Qwen3MoeForCausalLM"],
    }])
    ctx = {"recommended_replay": {}}
    _extract_patches_from_prs_tested(ctx, recipe, ["LlamaForCausalLM"])
    assert "patches" not in ctx.get("recommended_replay", {})


def test_extract_revert_creates_blocked():
    recipe = _recipe_with_prs([{
        "outcome": "REVERT",
        "patch_file": "vllm/fp8.py",
        "patch_content": "diff ...",
        "measured_gain_pct": -3.2,
        "applicable_arch": ["Qwen3MoeForCausalLM"],
        "error_class": "accuracy_regression",
    }])
    ctx = {}
    _extract_patches_from_prs_tested(ctx, recipe, ["Qwen3MoeForCausalLM"])
    assert len(ctx["blocked_patches"]) == 1
    assert ctx["blocked_patches"][0]["error_class"] == "accuracy_regression"


def test_revert_without_applicable_arch_not_blocked():
    """REVERT with no applicable_arch should NOT block (too broad)."""
    recipe = _recipe_with_prs([{
        "outcome": "REVERT",
        "patch_file": "vllm/fp8.py",
        "patch_content": "diff ...",
        "measured_gain_pct": -5.0,
        "applicable_arch": [],
    }])
    ctx = {}
    _extract_patches_from_prs_tested(ctx, recipe, ["LlamaForCausalLM"])
    assert "blocked_patches" not in ctx


def test_patches_sorted_by_gain_desc():
    recipe = _recipe_with_prs([
        {"outcome": "KEEP", "patch_file": "a.py", "patch_content": "d1", "measured_gain_pct": 5.0, "applicable_arch": []},
        {"outcome": "KEEP", "patch_file": "b.py", "patch_content": "d2", "measured_gain_pct": 25.0, "applicable_arch": []},
        {"outcome": "KEEP", "patch_file": "c.py", "patch_content": "d3", "measured_gain_pct": 10.0, "applicable_arch": []},
    ])
    ctx = {}
    _extract_patches_from_prs_tested(ctx, recipe, [])
    patches = ctx["recommended_replay"]["patches"]
    gains = [p["measured_gain_pct"] for p in patches]
    assert gains == [25.0, 10.0, 5.0]


def test_large_patch_content_excluded():
    """Patches > 50KB should have empty patch_content (use patch_ref instead)."""
    recipe = _recipe_with_prs([{
        "outcome": "KEEP",
        "patch_file": "big.py",
        "patch_content": "x" * 60000,
        "measured_gain_pct": 10.0,
        "applicable_arch": [],
    }])
    ctx = {}
    _extract_patches_from_prs_tested(ctx, recipe, [])
    patches = ctx["recommended_replay"]["patches"]
    assert patches[0]["patch_content"] == ""


def test_empty_prs_tested_no_effect():
    recipe = _recipe_with_prs([])
    ctx = {"recommended_replay": {"extra_server_args": "--foo"}}
    _extract_patches_from_prs_tested(ctx, recipe, ["LlamaForCausalLM"])
    assert "patches" not in ctx["recommended_replay"]
    assert "blocked_patches" not in ctx


def test_build_warm_start_context_includes_patches():
    """Integration: _build_warm_start_context populates patches from prs_tested."""
    recipe = {
        "canonical_id": "inference:test:mi300x:sglang:llama:llamaforcausallm:0.5.11:fp8",
        "best_config": {"extra_server_args": "--disable-radix-cache"},
        "best_throughput": 5000.0,
        "prs_tested": [{
            "outcome": "KEEP",
            "patch_file": "vllm/fp8.py",
            "patch_content": "diff --git ...",
            "measured_gain_pct": 24.9,
            "applicable_arch": ["LlamaForCausalLM"],
        }],
        "what_worked": [],
        "what_failed": [],
        "lessons": [],
        "pitfalls": [],
    }
    ctx = _build_warm_start_context(
        status="hit",
        tier="exact",
        confidence=1.0,
        canonical_id=recipe["canonical_id"],
        source="local",
        recipe=recipe,
        model_architectures=["LlamaForCausalLM"],
    )
    assert ctx["recommended_replay"]["extra_server_args"] == "--disable-radix-cache"
    assert len(ctx["recommended_replay"]["patches"]) == 1
    assert ctx["recommended_replay"]["patches"][0]["measured_gain_pct"] == 24.9
