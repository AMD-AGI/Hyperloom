"""E2E integration tests for warm-replay uncovered scenarios.

Covers KEEP/REVERT patch precedence, gbrain anti-pattern deserialization into
the blocklist, and framework write-back into prs_tested.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom.orchestrator.knowledge.cortex_t0 import (
    _extract_patches_from_prs_tested,
)
from hyperloom.orchestrator.knowledge.recipe_kb.gbrain_remote_client import _json_list


def _recipe_with_prs(prs_tested):
    return {
        "canonical_id": "inference:deepseek-r1:mi300x:sglang:llm:deepseekreasonermodel:0.5.11:fp8",
        "best_config": {"extra_server_args": "--tp 8"},
        "best_throughput": 5000.0,
        "prs_tested": prs_tested,
    }


def test_c3_keep_and_revert_same_patch_revert_wins():
    """C3: If same patch_file has both KEEP and REVERT, REVERT blocks it."""
    recipe = _recipe_with_prs([
        {
            "outcome": "KEEP",
            "patch_file": "vllm/attention/rocm_flash_attn.py",
            "patch_content": "diff --git a/vllm/attention ...",
            "measured_gain_pct": 12.5,
            "applicable_arch": ["DeepseekReasonerModel"],
            "repo": "ROCm/vllm",
        },
        {
            "outcome": "REVERT",
            "patch_file": "vllm/attention/rocm_flash_attn.py",
            "patch_content": "diff --git a/vllm/attention ...",
            "measured_gain_pct": -5.2,
            "applicable_arch": ["DeepseekReasonerModel"],
            "error_class": "perf_regression",
            "repo": "ROCm/vllm",
        },
    ])
    ctx = {}
    _extract_patches_from_prs_tested(ctx, recipe, ["DeepseekReasonerModel"])

    patches = (ctx.get("recommended_replay") or {}).get("patches") or []
    blocked = ctx.get("blocked_patches") or []

    assert len(blocked) == 1
    assert blocked[0]["patch_file"] == "vllm/attention/rocm_flash_attn.py"
    assert blocked[0]["error_class"] == "perf_regression"

    # KEEP is also extracted (the executor filters it at apply time via blocklist).
    assert len(patches) == 1

    blocked_files = {b["patch_file"] for b in blocked}
    assert patches[0]["patch_file"] in blocked_files


def test_c3_multiple_patches_partial_block():
    """C3 variant: only the REVERT'd patch is blocked, others pass."""
    recipe = _recipe_with_prs([
        {
            "outcome": "KEEP",
            "patch_file": "vllm/attention/rocm_flash_attn.py",
            "patch_content": "diff A",
            "measured_gain_pct": 12.5,
            "applicable_arch": ["DeepseekReasonerModel"],
        },
        {
            "outcome": "REVERT",
            "patch_file": "vllm/attention/rocm_flash_attn.py",
            "patch_content": "diff A",
            "measured_gain_pct": -5.2,
            "applicable_arch": ["DeepseekReasonerModel"],
            "error_class": "perf_regression",
        },
        {
            "outcome": "KEEP",
            "patch_file": "sglang/radix_cache.py",
            "patch_content": "diff B",
            "measured_gain_pct": 8.0,
            "applicable_arch": ["DeepseekReasonerModel"],
        },
    ])
    ctx = {}
    _extract_patches_from_prs_tested(ctx, recipe, ["DeepseekReasonerModel"])

    patches = ctx["recommended_replay"]["patches"]
    blocked = ctx["blocked_patches"]

    assert len(patches) == 2
    assert len(blocked) == 1
    assert blocked[0]["patch_file"] == "vllm/attention/rocm_flash_attn.py"

    blocked_files = {b["patch_file"] for b in blocked}
    assert "sglang/radix_cache.py" not in blocked_files


def test_c4_gbrain_prs_tested_roundtrip():
    """prs_tested stored as JSON string in gbrain page is decoded and used by
    cortex_t0 to produce blocked_patches."""
    prs_data = [
        {
            "outcome": "REVERT",
            "patch_file": "vllm/fp8_quant.py",
            "patch_content": "diff --git ...",
            "measured_gain_pct": -8.1,
            "applicable_arch": ["LlamaForCausalLM"],
            "error_class": "accuracy_regression",
            "repo": "ROCm/vllm",
        },
        {
            "outcome": "KEEP",
            "patch_file": "sglang/mem_pool.py",
            "patch_content": "diff --git ...",
            "measured_gain_pct": 15.3,
            "applicable_arch": ["LlamaForCausalLM"],
            "repo": "ROCm/sglang",
        },
    ]

    stored_json = json.dumps(prs_data)

    decoded = _json_list(stored_json)
    assert len(decoded) == 2
    assert decoded[0]["outcome"] == "REVERT"
    assert decoded[1]["outcome"] == "KEEP"

    recipe = {
        "canonical_id": "inference:llama3.3-70b:mi300x:sglang:llm:llamaforcausallm:0.5.11:fp8",
        "best_config": {"extra_server_args": "--tp 8"},
        "best_throughput": 4200.0,
        "prs_tested": decoded,
    }

    ctx = {}
    _extract_patches_from_prs_tested(ctx, recipe, ["LlamaForCausalLM"])

    patches = ctx["recommended_replay"]["patches"]
    blocked = ctx["blocked_patches"]

    assert len(patches) == 1
    assert patches[0]["patch_file"] == "sglang/mem_pool.py"
    assert patches[0]["measured_gain_pct"] == 15.3

    assert len(blocked) == 1
    assert blocked[0]["patch_file"] == "vllm/fp8_quant.py"
    assert blocked[0]["error_class"] == "accuracy_regression"


def test_c4_gbrain_empty_prs_tested_is_safe():
    """Gbrain returns empty/null prs_tested without crashing."""
    for value in (None, "", "[]", [], "null"):
        decoded = _json_list(value)
        assert decoded == [] or decoded is None or decoded == []


@dataclass
class _MockSharedState:
    model_architectures: list = field(default_factory=lambda: ["LlamaForCausalLM"])
    precision: str = "fp8"
    framework: str = "sglang"


@dataclass
class _MockCoordinator:
    """Minimal coordinator surface for testing framework write-back."""
    shared_state: _MockSharedState = field(default_factory=_MockSharedState)
    _local_recipe_row: dict = field(default_factory=dict)
    _amended: list = field(default_factory=list)
    _source_sid: str = "test-session-001"

    def _source_session_id(self):
        return self._source_sid

    def _read_local_recipe_row(self):
        return self._local_recipe_row

    def _kb_amend_recipe(self, recipe_overrides=None, **kwargs):
        self._amended.append(recipe_overrides or kwargs)


def _build_framework_entry(
    coord: _MockCoordinator,
    *,
    status: str,
    patch_path: str,
    delta_pct: float,
    pr_url: str = "https://github.com/ROCm/vllm/pull/42",
    repo: str = "ROCm/vllm",
    error_class: str = "",
) -> dict:
    """Simulate the framework result -> prs_tested entry construction."""
    from datetime import datetime, timezone

    outcome = "KEEP" if status == "kept" else "REVERT"
    ss = coord.shared_state
    entry = {
        "repo": repo,
        "number": 42,
        "outcome": outcome,
        "patch_file": patch_path,
        "measured_gain_pct": float(delta_pct),
        "applicable_arch": list(ss.model_architectures or []),
        "applicable_precision": str(ss.precision or ""),
        "applicable_platform": "rocm",
        "error_class": error_class if outcome == "REVERT" else "",
        "notes": f"{outcome}: test patch ({delta_pct:+.1f}%)",
        "source_session_id": coord._source_session_id(),
        "tested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    existing_prs = list(coord._local_recipe_row.get("prs_tested") or [])
    existing_prs.append(entry)
    coord._kb_amend_recipe(recipe_overrides={"prs_tested": existing_prs})
    return entry


def test_d1_framework_keep_writes_prs_tested():
    """D1: Framework KEEP result → prs_tested[KEEP] with positive gain."""
    coord = _MockCoordinator(
        _local_recipe_row={"prs_tested": []},
    )
    _build_framework_entry(
        coord,
        status="kept",
        patch_path="vllm/attention/rocm_flash_attn.py",
        delta_pct=18.7,
    )

    assert len(coord._amended) == 1
    written = coord._amended[0]["prs_tested"]
    assert len(written) == 1
    assert written[0]["outcome"] == "KEEP"
    assert written[0]["patch_file"] == "vllm/attention/rocm_flash_attn.py"
    assert written[0]["measured_gain_pct"] == 18.7
    assert written[0]["error_class"] == ""


def test_d2_framework_revert_writes_prs_tested():
    """D2: Framework REVERT result → prs_tested[REVERT] with negative gain."""
    coord = _MockCoordinator(
        _local_recipe_row={"prs_tested": []},
    )
    _build_framework_entry(
        coord,
        status="reverted",
        patch_path="sglang/radix_cache.py",
        delta_pct=-4.3,
        error_class="perf_regression",
    )

    assert len(coord._amended) == 1
    written = coord._amended[0]["prs_tested"]
    assert len(written) == 1
    assert written[0]["outcome"] == "REVERT"
    assert written[0]["patch_file"] == "sglang/radix_cache.py"
    assert written[0]["measured_gain_pct"] == -4.3
    assert written[0]["error_class"] == "perf_regression"


def test_d3_writeback_includes_arch_and_error_class():
    """D3: Write-back entry contains applicable_arch + error_class + platform."""
    coord = _MockCoordinator(
        shared_state=_MockSharedState(
            model_architectures=["Qwen3MoeForCausalLM", "MixtralForCausalLM"],
            precision="bf16",
        ),
        _local_recipe_row={"prs_tested": []},
    )
    _build_framework_entry(
        coord,
        status="reverted",
        patch_path="vllm/model_executor/moe.py",
        delta_pct=-12.0,
        error_class="server_crash",
    )

    written = coord._amended[0]["prs_tested"][0]
    assert written["applicable_arch"] == ["Qwen3MoeForCausalLM", "MixtralForCausalLM"]
    assert written["applicable_precision"] == "bf16"
    assert written["applicable_platform"] == "rocm"
    assert written["error_class"] == "server_crash"
    assert written["source_session_id"] == "test-session-001"
    assert "tested_at" in written


def test_d3_append_to_existing_prs_tested():
    """D3: Write-back appends to existing prs_tested list, not replace."""
    existing = [{
        "outcome": "KEEP",
        "patch_file": "old_patch.py",
        "measured_gain_pct": 5.0,
        "applicable_arch": ["LlamaForCausalLM"],
    }]
    coord = _MockCoordinator(
        _local_recipe_row={"prs_tested": existing},
    )
    _build_framework_entry(
        coord,
        status="kept",
        patch_path="new_patch.py",
        delta_pct=10.0,
    )

    written = coord._amended[0]["prs_tested"]
    assert len(written) == 2
    assert written[0]["patch_file"] == "old_patch.py"
    assert written[1]["patch_file"] == "new_patch.py"


def test_full_chain_gbrain_revert_blocks_at_executor():
    """Integration: REVERT from gbrain blocks patch at executor apply phase."""
    import subprocess

    gbrain_prs = json.dumps([{
        "outcome": "REVERT",
        "patch_file": "vllm/fp8.py",
        "patch_content": "diff --git a/vllm/fp8.py ...",
        "measured_gain_pct": -7.0,
        "applicable_arch": ["LlamaForCausalLM"],
        "error_class": "accuracy_regression",
    }, {
        "outcome": "KEEP",
        "patch_file": "vllm/fp8.py",
        "patch_content": "diff --git a/vllm/fp8.py ...",
        "measured_gain_pct": 20.0,
        "applicable_arch": ["LlamaForCausalLM"],
    }])

    decoded_prs = _json_list(gbrain_prs)

    recipe = {
        "canonical_id": "test:llama:mi300x:sglang:llm:llamaforcausallm:0.5.11:fp8",
        "best_config": {},
        "prs_tested": decoded_prs,
    }
    ctx = {}
    _extract_patches_from_prs_tested(ctx, recipe, ["LlamaForCausalLM"])

    patches = ctx["recommended_replay"]["patches"]
    blocked = ctx["blocked_patches"]

    params = {
        "patches": patches,
        "blocked_patches": blocked,
    }

    from hyperloom.orchestrator.actions.executors.baseline import (
        _apply_warm_patches,
    )

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td) / "output"
        output_dir.mkdir()
        # target_repo empty -> returns [] (no actual git apply needed)
        result = _apply_warm_patches(params, "", output_dir)
        assert result == []

        fake_repo = Path(td) / "repo"
        fake_repo.mkdir()
        subprocess.run(["git", "init"], cwd=str(fake_repo), capture_output=True)
        result = _apply_warm_patches(params, str(fake_repo), output_dir)
        # patch skipped due to blocklist (patch_file matches)
        assert result == []
