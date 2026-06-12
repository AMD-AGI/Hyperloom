# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for specialist prompt builder focus templates + section branches."""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.system_prompts import (
    specialist_prompt_builder as spb,
)
from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    build_specialist_prompts,
    build_specialist_prompts_for_domain,
    _render_measured_impact,
    _format_version_note,
)
from inference_optimizer.orchestrator.specialist_domains import get_domain


_DOMAIN_KEYS = (
    "serving_specialist",
    "kernel_switch_specialist",
    "comm_specialist",
    "compiler_specialist",
    "system_specialist",
    "pr_intel_specialist",
    "research_scout_specialist",
)


@pytest.mark.parametrize("domain_key", _DOMAIN_KEYS)
@pytest.mark.parametrize("framework", ["vllm", "atom"])
def test_build_for_each_domain_and_framework(domain_key, framework):
    sys_p, user_p = build_specialist_prompts_for_domain(
        task_id="t1", domain_key=domain_key, framework=framework,
        gpu_type="MI300X", tp=2, hbm_gb=192.0, peak_tflops=1300.0,
        precision="fp8", conc=64, isl=1024, osl=512, max_model_len=8192,
        arch_notes="MoE 8x7B", target_gap_notes="competitor at 2x",
        gap_canonical_id="gap.x", gap_layer="kernel", gap_symptom="slow",
        gap_evidence={"k": 1},
    )
    assert sys_p
    assert "## 2. HARDWARE CONTEXT" in user_p
    assert "concurrency: 64" in user_p


def test_unknown_domain_raises():
    with pytest.raises(ValueError):
        build_specialist_prompts_for_domain(task_id="t", domain_key="nope")


def _rich_inputs(**overrides):
    base = dict(
        task_id="t1",
        domain=get_domain("serving_specialist"),
        framework="vllm",
        framework_version="0.6.2",
        gpu_type="MI300X",
        allocated_gpu_ids=(0, 1),
        tp=2,
        hbm_gb=192.0,
        peak_tflops=1300.0,
        precision="fp8",
        conc=64,
        isl=1024,
        osl=512,
        max_model_len=8192,
        arch_notes="MoE",
        target_gap_notes="target gap note",
        gap_canonical_id="gap.x",
        gap_layer="kernel",
        gap_symptom="slow decode",
        gap_evidence={"step": 1},
        kb_subgraph={"nodes": [1, 2]},
        roofline_evidence={
            "roofline_snapshot_id": 7,
            "executive_summary": {
                "compute_pct": 70.0, "idle_pct": 10.0,
                "comm_pct": 3.0, "top_bottleneck": "MoE",
            },
            "kernel_roofline_top15": [
                {"kernel_id": "k1", "name": "gemm", "gpu_pct": 30.0,
                 "bound_type": "compute", "arithmetic_intensity": 12.3,
                 "efficiency_percent": 55.0, "compute_utilization_pct": 60.0,
                 "bandwidth_utilization_pct": 40.0,
                 "recommended_actions": ["fuse"]},
                "skip-non-dict",
            ],
            "hot_kernels_top15": [
                {"kernel_id": "h1", "name": "attn", "gpu_pct": 20.0,
                 "bottleneck": "memory", "source_file": "attn.py"},
            ],
            "analysis_md_path": "/tmp/analysis.md",
        },
        warm_start_recipe={"best_config": {"x": "1"}},
        warm_start_lessons=[
            {"confidence": 0.9, "attrs": {
                "statement": "enable cudagraph",
                "measured_impact": {"gain_pct": 5.0, "throughput_after": 100.0,
                                    "stack_depth_at_apply": 2,
                                    "measured_at": "2026-01-02T00:00:00"},
                "validated_count": 3,
                "source_session_ids": ["s1", "s2"],
                "framework_version": "0.6.1",
            }},
            {"attrs": {}},  # filtered (no statement)
        ],
        warm_start_pitfalls=[
            {"confidence": 0.8, "attrs": {
                "description": "do not enforce eager",
                "severity": "high", "validated_count": 2,
                "source_session_id": "s9",
            }},
            {"attrs": {}},  # filtered
        ],
        pr_feed=[{"title": "Fix MoE", "url": "http://pr/1", "labels": ["perf"]}],
        framework_source_roots=("/src/vllm",),
        source_hint_directories=("/src/vllm/v1",),
        notes="prev round residuals",
    )
    base.update(overrides)
    return SpecialistPromptInputs(**base)


def test_build_rich_user_prompt_sections():
    sys_p, user_p = build_specialist_prompts(_rich_inputs())
    assert "TraceLens snapshot #7" in user_p
    assert "Executive Summary" in user_p
    assert "`k1`" in user_p          # kernel roofline table
    assert "`h1`" in user_p          # hot kernels table
    assert "find-recipe result" in user_p
    assert "enable cudagraph" in user_p
    assert "+5.00%" in user_p        # measured_impact dict render
    assert "[from vllm@0.6.1, you're on 0.6.2]" in user_p  # version note
    assert "do not enforce eager" in user_p
    assert "Fix MoE" in user_p
    assert "/src/vllm" in user_p
    assert "NOTES FROM ORCHESTRATION" in user_p


def test_cold_start_directive():
    inp = SpecialistPromptInputs(
        task_id="t", domain=get_domain("serving_specialist"),
    )
    _, user_p = build_specialist_prompts(inp)
    assert "COLD-START MODE" in user_p


def test_research_hints_fallback():
    inp = SpecialistPromptInputs(
        task_id="t", domain=get_domain("serving_specialist"),
        research_hints="prior: enable aiter",
    )
    _, user_p = build_specialist_prompts(inp)
    assert "enable aiter" in user_p


def test_pr_monitor_unavailable():
    inp = SpecialistPromptInputs(
        task_id="t", domain=get_domain("serving_specialist"),
        pr_monitor_available=False,
    )
    _, user_p = build_specialist_prompts(inp)
    assert "pr_monitor unavailable" in user_p


def test_scope_domains_with_extra_tags():
    inp = _rich_inputs(
        scope="domains",
        extra_focus_tags=("comm_specialist", "compiler_specialist"),
    )
    sys_p, _ = build_specialist_prompts(inp)
    assert "Domain focus — comm_specialist" in sys_p


def test_scope_freeform():
    inp = _rich_inputs(scope="freeform", task_description="optimize the kv cache")
    sys_p, _ = build_specialist_prompts(inp)
    assert "Free-form mandate" in sys_p
    assert "optimize the kv cache" in sys_p


def test_bench_block_and_auto_retry():
    inp = _rich_inputs(bench=True, mode="patch",
                       auto_retry_reason="timeout on prior attempt")
    sys_p, _ = build_specialist_prompts(inp)
    assert "In-loop micro-bench" in sys_p
    assert "Auto-retry notice" in sys_p


# ---- small pure helpers ----
def test_render_measured_impact_variants():
    assert _render_measured_impact("just a string") == "just a string"
    assert _render_measured_impact(None) == ""
    assert _render_measured_impact(42) == "42"
    out = _render_measured_impact({"gain_pct": 1.0})
    assert "+1.00%" in out


def test_format_version_note_empty_when_same_or_unknown():
    inp = SpecialistPromptInputs(
        task_id="t", domain=get_domain("serving_specialist"),
        framework="vllm", framework_version="1.0",
    )
    assert _format_version_note(inp, {"framework_version": "1.0"}) == ""
    assert _format_version_note(inp, {}) == ""
    note = _format_version_note(inp, {"framework_version": "0.9"})
    assert "0.9" in note
