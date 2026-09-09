# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for specialist prompt builder focus templates + section branches."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    build_specialist_prompts,
    _render_measured_impact,
    _format_version_note,
)
from hyperloom.orchestrator.specialists.domains import SPECIALIST_DOMAINS, get_domain


# Derived, not copied: a hand-maintained list goes stale silently when a domain
# is added, and fails en masse when one is removed.
_DOMAIN_KEYS = tuple(sorted(d.key for d in SPECIALIST_DOMAINS))


@pytest.mark.parametrize("domain_key", _DOMAIN_KEYS)
@pytest.mark.parametrize("framework", ["vllm", "atom"])
def test_build_for_each_domain_and_framework(domain_key, framework):
    domain = get_domain(domain_key)
    assert domain is not None
    sys_p, user_p = build_specialist_prompts(
        SpecialistPromptInputs(
            task_id="t1",
            domain=domain,
            framework=framework,
            gpu_type="MI300X",
            tp=2,
            hbm_gb=192.0,
            peak_tflops=1300.0,
            precision="fp8",
            conc=64,
            isl=1024,
            osl=512,
            max_model_len=8192,
            arch_notes="MoE 8x7B",
            target_gap_notes="competitor at 2x",
            gap_canonical_id="gap.x",
            gap_layer="kernel_agent",
            gap_symptom="slow",
            gap_evidence={"k": 1},
        )
    )
    assert sys_p
    assert "## 2. HARDWARE CONTEXT" in user_p
    assert "concurrency: 64" in user_p


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
        gap_layer="kernel_agent",
        gap_symptom="slow decode",
        gap_evidence={"step": 1},
        kb_subgraph={"nodes": [1, 2]},
        roofline_evidence={
            "roofline_snapshot_id": 7,
            "executive_summary": {
                "compute_pct": 70.0,
                "idle_pct": 10.0,
                "comm_pct": 3.0,
                "top_bottleneck": "MoE",
            },
            "kernel_roofline_top15": [
                {
                    "kernel_id": "k1",
                    "name": "gemm",
                    "gpu_pct": 30.0,
                    "bound_type": "compute",
                    "arithmetic_intensity": 12.3,
                    "efficiency_percent": 55.0,
                    "compute_utilization_pct": 60.0,
                    "bandwidth_utilization_pct": 40.0,
                    "recommended_actions": ["fuse"],
                },
                "skip-non-dict",
            ],
            "hot_kernels_top15": [
                {"kernel_id": "h1", "name": "attn", "gpu_pct": 20.0, "bottleneck": "memory", "source_file": "attn.py"},
            ],
            "analysis_md_path": "/tmp/analysis.md",
        },
        warm_start_recipe={"best_config": {"x": "1"}},
        warm_start_lessons=[
            {
                "confidence": 0.9,
                "attrs": {
                    "statement": "enable cudagraph",
                    "measured_impact": {
                        "gain_pct": 5.0,
                        "throughput_after": 100.0,
                        "stack_depth_at_apply": 2,
                        "measured_at": "2026-01-02T00:00:00",
                    },
                    "validated_count": 3,
                    "source_session_ids": ["s1", "s2"],
                    "framework_version": "0.6.1",
                },
            },
            {"attrs": {}},  # filtered
        ],
        warm_start_pitfalls=[
            {
                "confidence": 0.8,
                "attrs": {
                    "description": "do not enforce eager",
                    "severity": "high",
                    "validated_count": 2,
                    "source_session_id": "s9",
                },
            },
            {"attrs": {}},  # filtered
        ],
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
    assert "`k1`" in user_p
    assert "`h1`" in user_p
    assert "find-recipe result" in user_p
    assert "enable cudagraph" in user_p
    assert "+5.00%" in user_p
    assert "[from vllm@0.6.1, you're on 0.6.2]" in user_p
    assert "do not enforce eager" in user_p
    assert "mcp__pr_monitor__" in user_p
    assert "/src/vllm" in user_p
    assert "NOTES FROM ORCHESTRATION" in user_p


def test_cold_start_directive():
    inp = SpecialistPromptInputs(
        task_id="t",
        domain=get_domain("serving_specialist"),
    )
    _, user_p = build_specialist_prompts(inp)
    assert "COLD-START MODE" in user_p


def test_cold_start_does_not_ask_for_a_field_the_safety_gate_forbids():
    """It used to direct the fallback proposals to carry ``confidence: low`` --
    a field in FORBIDDEN_PROPOSAL_FIELDS -- so a compliant specialist tripped
    the guard on exactly the round where it was the only source of ideas."""
    inp = SpecialistPromptInputs(
        task_id="t",
        domain=get_domain("serving_specialist"),
    )
    _, user_p = build_specialist_prompts(inp)

    cold_start = user_p[user_p.index("COLD-START MODE") :]
    assert "confidence: low" not in cold_start
    assert "provenance: domain_focus_default" in cold_start


def test_research_hints_fallback():
    inp = SpecialistPromptInputs(
        task_id="t",
        domain=get_domain("serving_specialist"),
        research_hints="prior: enable aiter",
    )
    _, user_p = build_specialist_prompts(inp)
    assert "enable aiter" in user_p


def test_pr_monitor_unavailable():
    inp = SpecialistPromptInputs(
        task_id="t",
        domain=get_domain("serving_specialist"),
        pr_monitor_available=False,
    )
    _, user_p = build_specialist_prompts(inp)
    assert "unavailable" in user_p


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


def test_gpu_autonomy_block_and_auto_retry():
    inp = _rich_inputs(bench=True, mode="patch", auto_retry_reason="timeout on prior attempt")
    sys_p, _ = build_specialist_prompts(inp)
    assert "On-GPU autonomy" in sys_p
    assert "specialists.rebench" in sys_p
    assert "Auto-retry notice" in sys_p
    assert "port 8888" not in sys_p


def test_no_gpu_iron_rule_retains_serving_port_boundary():
    inp = _rich_inputs(allocated_gpu_ids=())
    sys_p, _ = build_specialist_prompts(inp)
    assert "port 8888" in sys_p


def test_render_measured_impact_variants():
    assert _render_measured_impact("just a string") == "just a string"
    assert _render_measured_impact(None) == ""
    assert _render_measured_impact(42) == "42"
    out = _render_measured_impact({"gain_pct": 1.0})
    assert "+1.00%" in out


def test_format_version_note_empty_when_same_or_unknown():
    inp = SpecialistPromptInputs(
        task_id="t",
        domain=get_domain("serving_specialist"),
        framework="vllm",
        framework_version="1.0",
    )
    assert _format_version_note(inp, {"framework_version": "1.0"}) == ""
    assert _format_version_note(inp, {}) == ""
    note = _format_version_note(inp, {"framework_version": "0.9"})
    assert "0.9" in note


class TestTheAgentXWorkloadBranch:
    """A specialist on an agentic replay must not be handed the ISL/OSL knobs.

    ``state.isl``/``osl`` are inert placeholders there, so Section 2 renders the
    corpus distribution instead and the kernel hints drop their short-OSL advice.
    """

    _SHAPE = {
        "corpus_loader": "semianalysis_cc_traces_weka_062126",
        "corpus_entries": 393,
        "duration_s": 3600.0,
        "isl": {"p50": 94821, "p90": 163328, "p99": 506158},
        "osl": {"p50": 333, "p90": 1874, "p99": 6386},
        "prefix_cache_hit": 0.975,
    }

    def _prompt(self, domain_key: str, **over) -> str:
        domain = get_domain(domain_key)
        assert domain is not None
        kwargs = {
            "task_id": "t-agentx",
            "domain": domain,
            "framework": "vllm",
            "gpu_type": "MI355X",
            "tp": 8,
            "precision": "fp8",
            "conc": 8,
            "isl": 1024,
            "osl": 1024,
            **over,
        }
        # Both halves: the workload section lands in the user prompt, the
        # domain-focus hints in the system prompt.
        return "\n".join(build_specialist_prompts(SpecialistPromptInputs(**kwargs)))

    def test_agentx_replaces_the_isl_osl_rows_with_the_corpus(self):
        body = self._prompt("serving_specialist", benchmark_mode="agentx", agentx_corpus_shape=self._SHAPE)
        assert "ISL (input seq len)" not in body
        assert "OSL (output seq len)" not in body
        assert "AgentX agentic trace replay" in body
        assert "p50 94,821" in body
        assert "prefix cache hit ~97.5%" in body

    def test_agentx_states_the_graded_axis_in_the_workload_section(self):
        body = self._prompt("serving_specialist", benchmark_mode="agentx", agentx_corpus_shape=self._SHAPE)
        assert "E2E normalised interactivity P90" in body
        assert "RECORDED" in body

    def test_a_synthetic_session_still_gets_the_isl_osl_rows(self):
        body = self._prompt("serving_specialist")
        assert "- ISL (input seq len): 1024" in body
        assert "- OSL (output seq len): 1024" in body
        assert "AgentX agentic trace replay" not in body

    def test_the_kernel_hints_retract_their_short_osl_advice_under_agentx(self):
        """The canonical block advertises short-OSL tuning; that is wrong here."""
        synthetic = self._prompt("kernel_switch_specialist")
        agentic = self._prompt("kernel_switch_specialist", benchmark_mode="agentx", agentx_corpus_shape=self._SHAPE)
        assert "short OSL" in synthetic
        assert "does not apply to this workload" not in synthetic
        assert "does not apply to this workload" in agentic
        assert "long-KV decode GEMMs" in agentic

    def test_the_mode_alone_arms_the_branch_without_a_shape(self):
        """A session whose first measurement has not landed yet still says so."""
        body = self._prompt("serving_specialist", benchmark_mode="agentx")
        assert "ISL (input seq len)" not in body
        assert "E2E normalised interactivity P90" in body
