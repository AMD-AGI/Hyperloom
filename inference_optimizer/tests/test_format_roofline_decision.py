"""Roofline-v2 C5: ``_format_roofline_decision`` prompt renderer tests.

Pins the three rendering modes the C4b executor + C2 recorder
populate ``last_roofline_analysis`` into, and the integration into
``to_prompt_summary`` that surfaces the rendered block to the main
Orchestration LLM.

The orchestration.md prompt fragment teaches the LLM how to consume
this rendered output — those two artefacts are co-designed and must
stay in lockstep: every signal documented in orchestration.md
(``(not yet run)`` / ``DEGRADED ...`` / structured block with
``primary_bottleneck`` etc.) must be produced by this renderer.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.shared_state import (
    _MAX_NEXT_ACTIONS_RENDERED,
    _MAX_PRUNES_RENDERED,
    SharedState,
)


def _healthy_analysis(**overrides) -> dict:
    """Canonical well-formed C4b-style executor output (post C2 cleaning)."""
    base = {
        "snapshot_id": 4,
        "analyzed_at_iso": "2026-05-19T10:30:00+00:00",
        "analyzed_at_gain_pct": 3.2,
        "based_on_analysis_md": "/sessions/abc/select_kernels/analysis.md",
        "primary_bottleneck": "comm",
        "bottleneck_distribution": {
            "comm": 0.45, "compute": 0.30, "memory": 0.15,
            "latency": 0.05, "idle": 0.05,
        },
        "suggested_prunes": [
            {"family": "kernel_opt",
             "reason": "compute saturated 91.2%, no reusable_native in Top Ops",
             "confidence": "high"},
            {"family": "deep_kernel_analysis",
             "reason": "comm > 40% — kernel-level analysis won't help",
             "confidence": "medium"},
        ],
        "suggested_next_actions": [
            {"kind": "params",
             "rationale": "try enable_two_batch_overlap / aiter_allreduce_fusion",
             "priority": "high"},
            {"kind": "comm_optimization",
             "rationale": "rccl Allreduce is top-1 by gpu_pct (32%)",
             "priority": "high"},
            {"kind": "backends",
             "rationale": "try moe_a2a_backend=deepep",
             "priority": "medium"},
        ],
        "reprofile_recommended": False,
        "reprofile_reason": "",
        "raw_llm_response": "{...}",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Mode 1: not-yet-run
# ---------------------------------------------------------------------------
def test_not_yet_run_renders_explicit_marker():
    """Default empty cache → ``(not yet run)`` so the LLM clearly
    sees that proposing ``roofline`` is the next move."""
    s = SharedState()
    assert s.last_roofline_analysis == {}
    assert s._format_roofline_decision() == "(not yet run)"


# ---------------------------------------------------------------------------
# Mode 2: degraded
# ---------------------------------------------------------------------------
def test_degraded_renders_one_liner_with_error():
    s = SharedState()
    s.last_roofline_analysis = {
        "snapshot_id": 3,
        "analyzed_at_iso": "2026-05-19T10:00:00+00:00",
        "analyzed_at_gain_pct": 0.0,
        "based_on_analysis_md": "/p/analysis.md",
        "primary_bottleneck": "unknown",
        "bottleneck_distribution": {},
        "suggested_prunes": [],
        "suggested_next_actions": [],
        "reprofile_recommended": False,
        "reprofile_reason": "",
        "raw_llm_response": "",
        "error": "sub_agent_timeout_after_60s",
    }
    rendered = s._format_roofline_decision()
    assert rendered.startswith("DEGRADED")
    assert "snapshot=3" in rendered
    assert "analyzed_at_gain=0.00%" in rendered
    assert "sub_agent_timeout_after_60s" in rendered
    assert "action_scores priors" in rendered  # operator guidance


def test_degraded_with_default_error_fallback_when_field_missing():
    """C2 recorder may strip ``error`` field on certain inputs; the
    renderer falls back to ``no_advice_available``."""
    s = SharedState()
    s.last_roofline_analysis = {
        "snapshot_id": 2,
        "analyzed_at_iso": "2026-05-19T09:00:00+00:00",
        "analyzed_at_gain_pct": 0.0,
        "based_on_analysis_md": "/p/analysis.md",
        "primary_bottleneck": "unknown",
        "bottleneck_distribution": {},
        "suggested_prunes": [],
        "suggested_next_actions": [],
        "reprofile_recommended": False,
        "reprofile_reason": "",
        "raw_llm_response": "",
    }
    rendered = s._format_roofline_decision()
    assert "DEGRADED" in rendered
    assert "no_advice_available" in rendered


# ---------------------------------------------------------------------------
# Mode 3: healthy
# ---------------------------------------------------------------------------
def test_healthy_renders_full_block():
    s = SharedState()
    s.record_roofline_analysis(_healthy_analysis())

    rendered = s._format_roofline_decision()

    # Header
    assert "snapshot=4" in rendered
    assert "analyzed_at_gain=3.20%" in rendered
    assert "ts=2026-05-19T10:30:00+00:00" in rendered

    # Primary + distribution (primary marked with *)
    assert "primary_bottleneck=comm" in rendered
    assert "comm=45%*" in rendered  # primary marker
    assert "compute=30%" in rendered
    assert "memory=15%" in rendered

    # Prune list with confidence + family + reason
    assert "suggested_prunes" in rendered
    assert "HIGH" in rendered and "kernel_opt" in rendered
    assert "compute saturated 91.2%" in rendered
    assert "MED" in rendered and "deep_kernel_analysis" in rendered

    # Next actions
    assert "suggested_next_actions" in rendered
    assert "HIGH" in rendered and "params" in rendered
    assert "enable_two_batch_overlap" in rendered
    assert "comm_optimization" in rendered
    assert "moe_a2a_backend" in rendered

    # Re-profile section + analysis path footer
    assert "reprofile_recommended=false" in rendered
    assert "full_analysis_md=/sessions/abc/select_kernels/analysis.md" in rendered


def test_healthy_with_reprofile_recommended():
    s = SharedState()
    s.record_roofline_analysis(_healthy_analysis(
        reprofile_recommended=True,
        reprofile_reason="gain +4% since snapshot; comm likely shifted",
    ))
    rendered = s._format_roofline_decision()
    assert "reprofile_recommended=true" in rendered
    assert "gain +4%" in rendered
    assert "emit PROPOSE_ACTION 'profile'" in rendered


def test_healthy_caps_suggestions_at_constants():
    """Top-3 cap keeps prompt compact (see _MAX_PRUNES_RENDERED
    /_MAX_NEXT_ACTIONS_RENDERED constants); excess entries fall off
    the prompt view but remain in the cached dict."""
    assert _MAX_PRUNES_RENDERED == 3
    assert _MAX_NEXT_ACTIONS_RENDERED == 3

    over_capped = _healthy_analysis(
        suggested_prunes=[
            {"family": f"fam_{i}", "reason": f"r_{i}",
             "confidence": "high"}
            for i in range(10)
        ],
        suggested_next_actions=[
            {"kind": f"kind_{i}", "rationale": f"r_{i}",
             "priority": "high"}
            for i in range(10)
        ],
    )
    s = SharedState()
    s.record_roofline_analysis(over_capped)
    rendered = s._format_roofline_decision()

    # First 3 from each list present
    assert "fam_0" in rendered and "fam_1" in rendered and "fam_2" in rendered
    assert "kind_0" in rendered and "kind_2" in rendered
    # 4th onwards omitted
    assert "fam_3" not in rendered
    assert "kind_5" not in rendered


def test_healthy_handles_empty_prune_list_gracefully():
    """Healthy ``primary_bottleneck`` + empty prune list still renders
    the block (analyzer may have nothing to prune)."""
    s = SharedState()
    s.record_roofline_analysis(_healthy_analysis(
        suggested_prunes=[],
    ))
    rendered = s._format_roofline_decision()
    assert "DEGRADED" not in rendered
    assert "primary_bottleneck=comm" in rendered
    assert "suggested_prunes" not in rendered  # section omitted
    assert "suggested_next_actions" in rendered  # still present


def test_healthy_handles_malformed_distribution_safely():
    """Non-numeric distribution values silently dropped (already
    enforced by C2 recorder, double-checked at render time)."""
    s = SharedState()
    # Bypass recorder to inject deliberate garbage
    s.last_roofline_analysis = _healthy_analysis(
        bottleneck_distribution={"comm": "bad", "compute": 0.5},
    )
    rendered = s._format_roofline_decision()
    assert "compute=50%" in rendered
    # "comm=bad" must NOT appear in the rendered output
    assert "comm=bad" not in rendered


# ---------------------------------------------------------------------------
# Integration into to_prompt_summary
# ---------------------------------------------------------------------------
def test_to_prompt_summary_includes_roofline_section():
    s = SharedState()
    s.record_roofline_analysis(_healthy_analysis())
    out = s.to_prompt_summary()
    assert "last_roofline_analysis=" in out
    assert "primary_bottleneck=comm" in out
    assert "suggested_prunes" in out


def test_to_prompt_summary_includes_marker_when_not_yet_run():
    s = SharedState()
    out = s.to_prompt_summary()
    assert "last_roofline_analysis=(not yet run)" in out


def test_to_prompt_summary_does_not_break_when_degraded():
    s = SharedState()
    s.last_roofline_analysis = {
        "snapshot_id": 1, "analyzed_at_iso": "now",
        "analyzed_at_gain_pct": 0.0, "based_on_analysis_md": "/p",
        "primary_bottleneck": "unknown", "bottleneck_distribution": {},
        "suggested_prunes": [], "suggested_next_actions": [],
        "reprofile_recommended": False, "reprofile_reason": "",
        "raw_llm_response": "", "error": "x",
    }
    out = s.to_prompt_summary()
    assert "DEGRADED" in out


# ---------------------------------------------------------------------------
# Distribution formatter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dist,primary,expected_first", [
    ({"comm": 0.6, "compute": 0.4}, "comm", "comm=60%*"),
    ({"compute": 0.7, "memory": 0.2, "idle": 0.1}, "compute", "compute=70%*"),
    ({}, "unknown", "unavailable"),
])
def test_distribution_formatter_sorts_and_marks_primary(dist, primary, expected_first):
    rendered = SharedState._format_bottleneck_distribution(dist, primary)
    assert rendered.startswith(expected_first)


def test_distribution_formatter_handles_non_dict():
    assert SharedState._format_bottleneck_distribution([], "comm") == "unavailable"
    assert SharedState._format_bottleneck_distribution("garbage", "comm") == "unavailable"
