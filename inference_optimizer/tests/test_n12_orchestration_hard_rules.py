"""Roofline-v2 N12: orchestration.md hard rules for analysis.md → action mapping.

GPU-empirical context (DeepSeek-R1 session 16:06-02:00 of 2026-05-19):
N5 injected analysis.md, N10 persisted snapshots, N9/N11 cleaned up
profile + base64 noise. But the LLM still ran 10h / 12 tasks with 0
kernel_opt requests and 0 comm_optimization proposals — pure
backend/param grid scanning that found no gain because the report's
🔴 P1 recommendations (kernel rewrite for fmoe_fp8_blockscale_g1u1
underutilising FP8 peak, AllReduce+RMSNorm fusion saving 115 ms) were
never acted on.

N12 adds two new sections to orchestration.md as **hard rules**:

* "Roofline-v2 action ordering (HARD RULES — PolicyGate enforced)"
  — explains the staged flow (baseline → roofline → cheap actions →
  re-roofline → kernel_opt) and why the ordering matters (post-cheap
  kernel distribution differs from baseline). Forward-references the
  N13 PolicyGate enforcement so the LLM knows the rules are not
  optional.

* "Roofline-v2 analysis.md → action mapping (HARD RULES)" — maps each
  marker class in the TraceLens report to its target action:
  🔴/🟡 Compute Kernel Optimizations → kernel_opt
  🔴/🟢 Kernel Fusion Opportunities → kernel_opt (fused rewrite)
  🔴/🟡 System-Level Optimizations  → params / backends
  "GPU idle % > 30%" → scheduling / graph-capture flags via params
  "Exposed Communication % > 10%" → comm_optimization

These tests pin the new sections + their key phrases against
accidental removal during future doc edits. They do NOT test the
LLM's actual compliance (that's measured empirically in N8 GPU runs);
they only pin that the guidance text exists in the system prompt so
the LLM has a chance to follow it.
"""

from __future__ import annotations

from inference_optimizer.paths import asset_system_prompts_dir


def _load_orchestration_md() -> str:
    return (asset_system_prompts_dir() / "orchestration.md").read_text(
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Section presence
# ---------------------------------------------------------------------------
def test_orchestration_md_includes_action_ordering_hard_rules_section():
    text = _load_orchestration_md()
    assert "Roofline-v2 action ordering (HARD RULES — PolicyGate enforced)" in text


def test_orchestration_md_includes_analysis_md_to_action_mapping_section():
    text = _load_orchestration_md()
    assert "Roofline-v2 analysis.md → action mapping (HARD RULES)" in text


# ---------------------------------------------------------------------------
# Ordering rules content
# ---------------------------------------------------------------------------
def test_ordering_section_lists_5_stage_flow():
    """The staged flow is the central N12 contribution; every stage
    must be enumerated so the LLM (and operators reading the prompt
    snapshot) can verify the design intent."""
    text = _load_orchestration_md()
    for stage in (
        "baseline",
        "roofline",
        "Cheap exploration",
        "kernel_opt",
    ):
        assert stage in text, f"ordering section missing stage: {stage!r}"


def test_ordering_section_mentions_re_roofline_requirement_for_kernel_opt():
    """Stage 4 is the central insight — re-roofline before kernel_opt
    because cheap actions shift the kernel distribution."""
    text = _load_orchestration_md()
    assert "Do not propose `kernel_opt` until you have a fresh" in text
    assert "snapshot_id < 2" in text
    assert "backends_attempts < 1" in text
    assert "params_attempts < 1" in text


def test_ordering_section_documents_n13_escape_hatch():
    """The escape hatch must be discoverable from the prompt so a
    debug session can opt out."""
    text = _load_orchestration_md()
    assert "INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT" in text


def test_ordering_section_explains_why_ordering_matters():
    """The 'why' paragraph is the LLM's primary defence against
    blindly following a stale snapshot — pin a representative
    explanation phrase."""
    text = _load_orchestration_md()
    # Either of the two explanation framings is acceptable
    assert (
        "backend/param changes shift the kernel" in text
        or "kernel distribution. The 🔴 P1 in snapshot #1" in text
    )


# ---------------------------------------------------------------------------
# analysis.md → action mapping content
# ---------------------------------------------------------------------------
def test_mapping_section_lists_compute_kernel_to_kernel_opt():
    text = _load_orchestration_md()
    assert "Compute Kernel Optimizations" in text
    # Map to kernel_opt + REQUEST kind run_optimization
    assert "kernel_opt" in text
    assert "run_optimization" in text


def test_mapping_section_lists_fusion_opportunities_to_kernel_opt():
    text = _load_orchestration_md()
    assert "Kernel Fusion Opportunities" in text


def test_mapping_section_lists_system_level_to_params_backends():
    text = _load_orchestration_md()
    assert "System-Level Optimizations" in text
    assert "params" in text
    assert "backends" in text
    assert "discovered_flags" in text


def test_mapping_section_lists_idle_threshold_to_scheduling():
    """The "GPU idle % > 30%" rule is the catch-all for the
    most common R1/Qwen idle-bound workload pattern."""
    text = _load_orchestration_md()
    assert 'GPU idle %' in text
    assert '30%' in text


def test_mapping_section_lists_comm_threshold_to_comm_optimization():
    text = _load_orchestration_md()
    assert "Exposed Communication %" in text
    assert "comm_optimization" in text


def test_mapping_section_warns_against_guessing_a2a_backend_directly():
    """Operator hint: don't guess at `--moe-a2a-backend` values — use
    the dedicated `comm_optimization` action surface."""
    text = _load_orchestration_md()
    assert "Do NOT just guess" in text or "do not just guess" in text.lower()


# ---------------------------------------------------------------------------
# Hard rule emphasis (the LLM must see these as non-optional)
# ---------------------------------------------------------------------------
def test_ordering_and_mapping_marked_as_hard_rules_not_preferences():
    """Both new sections must be labelled 'HARD RULES' so the LLM
    treats them with the same weight as the pre-existing
    'Hard rules' section."""
    text = _load_orchestration_md()
    # Two "HARD RULES" headlines should be present (one per new section)
    assert text.count("HARD RULES") >= 2
    # Also: must reference PolicyGate enforcement (N13 backing)
    assert "PolicyGate" in text


def test_mapping_section_references_marker_classes():
    """The mapping section must reference the marker classes
    (🔴 / 🟡 / 🟢) used by TraceLens analysis.md so the LLM can
    pattern-match them in the injected report."""
    text = _load_orchestration_md()
    for marker in ("🔴", "🟡", "🟢"):
        assert marker in text, f"mapping section missing marker: {marker}"


# ---------------------------------------------------------------------------
# Pre-N12 sections must still be present (don't accidentally break
# the existing prompt structure)
# ---------------------------------------------------------------------------
def test_pre_n12_sections_intact():
    """N12 is additive — existing sections must still be present."""
    text = _load_orchestration_md()
    # Pre-existing sections from C5/N5/N9
    assert "How to consume the TraceLens Analysis" in text
    assert "NEVER propose `profile` directly" in text  # N9
    assert "design §6.5 N9" in text  # N9 design link
    # Output protocol
    assert "Output protocol" in text
    assert "emit_intent" in text
