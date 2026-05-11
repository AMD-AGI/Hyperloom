"""Unit tests for ``orchestrator/system_prompts/prompt_builder.py``.

These tests pin the public contract of :func:`build_orchestration_prompt`
without going through the CLI helpers so a future CLI rearrangement
doesn't silently break the prompt structure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_registry import (
    ActionRegistry,
    VALID_PIPELINE_PHASES,
)
from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    KERNEL_OWNED_ACTIONS,
    NO_KERNEL_ENABLED_ACTIONS,
    _PHASE_HEADERS,
    build_orchestration_prompt,
    default_enabled_actions,
)
from inference_optimizer.paths import asset_system_prompts_dir


@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry().load()


@pytest.fixture
def rules_path() -> Path:
    return asset_system_prompts_dir() / "orchestration.md"


# ---------------------------------------------------------------------------
# Default enabled-action sets
# ---------------------------------------------------------------------------
def test_default_enabled_actions_full_includes_kernel_actions():
    full = default_enabled_actions(no_kernel=False)
    assert set(KERNEL_OWNED_ACTIONS) <= set(full)
    assert "validate_stack" in full
    assert "report" in full
    assert "baseline" in full


def test_default_enabled_actions_no_kernel_excludes_all_kernel_actions():
    bare = default_enabled_actions(no_kernel=True)
    assert set(KERNEL_OWNED_ACTIONS).isdisjoint(set(bare))
    assert "profile" not in bare  # profile only feeds kernel-opt
    assert "validate_stack" in bare
    assert "baseline" in bare


def test_full_enabled_actions_match_registry_minus_pmc_optional(registry):
    """All enabled actions must be present in the registry."""
    for name in FULL_ENABLED_ACTIONS:
        assert registry.get(name) is not None, (
            f"FULL_ENABLED_ACTIONS lists {name!r} but it's not in registry"
        )
    for name in NO_KERNEL_ENABLED_ACTIONS:
        assert registry.get(name) is not None


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------
def _section_headers(prompt: str) -> list[str]:
    return [
        line.strip() for line in prompt.splitlines()
        if line.startswith("## ") or line.startswith("### ")
    ]


def test_full_prompt_has_seven_sections(registry, rules_path):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="gain_pct",
        objective_value=10.0,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    headers = _section_headers(text)
    expected_top_level = [
        "## 1. MISSION",
        "## 2. SESSION CONTEXT",
        "## 3. PIPELINE & TIME BUDGET",
        "## 4. ACTIONS YOU MAY USE",
        "## 5. DECISION FRAMEWORK (apply EVERY tick BEFORE emitting)",
        "## 6. KERNEL-OPT PIPELINE (sequential, no backtracking)",
        "## 7. RULES & OUTPUT PROTOCOL",
    ]
    actual_top = [h for h in headers if h.startswith("## ")]
    assert actual_top == expected_top_level


def test_no_kernel_prompt_drops_section_six(registry, rules_path):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=NO_KERNEL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="time_only",
        objective_value=None,
        max_minutes=60,
        rules_fragment_path=rules_path,
    )
    headers = [h for h in _section_headers(text) if h.startswith("## ")]
    assert "## 6. KERNEL-OPT PIPELINE (sequential, no backtracking)" not in headers
    # Other sections still present
    assert "## 1. MISSION" in headers
    assert "## 4. ACTIONS YOU MAY USE" in headers
    assert "## 7. RULES & OUTPUT PROTOCOL" in headers


def test_full_prompt_has_all_phase_subheaders(registry, rules_path):
    """ACTIONS YOU MAY USE groups by pipeline_phase; each phase used by the
    enabled set must appear as a ``###`` sub-header."""
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="time_only",
        objective_value=None,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    enabled_metas = [registry.get(n) for n in FULL_ENABLED_ACTIONS]
    expected_phases = {m.pipeline_phase for m in enabled_metas if m is not None}
    sub_headers = {h for h in _section_headers(text) if h.startswith("### ")}
    for phase in expected_phases:
        assert f"### {phase}" in sub_headers, (
            f"expected phase header '### {phase}' in catalogue, got {sub_headers}"
        )
        # Each phase MUST have a header line in the time-budget mapping
        # (defensive — the catalogue uses the same _PHASE_HEADERS contract).
        assert phase in _PHASE_HEADERS, (
            f"phase {phase!r} present in registry but missing from "
            f"_PHASE_HEADERS — update prompt_builder._PHASE_HEADERS"
        )


def test_phase_header_keys_subset_of_valid_phases():
    """Defensive: builder phase headers must be a subset of registry's
    VALID_PIPELINE_PHASES so we don't drift between schema and prompt."""
    assert set(_PHASE_HEADERS.keys()) <= VALID_PIPELINE_PHASES


# ---------------------------------------------------------------------------
# Action-level content
# ---------------------------------------------------------------------------
def test_every_enabled_action_appears_with_description(registry, rules_path):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="gain_pct",
        objective_value=10.0,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    for name in FULL_ENABLED_ACTIONS:
        meta = registry.get(name)
        assert meta is not None
        # Bullet line for the action
        assert f"**{name}**" in text, f"action {name!r} missing from catalogue"
        # And its description
        assert meta.description in text, (
            f"description for {name!r} missing or truncated"
        )


def test_kernel_owned_actions_marked_kernel_owned(registry, rules_path):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="time_only",
        objective_value=None,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    for name in KERNEL_OWNED_ACTIONS:
        # Marker on the bullet line so the LLM picks the REQUEST transport
        marker = f"**{name}** (KERNEL-OWNED)"
        assert marker in text, (
            f"expected '(KERNEL-OWNED)' tag next to {name!r}"
        )


def test_emit_hints_use_request_for_kernel_actions(registry, rules_path):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="time_only",
        objective_value=None,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    # propose_action hint format for non-kernel
    assert "propose_action{action_name='baseline'" in text
    assert "propose_action{action_name='backends'" in text
    # REQUEST hint for kernel-owned
    assert "REQUEST{target_agent='kernel'" in text


# ---------------------------------------------------------------------------
# Determinism + fragment fallback
# ---------------------------------------------------------------------------
def test_build_is_deterministic(registry, rules_path):
    args = dict(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="gain_pct",
        objective_value=10.0,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    assert build_orchestration_prompt(**args) == build_orchestration_prompt(**args)


def test_missing_rules_fragment_yields_placeholder(tmp_path, registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=NO_KERNEL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="time_only",
        objective_value=None,
        max_minutes=60,
        rules_fragment_path=tmp_path / "does-not-exist.md",
    )
    assert "## 7. RULES & OUTPUT PROTOCOL" in text
    assert "rules fragment not found" in text


def test_unknown_enabled_action_is_silently_skipped(registry, rules_path):
    """Pass a phantom action name; builder must skip it without raising."""
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=("baseline", "no_such_action", "report"),
        framework="sglang",
        objective_kind="time_only",
        objective_value=None,
        max_minutes=60,
        rules_fragment_path=rules_path,
        kernel_enabled=False,
    )
    assert "**baseline**" in text
    assert "**report**" in text
    assert "no_such_action" not in text


def test_explicit_kernel_enabled_override_wins(registry, rules_path):
    """Even with FULL set, if kernel_enabled=False is forced, no K-pipeline."""
    text = build_orchestration_prompt(
        action_registry=registry,
        # Caller forgot to filter — builder must STILL respect explicit flag
        enabled_actions=("baseline", "backends", "params", "report"),
        framework="sglang",
        kernel_enabled=False,
        objective_kind="time_only",
        objective_value=None,
        max_minutes=60,
        rules_fragment_path=rules_path,
    )
    assert "## 6. KERNEL-OPT PIPELINE" not in text


# ---------------------------------------------------------------------------
# Mission / time budget content
# ---------------------------------------------------------------------------
def test_mission_section_emphasises_cumulative_gain_and_validate_stack(
    registry, rules_path,
):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="gain_pct",
        objective_value=10.0,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    mission_block = text.split("## 2.")[0]
    assert "cumulative_gain" in mission_block
    assert "validate_stack" in mission_block


def test_time_budget_section_lists_all_enabled_phases(registry, rules_path):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="time_only",
        objective_value=None,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    pipeline_block = text.split("## 3. PIPELINE & TIME BUDGET")[1].split("## 4.")[0]
    enabled_metas = [registry.get(n) for n in FULL_ENABLED_ACTIONS]
    expected_phases = {m.pipeline_phase for m in enabled_metas if m is not None}
    for phase in expected_phases:
        assert f"**{phase}**" in pipeline_block, (
            f"phase {phase!r} missing from time budget summary"
        )
    assert "Sum of typical phase ETAs" in pipeline_block
    assert "max_minutes=120" in pipeline_block
    # Mandatory rule about validate_stack must be in the budget section so
    # the LLM treats it as part of the schedule, not a side note.
    assert "validate_stack" in pipeline_block
