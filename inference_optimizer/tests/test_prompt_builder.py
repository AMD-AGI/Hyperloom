# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests pinning :func:`build_orchestration_prompt`'s public contract."""

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


# Default enabled-action sets
def test_default_enabled_actions_full_includes_kernel_actions():
    full = default_enabled_actions(no_kernel=False)
    assert set(KERNEL_OWNED_ACTIONS) <= set(full)
    # Gap-10: validate_stack is deprecated; the merged ``explore`` action replaces it.
    assert "explore" in full
    assert "report" in full
    assert "baseline" in full


def test_default_enabled_actions_no_kernel_excludes_all_kernel_actions():
    bare = default_enabled_actions(no_kernel=True)
    assert set(KERNEL_OWNED_ACTIONS).isdisjoint(set(bare))
    assert "profile" not in bare  # profile only feeds kernel-opt
    # Gap-10: ``explore`` replaces the v0.6 backends/params/validate_stack triple.
    assert "explore" in bare
    assert "baseline" in bare


def test_full_enabled_actions_match_registry_minus_pmc_optional(registry):
    """All enabled actions must be present in the registry."""
    for name in FULL_ENABLED_ACTIONS:
        assert registry.get(name) is not None, (
            f"FULL_ENABLED_ACTIONS lists {name!r} but it's not in registry"
        )
    for name in NO_KERNEL_ENABLED_ACTIONS:
        assert registry.get(name) is not None


def test_recover_is_robustness_delegate_only_with_real_executor(registry):
    """``recover`` is ROBUSTNESS_DELEGATE_ONLY: real executor + metadata, but off the Orchestration prompt surface."""
    from inference_optimizer.cli import _REAL_EXECUTORS_FULL
    from inference_optimizer.orchestrator.action_executors.recover import (
        RecoverExecutor,
        recover_executor,
    )
    from inference_optimizer.protocol.action_surfaces import (
        ROBUSTNESS_DELEGATE_ONLY_ACTIONS,
    )

    assert "recover" not in FULL_ENABLED_ACTIONS
    assert "recover" not in NO_KERNEL_ENABLED_ACTIONS
    assert "recover" in ROBUSTNESS_DELEGATE_ONLY_ACTIONS
    assert registry.get("recover") is not None
    assert "recover" in _REAL_EXECUTORS_FULL
    assert _REAL_EXECUTORS_FULL["recover"] is recover_executor
    assert isinstance(recover_executor, RecoverExecutor)


# Output structure
def _section_headers(prompt: str) -> list[str]:
    return [
        line.strip() for line in prompt.splitlines()
        if line.startswith("## ") or line.startswith("### ")
    ]


def test_full_prompt_has_seven_sections(registry, rules_path):
    """v0.8 §3.3 — eight sections now (PHASE CONTRACT §3a added); legacy 1-7 headers preserved verbatim."""
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
        "## 3a. PHASE CONTRACT (v0.8 §3.2 / §3.3)",
        "## 4. ACTIONS YOU MAY USE",
        "## 5. DECISION FRAMEWORK (heuristics + facts — the next action is your call)",
        # N20-A "BACKENDS GRID CATALOGUE" + "PARAMS GRID CATALOGUE"
        # sections retired on this branch alongside the v0.6
        # backends / params executors (KB_design §3.4 / Dead-A).
        # The v0.8 ``explore`` action covers the same surface
        # internally; no per-action grid catalogue is rendered.
        "## 6. KERNEL-OPT REQUEST REFERENCE (payload templates — NOT a forced ordering)",
        # The former "## 6b. DYNAMIC ACTION (supplementary EXPLORE channel)"
        # section was removed when dynamic_action was folded into the unified
        # ``specialist`` (scope=domains). The cross-domain channel now lives
        # entirely inside §4 ACTIONS / the specialist dial documentation.
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
    assert "## 6. KERNEL-OPT REQUEST REFERENCE (payload templates — NOT a forced ordering)" not in headers
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
    # propose_action hint format for non-kernel.
    # v0.8 M3 + KB_gaps/Gap-10: ``backends`` was deprecated; the
    # canonical grid-runner is ``explore``.
    assert "propose_action{action_name='baseline'" in text
    assert "propose_action{action_name='explore'" in text
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
        enabled_actions=("baseline", "explore", "report"),
        framework="sglang",
        kernel_enabled=False,
        objective_kind="time_only",
        objective_value=None,
        max_minutes=60,
        rules_fragment_path=rules_path,
    )
    assert "## 6. KERNEL-OPT REQUEST REFERENCE" not in text


# ---------------------------------------------------------------------------
# Mission / time budget content
# ---------------------------------------------------------------------------
def test_mission_section_emphasises_cumulative_gain_and_stack_rebench(
    registry, rules_path,
):
    """v0.8 M3 + KB_gaps/Gap-10: the mission section emphasises the
    cumulative gain story; the standalone ``validate_stack`` keyword
    is gone (the rebench is inlined into ``explore``). We check for
    the new keyword ``stack rebench`` instead."""
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
    assert "stack rebench" in mission_block


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
    # The inlined stack rebench must be documented in the budget
    # section so the LLM treats it as part of the schedule.
    assert "stack rebench" in pipeline_block


# ---------------------------------------------------------------------------
# #144 last comment Layer 2: orchestrator must NOT pre-pin backends='claude'
# ---------------------------------------------------------------------------
def test_run_optimization_example_does_not_pin_backends_to_claude(
    registry, rules_path,
):
    """The ``run_optimization`` example in the kernel-opt pipeline section
    must NOT contain a literal ``backends: 'claude'`` (or any other backend
    pin). When the example carried that literal, the LLM echoed it on
    every kernel-opt request and ``kernel_optimization.choose_backends()``
    short-circuited to Claude only — even on hip_cpp+benchmark kernels
    that GEAK can rewrite (the exact regression closed in #144 last
    comment Layer 2)."""
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="time_only",
        objective_value=None,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    # The example block lives inside the kernel-opt pipeline section.
    assert "## 6. KERNEL-OPT PIPELINE" in text or "KERNEL-OPT PIPELINE" in text
    # Extract the K2 step block (where the run_optimization example lives).
    # We don't want to test on the entire prompt because legitimate
    # mentions of "claude" elsewhere (e.g. in the explanatory comment)
    # are fine.
    assert "kind: 'run_optimization'" in text
    k2_section = text.split("kind: 'run_optimization'", 1)[1]
    example_block = k2_section.split("budget_minutes")[0]
    # The example's `params:` block must NOT carry a `backends: 'claude'`
    # (or any other backend) literal.
    assert "backends: 'claude'" not in example_block, (
        "orchestrator example must not pin backends='claude' — that's the "
        "#144 last comment Layer 2 regression"
    )
    assert "backends: 'codex'" not in example_block
    assert "backends: 'geak'" not in example_block


def test_run_optimization_section_documents_auto_pick_rule(registry, rules_path):
    """The kernel-opt pipeline section must document that backends are
    auto-picked by the kernel-agent — so a future contributor doesn't
    re-add the literal pin "for clarity" and regress #144."""
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="time_only",
        objective_value=None,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    # Look for the auto-pick guidance in the kernel_opt section.
    # Main's prompt restructure (post-#207 merge) renamed "Step **K2**"
    # to "### kernel_opt — payload for run_optimization" and the next
    # subsection is "### integrate — payload"; split there.
    k2_section = text.split("kind: 'run_optimization'", 1)[1].split("### `integrate`")[0]
    assert "auto-pick" in k2_section.lower() or "auto-picks" in k2_section.lower(), (
        "kernel_opt step must document that backends are auto-picked"
    )
    assert "choose_backends" in k2_section, (
        "K2 step must reference the kernel-agent function that does the pick"
    )
    # And the historical regression must be called out so the rationale
    # survives a casual prompt-template cleanup.
    assert "#144" in k2_section
