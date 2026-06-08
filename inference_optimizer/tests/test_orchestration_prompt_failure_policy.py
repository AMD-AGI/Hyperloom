# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""FAILURE RECOVERY block contract tests for the orchestration prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    NO_KERNEL_ENABLED_ACTIONS,
    build_orchestration_prompt,
)
from inference_optimizer.paths import asset_system_prompts_dir


@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry().load()


@pytest.fixture
def rules_path() -> Path:
    return asset_system_prompts_dir() / "orchestration.md"


def _prompt(registry, rules_path, *, enabled=FULL_ENABLED_ACTIONS) -> str:
    return build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=enabled,
        framework="sglang",
        objective_kind="gain_pct",
        objective_value=10.0,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )


def test_prompt_contains_failure_recovery_header(registry, rules_path):
    text = _prompt(registry, rules_path)
    assert "### FAILURE RECOVERY" in text


def test_prompt_failure_recovery_names_state_surfaces(registry, rules_path):
    text = _prompt(registry, rules_path)
    for needle in (
        "last_<action>",
        "<action>_attempts",
        "last_action_failures",
        "baseline_failure_streak",
        "extras.fingerprint",
    ):
        assert needle in text, f"FAILURE RECOVERY missing surface: {needle!r}"


def test_prompt_failure_recovery_names_baseline_rules(registry, rules_path):
    text = _prompt(registry, rules_path)
    assert "RULE BF1" in text
    assert "RULE BF2" in text
    assert "RULE BF3" in text
    assert "baseline_no_param_change" in text


def test_prompt_failure_recovery_names_nonbaseline_rule(registry, rules_path):
    text = _prompt(registry, rules_path)
    assert "RULE F4" in text
    assert "policy_loop" in text


def test_prompt_failure_recovery_mentions_policy_rule_tag(registry, rules_path):
    """The exact PolicyGate rule string must appear so the LLM can correlate denial events with recovery."""
    text = _prompt(registry, rules_path)
    assert "baseline_no_param_change" in text


def test_prompt_failure_recovery_forbids_param_changes(registry, rules_path):
    """The prompt must explicitly tell the LLM not to change params."""
    text = _prompt(registry, rules_path)
    assert "do NOT change baseline params" in text or \
           "Do not tweak" in text


def test_failure_recovery_block_present_in_no_kernel_prompt(registry, rules_path):
    """Recovery semantics aren't kernel-specific; the block must also appear in the no-kernel prompt."""
    text = _prompt(registry, rules_path, enabled=NO_KERNEL_ENABLED_ACTIONS)
    assert "### FAILURE RECOVERY" in text
    assert "RULE BF1" in text


def test_failure_recovery_appears_after_decision_framework_header(
    registry, rules_path,
):
    """Anchor the block under section 5 so a header rename elsewhere doesn't swallow it."""
    text = _prompt(registry, rules_path)
    dframe_idx = text.index("## 5. DECISION FRAMEWORK")
    fr_idx = text.index("### FAILURE RECOVERY")
    rules_idx = text.index("## 7. RULES & OUTPUT PROTOCOL")
    assert dframe_idx < fr_idx < rules_idx
