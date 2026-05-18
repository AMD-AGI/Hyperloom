"""Tests for :mod:`critic_prompt_builder`."""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.system_prompts.critic_prompt_builder import (
    build_critic_prompt,
)
from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
    default_enabled_actions,
)
from inference_optimizer.paths import asset_system_prompts_dir


@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry().load()


def _rules_path():
    return asset_system_prompts_dir() / "critic.md"


def test_section_headers_present(registry):
    text = build_critic_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        kernel_enabled=True,
        max_minutes=120,
        rules_fragment_path=_rules_path(),
    )
    for header in (
        "## 1. MISSION",
        "## 2. RUN CONTEXT",
        "## 3. KNOWN ACTIONS",
        "## 4. DEFAULT VERDICT",
        "## 5. KERNEL-OWNED CARVE-OUT",
        "## 6. RULES",
        "## 7. OUTPUT PROTOCOL",
    ):
        assert header in text, f"missing {header}"


def test_deterministic(registry):
    kwargs = dict(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="vllm",
        kernel_enabled=True,
        max_minutes=60,
        rules_fragment_path=_rules_path(),
    )
    assert build_critic_prompt(**kwargs) == build_critic_prompt(**kwargs)


def test_full_prompt_contains_all_registered_actions(registry):
    """Regression guard: every action in _meta must appear in §3."""
    text = build_critic_prompt(
        action_registry=registry,
        enabled_actions=registry.names(),
        framework="sglang",
        kernel_enabled=True,
        max_minutes=60,
        rules_fragment_path=_rules_path(),
    )
    for name in registry.names():
        assert f"**{name}**" in text, f"action {name!r} missing from KNOWN ACTIONS"


def test_validate_stack_in_both_modes(registry):
    for no_kernel in (False, True):
        enabled = default_enabled_actions(no_kernel=no_kernel)
        text = build_critic_prompt(
            action_registry=registry,
            enabled_actions=enabled,
            framework="sglang",
            kernel_enabled=not no_kernel,
            max_minutes=60,
            rules_fragment_path=_rules_path(),
        )
        assert "validate_stack" in text, (
            f"validate_stack missing (no_kernel={no_kernel})"
        )


def test_no_kernel_mode_drops_kernel_owned(registry):
    text = build_critic_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=True),
        framework="sglang",
        kernel_enabled=False,
        max_minutes=60,
        rules_fragment_path=_rules_path(),
    )
    assert "## 5. KERNEL-OWNED CARVE-OUT" not in text
    for name in ("kernel_opt", "integrate", "deep_kernel_analysis"):
        assert f"**{name}**" not in text, (
            f"{name} should not appear in no-kernel catalogue"
        )
