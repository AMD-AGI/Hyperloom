"""FAILURE RECOVERY block contract tests for the orchestration prompt.

The block must:

* Appear under the DECISION FRAMEWORK section.
* Name the three SharedState surfaces the LLM is expected to consult
  (``last_<action>`` / ``<action>_attempts`` / ``last_action_failures``).
* Spell out the three decision rules (F1 same-fingerprint denial /
  F2 ``error_class='no_report'`` salvage miss / F3 ``subprocess_nonzero``
  heartbeat-out).
* Mention the two recovery knobs Orchestration may set in
  ``propose_action{params=...}``: ``benchmark_script`` and ``result_dir``.

Failing these assertions is a strong signal the prompt drifted away
from what PolicyGate actually enforces — the prompt must keep referring
to the rule names the Coordinator emits (``baseline_self_loop``,
``rescued_from_leaked_path:*``, etc.) so the LLM can correlate the two.
"""

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


def test_prompt_failure_recovery_lists_fingerprint_keys(registry, rules_path):
    """The eight ``_BASELINE_FINGERPRINT_KEYS`` must show up so the LLM
    knows which fields actually change what PolicyGate sees."""
    text = _prompt(registry, rules_path)
    expected_keys = (
        "benchmark_script", "result_dir", "extra_server_args",
        "extra_envs", "model_path", "gpu_type", "config_path",
        "disable_run_eval",
    )
    for key in expected_keys:
        assert key in text, f"fingerprint key {key!r} missing from prompt"


def test_prompt_failure_recovery_names_four_rules(registry, rules_path):
    text = _prompt(registry, rules_path)
    assert "RULE F1" in text
    assert "RULE F2" in text
    assert "RULE F3" in text
    assert "RULE F4" in text
    assert "policy_loop" in text


def test_prompt_failure_recovery_mentions_policy_rule_tag(registry, rules_path):
    """The exact PolicyGate rule string must appear so the LLM can
    correlate ``policy_denied{rule: 'baseline_self_loop'}`` with the
    recovery instructions."""
    text = _prompt(registry, rules_path)
    assert "baseline_self_loop" in text


def test_prompt_failure_recovery_has_recovery_example(registry, rules_path):
    text = _prompt(registry, rules_path)
    # A concrete propose_action example using the new params surface.
    assert "propose_action{action_name='baseline'" in text
    assert "sglang_mi300x.sh" in text


def test_failure_recovery_block_present_in_no_kernel_prompt(registry, rules_path):
    """Recovery semantics are not kernel-specific — the block must also
    appear in the no-kernel prompt (baseline / backends / params /
    sweep / validate_stack are all included there)."""
    text = _prompt(registry, rules_path, enabled=NO_KERNEL_ENABLED_ACTIONS)
    assert "### FAILURE RECOVERY" in text
    assert "RULE F1" in text


def test_failure_recovery_appears_after_decision_framework_header(
    registry, rules_path,
):
    """Anchor the block under section 5 so an unrelated header rename
    in section 4/6 doesn't accidentally swallow it."""
    text = _prompt(registry, rules_path)
    dframe_idx = text.index("## 5. DECISION FRAMEWORK")
    fr_idx = text.index("### FAILURE RECOVERY")
    rules_idx = text.index("## 7. RULES & OUTPUT PROTOCOL")
    assert dframe_idx < fr_idx < rules_idx
