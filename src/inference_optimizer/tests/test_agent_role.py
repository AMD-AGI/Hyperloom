"""Tests for orchestrator/agent_role.py — DESIGN §5.1."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator import agent_role as ar
from inference_optimizer.orchestrator.agent_role import (
    BackendType,
    ROLE_CRITIC,
    ROLE_EXECUTOR,
    ROLE_SAGE,
    ROLE_WATCHDOG,
    claude_role,
    codex_role,
    default_role_registry,
    load_system_prompt,
    roles_for_mode,
    system_prompts_dir,
)
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.intent_parser import IntentType


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_default_registry_has_four_roles():
    reg = default_role_registry()
    assert set(reg.keys()) == {"executor", "critic", "watchdog", "sage"}


def test_executor_is_claude_with_tools():
    assert ROLE_EXECUTOR.backend is BackendType.CLAUDE
    assert ROLE_EXECUTOR.no_tools is False
    assert ROLE_EXECUTOR.can_delegate_side_effects is True
    assert ROLE_EXECUTOR.can_mutate_core_state is False
    assert IntentType.PROPOSE_ACTION in ROLE_EXECUTOR.allowed_intents
    assert IntentType.DELEGATE in ROLE_EXECUTOR.allowed_intents
    assert IntentType.UPDATE_STATE in ROLE_EXECUTOR.allowed_intents


def test_watchdog_is_claude_with_tools_marathon_only():
    assert ROLE_WATCHDOG.backend is BackendType.CLAUDE
    assert ROLE_WATCHDOG.no_tools is False
    assert ROLE_WATCHDOG.can_delegate_side_effects is True
    assert ROLE_WATCHDOG.can_mutate_core_state is False


def test_critic_is_codex_no_tools_no_delegate():
    assert ROLE_CRITIC.backend is BackendType.CODEX
    assert ROLE_CRITIC.no_tools is True
    assert ROLE_CRITIC.can_delegate_side_effects is False
    assert ROLE_CRITIC.can_mutate_core_state is False
    # Codex roles must not be allowed to delegate or mutate state directly.
    assert IntentType.DELEGATE not in ROLE_CRITIC.allowed_intents
    assert IntentType.UPDATE_STATE not in ROLE_CRITIC.allowed_intents
    # But they review proposals (objection, vote).
    assert IntentType.OBJECTION in ROLE_CRITIC.allowed_intents
    assert IntentType.VOTE in ROLE_CRITIC.allowed_intents


def test_sage_is_codex_no_tools_with_propose_action():
    assert ROLE_SAGE.backend is BackendType.CODEX
    assert ROLE_SAGE.no_tools is True
    assert ROLE_SAGE.can_delegate_side_effects is False
    # Marathon Sage may propose strategic_review etc. (DESIGN §5.1.2).
    assert IntentType.PROPOSE_ACTION in ROLE_SAGE.allowed_intents
    assert IntentType.OBJECTION in ROLE_SAGE.allowed_intents
    # But never delegate workspace side-effects.
    assert IntentType.DELEGATE not in ROLE_SAGE.allowed_intents


def test_all_roles_can_send_message_and_alert():
    for role in (ROLE_EXECUTOR, ROLE_WATCHDOG, ROLE_CRITIC, ROLE_SAGE):
        assert IntentType.SEND_MESSAGE in role.allowed_intents
        assert IntentType.ALERT in role.allowed_intents
        assert IntentType.UPDATE_PERSONA in role.allowed_intents


# ---------------------------------------------------------------------------
# roles_for_mode
# ---------------------------------------------------------------------------
def test_roles_for_quick_mode_only_executor():
    rs = roles_for_mode(ExecutionMode.QUICK_PARAM_SWEEP)
    assert [r.name for r in rs] == ["executor"]


def test_roles_for_guided_executor_plus_critic():
    rs = roles_for_mode(ExecutionMode.GUIDED_KERNEL_OPT)
    assert [r.name for r in rs] == ["executor", "critic"]


def test_roles_for_marathon_full_roster():
    rs = roles_for_mode(ExecutionMode.MARATHON_MULTI_AGENT)
    assert [r.name for r in rs] == ["executor", "critic", "watchdog", "sage"]


def test_roles_for_mode_uses_supplied_registry():
    custom_executor = claude_role("executor", model="claude-opus-test")
    custom_critic = codex_role("critic", model="gpt-test")
    reg = {"executor": custom_executor, "critic": custom_critic}
    rs = roles_for_mode(ExecutionMode.GUIDED_KERNEL_OPT, registry=reg)
    assert rs == [custom_executor, custom_critic]


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def test_claude_role_factory_defaults():
    role = claude_role("executor-test")
    assert role.backend is BackendType.CLAUDE
    assert role.api_key_env == "ANTHROPIC_API_KEY"
    assert role.no_tools is False
    assert role.can_delegate_side_effects is True
    assert role.can_mutate_core_state is False


def test_codex_role_factory_forbids_delegation():
    role = codex_role("custom-critic")
    assert role.backend is BackendType.CODEX
    assert role.no_tools is True
    assert role.can_delegate_side_effects is False
    assert role.can_mutate_core_state is False


def test_codex_role_factory_with_custom_intents():
    role = codex_role(
        "auditor",
        allowed_intents=[IntentType.SEND_MESSAGE, IntentType.OBJECTION],
    )
    assert IntentType.SEND_MESSAGE in role.allowed_intents
    assert IntentType.OBJECTION in role.allowed_intents
    assert IntentType.PROPOSE_ACTION not in role.allowed_intents


# ---------------------------------------------------------------------------
# System prompt loader
# ---------------------------------------------------------------------------
def test_system_prompts_dir_exists():
    p = system_prompts_dir()
    assert p.exists() and p.is_dir()


def test_load_system_prompt_for_known_role():
    text = load_system_prompt("executor")
    # Real prompt is non-trivial; just verify we got something.
    assert text and len(text) > 50


def test_load_system_prompt_falls_back_for_unknown_role():
    text = load_system_prompt("__definitely_not_a_role__")
    assert "__definitely_not_a_role__" in text


def test_role_system_prompt_method():
    role = ROLE_CRITIC
    text = role.system_prompt()
    assert text and len(text) > 50


def test_load_system_prompt_uses_cache(monkeypatch, tmp_path):
    # We can't redirect the cached path, but we can verify the cache hits the
    # same string twice (cheap correctness check).
    a = load_system_prompt("executor")
    b = load_system_prompt("executor")
    assert a is b
