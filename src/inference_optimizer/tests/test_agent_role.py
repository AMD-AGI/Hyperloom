"""Tests for orchestrator/agent_role.py — v0.4 MVP roster
(executor / critic / triage / kernel — all Claude). See
standalone_agent_design §13.1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator import agent_role as ar
from inference_optimizer.orchestrator.agent_role import (
    BackendType,
    ROLE_CRITIC,
    ROLE_EXECUTOR,
    ROLE_KERNEL,
    ROLE_TRIAGE,
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
    assert set(reg.keys()) == {"executor", "critic", "triage", "kernel"}


def test_executor_is_claude_with_tools():
    assert ROLE_EXECUTOR.backend is BackendType.CLAUDE
    assert ROLE_EXECUTOR.no_tools is False
    assert ROLE_EXECUTOR.can_delegate_side_effects is True
    assert ROLE_EXECUTOR.can_mutate_core_state is False
    assert IntentType.PROPOSE_ACTION in ROLE_EXECUTOR.allowed_intents
    assert IntentType.DELEGATE in ROLE_EXECUTOR.allowed_intents
    assert IntentType.UPDATE_STATE in ROLE_EXECUTOR.allowed_intents
    # Plan A: only executor may emit REQUEST (target=kernel today).
    assert IntentType.REQUEST in ROLE_EXECUTOR.allowed_intents
    # v0.4: parliament removed → executor never had OBJECTION/VOTE anyway.


def test_kernel_is_claude_responder_only():
    """Kernel agent: Claude-backed, RESPONSE-only, no DELEGATE / no REQUEST."""
    assert ROLE_KERNEL.backend is BackendType.CLAUDE
    assert ROLE_KERNEL.no_tools is False
    assert ROLE_KERNEL.can_delegate_side_effects is False
    assert ROLE_KERNEL.can_mutate_core_state is False
    assert IntentType.RESPONSE in ROLE_KERNEL.allowed_intents
    # Forbidden: kernel agent never initiates RPCs / delegates / proposes.
    assert IntentType.REQUEST not in ROLE_KERNEL.allowed_intents
    assert IntentType.DELEGATE not in ROLE_KERNEL.allowed_intents
    assert IntentType.PROPOSE_ACTION not in ROLE_KERNEL.allowed_intents
    assert IntentType.UPDATE_STATE not in ROLE_KERNEL.allowed_intents
    # v0.4 — kernel never emits kill_task either.
    assert IntentType.KILL_TASK not in ROLE_KERNEL.allowed_intents
    # Allowed observability intents (inherited from _BASE_INTENTS).
    assert IntentType.SEND_MESSAGE in ROLE_KERNEL.allowed_intents
    assert IntentType.ALERT in ROLE_KERNEL.allowed_intents
    assert IntentType.ASK_QUESTION in ROLE_KERNEL.allowed_intents
    assert IntentType.UPDATE_PERSONA in ROLE_KERNEL.allowed_intents


def test_triage_is_claude_with_kill_task():
    """Triage (v0.4 MVP) — Claude-backed always-on, the ONLY role with
    KILL_TASK. PolicyGate further restricts source via
    KILL_TASK_SOURCE_ALLOWLIST.
    """
    assert ROLE_TRIAGE.backend is BackendType.CLAUDE
    assert ROLE_TRIAGE.no_tools is False
    assert ROLE_TRIAGE.can_delegate_side_effects is False
    assert ROLE_TRIAGE.can_mutate_core_state is False
    assert IntentType.KILL_TASK in ROLE_TRIAGE.allowed_intents
    assert IntentType.UPDATE_STATE in ROLE_TRIAGE.allowed_intents
    assert IntentType.ALERT in ROLE_TRIAGE.allowed_intents
    # Triage must never delegate / propose / request.
    assert IntentType.DELEGATE not in ROLE_TRIAGE.allowed_intents
    assert IntentType.PROPOSE_ACTION not in ROLE_TRIAGE.allowed_intents
    assert IntentType.REQUEST not in ROLE_TRIAGE.allowed_intents
    assert IntentType.RESPONSE not in ROLE_TRIAGE.allowed_intents


def test_only_executor_may_emit_request():
    """REQUEST is reserved for executor; every other role must not have it."""
    for role in (ROLE_CRITIC, ROLE_TRIAGE, ROLE_KERNEL):
        assert IntentType.REQUEST not in role.allowed_intents, (
            f"{role.name} must not emit REQUEST"
        )


def test_only_kernel_may_emit_response():
    """RESPONSE is reserved for kernel agent; every other role must not have it."""
    for role in (ROLE_EXECUTOR, ROLE_CRITIC, ROLE_TRIAGE):
        assert IntentType.RESPONSE not in role.allowed_intents, (
            f"{role.name} must not emit RESPONSE"
        )


def test_only_triage_may_emit_kill_task():
    """KILL_TASK on the role gate — PolicyGate further enforces via the
    source allowlist (see test_policy.py)."""
    for role in (ROLE_EXECUTOR, ROLE_CRITIC, ROLE_KERNEL):
        assert IntentType.KILL_TASK not in role.allowed_intents, (
            f"{role.name} must not emit KILL_TASK"
        )


def test_critic_is_claude_no_delegate():
    """v0.4 — critic flipped from Codex to Claude. No OBJECTION/VOTE
    (parliament removed); no delegate / propose_action / update_state.
    """
    assert ROLE_CRITIC.backend is BackendType.CLAUDE
    assert ROLE_CRITIC.no_tools is False
    assert ROLE_CRITIC.can_delegate_side_effects is False
    assert ROLE_CRITIC.can_mutate_core_state is False
    # Critic must not be allowed to delegate / mutate state / propose.
    assert IntentType.DELEGATE not in ROLE_CRITIC.allowed_intents
    assert IntentType.UPDATE_STATE not in ROLE_CRITIC.allowed_intents
    assert IntentType.PROPOSE_ACTION not in ROLE_CRITIC.allowed_intents
    # Critic emits verdicts via send_message; can also alert.
    assert IntentType.SEND_MESSAGE in ROLE_CRITIC.allowed_intents
    assert IntentType.ALERT in ROLE_CRITIC.allowed_intents


def test_all_roles_can_send_message_and_alert():
    for role in (ROLE_EXECUTOR, ROLE_CRITIC, ROLE_TRIAGE, ROLE_KERNEL):
        assert IntentType.SEND_MESSAGE in role.allowed_intents
        assert IntentType.ALERT in role.allowed_intents
        assert IntentType.UPDATE_PERSONA in role.allowed_intents


# ---------------------------------------------------------------------------
# roles_for_mode (v0.4 — triage always-on; guided/marathon roster identical)
# ---------------------------------------------------------------------------
def test_roles_for_quick_mode_is_executor_plus_triage():
    rs = roles_for_mode(ExecutionMode.QUICK_PARAM_SWEEP)
    assert [r.name for r in rs] == ["executor", "triage"]


def test_roles_for_guided_full_roster():
    rs = roles_for_mode(ExecutionMode.GUIDED_KERNEL_OPT)
    assert [r.name for r in rs] == ["executor", "critic", "kernel", "triage"]


def test_roles_for_marathon_same_as_guided():
    """v0.4 collapses guided/marathon roster to the same 4 agents."""
    rs = roles_for_mode(ExecutionMode.MARATHON_MULTI_AGENT)
    assert [r.name for r in rs] == ["executor", "critic", "kernel", "triage"]


def test_roles_for_mode_uses_supplied_registry():
    custom_executor = claude_role("executor", model="claude-opus-test")
    custom_critic = claude_role("critic", model="claude-opus-test",
                                can_delegate_side_effects=False)
    custom_kernel = claude_role("kernel", model="claude-opus-test",
                                can_delegate_side_effects=False)
    custom_triage = claude_role("triage", model="claude-opus-test",
                                can_delegate_side_effects=False)
    reg = {
        "executor": custom_executor,
        "critic": custom_critic,
        "kernel": custom_kernel,
        "triage": custom_triage,
    }
    rs = roles_for_mode(ExecutionMode.GUIDED_KERNEL_OPT, registry=reg)
    assert rs == [custom_executor, custom_critic, custom_kernel, custom_triage]


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
    """Codex factory retained for v0.5 even though no v0.4 role binds to it."""
    role = codex_role("custom-codex-role")
    assert role.backend is BackendType.CODEX
    assert role.no_tools is True
    assert role.can_delegate_side_effects is False
    assert role.can_mutate_core_state is False


def test_codex_role_factory_with_custom_intents():
    role = codex_role(
        "auditor",
        allowed_intents=[IntentType.SEND_MESSAGE, IntentType.ALERT],
    )
    assert IntentType.SEND_MESSAGE in role.allowed_intents
    assert IntentType.ALERT in role.allowed_intents
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


def test_load_system_prompt_for_triage_role_v04():
    """v0.4 triage system prompt should exist and be non-trivial."""
    text = load_system_prompt("triage")
    assert text and len(text) > 100
    assert "kill_task" in text.lower() or "kill task" in text.lower()


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
