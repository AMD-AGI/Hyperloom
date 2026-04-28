"""Tests for orchestrator/policy.py — DESIGN §10.5.7 / §10.5.8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pytest

from inference_optimizer.orchestrator.agent_role import (
    ROLE_CRITIC,
    ROLE_EXECUTOR,
    ROLE_SAGE,
    ROLE_WATCHDOG,
    default_role_registry,
)
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.feature_flags import build_feature_flags
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.policy import (
    CORE_STATE_FIELDS,
    DEFAULT_QUICK_ACTION_ALLOWLIST,
    PolicyDenied,
    PolicyGate,
    QUICK_BASH_DENYLIST,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_gate(
    mode: ExecutionMode = ExecutionMode.GUIDED_KERNEL_OPT,
    *,
    action_registry=None,
) -> PolicyGate:
    return PolicyGate(
        flags=build_feature_flags(mode),
        mode=mode,
        role_registry=default_role_registry(),
        action_registry=action_registry,
    )


def intent(t: IntentType, **payload) -> Intent:
    return Intent(type=t, payload=payload)


# ---------------------------------------------------------------------------
# Role gate
# ---------------------------------------------------------------------------
def test_unknown_agent_denied():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("nobody", intent(IntentType.SEND_MESSAGE, topic="event"))
    assert exc.value.rule == "role"


def test_role_cannot_emit_forbidden_intent_type():
    gate = make_gate()
    # Critic cannot delegate by spec.
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "critic", intent(IntentType.DELEGATE, action_name="kernel_opt")
        )
    assert exc.value.rule == "role"
    assert "critic" in str(exc.value)


def test_executor_can_send_message():
    gate = make_gate()
    gate.validate_intent(
        "executor",
        intent(IntentType.SEND_MESSAGE, topic="event", body_md="hi"),
    )  # no exception


def test_critic_can_objection_and_vote():
    gate = make_gate()
    gate.validate_intent(
        "critic",
        intent(IntentType.OBJECTION, target_msg_id="abc", reason="risky"),
    )
    gate.validate_intent(
        "critic", intent(IntentType.VOTE, target_msg_id="abc", vote="reject")
    )


# ---------------------------------------------------------------------------
# Delegate gate
# ---------------------------------------------------------------------------
def test_delegate_blocked_in_quick_mode_via_feature_flag():
    gate = make_gate(ExecutionMode.QUICK_PARAM_SWEEP)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(IntentType.DELEGATE, action_name="kernel_opt"),
        )
    assert exc.value.rule == "mode"
    assert "delegation disabled" in str(exc.value)


def test_delegate_codex_role_blocked_by_role():
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT)
    # Sage doesn't have DELEGATE in allowed_intents anyway -> role rule fires.
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "sage", intent(IntentType.DELEGATE, action_name="kernel_opt"),
        )
    assert exc.value.rule == "role"


def test_delegate_missing_action_name():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("executor", intent(IntentType.DELEGATE))
    assert exc.value.rule == "payload"


def test_delegate_in_guided_with_no_registry_accepts():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    gate.validate_intent(
        "executor",
        intent(IntentType.DELEGATE, action_name="anything_goes_in_guided"),
    )  # no exception


def test_delegate_quick_action_allowlist_enforced_when_mode_allows():
    # Force a fake mode where delegate flag is on but mode is quick-like.
    # We do this by overriding the gate directly.
    gate = PolicyGate(
        flags=build_feature_flags(ExecutionMode.GUIDED_KERNEL_OPT),
        mode=ExecutionMode.QUICK_PARAM_SWEEP,  # mismatch on purpose
        role_registry=default_role_registry(),
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(IntentType.DELEGATE, action_name="not_in_allowlist"),
        )
    assert exc.value.rule == "mode"


def test_delegate_quick_action_allowlist_passes_for_known_action():
    gate = PolicyGate(
        flags=build_feature_flags(ExecutionMode.GUIDED_KERNEL_OPT),
        mode=ExecutionMode.QUICK_PARAM_SWEEP,
        role_registry=default_role_registry(),
    )
    gate.validate_intent(
        "executor",
        intent(IntentType.DELEGATE, action_name="bench_runner"),
    )


# ---------------------------------------------------------------------------
# Action registry integration (Phase 4 will provide a real registry)
# ---------------------------------------------------------------------------
@dataclass
class _StubAction:
    name: str
    allowed_modes: tuple[ExecutionMode, ...]
    allowed_tools: tuple[str, ...] = ("emit_intent", "Read", "Bash")


class _StubRegistry:
    def __init__(self, actions: Iterable[_StubAction]):
        self._by_name = {a.name: a for a in actions}

    def get(self, name: str):
        return self._by_name.get(name)


def test_propose_action_rejected_when_registry_says_mode_unsupported():
    reg = _StubRegistry(
        [_StubAction("kernel_opt", (ExecutionMode.MARATHON_MULTI_AGENT,))]
    )
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT, action_registry=reg)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(IntentType.PROPOSE_ACTION, action_name="kernel_opt", predicted_gain_pct=3.0),
        )
    assert exc.value.rule == "mode"


def test_propose_action_unknown_action_with_registry_passes_in_v06():
    # PolicyGate doesn't reject unknown actions for *propose*; only delegate.
    reg = _StubRegistry([])
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT, action_registry=reg)
    gate.validate_intent(
        "executor",
        intent(
            IntentType.PROPOSE_ACTION,
            action_name="brand_new_action",
            predicted_gain_pct=2.0,
        ),
    )


def test_delegate_unknown_action_with_registry_rejected():
    reg = _StubRegistry([])
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT, action_registry=reg)
    with pytest.raises(PolicyDenied):
        gate.validate_intent(
            "executor",
            intent(IntentType.DELEGATE, action_name="never_heard_of"),
        )


def test_allowed_tools_for_action_uses_registry():
    reg = _StubRegistry(
        [_StubAction("kernel_opt", (ExecutionMode.MARATHON_MULTI_AGENT,),
                     allowed_tools=("emit_intent", "Read", "Bash", "Edit"))]
    )
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT, action_registry=reg)
    tools = gate.allowed_tools_for_action(
        ExecutionMode.MARATHON_MULTI_AGENT, "kernel_opt"
    )
    assert "Edit" in tools


# ---------------------------------------------------------------------------
# update_state gate
# ---------------------------------------------------------------------------
def test_update_state_blocks_core_field_for_executor():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(
                IntentType.UPDATE_STATE,
                changes={"current_best": "v2", "current_action": "x"},
            ),
        )
    assert exc.value.rule == "state_field"
    assert "current_best" in str(exc.value)


def test_update_state_blocks_core_field_for_watchdog():
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "watchdog",
            intent(IntentType.UPDATE_STATE, changes={"stop_reason": "x"}),
        )
    assert exc.value.rule == "state_field"


def test_update_state_passes_for_non_core_fields():
    gate = make_gate()
    gate.validate_intent(
        "executor",
        intent(
            IntentType.UPDATE_STATE,
            changes={"current_action": "kernel_opt", "current_tput": 1234.5},
        ),
    )


def test_update_state_rejects_empty_changes():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor", intent(IntentType.UPDATE_STATE, changes={})
        )
    assert exc.value.rule == "payload"


def test_update_state_codex_role_denied_by_role_rule():
    gate = make_gate()
    # Critic doesn't have UPDATE_STATE allowed at all.
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "critic",
            intent(IntentType.UPDATE_STATE, changes={"current_action": "x"}),
        )
    assert exc.value.rule == "role"


# ---------------------------------------------------------------------------
# send_message topic gate
# ---------------------------------------------------------------------------
def test_send_message_rejects_topic_outside_allowlist():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(IntentType.SEND_MESSAGE, topic="not-on-bus", body_md="x"),
        )
    assert exc.value.rule == "topic"


def test_send_message_accepts_known_topic():
    gate = make_gate()
    gate.validate_intent(
        "executor",
        intent(IntentType.SEND_MESSAGE, topic="event", body_md="hello"),
    )


# ---------------------------------------------------------------------------
# Bash allowlist / denylist
# ---------------------------------------------------------------------------
def test_quick_bash_denylist_always_fires():
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT)
    with pytest.raises(PolicyDenied) as exc:
        gate.check_bash_denylist("rm -rf /")
    assert exc.value.rule == "bash"


def test_quick_bash_allowlist_in_quick_mode():
    gate = make_gate(ExecutionMode.QUICK_PARAM_SWEEP)
    # Should pass (rocm-smi is on the allowlist).
    gate.validate_quick_bash("rocm-smi --showuse")
    # Should fail (random unknown command).
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_quick_bash("./scripts/random_thing.sh")
    assert exc.value.rule == "bash"


def test_quick_bash_outside_quick_is_noop_for_allowlist():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    gate.validate_quick_bash("./scripts/random_thing.sh")  # noop


# ---------------------------------------------------------------------------
# allowed_tools_for_agent
# ---------------------------------------------------------------------------
def test_allowed_tools_for_codex_role_is_empty():
    gate = make_gate()
    assert gate.allowed_tools_for_agent("critic") == []
    assert gate.allowed_tools_for_agent("sage") == []


def test_allowed_tools_for_claude_role_has_emit_intent():
    gate = make_gate()
    assert "emit_intent" in gate.allowed_tools_for_agent("executor")


def test_allowed_tools_for_unknown_agent_empty():
    gate = make_gate()
    assert gate.allowed_tools_for_agent("nobody") == []


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------
def test_core_state_fields_includes_current_best_and_stop_reason():
    assert "current_best" in CORE_STATE_FIELDS
    assert "stop_reason" in CORE_STATE_FIELDS


def test_default_quick_allowlist_non_empty():
    assert "bench_runner" in DEFAULT_QUICK_ACTION_ALLOWLIST
