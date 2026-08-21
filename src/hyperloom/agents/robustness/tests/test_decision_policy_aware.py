# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :class:`hyperloom.agents.robustness.decision.PolicyAware` (mirrors upstream PolicyGate rules)."""

from __future__ import annotations

import pytest

from hyperloom.agents.robustness.decision import PolicyAware, PolicyViolation
from hyperloom.agents.robustness.role.envelope import (
    Intent,
    IntentType,
    build_alert,
    build_delegate,
    build_escalate,
    build_heartbeat,
    build_prune_branch,
    build_send_message,
    build_update_state,
)


@pytest.fixture()
def policy() -> PolicyAware:
    return PolicyAware()


# Happy path — every builder output should be emit-safe


def test_heartbeat_passes(policy: PolicyAware):
    policy.assert_payload_complete(build_heartbeat())


def test_send_message_observation_passes(policy: PolicyAware):
    policy.assert_payload_complete(build_send_message("observation", body_md="x"))


@pytest.mark.parametrize("severity", ["low", "medium", "high"])
def test_alert_each_severity_passes(severity, policy: PolicyAware):
    policy.assert_payload_complete(build_alert(severity, "summary"))


def test_escalate_passes(policy: PolicyAware):
    policy.assert_payload_complete(build_escalate("r", "next"))


def test_prune_branch_passes(policy: PolicyAware):
    policy.assert_payload_complete(build_prune_branch("backends", "fail x3"))


def test_delegate_passes(policy: PolicyAware):
    policy.assert_payload_complete(build_delegate("recover"))


def test_delegate_recover_gpu_leak_payload_passes(policy: PolicyAware):
    """Pin the PolicyAware contract for the delegate(recover) payload against schema drift."""
    intent = build_delegate(
        action_name="recover",
        params={
            "reason": "gpu_memory_leaked",
            "force_gpu_cleanup": True,
            "evidence": {
                "consecutive_hits": 2,
                "per_gpu": [{"gpu_id": 0, "free_mb": 108.0}],
            },
        },
        idempotency_key="recover-gpu-leak-tick-7",
    )
    policy.assert_payload_complete(intent)
    assert intent.payload["params"]["force_gpu_cleanup"] is True
    assert intent.payload["idempotency_key"] == "recover-gpu-leak-tick-7"


def test_update_state_passes(policy: PolicyAware):
    policy.assert_payload_complete(build_update_state({"crash_count": 1}))


# Role gate


@pytest.mark.parametrize(
    "intent_type",
    [
        IntentType.PROPOSE_ACTION,
        IntentType.REQUEST,
        IntentType.RESPONSE,
        IntentType.REVIEW_VERDICT,
    ],
)
def test_role_blocks_disallowed_intents(intent_type, policy: PolicyAware):
    intent = Intent(type=intent_type, payload={})
    with pytest.raises(PolicyViolation) as excinfo:
        policy.assert_payload_complete(intent)
    assert excinfo.value.rule == "role"


# Required-field gate


def test_alert_missing_severity_is_payload_error(policy: PolicyAware):
    intent = Intent(type=IntentType.ALERT, payload={"summary": "x"})
    with pytest.raises(PolicyViolation) as excinfo:
        policy.assert_payload_complete(intent)
    assert excinfo.value.rule == "payload"


def test_alert_missing_summary_is_payload_error(policy: PolicyAware):
    intent = Intent(type=IntentType.ALERT, payload={"severity": "low"})
    with pytest.raises(PolicyViolation) as excinfo:
        policy.assert_payload_complete(intent)
    assert excinfo.value.rule == "payload"


def test_alert_unknown_severity_is_payload_error(policy: PolicyAware):
    intent = Intent(
        type=IntentType.ALERT,
        payload={"severity": "warn", "summary": "x"},
    )
    with pytest.raises(PolicyViolation) as excinfo:
        policy.assert_payload_complete(intent)
    assert excinfo.value.rule == "payload"


def test_prune_branch_required_fields(policy: PolicyAware):
    intent = Intent(type=IntentType.PRUNE_BRANCH, payload={"family": "f"})
    with pytest.raises(PolicyViolation):
        policy.assert_payload_complete(intent)


def test_escalate_missing_hint_is_payload_error(policy: PolicyAware):
    intent = Intent(
        type=IntentType.ESCALATE_STRATEGY_CHANGE,
        payload={"reason": "r"},
    )
    with pytest.raises(PolicyViolation) as excinfo:
        policy.assert_payload_complete(intent)
    assert excinfo.value.rule == "payload"


def test_escalate_invalid_severity_is_payload_error(policy: PolicyAware):
    intent = Intent(
        type=IntentType.ESCALATE_STRATEGY_CHANGE,
        payload={
            "reason": "r",
            "next_action_hint": "h",
            "severity": "extreme",
        },
    )
    with pytest.raises(PolicyViolation):
        policy.assert_payload_complete(intent)


def test_delegate_kernel_owned_action_is_blocked(policy: PolicyAware):
    intent = Intent(type=IntentType.DELEGATE, payload={"action_name": "kernel_opt"})
    with pytest.raises(PolicyViolation) as excinfo:
        policy.assert_payload_complete(intent)
    assert excinfo.value.rule == "delegate_action"


def test_update_state_core_field_blocked_with_state_field_rule(policy: PolicyAware):
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"current_best": {"tput": 1.0}}},
    )
    with pytest.raises(PolicyViolation) as excinfo:
        policy.assert_payload_complete(intent)
    assert excinfo.value.rule == "state_field"


def test_update_state_unknown_field_blocked(policy: PolicyAware):
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"random": 1}},
    )
    with pytest.raises(PolicyViolation) as excinfo:
        policy.assert_payload_complete(intent)
    assert excinfo.value.rule == "state_field"


def test_update_state_empty_changes_blocked(policy: PolicyAware):
    intent = Intent(type=IntentType.UPDATE_STATE, payload={"changes": {}})
    with pytest.raises(PolicyViolation) as excinfo:
        policy.assert_payload_complete(intent)
    assert excinfo.value.rule == "payload"


def test_send_message_missing_topic(policy: PolicyAware):
    intent = Intent(type=IntentType.SEND_MESSAGE, payload={})
    with pytest.raises(PolicyViolation) as excinfo:
        policy.assert_payload_complete(intent)
    assert excinfo.value.rule == "payload"
