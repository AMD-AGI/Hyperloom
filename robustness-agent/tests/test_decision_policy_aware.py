"""Unit tests for :class:`robustness_agent.decision.PolicyAware`.

The validator mirrors upstream PolicyGate.  These tests pin the
behaviour the reactor relies on:

* every robustness-allowed intent built via the helpers passes.
* missing / mistyped payload fields raise :class:`PolicyViolation` with
  the rule label upstream PolicyGate uses.
* validate_all collects every violation rather than short-circuiting.
"""

from __future__ import annotations

import pytest

from robustness_agent.decision import PolicyAware, PolicyViolation
from robustness_agent.role.envelope import (
    Intent,
    IntentType,
    build_alert,
    build_delegate,
    build_escalate,
    build_force_dispatch,
    build_heartbeat,
    build_kill_task,
    build_prune_branch,
    build_send_message,
    build_update_state,
)


@pytest.fixture()
def policy() -> PolicyAware:
    return PolicyAware()


# ---------------------------------------------------------------------------
# Happy path — every builder output should be emit-safe
# ---------------------------------------------------------------------------

def test_heartbeat_passes(policy: PolicyAware):
    policy.assert_payload_complete(build_heartbeat())


def test_send_message_observation_passes(policy: PolicyAware):
    policy.assert_payload_complete(build_send_message("observation", body_md="x"))


@pytest.mark.parametrize("severity", ["low", "medium", "high"])
def test_alert_each_severity_passes(severity, policy: PolicyAware):
    policy.assert_payload_complete(build_alert(severity, "summary"))


def test_escalate_passes(policy: PolicyAware):
    policy.assert_payload_complete(build_escalate("r", "next"))


def test_kill_task_passes(policy: PolicyAware):
    policy.assert_payload_complete(build_kill_task("t1", "stuck"))


def test_force_dispatch_passes(policy: PolicyAware):
    policy.assert_payload_complete(build_force_dispatch("t1", "blocked"))


def test_prune_branch_passes(policy: PolicyAware):
    policy.assert_payload_complete(build_prune_branch("backends", "fail x3"))


def test_delegate_passes(policy: PolicyAware):
    for action in ("accuracy_gate", "recover", "server_lifecycle"):
        policy.assert_payload_complete(build_delegate(action))


def test_delegate_recover_gpu_leak_payload_passes(policy: PolicyAware):
    """``gpu_memory_leaked`` -> delegate(recover) payload regression.

    The ActionLadder branch in :mod:`decision.action_ladder` builds this
    exact shape; we pin the PolicyAware contract so future schema
    changes are caught at build time rather than mid-tick.
    """
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


# ---------------------------------------------------------------------------
# Role gate
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Required-field gate
# ---------------------------------------------------------------------------

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


def test_kill_task_missing_task_id(policy: PolicyAware):
    intent = Intent(type=IntentType.KILL_TASK, payload={"reason": "x"})
    with pytest.raises(PolicyViolation) as excinfo:
        policy.assert_payload_complete(intent)
    assert excinfo.value.rule == "payload"


def test_kill_task_bad_scope_uses_kill_scope_rule(policy: PolicyAware):
    intent = Intent(
        type=IntentType.KILL_TASK,
        payload={"task_id": "t", "reason": "r", "scope": "process"},
    )
    with pytest.raises(PolicyViolation) as excinfo:
        policy.assert_payload_complete(intent)
    assert excinfo.value.rule == "kill_scope"


def test_force_dispatch_required_fields(policy: PolicyAware):
    intent = Intent(type=IntentType.FORCE_DISPATCH, payload={"task_id": ""})
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


# ---------------------------------------------------------------------------
# validate_all collects every violation
# ---------------------------------------------------------------------------

def test_validate_all_collects_multiple_violations(policy: PolicyAware):
    bad = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"current_best": 1, "random": 2}},
    )
    violations = policy.validate_all(bad)
    assert violations
    rules = [v.rule for v in violations]
    assert "state_field" in rules


def test_validate_all_returns_empty_on_valid_intent(policy: PolicyAware):
    assert policy.validate_all(build_heartbeat()) == []
