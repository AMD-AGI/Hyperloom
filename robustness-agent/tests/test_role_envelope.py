# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for :mod:`robustness_agent.role.envelope`.

Covers:

* every builder produces an :class:`Intent` whose ``to_envelope_item``
  round-trips through ``build_envelope_dict``.
* invalid arguments to builders raise ``ValueError`` (so a bad caller
  blows up before reaching the Coordinator).
* the static tables (``PAYLOAD_REQUIRED`` / role allowlist /
  robustness-only set) keep the shape upstream PolicyGate expects.
"""

from __future__ import annotations

import json

import pytest

from robustness_agent.role.envelope import (
    ALERT_SEVERITIES,
    BackendTurnResult,
    IntentType,
    KILL_TASK_ALLOWED_SCOPES,
    PAYLOAD_REQUIRED,
    ROBUSTNESS_ALLOWED_INTENTS,
    ROBUSTNESS_DELEGATE_ACTIONS,
    ROBUSTNESS_ONLY_INTENTS,
    ROBUSTNESS_STATE_FIELDS,
    build_alert,
    build_delegate,
    build_envelope_dict,
    build_escalate,
    build_force_dispatch,
    build_heartbeat,
    build_kill_task,
    build_prune_branch,
    build_send_message,
    build_update_state,
)


# ---------------------------------------------------------------------------
# Builders — happy path
# ---------------------------------------------------------------------------

def test_heartbeat_builder_uses_known_topic_and_body():
    intent = build_heartbeat()
    assert intent.type is IntentType.SEND_MESSAGE
    assert intent.payload["topic"] == "heartbeat"
    assert intent.payload["body_md"]
    assert intent.to_envelope_item() == {
        "intent_type": "send_message",
        "payload": {"topic": "heartbeat", "body_md": "ok (robustness-agent)"},
    }


def test_send_message_includes_optional_fields():
    intent = build_send_message(
        "observation",
        body_md="hello",
        to="orchestration",
        extras={"k": "v"},
    )
    assert intent.payload == {
        "topic": "observation",
        "body_md": "hello",
        "to": "orchestration",
        "k": "v",
    }


def test_alert_builder_serialises_with_detail():
    intent = build_alert(
        "high",
        "session pod failed",
        detail={"pod": "brain-0", "phase": "Failed"},
    )
    assert intent.type is IntentType.ALERT
    item = intent.to_envelope_item()
    assert item["payload"]["severity"] == "high"
    assert item["payload"]["summary"] == "session pod failed"
    assert item["payload"]["detail"] == {"pod": "brain-0", "phase": "Failed"}


def test_escalate_builder_carries_hint_and_severity():
    intent = build_escalate(
        "repeated_failure",
        "switch to fp16 baseline",
        severity="high",
    )
    assert intent.type is IntentType.ESCALATE_STRATEGY_CHANGE
    payload = intent.payload
    assert payload["reason"] == "repeated_failure"
    assert payload["next_action_hint"] == "switch to fp16 baseline"
    assert payload["severity"] == "high"


def test_kill_task_builder_pins_scope_to_task():
    intent = build_kill_task("task-123", "stuck for 10min")
    assert intent.type is IntentType.KILL_TASK
    assert intent.payload == {
        "task_id": "task-123",
        "reason": "stuck for 10min",
        "scope": "task",
    }


def test_force_dispatch_builder():
    intent = build_force_dispatch("task-456", "high-value blocked")
    assert intent.type is IntentType.FORCE_DISPATCH
    assert intent.payload == {"task_id": "task-456", "reason": "high-value blocked"}


def test_prune_branch_builder():
    intent = build_prune_branch("backends.sglang", "3 consecutive failures")
    assert intent.type is IntentType.PRUNE_BRANCH
    assert intent.payload == {"family": "backends.sglang", "reason": "3 consecutive failures"}


def test_delegate_builder_passes_params_and_idempotency():
    intent = build_delegate(
        "recover",
        params={"checkpoint": "latest"},
        idempotency_key="recover-1",
    )
    assert intent.type is IntentType.DELEGATE
    assert intent.payload == {
        "action_name": "recover",
        "params": {"checkpoint": "latest"},
        "idempotency_key": "recover-1",
    }


def test_update_state_builder_only_allows_robustness_fields():
    intent = build_update_state({"crash_count": 3, "current_action": "recover"})
    assert intent.type is IntentType.UPDATE_STATE
    assert intent.payload["changes"] == {"crash_count": 3, "current_action": "recover"}


# ---------------------------------------------------------------------------
# Builders — defensive errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "severity",
    ["", "warn", "critical", "INFO", None],
)
def test_alert_rejects_unknown_severity(severity):
    with pytest.raises(ValueError):
        build_alert(severity, "x")  # type: ignore[arg-type]


def test_alert_rejects_empty_summary():
    with pytest.raises(ValueError):
        build_alert("medium", "")


def test_escalate_requires_reason_and_hint():
    with pytest.raises(ValueError):
        build_escalate("", "hint")
    with pytest.raises(ValueError):
        build_escalate("reason", "")


@pytest.mark.parametrize(
    "task_id,reason",
    [("", "r"), ("t", ""), ("", "")],
)
def test_kill_task_requires_both_fields(task_id, reason):
    with pytest.raises(ValueError):
        build_kill_task(task_id, reason)


def test_delegate_rejects_kernel_owned_action():
    for kernel_owned in ("kernel_opt", "integrate", "deep_kernel_analysis"):
        with pytest.raises(ValueError):
            build_delegate(kernel_owned)


def test_delegate_accepts_only_handle_actions():
    for action in ROBUSTNESS_DELEGATE_ACTIONS:
        intent = build_delegate(action)
        assert intent.payload["action_name"] == action


def test_delegate_allowlist_includes_report_wind_down():
    """``report`` is the wind-down lever for ``deadline_imminent`` /
    ``recover_unsuccessful``. If this asserts trips check whether you
    are intentionally narrowing the allowlist; the ladder relies on
    ``build_delegate('report')`` not raising.
    """
    assert "report" in ROBUSTNESS_DELEGATE_ACTIONS
    intent = build_delegate(
        "report",
        params={"reason": "deadline_imminent"},
        idempotency_key="report-deadline-imminent-tick-7",
    )
    assert intent.payload == {
        "action_name": "report",
        "params": {"reason": "deadline_imminent"},
        "idempotency_key": "report-deadline-imminent-tick-7",
    }


def test_update_state_rejects_core_fields():
    with pytest.raises(ValueError):
        build_update_state({"current_best": {"tput": 1.0}})


def test_update_state_rejects_unknown_fields():
    with pytest.raises(ValueError):
        build_update_state({"random_field": 1})


def test_update_state_rejects_empty_changes():
    with pytest.raises(ValueError):
        build_update_state({})


# ---------------------------------------------------------------------------
# Envelope serialisation
# ---------------------------------------------------------------------------

def test_envelope_dict_is_json_serialisable():
    intents = [
        build_heartbeat(),
        build_alert("medium", "stall detected"),
        build_escalate("crash_count_high", "trigger recover"),
    ]
    env = build_envelope_dict(intents)
    payload = json.dumps(env)
    restored = json.loads(payload)
    assert restored == env
    assert [item["intent_type"] for item in restored["intents"]] == [
        "send_message",
        "alert",
        "escalate_strategy_change",
    ]


def test_intent_to_envelope_item_makes_a_copy_of_payload():
    intent = build_alert("low", "noise", detail={"k": 1})
    item = intent.to_envelope_item()
    item["payload"]["k"] = 2
    assert intent.payload["detail"]["k"] == 1


def test_backend_turn_result_default_fields():
    res = BackendTurnResult()
    assert res.intents == []
    assert res.raw_text == ""
    assert res.metadata == {}


# ---------------------------------------------------------------------------
# Static tables — invariants
# ---------------------------------------------------------------------------

def test_payload_required_covers_every_intent_type():
    for intent_type in IntentType:
        assert intent_type in PAYLOAD_REQUIRED, intent_type


def test_robustness_only_subset_of_allowed():
    assert ROBUSTNESS_ONLY_INTENTS.issubset(ROBUSTNESS_ALLOWED_INTENTS)


def test_robustness_only_set_matches_design_v06():
    assert ROBUSTNESS_ONLY_INTENTS == frozenset({
        IntentType.KILL_TASK,
        IntentType.FORCE_DISPATCH,
        IntentType.PRUNE_BRANCH,
        IntentType.ESCALATE_STRATEGY_CHANGE,
    })


def test_kill_task_scope_locked_to_task():
    assert KILL_TASK_ALLOWED_SCOPES == frozenset({"task"})


def test_alert_severities_are_low_medium_high():
    assert ALERT_SEVERITIES == frozenset({"low", "medium", "high"})


def test_robustness_state_fields_are_minimal():
    assert ROBUSTNESS_STATE_FIELDS == frozenset({"crash_count", "current_action"})
