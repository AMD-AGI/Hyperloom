# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for signal rules + classifier."""

from __future__ import annotations

import pytest

from robustness_agent.role.prompt_inputs import (
    InboxItem,
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import (
    Classifier,
    Symptom,
    SymptomSeverity,
    evaluate_crash_signals,
    evaluate_event_signals,
    evaluate_health_signals,
    evaluate_stall_signals,
)
from robustness_agent.signals.crash import CrashConfig
from robustness_agent.signals.event import EventConfig
from robustness_agent.signals.health import HealthConfig
from robustness_agent.signals.stall import StallConfig
from robustness_agent.sources.base import SourceData


def _ctx(
    *,
    crash_count: int = 0,
    current_action: str = "",
    inbox: list[InboxItem] | None = None,
    now_unix: float = 1_700_000_000.0,
) -> ReactorContext:
    return ReactorContext(
        tick_index=1,
        shared_state=SharedStateSnapshot(
            session_id="sess-1",
            crash_count=crash_count,
            current_action=current_action,
        ),
        inbox=list(inbox or []),
        now_unix=now_unix,
    )


# ---------------------------------------------------------------------------
# Stall
# ---------------------------------------------------------------------------

def test_stall_emits_medium_when_agent_silent_past_threshold():
    now = 1_000.0
    ctx = _ctx(now_unix=now)
    data = SourceData(
        coordinator_events=[
            {"agent": "kernel", "topic": "heartbeat", "ts": now - 600.0},
            {"agent": "orchestration", "topic": "heartbeat", "ts": now - 100.0},
        ],
    )
    out = evaluate_stall_signals(ctx, data, config=StallConfig(stall_timeout_s=300.0, severity_high_after_s=900.0))
    assert len(out) == 1
    assert out[0].name == "agent_stall"
    assert out[0].severity is SymptomSeverity.MEDIUM
    assert out[0].evidence["agent"] == "kernel"


def test_stall_escalates_to_high_after_long_silence():
    now = 1_000.0
    ctx = _ctx(now_unix=now)
    data = SourceData(
        coordinator_events=[{"agent": "kernel", "ts": now - 1500.0}],
    )
    out = evaluate_stall_signals(ctx, data, config=StallConfig(stall_timeout_s=300.0, severity_high_after_s=900.0))
    assert out and out[0].severity is SymptomSeverity.HIGH


def test_stall_silent_when_no_data_present():
    ctx = _ctx()
    data = SourceData()
    assert evaluate_stall_signals(ctx, data) == []


def test_stall_uses_iso_timestamps_too():
    now = 1_700_000_500.0
    ctx = _ctx(now_unix=now)
    data = SourceData(
        coordinator_events=[
            {"agent": "kernel", "ts": "2023-11-14T22:13:20+00:00"},  # ~1700000000
        ],
    )
    out = evaluate_stall_signals(ctx, data, config=StallConfig(stall_timeout_s=300.0))
    assert out and out[0].evidence["agent"] == "kernel"


# ---------------------------------------------------------------------------
# Crash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "crash_count,expected_name,expected_severity",
    [
        (0, None, None),
        (1, None, None),
        (2, "crash_count_rising", SymptomSeverity.MEDIUM),
        (5, "crash_count_high", SymptomSeverity.HIGH),
        (10, "crash_count_emergency", SymptomSeverity.HIGH),
    ],
)
def test_crash_signal_thresholds(crash_count, expected_name, expected_severity):
    ctx = _ctx(crash_count=crash_count, current_action="recover")
    out = evaluate_crash_signals(ctx, SourceData(), config=CrashConfig())
    if expected_name is None:
        assert out == []
    else:
        assert len(out) == 1
        assert out[0].name == expected_name
        assert out[0].severity is expected_severity


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

def test_repeated_policy_denied_emits_medium_alert():
    inbox = [
        InboxItem(
            seq=i,
            msg_id=f"m{i}",
            from_agent="orchestration",
            topic="observation",
            payload={"kind": "policy_denied", "rule": "kernel_owned_by_kernel_agent"},
        )
        for i in range(3)
    ]
    ctx = _ctx(inbox=inbox)
    out = evaluate_event_signals(ctx, SourceData(), config=EventConfig(policy_denied_threshold=3))
    assert any(s.name == "repeated_policy_denied" for s in out)
    sym = next(s for s in out if s.name == "repeated_policy_denied")
    assert sym.evidence["count"] == 3
    assert sym.evidence["top_rule"] == "kernel_owned_by_kernel_agent"


def test_policy_denied_below_threshold_is_silent():
    inbox = [
        InboxItem(
            seq=i,
            msg_id=f"m{i}",
            from_agent="orchestration",
            topic="observation",
            payload={"kind": "policy_denied"},
        )
        for i in range(2)
    ]
    ctx = _ctx(inbox=inbox)
    out = evaluate_event_signals(ctx, SourceData(), config=EventConfig(policy_denied_threshold=3))
    assert all(s.name != "repeated_policy_denied" for s in out)


def test_repeated_failure_groups_by_family():
    coord_events = [
        {
            "topic": "delegated_result",
            "agent": "coordinator",
            "payload": {"state": "failed", "kind": "kernel_opt", "task_id": "t1"},
        },
        {
            "topic": "delegated_result",
            "agent": "coordinator",
            "payload": {"state": "failed", "kind": "kernel_opt", "task_id": "t2"},
        },
        {
            "topic": "delegated_result",
            "agent": "coordinator",
            "payload": {"state": "succeeded", "kind": "kernel_opt", "task_id": "t3"},
        },
    ]
    data = SourceData(coordinator_events=coord_events)
    out = evaluate_event_signals(_ctx(), data, config=EventConfig(delegated_failure_threshold=2))
    assert any(s.name == "repeated_failure" for s in out)
    sym = next(s for s in out if s.name == "repeated_failure")
    assert sym.evidence["family"] == "kernel_opt"
    assert sym.evidence["count"] == 2


def test_recover_unsuccessful_fires_on_needs_review_with_gpu_unhealthy_error():
    coord_events = [
        {
            "topic": "delegated_result",
            "agent": "coordinator",
            "payload": {
                "task_id": "tsk-9",
                "kind": "recover",
                "state": "needs_review",
                "error_class": "gpu_unhealthy_after_gpureset",
                "force_gpu_cleanup": True,
                "gpureset_attempted": True,
                "post_free_mb_per_gpu": [{"gpu_id": 0, "free_mb": 12.0}],
            },
        },
    ]
    data = SourceData(coordinator_events=coord_events)
    out = evaluate_event_signals(_ctx(), data)
    sym = next((s for s in out if s.name == "recover_unsuccessful"), None)
    assert sym is not None
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["error_class"] == "gpu_unhealthy_after_gpureset"
    assert sym.evidence["task_id"] == "tsk-9"
    assert sym.evidence["post_free_mb_per_gpu"][0]["free_mb"] == 12.0


def test_recover_unsuccessful_silent_when_recover_succeeded():
    coord_events = [
        {
            "topic": "delegated_result",
            "agent": "coordinator",
            "payload": {
                "task_id": "tsk-10",
                "kind": "recover",
                "state": "succeeded",
                "force_gpu_cleanup": True,
                "gpureset_attempted": False,
            },
        },
    ]
    data = SourceData(coordinator_events=coord_events)
    out = evaluate_event_signals(_ctx(), data)
    assert all(s.name != "recover_unsuccessful" for s in out)


def test_recover_unsuccessful_uses_latest_result_when_multiple_recovers():
    # Old recover succeeded; newer recover failed → fire on the newer.
    coord_events = [
        {
            "topic": "delegated_result",
            "agent": "coordinator",
            "payload": {
                "task_id": "tsk-old",
                "kind": "recover",
                "state": "succeeded",
                "force_gpu_cleanup": True,
                "gpureset_attempted": False,
            },
        },
        {
            "topic": "delegated_result",
            "agent": "coordinator",
            "payload": {
                "task_id": "tsk-new",
                "kind": "recover",
                "state": "needs_review",
                "error_class": "gpu_unhealthy_after_soft_cleanup",
                "force_gpu_cleanup": True,
                "gpureset_attempted": False,
            },
        },
    ]
    data = SourceData(coordinator_events=coord_events)
    out = evaluate_event_signals(_ctx(), data)
    sym = next((s for s in out if s.name == "recover_unsuccessful"), None)
    assert sym is not None
    assert sym.evidence["task_id"] == "tsk-new"


def test_idempotency_replay_fires_when_same_payload_distinct_keys():
    """B4: distinct idempotency_keys + same payload hash → MEDIUM alert."""
    inbox = [
        InboxItem(
            seq=1, msg_id="m1", from_agent="orchestration",
            topic="proposal",
            payload={
                "action_name": "validate_stack",
                "params": {"optimization_stack": [{"a": "v"}]},
                "idempotency_key": "k-alpha",
            },
        ),
        InboxItem(
            seq=2, msg_id="m2", from_agent="orchestration",
            topic="proposal",
            payload={
                "action_name": "validate_stack",
                "params": {"optimization_stack": [{"a": "v"}]},
                "idempotency_key": "k-beta",
            },
        ),
    ]
    ctx = _ctx(inbox=inbox)
    out = evaluate_event_signals(
        ctx, SourceData(),
        config=EventConfig(idempotency_replay_threshold=2),
    )
    sym = next(s for s in out if s.name == "idempotency_replay")
    assert sym.severity is SymptomSeverity.MEDIUM
    assert sym.evidence["action_name"] == "validate_stack"
    assert set(sym.evidence["distinct_keys"]) == {"k-alpha", "k-beta"}


def test_idempotency_replay_silent_when_payloads_differ():
    inbox = [
        InboxItem(
            seq=1, msg_id="m1", from_agent="orchestration",
            topic="proposal",
            payload={
                "action_name": "validate_stack",
                "params": {"optimization_stack": [{"a": "v1"}]},
                "idempotency_key": "k-alpha",
            },
        ),
        InboxItem(
            seq=2, msg_id="m2", from_agent="orchestration",
            topic="proposal",
            payload={
                "action_name": "validate_stack",
                "params": {"optimization_stack": [{"a": "v2"}]},
                "idempotency_key": "k-beta",
            },
        ),
    ]
    ctx = _ctx(inbox=inbox)
    out = evaluate_event_signals(ctx, SourceData())
    assert all(s.name != "idempotency_replay" for s in out)


def test_idempotency_replay_silent_when_no_key():
    """No idempotency_key at all → not a key-bypass attempt, skip."""
    inbox = [
        InboxItem(
            seq=1, msg_id="m1", from_agent="orchestration",
            topic="proposal",
            payload={
                "action_name": "validate_stack",
                "params": {"x": 1},
            },
        ),
        InboxItem(
            seq=2, msg_id="m2", from_agent="orchestration",
            topic="proposal",
            payload={
                "action_name": "validate_stack",
                "params": {"x": 1},
            },
        ),
    ]
    ctx = _ctx(inbox=inbox)
    out = evaluate_event_signals(ctx, SourceData())
    assert all(s.name != "idempotency_replay" for s in out)


def test_recover_unsuccessful_detected_via_signature_when_kind_missing():
    # ``kind`` tag elided — still recognised as recover via the
    # ``force_gpu_cleanup`` + ``gpureset_attempted`` signature.
    coord_events = [
        {
            "topic": "delegated_result",
            "agent": "coordinator",
            "payload": {
                "task_id": "tsk-11",
                "state": "needs_review",
                "error_class": "gpu_unhealthy_after_soft_cleanup",
                "force_gpu_cleanup": True,
                "gpureset_attempted": False,
            },
        },
    ]
    data = SourceData(coordinator_events=coord_events)
    out = evaluate_event_signals(_ctx(), data)
    assert any(s.name == "recover_unsuccessful" for s in out)


def test_event_handles_combined_inbox_and_coordinator_events():
    inbox = [
        InboxItem(
            seq=1,
            msg_id="m1",
            from_agent="orchestration",
            topic="observation",
            payload={"kind": "policy_denied", "rule": "role"},
        ),
    ]
    coord_events = [
        {
            "topic": "observation",
            "agent": "orchestration",
            "payload": {"kind": "policy_denied", "rule": "role"},
        },
        {
            "topic": "observation",
            "agent": "orchestration",
            "payload": {"kind": "policy_denied", "rule": "payload"},
        },
    ]
    data = SourceData(coordinator_events=coord_events)
    ctx = _ctx(inbox=inbox)
    out = evaluate_event_signals(ctx, data, config=EventConfig(policy_denied_threshold=3))
    assert any(s.name == "repeated_policy_denied" for s in out)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_flags_failed_pod_as_high_severity():
    data = SourceData(
        session_pods=[
            {
                "pod": {"namespace": "ns", "name": "brain-0"},
                "role": "brain",
                "phase": "Failed",
                "assignment_id": "a1",
            },
        ],
    )
    out = evaluate_health_signals(_ctx(), data)
    assert out and out[0].severity is SymptomSeverity.HIGH
    assert out[0].name == "pod_not_running"


def test_health_flags_unknown_phase_as_medium():
    data = SourceData(
        session_pods=[
            {
                "pod": {"namespace": "ns", "name": "hands-0"},
                "role": "hands",
                "phase": "CrashLoopBackOff",
            },
        ],
    )
    out = evaluate_health_signals(_ctx(), data)
    assert out and out[0].severity is SymptomSeverity.MEDIUM


def test_health_silent_when_pods_running():
    data = SourceData(
        session_pods=[{"pod": {"namespace": "ns", "name": "p"}, "phase": "Running"}],
    )
    assert evaluate_health_signals(_ctx(), data) == []


def test_health_warns_no_metrics_after_threshold():
    now = 10_000.0
    data = SourceData(
        session_summary={
            "pods": [
                {
                    "pod": {"namespace": "ns", "name": "p1"},
                    "role": "hands",
                    "t_start": now - 1200.0,
                    "available_metrics": [],
                }
            ]
        }
    )
    out = evaluate_health_signals(_ctx(now_unix=now), data, config=HealthConfig(no_metrics_warn_s=600))
    assert any(s.name == "pod_no_metrics" for s in out)


def test_health_does_not_flag_recently_started_pods_with_no_metrics():
    now = 10_000.0
    data = SourceData(
        session_summary={
            "pods": [
                {
                    "pod": {"namespace": "ns", "name": "p2"},
                    "role": "hands",
                    "t_start": now - 60.0,
                    "available_metrics": [],
                }
            ]
        }
    )
    assert evaluate_health_signals(_ctx(now_unix=now), data, config=HealthConfig(no_metrics_warn_s=600)) == []


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def test_classifier_dedupes_same_subject_keeps_higher_severity():
    classifier = Classifier()

    def custom_a(ctx, data):
        return [Symptom(name="x", severity=SymptomSeverity.MEDIUM, summary="m", subject={"a": "1"})]

    def custom_b(ctx, data):
        return [Symptom(name="x", severity=SymptomSeverity.HIGH, summary="h", subject={"a": "1"})]

    classifier.extra_evaluators = [custom_a, custom_b]
    out = classifier.classify(SourceData(), _ctx())
    assert len(out) == 1
    assert out[0].severity is SymptomSeverity.HIGH


def test_classifier_orders_by_severity_descending():
    classifier = Classifier()

    def fn(ctx, data):
        return [
            Symptom(name="lo", severity=SymptomSeverity.LOW, summary="lo", subject={"k": "1"}),
            Symptom(name="hi", severity=SymptomSeverity.HIGH, summary="hi", subject={"k": "2"}),
            Symptom(name="md", severity=SymptomSeverity.MEDIUM, summary="md", subject={"k": "3"}),
        ]

    classifier.extra_evaluators = [fn]
    out = classifier.classify(SourceData(), _ctx())
    severities = [s.severity for s in out]
    assert severities == [SymptomSeverity.HIGH, SymptomSeverity.MEDIUM, SymptomSeverity.LOW]


def test_classifier_runs_all_default_rules():
    inbox = [
        InboxItem(
            seq=i,
            msg_id=f"m{i}",
            from_agent="orchestration",
            topic="observation",
            payload={"kind": "policy_denied", "rule": "role"},
        )
        for i in range(3)
    ]
    ctx = _ctx(crash_count=2, inbox=inbox)
    data = SourceData(
        session_pods=[{"pod": {"namespace": "ns", "name": "p"}, "phase": "Failed"}],
        coordinator_events=[],
    )
    classifier = Classifier(crash_config=CrashConfig(medium_threshold=2))
    out = classifier.classify(data, ctx)
    names = {s.name for s in out}
    assert "crash_count_rising" in names
    assert "repeated_policy_denied" in names
    assert "pod_not_running" in names
