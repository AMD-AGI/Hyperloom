"""Unit tests for the ``same_payload_loop`` signal (B1)."""

from __future__ import annotations

from hyperloom.agents.robustness.role.prompt_inputs import (
    InboxItem,
    ReactorContext,
    SharedStateSnapshot,
)
from hyperloom.agents.robustness.signals import SymptomSeverity
from hyperloom.agents.robustness.signals.repeated_payload import (
    RepeatedPayloadConfig,
    evaluate_repeated_payload_signals,
)
from hyperloom.agents.robustness.sources.base import SourceData


def _ctx() -> ReactorContext:
    return ReactorContext(
        tick_index=1,
        shared_state=SharedStateSnapshot(session_id="sess-1"),
        inbox=[],
        now_unix=1.0,
    )


def _validate_stack_event(
    *,
    task_id: str,
    state: str = "failed",
    stack: list | None = None,
    error_class: str = "RuntimeError",
) -> dict:
    return {
        "topic": "delegated_result",
        "agent": "coordinator",
        "payload": {
            "kind": "validate_stack",
            "family": "validate_stack",
            "state": state,
            "task_id": task_id,
            "error_class": error_class,
            "params": {
                "optimization_stack": stack or [
                    {"action": "backends", "variant_name": "fp8"}
                ],
                "config_path": "/tmp/baseline_config.yaml",
            },
            # Different idempotency_key each attempt — the smoking gun.
            "idempotency_key": f"validate-stack-tick-{task_id}",
        },
    }


def test_no_streak_silent_below_threshold():
    events = [
        _validate_stack_event(task_id="t1"),
        _validate_stack_event(task_id="t2"),
    ]
    data = SourceData(coordinator_events=events)
    out = evaluate_repeated_payload_signals(
        _ctx(), data, config=RepeatedPayloadConfig(streak_threshold=3),
    )
    assert out == []


def test_three_identical_payloads_fires_high():
    events = [
        _validate_stack_event(task_id="t1"),
        _validate_stack_event(task_id="t2"),
        _validate_stack_event(task_id="t3"),
    ]
    data = SourceData(coordinator_events=events)
    out = evaluate_repeated_payload_signals(
        _ctx(), data, config=RepeatedPayloadConfig(streak_threshold=3),
    )
    assert len(out) == 1
    sym = out[0]
    assert sym.name == "same_payload_loop"
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["family"] == "validate_stack"
    assert sym.evidence["count"] == 3
    assert sym.evidence["last_task_id"] == "t3"
    assert sym.evidence["top_error_class"] == "RuntimeError"


def test_different_idempotency_key_does_not_break_dedup():
    """The 2026-05-18 GPU-leak failure mode: distinct keys, same payload."""
    events = [
        _validate_stack_event(task_id=str(i))
        for i in range(5)
    ]
    data = SourceData(coordinator_events=events)
    out = evaluate_repeated_payload_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "same_payload_loop")
    assert sym.evidence["count"] == 5


def test_payload_change_resets_streak():
    events = [
        _validate_stack_event(task_id="t1", stack=[{"action": "a", "variant_name": "v1"}]),
        _validate_stack_event(task_id="t2", stack=[{"action": "a", "variant_name": "v1"}]),
        _validate_stack_event(task_id="t3", stack=[{"action": "b", "variant_name": "v2"}]),
    ]
    data = SourceData(coordinator_events=events)
    out = evaluate_repeated_payload_signals(
        _ctx(), data, config=RepeatedPayloadConfig(streak_threshold=3),
    )
    # After the t3 payload change the streak resets to length 1.
    assert all(s.name != "same_payload_loop" for s in out)


def test_success_resets_streak():
    events = [
        _validate_stack_event(task_id="t1"),
        _validate_stack_event(task_id="t2"),
        _validate_stack_event(task_id="t3", state="succeeded"),
        _validate_stack_event(task_id="t4"),
    ]
    data = SourceData(coordinator_events=events)
    out = evaluate_repeated_payload_signals(
        _ctx(), data, config=RepeatedPayloadConfig(streak_threshold=3),
    )
    assert all(s.name != "same_payload_loop" for s in out)


def test_baseline_family_uses_dedicated_projection():
    """``baseline`` payload fingerprint matches the upstream Coordinator's."""
    base_payload = {
        "kind": "baseline",
        "family": "baseline",
        "state": "failed",
        "task_id": "b{}",
        "error_class": "BaselineFailed",
        "params": {
            "benchmark_script": "sglang_mi300x.sh",
            "result_dir": "/workspace/hyperloom",
            "extra_sglang_args": ["--tp", "8"],
            "extra_envs": {"CONC": "8"},
            "model_path": "/wekafs/models/dsr1",
            "gpu_type": "mi300x",
            "config_path": "/tmp/baseline_config.yaml",
            "disable_run_eval": False,
        },
        "idempotency_key": "baseline-tick-X",
    }
    events = []
    for i in range(3):
        payload = {**base_payload, "task_id": f"b{i}",
                   "idempotency_key": f"baseline-tick-{i}"}
        events.append({"topic": "delegated_result", "agent": "coordinator", "payload": payload})
    data = SourceData(coordinator_events=events)
    out = evaluate_repeated_payload_signals(
        _ctx(), data, config=RepeatedPayloadConfig(streak_threshold=3),
    )
    sym = next(s for s in out if s.name == "same_payload_loop")
    assert sym.evidence["family"] == "baseline"


def test_other_topics_ignored():
    events = [
        {"topic": "heartbeat", "agent": "robustness", "payload": {}},
        {"topic": "alert", "agent": "robustness", "payload": {"severity": "high"}},
    ]
    data = SourceData(coordinator_events=events)
    out = evaluate_repeated_payload_signals(_ctx(), data)
    assert out == []


def test_inbox_and_coordinator_events_combined():
    inbox = [
        InboxItem(
            seq=1,
            msg_id="m1",
            from_agent="coordinator",
            topic="delegated_result",
            payload={
                "kind": "validate_stack",
                "family": "validate_stack",
                "state": "failed",
                "task_id": "i1",
                "error_class": "RuntimeError",
                "params": {
                    "optimization_stack": [{"action": "x", "variant_name": "v"}],
                    "config_path": "/tmp/c",
                },
                "idempotency_key": "k1",
            },
        ),
    ]
    coord_events = [
        _validate_stack_event(
            task_id="c1",
            stack=[{"action": "x", "variant_name": "v"}],
        ),
        _validate_stack_event(
            task_id="c2",
            stack=[{"action": "x", "variant_name": "v"}],
        ),
    ]
    ctx = ReactorContext(
        tick_index=1,
        shared_state=SharedStateSnapshot(session_id="sess-1"),
        inbox=inbox,
        now_unix=1.0,
    )
    # config_path differs so inbox + coord won't merge into one hash;
    # match them up:
    inbox[0].payload["params"]["config_path"] = (
        coord_events[0]["payload"]["params"]["config_path"]
    )
    data = SourceData(coordinator_events=coord_events)
    out = evaluate_repeated_payload_signals(
        ctx, data, config=RepeatedPayloadConfig(streak_threshold=3),
    )
    sym = next(s for s in out if s.name == "same_payload_loop")
    assert sym.evidence["count"] == 3
