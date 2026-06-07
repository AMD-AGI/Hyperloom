# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the ``same_payload_loop`` signal (B1)."""

from __future__ import annotations

from robustness_agent.role.prompt_inputs import (
    InboxItem,
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import SymptomSeverity
from robustness_agent.signals.repeated_payload import (
    RepeatedPayloadConfig,
    evaluate_repeated_payload_signals,
)
from robustness_agent.sources.base import SourceData


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
            "extra_server_args": ["--tp", "8"],
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


# ---------------------------------------------------------------------------
# Legacy ``extra_sglang_args`` envelopes still fingerprint to the same
# hash as canonical envelopes.
# ---------------------------------------------------------------------------
def _integrate_event(*, task_id: str, args_value: str, legacy: bool) -> dict:
    """Build an ``integrate`` family envelope. When ``legacy=True`` the
    payload carries the legacy ``extra_sglang_args`` key; otherwise the
    canonical ``extra_server_args`` key. Both rows are otherwise
    identical so the projection should fingerprint them to the same
    hash.
    """
    key = "extra_sglang_args" if legacy else "extra_server_args"
    return {
        "topic": "delegated_result",
        "agent": "coordinator",
        "payload": {
            "kind": "integrate",
            "family": "integrate",
            "state": "failed",
            "task_id": task_id,
            "error_class": "ApplyFailed",
            "params": {
                "kernel_id": "k1",
                "patch_path": "/tmp/p1.patch",
                key: args_value,
                "extra_envs": {"CONC": "8"},
            },
            "idempotency_key": f"integrate-{task_id}",
        },
    }


def test_legacy_extra_sglang_args_envelope_hashes_identically(recwarn):
    """G3 regression: a legacy ``extra_sglang_args`` envelope must
    fingerprint to the same hash as a canonical ``extra_server_args``
    envelope so the same_payload_loop signal still fires across a
    mixed-key burst (e.g. an in-flight rollout where some workers
    still emit the legacy key).
    """
    events = [
        _integrate_event(task_id="t1", args_value="--tp 4", legacy=True),
        _integrate_event(task_id="t2", args_value="--tp 4", legacy=False),
        _integrate_event(task_id="t3", args_value="--tp 4", legacy=True),
    ]
    data = SourceData(coordinator_events=events)
    out = evaluate_repeated_payload_signals(
        _ctx(), data, config=RepeatedPayloadConfig(streak_threshold=3),
    )
    sym = next(s for s in out if s.name == "same_payload_loop")
    assert sym.evidence["family"] == "integrate"
    assert sym.evidence["count"] == 3, (
        "legacy + canonical envelopes did not collapse to the same hash"
    )
    # At least one DeprecationWarning fired (one per legacy envelope
    # touched by ``_hash_for`` — exact count is implementation-detail
    # but >= 1 must hold for the audit channel).
    legacy_warnings = [
        w for w in recwarn.list
        if issubclass(w.category, DeprecationWarning)
        and "extra_sglang_args" in str(w.message)
    ]
    assert legacy_warnings, "no DeprecationWarning fired on legacy envelope"


def test_legacy_envelope_alone_still_fingerprints():
    """If every envelope in a streak uses the legacy key, the signal
    must still fire — the shim must normalise the key before the
    projection, not just on mixed-key bursts.
    """
    events = [
        _integrate_event(task_id="t1", args_value="--tp 4", legacy=True),
        _integrate_event(task_id="t2", args_value="--tp 4", legacy=True),
        _integrate_event(task_id="t3", args_value="--tp 4", legacy=True),
    ]
    data = SourceData(coordinator_events=events)
    out = evaluate_repeated_payload_signals(
        _ctx(), data, config=RepeatedPayloadConfig(streak_threshold=3),
    )
    sym = next(s for s in out if s.name == "same_payload_loop")
    assert sym.evidence["count"] == 3
