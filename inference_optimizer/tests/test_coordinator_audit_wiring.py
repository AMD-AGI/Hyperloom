"""End-to-end Coordinator wiring tests for the audit trail.

These exercise ``Coordinator._promote_to_shared_state`` (success path)
and ``Coordinator._handle_unpromotable_result`` (failure path) for the
six audit-trail kinds, asserting that:

* ``last_<action>`` snapshot fields are refreshed.
* ``<action>_attempts`` history grows by one entry per attempt.
* ``last_action_failures`` is populated for every failure (not just
  baseline — kernel-owned actions land here too).
* The existing ``baseline_failure_streak`` / ``stop_reason`` legacy
  behaviour is preserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.task_registry import Task
from inference_optimizer.paths import make_session_dir


def _heartbeat() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _mute_action_scoring(coordinator: Coordinator) -> None:
    """Clear the seeded ``action_scores`` to short-circuit the scoring
    helpers. These tests focus on audit-trail wiring, not the scoring
    matrix; bypassing it avoids tripping pre-existing
    ``_score_action_no_promote`` paths that aren't part of this plan.
    """
    coordinator.shared_state.action_scores = {}


def _mk_task(kind: str, task_id: str = "t-aud-1") -> Task:
    return Task(
        task_id=task_id,
        kind=kind,
        state="queued",
        params={},
        idempotency_key=f"idem-{task_id}",
    )


# ---------------------------------------------------------------------------
# Success path: _promote_to_shared_state records succeeded attempt
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_baseline_records_success_attempt(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_task("baseline", "t-base-1")
        result = {
            "output_throughput": 1500.0,
            "accuracy": 0.81,
            "materialized_config": "/tmp/baseline.with_envs.yaml",
            "workspace": "/runs/baseline/t-base-1",
        }
        await c._promote_to_shared_state("baseline", result, task=task)
        last = c.shared_state.last_baseline
        assert last
        assert last["status"] == "succeeded"
        assert last["decision"] == "promoted"
        assert last["key_metric"] == pytest.approx(1500.0)
        assert last["key_metric_kind"] == "output_throughput"
        assert last["extras"]["accuracy"] == 0.81
        assert len(c.shared_state.baseline_attempts) == 1
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_profile_records_success_attempt(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_task("profile", "t-prof-1")
        result = {
            "main_trace_path": "/tmp/trace.json",
            "output_throughput": 1234.5,
            "workspace": "/runs/profile/t-prof-1",
        }
        await c._promote_to_shared_state("profile", result, task=task)
        last = c.shared_state.last_profile
        assert last["status"] == "succeeded"
        assert last["decision"] == "promoted"
        assert last["extras"]["trace_path"] == "/tmp/trace.json"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_backends_records_success_attempt(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.current_best = {"action": "baseline", "tput": 800.0}
        task = _mk_task("backends", "t-be-1")
        result = {
            "winners": [{"name": "v1"}],
            "best_variant": {
                "name": "v1",
                "extra_sglang_args": "--foo",
                "extra_envs": {"K": "1"},
            },
            "output_throughput": 900.0,
            "best_gain_pct": 12.5,
            "base_tput": 800.0,
        }
        await c._promote_to_shared_state("backends", result, task=task)
        last = c.shared_state.last_backends
        assert last["status"] == "succeeded"
        # 900/800 = +12.5% — well above the 0.1% promote threshold.
        assert last["decision"] == "promoted"
        assert last["extras"]["best_variant_name"] == "v1"
        assert last["extras"]["candidate_extra_sglang_args"] == "--foo"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_sweep_records_discarded_attempt(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_task("sweep", "t-sw-1")
        result = {
            "grid_size": 4,
            "best_overall": {"name": "c8_isl1k_osl1k", "tput": 1200.0},
            "pareto_front": [{"name": "a"}, {"name": "b"}],
        }
        await c._promote_to_shared_state("sweep", result, task=task)
        last = c.shared_state.last_sweep_attempt = c.shared_state.last_sweep
        # Sweep never promotes a current_best so its decision is "discarded".
        assert c.shared_state.sweep_attempts[-1]["status"] == "succeeded"
        assert c.shared_state.sweep_attempts[-1]["decision"] == "discarded"
        assert c.shared_state.sweep_attempts[-1]["extras"]["grid_size"] == 4
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_validate_stack_records_success_attempt(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 1000.0
        c.shared_state.optimization_stack = [
            {"action": "backends", "variant_name": "v1"}
        ]
        task = _mk_task("validate_stack", "t-vs-1")
        result = {
            "output_throughput": 1100.0,
            "validated_stack_len": 1,
        }
        await c._promote_to_shared_state("validate_stack", result, task=task)
        last = c.shared_state.last_validate_stack
        assert last["status"] == "succeeded"
        assert last["decision"] == "promoted"
        assert last["extras"]["gain_pct"] == pytest.approx(10.0)
        assert last["extras"]["validated_stack_len"] == 1
    finally:
        await c.stop()


# ---------------------------------------------------------------------------
# Failure path: _handle_unpromotable_result records both attempt + failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_handle_unpromotable_baseline_records_failure(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_task("baseline", "t-fail-1")
        result = {
            "status": "failed",
            "error_class": "no_report",
            "error": "benchmark_report.json missing under runs/...",
            "workspace": "/runs/baseline/t-fail-1/benchmark_sglang_xyz",
            "reported_success": False,
        }
        await c._handle_unpromotable_result(task, result)
        # Per-action audit
        assert len(c.shared_state.baseline_attempts) == 1
        attempt = c.shared_state.baseline_attempts[-1]
        assert attempt["status"] == "failed"
        assert attempt["decision"] == "no_promote"
        assert attempt["error_class"] == "no_report"
        # Global rolling log
        assert len(c.shared_state.last_action_failures) == 1
        fail = c.shared_state.last_action_failures[-1]
        assert fail["action"] == "baseline"
        assert fail["error_class"] == "no_report"
        # Legacy baseline streak still increments
        assert c.shared_state.baseline_failure_streak == 1
        assert c.shared_state.stop_reason in ("", None)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_baseline_third_failure_sets_stop_reason(
    session_dir,
):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        for i in range(3):
            await c._handle_unpromotable_result(
                _mk_task("baseline", f"t-{i}"),
                {"status": "failed", "error_class": "no_report",
                 "error": "missing"},
            )
        assert c.shared_state.baseline_failure_streak == 3
        assert c.shared_state.stop_reason == "baseline_failed"
        assert len(c.shared_state.last_action_failures) == 3
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_records_for_non_baseline_kinds(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        # backends failure — should hit both per-action attempts and the
        # global failure log, but should NOT touch baseline_failure_streak.
        await c._handle_unpromotable_result(
            _mk_task("backends", "t-be-fail"),
            {"status": "failed", "error_class": "subprocess_nonzero",
             "error": "rc=1\nstderr blob"},
        )
        assert c.shared_state.baseline_failure_streak == 0
        assert c.shared_state.stop_reason in ("", None)
        assert len(c.shared_state.backends_attempts) == 1
        assert c.shared_state.backends_attempts[-1]["status"] == "failed"
        assert len(c.shared_state.last_action_failures) == 1
        fail = c.shared_state.last_action_failures[-1]
        assert fail["action"] == "backends"
        assert fail["stderr_tail"] is not None  # subprocess class triggers tail
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_kernel_action_records_global_only(
    session_dir,
):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        # Kernel-owned action: NOT in _AUDIT_ACTIONS, but failure should
        # still land in last_action_failures (global log is intentionally
        # comprehensive).
        await c._handle_unpromotable_result(
            _mk_task("kernel_opt", "t-ko-fail"),
            {"status": "failed", "error_class": "timeout",
             "error": "wall-clock exceeded"},
        )
        # No per-action audit attempt (kernel-owned actions stay outside
        # the kernel-parity audit set; they have bespoke recorders).
        assert not hasattr(c.shared_state, "kernel_opt_attempts_audit")
        # But the global log IS populated.
        assert len(c.shared_state.last_action_failures) == 1
        entry = c.shared_state.last_action_failures[-1]
        assert entry["action"] == "kernel_opt"
        assert entry["error_class"] == "timeout"
        assert entry["stderr_tail"] is not None
    finally:
        await c.stop()
