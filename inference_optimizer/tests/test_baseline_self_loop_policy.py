"""PolicyGate ``baseline_self_loop`` denial tests.

Coordinator denies a ``baseline`` propose_action / delegate when the
last two baseline attempts both failed with the same params fingerprint
AND the new proposal carries that same fingerprint. The denial fires
through ``_sequence_denial_for_action``, lands on the bus as a
``policy_denied`` observation, and stops the dispatcher from
instantiating a third doomed task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
)
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.orchestrator.task_registry import Task
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import target_baseline_json


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
    """v0.8 §3.9 — scoreboard retired. No-op back-compat shim."""
    return None


def _seed_target_analysis_marker(sd: Path) -> None:
    """Satisfy the unconditional ``target_analysis`` gate for tests that
    target a downstream rule (self-loop / baseline) and must reach it.
    """
    path = target_baseline_json(sd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "skipped",
                    "reason": "no_target_gpu_configured"}),
        encoding="utf-8",
    )


def _record_failed_attempt(c: Coordinator, params: dict[str, Any]) -> None:
    """Drive a single failed baseline through ``_handle_unpromotable_result``.

    This is the same path the dispatcher uses post-task, so the
    fingerprint extras land exactly where the self-loop helper reads.
    """
    task = Task(
        task_id=f"t-fail-{len(c.shared_state.baseline_attempts)}",
        kind="baseline",
        state="queued",
        params=params,
        idempotency_key=f"idem-{len(c.shared_state.baseline_attempts)}",
    )
    import asyncio
    asyncio.get_event_loop().run_until_complete(
        c._handle_unpromotable_result(
            task,
            {"status": "failed", "error_class": "no_report",
             "error": "benchmark_report.json missing"},
        )
    )


# ---------------------------------------------------------------------------
# Self-loop denial direct helper
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_baseline_self_loop_denies_third_attempt_with_same_fingerprint(
    session_dir,
):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        params = {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        for i in range(2):
            await c._handle_unpromotable_result(
                Task(
                    task_id=f"t-{i}", kind="baseline", state="queued",
                    params=params, idempotency_key=f"idem-{i}",
                ),
                {"status": "failed", "error_class": "no_report",
                 "error": "missing"},
            )
        denial = c._baseline_self_loop_denial(params)
        assert denial is not None
        assert denial.rule == "baseline_self_loop"
        assert "benchmark_script" in (denial.hint or "")
        assert "result_dir" in (denial.hint or "")
        assert "no_report" in (denial.hint or "")
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_baseline_self_loop_allows_changed_fingerprint(session_dir):
    """A proposal with a DIFFERENT fingerprint is not denied."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        bad = {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        for i in range(2):
            await c._handle_unpromotable_result(
                Task(
                    task_id=f"t-{i}", kind="baseline", state="queued",
                    params=bad, idempotency_key=f"idem-{i}",
                ),
                {"status": "failed", "error_class": "no_report",
                 "error": "missing"},
            )
        # Same action, but new override → denial does NOT fire.
        assert c._baseline_self_loop_denial(
            {"benchmark_script": "sglang_mi300x.sh"}
        ) is None
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_baseline_self_loop_quiet_after_success(session_dir):
    """A succeeded attempt between failures breaks the streak."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        params = {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        task = Task(
            task_id="t-fail", kind="baseline", state="queued",
            params=params, idempotency_key="idem-fail",
        )
        await c._handle_unpromotable_result(
            task,
            {"status": "failed", "error_class": "no_report",
             "error": "missing"},
        )
        # Now a success lands.
        ok_task = Task(
            task_id="t-ok", kind="baseline", state="queued",
            params=params, idempotency_key="idem-ok",
        )
        await c._promote_to_shared_state(
            "baseline",
            {"output_throughput": 1500.0,
             "materialized_config": "/tmp/m.yaml"},
            task=ok_task,
        )
        # Followed by another failure.
        task2 = Task(
            task_id="t-fail2", kind="baseline", state="queued",
            params=params, idempotency_key="idem-fail2",
        )
        await c._handle_unpromotable_result(
            task2,
            {"status": "failed", "error_class": "no_report",
             "error": "missing"},
        )
        # Only ONE failure in the consecutive tail → no denial.
        assert c._baseline_self_loop_denial(params) is None
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_baseline_self_loop_short_circuits_under_threshold(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        # Only one failure recorded — threshold is 2.
        await c._handle_unpromotable_result(
            Task(
                task_id="t-once", kind="baseline", state="queued",
                params={"benchmark_script": "dsr1_fp8_mi300x.sh"},
                idempotency_key="idem-once",
            ),
            {"status": "failed", "error_class": "no_report",
             "error": "missing"},
        )
        denial = c._baseline_self_loop_denial(
            {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        )
        assert denial is None
    finally:
        await c.stop()


# ---------------------------------------------------------------------------
# _sequence_denial_for_action wires baseline_self_loop in
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sequence_denial_returns_self_loop_for_baseline(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _seed_target_analysis_marker(session_dir)
    _mute_action_scoring(c)
    try:
        # Make baseline_tput > 0 so the "baseline must run first" rule
        # doesn't pre-empt us — we want to reach the self-loop branch.
        c.shared_state.baseline_tput = 1500.0
        params = {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        for i in range(2):
            await c._handle_unpromotable_result(
                Task(
                    task_id=f"t-{i}", kind="baseline", state="queued",
                    params=params, idempotency_key=f"idem-{i}",
                ),
                {"status": "failed", "error_class": "no_report",
                 "error": "missing"},
            )
        denial = c._sequence_denial_for_action(
            "baseline", proposed_params=params,
        )
        assert denial is not None
        assert denial.rule == "baseline_self_loop"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_sequence_denial_no_self_loop_for_other_actions(session_dir):
    """``backends`` (and other non-baseline kinds) skip the self-loop guard.

    We pre-satisfy the unrelated ``profile``/``trace_analyze`` prerequisite
    gates so the only remaining denial source is the self-loop helper —
    which is baseline-only, so a backends proposal must pass cleanly.
    """
    c = Coordinator(session_dir, backends=_silent_backends())
    _seed_target_analysis_marker(session_dir)
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 1500.0
        c.shared_state.last_profile_trace = "/tmp/fake-trace"
        c.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/fake-trace",
            # Roofline-v2 N3: backends now requires fresh analysis_md_text.
            "analysis_md_text": "FAKE_REPORT",
        }
        # Pretend the user tried a backends proposal with the same params
        # as a doomed baseline — self-loop rule must NOT fire on
        # backends, because the fingerprint surface is baseline-only today.
        denial = c._sequence_denial_for_action(
            "backends",
            proposed_params={"benchmark_script": "dsr1_fp8_mi300x.sh"},
        )
        assert denial is None
    finally:
        await c.stop()
