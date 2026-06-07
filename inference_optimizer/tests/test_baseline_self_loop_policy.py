# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PolicyGate ``baseline_no_param_change`` denial tests.

After a baseline failure the agent must retry with **identical** params.
Changing any knob (extra_server_args, benchmark_script, etc.) is denied
by ``_baseline_self_loop_denial`` with ``rule='baseline_no_param_change'``.
Same-fingerprint retries are allowed; the ``baseline_failure_streak``
counter terminates the run at 3 with ``stop_reason='baseline_failed'``.
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


async def _record_failed_baseline(
    c: Coordinator,
    params: dict[str, Any],
    error_class: str = "no_report",
) -> None:
    """Drive a single failed baseline through ``_handle_unpromotable_result``."""
    idx = len(c.shared_state.baseline_attempts or [])
    task = Task(
        task_id=f"t-fail-{idx}",
        kind="baseline",
        state="queued",
        params=params,
        idempotency_key=f"idem-{idx}",
    )
    await c._handle_unpromotable_result(
        task,
        {"status": "failed", "error_class": error_class,
         "error": "benchmark_report.json missing"},
    )


# ---------------------------------------------------------------------------
# Core: deny param changes after a baseline failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_denies_changed_params_after_single_failure(session_dir):
    """After ONE failure, changing any param is denied."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        original = {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        await _record_failed_baseline(c, original)

        changed = {"benchmark_script": "sglang_mi300x.sh"}
        denial = c._baseline_self_loop_denial(changed)
        assert denial is not None
        assert denial.rule == "baseline_no_param_change"
        assert "self-heal is disabled" in (denial.hint or "")
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_denies_extra_server_args_tweak_after_failure(session_dir):
    """Tweaking extra_server_args is the specific scenario we want to block."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        original = {"benchmark_script": "dsr1_fp8_mi300x.sh",
                    "extra_server_args": ""}
        await _record_failed_baseline(c, original)

        tweaked = {"benchmark_script": "dsr1_fp8_mi300x.sh",
                   "extra_server_args": "--max-num-seqs 128"}
        denial = c._baseline_self_loop_denial(tweaked)
        assert denial is not None
        assert denial.rule == "baseline_no_param_change"
    finally:
        await c.stop()


# ---------------------------------------------------------------------------
# Core: allow same-params retry (streak counter handles exit)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_allows_same_params_retry_after_failure(session_dir):
    """Same fingerprint → allowed (the streak counter handles termination)."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        params = {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        await _record_failed_baseline(c, params)
        assert c._baseline_self_loop_denial(params) is None
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_allows_same_params_retry_after_two_failures(session_dir):
    """Same params after two failures → still allowed (streak=2, one more to go)."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        params = {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        await _record_failed_baseline(c, params)
        await _record_failed_baseline(c, params)
        assert c._baseline_self_loop_denial(params) is None
    finally:
        await c.stop()


# ---------------------------------------------------------------------------
# Streak counter: 3 failures → stop_reason='baseline_failed'
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_three_failures_set_stop_reason(session_dir):
    """Three consecutive baseline failures set stop_reason='baseline_failed'."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        params = {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        for _ in range(3):
            await _record_failed_baseline(c, params)
        assert c.shared_state.stop_reason == "baseline_failed"
        assert c.shared_state.baseline_failure_streak == 3
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_two_failures_do_not_set_stop_reason(session_dir):
    """Two failures bump streak but do not terminate yet."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        params = {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        await _record_failed_baseline(c, params)
        await _record_failed_baseline(c, params)
        assert c.shared_state.baseline_failure_streak == 2
        assert c.shared_state.stop_reason != "baseline_failed"
    finally:
        await c.stop()


# ---------------------------------------------------------------------------
# Success resets the gate
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_success_resets_denial_gate(session_dir):
    """A succeeded baseline between failures resets the denial gate."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        params = {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        await _record_failed_baseline(c, params)

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

        # After success, changing params on next failure is fine.
        different = {"benchmark_script": "sglang_mi300x.sh"}
        assert c._baseline_self_loop_denial(different) is None
    finally:
        await c.stop()


# ---------------------------------------------------------------------------
# No prior failures → no denial
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_failures_no_denial(session_dir):
    """Fresh session with no baseline attempts → no denial."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        assert c._baseline_self_loop_denial(
            {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        ) is None
    finally:
        await c.stop()


# ---------------------------------------------------------------------------
# _sequence_denial_for_action integration
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sequence_denial_returns_no_param_change_for_baseline(
    session_dir,
):
    """_sequence_denial_for_action surfaces the no-param-change denial."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _seed_target_analysis_marker(session_dir)
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 1500.0
        original = {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        await _record_failed_baseline(c, original)

        denial = c._sequence_denial_for_action(
            "baseline",
            proposed_params={"benchmark_script": "sglang_mi300x.sh"},
        )
        assert denial is not None
        assert denial.rule == "baseline_no_param_change"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_sequence_denial_no_param_change_for_other_actions(session_dir):
    """Non-baseline actions are NOT affected by the baseline param freeze."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _seed_target_analysis_marker(session_dir)
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 1500.0
        c.shared_state.last_profile_trace = "/tmp/fake-trace"
        c.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/fake-trace",
            "analysis_md_text": "FAKE_REPORT",
        }
        denial = c._sequence_denial_for_action(
            "backends",
            proposed_params={"benchmark_script": "dsr1_fp8_mi300x.sh"},
        )
        assert denial is None
    finally:
        await c.stop()


# ---------------------------------------------------------------------------
# Audit trail carries error detail
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_failure_attempt_carries_error_excerpt(session_dir):
    """Each failed baseline attempt stores error_class and error_excerpt."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        params = {"benchmark_script": "dsr1_fp8_mi300x.sh"}
        await _record_failed_baseline(c, params, error_class="subprocess_nonzero")

        attempts = list(c.shared_state.baseline_attempts or [])
        assert len(attempts) >= 1
        last = attempts[-1]
        assert last["error_class"] == "subprocess_nonzero"
        assert last["error_excerpt"] is not None
        assert "benchmark_report.json missing" in last["error_excerpt"]
    finally:
        await c.stop()
