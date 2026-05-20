"""PolicyGate ``validate_stack`` skip-after-N-failures tests.

Coordinator drops the otherwise-mandatory ``validate_stack`` gate after
``_VALIDATE_STACK_FAIL_THRESHOLD`` consecutive failed validate_stack
attempts, so a doomed environment doesn't burn LLM calls every tick on
the same retry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    _VALIDATE_STACK_FAIL_THRESHOLD,
)
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
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


def _seed_target_analysis_marker(sd: Path) -> None:
    """Satisfy the unconditional ``target_analysis`` gate.

    Tests targeting the validate_stack gate must reach the validate_stack
    branch in ``_sequence_denial_for_action``; without this marker the
    earlier ``target_analysis must run first`` rule short-circuits.
    """
    path = target_baseline_json(sd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "skipped",
                    "reason": "no_target_gpu_configured"}),
        encoding="utf-8",
    )


def _bypass_upstream_gates(c: Coordinator) -> None:
    """Pre-seed every prerequisite the validate_stack gate sits behind.

    The gate is the *last* rule in ``_sequence_denial_for_action``: any
    earlier rule (baseline / profile / select_kernels / integrate) would
    short-circuit our integration test before the validate_stack branch
    is reached. We satisfy them all so the assertion isolates the
    behaviour under test.
    """
    c.shared_state.baseline_tput = 1500.0
    c.shared_state.last_profile_trace = "/tmp/fake-trace"
    c.shared_state.last_profile_pmc_summary = "/tmp/fake-pmc.json"
    c.shared_state.last_select_kernels = {"trace_input": "/tmp/fake-trace"}
    c.shared_state.action_scores = {}


def _seed_unvalidated_keep(c: Coordinator) -> None:
    """Make ``optimization_stack_has_unvalidated_keeps()`` return True.

    Coordinator only enforces the validate_stack gate while there is at
    least one KEEP on the stack that has not been validated end-to-end.
    """
    c.shared_state.optimization_stack = [
        {
            "action": "params",
            "variant_name": "v0",
            "extra_sglang_args": "--foo 1",
            "extra_envs": {},
            "tput": 1500.0,
        }
    ]
    c.shared_state.cumulative_gain_validated_stack_len = 0


async def _record_validate_stack_failure(c: Coordinator, idx: int) -> None:
    """Drive one failed validate_stack through ``_handle_unpromotable_result``.

    This is the same path the dispatcher uses post-task, so the failure
    lands on ``validate_stack_attempts`` exactly the way the gate-skip
    helper reads it.
    """
    task = Task(
        task_id=f"vs-fail-{idx}",
        kind="validate_stack",
        state="queued",
        params={},
        idempotency_key=f"idem-vs-{idx}",
    )
    await c._handle_unpromotable_result(
        task,
        {
            "status": "failed",
            "error_class": "tp_mismatch",
            "error": "TP=8 requested but only 4 GPUs visible",
        },
    )


# ---------------------------------------------------------------------------
# _validate_stack_gate_skipped() direct helper
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gate_skipped_after_threshold_failures(session_dir, caplog):
    """N consecutive failures → skip returns True and warns exactly once."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        for i in range(_VALIDATE_STACK_FAIL_THRESHOLD):
            await _record_validate_stack_failure(c, i)
        with caplog.at_level("WARNING"):
            assert c._validate_stack_gate_skipped() is True
            assert c._validate_stack_gate_skipped() is True
        warns = [r for r in caplog.records if "gate SKIPPED" in r.getMessage()]
        assert len(warns) == 1
        assert "tp_mismatch" in warns[0].getMessage()
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_gate_held_under_threshold(session_dir):
    """N-1 failures must NOT trip the skip — gate stays mandatory."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        for i in range(_VALIDATE_STACK_FAIL_THRESHOLD - 1):
            await _record_validate_stack_failure(c, i)
        assert c._validate_stack_gate_skipped() is False
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_gate_skip_resets_after_success(session_dir):
    """A successful validate_stack between failures resets the streak."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        for i in range(_VALIDATE_STACK_FAIL_THRESHOLD):
            await _record_validate_stack_failure(c, i)
        assert c._validate_stack_gate_skipped() is True
        c.shared_state.validate_stack_attempts.append(
            {
                "ts": "2026-01-01T00:00:00",
                "task_id": "vs-ok",
                "status": "succeeded",
                "decision": "promoted",
                "error_class": None,
                "error_excerpt": None,
                "extras": {},
            }
        )
        assert c._validate_stack_gate_skipped() is False
        assert c._validate_stack_gate_skip_warned is False
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_gate_denies_other_actions_under_threshold(session_dir):
    """Below threshold the gate still denies non-validate_stack actions."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _seed_target_analysis_marker(session_dir)
    try:
        _bypass_upstream_gates(c)
        _seed_unvalidated_keep(c)
        for i in range(_VALIDATE_STACK_FAIL_THRESHOLD - 1):
            await _record_validate_stack_failure(c, i)
        denial = c._sequence_denial_for_action("backends")
        assert denial is not None
        assert denial.rule == "validate_stack_required"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_gate_allows_other_actions_at_threshold(session_dir):
    """At threshold the gate is dropped — `backends` (and friends) pass."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _seed_target_analysis_marker(session_dir)
    try:
        _bypass_upstream_gates(c)
        _seed_unvalidated_keep(c)
        for i in range(_VALIDATE_STACK_FAIL_THRESHOLD):
            await _record_validate_stack_failure(c, i)
        assert c._sequence_denial_for_action("backends") is None
        assert c._sequence_denial_for_action("params") is None
        assert c._sequence_denial_for_action("report") is None
        assert c._sequence_denial_for_action("validate_stack") is None
    finally:
        await c.stop()
