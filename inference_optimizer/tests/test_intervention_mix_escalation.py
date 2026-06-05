"""Intervention-mix ledger telemetry.

``SharedState.record_intervention`` maintains a config-vs-code_patch
ledger + ``consecutive_config_only_rounds`` counter.
``to_intervention_mix_summary`` renders the ledger as neutral counts for
the Orchestration per-tick prompt — no directive. These tests pin the
counter bookkeeping and the telemetry rendering.
"""

from __future__ import annotations

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.task_registry import Task


def test_empty_ledger_renders_nothing():
    """No interventions recorded yet → empty string (no early-tick noise,
    nothing to escalate)."""
    s = SharedState(session_id="im-empty")
    assert s.to_intervention_mix_summary() == ""


def test_single_config_keep_shows_counts_without_escalation():
    s = SharedState(session_id="im-one")
    s.record_intervention(change_type="config", action="explore", task_id="t1")
    out = s.to_intervention_mix_summary()
    assert "config_keeps=1" in out
    assert "code_patch_keeps=0" in out
    assert "consecutive_config_only_rounds=1" in out
    assert "ESCALATION" not in out


def test_two_consecutive_config_only_rounds_render_counter():
    """consecutive_config_only_rounds is surfaced as a neutral count; no
    directive is emitted."""
    s = SharedState(session_id="im-escalate")
    s.record_intervention(change_type="config", action="explore", task_id="t1")
    s.record_intervention(change_type="config", action="explore", task_id="t2")
    out = s.to_intervention_mix_summary()
    assert "consecutive_config_only_rounds=2" in out
    assert "ESCALATION" not in out


def test_code_patch_keep_resets_counter():
    """A code_patch KEEP resets consecutive_config_only_rounds."""
    s = SharedState(session_id="im-reset")
    s.record_intervention(change_type="config", action="explore", task_id="t1")
    s.record_intervention(change_type="config", action="explore", task_id="t2")
    s.record_intervention(
        change_type="code_patch", action="integrate_patch", task_id="t3",
    )
    out = s.to_intervention_mix_summary()
    assert "config_keeps=2" in out
    assert "code_patch_keeps=1" in out
    assert "consecutive_config_only_rounds=0" in out
    assert "ESCALATION" not in out


def test_code_patch_attempt_records_in_ledger_without_keep():
    s = SharedState(session_id="im-attempt")
    s.record_intervention(
        change_type="code_patch_attempt",
        action="integrate_patch",
        task_id="t-revert",
    )
    out = s.to_intervention_mix_summary()
    assert "code_patch_keeps=0" in out
    assert "code_patch_attempts=1" in out


def test_reverted_integrate_patch_records_attempt_not_keep():
    coord = object.__new__(Coordinator)
    coord.shared_state = SharedState(session_id="im-coord-attempt")
    task = Task(
        task_id="t-integrate-revert",
        kind="integrate_patch",
        state="succeeded",
        params={},
        idempotency_key="t-integrate-revert",
    )

    coord._record_intervention_for_task(task, {
        "status": "reverted",
        "delta_pct": -1.0,
    })

    mix = coord.shared_state.get_intervention_mix()
    assert mix["total_code_patch"] == 0
    assert mix["total_code_patch_attempt"] == 1


def test_config_heavy_zero_patch_renders_counts_without_directive():
    """Config-heavy ledger (>=5 config keeps, 0 code_patch) renders the
    neutral counts; no directive is emitted."""
    s = SharedState(session_id="im-heavy")
    for i in range(5):
        s.record_intervention(
            change_type="config", action="explore", task_id=f"t{i}",
        )
    out = s.to_intervention_mix_summary()
    assert "config_keeps=5" in out
    assert "code_patch_keeps=0" in out
    assert "ESCALATION" not in out
