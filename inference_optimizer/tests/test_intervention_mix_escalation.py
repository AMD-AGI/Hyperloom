"""PR-A8 / D3 (Arbor-into-Hyperloom): intervention-mix escalation.

``SharedState.record_intervention`` maintains a config-vs-code_patch
ledger + ``consecutive_config_only_rounds`` counter. D3 wires the
CONSUMER: ``to_intervention_mix_summary`` renders the ledger for the
Orchestration per-tick prompt and emits an ESCALATION directive when the
session has been config-only for too long (Arbor's "do not settle for
config-only" rule). These tests pin the consumer contract.
"""

from __future__ import annotations

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.phase_state import depth_gate
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


def test_two_consecutive_config_only_rounds_escalate():
    """consecutive_config_only_rounds >= 2 trips the ESCALATION directive
    pointing at a code-patch serving_specialist."""
    s = SharedState(session_id="im-escalate")
    s.record_intervention(change_type="config", action="explore", task_id="t1")
    s.record_intervention(change_type="config", action="explore", task_id="t2")
    out = s.to_intervention_mix_summary()
    assert "consecutive_config_only_rounds=2" in out
    assert "ESCALATION" in out
    assert "serving_specialist" in out
    assert "integrate_patch" in out


def test_code_patch_keep_resets_counter_and_clears_escalation():
    """A code_patch KEEP resets consecutive_config_only_rounds → no
    escalation (Arbor: the counter resets when a code patch lands)."""
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


def test_code_patch_attempt_counts_for_depth_without_keep():
    s = SharedState(session_id="im-attempt")
    s.depth_tracker["consecutive_reverts"] = 3
    s.record_intervention(
        change_type="code_patch_attempt",
        action="integrate_patch",
        task_id="t-revert",
    )
    out = s.to_intervention_mix_summary()
    assert "code_patch_keeps=0" in out
    assert "code_patch_attempts=1" in out
    assert s.depth_tracker["code_patches_attempted"] == 1

    ok, blockers, _ = depth_gate(
        s.to_dict(),
        scout_runs_min=0,
        prs_fetched_min=0,
        pr_diffs_read_min=0,
        nvidia_refs_min=0,
        code_patches_min=1,
        reverts_to_evaluate=3,
    )
    assert ok is True
    assert not any("code_patches_attempted" in b for b in blockers)


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
    assert coord.shared_state.depth_tracker["code_patches_attempted"] == 1


def test_config_heavy_zero_patch_escalates_even_if_counter_low():
    """Config-heavy ledger (>=5 config keeps, 0 code_patch) escalates even
    when the consecutive run was interrupted by non-config entries."""
    s = SharedState(session_id="im-heavy")
    for i in range(5):
        s.record_intervention(
            change_type="config", action="explore", task_id=f"t{i}",
        )
    out = s.to_intervention_mix_summary()
    assert "config_keeps=5" in out
    assert "code_patch_keeps=0" in out
    assert "ESCALATION" in out
