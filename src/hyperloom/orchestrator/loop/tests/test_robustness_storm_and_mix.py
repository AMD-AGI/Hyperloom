# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Robustness storm detector + intervention-mix primitives."""

from __future__ import annotations

from hyperloom.orchestrator.state.shared_state import SharedState


# 1. Intervention-mix ledger basics
def test_intervention_mix_ledger_appends_entries():
    s = SharedState()
    assert s.intervention_mix == []
    s.record_intervention(
        change_type="config",
        action="explore",
        task_id="t1",
        delta_pct=2.3,
    )
    assert len(s.intervention_mix) == 1
    entry = s.intervention_mix[0]
    assert entry["change_type"] == "config"
    assert entry["action"] == "explore"
    assert entry["task_id"] == "t1"
    assert entry["delta_pct"] == 2.3
    assert "ts" in entry


def test_intervention_mix_normalises_change_type_casing():
    s = SharedState()
    s.record_intervention(change_type="CONFIG", action="explore")
    s.record_intervention(change_type="  code_patch  ", action="integrate_patch")
    assert s.intervention_mix[0]["change_type"] == "config"
    assert s.intervention_mix[1]["change_type"] == "code_patch"


# 2. consecutive_config_only_rounds counter
def test_consecutive_config_only_advances_on_explore_keep():
    s = SharedState()
    assert s.consecutive_config_only_rounds == 0
    s.record_intervention(change_type="config", action="explore", task_id="t1")
    assert s.consecutive_config_only_rounds == 1
    s.record_intervention(change_type="config", action="explore", task_id="t2")
    assert s.consecutive_config_only_rounds == 2


def test_consecutive_config_only_resets_on_code_patch_keep():
    s = SharedState()
    s.record_intervention(change_type="config", action="explore", task_id="t1")
    s.record_intervention(change_type="config", action="explore", task_id="t2")
    assert s.consecutive_config_only_rounds == 2
    s.record_intervention(
        change_type="code_patch",
        action="integrate_patch",
        task_id="t3",
    )
    assert s.consecutive_config_only_rounds == 0


def test_consecutive_config_only_ignores_unknown_change_type():
    """An unrecognised change_type leaves the counter unchanged."""
    s = SharedState()
    s.record_intervention(change_type="config", action="explore", task_id="t1")
    s.record_intervention(change_type="other", action="recover", task_id="t2")
    assert s.consecutive_config_only_rounds == 1


# 3. Coordinator hook: explore KEEP → config; integrate_patch kept → code_patch
def test_coordinator_intervention_hook_records_config_for_explore():
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.state.task_registry import Task

    c = Coordinator.__new__(Coordinator)
    c.shared_state = SharedState(session_id="pr-a8-config")
    task = Task(
        task_id="explore-1",
        kind="explore",
        state="succeeded",
        params={},
        idempotency_key="explore-1",
    )
    result = {
        "status": "succeeded",
        "winners": [{"name": "var1", "gain_pct": 3.5}],
        "best_variant": {"name": "var1", "gain_pct": 3.5},
    }
    c._record_intervention_for_task(task, result)
    assert len(c.shared_state.intervention_mix) == 1
    assert c.shared_state.intervention_mix[0]["change_type"] == "config"
    assert c.shared_state.consecutive_config_only_rounds == 1


def test_coordinator_intervention_hook_skips_explore_with_no_winners():
    """An empty-winners explore round records a ``config_attempt`` but doesn't advance the counter."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.state.task_registry import Task

    c = Coordinator.__new__(Coordinator)
    c.shared_state = SharedState(session_id="pr-a8-skip")
    task = Task(
        task_id="explore-2",
        kind="explore",
        state="succeeded",
        params={},
        idempotency_key="explore-2",
    )
    result = {"status": "succeeded", "winners": [], "best_variant": None}
    c._record_intervention_for_task(task, result)
    assert len(c.shared_state.intervention_mix) == 1
    assert c.shared_state.intervention_mix[0]["change_type"] == "config_attempt"
    assert c.shared_state.consecutive_config_only_rounds == 0


def test_coordinator_intervention_hook_records_code_patch_for_integrate_kept():
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.state.task_registry import Task

    c = Coordinator.__new__(Coordinator)
    c.shared_state = SharedState(session_id="pr-a8-code")
    c.shared_state.record_intervention(change_type="config", action="explore")
    c.shared_state.record_intervention(change_type="config", action="explore")
    assert c.shared_state.consecutive_config_only_rounds == 2

    task = Task(
        task_id="ip-1",
        kind="integrate_patch",
        state="succeeded",
        params={},
        idempotency_key="ip-1",
    )
    result = {
        "status": "kept",
        "delta_pct": 5.4,
        "patches_applied": ["patches/001.patch"],
    }
    c._record_intervention_for_task(task, result)
    assert c.shared_state.intervention_mix[-1]["change_type"] == "code_patch"
    assert c.shared_state.intervention_mix[-1]["delta_pct"] == 5.4
    assert c.shared_state.consecutive_config_only_rounds == 0


def test_coordinator_intervention_hook_records_integrate_attempts():
    """Non-KEEP integrate_patch attempts land on the ledger but don't reset the counter."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.state.task_registry import Task

    c = Coordinator.__new__(Coordinator)
    c.shared_state = SharedState(session_id="pr-a8-noop")
    task = Task(
        task_id="ip-2",
        kind="integrate_patch",
        state="succeeded",
        params={},
        idempotency_key="ip-2",
    )
    for status in ("reverted", "apply_failed", "rejected_by_critic", "applied_no_bench", "no_patches"):
        c._record_intervention_for_task(task, {"status": status})
    assert len(c.shared_state.intervention_mix) == 5
    assert {e["change_type"] for e in c.shared_state.intervention_mix} == {"code_patch_attempt"}
    assert c.shared_state.consecutive_config_only_rounds == 0


# 4. Advisory intervention-mix prompt summary
def test_intervention_mix_summary_no_escalation_when_balanced():
    s = SharedState()
    s.record_intervention(change_type="config", action="explore")
    s.record_intervention(change_type="code_patch", action="integrate_patch")
    out = s.to_intervention_mix_summary()
    assert "code_patch_keeps=1" in out
    assert "ESCALATION" not in out


def test_intervention_mix_summary_renders_counter_for_two_consecutive_config():
    s = SharedState()
    s.record_intervention(change_type="config", action="explore")
    s.record_intervention(change_type="config", action="explore")
    hint = s.to_intervention_mix_summary()
    assert hint
    assert "consecutive_config_only_rounds=2" in hint
    assert "ESCALATION" not in hint


def test_intervention_mix_summary_renders_counts_for_config_heavy():
    s = SharedState()
    for _ in range(5):
        s.record_intervention(change_type="config", action="explore")
    out = s.to_intervention_mix_summary()
    assert "config_keeps=5" in out
    assert "ESCALATION" not in out


def test_intervention_mix_summary_respects_threshold():
    s = SharedState()
    s.record_intervention(change_type="config", action="explore")
    # One config-only KEEP is below the escalation threshold of 2.
    out = s.to_intervention_mix_summary()
    assert "config_keeps=1" in out
    assert "ESCALATION" not in out
