# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the resume-time abandoned-dispatch sweep.
Auxiliary tests pin the closed dispatch_history schema, the
worktree-cleanup outcome enum, and the multi-restart idempotency
contract.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.dynamic_action_proposal import (
    DynamicActionStatus,
    TERMINAL_LIFECYCLE_STATUSES,
)
from inference_optimizer.orchestrator.dynamic_action_resume import (
    ABANDONED_HISTORY_FIELDS,
    AbandonedSweepResult,
    WORKTREE_CLEANUP_OUTCOMES,
    resume_abandon_dynamic_actions,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.session_paths import (
    dynamic_action_artifact_dir,
    dynamic_action_dispatch_history_path,
    dynamic_action_spec_path,
)


# ===========================================================================
# Helpers
# ===========================================================================
SCOPE = ["serving_specialist", "kernel_switch_specialist"]


def _seed_artifact(
    tmp_path: Path, dyn_id: str, *, round_index: int = 0,
) -> None:
    """Write the spec.json the resume hook needs to backfill a
    missing summary row (corner case A)."""
    art = dynamic_action_artifact_dir(tmp_path, dyn_id)
    art.mkdir(parents=True, exist_ok=True)
    dynamic_action_spec_path(tmp_path, dyn_id).write_text(
        json.dumps({
            "dyn_id": dyn_id,
            "round_index": round_index,
            "dispatched_at": "2026-05-29T00:00:00+00:00",
            "payload": {
                "motivation_gap_text": f"motivation {dyn_id}",
                "scope_domains": SCOPE,
                "side_effects_declared": ["framework_source"],
                "budget_hint": "medium",
            },
        }),
        encoding="utf-8",
    )


_HAPPY_PATH = (
    "DISPATCHED", "SUB_AGENT_RUNNING", "SUB_AGENT_DONE",
    "AWAITING_CRITIC", "INTEGRATING",
)

# Canonical predecessor lifecycle for each terminal status.
# state machine. The walker drops out of the happy path at the
# predecessor and emits one final transition to the terminal.
_TERMINAL_PREDECESSOR: dict[str, str] = {
    "TIMED_OUT": "SUB_AGENT_RUNNING",
    "FAILED": "SUB_AGENT_RUNNING",
    "COMPLETED_EMPTY": "SUB_AGENT_RUNNING",
    "CRITIC_REJECTED": "AWAITING_CRITIC",
    "INTEGRATE_FAILED": "INTEGRATING",
    "REVERTED": "INTEGRATING",
    "KEPT": "INTEGRATING",
    "ABANDONED": None,  # reachable from any non-terminal directly
}


def _walk_summary_to(state: SharedState, dyn_id: str, target: str) -> None:
    """Walk the per-dyn_id summary through the canonical happy path
    until ``status == target``. For terminals, walk to the canonical
    predecessor first then transition once to the target."""
    if target in _HAPPY_PATH:
        for st in _HAPPY_PATH:
            state.record_dynamic_action_outcome(dyn_id, status=st)
            if state.dynamic_actions[dyn_id]["status"] == target:
                return
        return
    predecessor = _TERMINAL_PREDECESSOR.get(target)
    if predecessor is not None:
        for st in _HAPPY_PATH:
            state.record_dynamic_action_outcome(dyn_id, status=st)
            if state.dynamic_actions[dyn_id]["status"] == predecessor:
                break
    state.record_dynamic_action_outcome(dyn_id, status=target)


def _init_git_base(tmp_path: Path) -> Path:
    base = tmp_path / "repo"
    base.mkdir()
    subprocess.run(["git", "init", "-q", str(base)], check=True)
    subprocess.run(
        ["git", "-C", str(base), "config", "user.email", "t@t"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(base), "config", "user.name", "t"],
        check=True,
    )
    (base / "x.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(base), "add", "x.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(base), "commit", "-q", "-m", "init"],
        check=True,
    )
    return base


# ===========================================================================
# Surface invariants
# ===========================================================================
def test_abandoned_history_fields_closed():
    assert ABANDONED_HISTORY_FIELDS == frozenset({
        "event", "ts", "previous_status",
        "coordinator_session_id", "worktree_cleanup_outcome",
        "artifact_missing",
    })


def test_worktree_cleanup_outcomes_closed_enum():
    assert WORKTREE_CLEANUP_OUTCOMES == frozenset({
        "success", "partial", "skipped",
    })


# ===========================================================================
# §10 #1 — single SUB_AGENT_RUNNING dyn_id is abandoned
# ===========================================================================
def test_p8_scenario_01_single_running_abandoned(tmp_path: Path):
    state = SharedState(session_id="sess-1")
    dyn_id = "dyn-0-1"
    _seed_artifact(tmp_path, dyn_id)
    state.record_dynamic_action_outcome(dyn_id, status="DISPATCHED")
    state.record_dynamic_action_outcome(dyn_id, status="SUB_AGENT_RUNNING")

    result = resume_abandon_dynamic_actions(
        session_dir=tmp_path,
        shared_state=state,
        coordinator_session_id="sess-1",
    )
    assert dyn_id in result.abandoned
    assert state.dynamic_actions[dyn_id]["status"] == (
        DynamicActionStatus.ABANDONED.value
    )

    history = dynamic_action_dispatch_history_path(tmp_path, dyn_id).read_text(
        encoding="utf-8",
    )
    rows = [json.loads(l) for l in history.splitlines() if l.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "abandoned_on_resume"
    assert row["previous_status"] == "SUB_AGENT_RUNNING"
    assert row["coordinator_session_id"] == "sess-1"
    assert row["worktree_cleanup_outcome"] in WORKTREE_CLEANUP_OUTCOMES
    assert set(row.keys()) <= ABANDONED_HISTORY_FIELDS


# ===========================================================================
# §10 #2 — multiple non-terminal dyn_ids handled independently
# ===========================================================================
def test_p8_scenario_02_multiple_non_terminal_all_abandoned(tmp_path: Path):
    state = SharedState(session_id="sess-2")
    # dyn-0-1 = DISPATCHED, dyn-0-2 = AWAITING_CRITIC, dyn-0-3 = INTEGRATING
    for dyn_id in ("dyn-0-1", "dyn-0-2", "dyn-0-3"):
        _seed_artifact(tmp_path, dyn_id)
    state.record_dynamic_action_outcome("dyn-0-1", status="DISPATCHED")
    _walk_summary_to(state, "dyn-0-2", "AWAITING_CRITIC")
    _walk_summary_to(state, "dyn-0-3", "INTEGRATING")

    result = resume_abandon_dynamic_actions(
        session_dir=tmp_path,
        shared_state=state,
        coordinator_session_id="sess-2",
    )
    assert set(result.abandoned) == {"dyn-0-1", "dyn-0-2", "dyn-0-3"}
    for dyn_id in ("dyn-0-1", "dyn-0-2", "dyn-0-3"):
        assert state.dynamic_actions[dyn_id]["status"] == (
            DynamicActionStatus.ABANDONED.value
        )


# ===========================================================================
# §10 #3 — terminal dyn_ids are no-ops
# ===========================================================================
def test_p8_scenario_03_terminal_dyn_ids_are_noop(tmp_path: Path):
    state = SharedState(session_id="sess-3")
    _seed_artifact(tmp_path, "dyn-0-1")
    _walk_summary_to(state, "dyn-0-1", "KEPT")
    _seed_artifact(tmp_path, "dyn-0-2")
    _walk_summary_to(state, "dyn-0-2", "FAILED")
    _seed_artifact(tmp_path, "dyn-0-3")
    _walk_summary_to(state, "dyn-0-3", "REVERTED")

    result = resume_abandon_dynamic_actions(
        session_dir=tmp_path,
        shared_state=state,
        coordinator_session_id="sess-3",
    )
    assert result.abandoned == []
    assert set(result.skipped_terminal) == {"dyn-0-1", "dyn-0-2", "dyn-0-3"}
    assert state.dynamic_actions["dyn-0-1"]["status"] == "KEPT"
    assert state.dynamic_actions["dyn-0-2"]["status"] == "FAILED"
    assert state.dynamic_actions["dyn-0-3"]["status"] == "REVERTED"


# ===========================================================================
# §10 #4 — artefact present + summary missing → recovery path
# ===========================================================================
def test_p8_scenario_04_artifact_only_rebuilds_summary(tmp_path: Path):
    state = SharedState(session_id="sess-4")
    dyn_id = "dyn-7-1"
    _seed_artifact(tmp_path, dyn_id, round_index=7)
    # No SharedState writes.
    assert dyn_id not in state.dynamic_actions

    result = resume_abandon_dynamic_actions(
        session_dir=tmp_path,
        shared_state=state,
        coordinator_session_id="sess-4",
    )
    assert dyn_id in result.abandoned
    assert dyn_id in result.summary_missing
    summary = state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.ABANDONED.value
    # Rebuilt fields propagated from spec.json.
    assert summary["scope_domains"] == SCOPE
    assert summary["round_index"] == 7
    assert summary["motivation_gap_short"].startswith("motivation")
    # Synthesised row flag preserved for audit.
    assert summary.get("synthesised_row") is True


# ===========================================================================
# §10 #5 — summary present + artefact missing → flag set
# ===========================================================================
def test_p8_scenario_05_summary_only_marks_artifact_missing(tmp_path: Path):
    state = SharedState(session_id="sess-5")
    dyn_id = "dyn-9-9"
    state.record_dynamic_action_outcome(dyn_id, status="DISPATCHED")
    state.record_dynamic_action_outcome(dyn_id, status="SUB_AGENT_RUNNING")
    # NO artefact dir created.

    result = resume_abandon_dynamic_actions(
        session_dir=tmp_path,
        shared_state=state,
        coordinator_session_id="sess-5",
    )
    assert dyn_id in result.abandoned
    assert dyn_id in result.artifact_missing
    summary = state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.ABANDONED.value
    assert summary.get("artifact_missing") is True
    # dispatch_history.jsonl is NOT written when the artefact dir is
    # missing — there is no place to land it.
    history_path = dynamic_action_dispatch_history_path(tmp_path, dyn_id)
    assert not history_path.exists()


# ===========================================================================
# §10 #6 — worktree cleanup failure does not block the sweep
# ===========================================================================
def test_p8_scenario_06_worktree_cleanup_failure_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    state = SharedState(session_id="sess-6")
    dyn_id = "dyn-0-1"
    _seed_artifact(tmp_path, dyn_id)
    state.record_dynamic_action_outcome(dyn_id, status="DISPATCHED")
    state.record_dynamic_action_outcome(dyn_id, status="SUB_AGENT_RUNNING")

    # Simulate cleanup failure: monkeypatch the cleanup helper to
    # return ``partial``.
    from inference_optimizer.orchestrator import dynamic_action_resume as mod
    monkeypatch.setattr(
        mod, "_cleanup_worktree_and_branch",
        lambda **kwargs: "partial",
    )

    result = resume_abandon_dynamic_actions(
        session_dir=tmp_path,
        shared_state=state,
        coordinator_session_id="sess-6",
    )
    # The sweep still abandons the dyn_id even though cleanup
    # partially failed.
    assert dyn_id in result.abandoned
    assert state.dynamic_actions[dyn_id]["status"] == (
        DynamicActionStatus.ABANDONED.value
    )
    rows = [
        json.loads(l)
        for l in dynamic_action_dispatch_history_path(
            tmp_path, dyn_id,
        ).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert rows[0]["worktree_cleanup_outcome"] == "partial"


# ===========================================================================
# §10 #7 — orchestration can dispatch a fresh dyn_id after sweep
# ===========================================================================
def test_p8_scenario_07_fresh_dispatch_after_abandoned(tmp_path: Path):
    """Verify the resume sweep does not corrupt the round-cap accounting:
    after abandoning ``dyn-3-1`` the next dispatch lands as a clean
    ``dyn-3-2`` via the existing record_dynamic_action_dispatch path."""
    state = SharedState(session_id="sess-7")
    # Seed initial dispatch ``dyn-3-1`` mid-flight
    dyn_id_old = "dyn-3-1"
    _seed_artifact(tmp_path, dyn_id_old, round_index=3)
    state.record_dynamic_action_dispatch(
        dyn_id_old,
        {"status": "DISPATCHED", "round_index": 3,
         "scope_domains": SCOPE, "motivation_gap_short": "first attempt"},
    )
    state.record_dynamic_action_outcome(dyn_id_old, status="SUB_AGENT_RUNNING")
    assert state.dynamic_action_round_count == 1

    resume_abandon_dynamic_actions(
        session_dir=tmp_path, shared_state=state,
        coordinator_session_id="sess-7",
    )
    assert state.dynamic_actions[dyn_id_old]["status"] == (
        DynamicActionStatus.ABANDONED.value
    )

    # Round counter is NOT mutated by the sweep (it is the
    # _on_enter_explore hook that resets it on a fresh phase entry).
    assert state.dynamic_action_round_count == 1

    # A second dispatch with a fresh dyn_id is accepted normally.
    dyn_id_new = "dyn-3-2"
    state.record_dynamic_action_dispatch(
        dyn_id_new,
        {"status": "DISPATCHED", "round_index": 3,
         "scope_domains": SCOPE, "motivation_gap_short": "rebuilt"},
    )
    assert state.dynamic_actions[dyn_id_new]["status"] == "DISPATCHED"
    assert state.dynamic_action_round_count == 2
    # The two dyn_ids coexist.
    assert state.dynamic_actions[dyn_id_old]["status"] == "ABANDONED"


# ===========================================================================
# §10 #8 — multiple restarts on the same session are idempotent
# ===========================================================================
def test_p8_scenario_08_multiple_restarts_idempotent(tmp_path: Path):
    state = SharedState(session_id="sess-8")
    dyn_id = "dyn-0-1"
    _seed_artifact(tmp_path, dyn_id)
    state.record_dynamic_action_outcome(dyn_id, status="DISPATCHED")
    state.record_dynamic_action_outcome(dyn_id, status="SUB_AGENT_RUNNING")

    first = resume_abandon_dynamic_actions(
        session_dir=tmp_path, shared_state=state,
        coordinator_session_id="sess-8",
    )
    assert first.abandoned == [dyn_id]

    # Second run: dyn_id is now ABANDONED (terminal) → skipped.
    second = resume_abandon_dynamic_actions(
        session_dir=tmp_path, shared_state=state,
        coordinator_session_id="sess-8",
    )
    assert second.abandoned == []
    assert dyn_id in second.skipped_terminal

    # dispatch_history.jsonl only carries one abandoned_on_resume row.
    rows = [
        json.loads(l)
        for l in dynamic_action_dispatch_history_path(
            tmp_path, dyn_id,
        ).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(rows) == 1


# ===========================================================================
# §10 #9 — re-dispatching the same motivation under a new dyn_id is
# independent of the abandoned predecessor
# ===========================================================================
def test_p8_scenario_09_redispatch_independent_of_abandoned(tmp_path: Path):
    state = SharedState(session_id="sess-9")
    _seed_artifact(tmp_path, "dyn-3-1", round_index=3)
    state.record_dynamic_action_outcome("dyn-3-1", status="DISPATCHED")
    state.record_dynamic_action_outcome("dyn-3-1", status="SUB_AGENT_RUNNING")
    resume_abandon_dynamic_actions(
        session_dir=tmp_path, shared_state=state,
        coordinator_session_id="sess-9",
    )
    # New dispatch under a different dyn_id with the SAME motivation.
    _seed_artifact(tmp_path, "dyn-3-2", round_index=3)
    state.record_dynamic_action_outcome("dyn-3-2", status="DISPATCHED", extra={
        "scope_domains": SCOPE,
        "motivation_gap_short": "motivation dyn-3-1",  # same text
        "round_index": 3,
    })
    # Walk to KEPT.
    for st in ("SUB_AGENT_RUNNING", "SUB_AGENT_DONE", "AWAITING_CRITIC",
               "INTEGRATING", "KEPT"):
        state.record_dynamic_action_outcome(
            "dyn-3-2", status=st,
            cumulative_gain=2.0 if st == "KEPT" else None,
        )
    # The abandoned predecessor is untouched.
    assert state.dynamic_actions["dyn-3-1"]["status"] == "ABANDONED"
    # The fresh dispatch reaches KEPT.
    assert state.dynamic_actions["dyn-3-2"]["status"] == "KEPT"
    assert state.dynamic_actions["dyn-3-2"]["cumulative_gain"] == 2.0


# ===========================================================================
# Helper / closed-schema invariants
# ===========================================================================
def test_dispatch_history_writer_rejects_unknown_outcome(tmp_path: Path):
    from inference_optimizer.orchestrator.dynamic_action_resume import (
        _append_abandoned_history,
    )
    art = dynamic_action_artifact_dir(tmp_path, "dyn-0-1")
    art.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        _append_abandoned_history(
            session_dir=tmp_path,
            dyn_id="dyn-0-1",
            previous_status="DISPATCHED",
            coordinator_session_id="x",
            worktree_cleanup_outcome="garbage",
            artifact_missing=False,
        )


def test_branch_name_pattern():
    from inference_optimizer.orchestrator.dynamic_action_resume import (
        _branch_name_for,
    )
    assert _branch_name_for("dyn-0-1") == "dynamic-dyn-0-1"
    assert _branch_name_for("dyn-9-9") == "dynamic-dyn-9-9"


def test_worktree_path_matches_runner(tmp_path: Path):
    """The cleanup helper must aim at the same path the runner used:
    runs/dynamic/<dyn_id>/worktree/."""
    from inference_optimizer.orchestrator.dynamic_action_resume import (
        _resolve_dynamic_worktree,
    )
    assert _resolve_dynamic_worktree(tmp_path, "dyn-0-1") == (
        tmp_path / "runs" / "dynamic" / "dyn-0-1" / "worktree"
    )


# ===========================================================================
# Worktree + branch live cleanup smoke test
# ===========================================================================
def test_cleanup_worktree_and_branch_with_real_git(tmp_path: Path):
    """End-to-end: spin up a real git repo, create a worktree +
    branch via the same commands the runner uses, then verify the
    sweep removes both."""
    from inference_optimizer.orchestrator.dynamic_action_resume import (
        _cleanup_worktree_and_branch,
    )
    base = _init_git_base(tmp_path)
    worktree = tmp_path / "runs" / "dynamic" / "dyn-0-1" / "worktree"
    worktree.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(base), "worktree", "add",
         "-b", "dynamic-dyn-0-1", str(worktree)],
        check=True,
    )
    assert worktree.is_dir()
    outcome = _cleanup_worktree_and_branch(
        session_dir=tmp_path,
        dyn_id="dyn-0-1",
        framework_source_roots=(str(base),),
    )
    assert outcome == "success"
    assert not worktree.exists()
    # Branch removed.
    branches = subprocess.run(
        ["git", "-C", str(base), "branch", "--list", "dynamic-dyn-0-1"],
        capture_output=True, text=True, check=True,
    )
    assert branches.stdout.strip() == ""


# ===========================================================================
# AbandonedSweepResult.to_log_line shape
# ===========================================================================
def test_sweep_result_log_line():
    r = AbandonedSweepResult(
        abandoned=["a", "b"],
        skipped_terminal=["c"],
        artifact_missing=["b"],
        summary_missing=[],
    )
    line = r.to_log_line()
    assert "abandoned=2" in line
    assert "skipped_terminal=1" in line
    assert "artifact_missing=1" in line
    assert "summary_missing=0" in line


# ===========================================================================
# ABANDONED is reachable from every non-terminal status (the writer
# accepts the transition).
# ===========================================================================
@pytest.mark.parametrize("starting_status", [
    "DISPATCHED", "SUB_AGENT_RUNNING", "SUB_AGENT_DONE",
    "AWAITING_CRITIC", "INTEGRATING",
])
def test_abandoned_reachable_from_every_non_terminal(
    tmp_path: Path, starting_status: str,
):
    state = SharedState(session_id="sess-x")
    dyn_id = "dyn-1-1"
    _seed_artifact(tmp_path, dyn_id)
    _walk_summary_to(state, dyn_id, starting_status)
    assert state.dynamic_actions[dyn_id]["status"] == starting_status

    resume_abandon_dynamic_actions(
        session_dir=tmp_path, shared_state=state,
        coordinator_session_id="sess-x",
    )
    assert state.dynamic_actions[dyn_id]["status"] == "ABANDONED"


# ===========================================================================
# Coordinator wrapper smoke (just confirms the method exists + is
# safe to call on a stub Coordinator).
# ===========================================================================
def test_coordinator_resume_hook_smoke(tmp_path: Path):
    from inference_optimizer.orchestrator.coordinator import Coordinator

    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(session_id="sess-c")
    # No dynamic_actions present — sweep is a no-op + does not raise.
    c._resume_abandon_dynamic_actions()
    assert c.shared_state.dynamic_actions == {}


# ===========================================================================
# §8.2 / §8.4 — multi-restart cycle on a mix of states converges
# without corrupting any row.
# ===========================================================================
def test_multi_restart_with_mixed_states(tmp_path: Path):
    state = SharedState(session_id="sess-mix")
    # Seed: 2 non-terminal + 2 terminal at different statuses
    for did, st in [
        ("dyn-0-1", "SUB_AGENT_RUNNING"),
        ("dyn-0-2", "AWAITING_CRITIC"),
        ("dyn-0-3", "KEPT"),
        ("dyn-0-4", "FAILED"),
    ]:
        _seed_artifact(tmp_path, did)
        _walk_summary_to(state, did, st)

    # Run sweep three times — second + third should be no-ops on
    # everything (the first run flipped the two non-terminal rows).
    r1 = resume_abandon_dynamic_actions(
        session_dir=tmp_path, shared_state=state,
        coordinator_session_id="boot-1",
    )
    r2 = resume_abandon_dynamic_actions(
        session_dir=tmp_path, shared_state=state,
        coordinator_session_id="boot-2",
    )
    r3 = resume_abandon_dynamic_actions(
        session_dir=tmp_path, shared_state=state,
        coordinator_session_id="boot-3",
    )
    assert set(r1.abandoned) == {"dyn-0-1", "dyn-0-2"}
    assert r2.abandoned == []
    assert r3.abandoned == []
    # KEPT / FAILED preserved.
    assert state.dynamic_actions["dyn-0-3"]["status"] == "KEPT"
    assert state.dynamic_actions["dyn-0-4"]["status"] == "FAILED"
