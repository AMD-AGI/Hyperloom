"""Exploration-depth gate: tracker bookkeeping, pure gate, verdict rewrite.

Covers:

1. ``SharedState`` depth_tracker counters / id-sets, the
   ``record_intervention`` link, and round-trip through state.json.
2. ``phase_state.depth_gate`` — only activates after the revert
   threshold, drops N/A (unsupplied) dimensions, and respects the
   enabled flag.
3. The Coordinator rewrites a steward stop / advance verdict to
   ``continue_explore`` when depth is insufficient, repeatedly, and
   honours the original verdict once IR-6 budget force-exit fires.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator import phase_state
from inference_optimizer.orchestrator.shared_state import SharedState


# ---------------------------------------------------------------------------
# 1. depth_tracker bookkeeping
# ---------------------------------------------------------------------------
def test_intervention_drives_patch_counters():
    s = SharedState()
    s.record_intervention(change_type="config", action="explore",
                          task_id="t1", delta_pct=1.0)
    s.record_intervention(change_type="code_patch", action="integrate_patch",
                          task_id="t2", delta_pct=2.0)
    snap = s.depth_snapshot()
    assert snap["config_changes_attempted"] == 1
    assert snap["code_patches_attempted"] == 1


def test_consecutive_reverts_increment_and_reset():
    s = SharedState()
    assert s.note_explore_outcome(promoted=False) == 1
    assert s.note_explore_outcome(promoted=False) == 2
    assert s.note_explore_outcome(promoted=True) == 0


def test_research_evidence_dedups():
    s = SharedState()
    added = s.register_research_evidence(
        prs_fetched=["p1", "p2", "p1"],
        pr_diffs_read=["d1"],
        nvidia_refs_compared=[],
    )
    assert added["prs_fetched"] == 2
    assert added["pr_diffs_read"] == 1
    again = s.register_research_evidence(prs_fetched=["p1", "p3"])
    assert again["prs_fetched"] == 1


def test_depth_tracker_roundtrip_and_failsoft():
    s = SharedState()
    s.note_explore_outcome(promoted=False)
    s.register_research_evidence(prs_fetched=["p1"])
    restored = SharedState.from_dict(s.to_dict())
    assert restored.depth_snapshot()["consecutive_reverts"] == 1
    assert restored.depth_snapshot()["prs_fetched"] == ["p1"]
    # Old state.json without the field degrades to defaults.
    legacy = SharedState.from_dict({"schema_version": 99})
    assert legacy.depth_gate_enabled() is True
    assert legacy.depth_snapshot()["consecutive_reverts"] == 0


# ---------------------------------------------------------------------------
# 2. depth_gate pure function
# ---------------------------------------------------------------------------
def _deep_revert_state() -> SharedState:
    s = SharedState()
    for _ in range(3):
        s.note_explore_outcome(promoted=False)
    return s


def test_gate_inactive_below_revert_threshold():
    s = SharedState()
    s.note_explore_outcome(promoted=False)
    satisfied, blockers, action = phase_state.depth_gate(s)
    assert satisfied is True
    assert blockers == []


def test_gate_enforces_deterministic_dims_only_when_unsupplied():
    s = _deep_revert_state()
    satisfied, blockers, action = phase_state.depth_gate(s)
    assert satisfied is False
    # scout + code patch are always enforced; PR/diff/nvidia are N/A.
    assert any(b.startswith("research_scout_runs") for b in blockers)
    assert any(b.startswith("code_patches_attempted") for b in blockers)
    assert not any(b.startswith("prs_fetched") for b in blockers)
    assert action


def test_gate_enforces_supplied_pr_dims():
    s = _deep_revert_state()
    s.bump_research_scout_runs()
    s.bump_research_scout_runs()
    s.record_intervention(change_type="code_patch", action="x",
                          task_id="t", delta_pct=1.0)
    s.register_research_evidence(prs_fetched=["1", "2"])
    satisfied, blockers, _ = phase_state.depth_gate(s)
    assert satisfied is False
    assert any(b.startswith("prs_fetched") for b in blockers)


def test_gate_satisfied_when_all_supplied_dims_met():
    s = _deep_revert_state()
    s.bump_research_scout_runs()
    s.bump_research_scout_runs()
    s.record_intervention(change_type="code_patch", action="x",
                          task_id="t", delta_pct=1.0)
    satisfied, blockers, _ = phase_state.depth_gate(s)
    assert satisfied is True
    assert blockers == []


def test_gate_disabled_is_always_satisfied():
    s = _deep_revert_state()
    s.set_depth_gate_enabled(False)
    satisfied, _, _ = phase_state.depth_gate(s)
    assert satisfied is True


def test_gate_threshold_overrides():
    s = _deep_revert_state()
    s.bump_research_scout_runs()  # 1 run
    s.record_intervention(change_type="code_patch", action="x",
                          task_id="t", delta_pct=1.0)
    # Default scout_runs_min=2 -> blocked.
    assert phase_state.depth_gate(s)[0] is False
    # Lower the bar -> satisfied.
    assert phase_state.depth_gate(s, scout_runs_min=1)[0] is True


# ---------------------------------------------------------------------------
# 3. Coordinator verdict rewrite
# ---------------------------------------------------------------------------
def _coord_with_state(state: SharedState):
    from inference_optimizer.orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord.shared_state = state
    coord._phase_budget_pct = {}
    return coord


def test_verdict_rewritten_to_continue_when_shallow(monkeypatch):
    from inference_optimizer.orchestrator import coordinator as coord_mod

    state = _deep_revert_state()
    state.explore_search = {"cursor": 4}
    coord = _coord_with_state(state)
    monkeypatch.setattr(
        coord_mod._phase_state, "should_force_exit_explore",
        lambda *a, **kw: (False, {}),
    )
    task = SimpleNamespace(task_id="task-x", params={})
    rec, next_gap = coord._apply_depth_gate_to_verdict(
        raw_rec="stop_session", next_gap="", task=task,
    )
    assert rec == "continue_explore"
    assert next_gap.startswith("gap.depth.")
    assert any(g.get("canonical_id") == next_gap for g in state.gaps)


def test_verdict_kept_when_force_exit(monkeypatch):
    from inference_optimizer.orchestrator import coordinator as coord_mod

    state = _deep_revert_state()
    state.explore_search = {"cursor": 4}
    coord = _coord_with_state(state)
    monkeypatch.setattr(
        coord_mod._phase_state, "should_force_exit_explore",
        lambda *a, **kw: (True, {"fired_reasons": ["session_remaining"]}),
    )
    task = SimpleNamespace(task_id="task-x", params={})
    rec, _ = coord._apply_depth_gate_to_verdict(
        raw_rec="stop_session", next_gap="", task=task,
    )
    assert rec == "stop_session"


def test_verdict_kept_when_depth_satisfied(monkeypatch):
    from inference_optimizer.orchestrator import coordinator as coord_mod

    state = _deep_revert_state()
    state.bump_research_scout_runs()
    state.bump_research_scout_runs()
    state.record_intervention(change_type="code_patch", action="x",
                              task_id="t", delta_pct=1.0)
    state.explore_search = {"cursor": 4}
    coord = _coord_with_state(state)
    monkeypatch.setattr(
        coord_mod._phase_state, "should_force_exit_explore",
        lambda *a, **kw: (False, {}),
    )
    task = SimpleNamespace(task_id="task-x", params={})
    rec, _ = coord._apply_depth_gate_to_verdict(
        raw_rec="advance_to_kernel", next_gap="", task=task,
    )
    assert rec == "advance_to_kernel"
