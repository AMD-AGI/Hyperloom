# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the long-horizon run state, event log, and prompt view.

These cover the file-backed state substrate that lets a long forge-loop run be
driven from files instead of an ever-growing prompt: atomic state save/load,
append-only event replay, the iteration reducer, and the bounded prompt-view
header (overview + retrieval pointers, detail left on disk).
"""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path

import pytest

from kernelforge.loop.prompt_view import render_long_horizon_header
from kernelforge.loop.run_state import (
    ORCHESTRATION_CIRCUIT_CLOSED,
    ORCHESTRATION_CIRCUIT_HALF_OPEN,
    ORCHESTRATION_CIRCUIT_OPEN,
    PHASE_EXPLOIT,
    PHASE_EXPLORE,
    PHASE_STALLED,
    SESSION_COMPLETED,
    SESSION_INTERRUPTED,
    SESSION_PAUSED,
    SCHEMA_VERSION,
    SESSION_RUNNING,
    _RECENT_RESULT_CACHE,
    CriticRuling,
    LoopStateStore,
    RunState,
    WorkspaceLockError,
    apply_iteration,
    apply_supervisor_attempt,
    apply_supervisor_intervention,
    begin_orchestration_probe,
    complete_orchestration_probe,
    finish_session,
    make_event,
    pin_iteration,
    reconcile_stale_running_session,
    should_resume,
    start_session,
)
from kernelforge.loop.runner import (
    LONG_HORIZON_OUTCOME_WINDOW,
    _long_horizon_header,
)

# The header's own rendering budgets, read from the definition that owns them so
# the expectations below cannot drift from the defaults the loop relies on.
_HEADER_PARAMS = inspect.signature(render_long_horizon_header).parameters
_HEADER_MAX_RECENT = _HEADER_PARAMS["max_recent"].default
_HEADER_MAX_CHARS = _HEADER_PARAMS["max_chars"].default


# ── state persistence ─────────────────────────────────────────────────────────
def test_save_load_roundtrip_preserves_control_state(tmp_path):
    store = LoopStateStore(str(tmp_path))
    state = store.load()
    apply_iteration(
        state,
        iteration=7,
        decision="KEEP",
        kept=True,
        wall_ms=0.5,
        mean_case_speedup=2.0,
        commit_hash="abc1234",
        plan="vectorize global loads",
        baseline_wall_ms=1.0,
        best_wall_ms=0.5,
        best_mean_case_speedup=2.0,
    )
    state.analysis.evidence_commit = "abc1234"
    state.analysis.evidence_mean_case_speedup = 2.0
    state.analysis.evidence_status = "profiled"
    store.save(state)

    reloaded = LoopStateStore(str(tmp_path)).load()
    assert reloaded.best.iteration == 7
    assert reloaded.best.wall_ms == 0.5
    assert reloaded.best.mean_case_speedup == 2.0
    assert reloaded.best.commit_hash == "abc1234"
    assert reloaded.baseline_wall_ms == 1.0
    assert reloaded.phase == PHASE_EXPLOIT
    assert reloaded.analysis.evidence_commit == "abc1234"
    assert reloaded.analysis.evidence_mean_case_speedup == 2.0
    assert reloaded.analysis.evidence_status == "profiled"


def test_save_leaves_no_temp_files(tmp_path):
    store = LoopStateStore(str(tmp_path))
    store.save(store.load())
    leftovers = list((tmp_path / "forge_experiments").glob(".run_state.*.tmp"))
    assert leftovers == []
    assert (tmp_path / "forge_experiments" / "run_state.json").exists()


def test_load_missing_returns_fresh(tmp_path):
    state = LoopStateStore(str(tmp_path)).load()
    assert state.iteration == 0
    assert state.best.iteration == 0
    assert state.phase == PHASE_EXPLORE


def test_load_corrupt_fails_closed(tmp_path):
    root = tmp_path / "forge_experiments"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_state.json").write_text("{ this is not valid json ]")
    with pytest.raises(ValueError, match="invalid run state checkpoint"):
        LoopStateStore(str(tmp_path)).load()


def test_load_noncurrent_schema_fails_closed(tmp_path):
    root = tmp_path / "forge_experiments"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": 7,
                "phase": PHASE_EXPLOIT,
                "best": {
                    "iteration": 5,
                    "wall_ms": 0.5,
                    "commit_hash": "abc1234",
                    "plan": "vectorize loads",
                },
            }
        )
    )

    with pytest.raises(ValueError, match="unsupported run state schema"):
        LoopStateStore(str(tmp_path)).load()


def test_load_v13_migrates_with_empty_analysis_anchor(tmp_path):
    """A v13 checkpoint crosses every version added since, not just the next."""
    store = LoopStateStore(str(tmp_path))
    payload = RunState().to_dict()
    payload["schema_version"] = 13
    payload.pop("analysis")
    payload.pop("last_critic")
    root = tmp_path / "forge_experiments"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_state.json").write_text(json.dumps(payload))

    migrated = store.load()

    assert migrated.schema_version == SCHEMA_VERSION
    assert migrated.analysis.evidence_commit == ""
    assert migrated.analysis.evidence_mean_case_speedup is None
    assert migrated.last_critic == CriticRuling()


def test_load_v14_migrates_with_no_critic_ruling(tmp_path):
    """What such a campaign knows is that it never recorded a verdict.

    An empty ruling divides the next round as an ordinary one, which is what a
    checkpoint written before the ruling existed can honestly support.
    """
    store = LoopStateStore(str(tmp_path))
    payload = RunState().to_dict()
    payload["schema_version"] = 14
    payload.pop("last_critic")
    root = tmp_path / "forge_experiments"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_state.json").write_text(json.dumps(payload))

    migrated = store.load()

    assert migrated.schema_version == SCHEMA_VERSION
    assert migrated.last_critic == CriticRuling()


def test_load_v17_migrates_with_a_campaign_clock_that_covers_its_planning(
    tmp_path,
):
    """A v17 checkpoint banked planning with no span to divide it by.

    That is why a resumed session published a planning share above 100: the
    cumulative numerator was divided by the current process's wall-clock. The
    migration has to supply a denominator such a checkpoint can actually
    support, and the rounds' own recorded wall-clock is it -- a lower bound on
    how long the campaign ran, and one that already covers the planning inside
    it, since no round's total is smaller than its own planning.
    """
    store = LoopStateStore(str(tmp_path))
    payload = RunState().to_dict()
    payload["schema_version"] = 17
    payload["round_costs"]["rounds"] = 3
    payload["round_costs"]["planning_total_sec"] = 2700.0
    payload["round_costs"]["total_sec"] = 3300.0
    payload["round_costs"].pop("campaign_sec")
    root = tmp_path / "forge_experiments"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_state.json").write_text(json.dumps(payload))

    migrated = store.load()

    assert migrated.schema_version == SCHEMA_VERSION
    assert migrated.round_costs.campaign_sec == 3300.0
    # A share, not a number several times its own definition.
    assert migrated.round_costs.planning_share_pct() == pytest.approx(100.0 * 2700.0 / 3300.0)


def test_load_v17_without_round_wall_clock_still_covers_its_planning(tmp_path):
    """The degenerate v17 shape: planning recorded, round totals missing.

    Falling back to the round wall-clock alone would hand back a span shorter
    than the planning charged to it -- the same broken division in durable
    form -- so the migration takes the larger of the two.
    """
    store = LoopStateStore(str(tmp_path))
    payload = RunState().to_dict()
    payload["schema_version"] = 17
    payload["round_costs"]["rounds"] = 2
    payload["round_costs"]["planning_total_sec"] = 2700.0
    payload["round_costs"]["total_sec"] = 0.0
    payload["round_costs"].pop("campaign_sec")
    root = tmp_path / "forge_experiments"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_state.json").write_text(json.dumps(payload))

    migrated = store.load()

    assert migrated.round_costs.campaign_sec == 2700.0
    assert migrated.round_costs.planning_share_pct() == pytest.approx(100.0)


def test_load_v18_seeds_the_stall_counter_from_the_shared_streak(tmp_path):
    """A v18 checkpoint held one counter for two questions.

    Its no-improvement streak was reset by every past supervisor intervention,
    so it understates how long the search has really been stuck. Seeding from
    it is the fail-safe direction: a resumed campaign can be a few iterations
    late to DIVERSIFY, but it can never claim a stall it did not measure.
    """
    store = LoopStateStore(str(tmp_path))
    payload = RunState().to_dict()
    payload["schema_version"] = 18
    payload["stall"]["no_improvement_iters"] = 4
    payload["stall"].pop("unresolved_stall_iters")
    root = tmp_path / "forge_experiments"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_state.json").write_text(json.dumps(payload))

    migrated = store.load()

    assert migrated.schema_version == SCHEMA_VERSION
    assert migrated.stall.no_improvement_iters == 4
    assert migrated.stall.unresolved_stall_iters == 4


def test_a_keep_clears_both_stall_counters():
    """One measured improvement ends the stall episode outright.

    The split counter must not latch: it is "iterations since the last real
    KEEP", so a KEEP zeroes it exactly as it zeroes the supervisor cooldown.
    """
    state = RunState()
    apply_iteration(
        state,
        iteration=1,
        decision="REVERT_PERF",
        kept=False,
        wall_ms=1.0,
        commit_hash="",
        plan="widen the tile",
        baseline_wall_ms=1.0,
        best_wall_ms=1.0,
    )
    apply_supervisor_intervention(state, iteration=1, stall_threshold=3)
    apply_iteration(
        state,
        iteration=2,
        decision="REVERT_PERF",
        kept=False,
        wall_ms=1.0,
        commit_hash="",
        plan="widen the tile again",
        baseline_wall_ms=1.0,
        best_wall_ms=1.0,
    )

    assert state.stall.no_improvement_iters == 1
    assert state.stall.unresolved_stall_iters == 2

    apply_iteration(
        state,
        iteration=3,
        decision="KEEP",
        kept=True,
        wall_ms=0.6,
        mean_case_speedup=1.6,
        commit_hash="kept-commit",
        plan="fuse the epilogue",
        baseline_wall_ms=1.0,
        best_wall_ms=0.6,
        best_mean_case_speedup=1.6,
    )

    assert state.stall.no_improvement_iters == 0
    assert state.stall.unresolved_stall_iters == 0


def test_session_transitions_preserve_campaign_and_advance_index():
    state = RunState()

    start_session(
        state,
        campaign_id="campaign-123",
        experiment_id="experiment-1",
    )
    assert state.campaign_id == "campaign-123"
    assert state.session_index == 1
    assert state.session_status == SESSION_RUNNING
    assert state.last_experiment_id == "experiment-1"

    finish_session(state, status=SESSION_PAUSED, reason="iteration_budget")
    assert state.session_status == SESSION_PAUSED
    assert state.termination_reason == "iteration_budget"

    start_session(state, experiment_id="experiment-2")
    assert state.campaign_id == "campaign-123"
    assert state.session_index == 2
    assert state.session_status == SESSION_RUNNING
    assert state.last_experiment_id == "experiment-2"
    assert state.termination_reason == ""

    finish_session(state, status=SESSION_COMPLETED, reason="target_met")
    assert state.session_status == SESSION_COMPLETED


def test_reconcile_stale_running_session_marks_interrupted():
    state = RunState(
        session_status=SESSION_RUNNING,
        termination_reason="",
    )

    assert reconcile_stale_running_session(state) is True
    assert state.session_status == SESSION_INTERRUPTED
    assert state.termination_reason == "stale_running_session_reconciled"
    assert reconcile_stale_running_session(state) is False


def test_iteration_reducer_advances_global_cursor_and_counters():
    state = RunState(next_iteration=8)

    apply_iteration(
        state,
        iteration=8,
        decision="KEEP",
        kept=True,
        wall_ms=0.7,
        commit_hash="deadbeef",
        plan="fuse epilogue",
        baseline_wall_ms=1.0,
        best_wall_ms=0.7,
    )
    apply_iteration(
        state,
        iteration=9,
        decision="REVERT_PERF",
        kept=False,
        wall_ms=0.8,
        commit_hash="",
        plan="increase tile size",
        baseline_wall_ms=1.0,
        best_wall_ms=0.7,
    )

    assert state.iteration == 9
    assert state.next_iteration == 10
    assert state.cumulative.iterations == 2
    assert state.cumulative.kept == 1
    assert state.cumulative.reverted == 1

    with pytest.raises(ValueError, match="iteration 9"):
        apply_iteration(
            state,
            iteration=9,
            decision="REVERT_PERF",
            kept=False,
            wall_ms=0.8,
            commit_hash="",
            plan="duplicate attempt",
            baseline_wall_ms=1.0,
            best_wall_ms=0.7,
        )


def test_intervention_count_persists_across_save_load(tmp_path):
    store = LoopStateStore(str(tmp_path))
    state = RunState()
    apply_supervisor_intervention(state, iteration=5)
    apply_supervisor_intervention(state, iteration=11)
    store.save(state)

    reloaded = LoopStateStore(str(tmp_path)).load()

    assert reloaded.intervention_count == 2
    assert reloaded.stall.last_supervisor_iter == 11
    assert reloaded.stall.last_supervisor_attempt_iter == 11


def test_supervisor_attempt_anchor_persists_without_intervention(tmp_path):
    store = LoopStateStore(str(tmp_path))
    state = RunState()
    state.stall.no_improvement_iters = 4
    apply_supervisor_attempt(state, iteration=7)
    store.save(state)

    reloaded = store.load()

    assert reloaded.stall.last_supervisor_attempt_iter == 7
    assert reloaded.stall.last_supervisor_iter == 0
    assert reloaded.stall.no_improvement_iters == 4
    assert reloaded.intervention_count == 0


def test_workspace_lock_rejects_concurrent_owner_and_can_be_reacquired(tmp_path):
    first = LoopStateStore(str(tmp_path))
    second = LoopStateStore(str(tmp_path))

    with first.workspace_lock():
        with pytest.raises(WorkspaceLockError, match="already in use"):
            with second.workspace_lock():
                pass

    with second.workspace_lock():
        assert second.lock_path.exists()


# ── event log ──────────────────────────────────────────────────────────────────
def test_append_and_read_events_in_order(tmp_path):
    store = LoopStateStore(str(tmp_path))
    store.append_event(make_event("iteration_started", 1, best_before_ms=1.0))
    store.append_event(make_event("iteration_result", 1, decision="REVERT_PERF", wall_ms=1.1))
    store.append_event(make_event("iteration_result", 2, decision="KEEP", wall_ms=0.9))

    events = store.read_events()
    assert [e["type"] for e in events] == [
        "iteration_started",
        "iteration_result",
        "iteration_result",
    ]
    assert events[-1]["decision"] == "KEEP"
    # ``None`` fields are dropped; ts/type/iter always present.
    assert all("ts" in e and "type" in e and "iter" in e for e in events)


def test_read_events_limit_and_skips_bad_lines(tmp_path):
    store = LoopStateStore(str(tmp_path))
    for i in range(5):
        store.append_event(make_event("iteration_result", i, decision="REVERT_PERF"))
    # A malformed line must be skipped by the full reader, not crash it.
    with open(store.events_path, "a") as f:
        f.write("{not json}\n")
    events = store.read_events()
    assert [e["iter"] for e in events] == [0, 1, 2, 3, 4]


def test_read_events_skips_valid_json_non_objects_and_primes_dicts_only(tmp_path):
    root = tmp_path / "forge_experiments"
    root.mkdir()
    valid_events = [
        make_event("iteration_result", 1, decision="REVERT_PERF"),
        make_event("iteration_result", 2, decision="KEEP"),
    ]
    (root / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(valid_events[0]),
                "null",
                json.dumps(["not", "an", "event"]),
                json.dumps("scalar"),
                "17",
                json.dumps({"type": "iteration_result", "iter": "bad"}),
                json.dumps({"type": 7, "iter": 3}),
                json.dumps({"type": "iteration_result", "iter": True}),
                "{malformed",
                json.dumps(valid_events[1]),
            ]
        )
        + "\n"
    )

    store = LoopStateStore(str(tmp_path))

    assert store.read_events() == valid_events
    assert store.recent_events(10) == valid_events
    assert all(isinstance(event, dict) for event in store.read_events())


def test_recent_events_served_from_cache_and_reopened(tmp_path):
    store = LoopStateStore(str(tmp_path))
    for i in range(10):
        store.append_event(make_event("iteration_result", i, decision="REVERT_PERF"))
    # Served from the in-memory cache (no full-file re-parse).
    assert [e["iter"] for e in store.recent_events(3)] == [7, 8, 9]
    # A new store primes its recent cache from disk.
    reopened = LoopStateStore(str(tmp_path))
    assert [e["iter"] for e in reopened.recent_events(2)] == [8, 9]


def test_make_event_drops_none_fields():
    ev = make_event("iteration_result", 3, plan=None, wall_ms=0.4, error_sig=None)
    assert "plan" not in ev
    assert "error_sig" not in ev
    assert ev["wall_ms"] == 0.4
    assert ev["iter"] == 3


# ── reducer ─────────────────────────────────────────────────────────────────────
def test_apply_iteration_keep_updates_best_and_resets_stall():
    state = RunState()
    state.stall.no_improvement_iters = 4
    apply_iteration(
        state,
        iteration=10,
        decision="KEEP",
        kept=True,
        wall_ms=0.7,
        commit_hash="deadbeef",
        plan="fuse epilogue",
        baseline_wall_ms=1.0,
        best_wall_ms=0.7,
    )
    assert state.best.iteration == 10
    assert state.best.wall_ms == 0.7
    assert state.stall.no_improvement_iters == 0
    assert 10 in state.pinned_iterations
    assert state.phase == PHASE_EXPLOIT


def test_apply_iteration_non_keep_increments_stall_then_stalled_phase():
    state = RunState()
    for i in range(1, 6):
        apply_iteration(
            state,
            iteration=i,
            decision="REVERT_PERF",
            kept=False,
            wall_ms=1.2,
            commit_hash="",
            plan="",
            baseline_wall_ms=1.0,
            best_wall_ms=1.0,
            stall_threshold=5,
        )
    assert state.stall.no_improvement_iters == 5
    assert state.phase == PHASE_STALLED


def test_an_api_error_is_counted_apart_from_a_revert_and_leaves_the_stall_alone():
    """A gateway outage measured nothing, so it is not an optimization outcome.

    Counting it as ``reverted`` understated the optimizer on a bad-gateway day, and
    extending the stall streak pulled in the supervisor to redirect an agent that
    never ran -- three consecutive outages read as "the optimizer stopped
    improving".
    """
    state = RunState()
    apply_iteration(
        state,
        iteration=1,
        decision="REVERT_PERF",
        kept=False,
        wall_ms=1.2,
        commit_hash="",
        plan="bigger tile",
        baseline_wall_ms=1.0,
        best_wall_ms=1.0,
        stall_threshold=3,
    )
    for i in (2, 3, 4):
        apply_iteration(
            state,
            iteration=i,
            decision="API_ERROR",
            kept=False,
            wall_ms=None,
            commit_hash="",
            plan="",
            baseline_wall_ms=1.0,
            best_wall_ms=1.0,
            stall_threshold=3,
        )

    assert state.cumulative.api_errors == 3
    assert state.cumulative.reverted == 1, "an outage is not a rejected candidate"
    assert state.cumulative.kept == 0
    # iterations stays the loop's own count: kept + reverted + api_errors.
    assert state.cumulative.iterations == 4
    assert state.stall.no_improvement_iters == 1, "only the real attempt counts"
    assert state.phase != PHASE_STALLED


def test_api_error_counters_survive_save_load(tmp_path):
    store = LoopStateStore(str(tmp_path))
    state = store.load()
    apply_iteration(
        state,
        iteration=1,
        decision="API_ERROR",
        kept=False,
        wall_ms=None,
        commit_hash="",
        plan="",
        baseline_wall_ms=1.0,
        best_wall_ms=1.0,
    )
    store.save(state)

    assert LoopStateStore(str(tmp_path)).load().cumulative.api_errors == 1


def test_orchestration_circuit_opens_and_half_open_probe_is_single_shot():
    state = RunState()
    for iteration in (1, 2, 3):
        apply_iteration(
            state,
            iteration=iteration,
            decision="ORCHESTRATION_ERROR",
            kept=False,
            wall_ms=None,
            commit_hash="",
            plan="",
            baseline_wall_ms=1.0,
            best_wall_ms=1.0,
            orchestration_error_threshold=3,
        )

    assert state.orchestration_error_streak == 3
    assert state.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_OPEN

    begin_orchestration_probe(state)
    assert state.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_HALF_OPEN
    apply_iteration(
        state,
        iteration=4,
        decision="ORCHESTRATION_ERROR",
        kept=False,
        wall_ms=None,
        commit_hash="",
        plan="",
        baseline_wall_ms=1.0,
        best_wall_ms=1.0,
        orchestration_error_threshold=3,
    )

    assert state.orchestration_error_streak == 4
    assert state.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_OPEN


def test_successful_half_open_probe_closes_and_clears_streak():
    state = RunState(
        orchestration_error_streak=3,
        orchestration_circuit_state=ORCHESTRATION_CIRCUIT_OPEN,
    )

    begin_orchestration_probe(state)
    complete_orchestration_probe(state)

    assert state.orchestration_error_streak == 0
    assert state.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_CLOSED


def test_apply_iteration_caps_pinned_iterations():
    state = RunState()
    for i in range(1, 20):
        apply_iteration(
            state,
            iteration=i,
            decision="KEEP",
            kept=True,
            wall_ms=1.0 / i,
            commit_hash=f"c{i}",
            plan="p",
            baseline_wall_ms=1.0,
            best_wall_ms=1.0 / i,
            max_pinned=8,
        )
    assert len(state.pinned_iterations) == 8
    assert state.pinned_iterations[-1] == 19


# ── prompt view ────────────────────────────────────────────────────────────────
def test_header_empty_on_cold_start():
    state = RunState()
    state.iteration = 1  # loop marks the current iteration before any result
    header = render_long_horizon_header(state, [make_event("iteration_started", 1)])
    assert header == ""


def test_header_contains_best_and_retrieval_pointers():
    state = RunState(baseline_wall_ms=1.0)
    apply_iteration(
        state,
        iteration=3,
        decision="KEEP",
        kept=True,
        wall_ms=0.5,
        commit_hash="abc",
        plan="vectorize loads",
        baseline_wall_ms=1.0,
        best_wall_ms=0.5,
    )
    events = [
        make_event("iteration_result", 3, decision="KEEP", plan="vectorize loads", wall_ms=0.5),
    ]
    header = render_long_horizon_header(state, events)
    assert "Long-Horizon Memory" in header
    assert "Current best: iter 3" in header
    assert "vectorize loads" in header
    # Retrieval map points the agent at on-disk detail.
    assert "forge_experiments/run_state.json" in header
    assert "forge_experiments/events.jsonl" in header
    assert "candidates/index.jsonl" in header
    assert "iter_NNN" in header


def test_header_bounded_by_max_chars():
    state = RunState(baseline_wall_ms=1.0)
    apply_iteration(
        state,
        iteration=1,
        decision="KEEP",
        kept=True,
        wall_ms=0.9,
        commit_hash="c1",
        plan="p",
        baseline_wall_ms=1.0,
        best_wall_ms=0.9,
    )
    events = [
        make_event(
            "iteration_result",
            i,
            decision="REVERT_PERF",
            plan=f"attempt number {i} with a fairly long descriptive plan text",
            wall_ms=1.0 + i / 100.0,
            error_sig=f"error signature {i} that is reasonably verbose to consume space",
        )
        for i in range(40)
    ]
    header = render_long_horizon_header(state, events, max_chars=800)
    assert len(header) <= 800


def test_header_keeps_retrieval_map_under_tight_budget():
    state = RunState(baseline_wall_ms=1.0)
    apply_iteration(
        state,
        iteration=1,
        decision="KEEP",
        kept=True,
        wall_ms=0.9,
        commit_hash="c1",
        plan="p",
        baseline_wall_ms=1.0,
        best_wall_ms=0.9,
    )
    events = [
        make_event("iteration_result", i, decision="REVERT_PERF", plan="x" * 40, error_sig="y" * 80) for i in range(20)
    ]
    header = render_long_horizon_header(state, events, max_recent=20, max_chars=600)
    assert len(header) <= 600
    # The retrieval map is never trimmed away, even when recent lines are dropped.
    assert "forge_experiments/run_state.json" in header
    assert "index.jsonl" in header


def test_header_can_expose_iteration_handoffs():
    state = RunState(baseline_wall_ms=1.0)
    apply_iteration(
        state,
        iteration=1,
        decision="REVERT_PERF",
        kept=False,
        wall_ms=1.1,
        commit_hash="",
        plan="try a new layout",
        baseline_wall_ms=1.0,
        best_wall_ms=1.0,
    )

    header = render_long_horizon_header(
        state,
        [],
        include_handoffs=True,
    )

    assert "forge_experiments/handoffs/iter_NNN.json" in header


def test_header_preserves_retrieval_map_when_budget_is_below_essential_content():
    state = RunState(baseline_wall_ms=1.0)
    apply_iteration(
        state,
        iteration=1,
        decision="REVERT_PERF",
        kept=False,
        wall_ms=1.1,
        commit_hash="",
        plan="test an intentionally tiny rendering budget",
        baseline_wall_ms=1.0,
        best_wall_ms=1.0,
    )

    header = render_long_horizon_header(state, [], max_chars=80)

    assert "forge_experiments/run_state.json" in header
    assert "forge_experiments/events.jsonl" in header
    assert "forge_experiments/candidates/index.jsonl" in header
    assert "forge_experiments/candidates/iter_NNN/" in header
    assert len(header) > 80


def test_header_shows_baseline_not_iter0_best_before_first_keep():
    state = RunState(baseline_wall_ms=1.0)
    apply_iteration(
        state,
        iteration=1,
        decision="REVERT_PERF",
        kept=False,
        wall_ms=1.1,
        commit_hash="",
        plan="bigger tile",
        baseline_wall_ms=1.0,
        best_wall_ms=1.0,
    )
    header = render_long_horizon_header(
        state, [make_event("iteration_result", 1, decision="REVERT_PERF", plan="bigger tile")]
    )
    assert "Baseline: mean case speedup 1.000000x" in header
    assert "Current best: iter 0" not in header


def _pin_hint(header: str) -> str:
    """The retrieval map's pin hint, or "" when the header renders no pins."""
    match = re.search(r"\(pinned[^)]*\)", header)
    return match.group(0) if match else ""


def _pinned_iterations(header: str) -> list[int]:
    """Iteration numbers the pin hint names, ignoring any rendered speedup."""
    return [int(token) for token in re.findall(r"(?<![\d.])\d+(?![\d.])", _pin_hint(header))]


def _pin_entries(header: str) -> list[str]:
    """The pin hint's entries, each exactly as the retrieval map renders it."""
    match = re.search(r"\(pinned: ([^)]*)\)", header)
    return [entry.strip() for entry in match.group(1).split(",")] if match else []


def _rendered_attempt_iterations(header: str) -> list[int]:
    """Iteration numbers of the recent-attempt fact lines the header renders."""
    return [int(token) for token in re.findall(r"^- iter (\d+)", header, re.MULTILINE)]


def _store_with_live_iteration_events(tmp_path, iterations: range) -> LoopStateStore:
    """A store fed the events a live iteration writes, for each iteration.

    Every iteration logs its search-policy decision, its analysis result and an
    iteration_started marker before the outcome, so a tail counted in raw events
    reaches roughly a quarter of the outcomes its length suggests.
    """
    store = LoopStateStore(str(tmp_path))
    for iteration in iterations:
        store.append_event(make_event("search_policy_decision", iteration, mode="EXPLOIT"))
        store.append_event(make_event("iteration_started", iteration, phase=PHASE_EXPLOIT))
        store.append_event(make_event("analysis_result", iteration, status="ready"))
        store.append_event(
            make_event(
                "iteration_result",
                iteration,
                decision="KEEP" if iteration == 3 else "REVERT_PERF",
                plan=f"attempt {iteration}",
                mean_case_speedup=1.2 if iteration == 3 else 1.0 + iteration / 1000.0,
            )
        )
    return store


def _state_with_best_pin_held_against_near_misses() -> RunState:
    """A state whose pin list holds the best lineage ahead of later pins.

    ``pin_iteration`` keeps the iteration behind the current best when the list
    overflows, so after enough later near-misses the best lineage sits at the
    front of a list longer than the header renders.
    """
    state = RunState(baseline_wall_ms=1.0)
    apply_iteration(
        state,
        iteration=3,
        decision="KEEP",
        kept=True,
        wall_ms=0.5,
        mean_case_speedup=1.2,
        commit_hash="kept",
        plan="vectorize global loads",
        baseline_wall_ms=1.0,
        best_wall_ms=0.5,
        best_mean_case_speedup=1.2,
    )
    for iteration in range(4, 12):
        pin_iteration(state, iteration)
    assert state.pinned_iterations == [3, 5, 6, 7, 8, 9, 10, 11]
    return state


def test_header_pin_hint_keeps_the_held_best_lineage_pin():
    """The map names the pin ``pin_iteration`` held against the near-misses.

    The pin list is capped at eight with the best lineage held at the front, so
    rendering only its tail drops exactly the pin the map exists to point at.
    """
    state = _state_with_best_pin_held_against_near_misses()

    header = render_long_horizon_header(state, [])

    assert _pinned_iterations(header) == [3, 7, 8, 9, 10, 11]


def test_header_pin_hint_marks_the_best_and_carries_measured_speedups():
    """Each pin says what it is, so a bare number is never all the agent gets.

    ``pinned_iterations`` holds iteration numbers alone, so the measured mean
    case speedups come from the best record and the supplied outcome events; a
    pin older than that window renders as its iteration number only.
    """
    state = _state_with_best_pin_held_against_near_misses()
    events = [
        make_event(
            "iteration_result",
            9,
            decision="REVERT_PERF",
            plan="stage the scales through LDS",
            mean_case_speedup=1.0031,
        ),
        make_event(
            "iteration_result",
            11,
            decision="REVERT_PERF",
            plan="unroll the tail loop",
            mean_case_speedup=1.0125,
        ),
    ]

    hint = _pin_hint(render_long_horizon_header(state, events))

    assert "3 best 1.200000x" in hint
    assert "9 1.003100x" in hint
    assert "11 1.012500x" in hint
    # Iterations 7, 8 and 10 are pinned but outside the supplied event window,
    # so they carry no score rather than a guessed one.
    assert re.search(r"\b7, 8\b", hint)
    assert re.search(r"\b10, 11\b", hint)


def test_loop_header_scores_every_pin_and_fills_the_recent_budget(tmp_path):
    """The loop hands the header a window counted in outcomes, so both fit in it.

    An iteration writes four events before its outcome, so the eight raw events
    this header used to be handed reached two outcomes: the recent list rendered
    a third of the budget it is allowed, and every pin older than those two
    outcomes rendered as a bare number, which reads as an attempt that measured
    nothing.
    """
    state = _state_with_best_pin_held_against_near_misses()
    store = _store_with_live_iteration_events(tmp_path, range(1, 12))

    header = _long_horizon_header(state, store)

    attempts = _rendered_attempt_iterations(header)
    assert attempts == [6, 7, 8, 9, 10, 11]
    assert len(attempts) == _HEADER_MAX_RECENT
    assert _pin_entries(header) == [
        "3 best 1.200000x",
        "7 1.007000x",
        "8 1.008000x",
        "9 1.009000x",
        "10 1.010000x",
        "11 1.011000x",
    ]
    # A wider window feeds more outcomes but renders no more of them: what the
    # header shows is still bounded by its own budgets.
    assert len(header) <= _HEADER_MAX_CHARS

    # The window this replaces, from the same log: eight raw events reached two
    # outcomes, so three of the six pins carried no measured speedup at all.
    stale = render_long_horizon_header(state, store.recent_events(8))
    assert _rendered_attempt_iterations(stale) == [10, 11]
    assert _pin_entries(stale) == [
        "3 best 1.200000x",
        "7",
        "8",
        "9",
        "10 1.010000x",
        "11 1.011000x",
    ]


def test_loop_header_window_covers_the_pin_cap_and_is_served_from_the_cache(tmp_path):
    """The window must span every pin the state can hold and be answerable.

    ``recent_results`` refuses a request wider than its cache rather than
    answering short, and the loop renders this header best-effort, so a window
    beyond the cache would cost every session its header instead.
    """
    state = RunState()
    for iteration in range(1, 3 * LONG_HORIZON_OUTCOME_WINDOW):
        pin_iteration(state, iteration)
    store = _store_with_live_iteration_events(tmp_path, range(1, LONG_HORIZON_OUTCOME_WINDOW + 2))

    window = store.recent_results(LONG_HORIZON_OUTCOME_WINDOW)

    assert len(state.pinned_iterations) <= LONG_HORIZON_OUTCOME_WINDOW
    assert _HEADER_MAX_RECENT <= LONG_HORIZON_OUTCOME_WINDOW <= _RECENT_RESULT_CACHE
    assert [event["iter"] for event in window] == list(range(2, LONG_HORIZON_OUTCOME_WINDOW + 2))
    with pytest.raises(ValueError, match="exceeds the cached outcome bound"):
        store.recent_results(_RECENT_RESULT_CACHE + 1)


# ── reducer / resume guards ─────────────────────────────────────────────────────
def test_no_best_recorded_before_first_keep():
    state = RunState()
    apply_iteration(
        state,
        iteration=1,
        decision="REVERT_PERF",
        kept=False,
        wall_ms=1.1,
        commit_hash="",
        plan="x",
        baseline_wall_ms=1.0,
        best_wall_ms=1.0,
    )
    # The baseline must NOT masquerade as a best (best stays iter 0 / wall None).
    assert state.best.iteration == 0
    assert state.best.wall_ms is None


def test_should_resume_only_when_commit_is_head():
    fresh = RunState()
    assert should_resume(fresh, "abc123") is False  # no recorded best

    state = RunState()
    state.best.commit_hash = "abc123"
    state.best.wall_ms = 0.5
    state.best.mean_case_speedup = 2.0
    assert should_resume(state, "abc123") is True
    assert should_resume(state, "def456") is False
    assert should_resume(state, "") is False

    no_wall = RunState()
    no_wall.best.commit_hash = "abc123"  # commit but never measured
    assert should_resume(no_wall, "abc123") is False


def test_supervisor_intervention_resets_the_cooldown_but_not_the_stall():
    """Advice is not a result, so only the cooldown window restarts.

    A run that has gone five iterations without a KEEP is exactly as stuck the
    moment after the supervisor answers as it was the moment before, and the
    phase label and the search-mode switch both read that fact.
    """
    from kernelforge.loop import run_state as run_state_module

    state = RunState()
    state.best.iteration = 2
    state.best.wall_ms = 0.8
    state.stall.no_improvement_iters = 5
    state.stall.unresolved_stall_iters = 5
    state.phase = PHASE_STALLED

    run_state_module.apply_supervisor_intervention(
        state,
        iteration=8,
        stall_threshold=5,
    )

    assert state.stall.no_improvement_iters == 0
    assert state.stall.unresolved_stall_iters == 5
    assert state.stall.last_supervisor_iter == 8
    assert state.phase == PHASE_STALLED


# ── schema guards ───────────────────────────────────────────────────────────────
def test_from_dict_rejects_payloads_that_are_not_the_current_shape():
    """A checkpoint is control state: a partial one must not load as defaults.

    Silently filling a missing field would resume with a fabricated anchor (a
    zeroed stall streak, an empty best) rather than the campaign's own.
    """
    valid = RunState().to_dict()

    with pytest.raises(ValueError, match="must be a JSON object"):
        RunState.from_dict(["not", "an", "object"])

    missing = dict(valid)
    missing.pop("stall")
    missing.pop("pinned_iterations")
    with pytest.raises(ValueError, match="missing fields: pinned_iterations, stall"):
        RunState.from_dict(missing)

    unknown = dict(valid)
    unknown["retired_control_field"] = 3
    with pytest.raises(ValueError, match="unknown fields: retired_control_field"):
        RunState.from_dict(unknown)


def test_from_dict_rejects_nested_records_that_are_not_the_current_shape():
    """The nested records carry the anchors, so they get the same guard."""
    valid = RunState().to_dict()

    not_object = dict(valid, best="abc1234")
    with pytest.raises(ValueError, match="run state best must be an object"):
        RunState.from_dict(not_object)

    incomplete = dict(valid)
    incomplete["stall"] = {"no_improvement_iters": 2}
    with pytest.raises(
        ValueError,
        match=("stall missing fields: last_supervisor_attempt_iter, last_supervisor_iter, unresolved_stall_iters"),
    ):
        RunState.from_dict(incomplete)

    extra = dict(valid)
    extra["cumulative"] = dict(valid["cumulative"], skipped=1)
    with pytest.raises(ValueError, match="cumulative has unknown fields: skipped"):
        RunState.from_dict(extra)


def test_from_dict_rejects_control_values_outside_their_domains():
    """Shape alone is not enough: an out-of-domain value is still unresumable."""
    valid = RunState().to_dict()

    with pytest.raises(ValueError, match="unsupported search mode"):
        RunState.from_dict(dict(valid, search_mode="TURBO"))

    with pytest.raises(ValueError, match="unsupported orchestration circuit state"):
        RunState.from_dict(dict(valid, orchestration_circuit_state="tripped"))

    # next_iteration is the cursor apply_iteration refuses to go behind; a zero
    # would let iteration 0 be replayed as fresh work.
    with pytest.raises(ValueError, match="next_iteration must be positive"):
        RunState.from_dict(dict(valid, next_iteration=0))


# ── session guards ──────────────────────────────────────────────────────────────
def test_start_session_refuses_foreign_and_completed_campaigns():
    state = RunState()
    start_session(state, campaign_id="campaign-123")
    finish_session(state, status=SESSION_PAUSED, reason="iteration_budget")

    with pytest.raises(ValueError, match="campaign mismatch"):
        start_session(state, campaign_id="campaign-999")
    # A rejected start must not consume a session slot or revive the campaign.
    assert state.session_index == 1
    assert state.session_status == SESSION_PAUSED

    start_session(state, campaign_id="campaign-123")
    finish_session(state, status=SESSION_COMPLETED, reason="target_met")
    with pytest.raises(ValueError, match="completed campaign"):
        start_session(state, campaign_id="campaign-123")
    assert state.session_index == 2
    assert state.session_status == SESSION_COMPLETED


def test_finish_session_requires_a_running_session_and_a_terminal_status():
    fresh = RunState()
    with pytest.raises(ValueError, match="no running session to finish"):
        finish_session(fresh, status=SESSION_PAUSED)

    state = RunState()
    start_session(state, campaign_id="campaign-123")
    with pytest.raises(ValueError, match="invalid terminal session status"):
        finish_session(state, status=SESSION_RUNNING, reason="still going")
    # The rejected transition leaves the session running, not half-finished.
    assert state.session_status == SESSION_RUNNING
    assert state.termination_reason == ""


def test_orchestration_probe_transitions_are_guarded_in_both_directions():
    """A probe is a single deliberate step out of an open circuit.

    Closing straight from open would clear the streak without a call ever
    succeeding, and probing a circuit that is not open would report a recovery
    that never happened.
    """
    closed = RunState()
    with pytest.raises(ValueError, match="only an open orchestration circuit"):
        begin_orchestration_probe(closed)
    assert closed.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_CLOSED

    opened = RunState(
        orchestration_error_streak=3,
        orchestration_circuit_state=ORCHESTRATION_CIRCUIT_OPEN,
    )
    with pytest.raises(ValueError, match="cannot complete a probe"):
        complete_orchestration_probe(opened)
    assert opened.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_OPEN
    assert opened.orchestration_error_streak == 3


def test_pin_iteration_dedupes_without_refreshing_recency():
    """Re-pinning an iteration is a no-op, not a bump to the head of the list.

    Eviction is by age, so treating a repeat pin as new would let one iteration
    that keeps coming up push the rest of the lineage out of the map.
    """
    state = RunState()
    for iteration in (3, 4, 5):
        pin_iteration(state, iteration)

    pin_iteration(state, 3)

    assert state.pinned_iterations == [3, 4, 5]


# ── store degradation ───────────────────────────────────────────────────────────
def test_workspace_lock_reacquire_and_release_are_idempotent(tmp_path):
    """The loop takes the lock once per session but may unwind it more than once.

    A second acquire must hand back the same held lock rather than block on the
    handle this process already owns, and a second release must not touch a lock
    the next session may already hold.
    """
    store = LoopStateStore(str(tmp_path))
    lock = store.workspace_lock()

    assert lock.acquire() is lock
    assert lock.acquire() is lock
    assert f"pid={os.getpid()}" in store.lock_path.read_text()

    lock.release()
    lock.release()

    # Released for real: another owner can take it.
    with LoopStateStore(str(tmp_path)).workspace_lock():
        pass


def test_store_construction_degrades_instead_of_raising(tmp_path, monkeypatch):
    """Persistence is best-effort: a broken workspace must not abort the loop."""

    def _boom(*args, **kwargs):
        raise OSError("read-only file system")

    with monkeypatch.context() as patched:
        patched.setattr(Path, "mkdir", _boom)
        store = LoopStateStore(str(tmp_path))
    assert store.degraded is True
    assert any("create root" in message for message in store.persistence_errors)

    with monkeypatch.context() as patched:
        patched.setattr(LoopStateStore, "read_events", _boom)
        primed = LoopStateStore(str(tmp_path))
    assert primed.degraded is True
    assert any("prime recent cache" in message for message in primed.persistence_errors)
    # A store that could not prime still answers, with an empty window.
    assert primed.recent_events(5) == []


def test_write_failures_degrade_the_store_but_still_feed_the_prompt_view(tmp_path):
    """Every write is best-effort, and the in-memory tails are updated first.

    An iteration whose disk append failed is still an iteration the next prompt
    must describe, so the cached view carries it even though events.jsonl never
    took it.
    """
    store = LoopStateStore(str(tmp_path))
    # Rename/open onto a directory fails, without depending on file modes.
    store.state_path.mkdir()
    store.events_path.mkdir()
    event = make_event("iteration_result", 1, decision="KEEP", wall_ms=0.5)

    store.save(RunState())
    store.append_event(event)

    assert store.degraded is True
    assert any("save" in message for message in store.persistence_errors)
    assert any("append" in message for message in store.persistence_errors)
    # A failed save leaves no half-written checkpoint behind.
    assert list(store.root.glob(".run_state.*.tmp")) == []
    assert store.recent_events(1) == [event]
    assert store.recent_results(1) == [event]

    # An unreadable log degrades rather than raising, and reads as empty.
    assert store.read_events() == []
    assert any("read" in message for message in store.persistence_errors)


def test_read_events_skips_blank_lines_and_windows_refuse_nonpositive_counts(tmp_path):
    root = tmp_path / "forge_experiments"
    root.mkdir()
    first = make_event("iteration_result", 1, decision="KEEP")
    second = make_event("iteration_result", 2, decision="REVERT_PERF")
    (root / "events.jsonl").write_text("\n".join(["", json.dumps(first), "   ", "", json.dumps(second), ""]) + "\n")

    store = LoopStateStore(str(tmp_path))

    assert store.read_events() == [first, second]
    # A window of zero (or less) is an empty window, never the whole tail.
    assert store.recent_events(0) == []
    assert store.recent_events(-1) == []
    assert store.recent_results(0) == []
    assert store.recent_results(-3) == []
