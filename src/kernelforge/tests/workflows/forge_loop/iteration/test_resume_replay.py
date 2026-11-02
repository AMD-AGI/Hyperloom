# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resuming from the event log: every record replayed once, and classified as the loop classified it."""

from __future__ import annotations

import asyncio

from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.campaign.run_state import LoopStateStore, RunState, make_event
from kernelforge.workflows.forge_loop.iteration.runner import IterationLoop
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _AMPLE_BUDGET_SEC,
    _make_loop,
    _no_change_agent,
)


def test_resume_replays_nonkeep_event_ahead_of_state(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    store = LoopStateStore(str(workspace))
    state = RunState(
        campaign_id="campaign",
        iteration=1,
        next_iteration=2,
        baseline_wall_ms=1.0,
    )
    state.cumulative.iterations = 1
    store.save(state)
    store.append_event(
        make_event(
            "iteration_result",
            2,
            decision="REVERT_PERF",
            plan="larger tile",
            wall_ms=1.1,
            best_after_ms=1.0,
            diversification_cycle_completed=True,
        )
    )
    loop.state_store = store
    loop.run_state = state

    planned, _, _, _ = loop._plan_resume_recovery(state, None)
    loop.run_state = planned
    store.save(planned)

    replayed = store.load()
    assert replayed.iteration == 2
    assert replayed.next_iteration == 3
    assert replayed.cumulative.iterations == 2
    assert replayed.cumulative.reverted == 1
    assert replayed.stall.no_improvement_iters == 1
    assert replayed.diversification_cycle_completed is True


def test_resume_replay_pins_sub_threshold_gains_but_not_regressions(
    tmp_path,
    monkeypatch,
):
    """The replay path must classify rejected candidates exactly as the loop does."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    store = LoopStateStore(str(workspace))
    state = RunState(campaign_id="campaign", baseline_wall_ms=1.0)
    store.save(state)
    store.append_event(
        make_event(
            "iteration_result",
            1,
            decision="REVERT_PERF",
            plan="vectorize the epilogue stores",
            wall_ms=0.998,
            mean_case_speedup=1.001918,
            best_after_ms=1.0,
            best_after_mean_case_speedup=1.0,
        )
    )
    store.append_event(
        make_event(
            "iteration_result",
            2,
            decision="REVERT_PERF",
            plan="raise the tile size",
            wall_ms=1.02,
            mean_case_speedup=0.984,
            best_after_ms=1.0,
            best_after_mean_case_speedup=1.0,
        )
    )
    loop.state_store = store
    loop.run_state = state

    planned, _, _, _ = loop._plan_resume_recovery(state, None)

    assert planned.next_iteration == 3
    assert planned.pinned_iterations == [1]


def test_a_resume_without_a_pending_keep_replays_each_event_exactly_once(tmp_path, monkeypatch):
    """Each contiguous non-KEEP record is applied once, so a resume cannot double-count an iteration.

    The planner used to walk the forward window twice when no KEEP was pending;
    the second walk could only ever re-apply what the first had, so it is the
    count that pins the behaviour, not the number of walks.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    store = LoopStateStore(str(workspace))
    state = RunState(campaign_id="campaign", iteration=0, next_iteration=1, baseline_wall_ms=1.0)
    store.save(state)
    for iteration in (1, 2, 3):
        store.append_event(
            make_event(
                "iteration_result",
                iteration,
                decision="REVERT_PERF",
                plan=f"attempt {iteration}",
                wall_ms=1.1,
                best_after_ms=1.0,
            )
        )
    loop.state_store = store
    loop.run_state = state

    applied: list[int] = []
    original = loop._apply_replayed_non_keep

    def _counting(planned, event):
        applied.append(int(event["iter"]))
        return original(planned, event)

    monkeypatch.setattr(loop, "_apply_replayed_non_keep", _counting)
    planned, _, _, _ = loop._plan_resume_recovery(state, None)

    assert applied == [1, 2, 3]
    assert planned.next_iteration == 4
    assert planned.cumulative.iterations == 3


def test_resume_replay_advances_the_relay_streak_only_for_the_active_profiles_search(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    store = LoopStateStore(str(workspace))
    state = RunState(campaign_id="campaign", iteration=0, next_iteration=1, baseline_wall_ms=1.0)
    state.agent_relay.active_profile_id = "claude"
    store.save(state)
    for iteration, decision, producer in (
        (1, "REVERT_PERF", "claude"),
        (2, "REVERT_PERF", None),
        (3, "API_ERROR", "claude"),
        (4, "BUILD_FAILED", "claude"),
    ):
        store.append_event(
            make_event(
                "iteration_result",
                iteration,
                decision=decision,
                plan=f"attempt {iteration}",
                wall_ms=1.1,
                best_after_ms=1.0,
                agent_profile_id=producer,
            )
        )
    loop.state_store = store
    loop.run_state = state

    planned, _, _, _ = loop._plan_resume_recovery(state, None)

    assert planned.next_iteration == 5
    assert planned.agent_relay.active_no_improvement_streak == 2


def test_resume_advances_past_abruptly_started_event(tmp_path, monkeypatch):
    first, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(first.run(runner_module.LoopAgents(agent_fn=_no_change_agent)))
    store = LoopStateStore(str(workspace))
    state = store.load()
    interrupted_iteration = state.next_iteration
    store.append_event(
        make_event(
            "iteration_started",
            interrupted_iteration,
            phase=state.phase,
        )
    )

    resumed = IterationLoop(
        first.ic,
        first.tracker,
        config=object(),
        resume=True,
    )
    monkeypatch.setattr(
        resumed,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if len(resumed.results) < 1 else 0.0,
    )
    results = asyncio.run(resumed.run(runner_module.LoopAgents(agent_fn=_no_change_agent)))

    assert [result.iteration for result in results] == [interrupted_iteration + 1]
    assert LoopStateStore(str(workspace)).load().next_iteration == (interrupted_iteration + 2)
