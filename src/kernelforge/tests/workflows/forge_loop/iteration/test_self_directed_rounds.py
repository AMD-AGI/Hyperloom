# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A loop without a planning service: the session chooses its direction, and no round is priced or probed for it."""

from __future__ import annotations

import asyncio

from kernelforge.workflows.forge_loop.campaign.run_state import (
    ORCHESTRATION_CIRCUIT_CLOSED,
    ORCHESTRATION_CIRCUIT_OPEN,
    LoopStateStore,
)
from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.iteration.runner import IterationLoop
from kernelforge.workflows.forge_loop.policy.round_budget import (
    ADMISSION_SESSION_SEC,
    DISPATCH_FLOOR_SEC,
    DISPATCH_SESSION_SEC,
    FIRST_ROUND_MEASUREMENT_SEC,
    PLANNING_FLOOR_SEC,
)
from kernelforge.workflows.forge_loop.search.search_policy import (
    OBJECTIVE_DISCOVER_NEW_MECHANISM,
    SEARCH_MODE_DIVERSIFY,
    SearchPolicyDecision,
)
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _AMPLE_BUDGET_SEC,
    _make_loop,
    _no_change_agent,
    _unused_supervisor,
)


def _recording_agent(histories: list[str]):
    async def agent(_kernel_path, history, session_sink):
        histories.append(history)
        session_sink["plan"] = "inspect only"
        return "No source change was needed."

    return agent


def test_a_session_without_a_plan_is_told_to_choose_its_direction_from_the_evidence(tmp_path, monkeypatch):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    histories: list[str] = []

    asyncio.run(
        loop.run(runner_module.LoopAgents(agent_fn=_recording_agent(histories), supervisor_fn=_unused_supervisor))
    )

    assert len(histories) == 1
    assert histories[0].startswith("## Direction (yours to choose)")
    assert "Read the Analysis report named `analysis_summary`" in histories[0]
    assert "do not repeat a direction they disproved" in histories[0]
    assert "Choose the direction with the strongest immediate gain on the score." in histories[0]
    assert "## Required optimization plan" not in histories[0]


def test_a_stalled_search_sends_the_session_to_a_different_mechanism_and_counts_it_as_the_diversified_round(
    tmp_path, monkeypatch
):
    """The session is the round's plan, so having run it is what returns the search to exploiting."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    histories: list[str] = []
    monkeypatch.setattr(
        loop.search_policy_engine,
        "decide",
        lambda *_args, **_kwargs: SearchPolicyDecision(
            mode=SEARCH_MODE_DIVERSIFY,
            reason_codes=("NO_IMPROVEMENT_STALL",),
            objective_kind=OBJECTIVE_DISCOVER_NEW_MECHANISM,
        ),
    )

    asyncio.run(
        loop.run(runner_module.LoopAgents(agent_fn=_recording_agent(histories), supervisor_fn=_unused_supervisor))
    )

    assert "choose a mechanism materially different from the ones already tried" in histories[0]
    assert LoopStateStore(str(workspace)).load().diversification_cycle_completed is True


def test_a_session_under_a_supervisor_ruling_takes_its_search_mode_from_the_ruling(tmp_path, monkeypatch):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    loop._supervisor_ruling = "Stop tuning the tile size; restructure the reduction."
    loop._search_policy_decision = None

    direction = loop._session_direction("")

    assert "The latest Supervisor Ruling below decides" in direction
    assert "strongest immediate gain on the score" not in direction


def test_a_loop_without_planning_is_not_refused_for_the_price_of_a_plan(tmp_path, monkeypatch):
    """What remains pays for a session and its measurement, but not for the planning a planned round would add."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    planned_round_price = PLANNING_FLOOR_SEC + ADMISSION_SESSION_SEC + FIRST_ROUND_MEASUREMENT_SEC
    remaining = planned_round_price - 1.0
    assert remaining >= max(
        loop.ic.budget_reserve_sec, DISPATCH_FLOOR_SEC, DISPATCH_SESSION_SEC + FIRST_ROUND_MEASUREMENT_SEC
    )
    monkeypatch.setattr(loop, "_time_remaining", lambda: remaining if not loop.results else 0.0)
    histories: list[str] = []

    asyncio.run(
        loop.run(runner_module.LoopAgents(agent_fn=_recording_agent(histories), supervisor_fn=_unused_supervisor))
    )

    assert len(histories) == 1
    state = LoopStateStore(str(workspace)).load()
    assert state.termination_reason == "budget_exhausted"
    assert state.round_costs.rounds == 0


def test_a_campaign_stopped_on_failed_orchestration_resumes_without_a_planner(tmp_path, monkeypatch):
    """Nothing is left for the open circuit to guard, so the resume closes it and runs its session."""
    first, workspace = _make_loop(tmp_path, monkeypatch, session_count=5)

    async def failed_planning(**_kwargs):
        return None, "synthesis failed"

    monkeypatch.setattr(first, "_run_orchestration", failed_planning)
    asyncio.run(
        first.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent, orchestration_service=object(), supervisor_fn=_unused_supervisor
            )
        )
    )
    assert LoopStateStore(str(workspace)).load().orchestration_circuit_state == ORCHESTRATION_CIRCUIT_OPEN

    resumed = IterationLoop(first.ic, first.tracker, config=first.config, resume=True)
    monkeypatch.setattr(resumed, "_time_remaining", lambda: _AMPLE_BUDGET_SEC if not resumed.results else 0.0)
    histories: list[str] = []
    asyncio.run(
        resumed.run(runner_module.LoopAgents(agent_fn=_recording_agent(histories), supervisor_fn=_unused_supervisor))
    )

    state = LoopStateStore(str(workspace)).load()
    assert len(histories) == 1
    assert state.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_CLOSED
    assert state.orchestration_error_streak == 0
    assert state.termination_reason == "budget_exhausted"
    events = [event["type"] for event in LoopStateStore(str(workspace)).read_events()]
    assert "orchestration_circuit_retired" in events
    assert "orchestration_circuit_half_open" not in events
