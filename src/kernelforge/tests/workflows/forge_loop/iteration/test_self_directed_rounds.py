# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A loop without a planning service: the session chooses its direction, and no round is priced or probed for it."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.workflows.forge_loop.campaign.run_state import (
    ORCHESTRATION_CIRCUIT_CLOSED,
    ORCHESTRATION_CIRCUIT_OPEN,
    LoopStateStore,
)
from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.iteration.runner import IterationLoop, IterationResult
from kernelforge.workflows.forge_loop.policy.round_budget import (
    ADMISSION_SESSION_SEC,
    DISPATCH_FLOOR_SEC,
    DISPATCH_SESSION_SEC,
    FIRST_ROUND_MEASUREMENT_SEC,
    PLANNING_FLOOR_SEC,
)
from kernelforge.workflows.forge_loop.search.agent_relay import (
    AgentAvailabilitySnapshot,
    AgentProfile,
    AgentRelayConfig,
    ProfileAvailability,
    ProviderAvailability,
)
from kernelforge.workflows.forge_loop.search.search_actors import SearchActorBundle, SearchActorPool
from kernelforge.workflows.forge_loop.search.search_policy import (
    OBJECTIVE_DISCOVER_NEW_MECHANISM,
    SEARCH_MODE_DIVERSIFY,
    SEARCH_MODE_EXPLOIT,
    SearchPolicyDecision,
    stall_thresholds,
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
    assert (
        "Choose the direction with the largest expected gain on the score, a structural change included."
        in histories[0]
    )
    assert "## Required optimization plan" not in histories[0]


def test_a_stalled_search_sends_the_session_to_a_different_mechanism_and_leaves_the_cycle_open(tmp_path, monkeypatch):
    """No plan was published, so nothing returns the search to exploiting until a session produces a new best."""
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
    assert LoopStateStore(str(workspace)).load().diversification_cycle_completed is False


def _relay_with_one_runnable_profile(agent, supervisor) -> SearchActorPool:
    """A campaign's relay in a session where only its primary passed the startup check, so a blockage is reviewed."""
    primary = AgentProfile("claude", "claude", "claude-opus-5")
    other = AgentProfile("codex", "codex", "gpt-6-astra")
    passed = ProfileAvailability("claude", primary.model, provider_available=True, model_available=True, available=True)
    failed = ProfileAvailability("codex", other.model, reason="startup check timed out")
    return SearchActorPool(
        config=AgentRelayConfig(primary_profile_id=primary.profile_id, profiles=(primary, other)),
        availability=AgentAvailabilitySnapshot(
            checked_at="2026-11-22T05:55:06Z",
            session_index=1,
            providers={"claude": ProviderAvailability(available=True), "codex": ProviderAvailability()},
            profiles={primary.profile_id: passed, other.profile_id: failed},
        ),
        bundles={primary.profile_id: SearchActorBundle(profile=primary, agent_fn=agent, supervisor_fn=supervisor)},
    )


@pytest.mark.parametrize("relay", [False, True], ids=["no relay", "relay with one runnable profile"])
def test_a_self_directed_loop_turns_after_one_session_without_a_new_best_and_is_reviewed_after_two(
    tmp_path, monkeypatch, relay
):
    """The stall lasts until a new best: every session in it is sent to a different mechanism, the third under a ruling.

    The review comes at the same session whether the loop schedules the Supervisor itself or reaches it through a
    relay that has no other profile to switch to.
    """
    thresholds = stall_thresholds(planned_rounds=False)
    loop, workspace = _make_loop(tmp_path, monkeypatch, supervise_after=thresholds.supervise_after, session_count=4)
    loop.ic.diversify_after = thresholds.diversify_after
    loop.config = SimpleNamespace(experiments_dir=workspace / "forge_experiments", gpu_target="gfx942")
    histories: list[str] = []
    modes: list[str] = []
    reviewed_at: list[int] = []

    async def supervisor(**kwargs):
        reviewed_at.append(kwargs["iteration"])
        return "\n# Supervisor Ruling\n\nRestructure the reduction.\n"

    async def agent(_kernel_path, history, session_sink):
        histories.append(history)
        modes.append(loop._search_policy_decision.mode)
        session_sink["plan"] = "inspect only"
        return "No source change was needed."

    search_actors = _relay_with_one_runnable_profile(agent, supervisor) if relay else None
    asyncio.run(
        loop.run(runner_module.LoopAgents(agent_fn=agent, supervisor_fn=supervisor, search_actors=search_actors))
    )

    assert modes == [SEARCH_MODE_EXPLOIT] + [SEARCH_MODE_DIVERSIFY] * 3
    assert reviewed_at == [3]
    assert "The search has gone 1 session(s) without a new best" in histories[1]
    for under_ruling in histories[2:]:
        assert "## Latest Supervisor Ruling" in under_ruling
        assert "take one of its directions that no session since the ruling has measured" in under_ruling


def test_a_session_under_a_supervisor_ruling_takes_its_search_mode_from_the_ruling(tmp_path, monkeypatch):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    loop._supervisor_ruling = "Stop tuning the tile size; restructure the reduction."
    loop._search_policy_decision = None

    direction = loop._session_direction("")

    assert "The latest Supervisor Ruling below is the planning authority while the search is stalled" in direction
    assert "take one of its directions that no session since the ruling has measured" in direction
    assert "largest expected gain on the score" not in direction


def test_a_stalled_search_without_planning_hands_the_supervisors_ruling_to_the_next_session(tmp_path, monkeypatch):
    """With no plan to write the ruling into, the session that follows reads it and is told it decides the mode."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, supervise_after=1, session_count=3)
    loop.ic.supervise_cooldown = 2
    loop.config = SimpleNamespace(experiments_dir=workspace / "forge_experiments", gpu_target="gfx942")
    histories: list[str] = []
    evidence: list[dict] = []
    ruling = "\n# Supervisor Ruling\n\nStop tuning tile sizes; restructure the K loop.\n"

    async def supervisor(**kwargs):
        evidence.append(json.loads(kwargs["evidence_context"]))
        return ruling

    async def editing_agent(kernel_path, history, session_sink):
        histories.append(history)
        session_sink["plan"] = "tune tiles"
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        return "another tile sweep"

    async def reverted_iteration(iteration, plan=""):
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="passed",
            wall_ms=1.1,
            mean_case_speedup=0.9,
            kept=False,
            bench_detail={"case_times": {"case": 1.1}},
        )

    monkeypatch.setattr(loop, "run_one_iteration", reverted_iteration)
    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=editing_agent, supervisor_fn=supervisor)))

    assert len(histories) == 3
    assert len(evidence) == 1
    # The Supervisor is shown that there is no plan rather than a stale path to one.
    assert evidence[0]["latest_optimization_plan"] is None
    for later in histories[1:]:
        assert later.startswith("## Direction (yours to choose)")
        assert "The latest Supervisor Ruling below is the planning authority" in later
        assert "## Latest Supervisor Ruling" in later
        assert "restructure the K loop" in later


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
