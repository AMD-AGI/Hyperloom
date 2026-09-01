# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The loop driven end to end over mock backends.

Every other test in this tree exercises one rule with the state it needs
already arranged. That is what let a phase ship with a zero budget, a
discovery lane that re-fetched one finished task forever, and an exit rung
nothing could reach: each defect was invisible to a test that set up the
condition the rule reads, and fatal to a run that had to arrive at it.

These two walk the machine instead and assert where it ends up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.phases import machine_state as ps


def _coordinator(session_dir: Path):
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        MockRobustnessBackend,
        ScriptedPlan,
    )

    # Orchestration says nothing: what the phases do on their own is the point.
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
    return Coordinator(
        session_dir,
        backends={
            "orchestration": MockBackend(silent, name="orch"),
            "critic": MockCriticBackend(),
            "robustness": MockRobustnessBackend(),
        },
    )


@pytest.fixture
def session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _chain(state: Any) -> list[tuple[str, str, str]]:
    return [
        (str(row.get("from_phase") or ""), str(row.get("to_phase") or ""), str(row.get("reason") or ""))
        for row in state.phase_history
    ]


@pytest.mark.asyncio
async def test_a_baseline_carries_the_run_into_the_optimisation_phase_with_work(session_dir: Path):
    """Entering the phase must produce dispatches, not just a history row.

    The phase held a 0.0 budget share and left on its first tick; before that
    the discovery lane re-keyed every request identically and re-fetched one
    finished task on every tick. Both read as "the phase ran" in a history row
    and as nothing at all in the task registry.

    No backend runs specialists here, so what the arms produce is a dispatch,
    not a result. That is the whole assertion: the empty-discovery streak is
    deliberately not read, because in this harness the round fails to run
    rather than running and finding nothing.
    """
    coord = _coordinator(session_dir)
    try:
        coord.shared_state.baseline_tput = 1500.0
        coord.shared_state.max_minutes = 180
        coord.shared_state.save(session_dir)

        for tick in range(1, 25):
            await coord.tick(tick)

        state = coord.shared_state
        assert state.phase == ps.PHASE_FRAMEWORK_AGENT
        assert state.phase_budget_pct[ps.PHASE_FRAMEWORK_AGENT] > 0.0
        queued = await coord.tasks.queued()
        running = await coord.tasks.running()
        kinds = {str((t.params or {}).get("task_kind") or "") for t in (*queued, *running)}
        assert kinds & {"candidate_discovery", "framework_local_explore"}
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_both_arms_dry_walks_the_rest_of_the_chain(session_dir: Path):
    """With nothing left to try, the run reaches CLOSE through every phase."""
    coord = _coordinator(session_dir)
    try:
        state = coord.shared_state
        state.baseline_tput = 1500.0
        state.max_minutes = 180
        # Source arm: no local exploration, and discovery past its retries.
        state.framework_local_explore_enabled = False
        state.framework_agent_empty_discoveries = 99
        # Config arm: trailing winners below the gain floor, rounds producing
        # nothing.
        state.explore_search = {"winners_history": [{"gain_pct": 0.01, "cycle": 0} for _ in range(6)]}
        state.specialist_rounds = [{"proposals_total": 0, "proposals_kept": 0, "cycle": 0} for _ in range(6)]
        state.save(session_dir)

        for tick in range(1, 12):
            await coord.tick(tick)

        assert state.phase == ps.PHASE_CLOSE
        visited = [to_phase for _, to_phase, _ in _chain(state)]
        assert visited[:2] == [ps.PHASE_PRELUDE, ps.PHASE_FRAMEWORK_AGENT]
        assert visited[-3:] == [ps.PHASE_KERNEL_AGENT, ps.PHASE_SWEEP, ps.PHASE_CLOSE]
        reasons = {reason for _, _, reason in _chain(state)}
        assert "optimize_no_more_leverage" in reasons
        for reason in reasons:
            assert ps.is_valid_phase_exit_reason(reason), reason
    finally:
        await coord.stop()
