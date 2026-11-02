# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Orchestration sees what a queued or running explore round holds and how far it has got.

A proposal in a round's grid is no longer waiting for a bench, so the untested queue and its backlog count leave it
out, and the round's task line names its variants and the latest progress of each.
"""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.actions.executors._proposal_identity import controls_of
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.machine_state import PHASE_FRAMEWORK_AGENT
from hyperloom.orchestrator.roles import Backend, MockBackend, ScriptedPlan
from hyperloom.orchestrator.state.task_registry import TaskSpec


@pytest.fixture
def coord(session_dir) -> Coordinator:
    plan = ScriptedPlan(
        turns=[],
        default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"}),
    )
    backends: dict[str, Backend] = {name: MockBackend(plan, name=name) for name in ("orchestration", "critic")}
    return Coordinator(session_dir, backends=backends)


def _queue_two_proposals(coord: Coordinator) -> None:
    coord.shared_state.phase = PHASE_FRAMEWORK_AGENT
    coord.shared_state.macro_cycle = 0
    coord.shared_state.specialist_rounds = [
        {
            "cycle": 0,
            "domain": "serving_specialist",
            "proposal_set": [
                {"name": "in-round", "extra_args": "--a 1"},
                {"name": "still-waiting", "extra_args": "--b 2"},
            ],
        }
    ]


def _grid_of(rows: list[dict]) -> list[dict]:
    """The grid the Coordinator's auto-bench builds from untested rows."""
    return [
        {
            "name": row["name"],
            "extra_args": row["extra_args"],
            "extra_envs": dict(row["extra_envs"]),
            **controls_of(row),
        }
        for row in rows
    ]


async def _running_round(coord: Coordinator, grid: list[dict]):
    task, _ = await coord.tasks.create_or_return_existing(
        TaskSpec(kind="explore", params={"grid": grid}, idempotency_key="auto-explore-c0-test")
    )
    await coord.tasks.transition(task.task_id, "running")
    return task


@pytest.mark.asyncio
async def test_a_proposal_in_a_running_round_is_named_not_offered_as_untested(coord):
    _queue_two_proposals(coord)
    in_round = [row for row in coord.shared_state.untested_proposal_rows() if row["name"] == "in-round"]
    await _running_round(coord, _grid_of(in_round))

    out = (await coord.conversation.compose_prompt("orchestration")).text
    queue = out.split("=== Untested proposals (current cycle) ===", 1)[1]

    offered = [line.split()[1] for line in queue.splitlines() if line.startswith("• ")]
    assert offered == ["still-waiting"]
    assert "already in a queued or running explore grid" in queue and "in-round" in queue
    assert "1 untested proposals" in coord.conversation._context_running_tasks_reader()


@pytest.mark.asyncio
async def test_a_running_round_names_its_variants_and_their_latest_progress(coord):
    grid = [{"name": "first"}, {"name": "second"}, {"name": "third"}]
    task = await _running_round(coord, grid)
    await coord.tasks.record_progress(task.task_id, {"unit": "variant", "label": "first", "status": "started"})
    await coord.tasks.record_progress(
        task.task_id, {"unit": "variant", "label": "first", "status": "succeeded", "output_throughput": 1290.6}
    )
    await coord.tasks.record_progress(task.task_id, {"unit": "variant", "label": "second", "status": "started"})

    line = next(
        line for line in coord.conversation._context_running_tasks_reader().splitlines() if task.task_id in line
    )

    assert "grid_as_dispatched=['first', 'second', 'third']" in line
    assert "variants=[first: succeeded 1290.6 tok/s, second: started]" in line
    assert "last_progress_age_sec=" in line


@pytest.mark.asyncio
async def test_a_round_with_no_progress_yet_says_so(coord):
    task = await _running_round(coord, [{"name": "only"}])

    line = next(
        line for line in coord.conversation._context_running_tasks_reader().splitlines() if task.task_id in line
    )

    assert "grid_as_dispatched=['only']" in line
    assert "variants=[]" in line
    assert "last_progress_age_sec" not in line
