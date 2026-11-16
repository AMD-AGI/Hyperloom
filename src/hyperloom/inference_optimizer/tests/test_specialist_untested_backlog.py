# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A new FRAMEWORK_AGENT specialist waits while two automatic grids' worth of proposals wait for a bench."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.actions.executors._proposal_identity import controls_of
from hyperloom.orchestrator.loop._untested_backlog import SPECIALIST_UNTESTED_BACKLOG_FULL, untested_backlog_denial
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.proposal_bench import UNTESTED_BACKLOG_CAP, ProposalBench
from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskSpec


@pytest.fixture
async def coord(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    silent = ScriptedPlan(turns=[])
    c = Coordinator(
        make_session_dir(),
        backends={"orchestration": MockBackend(silent, name="o"), "critic": MockBackend(silent, name="c")},
    )
    c.shared_state.phase = "FRAMEWORK_AGENT"
    yield c
    await c.stop()


def _round(count: int, *, prefix: str = "v") -> dict[str, Any]:
    return {
        "cycle": 0,
        "domain": "serving_specialist",
        "proposal_set": [{"name": f"{prefix}{i}", "extra_args": f"--{prefix}{i} 1"} for i in range(count)],
    }


def _grid_of(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "extra_args": row["extra_args"],
            "extra_envs": dict(row["extra_envs"]),
            **controls_of(row),
        }
        for row in rows
    ]


async def _grid(c: Coordinator, rows: list[dict[str, Any]], *, key: str, state: str) -> None:
    task, _ = await c.tasks.create_or_return_existing(
        TaskSpec(kind="explore", params={"grid": _grid_of(rows)}, idempotency_key=key)
    )
    await c.tasks.transition(task.task_id, "running")
    if state != "running":
        await c.tasks.transition(task.task_id, state)


async def _delegate(c: Coordinator) -> None:
    payload = {
        "action_name": "specialist",
        "params": {"mode": "research", "scope": "freeform", "task_description": "x"},
    }
    await c.router.handle_delegate("orchestration", Intent(type=IntentType.DELEGATE, payload=payload))


async def _queued_specialists(c: Coordinator) -> int:
    return sum(t.kind == "specialist" for t in await c.tasks.by_state("queued"))


async def _backlog_denials(c: Coordinator) -> int:
    observed = await c.bus.tail(topic="observation")
    return sum(
        m.payload.get("kind") == "policy_denied" and m.payload.get("rule") == SPECIALIST_UNTESTED_BACKLOG_FULL
        for m in observed
    )


def test_the_cap_is_two_automatic_grids():
    assert UNTESTED_BACKLOG_CAP == 2 * ProposalBench._AUTO_EXPLORE_GRID_CAP


async def test_a_specialist_is_refused_while_the_cap_of_proposals_waits(coord):
    coord.shared_state.specialist_rounds = [_round(UNTESTED_BACKLOG_CAP)]

    await _delegate(coord)

    assert await _queued_specialists(coord) == 0
    assert await _backlog_denials(coord) == 1


async def test_a_specialist_is_admitted_below_the_cap(coord):
    coord.shared_state.specialist_rounds = [_round(UNTESTED_BACKLOG_CAP - 1)]

    await _delegate(coord)

    assert await _queued_specialists(coord) == 1
    assert await _backlog_denials(coord) == 0


@pytest.mark.parametrize(
    ("key", "state"),
    [("orchestration-grid", "running"), ("auto-explore-c0-ended", "failed")],
    ids=["in_a_running_grid", "carried_by_a_finished_automatic_grid"],
)
async def test_a_proposal_no_longer_waiting_for_a_bench_does_not_count(coord, key, state):
    coord.shared_state.specialist_rounds = [_round(UNTESTED_BACKLOG_CAP)]
    await _grid(coord, coord.shared_state.untested_proposal_rows()[:1], key=key, state=state)

    await _delegate(coord)

    assert await _queued_specialists(coord) == 1
    assert await _backlog_denials(coord) == 0


async def test_a_specialist_landing_at_the_cap_still_records_every_proposal_it_returns(coord):
    coord.shared_state.specialist_rounds = [_round(UNTESTED_BACKLOG_CAP)]

    coord.shared_state.record_specialist_round(_round(4, prefix="late"))
    await _delegate(coord)

    assert len(coord.shared_state.untested_proposal_rows()) == UNTESTED_BACKLOG_CAP + 4
    assert await _backlog_denials(coord) == 1


def _full_state(phase: str = "FRAMEWORK_AGENT") -> SharedState:
    state = SharedState()
    state.phase = phase
    state.specialist_rounds = [_round(UNTESTED_BACKLOG_CAP)]
    return state


_NO_TASKS = SimpleNamespace(
    queued_context_sync=list, running_context_sync=list, with_key_prefix_sync=lambda kind, key_prefix: []
)


@pytest.mark.parametrize(
    ("action_name", "params", "phase"),
    [
        ("explore", {"grid": []}, "FRAMEWORK_AGENT"),
        ("specialist", {"mode": "research"}, "ENABLEMENT"),
        ("specialist", {"domain": "custom_specialist"}, "FRAMEWORK_AGENT"),
    ],
    ids=["another_action", "outside_framework_agent", "operator_custom_specialist"],
)
def test_the_cap_holds_back_only_a_framework_agent_specialist_orchestration_may_dispatch(action_name, params, phase):
    assert untested_backlog_denial(action_name, params, _full_state(phase), _NO_TASKS) is None
    assert untested_backlog_denial("specialist", {"mode": "research"}, _full_state(), _NO_TASKS) is not None


def test_orchestration_is_told_the_cap_and_the_rule():
    from hyperloom.orchestrator.prompts import prompt_builder

    md = (Path(prompt_builder.__file__).parent / "orchestration.md").read_text(encoding="utf-8")

    assert f"While {UNTESTED_BACKLOG_CAP} untested proposals" in md
    assert SPECIALIST_UNTESTED_BACKLOG_FULL in md
