# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Coordinator's forced specialist for a stalled domain is held, not refused, while an admission gate would deny it."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.proposal_bench import UNTESTED_BACKLOG_CAP
from hyperloom.orchestrator.specialists.dispatch import SpecialistDispatchCollaborator
from hyperloom.orchestrator.tests.mock_backend import MockBackend, ScriptedPlan


@pytest.fixture
async def coord(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    # The real warmup stamps the session's framework onto every dispatch.
    monkeypatch.setattr(
        SpecialistDispatchCollaborator,
        "warm_specialist_params",
        AsyncMock(side_effect=lambda params: params.setdefault("framework", "sglang")),
    )
    silent = ScriptedPlan(turns=[])
    c = Coordinator(
        make_session_dir(),
        backends={"orchestration": MockBackend(silent, name="o"), "critic": MockBackend(silent, name="c")},
    )
    state = c.shared_state
    state.phase = "FRAMEWORK_AGENT"
    source_root = tmp_path / "framework"
    (source_root / ".git").mkdir(parents=True)
    state.framework_repo_path = str(source_root)
    # serving_specialist (anchor framework) idles past the force threshold with a gap pending.
    for _ in range(10):
        state.bump_domain_round_counters()
    state.upsert_gap({"canonical_id": "gap.framework.scheduler.s1", "domain_hint": "serving_specialist"})
    yield c
    await c.stop()


def _out_of_session_time(c: Coordinator) -> None:
    c.shared_state.max_minutes = 60
    c.shared_state.elapsed_minutes = lambda **_kw: 59.0  # type: ignore[method-assign]


def _backlog_full(c: Coordinator) -> None:
    c.shared_state.specialist_rounds = [
        {
            "cycle": 0,
            "domain": "serving_specialist",
            "proposal_set": [{"name": f"v{i}", "extra_args": f"--v{i} 1"} for i in range(UNTESTED_BACKLOG_CAP)],
        }
    ]


async def _queued_specialists(c: Coordinator) -> int:
    return sum(t.kind == "specialist" for t in await c.tasks.by_state("queued"))


async def _policy_denials(c: Coordinator) -> list[dict[str, Any]]:
    observed = await c.bus.tail(topic="observation")
    return [m.payload for m in observed if m.payload.get("kind") == "policy_denied"]


@pytest.mark.parametrize(
    ("deny", "rule"),
    [(_out_of_session_time, "time_budget"), (_backlog_full, "specialist_untested_backlog_full")],
    ids=["time_budget", "untested_backlog_full"],
)
async def test_a_force_an_admission_gate_would_refuse_is_held_every_tick_without_a_denial(coord, caplog, deny, rule):
    deny(coord)
    assert (
        coord.dispatcher.admission_denial_for_action("specialist", params={"domain": "serving_specialist"}).rule == rule
    )

    with caplog.at_level(logging.INFO, logger="hyperloom.orchestrator.specialists.dispatch"):
        for _ in range(2):
            await coord.specialist_dispatch.maybe_force_stalled_domain_specialist()

    assert await _queued_specialists(coord) == 0
    assert await _policy_denials(coord) == []
    assert "stalled-domain force: dispatched" not in caplog.text


async def test_a_force_the_router_leaves_without_a_task_is_not_logged_as_dispatched(coord, caplog, monkeypatch):
    # A refusal the admission gates do not predict (the policy gate's own rules): the router creates no task.
    monkeypatch.setattr(coord.router, "handle_intent", AsyncMock())

    with caplog.at_level(logging.INFO, logger="hyperloom.orchestrator.specialists.dispatch"):
        await coord.specialist_dispatch.maybe_force_stalled_domain_specialist()

    coord.router.handle_intent.assert_awaited_once()
    assert await _queued_specialists(coord) == 0
    assert "stalled-domain force: dispatched" not in caplog.text


async def test_an_admitted_force_is_submitted_once_and_logged_as_dispatched(coord, caplog):
    with caplog.at_level(logging.INFO, logger="hyperloom.orchestrator.specialists.dispatch"):
        for _ in range(2):
            await coord.specialist_dispatch.maybe_force_stalled_domain_specialist()

    assert await _queued_specialists(coord) == 1
    assert await _policy_denials(coord) == []
    assert caplog.text.count("stalled-domain force: dispatched domain=serving_specialist") == 1
