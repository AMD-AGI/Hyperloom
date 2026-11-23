# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist_done bookkeeping tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
from hyperloom.orchestrator.policy.gate import SPECIALIST_FROM_AGENT_PREFIX
from hyperloom.orchestrator.state.shared_state import SharedState
from ._dispatch_helpers import pump_until_settled
from .conftest import use_fake_specialist_cli
from hyperloom.orchestrator.state.task_registry import TaskSpec


@dataclass
class _StubTask:
    """Task-shaped stub (only ``task_id`` and ``params`` are inspected)."""

    task_id: str
    kind: str = "specialist"
    params: dict[str, Any] = field(default_factory=dict)


class _StubSharedState(SharedState):
    """SharedState that records the bookkeeping calls instead of persisting them."""

    def __init__(self):
        super().__init__()
        self.specialist_rounds: list[dict[str, Any]] = []
        self.saved: int = 0

    def record_specialist_round(self, entry: dict[str, Any]) -> None:
        # Mirror SharedState's idempotence-on-round_id behaviour.
        round_id = str(entry.get("round_id") or "").strip()
        if round_id:
            for i, prev in enumerate(self.specialist_rounds):
                if str(prev.get("round_id") or "") == round_id:
                    self.specialist_rounds[i] = dict(entry)
                    return
        self.specialist_rounds.append(dict(entry))

    def save(self, _session_dir) -> None:
        self.saved += 1


class _StubTaskRegistry:
    """Minimal TaskRegistry stub; ``get`` returns the registered task or raises ``TaskNotFound``."""

    def __init__(self):
        self._tasks: dict[str, _StubTask] = {}

    def register(self, task: _StubTask) -> None:
        self._tasks[task.task_id] = task

    async def get(self, task_id: str) -> _StubTask:
        if task_id not in self._tasks:
            from hyperloom.orchestrator.state.task_registry import TaskNotFound

            raise TaskNotFound(f"task {task_id} not found")
        return self._tasks[task_id]


# Fixture: lean Coordinator stand-in
@pytest.fixture
def coord(tmp_path: Path):
    """Build a Coordinator via ``__new__`` with just enough attributes for specialist lifecycle methods."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _StubSharedState()
    c.tasks = _StubTaskRegistry()
    c.knowledge_plane = None
    c.knowledge_plane = KnowledgePlane(recipe_kb=None)
    c.bus = SimpleNamespace(record_observation=AsyncMock())
    return c


def _done_payload(
    *,
    domain: str = "serving_specialist",
    gap: str = "gap.attention.fp8_kv",
    proposals: list | None = None,
    no_proposals: bool = False,
    summary: str = "Specialist found candidate variants",
    confidence: float = 0.7,
) -> dict[str, Any]:
    if proposals is None:
        proposals = (
            []
            if no_proposals
            else [
                {
                    "variant_name": "max_seqs_512",
                    "extra_server_args": "--max-num-seqs 512",
                    "rationale": "moderate concurrency bump",
                },
            ]
        )
    return {
        "domain": domain,
        "gap_canonical_id": gap,
        "proposal_set": proposals,
        "summary": summary,
        "reason": "no_findings" if not proposals else "kb_evidence",
        "confidence": confidence,
        "new_findings": ["fp8 kv cache stable above bs=128"],
        "residual_questions": [],
    }


# 1. record_specialist_result — direct bookkeeping unit tests
@pytest.mark.asyncio
async def test_record_specialist_result_non_empty_proposal_set(coord):
    """Non-empty proposal_set: ledger +1 row, save called."""
    task = _StubTask(task_id="task-1", params={})
    coord.tasks.register(task)

    payload = _done_payload(domain="serving_specialist")
    await coord.specialist_dispatch.record_specialist_result(
        task=task,
        done_payload=payload,
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}task-1",
    )

    state: _StubSharedState = coord.shared_state
    assert len(state.specialist_rounds) == 1
    row = state.specialist_rounds[0]
    assert row["task_id"] == "task-1"
    assert row["domain"] == "serving_specialist"
    assert row["gap_canonical_id"] == "gap.attention.fp8_kv"
    assert row["proposals_total"] == 1
    assert row["round_id"] == "task-1"
    assert state.saved == 1


@pytest.mark.asyncio
async def test_record_specialist_result_enqueues_build_request(coord):
    task = _StubTask(task_id="build-spec", params={"enablement": True})
    coord.tasks.register(task)
    coord.enablement_build.maybe_enqueue_specialist_requested_build = AsyncMock()
    payload = _done_payload(no_proposals=True)
    payload["needs_targeted_build"] = {
        "component": "aiter",
        "capability": "deepseek_v4_decode",
        "ref": "v0.1.15.post2",
    }

    await coord.specialist_dispatch.record_specialist_result(
        task=task,
        done_payload=payload,
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}build-spec",
    )

    coord.enablement_build.maybe_enqueue_specialist_requested_build.assert_awaited_once_with(
        task_id="build-spec",
        payload=payload,
    )


@pytest.mark.asyncio
async def test_record_specialist_result_empty_proposal_set(coord):
    """An empty proposal_set still records a round; proposals_total carries it."""
    task = _StubTask(task_id="task-empty-1", params={})
    coord.tasks.register(task)

    payload = _done_payload(no_proposals=True, domain="kernel_switch_specialist")
    await coord.specialist_dispatch.record_specialist_result(
        task=task,
        done_payload=payload,
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}task-empty-1",
    )

    state: _StubSharedState = coord.shared_state
    assert len(state.specialist_rounds) == 1
    assert state.specialist_rounds[0]["proposals_total"] == 0


@pytest.mark.asyncio
async def test_record_specialist_result_idempotent_on_round_id(coord):
    """The same explicit round_id overwrites instead of appending (resume doesn't dupe)."""
    task = _StubTask(
        task_id="t-resume",
        params={"round_id": "round-7"},
    )
    coord.tasks.register(task)

    await coord.specialist_dispatch.record_specialist_result(
        task=task,
        done_payload=_done_payload(),
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t-resume",
    )
    await coord.specialist_dispatch.record_specialist_result(
        task=task,
        done_payload=_done_payload(proposals=[]),
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t-resume",
    )

    state: _StubSharedState = coord.shared_state
    assert len(state.specialist_rounds) == 1
    assert state.specialist_rounds[0]["proposals_total"] == 0
    assert state.specialist_rounds[0]["round_id"] == "round-7"


# 4. build_specialist_round_entry — output shape
@pytest.mark.asyncio
async def test_build_specialist_round_entry_carries_full_payload(coord):
    """The entry carries the full field set the timeline round product expects."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    coord_obj = Coordinator.__new__(Coordinator)
    task = _StubTask(
        task_id="t-build",
        params={"round_id": "round-9", "source_phase": "KERNEL_AGENT"},
    )
    payload = _done_payload(
        domain="serving_specialist",
        proposals=[
            {"variant_name": "v1"},
            {"variant_name": "v2"},
        ],
        confidence=0.62,
    )
    entry = coord_obj.specialist_dispatch.build_specialist_round_entry(
        task=task,
        done_payload=payload,
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t-build",
    )

    expected_keys = {
        "round_id",
        "task_id",
        "source",
        "completed_at",
        "domain",
        "gap_canonical_id",
        "proposals_total",
        "proposal_set",
        "summary",
        "reason",
        "confidence",
        "new_findings",
        "residual_questions",
        "source_phase",
    }
    assert expected_keys.issubset(entry.keys())
    assert entry["round_id"] == "round-9"
    assert entry["task_id"] == "t-build"
    assert entry["proposals_total"] == 2
    assert entry["confidence"] == 0.62
    assert entry["source_phase"] == "KERNEL_AGENT"


@pytest.mark.asyncio
async def test_build_specialist_round_entry_round_id_falls_back_to_task_id(coord):
    task = _StubTask(task_id="task-no-round", params={})
    entry = coord.specialist_dispatch.build_specialist_round_entry(
        task=task,
        done_payload=_done_payload(),
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}task-no-round",
    )
    assert entry["round_id"] == "task-no-round"


@pytest.mark.asyncio
async def test_build_specialist_round_entry_records_run_error_and_notes(coord):
    task = _StubTask(task_id="t-err", params={})
    entry = coord.specialist_dispatch.build_specialist_round_entry(
        task=task,
        done_payload={**_done_payload(), "_specialist_notes": ["n1", 2]},
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t-err",
        run_error="dispatch failed",
    )
    assert entry["status"] == "failed"
    assert entry["error"] == entry["run_error"] == "dispatch failed"
    assert entry["notes"] == ["n1", "2"]


# 5. End-to-end: dispatcher exit hook bookkeeping
@pytest.mark.asyncio
async def test_dispatcher_hook_calls_bookkeeping_on_specialist_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end via the dispatcher exit hook: one specialist task lands the four bookkeeping mutations."""
    import argparse

    from hyperloom.inference_optimizer.cli.executors import _build_specialist_executor
    from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from hyperloom.orchestrator.state.task_registry import Task
    from hyperloom.orchestrator.tests.mock_backend import MockBackend as MockOrchBackend, MockTurn, ScriptedPlan

    done_payload = _done_payload(
        domain="serving_specialist",
        proposals=[
            {
                "variant_name": "moe_exp_par",
                "extra_server_args": "--expert-parallel-size 8",
                "extra_envs": {},
            },
        ],
    )
    use_fake_specialist_cli(tmp_path, monkeypatch, behavior="done_only", payload=done_payload)
    spec_args = argparse.Namespace(
        claude_model="claude-3-5-sonnet-latest",
        specialist_model=None,
        specialist_max_turns=4,
        research_lane_capacity=1,
        specialist_mcp_config=None,
    )
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockOrchBackend(idle_plan),
        "critic": MockOrchBackend(idle_plan),
    }

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        advisory=AdvisoryServices(knowledge_plane=None),
    )
    executor = _build_specialist_executor(
        spec_args,
        session_dir=session_dir,
        knowledge_plane=None,
    )
    coord.sub.register_executor("specialist", executor)

    # Enqueue directly through TaskRegistry to test the dispatcher hook, not the upstream intent flow.
    task = Task(
        task_id="t-e2e-1",
        kind="specialist",
        state="queued",
        params={
            "domain": "serving_specialist",
            "framework": "sglang",
            "gap_canonical_id": "gap.attention.fp8_kv",
            "max_turns": 4,
        },
        idempotency_key="t-e2e-1",
        requires_lanes=tuple(),
    )
    await coord.tasks.create_or_return_existing(
        TaskSpec(
            kind=task.kind,
            params=task.params,
            idempotency_key=task.idempotency_key,
        ),
    )
    await coord.tick(n=1)
    await pump_until_settled(coord.dispatcher)

    assert len(coord.shared_state.specialist_rounds) == 1, (
        "dispatcher hook should have triggered record_specialist_round"
    )
    row = coord.shared_state.specialist_rounds[0]
    assert row["domain"] == "serving_specialist"
    assert row["proposals_total"] == 1
    workspace = session_dir / "runs" / "specialist"
    assert workspace.exists()
    assert any(workspace.iterdir()), "specialist workspace should be non-empty"


# 8. SharedState.save failures around the specialist bookkeeping
def _record_result_run(coord):
    task = _StubTask(task_id="task-1", params={})
    coord.tasks.register(task)

    def run():
        return coord.specialist_dispatch.record_specialist_result(
            task=task, done_payload=_done_payload(), source=f"{SPECIALIST_FROM_AGENT_PREFIX}task-1"
        )

    return run


def _save_raising(exc: Exception):
    def _save(_session_dir) -> None:
        raise exc

    return _save


@pytest.mark.asyncio
async def test_save_os_error_is_logged_and_the_bookkeeping_stays(coord, caplog):
    run = _record_result_run(coord)
    err = OSError("no space left on device")
    coord.shared_state.save = _save_raising(err)

    with caplog.at_level("ERROR"):
        await run()

    assert [r["round_id"] for r in coord.shared_state.specialist_rounds] == ["task-1"]
    assert [r.exc_info[1] for r in caplog.records if r.exc_info] == [err]


@pytest.mark.asyncio
async def test_save_serialization_defect_propagates(coord):
    run = _record_result_run(coord)
    coord.shared_state.save = _save_raising(TypeError("Object of type set is not JSON serializable"))

    with pytest.raises(TypeError, match="not JSON serializable"):
        await run()
