# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""v0.8 §3.5 §10 / M5 — specialist_done bookkeeping (KB_gaps/Gap-03).

Gap-03 root cause: ``_handle_intent`` lacked a SPECIALIST_DONE branch, so the
intent degraded to an observation and dropped its bookkeeping. Exercises
``_record_specialist_result``, the intent-routing path, the dispatcher exit
hook, streak semantics, round_id idempotence, and unknown-task defense.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from inference_optimizer.protocol.intent import (
    Intent, IntentType,
)
from inference_optimizer.orchestrator.policy import SPECIALIST_FROM_AGENT_PREFIX


# Helpers — minimal stand-ins
@dataclass
class _StubTask:
    """Task-shaped stub (only ``task_id`` and ``params`` are inspected)."""
    task_id: str
    kind: str = "specialist"
    params: dict[str, Any] = field(default_factory=dict)


class _StubSharedState:
    """SharedState stand-in that records the bookkeeping calls."""

    def __init__(self):
        self.specialist_rounds: list[dict[str, Any]] = []
        self.specialist_domain_empty_streak: dict[str, int] = {}
        self.last_specialist: dict[str, Any] = {}
        self.saved: int = 0

    def record_specialist_round(self, entry: dict[str, Any]) -> None:
        # Mirror the real SharedState's idempotence-on-round_id behaviour.
        round_id = str(entry.get("round_id") or "").strip()
        if round_id:
            for i, prev in enumerate(self.specialist_rounds):
                if str(prev.get("round_id") or "") == round_id:
                    self.specialist_rounds[i] = dict(entry)
                    return
        self.specialist_rounds.append(dict(entry))

    def bump_specialist_domain_empty_streak(
        self, domain: str, *, empty: bool,
    ) -> int:
        d = domain or "unknown"
        if empty:
            self.specialist_domain_empty_streak[d] = (
                self.specialist_domain_empty_streak.get(d, 0) + 1
            )
        else:
            self.specialist_domain_empty_streak[d] = 0
        return self.specialist_domain_empty_streak[d]

    def update_last_specialist(self, snapshot: dict[str, Any]) -> None:
        self.last_specialist = dict(snapshot)

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
            from inference_optimizer.orchestrator.task_registry import TaskNotFound
            raise TaskNotFound(f"task {task_id} not found")
        return self._tasks[task_id]


# Fixture: lean Coordinator stand-in
@pytest.fixture
def coord(tmp_path: Path):
    """Build a Coordinator via ``__new__`` with just enough attributes for the Gap-03 methods."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _StubSharedState()
    c.tasks = _StubTaskRegistry()
    c._record_observation = AsyncMock()  # type: ignore[method-assign]
    return c


def _done_payload(
    *,
    domain: str = "serving_specialist",
    gap: str = "gap.attention.fp8_kv",
    proposals: list | None = None,
    empty: bool = False,
    summary: str = "Specialist found candidate variants",
    confidence: float = 0.7,
) -> dict[str, Any]:
    if proposals is None:
        proposals = [] if empty else [
            {
                "variant_name": "max_seqs_512",
                "extra_server_args": "--max-num-seqs 512",
                "rationale": "moderate concurrency bump",
            },
        ]
    return {
        "domain": domain,
        "gap_canonical_id": gap,
        "proposal_set": proposals,
        "empty": empty or len(proposals) == 0,
        "summary": summary,
        "reason": "kb_evidence" if not empty else "no_findings",
        "confidence": confidence,
        "new_findings": ["fp8 kv cache stable above bs=128"],
        "residual_questions": [],
    }


# 1. _record_specialist_result — direct bookkeeping unit tests
@pytest.mark.asyncio
async def test_record_specialist_result_non_empty_proposal_set(coord):
    """Non-empty proposal_set: ledger +1 row, streak reset, last_specialist mirrored, save called."""
    task = _StubTask(task_id="task-1", params={})
    coord.tasks.register(task)

    payload = _done_payload(domain="serving_specialist")
    await coord._record_specialist_result(
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
    assert row["empty"] is False
    assert row["proposals_total"] == 1
    assert row["round_id"] == "task-1"
    assert state.specialist_domain_empty_streak.get("serving_specialist", 0) == 0
    assert state.last_specialist["task_id"] == "task-1"
    assert state.last_specialist["empty"] is False
    assert state.last_specialist["proposals_total"] == 1
    assert state.last_specialist["domain"] == "serving_specialist"
    assert state.saved == 1
    coord._record_observation.assert_awaited_once()
    args, kwargs = coord._record_observation.call_args
    assert args[1] == "observation"
    assert args[2]["kind"] == "specialist_done_recorded"


@pytest.mark.asyncio
async def test_record_specialist_result_empty_proposal_set_bumps_streak(coord):
    """Empty proposal_set: ledger row stays (empty=True), streak +1."""
    task = _StubTask(task_id="task-empty-1", params={})
    coord.tasks.register(task)

    payload = _done_payload(empty=True, domain="kernel_switch_specialist")
    await coord._record_specialist_result(
        task=task,
        done_payload=payload,
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}task-empty-1",
    )

    state: _StubSharedState = coord.shared_state
    assert len(state.specialist_rounds) == 1
    assert state.specialist_rounds[0]["empty"] is True
    assert state.specialist_domain_empty_streak.get("kernel_switch_specialist") == 1
    assert state.last_specialist["empty"] is True


@pytest.mark.asyncio
async def test_record_specialist_result_streak_accumulates_then_resets(coord):
    """Empty × 3 then non-empty: streak goes 1 → 2 → 3 → 0."""
    state: _StubSharedState = coord.shared_state
    for n in range(1, 4):
        task = _StubTask(task_id=f"task-{n}", params={})
        coord.tasks.register(task)
        await coord._record_specialist_result(
            task=task,
            done_payload=_done_payload(empty=True, domain="comm_specialist"),
            source=f"{SPECIALIST_FROM_AGENT_PREFIX}task-{n}",
        )
        assert (
            state.specialist_domain_empty_streak["comm_specialist"] == n
        )

    task4 = _StubTask(task_id="task-4", params={})
    coord.tasks.register(task4)
    await coord._record_specialist_result(
        task=task4,
        done_payload=_done_payload(empty=False, domain="comm_specialist"),
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}task-4",
    )
    assert state.specialist_domain_empty_streak["comm_specialist"] == 0


@pytest.mark.asyncio
async def test_record_specialist_result_per_domain_streak_independent(coord):
    """Two different domains keep independent streak counters."""
    state: _StubSharedState = coord.shared_state

    for tid, dom in [
        ("t-fw-1", "serving_specialist"),
        ("t-kn-1", "kernel_switch_specialist"),
        ("t-fw-2", "serving_specialist"),
    ]:
        t = _StubTask(task_id=tid, params={})
        coord.tasks.register(t)
        await coord._record_specialist_result(
            task=t,
            done_payload=_done_payload(empty=True, domain=dom),
            source=f"{SPECIALIST_FROM_AGENT_PREFIX}{tid}",
        )

    assert state.specialist_domain_empty_streak["serving_specialist"] == 2
    assert state.specialist_domain_empty_streak["kernel_switch_specialist"] == 1


@pytest.mark.asyncio
async def test_record_specialist_result_idempotent_on_round_id(coord):
    """The same explicit round_id overwrites instead of appending (resume doesn't dupe)."""
    task = _StubTask(
        task_id="t-resume",
        params={"round_id": "round-7"},
    )
    coord.tasks.register(task)

    await coord._record_specialist_result(
        task=task,
        done_payload=_done_payload(empty=False),
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t-resume",
    )
    await coord._record_specialist_result(
        task=task,
        done_payload=_done_payload(empty=True, proposals=[]),
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t-resume",
    )

    state: _StubSharedState = coord.shared_state
    assert len(state.specialist_rounds) == 1
    assert state.specialist_rounds[0]["empty"] is True
    assert state.specialist_rounds[0]["round_id"] == "round-7"


# 2. _handle_specialist_done — intent routing path
@pytest.mark.asyncio
async def test_handle_specialist_done_routes_known_task(coord):
    """A known task_id triggers the full bookkeeping pass."""
    task = _StubTask(task_id="task-route", params={})
    coord.tasks.register(task)

    intent = Intent(
        type=IntentType.SPECIALIST_DONE,
        payload=_done_payload(domain="serving_specialist"),
    )
    await coord._handle_specialist_done(
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}task-route",
        intent=intent,
    )

    state: _StubSharedState = coord.shared_state
    assert len(state.specialist_rounds) == 1
    assert state.last_specialist["task_id"] == "task-route"


@pytest.mark.asyncio
async def test_handle_specialist_done_unknown_task_logs_and_skips(coord, caplog):
    """An unknown task_id logs a warning and skips bookkeeping (defense-in-depth)."""
    intent = Intent(
        type=IntentType.SPECIALIST_DONE,
        payload=_done_payload(domain="serving_specialist"),
    )
    await coord._handle_specialist_done(
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}unknown-id",
        intent=intent,
    )
    state: _StubSharedState = coord.shared_state
    assert state.specialist_rounds == []
    assert state.specialist_domain_empty_streak == {}


@pytest.mark.asyncio
async def test_handle_specialist_done_bad_source_prefix(coord):
    """A source not prefixed with ``specialist:`` → empty task_id → no bookkeeping."""
    intent = Intent(
        type=IntentType.SPECIALIST_DONE,
        payload=_done_payload(),
    )
    await coord._handle_specialist_done(
        source="orchestration",
        intent=intent,
    )
    state: _StubSharedState = coord.shared_state
    assert state.specialist_rounds == []


# 3. _task_id_from_specialist_source helper
def test_task_id_from_specialist_source_extracts_prefix():
    from inference_optimizer.orchestrator.coordinator import Coordinator

    assert Coordinator._task_id_from_specialist_source(
        "specialist:abc-123",
    ) == "abc-123"


def test_task_id_from_specialist_source_returns_empty_for_bad():
    from inference_optimizer.orchestrator.coordinator import Coordinator

    assert Coordinator._task_id_from_specialist_source("orchestration") == ""
    assert Coordinator._task_id_from_specialist_source("") == ""
    assert Coordinator._task_id_from_specialist_source(
        "robustness",
    ) == ""


# 4. _build_specialist_round_entry — output shape
@pytest.mark.asyncio
async def test_build_specialist_round_entry_carries_full_payload(coord):
    """The entry carries the full field set the breakdown ``specialist_runs[]`` consumer expects (KB_design §3.12 §4.3)."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    coord_obj = Coordinator.__new__(Coordinator)
    task = _StubTask(
        task_id="t-build",
        params={"round_id": "round-9"},
    )
    payload = _done_payload(
        domain="serving_specialist",
        proposals=[
            {"variant_name": "v1"},
            {"variant_name": "v2"},
        ],
        confidence=0.62,
    )
    entry = coord_obj._build_specialist_round_entry(
        task=task,
        done_payload=payload,
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t-build",
    )

    expected_keys = {
        "round_id", "task_id", "source", "completed_at",
        "domain", "gap_canonical_id", "empty",
        "proposals_total", "proposal_set",
        "summary", "reason", "confidence",
        "new_findings", "residual_questions",
    }
    assert expected_keys.issubset(entry.keys())
    assert entry["round_id"] == "round-9"
    assert entry["task_id"] == "t-build"
    assert entry["proposals_total"] == 2
    assert entry["empty"] is False
    assert entry["confidence"] == 0.62


@pytest.mark.asyncio
async def test_build_specialist_round_entry_round_id_falls_back_to_task_id(coord):
    from inference_optimizer.orchestrator.coordinator import Coordinator

    coord_obj = Coordinator.__new__(Coordinator)
    task = _StubTask(task_id="task-no-round", params={})
    entry = coord_obj._build_specialist_round_entry(
        task=task,
        done_payload=_done_payload(),
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}task-no-round",
    )
    assert entry["round_id"] == "task-no-round"


# 5. End-to-end: dispatcher exit hook bookkeeping
@pytest.mark.asyncio
async def test_dispatcher_hook_calls_bookkeeping_on_specialist_task(
    tmp_path: Path,
):
    """End-to-end via the dispatcher exit hook: one specialist task lands the four bookkeeping mutations."""
    from inference_optimizer.cli import _build_specialist_executor
    from inference_optimizer.orchestrator.backends.mock_backend import (
        MockBackend, MockTurn, ScriptedPlan,
    )
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.protocol.intent import IntentType
    from inference_optimizer.orchestrator.backends.mock_backend import (
        MockBackend as MockOrchBackend,
    )
    from inference_optimizer.orchestrator.agent_role import default_role_registry

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
    plan = ScriptedPlan(turns=[MockTurn(intents=[
        Intent(type=IntentType.SPECIALIST_DONE, payload=done_payload),
    ])])

    import inference_optimizer.cli_executors as cli_mod
    real_claude_cls = cli_mod.ClaudeBackend
    cli_mod.ClaudeBackend = lambda **_kw: MockBackend(plan, name="spec-mock")
    try:
        import argparse
        spec_args = argparse.Namespace(
            claude_model="claude-3-5-sonnet-latest",
            specialist_model=None,
            specialist_max_turns=4,
            specialist_per_turn_max_seconds=300.0,
            research_lane_capacity=1,
            # PR-A2: in-process dispatch so the mocked ClaudeBackend is used.
            specialist_dispatch_mode="inprocess",
            specialist_mcp_config=None,
        )
        idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
        backends = {
            "orchestration": MockOrchBackend(idle_plan),
            "kernel":        MockOrchBackend(idle_plan),
            "critic":        MockOrchBackend(idle_plan),
            "robustness":    MockOrchBackend(idle_plan),
        }

        session_dir = tmp_path / "session"
        session_dir.mkdir()
        coord = Coordinator(
            session_dir=session_dir,
            backends=backends,
            role_registry=default_role_registry(),
            cortex_kb=None,
            knowledge_plane=None,
        )
        executor = _build_specialist_executor(
            spec_args, session_dir=session_dir, knowledge_plane=None,
        )
        coord.sub.register_executor("specialist", executor)

        # Enqueue directly through TaskRegistry to test the dispatcher hook, not the upstream intent flow.
        from inference_optimizer.orchestrator.task_registry import Task
        task = Task(
            task_id="t-e2e-1",
            kind="specialist",
            state="queued",
            params={
                "domain": "serving_specialist",
                "gap_canonical_id": "gap.attention.fp8_kv",
                "max_turns": 4,
                "pr_feed": [],
            },
            idempotency_key="t-e2e-1",
            requires_lanes=tuple(),
        )
        await coord.tasks.create_or_return_existing(
            kind=task.kind, params=task.params,
            idempotency_key=task.idempotency_key,
        )
        await coord.tick(n=1)
    finally:
        cli_mod.ClaudeBackend = real_claude_cls

    assert len(coord.shared_state.specialist_rounds) == 1, (
        "dispatcher hook should have triggered record_specialist_round"
    )
    row = coord.shared_state.specialist_rounds[0]
    assert row["domain"] == "serving_specialist"
    assert row["proposals_total"] == 1
    assert row["empty"] is False
    assert coord.shared_state.specialist_domain_empty_streak.get(
        "serving_specialist", -1
    ) == 0
    assert coord.shared_state.last_specialist.get("domain") == "serving_specialist"
    workspace = session_dir / "runs" / "specialist"
    assert workspace.exists()
    assert any(workspace.iterdir()), "specialist workspace should be non-empty"


# 6. Intent routing branch wired into _handle_intent
def test_handle_intent_dispatch_table_has_specialist_done_branch():
    """Gap-03 regression: the dispatch table routes SPECIALIST_DONE to ``_handle_specialist_done``."""
    import inspect

    from inference_optimizer.orchestrator.coordinator import Coordinator

    src = inspect.getsource(Coordinator._handle_intent)
    assert "IntentType.SPECIALIST_DONE" in src, (
        "_handle_intent must dispatch SPECIALIST_DONE (KB_gaps/Gap-03)"
    )
    assert "_handle_specialist_done" in src, (
        "_handle_intent must route SPECIALIST_DONE to "
        "_handle_specialist_done"
    )
