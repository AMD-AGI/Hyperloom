# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Blocker #2 (safety net) — ``_materialize_approved_proposal`` must
re-check the auto-roofline gate after Critic approval.

The race the safety-net catches: Critic approves a gated proposal while
the watermark fires in the same tick. The cheaper gate in
``_handle_propose_action`` cannot see this — the proposal was already
emitted before the watermark crossed. Without the second gate the
explore task dispatches against a stale ``analysis.md`` snapshot.

Required behaviour:

  * Materialise defers the proposal onto
    ``_proposals_awaiting_roofline`` (FIFO) instead of dropping or
    dispatching, so the Critic round-trip is not wasted.
  * An ``observation`` envelope with
    ``kind='proposal_materialize_blocked'`` is recorded for the audit
    trail.
  * Once the analysis task lands and the gate clears,
    ``_drain_proposals_awaiting_roofline`` re-runs materialise and the
    task is finally created.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    CoordinatorState,
    PendingProposal,
)


@dataclass
class _BareState:
    baseline_tput: float = 100.0
    baseline_runtime_sec: float = 0.0
    explore_overtime_kill_ratio: float = 0.0
    cumulative_gain_validated: float = 0.0
    last_roofline_tput: float = 100.0
    auto_roofline_pending_task_id: str = ""
    enable_roofline: bool = True
    current_best: dict[str, Any] = field(default_factory=dict)
    baseline_config_path: str = ""
    policy_denial_history: list[dict[str, Any]] = field(default_factory=list)
    tick: int = 0


@dataclass
class _StubTask:
    task_id: str
    kind: str
    state: str = "running"
    params: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""


class _StubTaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, _StubTask] = {}
        self._by_idem: dict[str, _StubTask] = {}

    async def get(self, task_id: str) -> _StubTask:
        from inference_optimizer.orchestrator.task_registry import TaskNotFound

        t = self._tasks.get(task_id)
        if t is None:
            raise TaskNotFound(task_id)
        return t

    async def create_or_return_existing(
        self, *, kind: str, params: dict, idempotency_key: str, **_extras: Any,
    ):
        existing = self._by_idem.get(idempotency_key)
        if existing is not None:
            return existing, True
        import uuid as _uuid
        from inference_optimizer.orchestrator.task_registry import Task

        task = Task(
            task_id=_uuid.uuid4().hex,
            kind=kind,
            state="queued",
            params=dict(params),
            idempotency_key=idempotency_key,
        )
        self._tasks[task.task_id] = task  # type: ignore[assignment]
        self._by_idem[idempotency_key] = task  # type: ignore[assignment]
        return task, False


class _RecordingBus:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def append_and_seq(self, msg: Any) -> Any:
        self.messages.append(msg)
        return msg


@pytest.fixture
def coord(tmp_path: Path) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.tasks = _StubTaskRegistry()
    c.bus = _RecordingBus()
    c.cortex_kb = None
    c.state = CoordinatorState()
    c._proposals_awaiting_roofline = []
    return c


def _explore_proposal() -> PendingProposal:
    return PendingProposal(
        proposal_msg_id="proposal-msg-1",
        from_agent="orchestration",
        action_name="explore",
        predicted_gain_pct=5.0,
        payload={
            "action_name": "explore",
            "predicted_gain_pct": 5.0,
            "params": {"grid": [{"name": "v1"}]},
        },
    )


@pytest.mark.asyncio
async def test_materialize_dispatches_explore_while_roofline_pending(
    coord: Coordinator,
):
    """P3-b: auto-roofline is advisory. A Critic-approved explore
    proposal dispatches immediately even while a roofline task is in
    flight — it is NOT parked on the deferred queue."""
    from inference_optimizer.orchestrator.task_registry import Task

    pending = Task(
        task_id="rl-pending-id",
        kind="roofline",
        state="running",
        params={},
        idempotency_key="internal-analysis-watermark_crossed",
    )
    coord.tasks._tasks[pending.task_id] = pending  # type: ignore[assignment]
    coord.shared_state.auto_roofline_pending_task_id = pending.task_id

    proposal = _explore_proposal()
    await coord._materialize_approved_proposal(proposal)

    # Explore Task created right away; nothing parked, no blocked audit.
    explore_tasks = [
        t for t in coord.tasks._tasks.values() if t.kind == "explore"
    ]
    assert len(explore_tasks) == 1
    assert coord._proposals_awaiting_roofline == []
    obs = [m for m in coord.bus.messages if getattr(m, "topic", "") == "observation"]
    assert not any(
        getattr(m, "payload", {}).get("kind") == "proposal_materialize_blocked"
        for m in obs
    )


@pytest.mark.asyncio
async def test_drain_creates_task_once_roofline_clears(
    coord: Coordinator, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HYPERLOOM_ROOFLINE_DISPATCH_GATE", "1")
    """Once the analysis task lands and the pending field is cleared,
    :meth:`_drain_proposals_awaiting_roofline` re-runs materialise and
    finally creates the explore Task."""
    from inference_optimizer.orchestrator.task_registry import Task

    pending = Task(
        task_id="rl-pending-id",
        kind="roofline",
        state="running",
        params={},
        idempotency_key="internal-analysis-watermark_crossed",
    )
    coord.tasks._tasks[pending.task_id] = pending  # type: ignore[assignment]
    coord.shared_state.auto_roofline_pending_task_id = pending.task_id

    # Defer once.
    proposal = _explore_proposal()
    await coord._materialize_approved_proposal(proposal)
    assert len(coord._proposals_awaiting_roofline) == 1

    # Simulate completion: the promote-path clears the field, then
    # invokes the drain.
    coord.shared_state.auto_roofline_pending_task_id = ""
    await coord._drain_proposals_awaiting_roofline()

    # Deferred queue drained; an explore task is now in the registry.
    assert coord._proposals_awaiting_roofline == []
    explore_tasks = [
        t for t in coord.tasks._tasks.values() if t.kind == "explore"
    ]
    assert len(explore_tasks) == 1
    assert explore_tasks[0].idempotency_key == "approved-proposal-msg-1"


@pytest.mark.asyncio
async def test_drain_re_defers_if_another_roofline_armed(
    coord: Coordinator, monkeypatch: pytest.MonkeyPatch,
):
    """Edge case the drain must handle: a second watermark fires while
    the queue is being drained. The re-materialise hits the gate again
    and re-parks the proposal rather than dispatching against the new
    stale snapshot. (Legacy blocking gate, opt-in.)"""
    monkeypatch.setenv("HYPERLOOM_ROOFLINE_DISPATCH_GATE", "1")
    from inference_optimizer.orchestrator.task_registry import Task

    rl1 = Task(
        task_id="rl-1",
        kind="roofline",
        state="running",
        params={},
        idempotency_key="internal-analysis-watermark_crossed",
    )
    coord.tasks._tasks[rl1.task_id] = rl1  # type: ignore[assignment]
    coord.shared_state.auto_roofline_pending_task_id = rl1.task_id

    proposal = _explore_proposal()
    await coord._materialize_approved_proposal(proposal)
    assert len(coord._proposals_awaiting_roofline) == 1

    # First roofline finishes (field cleared) — but a second watermark
    # immediately fires and stamps a new pending id before the drain
    # actually runs.
    rl2 = Task(
        task_id="rl-2",
        kind="roofline",
        state="running",
        params={},
        idempotency_key="internal-analysis-explore_keep_watermark",
    )
    coord.tasks._tasks[rl2.task_id] = rl2  # type: ignore[assignment]
    coord.shared_state.auto_roofline_pending_task_id = rl2.task_id

    await coord._drain_proposals_awaiting_roofline()

    # Re-parked, not dispatched.
    assert len(coord._proposals_awaiting_roofline) == 1
    assert all(
        t.kind != "explore" for t in coord.tasks._tasks.values()
    )
