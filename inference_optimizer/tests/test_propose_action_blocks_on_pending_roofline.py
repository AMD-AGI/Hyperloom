"""Blocker #2 (cheap path) — ``_handle_propose_action`` must consult the
auto-roofline gate before paying for the Critic round-trip.

Without this gate, an LLM-emitted ``propose_action{action='explore'}``
with a grid (the only path that goes through propose, not delegate) would
fan out to the Critic while the PRELUDE / watermark analysis task is
still in flight — wasting a round-trip + risking dispatch against a stale
``analysis.md`` snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType


@dataclass
class _BareState:
    baseline_tput: float = 100.0
    cumulative_gain_validated: float = 0.0
    last_roofline_tput: float = 100.0
    auto_roofline_pending_task_id: str = ""
    enable_roofline: bool = True
    policy_denial_history: list[dict[str, Any]] = field(default_factory=list)
    tick: int = 0
    pruned_families: set[str] = field(default_factory=set)
    cortex_session_id: str = ""

    def is_pruned(self, family: str) -> bool:
        return family in self.pruned_families

    def prune_family(self, family: str) -> bool:
        self.pruned_families.add(family)
        return True

    def record_policy_denial(self, **kw: Any) -> int:
        self.policy_denial_history.append(kw)
        return len(self.policy_denial_history)

    def set_stop_reason(self, *_a: Any, **_kw: Any) -> None:
        return None


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

    async def get(self, task_id: str) -> _StubTask:
        from inference_optimizer.orchestrator.task_registry import TaskNotFound

        t = self._tasks.get(task_id)
        if t is None:
            raise TaskNotFound(task_id)
        return t


class _RecordingBus:
    """Captures every emitted Message so the test can assert on
    ``observation`` envelopes (the policy-denial path emits one)."""

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
    from inference_optimizer.orchestrator.coordinator import CoordinatorState

    c.state = CoordinatorState()
    # ``_handle_propose_action`` calls
    # ``_sequence_denial_for_action`` after the roofline gate. We
    # short-circuit that here so the test focuses on the gate itself —
    # full sequence-gate coverage lives in test_policy_sequence_*.py.
    c._sequence_denial_for_action = lambda *_a, **_kw: None  # type: ignore[assignment]
    return c


@pytest.mark.asyncio
async def test_propose_action_blocks_explore_while_roofline_pending(
    coord: Coordinator,
):
    """Explore-with-grid is the only path that re-routes through
    ``_handle_propose_action``. With a roofline task pending, the gate
    must drop the proposal *before* a ``proposal`` message hits the bus
    (which is what the Critic agent reads)."""
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

    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "explore",
            "predicted_gain_pct": 5.0,
            "params": {"grid": [{"name": "v1"}]},
        },
    )
    await coord._handle_propose_action("orchestration", intent)

    # No ``proposal`` envelope must have reached the bus (gate fired
    # before bus.append_and_seq for the proposal). The only envelope
    # in flight is the denial observation.
    topics = [getattr(m, "topic", None) for m in coord.bus.messages]
    assert "proposal" not in topics, (
        "propose_action gate failed — Critic would see a stale-roofline "
        "explore proposal: %r" % topics
    )
    assert "observation" in topics
    # PendingProposal table must remain empty so no later verdict can
    # re-animate the rejected proposal.
    assert coord.state.pending_proposals == {}
    # Denial counted in the streak ledger via record_policy_denial.
    assert len(coord.shared_state.policy_denial_history) == 1
    assert coord.shared_state.policy_denial_history[0]["rule"] == (
        "wait_for_auto_roofline"
    )


@pytest.mark.asyncio
async def test_propose_action_passes_through_non_gated_action(
    coord: Coordinator,
):
    """Non-gated actions (``report``) must dispatch even while a
    roofline task is in flight — the gate is per-action, not global."""
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

    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "report",
            "predicted_gain_pct": 0.0,
            "params": {},
        },
    )
    await coord._handle_propose_action("orchestration", intent)

    # Non-gated action proceeds: a ``proposal`` envelope is emitted and
    # the PendingProposal entry is created.
    topics = [getattr(m, "topic", None) for m in coord.bus.messages]
    assert "proposal" in topics
    assert len(coord.state.pending_proposals) == 1
