# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression (loosen P2_12): the ``wait_for_auto_roofline`` deny is gone — proposals proceed while an analysis task is in flight."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType


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
    c.bus = _RecordingBus()
    c.cortex_kb = None
    from inference_optimizer.orchestrator.coordinator import CoordinatorState

    c.state = CoordinatorState()
    c._sequence_denial_for_action = lambda *_a, **_kw: None  # type: ignore[assignment]
    return c


def test_roofline_pending_gate_hooks_removed():
    """The deny helpers and gated-action set must be gone from the Coordinator surface."""
    for attr in (
        "_roofline_denial_for_action",
        "_auto_roofline_pending_denial",
        "_defer_approved_proposal_for_roofline",
        "_drain_proposals_awaiting_roofline",
        "_proposals_awaiting_roofline",
    ):
        assert not hasattr(Coordinator, attr), (
            f"Coordinator.{attr} unexpectedly resurrected — the "
            f"P2_12 deletion would silently re-introduce the "
            f"wait_for_auto_roofline gate."
        )
    import inference_optimizer.orchestrator.coordinator as coord_mod
    assert not hasattr(coord_mod, "_ROOFLINE_GATED_ACTIONS")


@pytest.mark.asyncio
async def test_propose_action_passes_through_while_roofline_pending(
    coord: Coordinator,
):
    """An ``explore`` proposal lands on the bus even with an in-flight auto-roofline task (no deferral, no denial)."""
    coord.shared_state.auto_roofline_pending_task_id = "rl-pending-id"

    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "explore",
            "predicted_gain_pct": 5.0,
            "params": {"grid": [{"name": "v1"}]},
        },
    )
    await coord._handle_propose_action("orchestration", intent)

    topics = [getattr(m, "topic", None) for m in coord.bus.messages]
    assert "proposal" in topics, (
        "explore proposal must reach the bus while an analysis task is "
        "in flight: %r" % topics
    )
    assert len(coord.state.pending_proposals) == 1
    assert coord.shared_state.policy_denial_history == []
