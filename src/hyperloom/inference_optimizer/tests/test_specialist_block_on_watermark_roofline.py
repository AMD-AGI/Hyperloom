# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Watermark-roofline trigger tests (single path).

Pins ``Coordinator._needs_roofline_for_watermark`` (bootstrap/ratio/re-arm
guards) and ``_maybe_enqueue_watermark_roofline`` (enqueue + pending stamp).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator


@dataclass
class _BareState:
    baseline_tput: float = 100.0
    cumulative_gain_validated: float = 0.0
    last_roofline_tput: float = 0.0
    auto_roofline_pending_task_id: str = ""
    enable_roofline: bool = True
    current_best: dict[str, Any] = field(default_factory=dict)
    last_baseline: dict[str, Any] = field(default_factory=dict)
    save_count: int = 0

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1


@dataclass
class _StubTask:
    task_id: str
    kind: str
    state: str = "queued"
    params: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""


class _StubTaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, _StubTask] = {}
        self._by_idem: dict[str, _StubTask] = {}

    async def create_or_return_existing(
        self,
        *,
        kind: str,
        params: dict,
        idempotency_key: str,
        **_extras: Any,
    ):
        existing = self._by_idem.get(idempotency_key)
        if existing is not None:
            return existing, True
        import uuid as _uuid
        from hyperloom.orchestrator.state.task_registry import Task

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

    async def get(self, task_id: str):
        from hyperloom.orchestrator.state.task_registry import TaskNotFound

        t = self._tasks.get(task_id)
        if t is None:
            raise TaskNotFound(task_id)
        return t

    def _set_state(self, task_id: str, new_state: str) -> None:
        t = self._tasks.get(task_id)
        if t is not None:
            t.state = new_state  # type: ignore[assignment]


@pytest.fixture
def coord(tmp_path: Path) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.tasks = _StubTaskRegistry()
    c.knowledge_plane = None
    c.role_registry = {"kernel_agent": object()}
    return c


# _needs_roofline_for_watermark — guards + threshold
def test_watermark_check_false_before_first_roofline(coord: Coordinator):
    """Bootstrap guard: with ``last_roofline_tput=0`` the watermark never fires."""
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.last_roofline_tput = 0.0
    coord.shared_state.cumulative_gain_validated = 50.0
    assert coord._needs_roofline_for_watermark() is False


def test_watermark_check_false_below_ratio(coord: Coordinator):
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.last_roofline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 9.0
    assert coord._needs_roofline_for_watermark() is False


def test_watermark_check_true_at_ratio(coord: Coordinator):
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.last_roofline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 10.0
    assert coord._needs_roofline_for_watermark() is True


def test_watermark_check_true_above_ratio_compound(coord: Coordinator):
    """Compound step: the 10% step is computed against the last roofline measurement, not baseline."""
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.last_roofline_tput = 110.0
    coord.shared_state.cumulative_gain_validated = 21.0
    assert coord._needs_roofline_for_watermark() is True


def test_watermark_check_false_when_already_pending(coord: Coordinator):
    """Re-arm guard: when a roofline task is already pending the check returns False."""
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.last_roofline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 20.0
    coord.shared_state.auto_roofline_pending_task_id = "in-flight-rl"
    assert coord._needs_roofline_for_watermark() is False


def test_watermark_ratio_resolver_returns_default():
    """The module-level resolver returns the fixed 1.10 ratio."""
    from hyperloom.orchestrator.loop.coordinator import (
        _resolve_roofline_watermark_ratio,
    )

    assert _resolve_roofline_watermark_ratio() == 1.10


# _maybe_enqueue_watermark_roofline — enqueue + pending stamp
@pytest.mark.asyncio
async def test_maybe_enqueue_watermark_skips_when_not_needed(coord: Coordinator):
    coord.shared_state.last_roofline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 5.0
    fired = await coord._maybe_enqueue_watermark_roofline(reason="explore_keep")
    assert fired is False
    assert coord.shared_state.auto_roofline_pending_task_id == ""
    assert coord.tasks._by_idem == {}


@pytest.mark.asyncio
async def test_maybe_enqueue_watermark_enqueues_when_crossed(coord: Coordinator):
    coord.shared_state.last_roofline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 10.0
    fired = await coord._maybe_enqueue_watermark_roofline(
        reason="explore_keep_watermark",
    )
    assert fired is True
    assert "internal-analysis-explore_keep_watermark" in coord.tasks._by_idem
    task = coord.tasks._by_idem["internal-analysis-explore_keep_watermark"]
    assert task.kind == "roofline"
    assert coord.shared_state.auto_roofline_pending_task_id == task.task_id


@pytest.mark.asyncio
async def test_maybe_enqueue_watermark_dedups_per_reason(coord: Coordinator):
    """Two same-reason callers collapse to a single task via the idempotency key."""
    coord.shared_state.last_roofline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 15.0
    first = await coord._maybe_enqueue_watermark_roofline(reason="dup_reason")
    # Clear the pending field to force re-entry past the re-arm guard.
    pending_id = coord.shared_state.auto_roofline_pending_task_id
    coord.shared_state.auto_roofline_pending_task_id = ""
    second = await coord._maybe_enqueue_watermark_roofline(reason="dup_reason")
    assert first is True and second is True
    task = coord.tasks._by_idem["internal-analysis-dup_reason"]
    assert task.task_id == pending_id
