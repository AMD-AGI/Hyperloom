"""EXPLORE-entry auto-roofline + specialist-dispatch block tests.

When ``--use-roofline-composite=true`` the Coordinator's
``_on_enter_explore`` hook auto-enqueues a ``roofline`` task and
stamps :attr:`SharedState.auto_roofline_pending_task_id`. First-round
specialist dispatches in the same EXPLORE phase must be denied by
``_auto_roofline_pending_denial`` so the specialists run with the
fresh ``analysis.md`` snapshot in their prompt's ROOFLINE EVIDENCE
section.

Once the auto-roofline reaches a terminal state, the field is cleared
either via ``_promote_to_shared_state`` (success path) or via the
race-safe in-line check inside ``_auto_roofline_pending_denial``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.policy import PolicyDenied


# ---------------------------------------------------------------------------
# Minimal stubs — TaskRegistry double mirroring just the contract we need.
# ---------------------------------------------------------------------------
@dataclass
class _BareState:
    use_roofline_composite: bool = True
    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    cumulative_gain_validated: float = 0.0
    auto_roofline_pending_task_id: str = ""
    baseline_config_path: str = ""
    current_best: dict[str, Any] = field(default_factory=dict)
    last_baseline: dict[str, Any] = field(default_factory=dict)
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    save_count: int = 0

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1


@dataclass
class _StubTask:
    task_id: str
    state: str = "queued"


class _StubTaskRegistry:
    """Tiny double that tracks a single in-flight roofline task."""

    def __init__(self):
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
        from inference_optimizer.orchestrator.task_registry import Task

        task = Task(
            task_id=_uuid.uuid4().hex,
            kind=kind,
            state="queued",
            params=dict(params),
            idempotency_key=idempotency_key,
        )
        self._tasks[task.task_id] = task   # type: ignore[assignment]
        self._by_idem[idempotency_key] = task   # type: ignore[assignment]
        return task, False

    async def get(self, task_id: str):
        from inference_optimizer.orchestrator.task_registry import TaskNotFound

        t = self._tasks.get(task_id)
        if t is None:
            raise TaskNotFound(task_id)
        return t

    def _set_state(self, task_id: str, new_state: str) -> None:
        t = self._tasks.get(task_id)
        if t is not None:
            t.state = new_state   # type: ignore[assignment]


@pytest.fixture
def coord(tmp_path: Path):
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.tasks = _StubTaskRegistry()
    c.knowledge_plane = None
    c.role_registry = {"kernel": object()}
    return c


# ===========================================================================
# 1. _on_enter_explore fires auto-roofline + sets pending field
# ===========================================================================
@pytest.mark.asyncio
async def test_on_enter_explore_enqueues_roofline_when_composite_on(coord):
    coord.shared_state.use_roofline_composite = True
    coord.shared_state.last_trace_analyze = {}  # no snapshot yet
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
    ]
    await coord._on_enter_explore(from_phase="PRELUDE")

    # Auto-roofline enqueued under the EXPLORE-scoped idempotency key.
    assert "internal-roofline-explore_entry" in coord.tasks._by_idem
    task = coord.tasks._by_idem["internal-roofline-explore_entry"]
    assert task.kind == "roofline"
    # Pending field stamped so _handle_delegate can block specialists.
    assert coord.shared_state.auto_roofline_pending_task_id == task.task_id
    # Audit trail.
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence.get("auto_roofline_enqueued") is True
    assert evidence.get("auto_roofline_task_id") == task.task_id


@pytest.mark.asyncio
async def test_on_enter_explore_noop_when_composite_disabled(coord):
    coord.shared_state.use_roofline_composite = False
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
    ]
    await coord._on_enter_explore(from_phase="PRELUDE")
    # No roofline enqueued, no pending field, no evidence stamp.
    assert coord.tasks._by_idem == {}
    assert coord.shared_state.auto_roofline_pending_task_id == ""
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert "auto_roofline_enqueued" not in evidence


@pytest.mark.asyncio
async def test_on_enter_explore_noop_when_snapshot_fresh(coord):
    """Fresh snapshot present + gain drift within band → no re-roof."""
    coord.shared_state.use_roofline_composite = True
    coord.shared_state.last_trace_analyze = {
        "analysis_md_text": "## Executive Summary\n| Compute % | 50% |",
        "roofline_baseline_gain_at_snapshot": 4.0,
    }
    coord.shared_state.cumulative_gain_validated = 6.0   # delta 2% < 10%
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
    ]
    await coord._on_enter_explore(from_phase="PRELUDE")

    assert coord.tasks._by_idem == {}
    assert coord.shared_state.auto_roofline_pending_task_id == ""


# ===========================================================================
# 2. _auto_roofline_pending_denial — denies while task in-flight
# ===========================================================================
@pytest.mark.asyncio
async def test_pending_denial_blocks_specialist_during_in_flight(coord):
    """While the auto-roofline is still queued/running, specialist
    dispatch must be denied with the canonical
    ``specialist_wait_for_auto_roofline`` rule."""
    # Simulate _on_enter_explore having enqueued a roofline that's
    # still running.
    from inference_optimizer.orchestrator.task_registry import Task

    pending = Task(
        task_id="rl-task-1", kind="roofline", state="running",
        params={}, idempotency_key="internal-roofline-explore_entry",
    )
    coord.tasks._tasks[pending.task_id] = pending
    coord.shared_state.auto_roofline_pending_task_id = pending.task_id

    denied = await coord._auto_roofline_pending_denial()
    assert isinstance(denied, PolicyDenied)
    assert denied.rule == "specialist_wait_for_auto_roofline"
    assert "rl-task-1" in str(denied)


@pytest.mark.asyncio
async def test_pending_denial_clears_field_on_terminal_state(coord):
    """If the roofline already reached a terminal state but
    ``_promote_to_shared_state`` hasn't cleared the field yet
    (race), the denial helper itself clears the field and returns
    ``None`` so the dispatch proceeds."""
    from inference_optimizer.orchestrator.task_registry import Task

    done = Task(
        task_id="rl-task-2", kind="roofline", state="succeeded",
        params={}, idempotency_key="internal-roofline-explore_entry",
    )
    coord.tasks._tasks[done.task_id] = done
    coord.shared_state.auto_roofline_pending_task_id = done.task_id

    denied = await coord._auto_roofline_pending_denial()
    assert denied is None
    assert coord.shared_state.auto_roofline_pending_task_id == ""


@pytest.mark.asyncio
async def test_pending_denial_returns_none_when_field_empty(coord):
    coord.shared_state.auto_roofline_pending_task_id = ""
    denied = await coord._auto_roofline_pending_denial()
    assert denied is None


@pytest.mark.asyncio
async def test_pending_denial_clears_field_when_task_missing(coord):
    """Corrupt resume edge: the field points at a task that no longer
    exists in the registry. The helper must clear the pointer instead
    of permanently blocking specialists."""
    coord.shared_state.auto_roofline_pending_task_id = "missing-task-id"
    denied = await coord._auto_roofline_pending_denial()
    assert denied is None
    assert coord.shared_state.auto_roofline_pending_task_id == ""
