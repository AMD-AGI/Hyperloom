"""Watermark-roofline trigger + dispatch-block tests (single path).

After Tasks 1-9 collapsed the legacy composite-on/off bifurcation,
roofline is auto-managed by the Coordinator in exactly two situations:

  1. PRELUDE bootstrap (after baseline lands), driven by the
     baseline-completion path in ``_promote_to_shared_state``.
  2. Mid-run watermark crossing — whenever measured tput crosses
     ``last_roofline_tput * 1.10``, the single ``cumulative_gain_validated``
     writer (explore-KEEP and kernel-integrate-KEEP paths) calls
     :meth:`Coordinator._maybe_enqueue_watermark_roofline`.

This file pins:

* :meth:`Coordinator._needs_roofline_for_watermark` — bootstrap guard,
  ratio threshold, and re-arm guard (pending field).
* :meth:`Coordinator._maybe_enqueue_watermark_roofline` — enqueue +
  ``auto_roofline_pending_task_id`` stamping.
* :meth:`Coordinator._auto_roofline_pending_denial` — blocks every
  member of the gated action set while a roofline is in-flight, and
  returns ``None`` for actions outside the set.
* The gated action set itself (specialist / explore / kernel_opt /
  integrate / deep_kernel_analysis / operator_tuning /
  vendor_kernel_config).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.policy import PolicyDenied


# ---------------------------------------------------------------------------
# Stubs — minimal SharedState + TaskRegistry doubles. Mirror just the
# contract Coordinator's helpers touch.
# ---------------------------------------------------------------------------
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

    async def get(self, task_id: str):
        from inference_optimizer.orchestrator.task_registry import TaskNotFound

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
    c.role_registry = {"kernel": object()}
    return c


# ===========================================================================
# 1. _needs_roofline_for_watermark — guards + threshold
# ===========================================================================
def test_watermark_check_false_before_first_roofline(coord: Coordinator):
    """Bootstrap guard: with ``last_roofline_tput=0`` the watermark
    must never fire — PRELUDE-bootstrap is the sole entry point."""
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.last_roofline_tput = 0.0
    coord.shared_state.cumulative_gain_validated = 50.0  # well past 10%
    assert coord._needs_roofline_for_watermark() is False


def test_watermark_check_false_below_ratio(coord: Coordinator):
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.last_roofline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 9.0  # 109 / 100 < 1.10
    assert coord._needs_roofline_for_watermark() is False


def test_watermark_check_true_at_ratio(coord: Coordinator):
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.last_roofline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 10.0  # 110 / 100 == 1.10
    assert coord._needs_roofline_for_watermark() is True


def test_watermark_check_true_above_ratio_compound(coord: Coordinator):
    """Compound step: after a roofline anchored at 110 tok/s (10%
    over baseline 100), the next trigger must fire at 121 tok/s
    (21% over baseline) — the 10% step is computed against the
    last roofline measurement, not against baseline."""
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.last_roofline_tput = 110.0
    coord.shared_state.cumulative_gain_validated = 21.0  # cur=121, 121/110=1.10
    assert coord._needs_roofline_for_watermark() is True


def test_watermark_check_false_when_already_pending(coord: Coordinator):
    """Re-arm guard: a single watermark crossing must enqueue only one
    roofline. If a task is already pending, the check returns False
    until the field is cleared."""
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.last_roofline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 20.0
    coord.shared_state.auto_roofline_pending_task_id = "in-flight-rl"
    assert coord._needs_roofline_for_watermark() is False


# ===========================================================================
# 2. _maybe_enqueue_watermark_roofline — enqueue + pending stamp
# ===========================================================================
@pytest.mark.asyncio
async def test_maybe_enqueue_watermark_skips_when_not_needed(coord: Coordinator):
    coord.shared_state.last_roofline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 5.0  # below threshold
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
    # Task created with the reason-scoped idempotency key.
    assert "internal-analysis-explore_keep_watermark" in coord.tasks._by_idem
    task = coord.tasks._by_idem["internal-analysis-explore_keep_watermark"]
    assert task.kind == "roofline"
    # Pending field stamped so downstream dispatches are blocked.
    assert coord.shared_state.auto_roofline_pending_task_id == task.task_id


@pytest.mark.asyncio
async def test_maybe_enqueue_watermark_dedups_per_reason(coord: Coordinator):
    """Two callers in the same tick with the same reason collapse to a
    single task via the idempotency key."""
    coord.shared_state.last_roofline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 15.0
    first = await coord._maybe_enqueue_watermark_roofline(reason="dup_reason")
    # _maybe_enqueue_watermark_roofline sets the pending field, which
    # short-circuits the re-arm guard on the second call. Simulate
    # the cleared field to force re-entry through the second branch.
    pending_id = coord.shared_state.auto_roofline_pending_task_id
    coord.shared_state.auto_roofline_pending_task_id = ""
    second = await coord._maybe_enqueue_watermark_roofline(reason="dup_reason")
    assert first is True and second is True
    # Same task returned by the registry (idempotency dedup).
    task = coord.tasks._by_idem["internal-analysis-dup_reason"]
    assert task.task_id == pending_id


# ===========================================================================
# 3. _auto_roofline_pending_denial — gates the right action set
# ===========================================================================
_GATED_ACTIONS = (
    "specialist",
    "explore",
    "kernel_opt",
    "integrate",
    "deep_kernel_analysis",
    "operator_tuning",
    "vendor_kernel_config",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("action_name", _GATED_ACTIONS)
async def test_pending_denial_blocks_every_gated_action(
    coord: Coordinator, action_name: str,
):
    from inference_optimizer.orchestrator.task_registry import Task

    pending = Task(
        task_id="rl-pending",
        kind="roofline",
        state="running",
        params={},
        idempotency_key="internal-analysis-watermark_crossed",
    )
    coord.tasks._tasks[pending.task_id] = pending  # type: ignore[assignment]
    coord.shared_state.auto_roofline_pending_task_id = pending.task_id

    denied = await coord._auto_roofline_pending_denial(action_name=action_name)
    assert isinstance(denied, PolicyDenied)
    assert denied.rule == "wait_for_auto_roofline"
    # The denial message names the action being blocked so the LLM
    # gets a useful hint instead of a generic "dispatch" label.
    assert action_name in str(denied)


@pytest.mark.asyncio
async def test_pending_denial_passes_through_when_no_task(coord: Coordinator):
    coord.shared_state.auto_roofline_pending_task_id = ""
    denied = await coord._auto_roofline_pending_denial(action_name="specialist")
    assert denied is None


@pytest.mark.asyncio
async def test_pending_denial_clears_field_on_terminal_state(coord: Coordinator):
    """Race: the roofline task already finished but the promote-path
    has not yet cleared the field. The helper itself clears it and
    returns ``None`` so the dispatch proceeds."""
    from inference_optimizer.orchestrator.task_registry import Task

    done = Task(
        task_id="rl-done",
        kind="roofline",
        state="succeeded",
        params={},
        idempotency_key="internal-analysis-watermark_crossed",
    )
    coord.tasks._tasks[done.task_id] = done  # type: ignore[assignment]
    coord.shared_state.auto_roofline_pending_task_id = done.task_id

    denied = await coord._auto_roofline_pending_denial(action_name="explore")
    assert denied is None
    assert coord.shared_state.auto_roofline_pending_task_id == ""


@pytest.mark.asyncio
async def test_pending_denial_clears_field_when_task_missing(coord: Coordinator):
    """Corrupt resume edge: the field points at a task the registry no
    longer knows about. Clear the pointer rather than permanently
    blocking dispatches."""
    coord.shared_state.auto_roofline_pending_task_id = "missing-rl-id"
    denied = await coord._auto_roofline_pending_denial(action_name="kernel_opt")
    assert denied is None
    assert coord.shared_state.auto_roofline_pending_task_id == ""
