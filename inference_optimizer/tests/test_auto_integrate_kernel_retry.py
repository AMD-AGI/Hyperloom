# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""KERNEL-phase auto-integrate retry for un-exhausted integration faults.

Covers ``Coordinator._auto_enqueue_pending_integrations`` re-dispatching a
retryable integration fault inside the KERNEL phase (rather than deferring to
the SWEEP-entry drain), the recorded-attempt-count in-flight guard, and the
``SharedState.integrate_attempt_count_for_kernel`` helper that powers it.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.shared_state import SharedState


def _ok_result(
    kernel_id: str,
    decision: str,
    micro: float,
    source_file: str = "",
    artifact: str = "",
) -> dict:
    """kernel_optimization_handler-shaped result (mirrors test_shared_state_kernel_opt)."""
    return {
        "status": "ok",
        "kernel_id": kernel_id,
        "source_file": source_file,
        "proposal": {"decision": decision, "reasons": []},
        "verification": {
            "micro_speedup": micro,
            "best_artifact_path": artifact,
            "compile_passed": True,
            "correctness_passed": True,
        },
    }


def _integrate_result(
    kernel_id: str,
    *,
    decision: str | None = None,
    status: str = "ok",
    error_class: str | None = None,
    patch_path: str = "",
    target_file: str = "",
    gain_pct: float | None = None,
) -> dict:
    """Integrate E2E result envelope (kernel integrate path)."""
    return {
        "status": status,
        "decision": decision,
        "kernel_id": kernel_id,
        "patch_path": patch_path or f"/tmp/{kernel_id}_opt.py",
        "target_file": target_file,
        "error_class": error_class,
        "gain_pct": gain_pct,
    }


class _FakeBus:
    """Captures append_and_seq messages without a DB."""

    def __init__(self):
        self.sent: list = []

    async def append_and_seq(self, msg) -> int:
        self.sent.append(msg)
        return len(self.sent)


def _coord(state: SharedState) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.shared_state = state
    c.bus = _FakeBus()
    return c


def _dispatched_kids(coord: Coordinator) -> list[str]:
    return [m.payload.get("kernel_id") for m in coord.bus.sent]


# integrate_attempt_count_for_kernel helper
def test_integrate_attempt_count_unknown_is_zero():
    state = SharedState()
    assert state.integrate_attempt_count_for_kernel("k001") == 0
    assert state.integrate_attempt_count_for_kernel("") == 0


def test_integrate_attempt_count_sums_across_patch_keys():
    state = SharedState()
    state.kernel_integrate_attempts = {
        "k001|/tmp/a.py|": {"kernel_id": "k001", "attempt_count": 2},
        "k001|/tmp/b.py|": {"kernel_id": "k001", "attempt_count": 1},
        "k002|/tmp/c.py|": {"kernel_id": "k002", "attempt_count": 5},
        "junk": "not-a-dict",
    }
    assert state.integrate_attempt_count_for_kernel("k001") == 3
    assert state.integrate_attempt_count_for_kernel("k002") == 5


# _auto_enqueue_pending_integrations — first dispatch + in-flight guard
@pytest.mark.asyncio
async def test_auto_enqueue_dispatches_pending_keep_once_then_guards_inflight():
    state = SharedState()
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "KEEP",
            3.0,
            source_file="/p/a.py",
            artifact="/tmp/k001_opt.py",
        )
    )
    coord = _coord(state)

    await coord._auto_enqueue_pending_integrations()
    assert _dispatched_kids(coord) == ["k001"]

    # No integrate recorded yet (request still in flight) -> no duplicate.
    await coord._auto_enqueue_pending_integrations()
    assert _dispatched_kids(coord) == ["k001"]


# Retryable fault is re-dispatched in KERNEL phase
@pytest.mark.asyncio
async def test_auto_enqueue_retries_unexhausted_fault():
    state = SharedState()
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "KEEP",
            3.0,
            source_file="/p/a.py",
            artifact="/tmp/k001_opt.py",
        )
    )
    coord = _coord(state)

    # First dispatch (mark = 0 recorded attempts).
    await coord._auto_enqueue_pending_integrations()
    assert _dispatched_kids(coord) == ["k001"]

    # The in-flight integrate completes as a fault (recorded -> count advances).
    entry = state.record_kernel_integrate_result(
        _integrate_result(
            "k001",
            decision="REVERT",
            status="failed",
            error_class="rebaseline_exception",
            patch_path="/tmp/k001_opt.py",
        )
    )
    assert entry.get("retryable") is True

    # Count advanced past the mark -> retry is dispatched.
    await coord._auto_enqueue_pending_integrations()
    assert _dispatched_kids(coord) == ["k001", "k001"]


# Exhausted fault budget -> kernel leaves pending -> no further dispatch
@pytest.mark.asyncio
async def test_auto_enqueue_stops_after_fault_budget_exhausted():
    state = SharedState()
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "KEEP",
            3.0,
            source_file="/p/a.py",
            artifact="/tmp/k001_opt.py",
        )
    )
    coord = _coord(state)

    # Two faults exhaust the budget (2) -> rejected.
    for _ in range(2):
        await coord._auto_enqueue_pending_integrations()
        state.record_kernel_integrate_result(
            _integrate_result(
                "k001",
                decision="REVERT",
                status="failed",
                error_class="apply_failed",
                patch_path="/tmp/k001_opt.py",
            )
        )

    assert "k001" in state.rejected_kernel_ids
    before = len(coord.bus.sent)
    await coord._auto_enqueue_pending_integrations()
    assert len(coord.bus.sent) == before  # no new dispatch after rejection
    assert _dispatched_kids(coord) == ["k001", "k001"]


# KEEP integrate leaves pending -> not re-dispatched
@pytest.mark.asyncio
async def test_auto_enqueue_no_redispatch_after_keep():
    state = SharedState()
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "KEEP",
            3.0,
            source_file="/p/a.py",
            artifact="/tmp/k001_opt.py",
        )
    )
    coord = _coord(state)

    await coord._auto_enqueue_pending_integrations()
    assert _dispatched_kids(coord) == ["k001"]

    # Integrate KEEPs -> lands in optimization_stack -> leaves pending.
    state.record_kernel_integrate_result(
        _integrate_result(
            "k001",
            decision="KEEP",
            status="ok",
            gain_pct=5.0,
            patch_path="/tmp/k001_opt.py",
            target_file="/p/a.py",
        )
    )
    state.optimization_stack.append(
        {
            "action": "integrate",
            "kernel_id": "k001",
            "target_file": "/p/a.py",
            "tput": 4500.0,
        }
    )

    before = len(coord.bus.sent)
    await coord._auto_enqueue_pending_integrations()
    assert len(coord.bus.sent) == before


# Genuine REVERT leaves pending -> not re-dispatched
@pytest.mark.asyncio
async def test_auto_enqueue_no_redispatch_after_genuine_revert():
    state = SharedState()
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "KEEP",
            3.0,
            source_file="/p/a.py",
            artifact="/tmp/k001_opt.py",
        )
    )
    coord = _coord(state)

    await coord._auto_enqueue_pending_integrations()
    assert _dispatched_kids(coord) == ["k001"]

    state.record_kernel_integrate_result(
        _integrate_result(
            "k001",
            decision="REVERT",
            status="ok",
            gain_pct=-3.0,
            patch_path="/tmp/k001_opt.py",
        )
    )
    assert "k001" in state.rejected_kernel_ids

    before = len(coord.bus.sent)
    await coord._auto_enqueue_pending_integrations()
    assert len(coord.bus.sent) == before


# Empty pending -> no-op
@pytest.mark.asyncio
async def test_auto_enqueue_noop_when_nothing_pending():
    coord = _coord(SharedState())
    await coord._auto_enqueue_pending_integrations()
    assert coord.bus.sent == []
