# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KERNEL-phase auto-integrate retry for un-exhausted integration faults."""

from __future__ import annotations

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState


def _ok_result(
    kernel_id: str,
    decision: str,
    micro: float,
    source_file: str = "",
    artifact: str = "",
) -> dict:
    """kernel_optimization_handler-shaped result."""
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
    integration_id: str = "",
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
        "integration_id": integration_id,
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


def _dispatched_integration_id(coord: Coordinator) -> str:
    """The ``integration_id`` carried by the most recent integrate request."""
    return str(coord.bus.sent[-1].payload.get("integration_id") or "")


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
