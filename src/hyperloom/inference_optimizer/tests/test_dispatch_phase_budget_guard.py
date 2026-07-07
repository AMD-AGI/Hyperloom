# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Phase-budget dispatch guard (KERNEL/EXPLORE interleave-stall fix).

Regression for the bug where a long serially-drained KERNEL/EXPLORE grid kept
``_pump_dispatcher_once`` from returning, starving the per-phase cyclic budget
exit so the phase machine never advanced.
"""

from __future__ import annotations

import time
import types

import hyperloom.orchestrator.phases.machine_state as ps
from hyperloom.orchestrator.loop.coordinator import Coordinator


def _stub(*, phase: str, elapsed_h: float, max_minutes: int = 2880, cycle_minutes: float = 360.0):
    """Build a minimal coordinator-like stub for the guard method."""
    now = time.time()
    state = types.SimpleNamespace(
        phase=phase,
        max_minutes=max_minutes,
        cycle_minutes=cycle_minutes,
        phase_started_unix=now - elapsed_h * 3600.0,
        phase_budget_pct={"PRELUDE": 0.03, "FRAMEWORK_AGENT": 0.0, "EXPLORE": 0.45, "KERNEL_AGENT": 0.38, "SWEEP": 0.12, "CLOSE": 0.02},
    )
    stub = types.SimpleNamespace(
        shared_state=state,
        _phase_budget_pct=state.phase_budget_pct,
        _BUDGET_GATED_DISPATCH_PHASES=Coordinator._BUDGET_GATED_DISPATCH_PHASES,
    )
    return stub


def _paused(stub) -> bool:
    return Coordinator._dispatch_paused_for_phase_budget(stub)  # type: ignore[arg-type]


# KERNEL budget = 0.38 * 6h ≈ 2.28h.
def test_kernel_over_budget_pauses(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CYCLIC_PHASES", "1")
    stub = _stub(phase="KERNEL_AGENT", elapsed_h=9.0)
    assert _paused(stub) is True


def test_kernel_under_budget_not_paused(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CYCLIC_PHASES", "1")
    stub = _stub(phase="KERNEL_AGENT", elapsed_h=1.0)  # < 2.28h
    assert _paused(stub) is False


def test_explore_over_budget_pauses(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CYCLIC_PHASES", "1")
    stub = _stub(phase="EXPLORE", elapsed_h=4.0)  # EXPLORE budget 0.45*6h=2.7h
    assert _paused(stub) is True


# PRELUDE / SWEEP / CLOSE are never gated (mandatory-work phases).
def test_prelude_never_paused_even_over_budget(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CYCLIC_PHASES", "1")
    stub = _stub(phase="PRELUDE", elapsed_h=9.0)
    assert _paused(stub) is False


def test_sweep_never_paused(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CYCLIC_PHASES", "1")
    stub = _stub(phase="SWEEP", elapsed_h=9.0)
    assert _paused(stub) is False


# Short bounded run (≤24h) is not "long" → phases anchored on whole session, no cyclic pause.
def test_short_bounded_run_not_paused(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CYCLIC_PHASES", "1")
    stub = _stub(phase="KERNEL_AGENT", elapsed_h=9.0, max_minutes=120, cycle_minutes=360.0)
    assert ps.is_long_run(stub.shared_state) is False
    assert _paused(stub) is False


# Cyclic disabled → no pause (legacy monotonic behaviour).
def test_cyclic_disabled_not_paused(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CYCLIC_PHASES", "0")
    stub = _stub(phase="KERNEL_AGENT", elapsed_h=9.0)
    assert _paused(stub) is False
