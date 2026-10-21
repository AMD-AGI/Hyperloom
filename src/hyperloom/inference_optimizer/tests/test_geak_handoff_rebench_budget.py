# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KERNEL work hands back with enough of the phase left to measure its result: GEAK and the rewrite Controller.

GEAK's result is re-measured by an explore round measured the way the baseline was, so what GEAK hands back is that
round and a quarter of a round to spare: both passes after a baseline that ran a warm pass, its one pass otherwise.
The Controller hands back one measurement.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors._subprocess_kill import resolve_benchmark_timeouts
from hyperloom.orchestrator.phases import machine_state as ps

SESSION_LEFT = 40000.0
CLOSING_RESERVE = 1800.0
PHASE_LEFT = 30000.0
MARGIN = 300
BOOT = 600.0
BENCH = 5400.0
WARM = 3600.0
ONE_ROUND = BOOT + BENCH
SPARE = ONE_ROUND / 4


@pytest.fixture
def coordinator(session_dir, monkeypatch):
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import MockBackend, MockCriticBackend, ScriptedPlan
    from hyperloom.orchestrator.state.shared_state import SharedState

    for name in ("HYPERLOOM_AGENTX", "GEAK_BUDGET_MARGIN_S", "GEAK_E2E_TIMEOUT_S"):
        monkeypatch.delenv(name, raising=False)
    orchestration = MockBackend(ScriptedPlan(turns=[]), name="orchestration")
    coord = Coordinator(session_dir, backends={"orchestration": orchestration, "critic": MockCriticBackend()})
    coord._run_deadline = SimpleNamespace(remaining=lambda: SESSION_LEFT)
    monkeypatch.setattr(SharedState, "closing_reserve_sec", lambda _self: CLOSING_RESERVE)
    monkeypatch.setattr(ps, "phase_budget_remaining_seconds", lambda _state, **_kw: PHASE_LEFT)
    state = coord.shared_state
    state.baseline_runtime_sec = BOOT + BENCH
    state.baseline_post_ready_runtime_sec = BENCH
    return coord


@pytest.mark.parametrize("benchmark_mode", ["agentx", ""], ids=["agentx", "fixed_isl_osl"])
def test_after_a_single_pass_baseline_geak_returns_with_its_round_and_a_spare_quarter_left(coordinator, benchmark_mode):
    coordinator.shared_state.benchmark_mode = benchmark_mode

    runner, kill, budget_known = coordinator.phase_kernel._geak_timeouts()

    kill_expected = int(PHASE_LEFT - (ONE_ROUND + SPARE))
    assert (runner, kill, budget_known) == (kill_expected - MARGIN, kill_expected, True)


def test_after_a_warm_pass_geak_returns_with_both_passes_and_a_spare_quarter_left(coordinator):
    coordinator.shared_state.baseline_warm_runtime_sec = WARM

    _runner, kill, _known = coordinator.phase_kernel._geak_timeouts()

    assert kill == int(PHASE_LEFT - (ONE_ROUND + WARM + SPARE))


def test_without_a_measured_baseline_geak_holds_back_the_benchmark_hard_cap(coordinator):
    state = coordinator.shared_state
    state.baseline_runtime_sec = 0.0
    state.baseline_post_ready_runtime_sec = 0.0

    _runner, kill, _known = coordinator.phase_kernel._geak_timeouts()

    assert kill == int(PHASE_LEFT - resolve_benchmark_timeouts()[1])


def test_geak_gets_no_time_when_the_measurement_alone_overruns_the_phase(coordinator, monkeypatch):
    monkeypatch.setattr(ps, "phase_budget_remaining_seconds", lambda _state, **_kw: ONE_ROUND + SPARE - 1.0)

    runner, kill, budget_known = coordinator.phase_kernel._geak_timeouts()

    # Zero falls under GEAK_MIN_RUN_S, so the hand-off is skipped rather than started without room to judge it.
    assert (runner, kill, budget_known) == (0, 0, True)


@pytest.mark.parametrize(
    ("phase_left", "hard_expected"),
    [(PHASE_LEFT, int(PHASE_LEFT - (BOOT + BENCH))), (BOOT + BENCH - 1.0, 0)],
    ids=["funded", "measurement_alone_overruns"],
)
def test_the_rewrite_controller_returns_with_one_measurement_left_in_the_phase(
    coordinator, monkeypatch, phase_left, hard_expected
):
    monkeypatch.setattr(ps, "phase_budget_remaining_seconds", lambda _state, **_kw: phase_left)
    monkeypatch.setattr(ps, "session_remaining_seconds", lambda _state, **_kw: SESSION_LEFT)
    monkeypatch.setattr(ps, "phase_cap_seconds", lambda _state: None)

    soft, hard = coordinator.phase_kernel._kernel_rewrite_controller_timeouts()

    # Zero takes the no_result branch: the Controller is not started without room to integrate its patches.
    assert (soft, hard) == (max(0, hard_expected - 30), hard_expected)
