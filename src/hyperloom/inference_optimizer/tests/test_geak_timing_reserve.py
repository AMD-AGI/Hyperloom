# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GEAK leaves the declared canonical rebench and closing allowance intact."""

from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.phases import kernel


@pytest.mark.parametrize(
    ("remaining", "phase_remaining", "reserve", "expected"),
    [
        (23400, 18300, None, (18000, 18300, True)),
        (23400, 18300, "5400", (17580, 17880, True)),
        (24300, 18225, "5400", (17925, 18225, True)),
        (6000, 5000, "5400", (180, 480, True)),
        (5000, 4500, "5400", (0, 0, True)),
    ],
)
def test_declared_rebench_is_reserved_before_the_runner_margin(
    monkeypatch, remaining, phase_remaining, reserve, expected
):
    monkeypatch.setenv("GEAK_E2E_TIMEOUT_S", "7200")
    monkeypatch.setenv("GEAK_BUDGET_MARGIN_S", "300")
    if reserve is None:
        monkeypatch.delenv("GEAK_REBENCH_RESERVE_S", raising=False)
    else:
        monkeypatch.setenv("GEAK_REBENCH_RESERVE_S", reserve)
    monkeypatch.setattr(kernel._phase_state, "phase_budget_remaining_seconds", lambda *a, **kw: phase_remaining)
    phase = SimpleNamespace(
        _coord=SimpleNamespace(run_deadline=SimpleNamespace(remaining=lambda: remaining)),
        shared_state=SimpleNamespace(closing_reserve_sec=lambda: 120),
    )
    assert kernel.KernelPhase._geak_timeouts(phase) == expected


def test_unbounded_runner_keeps_existing_explicit_timeout(monkeypatch):
    monkeypatch.setenv("GEAK_E2E_TIMEOUT_S", "7200")
    monkeypatch.setenv("GEAK_REBENCH_RESERVE_S", "5400")
    phase = SimpleNamespace(_coord=SimpleNamespace(run_deadline=None))
    assert kernel.KernelPhase._geak_timeouts(phase) == (7200, 7800, False)
