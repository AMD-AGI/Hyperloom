# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hold the GEAK delegation phase at the gate in front of its runner launch."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.phases.kernel import KernelPhase


def stop_geak_before_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave GEAK no run budget, so the phase returns once the handoff is written."""
    monkeypatch.setattr(KernelPhase, "_geak_timeouts", lambda _self: (0, 0, True))


def forbid_geak_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the phase gets past its recovery short-circuit to the runner launch."""

    def _launch_reached(_self: KernelPhase) -> tuple[int, int, bool]:
        raise AssertionError("GEAK runner launch reached")

    monkeypatch.setattr(KernelPhase, "_geak_timeouts", _launch_reached)
