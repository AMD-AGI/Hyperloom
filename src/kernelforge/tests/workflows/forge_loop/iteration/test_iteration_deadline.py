# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An external task deadline bounds every iteration admission decision."""

from __future__ import annotations

import pytest

from kernelforge.workflows.forge_loop.iteration import round_admission
from kernelforge.workflows.forge_loop.iteration.iteration_types import IterationConfig
from kernelforge.workflows.forge_loop.iteration.round_admission import RoundAdmissionMixin
from kernelforge.workflows.forge_loop.campaign.run_state import LoopStateStore, RunState


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    loop = RoundAdmissionMixin()
    loop.start_time = 1000.0
    loop.ic = IterationConfig(
        kernel_file="kernel.py",
        driver_script="driver.py",
        max_time_hours=1.0,
        deadline_unix=2800.0,
    )
    loop.run_state = RunState()
    loop.state_store = LoopStateStore(str(tmp_path))
    monkeypatch.setattr(round_admission.time, "time", lambda: 1600.0)
    return loop


@pytest.mark.parametrize(
    ("max_hours", "deadline", "expected_remaining"),
    [
        (1.0, 2800.0, 1200.0),
        (0.25, 5000.0, 300.0),
        (1.0, None, 3000.0),
        (1.0, 1600.0, 0.0),
        (1.0, 1599.0, 0.0),
        (0.1, 5000.0, 0.0),
    ],
)
def test_remaining_time_honors_the_earliest_deadline(campaign, max_hours, deadline, expected_remaining):
    campaign.ic.max_time_hours = max_hours
    campaign.ic.deadline_unix = deadline

    assert campaign._time_remaining() == expected_remaining


def test_external_deadline_holds_the_existing_finalization_reserve(campaign):
    assert campaign.ic.budget_reserve_sec == 1800
    assert campaign._is_budget_exhausted() is True
    assert campaign._analysis_deadline_unix() == 1600.0


def test_round_admission_cannot_spend_the_floored_hour_past_external_deadline(campaign):
    assert campaign._admit_next_round(1) is None
    assert campaign.termination_reason == "round_budget_exhausted"


def test_dispatch_cannot_spend_the_floored_hour_past_external_deadline(campaign):
    assert campaign._admit_dispatch(1) is False
    assert campaign.termination_reason == "round_budget_exhausted"


def test_equal_reserve_remains_admissible_without_double_reserving(campaign, monkeypatch):
    monkeypatch.setattr(round_admission.time, "time", lambda: 1000.0)

    assert campaign._time_remaining() == campaign.ic.budget_reserve_sec
    assert campaign._is_budget_exhausted() is False
    assert campaign._admit_dispatch(1) is True
