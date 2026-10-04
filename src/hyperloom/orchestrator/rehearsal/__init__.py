# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Play a whole bring-up round in this process: no GPU, no subprocess, no waiting."""

from __future__ import annotations

from hyperloom.orchestrator.rehearsal.clock import VirtualClock, installed_clock
from hyperloom.orchestrator.rehearsal.launch import (
    RecordedLaunch,
    ScenarioExhausted,
    ScriptedLaunchBackend,
)
from hyperloom.orchestrator.rehearsal.scenario import (
    DIED_SILENTLY,
    HANG,
    READY,
    STAGE_FAILED,
    LaunchAttempt,
    LaunchScenario,
    ScenarioError,
    boot_log_for,
)
from hyperloom.orchestrator.rehearsal.specialist import (
    COMPLETED,
    CRASHED,
    EMPTY,
    HUNG,
    ScriptedSpecialist,
    SpecialistStep,
)

__all__ = [
    "COMPLETED",
    "CRASHED",
    "DIED_SILENTLY",
    "EMPTY",
    "HANG",
    "HUNG",
    "READY",
    "STAGE_FAILED",
    "LaunchAttempt",
    "LaunchScenario",
    "RecordedLaunch",
    "ScenarioError",
    "ScenarioExhausted",
    "ScriptedLaunchBackend",
    "ScriptedSpecialist",
    "SpecialistStep",
    "VirtualClock",
    "boot_log_for",
    "installed_clock",
]
