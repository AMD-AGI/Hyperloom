# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Play a whole bring-up round in this process: no GPU, no subprocess, no waiting."""

from __future__ import annotations

from hyperloom.orchestrator.rehearsal.clock import VirtualClock, installed_clock
from hyperloom.orchestrator.rehearsal.launch import (
    RecordedRound,
    ScenarioExhausted,
    ScriptedRounds,
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

__all__ = [
    "DIED_SILENTLY",
    "HANG",
    "READY",
    "STAGE_FAILED",
    "LaunchAttempt",
    "LaunchScenario",
    "RecordedRound",
    "ScenarioError",
    "ScenarioExhausted",
    "ScriptedRounds",
    "VirtualClock",
    "boot_log_for",
    "installed_clock",
]
