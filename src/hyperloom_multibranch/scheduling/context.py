# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What every part of one session's scheduler shares: :class:`SchedulerContext`."""

from __future__ import annotations

from dataclasses import dataclass

from hyperloom_multibranch.contracts.notebook import Notebook
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import Roster


@dataclass(frozen=True)
class SchedulerContext:
    """One session's scheduler state and services: the session directory, the roster, the scheduler's notebook
    handle, the decision log, the registry of every process the session starts and the clock it runs on."""

    layout: SessionLayout
    roster: Roster
    notebook: Notebook
    decisions: DecisionLog
    processes: ProcessRegistry
    clock: Clock


__all__ = ["SchedulerContext"]
