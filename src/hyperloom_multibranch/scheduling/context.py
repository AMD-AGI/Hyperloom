# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What every part of one session's scheduler shares: :class:`SchedulerContext`."""

from __future__ import annotations

from dataclasses import dataclass

from hyperloom_multibranch.contracts.fork import ReservedBranches
from hyperloom_multibranch.contracts.blackboard import Blackboard
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import Roster


@dataclass(frozen=True)
class SchedulerContext:
    """One session's scheduler state and services: the session directory, the roster, the scheduler's blackboard
    handle, the decision log, the registry of every process the session starts, the clock it runs on, the fork
    desk that admits and queues the rule gate's branch requests onto the roster, and the rule gate's side of the
    branches the desk reserved, which lands each admitted fork and records each voided fork and repair."""

    layout: SessionLayout
    roster: Roster
    blackboard: Blackboard
    decisions: DecisionLog
    processes: ProcessRegistry
    clock: Clock
    desk: ForkDesk
    reserved: ReservedBranches


__all__ = ["SchedulerContext"]
