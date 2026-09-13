# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Out-of-band supervision: spawn and stop the watcher in :mod:`.watch`."""

from __future__ import annotations

from hyperloom.orchestrator.supervisor.launcher import (
    SUPERVISOR_ENABLE_ENV,
    SUPERVISOR_STALL_ENV,
    spawn_supervisor,
    stop_supervisor,
    tick_stall_sec,
)

__all__ = [
    "SUPERVISOR_ENABLE_ENV",
    "SUPERVISOR_STALL_ENV",
    "spawn_supervisor",
    "stop_supervisor",
    "tick_stall_sec",
]
