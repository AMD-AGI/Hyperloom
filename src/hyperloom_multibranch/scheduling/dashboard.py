# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session dashboard in ``MB-dashboard/``: rebuilt every judge interval during EXPLORE and once more at CLOSE.

Each build reads the blackboard, the frontier-scored progress and the toolbox's queue at that moment.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from hyperloom_multibranch.contracts.blackboard.interfaces import ProgressQuery, SnapshotQuery
from hyperloom_multibranch.contracts.toolbox import QueueState, Toolbox
from hyperloom_multibranch.blackboard.dashboard import DashboardFiles, build_dashboard, write_dashboard


class SessionDashboard:
    """Writes the dashboard of ``blackboard`` into ``directory``; ``metric`` names the gains it shows."""

    def __init__(
        self, blackboard: SnapshotQuery, progress: ProgressQuery, toolbox: Toolbox, metric: str, directory: Path
    ) -> None:
        self._blackboard = blackboard
        self._progress = progress
        self._toolbox = toolbox
        self._metric = metric
        self._directory = directory

    async def refresh(self) -> DashboardFiles:
        """Rebuild the dashboard with the queue the running toolbox reports now."""

        return await self.write(await self._toolbox.queue_state())

    async def write(self, queue: QueueState | None) -> DashboardFiles:
        """Rebuild the dashboard with ``queue``, ``None`` when the toolbox reported none."""

        snapshot = await self._blackboard.snapshot()
        dashboard = build_dashboard(snapshot, await self._progress.progress(), metric=self._metric, queue=queue)
        return await asyncio.to_thread(write_dashboard, self._directory, dashboard)


__all__ = ["SessionDashboard"]
