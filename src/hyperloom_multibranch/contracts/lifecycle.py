# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How the scheduler starts, stops and resizes the long-running specialist on a branch."""

from __future__ import annotations

from typing import Protocol

from hyperloom_multibranch.contracts.fork import RepairRequest


class SpecialistLifecycle(Protocol):
    """Long-running specialists, one per branch; the runtime restarts a crashed one on the same branch by itself.

    ``brief`` is the startup brief the scheduler hands a new specialist (the warm-start leads for the first batch, the
    cross-session brief for a fill); the runtime puts it into the specialist's first prompt and every restart prompt.
    ``repair`` is the request an admitted repair branch was opened for, started at its intent's parent commit; every
    prompt of the branch's specialist shows it as the branch's task, whatever brief a launch is rebuilt with.
    A started branch may have one test running until ``set_k`` says otherwise, and shrinking a branch to nothing is
    ``stop``, not ``set_k(branch, 0)``. Every process the runtime starts goes through the session's process registry,
    so the session stop kills it even when ``stop`` hangs.
    """

    async def start(
        self, branch: str, start_commit: str, brief: str = "", repair: RepairRequest | None = None
    ) -> None: ...

    async def stop(self, branch: str) -> None:
        """Stop the branch's specialist; returns once no process of it is left."""

    async def set_k(self, branch: str, k: int) -> None:
        """How many tests the branch's specialist may have running at once."""


__all__ = ["SpecialistLifecycle"]
