# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How the scheduler starts, stops and resizes the long-running specialist on a branch."""

from __future__ import annotations

from typing import Protocol


class SpecialistLifecycle(Protocol):
    """Long-running specialists, one per branch; the runtime restarts a crashed one on the same branch by itself.

    The runtime reads the branch's startup brief from the notebook. A started branch may have one test running until
    ``set_k`` says otherwise, and shrinking a branch to nothing is ``stop``, not ``set_k(branch, 0)``. Every process
    the runtime starts goes through the session's process registry, so the session stop kills it even when ``stop``
    hangs.
    """

    async def start(self, branch: str, start_commit: str) -> None: ...

    async def stop(self, branch: str) -> None:
        """Stop the branch's specialist; returns once no process of it is left."""

    async def set_k(self, branch: str, k: int) -> None:
        """How many tests the branch's specialist may have running at once."""


__all__ = ["SpecialistLifecycle"]
