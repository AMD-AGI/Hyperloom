# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How the scheduler starts, stops and resizes the long-running specialist on a branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpecialistLaunch:
    """``k`` is how many tests the branch may have in flight; ``brief`` is the rendered launch brief a refill carries."""

    branch: str
    start_commit: str
    k: int
    brief: str | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("a specialist needs room for at least one test in flight")


class SpecialistLifecycle(Protocol):
    """One specialist per branch; shrinking a branch to nothing is ``stop``, not ``set_k(branch, 0)``."""

    async def start(self, launch: SpecialistLaunch) -> None: ...

    async def stop(self, branch: str) -> None: ...

    async def set_k(self, branch: str, k: int) -> None: ...


__all__ = ["SpecialistLaunch", "SpecialistLifecycle"]
