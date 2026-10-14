# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The peers a session runs on and what they are built on.

A :data:`PeerFactory` builds the specialist lifecycle, the toolbox and the experiment repository on a
:class:`PeerSetup`: the session directory, the workload the session optimizes, the scheduler's answer to the rule
gate's branch requests, the session's process registry and its blob store. The rule gate itself is no peer of the
scheduler: the specialist runtime builds it on the branch requests.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom.contracts.recipe import Recipe
from hyperloom.measure.compile import BlobStore
from hyperloom_multibranch.contracts.fork import BranchRequests
from hyperloom_multibranch.contracts.lifecycle import SpecialistLifecycle
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.settings import Workload

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

#: Writes the baseline recipe as the experiment repository's root commit, which starts the integration branch, and
#: returns that commit. The repository contract has no such call: its git implementation is created with its root.
RootWriter = Callable[[Recipe], Awaitable[str]]


@dataclass(frozen=True)
class Peers:
    """``repo`` is read and written only once ``write_root`` has written the root commit, which PRELUDE does."""

    lifecycle: SpecialistLifecycle
    toolbox: RealToolbox
    repo: ExperimentRepo
    write_root: RootWriter


def git_root_writer(path: Path) -> RootWriter:
    """Creates the session's git experiment repository at ``path`` with the baseline as its root; a
    :class:`~hyperloom_multibranch.repo.git_repo.GitExperimentRepo` on ``path`` reads and writes it from then on."""

    async def write_root(baseline: Recipe) -> str:
        created = await GitExperimentRepo.create(path, baseline)
        return await created.head(INTEGRATION_BRANCH)

    return write_root


@dataclass(frozen=True)
class PeerSetup:
    """What one session's peers are built on.

    ``workload`` is the measurement surface: the toolbox measures every test and sweep of the session on it, so a
    test request carries no operating point of its own. ``desk`` is what the specialist runtime builds its rule gates
    on. ``processes`` is the session's process registry: the session stop kills what is registered there even when a
    peer's own stop hangs, and reports what outlives that kill. The lifecycle contract does not require a lifecycle to
    register its processes there, so a specialist process it does not register is stopped only by the lifecycle's own
    ``stop``. The toolbox's sandboxes are its own: the session stop releases them by closing the toolbox. ``blobs``
    holds the patch and kernel files the recipes name; the toolbox's environment compiler reads them from it.
    """

    layout: SessionLayout
    workload: Workload
    desk: BranchRequests
    processes: ProcessRegistry
    blobs: BlobStore


PeerFactory = Callable[[PeerSetup], Peers]


class PeersUnavailable(RuntimeError):
    """A peer module a session needs has no implementation; ``modules`` names them."""

    def __init__(self, modules: tuple[str, ...]) -> None:
        super().__init__("no implementation yet in: " + ", ".join(modules))
        self.modules = modules


__all__ = [
    "PeerFactory",
    "PeerSetup",
    "Peers",
    "PeersUnavailable",
    "RootWriter",
    "git_root_writer",
]
