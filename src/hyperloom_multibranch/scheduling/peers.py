# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The peers a session runs on and what they are built on.

A :data:`PeerFactory` builds the specialist lifecycle, the toolbox, the experiment repository and the rule gate's
session service on a :class:`PeerSetup`: the session directory, the workload the session optimizes, the scheduler's
answer to the rule gate's branch requests, the session's process registry and its blob store. The specialist runtime
builds its own rule gates on the branch requests; the session gate judges the tests the scheduler submits itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hyperloom.measure.compile import BlobStore
from hyperloom_multibranch.contracts.fork import ForkAdmissionService
from hyperloom_multibranch.contracts.gate import SessionGate
from hyperloom_multibranch.contracts.lifecycle import SpecialistLifecycle
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.settings import Workload

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox


@dataclass(frozen=True)
class Peers:
    """``repo`` holds nothing until PRELUDE roots it with ``init_root``; ``gate`` writes the baseline PRELUDE measures."""

    lifecycle: SpecialistLifecycle
    toolbox: RealToolbox
    repo: ExperimentRepo
    gate: SessionGate


@dataclass(frozen=True)
class PeerSetup:
    """What one session's peers are built on.

    ``workload`` is the measurement surface: the toolbox measures every test and sweep of the session on it, so a
    test request carries no operating point of its own. ``desk`` is what the specialist runtime builds its rule gates
    on. ``processes`` is the session's process registry: the session stop kills what is registered there even when a
    peer's own stop hangs, and reports what outlives that kill; the specialist runtime starts every process of its
    own through it. The toolbox's sandboxes are its own: the session stop releases them by closing the toolbox. ``blobs``
    holds the patch and kernel files the recipes name; the toolbox's environment compiler reads them from it.
    """

    layout: SessionLayout
    workload: Workload
    desk: ForkAdmissionService
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
]
