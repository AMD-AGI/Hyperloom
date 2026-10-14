# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The peers a session runs on, and the placeholder that wires them until their owners' modules land.

A :data:`PeerFactory` builds the specialist lifecycle, the toolbox, the rule gate and the experiment repository on a
:class:`PeerSetup`: the session directory, the workload the session optimizes, the scheduler's fork admission desk for
the rule gate, and the session's process registry. No owner has delivered a real
implementation yet, so :func:`mb_fake_load_peers` refuses, and the command line fails before it creates anything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from hyperloom_multibranch.contracts.scheduling import (
    ExperimentRepo,
    ForkAdmissionService,
    RuleGate,
    SpecialistLifecycle,
    Toolbox,
)
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.settings import Workload

#: The modules a real session needs and that hold no implementation yet; emptied as each module lands, and deleted
#: with :func:`mb_fake_load_peers` once the last one has.
MB_FAKE_MISSING_PEER_MODULES: tuple[str, ...] = (
    "hyperloom_multibranch.specialist",
    "hyperloom_multibranch.toolbox",
    "hyperloom_multibranch.rulegate",
    "hyperloom_multibranch.repo",
)


@dataclass(frozen=True)
class Peers:
    lifecycle: SpecialistLifecycle
    toolbox: Toolbox
    gate: RuleGate
    repo: ExperimentRepo


@dataclass(frozen=True)
class PeerSetup:
    """What one session's peers are built on.

    ``workload`` is the measurement surface: the toolbox measures every test and sweep of the session on it, so a
    test request carries no operating point of its own. The lifecycle and the toolbox start every process of theirs
    through ``processes``, so the session stop kills it even when a peer's own stop hangs, and the session reports it
    when it outlives that kill.
    """

    layout: SessionLayout
    workload: Workload
    desk: ForkAdmissionService
    processes: ProcessRegistry


PeerFactory = Callable[[PeerSetup], Peers]


class PeersUnavailable(RuntimeError):
    """A peer module a session needs has no implementation; ``modules`` names them."""

    def __init__(self, modules: tuple[str, ...]) -> None:
        super().__init__("no implementation yet in: " + ", ".join(modules))
        self.modules = modules


def mb_fake_load_peers() -> PeerFactory:
    """Always raises :class:`PeersUnavailable` naming :data:`MB_FAKE_MISSING_PEER_MODULES`.

    Stands in for the loader of the specialist runtime, the toolbox, the rule gate and the experiment repository;
    replaced by the real loader once those modules land. It takes no argument so the command line can call it
    before it creates the session directory.
    """

    raise PeersUnavailable(MB_FAKE_MISSING_PEER_MODULES)


__all__ = [
    "MB_FAKE_MISSING_PEER_MODULES",
    "PeerFactory",
    "PeerSetup",
    "Peers",
    "PeersUnavailable",
    "mb_fake_load_peers",
]
