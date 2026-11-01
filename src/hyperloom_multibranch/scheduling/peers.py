# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The peers a session runs on and what they are built on.

A :data:`PeerFactory` builds the specialist lifecycle, the toolbox, the experiment repository, the rule gate's
session service, the session's MCP service and its briefs on a :class:`PeerSetup`: the session directory, the workload the session optimizes, the scheduler's
answer to the rule gate's branch requests, the session's process registry, its blob store, the critic that
reviews a declared correctness fix and the reference material the run was given. The specialist runtime builds its own rule gates on the branch requests; the
session gate judges the tests the scheduler submits itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom.measure.compile import BlobStore
from hyperloom_multibranch.contracts.fork import ForkAdmissionService, ReservedBranches
from hyperloom_multibranch.contracts.gate import RegressionCheck, SessionGate
from hyperloom_multibranch.contracts.lifecycle import SpecialistLifecycle
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.blackboard.rsi.briefing import SessionBriefs
from hyperloom_multibranch.rulegate.critic import FixReviewer
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.settings import AGENT_PORT, Workload

if TYPE_CHECKING:
    from hyperloom.hosted.mcp_server import McpServer
    from hyperloom_multibranch.specialist.dossier_pass import DossierPass
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox


@dataclass(frozen=True)
class Peers:
    """``repo`` holds nothing until PRELUDE roots it with ``init_root``; ``gate`` writes the baseline PRELUDE measures.

    ``mcp`` serves the specialist tools; the session stop stops it. ``briefs`` brief every specialist; they are built
    before PRELUDE has read the warm-start leads, which the session binds to them once it has. ``base_image_digest``
    is the image the toolbox measures on, which every recipe must carry: PRELUDE roots the session on it.
    ``reserved`` is the rule gate that asks for the forks and repairs: it lands each fork the fork desk admits before
    the scheduler starts it, and records each fork and repair the scheduler voids. ``regression`` runs the
    regression set the integrator asks for before a merge and FINAL for every candidate. ``on_this_machine`` says
    whether the toolbox measures on this machine's cards, which the PRELUDE child runs on too; PRELUDE starts none
    where it does not. ``dossier_pass`` runs the session's dossier pass before EXPLORE opens its agent branches; a
    session that runs no agent has none.
    """

    lifecycle: SpecialistLifecycle
    toolbox: RealToolbox
    repo: ExperimentRepo
    gate: SessionGate
    mcp: McpServer
    briefs: SessionBriefs
    base_image_digest: str
    reserved: ReservedBranches
    regression: RegressionCheck
    on_this_machine: bool
    dossier_pass: DossierPass | None = None


@dataclass(frozen=True)
class PeerSetup:
    """What one session's peers are built on.

    ``workload`` is the measurement surface: the toolbox measures every test and sweep of the session on it, so a
    test request carries no operating point of its own. ``desk`` is what the specialist runtime builds its rule gates
    on. ``processes`` is the session's process registry: the session stop kills what is registered there even when a
    peer's own stop hangs, and reports what outlives that kill; the specialist runtime registers every process the AI
    work socket starts for it with ``processes.adopt``. The toolbox's sandboxes are its own: the session stop releases
    them by closing the toolbox. ``blobs`` holds the patch and kernel files the recipes name; the toolbox's
    environment compiler reads them from it. ``critic`` decides whether the defect a proven correctness fix removes is
    reachable under the workload. ``references`` names each directory of reference material the run was given, already
    copied into the session, for the specialist role to send the specialists to. ``dev_leases`` is how many dev
    sandboxes the session lends its specialists at once, from the toolbox's cards; 0 lends none. ``agents`` are the
    third-party agents the user named, each run on an autonomous branch of its own, ``agent_pack`` the pack an
    Arbor among them runs (``""`` for the adapter's default), ``agent_launch`` the file of the launch its recipe's
    adds to (``""`` when the recipe's is the whole launch), ``agent_port`` the first port of their port blocks, and
    ``agent_dossier`` the dossier the run gives every agent branch with its reviewed map (``""`` for none).
    """

    layout: SessionLayout
    workload: Workload
    desk: ForkAdmissionService
    processes: ProcessRegistry
    blobs: BlobStore
    critic: FixReviewer
    references: Mapping[str, Path]
    dev_leases: int
    agents: tuple[str, ...]
    agent_pack: str = ""
    agent_launch: str = ""
    agent_port: int = AGENT_PORT
    agent_dossier: str = ""


PeerFactory = Callable[[PeerSetup], Peers]


class PeersUnavailable(RuntimeError):
    """A peer a session needs cannot be built on this machine; the message says what is missing."""


__all__ = [
    "PeerFactory",
    "PeerSetup",
    "Peers",
    "PeersUnavailable",
]
