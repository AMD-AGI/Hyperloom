# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's dossier pass: Arbor run once before the agent branches open, to map the directions they split.

With two or more Arbor agents and no map given, the session runs the first Arbor source once as its dossier pass
(:data:`~hyperloom_multibranch.specialist.arbor_adapter.DOSSIER_PASS`, ``component.stop_at: "map"``), as Arbor's
``docs/hosts.md`` describes it:
- **where:** on a worktree at the commit the branches open on, with that commit's launch, on the cards of one lease;
- **with what:** the root's shares as its ``component.profile``, until its deadline;
- **what it leaves:** the dossier and its map, then exit 0 with ``mapped``. The map is what the session splits
  (:mod:`~hyperloom_multibranch.scheduling.agent_plans`), and the dossier is copied into each branch's checkout.

No branch holds the pass. Its lease is charged to the integration branch, as PRELUDE's child's is, and it writes nothing
to the blackboard. Whatever ends it, its process is killed and its cards are given back. A pass that sets no map says
why (:class:`DossierOutcome`), and the session falls back to the generic map.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom.sandbox import SandboxUnavailable
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import BranchLease, LeaseCharge, RequestRejected
from hyperloom_multibranch.specialist.agent_onboarding import (
    ROOT_PROFILE_SOURCE,
    START_SECONDS,
    AgentSandbox,
    AgentSpec,
    ModelRefused,
    OnboardingSetup,
    ReadyAgent,
    make_ready,
    tail,
)
from hyperloom_multibranch.specialist.arbor_adapter import (
    DOSSIER_PASS,
    EXIT_REFUSED,
    MAP_FILE,
    MAPPED,
    mapped,
    pass_dossier,
    refused_reasons,
)
from hyperloom_multibranch.specialist.runtime import PreflightFailed
from hyperloom_multibranch.specialist.tools import sync_to_commit
from hyperloom_multibranch.specialist.worktree import BranchWorktree, WorktreeError

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

#: The holder of the pass's lease, and the name of its worktree and directory beside the branches': it is no branch.
DOSSIER_HOLDER = "dossier"
#: The pass's card time is the integration branch's, as the PRELUDE child's is.
DOSSIER_CHARGE = LeaseCharge(LeaseKind.PRELUDE_CHILD, INTEGRATION_BRANCH)
POLL_SECONDS = 10.0
#: How long past its deadline the pass has to write its exit and end of itself before it is killed: Arbor ends a run
#: by its deadline on its own.
EXIT_GRACE_SECONDS = 120.0


@dataclass(frozen=True)
class DossierStart:
    """``source``, the Arbor to run; ``start_commit``, the commit the branches open on; ``deadline``, the clock time the
    pass ends by; ``shares``, the root's shares for its profile (``None`` without), which came from
    ``shares_source``."""

    source: str
    start_commit: str
    deadline: float
    shares: OpShares | None = None
    shares_source: str = ROOT_PROFILE_SOURCE


@dataclass(frozen=True)
class DossierOutcome:
    """How the pass ended: ``map``, the directions it set, and ``dossier``, the dossier it wrote, each ``None`` without;
    ``reason``, :data:`~hyperloom_multibranch.specialist.arbor_adapter.MAPPED` when it set its map, else why not."""

    map: Path | None
    dossier: Path | None
    reason: str


DossierPass = Callable[[DossierStart], Awaitable[DossierOutcome]]


@dataclass(frozen=True)
class DossierSetup:
    """What the pass runs on: ``spec``, the session's agent spec with the pass's port and client; ``repo`` and
    ``toolbox``; ``sandbox``, the sandbox of a lease; ``worktree``, the pass's own; and ``agent_dir``, where its Arbor
    is fetched. ``start_seconds`` is how long its start may take (:class:`OnboardingSetup`), ``exit_grace_seconds`` how
    long past its deadline it has to end of itself."""

    spec: AgentSpec
    repo: ExperimentRepo
    toolbox: RealToolbox
    sandbox: Callable[[BranchLease], AgentSandbox]
    worktree: BranchWorktree
    agent_dir: Path
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    poll_seconds: float = POLL_SECONDS
    start_seconds: float = START_SECONDS
    exit_grace_seconds: float = EXIT_GRACE_SECONDS


async def _arbor_only(_checkout: Path, _prompt: str) -> str:
    raise ModelRefused("the dossier pass runs Arbor, which is not onboarded by a model, and this source is not Arbor")


async def run_dossier_pass(start: DossierStart, setup: DossierSetup) -> DossierOutcome:
    """Run the dossier pass of ``start`` on ``setup``, and say how it ended; never raises for the pass's own failures."""

    worktree = setup.worktree
    try:
        await sync_to_commit(setup.repo, start.start_commit, worktree)
        recipe = await setup.repo.read_recipe(start.start_commit)
    except (WorktreeError, OSError) as error:
        return DossierOutcome(None, None, f"its worktree could not be laid out: {error}")
    seconds = start.deadline - setup.clock()
    if seconds <= 0:
        return DossierOutcome(None, None, "no time was left before its deadline")
    spec = replace(
        setup.spec,
        server_args=recipe.launch.server_args,
        envs=dict(recipe.launch.envs),
        workspace=str(worktree.path),
        seconds=seconds,
        deadline=start.deadline,
        root=worktree.commit(),
        root_shares=start.shares,
        root_shares_source=start.shares_source,
        plan=None,
    )
    try:
        lease = await setup.toolbox.lease_branch_sandbox(
            DOSSIER_HOLDER, spec.tp, charge=DOSSIER_CHARGE, lifetime_sec=seconds + setup.exit_grace_seconds
        )
    except (RequestRejected, SandboxUnavailable) as error:
        return DossierOutcome(None, None, f"no cards were leased for it: {error}")
    ready: ReadyAgent | None = None
    try:
        onboarding = OnboardingSetup(
            setup.sandbox(lease),
            _arbor_only,
            run=DOSSIER_PASS,
            start_seconds=setup.start_seconds,
            poll_seconds=setup.poll_seconds,
            sleep=setup.sleep,
        )
        try:
            ready = await make_ready(start.source, replace(spec, cards=lease.gpu_ids), setup.agent_dir, onboarding)
        except PreflightFailed as failed:
            return DossierOutcome(None, None, f"it did not start: {'; '.join(failed.reasons)}")
        return await _ended(ready, start, setup)
    finally:
        if ready is not None and ready.process.returncode is None:
            await ready.process.kill()
        await asyncio.shield(setup.toolbox.release_branch_sandbox(lease.lease_id))


async def _ended(ready: ReadyAgent, start: DossierStart, setup: DossierSetup) -> DossierOutcome:
    """Wait for the pass to end, until a little past its deadline, and read how it did."""

    process, checkout = ready.process, ready.source_dir
    while process.returncode is None and setup.clock() < start.deadline + setup.exit_grace_seconds:
        await setup.sleep(setup.poll_seconds)
    session = checkout / DOSSIER_PASS.session_dir
    dossier = pass_dossier(checkout, session)
    if process.returncode is None:
        return DossierOutcome(None, dossier, "it set no map by its deadline and was stopped")
    if process.returncode == 0 and mapped(session):
        return DossierOutcome(session / MAP_FILE, dossier, MAPPED)
    if process.returncode == EXIT_REFUSED:
        why = refused_reasons(checkout / ready.manifest.snapshots_refused) or tail(await process.output())
        return DossierOutcome(None, None, f"it refused its config: {why}")
    if process.returncode == 0:
        said = _exit_reason(checkout / ready.manifest.snapshots_exit)
        return DossierOutcome(None, dossier, f"it ended without a map: {said}")
    return DossierOutcome(None, dossier, f"it exited {process.returncode}: {tail(await process.output())}")


def _exit_reason(path: Path) -> str:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "it wrote no exit reason"
    reason = document.get("reason") if isinstance(document, dict) else None
    return reason if isinstance(reason, str) and reason else "its exit file names no reason"


__all__ = [
    "DOSSIER_CHARGE",
    "DOSSIER_HOLDER",
    "DossierOutcome",
    "DossierPass",
    "DossierSetup",
    "DossierStart",
    "run_dossier_pass",
]
