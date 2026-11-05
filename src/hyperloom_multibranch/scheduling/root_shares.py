# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The root's op shares, for the agents' map: waited for until the root profile's analysis delivered them.

PRELUDE profiles the root and has the toolbox analyse the profile, naming the analysis's job. Before the session maps
its agents' directions, it waits for that job and, once the job succeeded, for the root's op shares to reach the
blackboard with its result, at most :data:`WAIT_SECONDS` in all, then takes them from the blackboard.

They are the shares the toolbox took from the analysis (``toolbox.tools.op_shares``): TraceLens' ops, or the kernels
its report route ranks, which still give shares on a CUDA-graph trace, where TraceLens' op decomposition degrades.
Arbor's map generator and its dossier pass take kernel names as they take ops. A job that did not succeed leaves the
session mapping without shares at once; one whose analysis gave none, once the wait is over.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.specialist.agent_onboarding import ROOT_PROFILE_SOURCE

if TYPE_CHECKING:
    from hyperloom_multibranch.contracts.blackboard.interfaces import RecordSearch
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

#: How long the session waits for the root profile's job and its shares before it maps without them.
WAIT_SECONDS = 600.0
POLL_SECONDS = 5.0


@dataclass(frozen=True)
class RootShares:
    """The root's shares, and what they are, as the profile file Arbor reads names them (``source``)."""

    shares: OpShares
    source: str


@dataclass(frozen=True)
class SharesRead:
    """What :func:`root_shares` found: the shares (``None`` without), and what it did, for the decision log."""

    shares: RootShares | None
    detail: dict[str, object]


@dataclass(frozen=True)
class ShareSources:
    """Where :func:`root_shares` reads: the toolbox's jobs and the blackboard's op shares, on the session's clock.
    ``toolbox_dir``, the session's toolbox directory, is not read: the shares come from the blackboard."""

    blackboard: RecordSearch
    toolbox: RealToolbox
    clock: Clock
    toolbox_dir: Path


async def root_shares(
    sources: ShareSources, root_commit: str, job: str, wait_seconds: float = WAIT_SECONDS
) -> SharesRead:
    """The op shares of ``root_commit`` once its profile's ``job`` ended and, when it succeeded, they reached the
    blackboard, waited for at most ``wait_seconds`` in all."""

    clock = sources.clock
    waited, landing = await _wait(sources.toolbox, clock, job, wait_seconds)
    own = await _landed(sources.blackboard, clock, root_commit, landing)
    detail: dict[str, object] = {"job_id": job, "job": waited}
    if own is None:
        return SharesRead(None, {**detail, "taken": "none: no op shares"})
    return SharesRead(RootShares(own, ROOT_PROFILE_SOURCE), {**detail, "taken": "op shares", "ops": len(own.shares)})


async def _wait(toolbox: RealToolbox, clock: Clock, job: str, seconds: float) -> tuple[str, float]:
    """Wait while ``job`` runs, at most ``seconds``: how it ended, as the decision says it, and until when its op
    shares may still reach the blackboard: the end of those ``seconds`` once it succeeded, its result carrying them,
    else now."""

    if not job:
        return "no root profile was started", clock.now()
    from hyperloom_multibranch.toolbox.tools.base import JobStatus  # the toolbox's tools load only when one runs

    until = clock.now() + seconds
    while True:
        try:
            status = toolbox.tool_status(job)
        except LookupError:
            return "the toolbox knows no such job", clock.now()
        if status is not JobStatus.RUNNING:
            return status.value, until if status is JobStatus.SUCCEEDED else clock.now()
        if clock.now() >= until:
            return f"still running after {seconds:.0f} s", until
        await clock.sleep(POLL_SECONDS)


async def _landed(blackboard: RecordSearch, clock: Clock, root_commit: str, until: float) -> OpShares | None:
    """The op shares of ``root_commit`` on the blackboard, read every poll until they are there or ``until`` passed;
    ``None`` without. A job's status turns final just before its result, with the shares, reaches the blackboard."""

    while True:
        own = await blackboard.op_shares(root_commit)
        if own is not None and own.shares:
            return own
        if clock.now() >= until:
            return None
        await clock.sleep(POLL_SECONDS)


__all__ = [
    "WAIT_SECONDS",
    "RootShares",
    "ShareSources",
    "SharesRead",
    "root_shares",
]
