# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The root's shares, for the agents' map: waited for, and read from the root profile's kernels when TraceLens gave no
op shares.

PRELUDE starts the analysis of the root's profile (job ``prelude:<run id>``) and does not wait for it: it ends about a
minute into EXPLORE. Before the session maps its agents' directions, it waits for that job, at most
:data:`WAIT_SECONDS`, then takes the root's op shares from the blackboard.

On a CUDA-graph trace, TraceLens' op decomposition degrades and the job records no op shares. Its kernel table is still
written (``kernel_shares_path`` in the job's ``result.json``), and is read here for Arbor alone: Arbor's map generator
and its dossier pass take kernel names as well as ops. The blackboard's op shares, which MB's own scheduling reads, stay
as they are.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from hyperloom_multibranch.contracts.blackboard.records import MAX_OP_SHARES, OpShares
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.specialist.agent_onboarding import ROOT_PROFILE_SOURCE

if TYPE_CHECKING:
    from hyperloom_multibranch.contracts.blackboard.interfaces import RecordSearch
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

#: How long the session waits for the root profile's job before it maps without it.
WAIT_SECONDS = 600.0
POLL_SECONDS = 5.0
#: Where the toolbox's jobs write their results, below the session's toolbox directory, one directory per job named
#: by its quoted id (``toolbox.leased_toolbox``, ``toolbox.tools.base``).
TOOL_RESULTS_DIR, RESULT_FILE = "tool-results", "result.json"
KERNEL_SHARES_KEY = "kernel_shares_path"
NAME_COLUMN, SHARE_COLUMN = "name", "Percentage (%)"
KERNEL_SHARES_SOURCE = "PRELUDE root profile's kernel shares (TraceLens degraded: no op shares)"


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
    """Where :func:`root_shares` reads: the blackboard's op shares, the toolbox's jobs and the directory they write
    their results in, on the session's clock."""

    blackboard: RecordSearch
    toolbox: RealToolbox
    clock: Clock
    toolbox_dir: Path


async def root_shares(
    sources: ShareSources, root_commit: str, job: str, wait_seconds: float = WAIT_SECONDS
) -> SharesRead:
    """The shares of ``root_commit`` once its profile's ``job`` ended or ``wait_seconds`` passed: its op shares, else
    the job's kernel shares from the toolbox's directory."""

    blackboard, toolbox, clock, toolbox_dir = sources.blackboard, sources.toolbox, sources.clock, sources.toolbox_dir
    waited = await _wait(toolbox, clock, job, wait_seconds)
    detail: dict[str, object] = {"job_id": job, "job": waited}
    own = await blackboard.op_shares(root_commit)
    if own is not None and own.shares:
        return SharesRead(
            RootShares(own, ROOT_PROFILE_SOURCE), {**detail, "taken": "op shares", "ops": len(own.shares)}
        )
    kernels = kernel_shares(toolbox_dir / TOOL_RESULTS_DIR / quote(job, safe="") / RESULT_FILE) if job else {}
    if not kernels:
        return SharesRead(None, {**detail, "taken": "none: no op shares and no kernel table"})
    shares = OpShares(root_commit, job, "", kernels)
    return SharesRead(
        RootShares(shares, KERNEL_SHARES_SOURCE), {**detail, "taken": "kernel shares", "kernels": len(kernels)}
    )


async def _wait(toolbox: RealToolbox, clock: Clock, job: str, seconds: float) -> str:
    """Wait while ``job`` runs, at most ``seconds``; how it ended, as the decision says it."""

    if not job:
        return "no root profile was started"
    from hyperloom_multibranch.toolbox.tools.base import JobStatus  # the toolbox's tools load only when one runs

    until = clock.now() + seconds
    while True:
        try:
            status = toolbox.tool_status(job)
        except LookupError:
            return "the toolbox knows no such job"
        if status is not JobStatus.RUNNING:
            # A job's status turns final just before its result reaches the blackboard: read it a poll later.
            await clock.sleep(POLL_SECONDS)
            return status.value
        if clock.now() >= until:
            return f"still running after {seconds:.0f} s"
        await clock.sleep(POLL_SECONDS)


def kernel_shares(result: Path, limit: int = MAX_OP_SHARES) -> dict[str, float]:
    """The ``limit`` largest kernel shares, in percent, of the kernel table the job's ``result`` file names; empty when
    it names none or cannot be read. A kernel the table lists in several rows has their shares summed."""

    try:
        document = json.loads(result.read_text(encoding="utf-8"))
        table = Path(str((document.get("result") or {}).get(KERNEL_SHARES_KEY) or ""))
        with table.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, ValueError, AttributeError, csv.Error):
        return {}
    shares: dict[str, float] = {}
    for row in rows:
        name = str(row.get(NAME_COLUMN) or "").strip()
        try:
            share = float(row.get(SHARE_COLUMN) or "")
        except ValueError:
            continue
        if name and math.isfinite(share) and share > 0:
            shares[name] = shares.get(name, 0.0) + share
    return dict(sorted(shares.items(), key=lambda item: (-item[1], item[0]))[:limit])


__all__ = [
    "KERNEL_SHARES_SOURCE",
    "WAIT_SECONDS",
    "RootShares",
    "ShareSources",
    "SharesRead",
    "kernel_shares",
    "root_shares",
]
