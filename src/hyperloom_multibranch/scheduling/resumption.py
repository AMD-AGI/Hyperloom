# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session that ended before FINAL as read back from its directory (:func:`persisted_run`), whether it can be
resumed (:meth:`~hyperloom_multibranch.scheduling.lifecycle.Session.resume`), and the deadline ``run --resume`` resumes
it up to.

``run --resume`` admits it with :func:`resumable_run` and gives back the time it was down
(:func:`deadline_after_downtime`); a hosted restart admits it with
:func:`~hyperloom_multibranch.scheduling.hosted_run.hosted_resume_deadline` and gives nothing back. Both refuse with
:class:`ResumeRefused`, which leaves the session to be finished instead.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path

from hyperloom_common.jsonio import read_jsonl
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.interfaces import NotRecorded
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord
from hyperloom_multibranch.scheduling.budget import MB_SECONDS_PER_HOUR, explore_gpu_seconds
from hyperloom_multibranch.scheduling.close import ended_before_final
from hyperloom_multibranch.scheduling.decisions import downtime_until
from hyperloom_multibranch.scheduling.explore import spent_gpu_seconds
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.manifest import code_version, read_manifest
from hyperloom_multibranch.scheduling.prelude import baseline_decision
from hyperloom_multibranch.scheduling.settings import RunSettings
from hyperloom_multibranch.scheduling.stops import SessionStopReason

#: The writer name a resume check reads a session's blackboard as; it writes nothing.
RESUME_READER = "resume-check"


@dataclass(frozen=True)
class PersistedRun:
    """A session directory that ended without FINAL, as
    :meth:`~hyperloom_multibranch.scheduling.lifecycle.Session.finish` takes it again; ``ended`` is the stop reason
    that session reported, or :data:`SessionStopReason.KILLED` when it was killed before it could report one, and the
    second report carries it."""

    layout: SessionLayout
    settings: RunSettings
    ended: SessionStopReason


async def recorded_baseline(blackboard: LocalBlackboard) -> BaselineRecord | None:
    """The baseline the rule gate wrote on ``blackboard``, ``None`` while it has written none."""

    try:
        return await blackboard.baseline()
    except NotRecorded:
        return None


def persisted_run(root: Path) -> PersistedRun:
    """The session in ``root``, read back from what the session manifest and CLOSE left there, or from what
    a session that was killed left without a CLOSE; raises ValueError naming what keeps FINAL from running over it.
    Only the build it ran on may read back what it measured. Creates nothing. Blocks."""

    persisted = persisted_session(root)
    ran_on, here = read_manifest(persisted.layout.manifest).code_version, code_version()
    if ran_on != here:
        raise ValueError(
            f"the session in {root} ran on {ran_on} and this build is {here}; FINAL and CLOSE read back what that "
            "build persisted, so run them from it"
        )
    return persisted


def persisted_session(root: Path) -> PersistedRun:
    """:func:`persisted_run` on any build: for a session that recorded no baseline, which a restart only closes, so
    nothing it measured is read back."""

    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")
    layout = SessionLayout(root)
    try:
        manifest = read_manifest(layout.manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{root} is no multi-branch session that ran: {exc}") from exc
    if not layout.root_recipe.is_file():
        raise ValueError(f"the session in {root} committed no root, so FINAL has nothing to validate")
    return PersistedRun(layout, manifest.settings, ended_before_final(layout))


class ResumeRefused(ValueError):
    """Why a session that ended before FINAL cannot be resumed; it can still be finished."""


def require_resumable(persisted: PersistedRun) -> None:
    """Raise ResumeRefused when a resume has nothing to continue ``persisted`` from: PRELUDE recorded no baseline or
    not the decision that names it, the session was to stop once its map was made, or it already spent the GPU hours
    EXPLORE has under ``--max-gpu-hours``, which would send the resume straight to FINAL. Blocks."""

    root = persisted.layout.root
    if baseline_decision(read_jsonl(persisted.layout.decisions)) is None:
        raise ResumeRefused(f"the session in {root} recorded no baseline to resume from; start a new session")
    if persisted.settings.stop_after_map:
        raise ResumeRefused(f"the session in {root} was to stop once its map was made, so it has no EXPLORE to resume")
    explore_gpu = explore_gpu_seconds(persisted.settings.max_gpu_hours)
    if explore_gpu != math.inf:
        reader = LocalBlackboard(persisted.layout.blackboard, Writer(Role.OBSERVER, RESUME_READER))
        spent = spent_gpu_seconds(asyncio.run(reader.progress()))
        if spent >= explore_gpu:
            raise ResumeRefused(
                f"the session in {root} spent {spent / MB_SECONDS_PER_HOUR:.2f} GPU hours, all EXPLORE has under --max-gpu-hours"
                f" {persisted.settings.max_gpu_hours:g}, so it has no EXPLORE to resume"
            )


def resumable_run(root: Path) -> PersistedRun:
    """:func:`persisted_run` for ``run --resume``; raises what
    :func:`require_resumable` raises, and ResumeRefused too when the session was measured in a grant's sandboxes,
    which a resume on this machine does not lease again: a hosted restart resumes it instead. Writes nothing."""

    persisted = persisted_run(root)
    require_resumable(persisted)
    if persisted.layout.runner_source.exists():
        raise ResumeRefused(
            f"the session in {root} was measured in a grant's sandboxes and a resume measures on this machine's cards;"
            " start a new session"
        )
    return persisted


def deadline_after_downtime(layout: SessionLayout, now: float) -> float:
    """The deadline of the session in ``layout`` moved by the time it was down until ``now``: what ``run --resume``
    gives back. GPU hours are not given back: the session reads what it spent from its blackboard."""

    return read_manifest(layout.manifest).deadline + downtime_until(layout.decisions, now)


__all__ = [
    "RESUME_READER",
    "PersistedRun",
    "ResumeRefused",
    "deadline_after_downtime",
    "persisted_run",
    "persisted_session",
    "recorded_baseline",
    "require_resumable",
    "resumable_run",
]
