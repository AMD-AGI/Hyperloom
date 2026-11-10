# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a session that ended before FINAL can be resumed (:meth:`~hyperloom_multibranch.scheduling.lifecycle.
Session.resume`), and the deadline ``run --resume`` resumes it up to.

``run --resume`` admits it with :func:`resumable_run` and gives back the time it was down
(:func:`deadline_after_downtime`); a hosted restart admits it with
:func:`~hyperloom_multibranch.scheduling.hosted_run.hosted_resume_deadline` and gives nothing back. Both refuse with
:class:`ResumeRefused`, which leaves the session to be finished instead.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom_common.jsonio import read_jsonl
from hyperloom_multibranch.scheduling.decisions import downtime_until
from hyperloom_multibranch.scheduling.launch_isolation import HOSTED
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.lifecycle import PersistedRun, persisted_run, prelude_landed
from hyperloom_multibranch.scheduling.manifest import read_manifest
from hyperloom_multibranch.scheduling.prelude import baseline_decision


class ResumeRefused(ValueError):
    """Why a session that ended before FINAL cannot be resumed; it can still be finished."""


def require_resumable(persisted: PersistedRun) -> None:
    """Raise ResumeRefused when a resume has nothing to continue ``persisted`` from: PRELUDE recorded no baseline or
    not the decision that names it, or the session was to stop once its map was made. Blocks."""

    root = persisted.layout.root
    if not prelude_landed(root) or baseline_decision(read_jsonl(persisted.layout.decisions)) is None:
        raise ResumeRefused(f"the session in {root} recorded no baseline to resume from; start a new session")
    if persisted.settings.stop_after_map:
        raise ResumeRefused(f"the session in {root} was to stop once its map was made, so it has no EXPLORE to resume")


def resumable_run(root: Path) -> PersistedRun:
    """:func:`~hyperloom_multibranch.scheduling.lifecycle.persisted_run` for ``run --resume``; raises what
    :func:`require_resumable` raises, and ResumeRefused too when the session was measured in a grant's sandboxes,
    which a resume on this machine does not lease again: a hosted restart resumes it instead. Writes nothing."""

    persisted = persisted_run(root)
    require_resumable(persisted)
    isolation = read_manifest(persisted.layout.manifest).agent_isolation
    if isolation is not None and isolation.tests == HOSTED:
        raise ResumeRefused(
            f"the session in {root} was measured in a grant's sandboxes and a resume measures on this machine's cards;"
            " start a new session"
        )
    return persisted


def deadline_after_downtime(layout: SessionLayout, now: float) -> float:
    """The deadline of the session in ``layout`` moved by the time it was down until ``now``: what ``run --resume``
    gives back. GPU hours are not given back: the session reads what it spent from its blackboard."""

    return read_manifest(layout.manifest).deadline + downtime_until(layout.decisions, now)


__all__ = ["ResumeRefused", "deadline_after_downtime", "require_resumable", "resumable_run"]
