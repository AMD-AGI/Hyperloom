# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The native builds each branch's tests wait for, as the judge sees them: a branch whose tests wait only on a compiler
is building, not stalled."""

from __future__ import annotations

from collections.abc import Iterable

from hyperloom_multibranch.contracts.blackboard.records import TestRecord
from hyperloom_multibranch.contracts.toolbox import BuildState, BuildStatus, Toolbox


async def waiting_builds(toolbox: Toolbox, records: Iterable[TestRecord]) -> dict[str, tuple[BuildStatus, ...]]:
    """Each branch's native builds that a test of it in flight waits for and that are not ready, once per build, as
    the toolbox reports each test's progress; a branch with none is left out. Only the tests of a branch the toolbox
    counts a test waiting for its builds of are asked about."""

    pending = (await toolbox.queue_state()).pending_build_by_branch
    waiting: dict[str, dict[str, BuildStatus]] = {}
    for record in records:
        branch = record.request.branch
        if not record.in_flight or not pending.get(branch):
            continue
        progress = await toolbox.progress(record.request_id)
        builds = () if progress is None else progress.builds
        waiting.setdefault(branch, {}).update(
            {build.key: build for build in builds if build.state is not BuildState.READY}
        )
    return {branch: tuple(builds.values()) for branch, builds in waiting.items() if builds}


__all__ = ["waiting_builds"]
