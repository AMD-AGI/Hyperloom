# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the roster entry's status, the branch cap's count and the K limit."""

from __future__ import annotations

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.scheduling.roster import (
    BranchEntry,
    BranchKind,
    BranchOrigin,
    BranchStatus,
    BranchStop,
    RosterCaps,
    StopCause,
)


def _entry(*, start: str = "c1", stop: BranchStop | None = None) -> BranchEntry:
    return BranchEntry("b1", BranchKind.SPECIALIST, BranchOrigin.FIRST_BATCH, start, k=1, opened_at=0.0, stop=stop)


def test_status_follows_start_commit_and_stop() -> None:
    assert _entry(start="").status is BranchStatus.ADMITTED
    assert _entry().status is BranchStatus.RUNNING
    assert _entry(stop=BranchStop(StopCause.JUDGE, "no gain", 1.0)).status is BranchStatus.STOPPED
    assert _entry(start="", stop=BranchStop(StopCause.SESSION_END, "time", 1.0)).status is BranchStatus.STOPPED


def test_cap_counts_admitted_and_running_specialists_only() -> None:
    assert _entry().counts_toward_cap
    assert _entry(start="").counts_toward_cap
    assert not _entry(stop=BranchStop(StopCause.JUDGE, "no gain", 1.0)).counts_toward_cap
    integration = BranchEntry(INTEGRATION_BRANCH, BranchKind.INTEGRATION, BranchOrigin.ROOT, "root", 1, 0.0)
    assert not integration.counts_toward_cap


def test_k_is_cut_to_one_through_the_cap() -> None:
    caps = RosterCaps(max_branches=4, max_k_per_branch=2, max_parallel_tests=8)
    assert [caps.clamp_k(k) for k in (-3, 0, 1, 2, 9)] == [1, 1, 1, 2, 2]


@pytest.mark.parametrize("field", ["max_branches", "max_k_per_branch", "max_parallel_tests"])
def test_caps_are_at_least_one(field: str) -> None:
    values = {"max_branches": 4, "max_k_per_branch": 2, "max_parallel_tests": 8, field: 0}
    with pytest.raises(ValueError, match=field):
        RosterCaps(**values)
