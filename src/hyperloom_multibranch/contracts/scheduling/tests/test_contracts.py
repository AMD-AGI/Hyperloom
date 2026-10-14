# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the quota policy, the fork admission answer and the queue status the scheduling side exchanges."""

from __future__ import annotations

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.scheduling import (
    PRIORITY_ORDER,
    AdmissionStatus,
    BranchQuota,
    ForkAdmission,
    PriorityClass,
    QueueStatus,
    QuotaPolicy,
)


def test_priority_order_follows_the_design() -> None:
    assert PRIORITY_ORDER == (
        PriorityClass.SESSION_MEASUREMENT,
        PriorityClass.CONFIRMATION_RETEST,
        PriorityClass.REPRO_CHECK,
        PriorityClass.INTEGRATION_MERGE,
        PriorityClass.PROFILE_REFRESH,
        PriorityClass.EXPLORATION,
    )


def test_confirmations_and_repro_checks_do_not_take_branch_k() -> None:
    exempt = {priority for priority in PriorityClass if not QuotaPolicy.uses_branch_k(priority)}
    assert exempt == {
        PriorityClass.SESSION_MEASUREMENT,
        PriorityClass.CONFIRMATION_RETEST,
        PriorityClass.REPRO_CHECK,
    }


def test_policy_answers_k_per_branch_and_none_for_a_branch_without_quota() -> None:
    policy = QuotaPolicy(8, (BranchQuota("b1", 2), BranchQuota(INTEGRATION_BRANCH, 1)))
    assert policy.k_of("b1") == 2
    assert policy.k_of(INTEGRATION_BRANCH) == 1
    assert policy.k_of("stopped") is None


@pytest.mark.parametrize("k", [0, -1])
def test_a_branch_quota_is_at_least_one(k: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BranchQuota("b1", k)


def test_policy_refuses_a_zero_global_cap_and_a_branch_listed_twice() -> None:
    with pytest.raises(ValueError, match="max_parallel_tests"):
        QuotaPolicy(0, ())
    with pytest.raises(ValueError, match="more than one quota"):
        QuotaPolicy(4, (BranchQuota("b1", 1), BranchQuota("b1", 2)))


def test_only_an_accepted_fork_names_its_branch() -> None:
    assert ForkAdmission(AdmissionStatus.ACCEPTED, "slot free", "b5").branch == "b5"
    assert ForkAdmission(AdmissionStatus.REJECTED, "at the branch cap").branch == ""
    with pytest.raises(ValueError):
        ForkAdmission(AdmissionStatus.ACCEPTED, "slot free")
    with pytest.raises(ValueError):
        ForkAdmission(AdmissionStatus.PENDING, "at the branch cap", "b5")


def test_queue_is_idle_only_with_nothing_queued_or_running() -> None:
    assert QueueStatus().idle
    assert QueueStatus(queued={"exploration": 0}, running={"b1": 0}).idle
    assert not QueueStatus(running={"b1": 1}).idle
    assert not QueueStatus(queued={"exploration": 2}).idle
