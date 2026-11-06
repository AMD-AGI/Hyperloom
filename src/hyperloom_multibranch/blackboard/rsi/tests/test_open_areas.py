# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The open areas the judge reads: the integration head's largest ceilings less what is worked on or failed."""

from __future__ import annotations

import dataclasses

from hyperloom_multibranch.contracts.blackboard.peers import Intent, IntentLayer
from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot, OpShares
from hyperloom_multibranch.contracts.blackboard.vocab import ClaimStatus
from hyperloom_multibranch.blackboard.rsi.open_areas import OpenOp, open_areas
from hyperloom_multibranch.blackboard.rsi.playbook import H0, replaced
from hyperloom_multibranch.blackboard.rsi.session import judge_open_areas
from hyperloom_multibranch.blackboard.rsi.tests.trees import branch, tree
from hyperloom_multibranch.blackboard.tests.sessions import METRIC

HOUR = 3600.0
SHARES = {
    "paged_attention_ll4mi": 30.0,
    "fused_moe_kernel": 25.0,
    "gemm_a8w8_blockscale": 10.0,
    "custom_op_x": 4.0,
    "rmsnorm": 2.0,
}


def _worked(snapshot: BlackboardSnapshot) -> BlackboardSnapshot:
    """The scripted session with the integration head profiled, s1 holding a claim on MoE, and attention failed a
    second time."""

    claim = dataclasses.replace(
        snapshot.claims[0],
        status=ClaimStatus.HELD,
        holder="s1",
        intent=Intent("s1", "c1", "moe", "fused moe", IntentLayer.KERNEL),
    )
    split = next(record for record in snapshot.records if record.request_id == "r2")
    again = dataclasses.replace(split, request=dataclasses.replace(split.request, request_id="r2b"))
    return dataclasses.replace(
        snapshot,
        claims=(claim, *snapshot.claims[1:]),
        records=(*snapshot.records, again),
        op_shares={"i1": OpShares("i1", "job-1", "p1", SHARES)},
    )


def test_the_open_areas_are_the_heads_largest_ceilings_less_the_families_worked_on_or_failed(
    snapshot: BlackboardSnapshot,
) -> None:
    found = open_areas(_worked(snapshot), seconds_left=4 * HOUR, first_commit_minutes=42.0)

    assert found.profiled == "i1" and not found.too_late
    assert found.ops == (
        OpenOp("gemm_a8w8_blockscale", "gemm", 10.0),
        OpenOp("custom_op_x", None, 4.0),
        OpenOp("rmsnorm", "norm", 2.0),
    )
    assert (dict(found.covered), dict(found.failed)) == ({"moe": ("s1",)}, {"attention": 2})
    assert found.render().splitlines() == [
        "Open areas for a new branch, from the profile of i1:",
        "- gemm_a8w8_blockscale (gemm): 10.0% of GPU time, at most +11.1%",
        "- custom_op_x (no op family): 4.0% of GPU time, at most +4.2%",
        "- rmsnorm (norm): 2.0% of GPU time, at most +2.0%",
        "Left out: moe, worked on by s1; attention, failed 2 times this session.",
    ]


def test_too_little_time_left_for_a_first_commit_says_to_open_no_branch(snapshot: BlackboardSnapshot) -> None:
    worked = _worked(snapshot)

    late = open_areas(worked, seconds_left=60.0, first_commit_minutes=42.0)
    quick = open_areas(
        worked, seconds_left=60.0, first_commit_minutes=42.0, playbook=replaced(H0, last_open_minutes=0.5)
    )
    strict = open_areas(
        worked, seconds_left=4 * HOUR, first_commit_minutes=42.0, playbook=replaced(H0, failed_op_limit=3)
    )

    assert late.too_late and late.render().endswith(
        "Time left: 1 min, short of the 42 min a new branch takes to its first commit: open no new branch."
    )
    assert not quick.too_late and "attention" not in strict.failed


def test_without_a_profile_the_open_areas_say_so_and_the_judge_reads_them_once_the_session_has_one(
    snapshot: BlackboardSnapshot,
) -> None:
    bare = open_areas(snapshot, seconds_left=4 * HOUR, first_commit_minutes=None)
    past = (tree("past", branch("s1", [(1800.0, 3.0)])),)

    assert bare.render().startswith("Open areas for a new branch: neither the integration head nor a commit")
    assert (
        judge_open_areas(_worked(snapshot), past, METRIC, seconds_left=4 * HOUR)
        .render()
        .startswith("Open areas for a new branch, from the profile of i1:")
    )


def test_without_archived_sessions_only_the_playbook_s_last_open_minutes_says_it_is_too_late(
    snapshot: BlackboardSnapshot,
) -> None:
    unknown = judge_open_areas(_worked(snapshot), (), METRIC, seconds_left=0.0)
    capped = judge_open_areas(
        _worked(snapshot), (), METRIC, seconds_left=0.0, playbook=replaced(H0, last_open_minutes=30)
    )

    assert unknown.first_commit_minutes is None and not unknown.too_late
    assert unknown.render().startswith("Open areas for a new branch, from the profile of i1:")
    assert capped.too_late and capped.render().endswith("open no new branch.")
