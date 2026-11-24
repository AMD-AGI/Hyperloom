# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The open areas the judge reads: the integration head's largest ceilings less the ops worked on or spent."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping

from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.blackboard.exhausted import EXHAUSTED_PREFIX, ExhaustedWhy, latest_exhausted
from hyperloom_multibranch.contracts.blackboard.peers import Intent, IntentLayer, TestKind, Validity
from hyperloom_multibranch.contracts.blackboard.records import (
    BlackboardSnapshot,
    BranchLogRecord,
    BranchNode,
    CommitNode,
    CommittedCommit,
    MapDirection,
    OpShares,
    TestRecord,
)
from hyperloom_multibranch.contracts.blackboard.vocab import ClaimStatus, CommitOrigin, LineageKind
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.blackboard.rsi.open_areas import FocusHistory, OpenOp, open_areas, stopped_on_focus
from hyperloom_multibranch.blackboard.rsi.playbook import H0, replaced
from hyperloom_multibranch.blackboard.rsi.session import judge_open_areas
from hyperloom_multibranch.blackboard.rsi.tests.trees import branch, tree
from hyperloom_multibranch.blackboard.tests.factories import intent, request, result
from hyperloom_multibranch.blackboard.tests.sessions import METRIC

HOUR = 3600.0
ACT_MUL = "_act_mul_and_dynamic_fp8_group_quant_kernel"
ALLREDUCE = "aiter::cross_device_reduce_1stage"
SHARES = {
    ACT_MUL: 30.0,
    ALLREDUCE: 25.0,
    "gemm_a8w8_blockscale": 10.0,
    "paged_attention_ll4mi": 4.0,
    "rmsnorm": 2.0,
}


def _worked(snapshot: BlackboardSnapshot, *focused: BranchNode, head: str = "i1") -> BlackboardSnapshot:
    """The scripted session with the integration head at ``head`` and profiled, s2's attention test failed a second
    time, and the ``focused`` branches opened beside s1 and s2."""

    split = next(record for record in snapshot.records if record.request_id == "r2")
    again = dataclasses.replace(split, request=dataclasses.replace(split.request, request_id="r2b"))
    nodes = [
        dataclasses.replace(node, head_commit=head) if node.branch == INTEGRATION_BRANCH else node
        for node in snapshot.lineage.branches
    ]
    return dataclasses.replace(
        snapshot,
        records=(*snapshot.records, again),
        lineage=dataclasses.replace(snapshot.lineage, branches=(*nodes, *focused)),
        op_shares={head: OpShares(head, "job-1", "p1", SHARES)},
    )


def _focused(name: str, focus: str, *, start: str = "i1", running: bool = False) -> BranchNode:
    stopped = {} if running else {"stop_reason": "gave_up: nothing to test", "stopped_at": 1.0}
    return BranchNode(name, LineageKind.OPEN, start, start, focus=focus, **stopped)


def _committed(snapshot: BlackboardSnapshot, branch_name: str, gain_pct: float) -> BlackboardSnapshot:
    commit = CommittedCommit(f"k-{branch_name}", branch_name, "c0", f"r-{branch_name}", gain_pct, False, False)
    return dataclasses.replace(snapshot, committed=(*snapshot.committed, commit))


def test_the_ops_running_branches_are_focused_on_are_left_out_and_failed_tests_leave_out_none(
    snapshot: BlackboardSnapshot,
) -> None:
    worked = _worked(
        snapshot, _focused("b3", ACT_MUL, running=True), _focused("b4", "gemm_a8w8_blockscale", running=True)
    )

    found = open_areas(worked, seconds_left=4 * HOUR, first_commit_minutes=42.0)

    assert found.profiled == "i1" and not found.too_late and not found.spent
    assert found.ops == (
        OpenOp(ALLREDUCE, None, 25.0),
        OpenOp("paged_attention_ll4mi", "attention", 4.0),
        OpenOp("rmsnorm", "norm", 2.0),
    )
    assert dict(found.covered) == {ACT_MUL: ("b3",), "gemm_a8w8_blockscale": ("b4",)}
    assert found.render().splitlines() == [
        "Open areas for a new branch, from the profile of i1:",
        f"- {ALLREDUCE} (no op family): 25.0% of GPU time, at most +33.3%",
        "- paged_attention_ll4mi (attention): 4.0% of GPU time, at most +4.2%",
        "- rmsnorm (norm): 2.0% of GPU time, at most +2.0%",
        f"Left out: {ACT_MUL}, worked on by b3; gemm_a8w8_blockscale, worked on by b4.",
    ]


def test_a_running_branch_opened_without_a_focus_covers_the_ops_of_the_family_it_holds_a_claim_on(
    snapshot: BlackboardSnapshot,
) -> None:
    claim = dataclasses.replace(
        snapshot.claims[0],
        status=ClaimStatus.HELD,
        holder="s1",
        intent=Intent("s1", "c1", "gemm", "blockscale gemm tuning", IntentLayer.KERNEL),
    )
    worked = dataclasses.replace(_worked(snapshot), claims=(claim, *snapshot.claims[1:]))

    found = open_areas(worked, seconds_left=4 * HOUR, first_commit_minutes=42.0)

    assert "gemm_a8w8_blockscale" not in [item.op for item in found.ops]
    assert dict(found.covered) == {"gemm_a8w8_blockscale": ("s1",)}


def _mapped(snapshot: BlackboardSnapshot, *directions: MapDirection) -> BlackboardSnapshot:
    """``snapshot`` with the agent map of ``directions``, its agent branch b7 running beside b3, focused on the
    activation, and b8 stopped."""

    worked = _worked(
        snapshot, _focused("b3", ACT_MUL, running=True), _focused("b7", "", running=True), _focused("b8", "")
    )
    return dataclasses.replace(worked, agent_map=directions)


def test_an_op_a_direction_held_by_a_running_agent_branch_names_is_left_out_as_worked_on_by_that_branch(
    snapshot: BlackboardSnapshot,
) -> None:
    mapped = _mapped(
        snapshot,
        MapDirection("m-3", "Blockscale GEMM tiles", "open", "largest gemm_a8w8_blockscale 10%", holder="b7"),
        MapDirection("m-4", "Fold the activation into the GEMM", "open", holder="b7", ops=(f"{ACT_MUL}.kd",)),
    )

    found = open_areas(mapped, seconds_left=4 * HOUR, first_commit_minutes=42.0)

    assert [item.op for item in found.ops] == [ALLREDUCE, "paged_attention_ll4mi", "rmsnorm"]
    assert dict(found.covered) == {
        ACT_MUL: ("b3", "b7 (agent map, m-4)"),
        "gemm_a8w8_blockscale": ("b7 (agent map, m-3)",),
    }
    assert found.render().splitlines()[-1] == (
        f"Left out: {ACT_MUL}, worked on by b3, b7 (agent map, m-4); "
        "gemm_a8w8_blockscale, worked on by b7 (agent map, m-3)."
    )


def test_the_ops_of_a_direction_whose_agent_branch_stopped_or_that_the_map_set_aside_stay_open(
    snapshot: BlackboardSnapshot,
) -> None:
    mapped = _mapped(
        snapshot,
        MapDirection("m-5", "Decode attention", "open", "paged_attention_ll4mi is 4%", holder="b8"),
        MapDirection("m-6", "Fuse rmsnorm", "killed", "rmsnorm is 2% of the profile", holder="b7"),
    )

    found = open_areas(mapped, seconds_left=4 * HOUR, first_commit_minutes=42.0)

    assert "paged_attention_ll4mi" in [item.op for item in found.ops] and "rmsnorm" in [item.op for item in found.ops]
    assert dict(found.covered) == {ACT_MUL: ("b3",)}


def test_an_op_whose_focused_branches_on_the_head_ended_without_a_commit_is_spent_until_the_head_moves(
    snapshot: BlackboardSnapshot,
) -> None:
    branches = (
        _focused("b3", ALLREDUCE, start="c0"),
        _focused("b4", ALLREDUCE, start="c0"),
        _focused("b5", ALLREDUCE),
        _focused("b6", ALLREDUCE),
    )
    worked = _committed(_worked(snapshot, *branches), "b3", 3.2)

    found = open_areas(worked, seconds_left=4 * HOUR, first_commit_minutes=42.0)
    tolerant = open_areas(
        worked, seconds_left=4 * HOUR, first_commit_minutes=42.0, playbook=replaced(H0, failed_op_limit=3)
    )
    moved = open_areas(
        _committed(_worked(snapshot, *branches, head="i2"), "b3", 3.2), seconds_left=4 * HOUR, first_commit_minutes=42.0
    )

    assert [item.op for item in found.ops] == [ACT_MUL, "gemm_a8w8_blockscale", "paged_attention_ll4mi", "rmsnorm"]
    assert dict(found.spent) == {ALLREDUCE: FocusHistory(4, 3, 1, 3.2)}
    assert found.render().splitlines()[-2:] == [
        "Spent focus, left out until the integration head moves: the branches focused on it here ended without a "
        "commit. Open one again only on an angle they did not take:",
        f"- {ALLREDUCE}: opened 4, ended without a commit 3, commits 1, best +3.2%",
    ]
    assert not tolerant.spent and not moved.spent
    assert ALLREDUCE in [item.op for item in tolerant.ops] and ALLREDUCE in [item.op for item in moved.ops]


CRASHED = "gave_up: its specialist gave up after 4 crashes"
QUIET = "gave_up: its specialist gave up: its sessions in a row found nothing to test"
#: The stop of a specialist that crashed once, was restarted and then found nothing to test twice: the writer states
#: the idle give-up, whatever crashes came before it, so it reads as the quiet one.
CRASHED_THEN_IDLE = QUIET


def _ended(name: str, reason: str) -> BranchNode:
    """A branch focused on the allreduce op on the head that stopped for ``reason``."""

    return dataclasses.replace(_focused(name, ALLREDUCE), stop_reason=reason)


def _judged_on(snapshot: BlackboardSnapshot, branch_name: str) -> BlackboardSnapshot:
    """``snapshot`` with one more judged test, of ``branch_name``."""

    judged = next(record for record in snapshot.records if record.judged)
    moved = dataclasses.replace(
        judged, request=dataclasses.replace(judged.request, request_id=f"j-{branch_name}", branch_id=branch_name)
    )
    return dataclasses.replace(snapshot, records=(*snapshot.records, moved))


def _crashed_out(snapshot: BlackboardSnapshot, count: int) -> BlackboardSnapshot:
    """The session with ``count`` branches focused on the allreduce op that crashed out before a judged test."""

    return _worked(snapshot, *(_ended(f"b{8 + index}", CRASHED) for index in range(count)))


def test_branches_whose_specialist_only_crashed_before_a_judged_test_count_toward_twice_the_limit(
    snapshot: BlackboardSnapshot,
) -> None:
    crashed = _worked(snapshot, _ended("b8", CRASHED), _ended("b9", f"{CRASHED}, 2 of them with a lost heartbeat"))
    quiet = _worked(snapshot, _ended("b8", QUIET), _ended("b9", "judge: stalled"))
    idle_after_crash = _worked(snapshot, _ended("b8", CRASHED_THEN_IDLE), _ended("b9", CRASHED_THEN_IDLE))
    mixed = _worked(snapshot, _ended("b8", CRASHED), _ended("b9", QUIET))
    tested = _judged_on(_worked(snapshot, _ended("b8", CRASHED), _ended("b9", QUIET)), "b8")

    found = {
        name: open_areas(worked, seconds_left=4 * HOUR, first_commit_minutes=42.0)
        for name, worked in (
            ("crashed", crashed),
            ("quiet", quiet),
            ("idle_after_crash", idle_after_crash),
            ("mixed", mixed),
            ("tested", tested),
            ("three", _crashed_out(snapshot, 3)),
            ("four", _crashed_out(snapshot, 4)),
        )
    }
    one_bare = open_areas(
        mixed, seconds_left=4 * HOUR, first_commit_minutes=42.0, playbook=replaced(H0, failed_op_limit=1)
    )

    assert ALLREDUCE in [item.op for item in found["crashed"].ops] and not found["crashed"].spent
    assert dict(found["quiet"].spent) == {ALLREDUCE: FocusHistory(2, 2)}
    assert dict(found["idle_after_crash"].spent) == {ALLREDUCE: FocusHistory(2, 2)}
    assert ALLREDUCE in [item.op for item in found["mixed"].ops] and not found["mixed"].spent
    assert dict(one_bare.spent) == {ALLREDUCE: FocusHistory(2, 1, crashed_out=1)}
    assert dict(found["tested"].spent) == {ALLREDUCE: FocusHistory(2, 2)}
    assert ALLREDUCE in [item.op for item in found["three"].ops] and not found["three"].spent
    assert dict(found["four"].spent) == {ALLREDUCE: FocusHistory(4, 0, crashed_out=4)}
    assert found["four"].render().splitlines()[-1] == (
        f"- {ALLREDUCE}: opened 4, ended without a commit 0, crashed out before a judged test 4, no commit"
    )


def test_only_a_give_up_after_crashes_reads_as_a_branch_that_crashed_out() -> None:
    reasons = {
        CRASHED: True,
        f"{CRASHED}, 1 of them with a lost heartbeat": True,
        QUIET: False,
        "gave_up: its specialist could not be started: no account": False,
        "judge: its specialist gave up after 4 crashes": False,
        "session_end: explore ended": False,
    }

    assert {reason: _ended("b8", reason).crashed_out for reason in reasons} == reasons
    assert not _focused("b8", ALLREDUCE, running=True).crashed_out


def _exhausted(
    snapshot: BlackboardSnapshot,
    branch_name: str,
    why: ExhaustedWhy,
    head: str,
    detail: str = "every allreduce variant within 1% of the default",
) -> BlackboardSnapshot:
    """``snapshot`` with ``branch_name``'s log ending in an EXHAUSTED record of ``why`` on ``head``."""

    record = {
        "direction": "Small-M GEMM kernels",
        "head": head,
        "tried": ["r9"],
        "why": why.value,
        "detail": detail,
        "reopen_if": "the batch grows past 256",
    }
    entry = BranchLogEntry(branch_name, BranchLogType.RESULT, EXHAUSTED_PREFIX + json.dumps(record))
    logs = dict(snapshot.branch_logs)
    logs[branch_name] = (*logs.get(branch_name, ()), BranchLogRecord(f"e-{len(logs)}", 2.0, entry))
    return dataclasses.replace(snapshot, branch_logs=logs)


def _stopped(snapshot: BlackboardSnapshot, branch_name: str, reason: str) -> BlackboardSnapshot:
    nodes = tuple(
        dataclasses.replace(node, stop_reason=reason) if node.branch == branch_name else node
        for node in snapshot.lineage.branches
    )
    return dataclasses.replace(snapshot, lineage=dataclasses.replace(snapshot.lineage, branches=nodes))


def _gave_up(snapshot: BlackboardSnapshot, branch_name: str) -> BlackboardSnapshot:
    """``snapshot`` with ``branch_name`` stopped as the scheduler stops a give-up, quoting its latest record."""

    record = latest_exhausted(item.entry for item in snapshot.branch_logs[branch_name])
    assert record is not None
    return _stopped(snapshot, branch_name, f"gave_up: its specialist gave up: {record.summary()}")


def test_an_op_a_focused_branch_gave_up_on_quoting_a_done_record_on_the_head_is_spent_and_says_why(
    snapshot: BlackboardSnapshot,
) -> None:
    worked = _committed(_worked(snapshot, _focused("b8", ALLREDUCE, start="c0"), _focused("b9", ALLREDUCE)), "b8", 3.0)
    exhausted = _gave_up(_exhausted(worked, "b9", ExhaustedWhy.NO_CEILING_ABOVE_BAR, "i1"), "b9")

    found = open_areas(exhausted, seconds_left=4 * HOUR, first_commit_minutes=42.0)

    assert ALLREDUCE not in [item.op for item in found.ops]
    history = found.spent[ALLREDUCE]
    assert (history.opened, history.ended_without_commit, history.commits, history.best_gain_pct) == (2, 1, 1, 3.0)
    assert history.exhausted is not None and history.exhausted.why is ExhaustedWhy.NO_CEILING_ABOVE_BAR
    assert found.render().splitlines()[-2:] == [
        "Spent focus, exhausted here and left out until the integration head moves: a branch focused on it gave up "
        "on this head for the reason shown. Open one again only when its 'worth again if' holds:",
        f"- {ALLREDUCE}: no_ceiling_above_bar: every allreduce variant within 1% of the default; worth again if the "
        "batch grows past 256 (opened 2, ended without a commit 1, commits 1, best +3.0%)",
    ]


def test_a_done_record_of_one_direction_spends_nothing_when_the_branch_stopped_without_giving_up_on_it(
    snapshot: BlackboardSnapshot,
) -> None:
    exhausted = _exhausted(_worked(snapshot, _focused("b9", ALLREDUCE)), "b9", ExhaustedWhy.NO_CEILING_ABOVE_BAR, "i1")

    for reason in (
        "judge: stalled 6 tests since it opened",
        "session_end: time_exhausted",
        "gave_up: its specialist gave up after 3 crashes",
        "gave_up: its specialist gave up: its sessions in a row found nothing to test",
    ):
        stopped = _stopped(exhausted, "b9", reason)
        found = open_areas(stopped, seconds_left=4 * HOUR, first_commit_minutes=42.0)

        assert ALLREDUCE in [item.op for item in found.ops] and ALLREDUCE not in found.spent, reason
        (item,) = stopped_on_focus(stopped, ALLREDUCE)
        assert item.exhausted is not None and item.gave_up_on is None, reason


def test_records_on_other_heads_of_another_reason_or_of_branches_on_another_focus_leave_the_op_open(
    snapshot: BlackboardSnapshot,
) -> None:
    worked = _committed(
        _worked(
            snapshot,
            _focused("b3", ALLREDUCE, start="c0"),
            _focused("b4", ALLREDUCE),
            _focused("b5", ALLREDUCE),
            _focused("b7", ACT_MUL),
            _focused("b6", ""),
        ),
        "b5",
        2.0,
    )
    tool = "custom_all_reduce.py raised: not supported"
    for name, why, head, detail in (
        ("b3", ExhaustedWhy.ALL_TESTED_OR_CLAIMED, "c0", ""),
        ("b4", ExhaustedWhy.BLOCKED_BY_TOOL, "i1", tool),
        ("b5", ExhaustedWhy.NO_CEILING_ABOVE_BAR, "i1", ""),
        ("b5", ExhaustedWhy.NEEDS_CAPABILITY, "i1", tool),
        ("b6", ExhaustedWhy.NO_CEILING_ABOVE_BAR, "i1", ""),
        ("b7", ExhaustedWhy.NO_CEILING_ABOVE_BAR, "i1", ""),
    ):
        worked = _gave_up(_exhausted(worked, name, why, head, detail or "every variant within 1% of the default"), name)

    found = open_areas(worked, seconds_left=4 * HOUR, first_commit_minutes=42.0)

    assert ALLREDUCE in [item.op for item in found.ops] and ALLREDUCE not in found.spent
    records = {item.node.branch: item.exhausted for item in stopped_on_focus(worked, ALLREDUCE)}
    assert {name: record.why for name, record in records.items() if record is not None} == {
        "b3": ExhaustedWhy.ALL_TESTED_OR_CLAIMED,
        "b4": ExhaustedWhy.BLOCKED_BY_TOOL,
        "b5": ExhaustedWhy.NEEDS_CAPABILITY,
    }


def test_an_op_exhausted_on_an_older_head_opens_again_once_the_integration_head_moves(
    snapshot: BlackboardSnapshot,
) -> None:
    def exhausted_at(head: str) -> BlackboardSnapshot:
        worked = _worked(snapshot, _focused("b9", ALLREDUCE), head=head)
        return _gave_up(_exhausted(worked, "b9", ExhaustedWhy.ALL_TESTED_OR_CLAIMED, "i1"), "b9")

    on_head = open_areas(exhausted_at("i1"), seconds_left=4 * HOUR, first_commit_minutes=42.0)
    moved = open_areas(exhausted_at("i2"), seconds_left=4 * HOUR, first_commit_minutes=42.0)

    assert ALLREDUCE in on_head.spent and ALLREDUCE not in [item.op for item in on_head.ops]
    assert not moved.spent and ALLREDUCE in [item.op for item in moved.ops]


def _recipes(snapshot: BlackboardSnapshot, digests: Mapping[str, str]) -> BlackboardSnapshot:
    """``snapshot`` with each commit of ``digests`` holding that recipe digest, as a branch commit when it is new."""

    commits = tuple(
        dataclasses.replace(node, recipe_digest=digests.get(node.commit_id, node.recipe_digest))
        for node in snapshot.lineage.commits
    )
    known = {node.commit_id for node in commits}
    added = tuple(
        CommitNode(commit, "c0", "b9", CommitOrigin.COMMIT, recipe_digest=digest)
        for commit, digest in digests.items()
        if commit not in known
    )
    return dataclasses.replace(snapshot, lineage=dataclasses.replace(snapshot.lineage, commits=(*commits, *added)))


def test_an_op_given_up_on_a_commit_of_the_integration_heads_recipe_is_spent_as_on_the_head_itself(
    snapshot: BlackboardSnapshot,
) -> None:
    def gave_up_at(digest: str) -> BlackboardSnapshot:
        worked = _recipes(_worked(snapshot, _focused("b9", ALLREDUCE)), {"i1": "sha256:head", "k9": digest})
        return _gave_up(_exhausted(worked, "b9", ExhaustedWhy.ALL_TESTED_OR_CLAIMED, "k9"), "b9")

    same = open_areas(gave_up_at("sha256:head"), seconds_left=4 * HOUR, first_commit_minutes=42.0)
    other = open_areas(gave_up_at("sha256:other"), seconds_left=4 * HOUR, first_commit_minutes=42.0)

    record = same.spent[ALLREDUCE].exhausted
    assert record is not None and record.head == "k9" and ALLREDUCE not in [item.op for item in same.ops]
    assert not other.spent and ALLREDUCE in [item.op for item in other.ops]


def test_branches_opened_on_a_commit_of_the_integration_heads_recipe_spend_its_focus_as_on_the_head(
    snapshot: BlackboardSnapshot,
) -> None:
    def opened_on(digest: str) -> BlackboardSnapshot:
        branches = (_focused("b5", ALLREDUCE, start="k8"), _focused("b6", ALLREDUCE, start="k8"))
        return _recipes(_worked(snapshot, *branches), {"i1": "sha256:head", "k8": digest})

    same = open_areas(opened_on("sha256:head"), seconds_left=4 * HOUR, first_commit_minutes=42.0)
    other = open_areas(opened_on("sha256:other"), seconds_left=4 * HOUR, first_commit_minutes=42.0)

    assert dict(same.spent) == {ALLREDUCE: FocusHistory(2, 2)}
    assert not other.spent and ALLREDUCE in [item.op for item in other.ops]


def test_too_little_time_left_for_a_first_commit_says_to_open_no_branch(snapshot: BlackboardSnapshot) -> None:
    worked = _worked(snapshot)

    late = open_areas(worked, seconds_left=60.0, first_commit_minutes=42.0)
    quick = open_areas(
        worked, seconds_left=60.0, first_commit_minutes=42.0, playbook=replaced(H0, last_open_minutes=0.5)
    )

    assert late.too_late and late.render().endswith(
        "Time left: 1 min, short of the 42 min a new branch takes to its first commit: open no new branch."
    )
    assert not quick.too_late


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


def _profile_measured(
    snapshot: BlackboardSnapshot, commit: str, validity: Validity = Validity.SUCCEEDED
) -> BlackboardSnapshot:
    """``snapshot`` with a PROFILE test of the integration branch on ``commit`` that ended ``validity``."""

    profiled = TestRecord(
        request(f"p-{commit}", intent(INTEGRATION_BRANCH, commit, "profile", commit), kind=TestKind.PROFILE),
        result(f"p-{commit}", validity=validity),
    )
    return dataclasses.replace(snapshot, records=(*snapshot.records, profiled))


def test_a_profile_that_left_no_op_shares_reads_as_unknown_not_as_unprofiled_nor_as_out_of_ops(
    snapshot: BlackboardSnapshot,
) -> None:
    on_head = open_areas(_profile_measured(snapshot, "i1"), seconds_left=4 * HOUR, first_commit_minutes=None)
    on_root = open_areas(_profile_measured(snapshot, "c0"), seconds_left=4 * HOUR, first_commit_minutes=None)
    failed = open_areas(
        _profile_measured(snapshot, "i1", Validity.CANDIDATE_FAILED), seconds_left=4 * HOUR, first_commit_minutes=None
    )

    assert (on_head.unanalysed, on_head.profiled, on_head.ops) == ("i1", "", ())
    assert not on_head.exhausted and not on_root.exhausted
    assert on_head.render() == (
        "Open areas for a new branch: unknown. The integration head i1 is profiled, its trace not analysed: it left no "
        "op shares, so no op is known."
    )
    assert on_root.render().startswith(
        "Open areas for a new branch: unknown. A commit the integration head builds on c0"
    )
    assert failed.unanalysed == "" and failed.render().startswith(
        "Open areas for a new branch: neither the integration head nor a commit"
    )


def test_a_profile_of_another_commit_of_the_heads_recipe_that_left_no_op_shares_reads_as_unknown(
    snapshot: BlackboardSnapshot,
) -> None:
    def measured_on(digest: str) -> BlackboardSnapshot:
        return _recipes(_profile_measured(snapshot, "k7"), {"i1": "sha256:head", "k7": digest})

    same = open_areas(measured_on("sha256:head"), seconds_left=4 * HOUR, first_commit_minutes=None)
    other = open_areas(measured_on("sha256:other"), seconds_left=4 * HOUR, first_commit_minutes=None)

    assert same.unanalysed == "i1" and other.unanalysed == ""


def test_a_profile_with_every_op_worked_on_or_spent_is_out_of_ops_whatever_profile_test_measured_it(
    snapshot: BlackboardSnapshot,
) -> None:
    every = [_focused(f"b{n}", op, running=True) for n, op in enumerate(SHARES)]

    found = open_areas(
        _profile_measured(_worked(snapshot, *every), "i1"), seconds_left=4 * HOUR, first_commit_minutes=None
    )

    assert found.exhausted and found.unanalysed == ""
    assert found.render().splitlines()[:2] == [
        "Open areas for a new branch, from the profile of i1:",
        "- none: every large op is worked on or spent",
    ]


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


def test_shares_inherited_from_a_commit_of_another_recipe_name_both_commits(snapshot: BlackboardSnapshot) -> None:
    inherited = dataclasses.replace(snapshot, op_shares={"c0": OpShares("c0", "job-1", "p1", SHARES)})

    found = open_areas(inherited, seconds_left=4 * HOUR, first_commit_minutes=None)

    assert found.profiled == "c0"
    assert found.render().splitlines()[0] == (
        "Open areas for a new branch, from the profile of c0, which the integration head i1 builds on: the head's own "
        "recipe has no profile, so its shares may differ:"
    )
