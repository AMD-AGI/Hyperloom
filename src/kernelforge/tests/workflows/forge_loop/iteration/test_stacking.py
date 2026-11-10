# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stacking two rejected gains once single patches stall: which pair is measured, when, and what the queue gives up for it."""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import replace
from pathlib import Path

from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.campaign.archive import CandidateRecord
from kernelforge.workflows.forge_loop.campaign.run_state import LoopStateStore, RunState
from kernelforge.workflows.forge_loop.iteration.merge_candidates import (
    MERGE_ATTEMPT_STALL_THRESHOLD,
    MERGE_PRECEDENCE_STREAK_LIMIT,
)
from kernelforge.workflows.forge_loop.iteration.runner import IterationResult
from kernelforge.workflows.forge_loop.iteration import campaign_runtime
from kernelforge.workflows.forge_loop.iteration import lane_runtime
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _AMPLE_BUDGET_SEC,
    _counting_plan,
    _diff_of,
    _lane_agent_factory,
    _make_loop,
    _no_change_agent,
    _reduction_loop,
    _unused_supervisor,
)


def _wide_kernel(loop, workspace, width):
    """Commit a kernel of ``width`` numbered lines, archive against it, and diff one-line edits of it."""
    loop.archive = campaign_runtime.CandidateArchive(str(workspace), loop.ic.kernel_file)
    kernel = workspace / "kernel.py"
    kernel.write_text("\n".join(f"line_{n} = {n}" for n in range(width)) + "\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "wide kernel"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    canonical = kernel.read_text()

    def diff(old: str, new: str) -> str:
        kernel.write_text(canonical.replace(old, new))
        change = subprocess.run(["git", "diff"], cwd=workspace, capture_output=True, text=True).stdout
        kernel.write_text(canonical)
        return change

    return canonical, diff


def _stackable_workspace(loop, workspace, *, candidates=2):
    """Archive rejected gains whose diffs touch well-separated parts of the kernel.

    Each candidate wins one case and loses the other, alternating, so any two of
    opposite parity are mutually complementary and the field of selectable pairs
    grows as the square of the count -- which is what a streak draws on. The
    lines are further apart than a diff hunk carries context, so any two of the
    patches apply over each other.
    """
    edits = (
        (0, "line_0 = 0", "line_0 = 100", "prefill"),
        (11, "line_11 = 11", "line_11 = 111", "decode"),
        (22, "line_22 = 22", "line_22 = 122", "prefill"),
        (33, "line_33 = 33", "line_33 = 133", "decode"),
    )[:candidates]
    canonical, _diff = _wide_kernel(loop, workspace, max(line for line, *_rest in edits) + 1)

    for iteration, (_line, old, new, case) in enumerate(edits, start=1):
        runs = [
            {
                "prefill": (0.9 if case == "prefill" else 1.0) * jitter,
                "decode": (0.9 if case == "decode" else 1.0) * jitter,
            }
            for jitter in (1.0, 1.0005, 0.9995)
        ]
        loop.archive.record(
            CandidateRecord(
                iteration=iteration,
                decision="REVERT_PERF",
                validation_passed=True,
                mean_case_speedup=1.002 + iteration / 1000.0,
                bench_detail={
                    "case_times": {
                        "prefill": 0.9 if case == "prefill" else 1.0,
                        "decode": 0.9 if case == "decode" else 1.0,
                    },
                    "measurements": [{"success": True, "case_times": run, "unscored_cases": []} for run in runs],
                },
                change_diff=_diff(old, new),
                plan=f"tune {case}",
            )
        )
    return canonical


def test_a_stalled_run_stacks_two_rejected_gains(tmp_path, monkeypatch):
    """The cheapest thing to try once single patches stop clearing the gate.

    Neither candidate passed alone, they win on different cases, and stacking
    them spends a measurement but no Implementer session.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    canonical = _stackable_workspace(loop, workspace)

    pair = loop._select_merge_attempt()

    assert pair is not None
    assert {item.iteration for item in pair} == {1, 2}

    staged, obstacle = loop._stage_merge_attempt(pair)

    assert staged
    assert obstacle == ""
    assert (workspace / "kernel.py").read_text() != canonical
    assert "line_0 = 100" in (workspace / "kernel.py").read_text()
    assert "line_11 = 111" in (workspace / "kernel.py").read_text()


def test_a_stack_does_not_take_an_iteration_that_is_holding_a_plan(tmp_path, monkeypatch):
    """The round's plan has only one consumer, and a stack is not it.

    A fan-out round can come back with an empty queue while still holding the
    plan it bought -- one lane plan, a dispatch the budget refused, a lane
    failure -- and the queue being empty is exactly the condition that lets a
    stack take the iteration. A stacked iteration records a result, which is
    what stops the next process from recovering the round, so planning
    (dispatch, every specialist, synthesis) would be paid for and thrown away.
    The stall that selected the pair is untouched by spending the plan, so the
    attempt is deferred, not lost.
    """
    cases = {"prefill": 1.0, "decode": 1.0}
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True, baseline_case_times=cases)
    loop.ic = replace(loop.ic, lanes=2)
    _stackable_workspace(loop, workspace)
    subprocess.run(
        ["git", "checkout", "-b", "test-loop"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    state = RunState(
        campaign_id="campaign",
        baseline_case_times=dict(cases),
        best_case_times=dict(cases),
        head_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    state.stall.unresolved_stall_iters = 2
    LoopStateStore(str(workspace)).save(state)
    rounds: list[int] = []
    monkeypatch.setattr(loop, "_run_orchestration", _counting_plan(loop, workspace, rounds))

    async def _no_candidates(**_kwargs):
        return []

    monkeypatch.setattr(lane_runtime, "run_lanes", _no_candidates)

    asyncio.run(
        loop.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent,
                agent_factory=_lane_agent_factory({}),
                orchestration_service=object(),
                supervisor_fn=_unused_supervisor,
            )
        )
    )

    events = LoopStateStore(str(workspace)).read_events()

    assert [e["type"] for e in events if e["type"].startswith("merge_attempt")] == []
    # One round planned, and the iteration spent it rather than buying another.
    assert rounds == [2]
    assert [e.get("decision") for e in events if e["type"] == "iteration_result"] == ["NO_CHANGES"]
    # Two things at once: the pair and the stall the branch needs were both in
    # place -- so the assertions above are about precedence, not about a stack
    # that could never have formed -- and the attempt is deferred, not retired.
    assert loop._select_merge_attempt() is not None


def _three_candidate_workspace(loop, workspace):
    """Archive three rejected gains: the best pair clashes, the runner-up does not.

    Iterations 1 and 2 rewrite the same line to different values, so they cover
    the most cases between them and cannot both be applied. Iteration 3 wins one
    of the same cases as 2 from the other end of the file, so (1, 3) is the pair
    left once (1, 2) is out and it stages cleanly.
    """
    canonical, _diff = _wide_kernel(loop, workspace, 12)

    plan = (
        (1, "line_0 = 0", "line_0 = 100", {"prefill"}),
        (2, "line_0 = 0", "line_0 = 200", {"decode", "mixed"}),
        (3, "line_11 = 11", "line_11 = 111", {"decode"}),
    )
    for iteration, old, new, wins in plan:
        times = {case: 0.9 if case in wins else 1.0 for case in ("prefill", "decode", "mixed")}
        loop.archive.record(
            CandidateRecord(
                iteration=iteration,
                decision="REVERT_PERF",
                validation_passed=True,
                mean_case_speedup=1.002 + iteration / 1000.0,
                bench_detail={
                    "case_times": times,
                    "measurements": [
                        {
                            "success": True,
                            "case_times": {case: value * jitter for case, value in times.items()},
                            "unscored_cases": [],
                        }
                        for jitter in (1.0, 1.0005, 0.9995)
                    ],
                },
                change_diff=_diff(old, new),
                plan=f"tune {iteration}",
            )
        )
    return canonical


def test_a_pair_that_would_not_stage_is_not_selected_again(tmp_path, monkeypatch):
    """A textual clash between two archived diffs is the same clash next stall.

    The selector returns the pair covering the most cases, so a decline the run
    does not remember wins every later selection, fails identically, and blocks
    the runner-up that would have staged for the rest of the campaign. Nothing
    in the archive records it: a pair that never reached a measurement is never
    archived.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    _three_candidate_workspace(loop, workspace)

    clashing = loop._select_merge_attempt()

    assert clashing is not None
    assert {item.iteration for item in clashing} == {1, 2}

    staged, obstacle = loop._stage_merge_attempt(clashing)

    assert staged == ""
    assert obstacle

    loop._decline_merge_attempt(4, clashing, obstacle)
    runner_up = loop._select_merge_attempt()

    assert runner_up is not None
    assert {item.iteration for item in runner_up} == {1, 3}
    assert loop._stage_merge_attempt(runner_up)[0]


def test_a_tree_that_carried_work_does_not_retire_the_pair(tmp_path, monkeypatch):
    """The one obstacle the pair is not responsible for and next iteration clears."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    _three_candidate_workspace(loop, workspace)
    pair = loop._select_merge_attempt()
    assert pair is not None

    loop._decline_merge_attempt(4, pair, loop.TREE_ALREADY_DIRTY_OBSTACLE)

    again = loop._select_merge_attempt()

    assert again is not None
    assert {item.iteration for item in again} == {item.iteration for item in pair}


def test_a_streak_refusal_does_not_retire_the_pair(tmp_path, monkeypatch):
    """The other obstacle that is a fact about the iteration, not about the pair.

    A refusal is ruled on before ``_stage_merge_attempt`` runs, so the pair's
    diffs are never read, let alone applied to each other -- there is no verdict
    on them to remember. Remembering one anyway turns the streak limit from a
    deferral into a drop, and costs the campaign a measurement that nothing was
    ever wrong with.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    _three_candidate_workspace(loop, workspace)
    pair = loop._select_merge_attempt()
    assert pair is not None
    loop._merge_precedence_streak = MERGE_PRECEDENCE_STREAK_LIMIT
    refusal = loop._merge_attempt_refusal()
    assert refusal

    loop._decline_merge_attempt(4, pair, refusal, about_the_iteration=True)

    # Reported, because a selected pair that reached no measurement is not the
    # same event as no pair at all -- and still selectable, because the report
    # was about the iteration.
    assert [
        (item["obstacle"], item["first_iteration"], item["second_iteration"])
        for item in LoopStateStore(str(workspace)).read_events()
        if item.get("type") == "merge_attempt_declined"
    ] == [(refusal, pair[0].iteration, pair[1].iteration)]
    again = loop._select_merge_attempt()

    assert again is not None
    assert {item.iteration for item in again} == {item.iteration for item in pair}


def test_an_archive_that_lost_a_diff_says_so_and_does_not_claim_a_conflict(tmp_path, monkeypatch):
    """A missing entry and a clashing patch ask for opposite responses.

    A clash is a fact about two candidates and says the archive is working. An
    entry the archive cannot produce says it lost a candidate it claims to hold,
    which the retrieval map and every resumed run are also reading.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    _stackable_workspace(loop, workspace)
    pair = loop._select_merge_attempt()
    assert pair is not None
    Path(loop.archive._iter_dir(2) / "change.diff").unlink()

    staged, obstacle = loop._stage_merge_attempt(pair)

    assert staged == ""
    assert obstacle == "iteration 2's archived diff is missing or unreadable"


def test_stacking_can_be_turned_off(tmp_path, monkeypatch):
    """It changes what the ordinary single-session path does at every --lanes.

    An operator comparing against a run that predates it needs the older
    behaviour back, and no amount of stall state should reach it.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, merge_stacking=False)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 9
    _stackable_workspace(loop, workspace)

    assert loop._select_merge_attempt() is None


def test_stacking_never_discards_work_it_did_not_stage(tmp_path, monkeypatch):
    """Returning to canonical takes every tracked edit, not just the staged ones.

    Two patches that clash send the tree back to HEAD, which would delete an
    edit that was already there. The loop should reach this on a clean tree, but
    that is an invariant of earlier paths -- a stacking attempt must not enforce
    it by destroying the evidence that it was broken.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    _stackable_workspace(loop, workspace)
    pair = loop._select_merge_attempt()
    assert pair is not None

    kernel = workspace / "kernel.py"
    uncommitted = kernel.read_text() + "# work this iteration did not create\n"
    kernel.write_text(uncommitted)

    staged, obstacle = loop._stage_merge_attempt(pair)

    assert staged == ""
    assert obstacle == "the working tree already carried uncommitted work"
    assert kernel.read_text() == uncommitted


def test_a_supervisor_intervention_does_not_retire_a_stack_worth_measuring(tmp_path, monkeypatch):
    """The stall a stack answers to is the one only a KEEP clears.

    A memo redirects the next Implementer session; it does not measure the two
    complementary gains already sitting in the archive. Gating on the cooldown
    counter meant each of the 37 interventions across the 2026-08 archives
    silently retired a stall this mechanism was written for.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    loop.run_state.stall.no_improvement_iters = 0
    _stackable_workspace(loop, workspace)

    pair = loop._select_merge_attempt()

    assert pair is not None
    assert {item.iteration for item in pair} == {1, 2}


def test_stacking_waits_until_single_patches_have_stalled(tmp_path, monkeypatch):
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 1
    _stackable_workspace(loop, workspace)

    assert loop._select_merge_attempt() is None


def _stalled_loop_behind_a_full_queue(
    tmp_path,
    monkeypatch,
    *,
    stall=MERGE_ATTEMPT_STALL_THRESHOLD,
    candidates=2,
    iterations=1,
):
    """A stalled run holding two bought candidates and a field of stackable gains."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = stall
    _stackable_workspace(loop, workspace, candidates=candidates)
    monkeypatch.setattr(
        loop,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if len(loop.results) < iterations else 0.0,
    )
    # The archive is already two iterations deep, which is what a stall means.
    loop.resume = True
    subprocess.run(
        ["git", "checkout", "-B", loop.ic.git_branch],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    kernel = workspace / "kernel.py"
    loop._lane_queue = [
        campaign_runtime.LaneResult(
            lane_id=str(lane),
            plan=f"tune line {line}",
            diff=_diff_of(
                workspace,
                lambda line=line: kernel.write_text(
                    kernel.read_text().replace(f"line_{line} = {line}", f"line_{line} = {line}00")
                ),
            ),
        )
        for lane, line in ((1, 5), (2, 6))
    ]
    loop._persist_lane_queue()
    loop.run_state.baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.git_branch = loop.ic.git_branch
    loop.run_state.head_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.state_store.save(loop.run_state)
    return loop, workspace


async def _reverting_iteration(_iteration, **_kwargs):
    """A candidate that measured cleanly and the KEEP gate turned down.

    Which is the outcome a stacked attempt has to have for a streak to be
    possible at all: it leaves ``unresolved_stall_iters`` higher than it found
    it. An ``IterationResult`` the constructor rejects would be caught by the
    loop's own crash guard and archived as a CRASH instead, which is a
    different decision on a different path.
    """
    return IterationResult(
        iteration=_iteration,
        duration_sec=0.0,
        validation_passed=True,
        validation_summary="",
        mean_case_speedup=1.0,
        kept=False,
    )


def test_a_queue_that_never_empties_does_not_starve_stacking(tmp_path, monkeypatch):
    """The queue holds the iteration only while it is still the thing working.

    A fan-out round refills whenever the queue drains, so on the thirty archived
    runs of 2026-08-22 and 08-23 a candidate was waiting on 409 of 549
    iterations. Deferring to that unconditionally is a gate stacking can never
    pass, and the mechanism ran 5 times on archives holding 20 pairs. Once the
    run is as stalled as a stack requires, the queue yields -- and it yields the
    iteration, not the candidates, which are still queued afterwards.
    """
    loop, workspace = _stalled_loop_behind_a_full_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(loop, "run_one_iteration", _reverting_iteration)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    events = LoopStateStore(str(workspace)).read_events()
    staged = [item for item in events if item.get("type") == "merge_attempt_staged"]
    precedence = [item for item in events if item.get("type") == "merge_took_precedence"]

    assert len(staged) == 1
    # The two counts answer different questions and the second cannot be read
    # off the first: how often a stack was measured, and how often one went
    # ahead of a candidate already paid for.
    assert len(precedence) == 1
    assert precedence[0]["lane_queue_depth"] == 2
    assert precedence[0]["unresolved_stall_iters"] == MERGE_ATTEMPT_STALL_THRESHOLD
    assert [item.plan for item in loop._lane_queue] == ["tune line 5", "tune line 6"]


def test_a_fresh_run_still_measures_the_candidates_it_bought(tmp_path, monkeypatch):
    """Precedence is the stall's, not stacking's.

    A queued candidate is kept 55.1% of the time while the search is still
    producing and 33.7% from the stall threshold on; the first of those numbers
    is why the queue keeps the iteration until the run has stopped resolving.
    """
    loop, workspace = _stalled_loop_behind_a_full_queue(tmp_path, monkeypatch, stall=MERGE_ATTEMPT_STALL_THRESHOLD - 1)
    monkeypatch.setattr(loop, "run_one_iteration", _reverting_iteration)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    events = LoopStateStore(str(workspace)).read_events()

    assert not [item for item in events if item.get("type") == "merge_took_precedence"]
    assert [item.plan for item in loop._lane_queue] == ["tune line 6"]


def test_taking_precedence_and_failing_to_stage_costs_the_queue_nothing(tmp_path, monkeypatch):
    """A stack that cannot be built is not a turn anyone spent.

    The obstacle is still reported -- a selected pair that reaches no
    measurement is the failure this mechanism's counters exist to expose -- and
    the candidate that would have been displaced is measured by the same
    iteration, so nothing counts as displaced that was not.
    """
    loop, workspace = _stalled_loop_behind_a_full_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(
        loop,
        "_stage_merge_attempt",
        lambda pair: ("", "iteration 1's diff would not apply over the other's"),
    )
    monkeypatch.setattr(loop, "run_one_iteration", _reverting_iteration)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    events = LoopStateStore(str(workspace)).read_events()

    assert [item["obstacle"] for item in events if item.get("type") == "merge_attempt_declined"] == [
        "iteration 1's diff would not apply over the other's"
    ]
    assert not [item for item in events if item.get("type") == "merge_took_precedence"]
    assert not [item for item in events if item.get("type") == "merge_attempt_staged"]
    assert [item.plan for item in loop._lane_queue] == ["tune line 6"]


def _longest_consecutive_run(iterations):
    """The longest stretch of back-to-back iterations in a sorted list."""
    longest = streak = 0
    previous = None
    for value in iterations:
        streak = streak + 1 if previous is not None and value == previous + 1 else 1
        longest = max(longest, streak)
        previous = value
    return longest


def test_a_merge_streak_is_bounded_so_the_queue_is_reached(tmp_path, monkeypatch):
    """A stall the archive keeps answering is not a licence to hold the loop.

    A stacked attempt reverts, so it leaves ``unresolved_stall_iters`` higher
    than it found it, and it drains nothing, so nothing about having run one
    makes the next one less likely. The pairs give out eventually -- the stack
    a streak archives carries the stacking prefix and ``eligible_candidates``
    skips it, so the pool is frozen while the streak spends it -- but only
    after as many iterations as the pool has pairs, which goes as the square of
    the pool. The queue-empty branch is where the next round is priced, so a
    streak that runs that long is a campaign that never asks whether it can
    still afford one.

    Four archived gains offer four complementary pairs here, which is enough
    to hold two lane candidates for four iterations. What the limit must do to
    them is defer, not drop: all four are still measured, in streaks of at most
    two, and the iteration the limit takes back goes to the queue.
    """
    loop, workspace = _stalled_loop_behind_a_full_queue(tmp_path, monkeypatch, candidates=4, iterations=5)
    monkeypatch.setattr(loop, "run_one_iteration", _reverting_iteration)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    # Each of those iterations has to have reached the KEEP gate and been
    # turned down there, because that is the outcome that leaves the stall
    # standing and lets the next stack be selected. An iteration that raises
    # instead is caught by the loop's own crash guard and archived as a CRASH,
    # which leaves the merge events below asserting over a streak driven by a
    # different decision on a different path.
    assert [result.crashed for result in loop.results] == [False] * 5

    events = LoopStateStore(str(workspace)).read_events()
    staged = [item["iter"] for item in events if item.get("type") == "merge_attempt_staged"]
    precedence = [item for item in events if item.get("type") == "merge_took_precedence"]

    assert _longest_consecutive_run(staged) == MERGE_PRECEDENCE_STREAK_LIMIT
    # The limit defers a pair rather than refusing it: all four are still
    # measured inside these five iterations.
    assert len(staged) == 4
    # And the iteration the limit took back went to the queue, which is one
    # shallower for the last two stacks than it was for the first two.
    assert [item["lane_queue_depth"] for item in precedence] == [2, 2, 1, 1]
    assert [(item["iter"], item["obstacle"]) for item in events if item.get("type") == "merge_attempt_declined"] == [
        (
            staged[1] + 1,
            "2 stacked iterations have run back to back without the queue being reached",
        )
    ]


def test_what_precedence_records_is_a_queue_depth(tmp_path, monkeypatch):
    """The queue's length is not the number of measurements a stack displaces.

    ``_take_lane_candidate`` returns a single candidate, and only after
    ``_next_lane_candidate`` has dropped every entry that would move the
    measurement surface or whose diff no longer applies. So a stack goes ahead
    of at most one measurement and possibly none, and which it is cannot be
    known without popping the queue and writing the tree. Here one of the three
    entries could never have been measured at all and one is still queued when
    the run ends, against a recorded depth of three.
    """
    loop, workspace = _stalled_loop_behind_a_full_queue(tmp_path, monkeypatch, iterations=2)
    loop._lane_queue.insert(
        0,
        campaign_runtime.LaneResult(
            lane_id="surface",
            plan="tune the driver",
            diff=_diff_of(
                workspace,
                lambda: (workspace / "driver.py").write_text("print('bypass')\n"),
            ),
        ),
    )
    loop._persist_lane_queue()
    monkeypatch.setattr(loop, "run_one_iteration", _reverting_iteration)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    events = LoopStateStore(str(workspace)).read_events()

    assert [item["lane_queue_depth"] for item in events if item.get("type") == "merge_took_precedence"] == [3]
    assert [item.plan for item in loop._lane_queue] == ["tune line 6"]
