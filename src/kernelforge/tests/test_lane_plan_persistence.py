# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a fan-out round leaves on disk, and what a later process may reuse.

A round of ``N`` lanes buys ``N`` synthesized plans before it runs a single
session. Only lane 1's was ever published, so the rest lived in one process's
memory: nothing could say afterwards what lanes 2..N had been asked to do, and a
process that died mid-round had to buy the same plans over again.

The first half of this file is about publishing all of them under names that
leave the single-session path's own file exactly where it was. The second half
is about the one round a later process may pick those plans back up from -- the
iteration that started and never reported a result -- and about every reason it
must refuse to.

The last part is about what the round produced rather than what it asked for. A
round spends its candidates one per iteration, so a process that ends with any
of them unspent -- a budget that ran out mid-round is the ordinary way to end,
not only a crash -- is throwing away finished Implementer sessions whose lane
workspaces are already deleted. Those are the expensive part of a round, so they
are published too.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.loop import runner as runner_module
from kernelforge.loop.fanout import LaneResult
from kernelforge.loop.run_state import LoopStateStore, RunState, make_event
from kernelforge.loop.runner import IterationConfig, IterationLoop
from kernelforge.orchestrator.contracts import PlanCriticOutcome
from kernelforge.tracker import ExperimentTracker


class _NoopEvolver:
    def on_experiment_complete(self, experiment):
        return {}


def _loop(tmp_path, monkeypatch):
    """A loop with a committed workspace and a durable event log."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel():\n    return 1\n")
    (workspace / "driver.py").write_text("pass\n")

    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "KernelForge Tests"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    return _open_loop(workspace, tmp_path, monkeypatch), workspace


def _reopen(workspace, tmp_path, monkeypatch):
    """A second loop over a workspace the process before it left behind."""
    return _open_loop(workspace, tmp_path / "reopened", monkeypatch)


def _open_loop(workspace, experiments_root, monkeypatch):
    monkeypatch.setattr(runner_module, "force_jit_rebuild", lambda _files: None)

    loop = IterationLoop(
        IterationConfig(
            kernel_file=str(workspace / "kernel.py"),
            driver_script=str(workspace / "driver.py"),
            baseline_wall_ms=1.0,
            baseline_case_times={"case": 1.0},
            max_time_hours=1.0,
            git_branch="test-lanes",
            workspace_dir=str(workspace),
            lanes=3,
        ),
        ExperimentTracker(experiments_root / "experiments"),
        config=object(),
        evolver=_NoopEvolver(),
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    # Per-run state the loop sets up when it starts, which these tests reach
    # without starting it. The clock matters: a round is dispatched only when
    # the budget can still pay for a session and its measurement, and an unset
    # start time reads as a campaign that has already spent its hour.
    loop._lane_queue = []
    loop.start_time = time.time()
    return loop


def _round(loop, iteration, plans, *, analysis_commit="commit-a"):
    """Publish one round exactly as ``_run_orchestration`` publishes it."""
    return loop._persist_lane_plans(iteration, plans, analysis_commit=analysis_commit)


def _started(loop, iteration):
    loop.state_store.append_event(make_event("iteration_started", iteration))


def _finished(loop, iteration):
    loop.state_store.append_event(make_event("iteration_result", iteration, decision="REVERT_PERF"))


def _head_commit(loop) -> str:
    """The commit a round planned now would be attributed to."""
    return loop._canonical_commit()


# --------------------------------------------------------------------------
# Publishing
# --------------------------------------------------------------------------


def test_every_lane_plan_is_published_not_only_the_first(tmp_path, monkeypatch):
    """Each plan is a separately paid-for answer, so each gets a file."""
    loop, workspace = _loop(tmp_path, monkeypatch)

    _round(loop, 7, ["widen the loads", "stage through LDS", "fuse the epilogue"])

    root = workspace / "forge_experiments" / "orchestration" / "iter_007"
    assert (root / "optimization_plan.md").read_text() == "widen the loads\n"
    assert (root / "lane_002.md").read_text() == "stage through LDS\n"
    assert (root / "lane_003.md").read_text() == "fuse the epilogue\n"


def test_lane_one_keeps_the_name_the_rest_of_the_loop_reads(tmp_path, monkeypatch):
    """The archive, the handoffs and the supervisor all read this file by name.

    None of them knows how wide the round was, so widening a round must not
    move the plan they are pointed at.
    """
    loop, _workspace = _loop(tmp_path, monkeypatch)

    published = _round(loop, 2, ["widen the loads", "stage through LDS"])

    assert published.name == "optimization_plan.md"
    assert published == loop._lane_plan_path(2, 1)


def test_a_single_lane_round_publishes_exactly_what_it_always_did(tmp_path, monkeypatch):
    """--lanes 1 predates all of this and its artifacts must be unchanged."""
    loop, workspace = _loop(tmp_path, monkeypatch)

    _round(loop, 1, ["widen the loads"])

    root = workspace / "forge_experiments" / "orchestration" / "iter_001"
    assert sorted(path.name for path in root.glob("*.md")) == ["optimization_plan.md"]


def test_replanning_one_iteration_narrower_removes_the_wider_leftovers(tmp_path, monkeypatch):
    """A fan-out that loses its lane copies re-plans the same iteration at one.

    A leftover ``lane_003.md`` from the abandoned wider round would be read
    back as a plan this iteration never issued.
    """
    loop, workspace = _loop(tmp_path, monkeypatch)
    _round(loop, 4, ["widen the loads", "stage through LDS", "fuse the epilogue"])

    _round(loop, 4, ["rewrite the whole thing"], analysis_commit="commit-b")

    root = workspace / "forge_experiments" / "orchestration" / "iter_004"
    assert sorted(path.name for path in root.glob("*.md")) == ["optimization_plan.md"]
    assert loop._load_lane_plans(4) == ("commit-b", ["rewrite the whole thing"])


def test_an_empty_plan_is_refused_rather_than_published(tmp_path, monkeypatch):
    """An empty plan would be handed to a lane as an empty instruction."""
    loop, _workspace = _loop(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        loop._persist_lane_plans(1, [], analysis_commit="commit-a")
    with pytest.raises(ValueError):
        loop._persist_lane_plans(1, ["widen the loads", "   "], analysis_commit="commit-a")


def test_plans_without_the_commit_they_describe_are_refused(tmp_path, monkeypatch):
    """Unattributed plans could never be reused, so they are not published."""
    loop, _workspace = _loop(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        loop._persist_lane_plans(1, ["widen the loads"], analysis_commit="")


def test_published_plans_are_read_back_in_lane_order(tmp_path, monkeypatch):
    """Lane order is the assignment; a reordered read would swap two lanes."""
    loop, _workspace = _loop(tmp_path, monkeypatch)
    plans = ["widen the loads", "stage through LDS", "fuse the epilogue"]

    _round(loop, 9, plans, analysis_commit="commit-a")

    assert loop._load_lane_plans(9) == ("commit-a", plans)


def test_a_round_that_published_nothing_reads_back_as_nothing(tmp_path, monkeypatch):
    """Most iterations run no orchestration at all; that is not a failure."""
    loop, _workspace = _loop(tmp_path, monkeypatch)

    assert loop._load_lane_plans(3) is None


def test_plans_without_their_manifest_are_read_back_as_nothing(tmp_path, monkeypatch):
    """The manifest is written last, so plan files without it are a dead round.

    Treating them as publishable would let a process that died mid-write hand
    the next one a set it never finished assembling.
    """
    loop, _workspace = _loop(tmp_path, monkeypatch)
    _round(loop, 5, ["widen the loads", "stage through LDS"])

    loop._lane_plan_manifest_path(5).unlink()

    assert loop._load_lane_plans(5) is None


def test_a_missing_lane_is_damage_rather_than_a_narrower_round(tmp_path, monkeypatch):
    """The manifest counted that lane, and it was paid for like the rest."""
    loop, _workspace = _loop(tmp_path, monkeypatch)
    _round(loop, 5, ["widen the loads", "stage through LDS", "fuse the epilogue"])

    loop._lane_plan_path(5, 2).unlink()

    assert loop._load_lane_plans(5) is None


# --------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------


def test_plans_of_a_round_that_never_reported_a_result_are_recovered(tmp_path, monkeypatch):
    """This is the round the crash cost: bought in full, dispatched never.

    The iteration asking has already marked itself started, because the loop
    does that before it plans anything. It is therefore itself started and
    unfinished, and answering for it rather than for the round behind it is
    how nothing at all gets recovered.
    """
    loop, _workspace = _loop(tmp_path, monkeypatch)
    plans = ["widen the loads", "stage through LDS", "fuse the epilogue"]
    _round(loop, 6, plans, analysis_commit=_head_commit(loop))
    _started(loop, 6)
    _started(loop, 7)

    assert loop._recoverable_lane_plans(7) == (6, plans)


def test_plans_of_a_round_that_reported_a_result_are_not_recovered(tmp_path, monkeypatch):
    """A finished round spent its plans, and the loop has ruled on them.

    Handing them back would re-issue directions that were already measured.
    """
    loop, _workspace = _loop(tmp_path, monkeypatch)
    _round(
        loop,
        6,
        ["widen the loads", "stage through LDS"],
        analysis_commit=_head_commit(loop),
    )
    _started(loop, 6)
    _finished(loop, 6)

    assert loop._recoverable_lane_plans(7) is None


def test_only_the_latest_unfinished_round_is_answered_for(tmp_path, monkeypatch):
    """An older gap is an iteration the loop already moved past."""
    loop, _workspace = _loop(tmp_path, monkeypatch)
    _round(
        loop,
        2,
        ["widen the loads", "stage through LDS"],
        analysis_commit=_head_commit(loop),
    )
    _started(loop, 2)
    _started(loop, 3)
    _finished(loop, 3)
    _started(loop, 4)

    assert loop._recoverable_lane_plans(4) is None


def test_plans_written_against_another_commit_are_refused(tmp_path, monkeypatch):
    """A KEEP recovered before this runs leaves the plans describing old code."""
    loop, _workspace = _loop(tmp_path, monkeypatch)
    _round(
        loop,
        6,
        ["widen the loads", "stage through LDS"],
        analysis_commit="a-commit-the-tree-has-moved-off",
    )
    _started(loop, 6)

    assert loop._recoverable_lane_plans(7) is None


def test_plans_whose_manifest_cannot_be_read_are_refused(tmp_path, monkeypatch):
    """Without the manifest nothing can vouch for what the round planned."""
    loop, _workspace = _loop(tmp_path, monkeypatch)
    _round(
        loop,
        6,
        ["widen the loads", "stage through LDS"],
        analysis_commit=_head_commit(loop),
    )
    _started(loop, 6)
    loop._lane_plan_manifest_path(6).write_text("{ truncated")

    assert loop._recoverable_lane_plans(7) is None


def test_a_single_recovered_plan_is_not_a_fan_out_round(tmp_path, monkeypatch):
    """One plan is the ordinary path, which plans its own round regardless."""
    loop, _workspace = _loop(tmp_path, monkeypatch)
    _round(loop, 6, ["widen the loads"], analysis_commit=_head_commit(loop))
    _started(loop, 6)

    assert loop._recoverable_lane_plans(7) is None


def test_the_current_iteration_is_never_recovered_from_itself(tmp_path, monkeypatch):
    """Its own ``iteration_started`` is on the log before the round is planned."""
    loop, _workspace = _loop(tmp_path, monkeypatch)
    _round(
        loop,
        6,
        ["widen the loads", "stage through LDS"],
        analysis_commit=_head_commit(loop),
    )
    _started(loop, 6)

    assert loop._recoverable_lane_plans(6) is None


def _dispatch_without_planning(loop, monkeypatch) -> dict:
    """Let a round reach its lanes, and fail loudly if it plans on the way."""
    dispatched: dict = {}

    async def _never(**_kwargs):
        raise AssertionError("a recovered round must not be planned again")

    async def _fill(*, iteration=0, agent_factory, lane_plans):
        dispatched["plans"] = list(lane_plans)

    monkeypatch.setattr(loop, "_run_orchestration", _never)
    monkeypatch.setattr(loop, "_fill_lane_queue", _fill)
    return dispatched


async def test_a_recovered_round_runs_its_lanes_without_planning_again(tmp_path, monkeypatch):
    """The tokens the crash cost are the planning; that is what is saved."""
    loop, _workspace = _loop(tmp_path, monkeypatch)
    plans = ["widen the loads", "stage through LDS", "fuse the epilogue"]
    _round(loop, 6, plans, analysis_commit=_head_commit(loop))
    _started(loop, 6)
    dispatched = _dispatch_without_planning(loop, monkeypatch)

    await loop._fan_out_round(
        iteration=7,
        orchestration_service=None,
        agent_factory=None,
    )

    assert dispatched["plans"] == plans
    assert loop._last_lane_plans == plans


async def test_a_recovered_round_republishes_under_its_own_iteration(tmp_path, monkeypatch):
    """Otherwise a second crash loses plans the first one had already saved.

    The recovered round is the one that runs the plans, so it has to be as
    recoverable as the round it inherited them from -- and its own artifacts
    have to say what it did.
    """
    loop, _workspace = _loop(tmp_path, monkeypatch)
    plans = ["widen the loads", "stage through LDS", "fuse the epilogue"]
    _round(loop, 6, plans, analysis_commit=_head_commit(loop))
    _started(loop, 6)
    _dispatch_without_planning(loop, monkeypatch)

    await loop._fan_out_round(
        iteration=7,
        orchestration_service=None,
        agent_factory=None,
    )

    assert loop._load_lane_plans(7) == (_head_commit(loop), plans)
    assert loop._latest_optimization_plan_path == str(loop._lane_plan_path(7, 1))

    _started(loop, 7)

    assert loop._recoverable_lane_plans(8) == (7, plans)


async def test_a_republish_that_fails_costs_the_round_not_the_campaign(tmp_path, monkeypatch, capsys):
    """A workspace too full for Markdown is too full for N workspace copies.

    The ordinary single-session path handles the iteration from here, which is
    the loop's standing answer to a lane-infrastructure failure. It is handed
    nothing, because nothing reached disk and nothing was spent: this is the one
    fallback that has to plan for itself.
    """
    loop, _workspace = _loop(tmp_path, monkeypatch)
    _round(
        loop,
        6,
        ["widen the loads", "stage through LDS"],
        analysis_commit=_head_commit(loop),
    )
    _started(loop, 6)
    _dispatch_without_planning(loop, monkeypatch)

    def _no_room(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(loop, "_persist_lane_plans", _no_room)

    held = await loop._fan_out_round(
        iteration=7,
        orchestration_service=None,
        agent_factory=None,
    )

    assert held is None
    assert loop._lane_queue == []
    assert "No space left on device" in capsys.readouterr().out


async def test_a_recovered_round_that_cannot_dispatch_spends_its_plans_anyway(tmp_path, monkeypatch):
    """Plans a crash left behind describe this tree and cost this round nothing.

    Losing the lane copies loses the sessions, not the planning, so the
    single-session path that takes the iteration over is handed the round's own
    republished plan rather than sent to buy the same answer again.
    """
    loop, _workspace = _loop(tmp_path, monkeypatch)
    plans = ["widen the loads", "stage through LDS"]
    _round(loop, 6, plans, analysis_commit=_head_commit(loop))
    _started(loop, 6)
    _dispatch_without_planning(loop, monkeypatch)

    async def _no_room(**_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(loop, "_fill_lane_queue", _no_room)

    held = await loop._fan_out_round(
        iteration=7,
        orchestration_service=None,
        agent_factory=None,
    )

    assert held == (loop._lane_plan_path(7, 1), "")


async def test_a_round_with_nothing_to_recover_is_planned_as_usual(tmp_path, monkeypatch):
    """Guards the recovery assertions above from passing for the wrong reason."""
    loop, _workspace = _loop(tmp_path, monkeypatch)
    dispatched: dict = {}

    async def _plan(*, iteration, lanes, **_kwargs):
        plans = ["widen the loads", "stage through LDS"]
        loop._last_lane_plans = plans
        return _round(loop, iteration, plans, analysis_commit=_head_commit(loop)), ""

    async def _fill(*, iteration=0, agent_factory, lane_plans):
        dispatched["plans"] = list(lane_plans)

    monkeypatch.setattr(loop, "_run_orchestration", _plan)
    monkeypatch.setattr(loop, "_fill_lane_queue", _fill)

    await loop._fan_out_round(
        iteration=1,
        orchestration_service=None,
        agent_factory=None,
    )

    assert dispatched["plans"] == ["widen the loads", "stage through LDS"]


async def test_a_round_whose_planning_spent_the_budget_keeps_its_plans(tmp_path, monkeypatch):
    """The production death, refused -- and refused without losing the plans.

    Planning is bought before anyone can know what it cost, so the round is
    stopped between its plans and its sessions. What that leaves on disk is an
    iteration that started and reported no result, holding published plans:
    exactly the state the recovery above picks a round back up from.
    """
    loop, _workspace = _loop(tmp_path, monkeypatch)
    plans = ["widen the loads", "stage through LDS", "fuse the epilogue"]
    dispatched: dict = {}

    async def _plan(*, iteration, lanes, **_kwargs):
        loop._last_lane_plans = plans
        # Planning returns with 7.3 minutes left, as the round that was killed
        # in production did: enough to start three lane sessions, not enough to
        # measure what any of them writes.
        monkeypatch.setattr(loop, "_time_remaining", lambda: 7.3 * 60.0)
        return (
            _round(loop, iteration, plans, analysis_commit=_head_commit(loop)),
            "",
        )

    async def _fill(*, agent_factory, lane_plans):
        dispatched["plans"] = list(lane_plans)

    monkeypatch.setattr(loop, "_run_orchestration", _plan)
    monkeypatch.setattr(loop, "_fill_lane_queue", _fill)
    _started(loop, 1)

    held = await loop._fan_out_round(
        iteration=1,
        orchestration_service=None,
        agent_factory=None,
    )

    assert dispatched == {}
    assert held == (loop._lane_plan_path(1, 1), "")
    assert loop.termination_reason == "round_budget_exhausted"
    # The next process, which marks its own iteration started before it plans.
    _started(loop, 2)
    assert loop._recoverable_lane_plans(2) == (1, plans)


async def test_a_recovered_round_refused_for_budget_stays_recoverable(tmp_path, monkeypatch):
    """A refusal must not lose plans a crash already saved once.

    The recovered round is the iteration the next process will find unfinished,
    so the plans have to be republished under it before the round is priced --
    otherwise the refusal quietly retires the very plans it is protecting.
    """
    loop, _workspace = _loop(tmp_path, monkeypatch)
    plans = ["widen the loads", "stage through LDS", "fuse the epilogue"]
    _round(loop, 6, plans, analysis_commit=_head_commit(loop))
    _started(loop, 6)
    dispatched = _dispatch_without_planning(loop, monkeypatch)
    monkeypatch.setattr(loop, "_time_remaining", lambda: 7.3 * 60.0)
    _started(loop, 7)

    await loop._fan_out_round(
        iteration=7,
        orchestration_service=None,
        agent_factory=None,
    )

    assert dispatched == {}
    assert loop.termination_reason == "round_budget_exhausted"
    assert loop._load_lane_plans(7) == (_head_commit(loop), plans)
    _started(loop, 8)
    assert loop._recoverable_lane_plans(8) == (7, plans)


async def test_a_planned_round_records_the_commit_recovery_compares_against(tmp_path, monkeypatch):
    """The seam between publishing a round and picking it back up.

    Planning attributes its plans to the analysis context's commit, and recovery
    compares against the canonical commit. Those are derived in different places,
    and if they ever drift apart nothing goes red: the manifest is still written,
    recovery still refuses, and the feature simply stops firing. So the real
    orchestration path is run here rather than stubbed.
    """
    loop, _workspace = _loop(tmp_path, monkeypatch)
    loop.config = SimpleNamespace(
        experiments_dir=Path(loop.ic.workspace_dir) / "forge_experiments",
        gpu_target="gfx942",
    )
    loop._active_analysis_context = loop._build_orchestration_context()
    plans = ("widen the loads", "stage through LDS")

    class OrchestrationService:
        async def run(self, context, **_kwargs):
            return SimpleNamespace(
                dispatch_plan=None,
                specialist_outcomes=(),
                optimization_plans=plans,
                optimization_plan_executable=True,
                optimization_plan_draft="",
                structured_output_diagnostics={},
                plan_critic=None,
                plan_revised=False,
            )

    await loop._run_orchestration(
        iteration=4,
        orchestration_service=OrchestrationService(),
        lanes=2,
    )

    assert loop._load_lane_plans(4) == (loop._canonical_commit(), list(plans))

    _started(loop, 4)

    assert loop._recoverable_lane_plans(5) == (4, list(plans))


# --------------------------------------------------------------------------
# The candidates a round bought and has not yet measured
# --------------------------------------------------------------------------


def _queued(loop, *lanes):
    """Queue candidates the way a finished round's sessions leave them."""
    loop._lane_queue = [LaneResult(lane_id=lane_id, plan=plan, diff=diff) for lane_id, plan, diff in lanes]
    loop._persist_lane_queue()


def _widen(workspace):
    """A kernel with room for two candidates that do not overlap."""
    kernel = workspace / "kernel.py"
    kernel.write_text("\n".join(f"line_{n} = {n}" for n in range(12)) + "\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "wide"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    canonical = kernel.read_text()

    def diff_of(old, new):
        kernel.write_text(canonical.replace(old, new))
        patch = subprocess.run(["git", "diff"], cwd=workspace, capture_output=True, text=True).stdout
        kernel.write_text(canonical)
        return patch

    return diff_of


def test_a_candidate_a_round_bought_outlives_the_process(tmp_path, monkeypatch):
    """The sessions are the expensive part of a round, and they are finished.

    Losing them is not the same trade as losing a plan: a plan can be bought
    again, whereas the session that wrote this diff has already run and its
    lane workspace has already been deleted.
    """
    loop, workspace = _loop(tmp_path, monkeypatch)
    _queued(loop, ("1", "widen the loads", "patch-1"), ("2", "stage LDS", "patch-2"))

    later = _reopen(workspace, tmp_path, monkeypatch)
    later._restore_lane_queue()

    assert [(item.lane_id, item.plan, item.diff) for item in later._lane_queue] == [
        ("1", "widen the loads", "patch-1"),
        ("2", "stage LDS", "patch-2"),
    ]


def test_a_measured_candidate_is_not_left_for_the_next_process(tmp_path, monkeypatch):
    """Reusing one the loop has already ruled on would re-run a spent lane."""
    loop, workspace = _loop(tmp_path, monkeypatch)
    diff_of = _widen(workspace)
    _queued(
        loop,
        ("1", "tune prefill", diff_of("line_0 = 0", "line_0 = 100")),
        ("2", "tune decode", diff_of("line_11 = 11", "line_11 = 111")),
    )

    taken = loop._take_lane_candidate()

    later = _reopen(workspace, tmp_path, monkeypatch)
    later._restore_lane_queue()

    assert taken is not None and taken.lane_id == "1"
    assert [item.lane_id for item in later._lane_queue] == ["2"]


def test_a_spent_round_leaves_no_queue_behind(tmp_path, monkeypatch):
    """An empty queue is the absence of a file, not a file holding nothing."""
    loop, workspace = _loop(tmp_path, monkeypatch)
    diff_of = _widen(workspace)
    _queued(loop, ("1", "tune prefill", diff_of("line_0 = 0", "line_0 = 100")))

    assert loop._lane_queue_path().exists()

    loop._take_lane_candidate()
    loop._git_discard_worktree()

    later = _reopen(workspace, tmp_path, monkeypatch)
    later._restore_lane_queue()

    assert not loop._lane_queue_path().exists()
    assert later._lane_queue == []


def test_a_round_that_queued_nothing_publishes_nothing(tmp_path, monkeypatch):
    """Guards the assertions above from passing for the wrong reason."""
    loop, workspace = _loop(tmp_path, monkeypatch)

    loop._persist_lane_queue()

    later = _reopen(workspace, tmp_path, monkeypatch)
    later._restore_lane_queue()

    assert not loop._lane_queue_path().exists()
    assert later._lane_queue == []


def test_a_queue_that_cannot_be_written_costs_durability_only(tmp_path, monkeypatch, capsys):
    """The candidates are in memory and this process still measures them.

    Ending a campaign because a few KB of JSON would not land would cost the
    run more than the durability it was protecting.
    """
    loop, _workspace = _loop(tmp_path, monkeypatch)
    loop._lane_queue = [LaneResult(lane_id="1", plan="widen", diff="patch-1")]

    def _no_room(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(runner_module, "atomic_write_json", _no_room, raising=False)
    monkeypatch.setattr("kernelforge.loop.recovery.atomic_write_json", _no_room)

    loop._persist_lane_queue()

    assert [item.lane_id for item in loop._lane_queue] == ["1"]
    assert "No space left on device" in capsys.readouterr().out


def test_a_restored_queue_is_measured_before_a_new_round_is_planned(tmp_path, monkeypatch):
    """Fanning out on top of unspent candidates would buy a round twice over."""
    loop, workspace = _loop(tmp_path, monkeypatch)
    diff_of = _widen(workspace)
    _queued(loop, ("1", "tune prefill", diff_of("line_0 = 0", "line_0 = 100")))

    later = _reopen(workspace, tmp_path, monkeypatch)
    later._restore_lane_queue()

    # The loop only fans out when nothing is queued, so a restored queue is
    # what the next iteration measures.
    assert later._lane_queue != []


# --------------------------------------------------------------------------
# The verdict that outlives the round it judged
# --------------------------------------------------------------------------


def _reviewed(loop, iteration, verdict, review):
    """Record one critic ruling exactly as ``_run_orchestration`` records it."""
    root = loop._orchestration_root(iteration)
    root.mkdir(parents=True, exist_ok=True)
    (root / "critic_review.md").write_text(review + "\n", encoding="utf-8")
    loop._record_critic_ruling(
        iteration,
        PlanCriticOutcome(verdict=verdict, review=review),
    )
    loop.state_store.save(loop.run_state)


def test_a_replace_verdict_outlives_the_process_that_recorded_it(tmp_path, monkeypatch):
    """A critic rules on a round already planned, so its verdict is spent next.

    The budget routinely ends between those two rounds, and a ruling held only
    in memory died exactly there: the process that resumed divided the route
    the critic had just called dominated.
    """
    loop, workspace = _loop(tmp_path, monkeypatch)
    _reviewed(loop, 4, "REPLACE", "A CK GEMM already exists for this shape.")

    later = _reopen(workspace, tmp_path, monkeypatch)
    later.run_state = later.state_store.load()
    later._restore_critic_ruling()

    assert later._last_critic_verdict == "REPLACE"
    assert "A CK GEMM already exists" in later._last_critic_review


def test_a_ruling_whose_review_is_gone_is_not_resumed(tmp_path, monkeypatch):
    """The challenge lives in the review, not in the word REPLACE.

    A verdict restored without one would ask the round to validate an
    alternative nobody ever named.
    """
    loop, workspace = _loop(tmp_path, monkeypatch)
    _reviewed(loop, 4, "REPLACE", "A CK GEMM already exists for this shape.")
    (loop._orchestration_root(4) / "critic_review.md").unlink()

    later = _reopen(workspace, tmp_path, monkeypatch)
    later.run_state = later.state_store.load()
    later._restore_critic_ruling()

    assert later._last_critic_verdict == ""
    assert later.run_state.last_critic.verdict == ""


def test_a_fail_open_review_records_no_ruling(tmp_path, monkeypatch):
    """Its artifact holds the outage that stopped it, not a judgement."""
    loop, _workspace = _loop(tmp_path, monkeypatch)

    loop._record_critic_ruling(
        4,
        PlanCriticOutcome(
            verdict="REVISE",
            error="backend timed out",
            verdict_source="error",
        ),
    )

    assert loop.run_state.last_critic.verdict == ""
    assert loop.run_state.last_critic.review_path == ""
