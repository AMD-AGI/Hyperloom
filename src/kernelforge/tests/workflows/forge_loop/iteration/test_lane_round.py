# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One round: its lanes work in copies of the workspace, and what they leave is merged into the one candidate measured."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from kernelforge.config import Config
from kernelforge.evaluation.scoring import GateScoring
from kernelforge.utils.device_queue import RUNNER_CONFIG_NAME, RUNNER_NAME
from kernelforge.workflows.forge_loop.bootstrap.lanes import LaneBinding, _make_lane_agent_factory
from kernelforge.workflows.forge_loop.campaign.experiments_layout import ExperimentsLayout
from kernelforge.workflows.forge_loop.campaign.run_state import LoopStateStore
from kernelforge.workflows.forge_loop.iteration.fanout import (
    LANE_ROOT_PREFIX,
    LaneAssignment,
    LaneResult,
    LaneRound,
    campaign_device_queue_path,
    lane_diff,
)
from kernelforge.workflows.forge_loop.roles.agent import ImplementerSetup, ImplementerTask
from kernelforge.workflows.forge_loop.iteration.iteration_types import LoopAgents
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _initialize_workspace,
    _make_loop,
    _no_change_agent,
    _stub_measurement,
    _unused_supervisor,
    lane_agents,
)

MERGE_HEADING = "## Direction: merge this round's lanes"


def _decisions(workspace: Path) -> list[str]:
    return [
        event["decision"]
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]


def _lane_copies(workspace: Path) -> list[Path]:
    return [path for path in workspace.parent.iterdir() if path.name.startswith(LANE_ROOT_PREFIX)]


def _verified(session_sink: dict, score: float, case_ms: float) -> None:
    """What a lane's own stop check leaves when it found the candidate correct and measured it."""
    session_sink.update(
        gate_passed=True,
        mean_case_speedup=score,
        benchmark_measurement={"success": True, "case_times": {"case": case_ms}},
    )


# ── the workspace copies ──────────────────────────────────────────────────────


async def test_a_lane_copy_leaves_the_analysis_evidence_behind_and_carries_its_device_runner(tmp_path, monkeypatch):
    workspace, _kernel, _driver = _initialize_workspace(tmp_path, monkeypatch)
    layout = ExperimentsLayout.of(workspace)
    (layout.analysis / "abc123").mkdir(parents=True)
    (layout.analysis / "abc123" / "report.md").write_text("gigabytes of profiles stand behind this\n")
    layout.lessons.mkdir(parents=True)
    (layout.lessons / "iter_001.md").write_text("a record\n")
    (workspace / "build.cache").write_text("untracked build state\n")
    lanes = LaneRound(
        workspace_dir=str(workspace),
        parent_dir=str(tmp_path),
        lanes=[LaneAssignment("1"), LaneAssignment("2", "b")],
        hold_limit_sec=90,
    )

    seats = await lanes.open()
    try:
        for seat in seats:
            copy = ExperimentsLayout.of(seat.lane_dir)
            assert (seat.lane_dir / "kernel.py").read_text() == (workspace / "kernel.py").read_text()
            assert (seat.lane_dir / "build.cache").is_file()
            assert (copy.lessons / "iter_001.md").is_file()
            assert not copy.analysis.exists()
            # The runner and its configuration are the lane's, and no lane's diff carries them.
            config = json.loads((seat.lane_dir / RUNNER_CONFIG_NAME).read_text())
            assert seat.device_runner == seat.lane_dir / RUNNER_NAME and seat.device_runner.is_file()
            assert config["queue_root"] == str(campaign_device_queue_path(workspace))
            assert config["lane"] == seat.assignment.lane_id
            assert config["hold_limit_sec"] == 90
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=seat.lane_dir, capture_output=True, text=True, check=True
            )
            assert RUNNER_NAME not in status.stdout
        assert seats[0].lane_dir != seats[1].lane_dir
    finally:
        lanes.close()

    assert _lane_copies(workspace) == []


async def test_a_lane_diff_carries_the_new_files_it_is_given_and_leaves_the_lanes_index_alone(tmp_path, monkeypatch):
    workspace, _kernel, _driver = _initialize_workspace(tmp_path, monkeypatch)
    lanes = LaneRound(
        workspace_dir=str(workspace), parent_dir=str(tmp_path), lanes=[LaneAssignment("1")], hold_limit_sec=90
    )
    (seat,) = await lanes.open()
    try:
        (seat.lane_dir / "kernel.py").write_text("def kernel():\n    return 2\n")
        (seat.lane_dir / "configs").mkdir()
        (seat.lane_dir / "configs" / "small.json").write_text("{}\n")

        tracked_only = lane_diff(seat.lane_dir)
        with_new_file = lane_diff(seat.lane_dir, ["configs/small.json"])
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"], cwd=seat.lane_dir, capture_output=True, text=True, check=True
        )
    finally:
        lanes.close()

    assert "configs/small.json" not in tracked_only
    assert "return 2" in with_new_file and "configs/small.json" in with_new_file
    assert staged.stdout == ""


def _worktree_workspace(tmp_path: Path) -> Path:
    """A workspace that is a git worktree: its ``.git`` is a file pointing into another repository."""
    main = tmp_path / "main"
    main.mkdir()
    (main / "kernel.py").write_text("VALUE = 0\n")
    (main / "forge_driver.py").write_text("pass\n")
    for command in (
        ["git", "init"],
        ["git", "config", "user.email", "tests@example.com"],
        ["git", "config", "user.name", "KernelForge Tests"],
        ["git", "add", "kernel.py", "forge_driver.py"],
        ["git", "commit", "-m", "initial"],
    ):
        subprocess.run(command, cwd=main, check=True, capture_output=True)
    workspace = tmp_path / "workspace"
    subprocess.run(
        ["git", "worktree", "add", "-b", "campaign", str(workspace)], cwd=main, check=True, capture_output=True
    )
    (workspace / "build").mkdir()
    (workspace / "build" / "cached.o").write_text("untracked build output\n")
    return workspace


def _git_output(workspace: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True, text=True).stdout


async def test_a_lane_of_a_worktree_workspace_gets_its_own_index(tmp_path):
    """`cp -a` copies the .git pointer file, so a lane would share the workspace's index."""
    workspace = _worktree_workspace(tmp_path)
    lanes = LaneRound(
        workspace_dir=str(workspace), parent_dir=str(tmp_path), lanes=[LaneAssignment("1")], hold_limit_sec=90
    )

    (seat,) = await lanes.open()
    try:
        (seat.lane_dir / "kernel.py").write_text("VALUE = 1\n")
        subprocess.run(["git", "add", "kernel.py"], cwd=seat.lane_dir, check=True, capture_output=True)
        diff = lane_diff(seat.lane_dir)
    finally:
        lanes.close()

    assert "VALUE = 1" in diff
    assert _git_output(workspace, "status", "--porcelain") == "?? build/\n"
    assert _git_output(workspace, "diff", "HEAD", "--", ".") == ""
    assert (workspace / "kernel.py").read_text() == "VALUE = 0\n"


async def test_lanes_refuse_to_copy_a_workspace_that_will_not_fit(tmp_path, monkeypatch):
    """A full filesystem must be a refusal with numbers, not a half-copied lane."""
    workspace, _kernel, _driver = _initialize_workspace(tmp_path, monkeypatch)
    before = set(tmp_path.iterdir())
    room = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _directory: room._replace(free=4096))
    lanes = LaneRound(
        workspace_dir=str(workspace),
        parent_dir=str(tmp_path),
        lanes=[LaneAssignment("1", "a"), LaneAssignment("2", "b")],
        hold_limit_sec=90,
    )

    with pytest.raises(RuntimeError) as error:
        await lanes.open()

    assert lanes.seats == []
    assert "4096" in str(error.value)
    assert set(tmp_path.iterdir()) == before


# ── a round of one lane ───────────────────────────────────────────────────────


def _no_lane_copies(_seat):
    raise AssertionError("a round of one lane works in the workspace itself and builds no lane session")


def test_a_round_of_one_lane_works_in_the_workspace_itself(tmp_path, monkeypatch):
    """A copy separates lanes that run side by side; one lane has nothing to be separated from."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    _stub_measurement(monkeypatch, walls=[0.5])
    worked_in: list[Path] = []
    copies_while_working: list[list[Path]] = []
    scorings: list[GateScoring] = []

    async def agent(kernel_path, history, session_sink, gate_scoring):
        worked_in.append(Path(kernel_path))
        copies_while_working.append(_lane_copies(workspace))
        scorings.append(gate_scoring)
        assert MERGE_HEADING not in history and "## This lane's focus" not in history
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        session_sink["plan"] = "return the faster constant"
        return "changed the constant"

    asyncio.run(loop.run(LoopAgents(agent_fn=agent, agent_factory=_no_lane_copies, supervisor_fn=_unused_supervisor)))

    assert [path.resolve() for path in worked_in] == [(workspace / "kernel.py").resolve()]
    assert copies_while_working == [[]]
    # With no merge to use anything less, the session's stop is judged against the current best.
    assert [scoring.best_mean_case_speedup for scoring in scorings] == [1.0]
    assert _decisions(workspace) == ["KEEP"]
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 2\n"
    # Lane artifacts are what a merge reads; a round with nothing to merge files none.
    assert not ExperimentsLayout.of(workspace).round(1).exists()


def _move_the_repository_out(workspace: Path, monkeypatch) -> Path:
    """Keep the workspace's repository elsewhere and select it through the environment, as forge-fuse does."""
    repository = workspace.parent / "shadow.git"
    (workspace / ".git").rename(repository)
    monkeypatch.setenv("GIT_DIR", str(repository))
    monkeypatch.setenv("GIT_WORK_TREE", str(workspace))
    return repository


def test_a_round_of_one_lane_keeps_its_change_on_a_workspace_whose_repository_is_kept_elsewhere(tmp_path, monkeypatch):
    """A framework tree edited where it is installed cannot be copied: git in a copy would still read the tree.

    In a lane copy the session's edit came back as an empty diff and the round had no candidate.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    _stub_measurement(monkeypatch, walls=[0.5])
    repository = _move_the_repository_out(workspace, monkeypatch)

    async def agent(kernel_path, _history, session_sink):
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        return "changed the constant"

    asyncio.run(loop.run(LoopAgents(agent_fn=agent, agent_factory=_no_lane_copies, supervisor_fn=_unused_supervisor)))

    assert _decisions(workspace) == ["KEEP"]
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 2\n"
    kept = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"], capture_output=True, text=True, check=True
    )
    assert kept.stdout.split() == ["kernel.py"]
    assert repository.is_dir() and not (workspace / ".git").exists()
    assert _lane_copies(workspace) == []


async def test_lanes_side_by_side_are_refused_on_a_workspace_whose_repository_is_kept_elsewhere(tmp_path, monkeypatch):
    """Refused with the reason rather than run: every lane's diff would be read from the workspace, and be empty."""
    workspace, _kernel, _driver = _initialize_workspace(tmp_path, monkeypatch)
    _move_the_repository_out(workspace, monkeypatch)
    before = set(tmp_path.iterdir())
    lanes = LaneRound(
        workspace_dir=str(workspace),
        parent_dir=str(tmp_path),
        lanes=[LaneAssignment("1", "a"), LaneAssignment("2", "b")],
        hold_limit_sec=90,
    )

    with pytest.raises(RuntimeError, match="GIT_DIR and GIT_WORK_TREE"):
        await lanes.open()

    assert lanes.seats == []
    assert set(tmp_path.iterdir()) == before


def test_the_campaigns_lane_factory_carries_the_scoring_state_in_and_the_lanes_result_out(tmp_path, monkeypatch):
    """The factory a campaign is started with stands between the loop and the session it builds.

    What the loop hands a session has to cross it in both directions: the scoring state the lane's stop check
    measures against, and the sink the round reads the lane's verdict and score from.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times={"case": 4.0})
    loop.ic.lanes = 2
    monkeypatch.setattr(loop, "_lane_focuses", lambda: ["the small cases", "the large cases"])
    _stub_measurement(monkeypatch, walls=[0.5])
    handed: list[GateScoring] = []
    hold_limits: list[int] = []

    class LaneAgent:
        """Stand in for the Implementer session the factory builds for one lane."""

        async def __call__(self, kernel_path, history, session_sink, gate_scoring):
            handed.append(gate_scoring)
            lane = 2 if "Yours is lane 2" in history else 1
            Path(kernel_path).write_text(f"def kernel():\n    return {10 * lane}\n")
            _verified(session_sink, score=1.0 + lane, case_ms=4.0 / (1.0 + lane))
            return f"lane {lane} changed the constant"

        def close(self):
            """Nothing is held between sessions."""

    build = _make_lane_agent_factory(
        make_agent=lambda _config, _setup, usage=None: LaneAgent(),
        config=Config(workspace=str(workspace), agent_backend="claude"),
        binding=LaneBinding(
            workspace_dir=str(workspace), driver=loop.ic.driver_script, source_files=[loop.ic.kernel_file]
        ),
        setup=ImplementerSetup(ImplementerTask(program_md="Optimize the kernel.")),
    )

    def factory(seat):
        hold_limits.append(seat.hold_limit_sec)
        return build(seat)

    asyncio.run(
        loop.run(replace(lane_agents(_no_change_agent, supervisor_fn=_unused_supervisor), agent_factory=factory))
    )

    assert len(handed) == 2
    for scoring in handed:
        assert scoring.baseline_case_times == {"case": 4.0}
        assert scoring.best_mean_case_speedup == 1.0
    # A command is stopped as hung only after twice what the engine allows one of its own benchmark runs.
    assert hold_limits == [2 * loop.ic.bench_timeout_sec] * 2
    lanes = ExperimentsLayout.of(workspace).round(1)
    filed = {name: json.loads((lanes / name / "lane.json").read_text()) for name in ("lane_1", "lane_2")}
    assert [(lane["verified"], lane["mean_case_speedup"]) for lane in filed.values()] == [(True, 2.0), (True, 3.0)]
    assert filed["lane_2"]["case_times"] == {"case": 4.0 / 3.0}
    # The merge session took nothing, so the better lane stands as the candidate.
    assert _decisions(workspace) == ["KEEP"]
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 20\n"


@pytest.mark.parametrize(
    ("lanes", "focuses", "expected"),
    [
        (1, ["a", "b"], [""]),
        (3, [], [""]),
        (3, ["only one"], [""]),
        (2, ["a", "b", "c"], ["a", "b"]),
        (3, ["a", "b"], ["a", "b"]),
    ],
    ids=["one lane allowed", "nothing divided", "one focus", "more focuses than lanes", "fewer focuses than lanes"],
)
def test_a_round_runs_one_lane_per_focus_up_to_the_campaigns_width(tmp_path, monkeypatch, lanes, focuses, expected):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic.lanes = lanes
    monkeypatch.setattr(loop, "_lane_focuses", lambda: list(focuses))

    assert [lane.focus for lane in loop._round_assignments()] == expected


# ── a round of several lanes ──────────────────────────────────────────────────


def _two_lane_loop(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic.lanes = 2
    monkeypatch.setattr(loop, "_lane_focuses", lambda: ["the small cases", "the large cases"])
    return loop, workspace


def test_two_changed_lanes_are_merged_by_a_session_that_starts_from_the_better_one(tmp_path, monkeypatch):
    loop, workspace = _two_lane_loop(tmp_path, monkeypatch)
    _stub_measurement(monkeypatch, walls=[0.4])
    merges: list[dict] = []
    lane_histories: dict[int, str] = {}

    async def agent(kernel_path, history, session_sink, gate_scoring):
        if MERGE_HEADING in history:
            merges.append({"history": history, "tree": Path(kernel_path).read_text(), "scoring": gate_scoring})
            Path(kernel_path).write_text("def kernel():\n    return 30\n")
            session_sink["plan"] = "dispatch between the two lanes"
            return "lane 2 is in; lane 1 is in for the small cases"
        lane = 2 if "Yours is lane 2" in history else 1
        lane_histories[lane] = history
        Path(kernel_path).write_text(f"def kernel():\n    return {10 * lane}\n")
        session_sink["plan"] = f"lane {lane}"
        _verified(session_sink, score=1.0 + 0.5 * lane, case_ms=1.0 / lane)
        return f"lane {lane} specialised its cases"

    asyncio.run(loop.run(lane_agents(agent, supervisor_fn=_unused_supervisor)))

    assert "## This lane's focus" in lane_histories[1] and "the small cases" in lane_histories[1]
    assert "- lane 2: the large cases" in lane_histories[1]
    (merge,) = merges
    # The canonical tree already held the round's best lane. The merge's stop is judged against the current best, as
    # the candidate will be; the lane's own score is one measurement the merge is told about, not its bar.
    assert merge["tree"] == "def kernel():\n    return 20\n"
    assert merge["scoring"].best_mean_case_speedup == 1.0
    assert merge["scoring"].starts_on_measured_candidate is True
    assert "This workspace already holds lane 2's result" in merge["history"]
    assert "score 2.0" in merge["history"]
    assert "leave the workspace exactly as you found it" in merge["history"]
    assert "judged against the campaign's current best (score 1.0)" in " ".join(merge["history"].split())
    brief = merge["history"]
    assert "- lane 1: stop check passed, score 1.5" in brief
    assert str(ExperimentsLayout.of(workspace.resolve()).round(1) / "lane_1" / "change.diff") in brief
    assert "lane 1 specialised its cases" in brief
    assert _decisions(workspace) == ["KEEP"]
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 30\n"
    assert _lane_copies(workspace) == []


def test_lanes_that_did_not_beat_the_base_are_merged_onto_the_base(tmp_path, monkeypatch):
    """A lane that is correct but no faster overall may still be faster on some cases, so it still reaches the merge."""
    loop, workspace = _two_lane_loop(tmp_path, monkeypatch)
    _stub_measurement(monkeypatch, walls=[0.8])
    merges: list[dict] = []

    async def agent(kernel_path, history, session_sink, gate_scoring):
        if MERGE_HEADING in history:
            merges.append({"history": history, "tree": Path(kernel_path).read_text(), "scoring": gate_scoring})
            Path(kernel_path).write_text("def kernel():\n    return 30\n")
            return "took each lane for the cases it won"
        lane = 2 if "Yours is lane 2" in history else 1
        Path(kernel_path).write_text(f"def kernel():\n    return {10 * lane}\n")
        if lane == 1:
            _verified(session_sink, score=0.9, case_ms=1.1)
        return f"lane {lane}"

    asyncio.run(loop.run(lane_agents(agent, supervisor_fn=_unused_supervisor)))

    (merge,) = merges
    assert merge["tree"] == "def kernel():\n    return 1\n"
    # The workspace holds no lane's result, so whatever the merge leaves is its own candidate.
    assert merge["scoring"].best_mean_case_speedup == 1.0
    assert merge["scoring"].starts_on_measured_candidate is False
    assert "This workspace holds the round's starting point" in merge["history"]
    assert "- lane 1: stop check passed, score 0.9" in merge["history"]
    assert "- lane 2: unverified." in merge["history"]
    assert _decisions(workspace) == ["KEEP"]


def test_a_merge_that_takes_nothing_leaves_the_lane_it_started_from_as_the_candidate(tmp_path, monkeypatch):
    """The other lane's change was measured as not worth taking: the round still has its best lane to measure."""
    loop, workspace = _two_lane_loop(tmp_path, monkeypatch)
    _stub_measurement(monkeypatch, walls=[0.5])
    merged: list[str] = []

    async def agent(kernel_path, history, session_sink):
        if MERGE_HEADING in history:
            merged.append(Path(kernel_path).read_text())
            session_sink["plan"] = "lane 1 measured slower on every case; nothing taken"
            return "lane 1 is not in: slower on every case"
        lane = 2 if "Yours is lane 2" in history else 1
        Path(kernel_path).write_text(f"def kernel():\n    return {10 * lane}\n")
        _verified(session_sink, score=1.0 + 0.5 * lane, case_ms=1.0 / lane)
        return f"lane {lane}"

    asyncio.run(loop.run(lane_agents(agent, supervisor_fn=_unused_supervisor)))

    assert merged == ["def kernel():\n    return 20\n"]
    assert _decisions(workspace) == ["KEEP"]
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 20\n"


def test_a_round_whose_only_changed_lane_needs_no_merge_is_measured_as_it_stands(tmp_path, monkeypatch):
    loop, workspace = _two_lane_loop(tmp_path, monkeypatch)
    _stub_measurement(monkeypatch, walls=[0.5])
    measured: list[dict] = []
    real_measure = loop.run_one_iteration

    async def recording(iteration, plan="", **kwargs):
        measured.append(kwargs)
        return await real_measure(iteration, plan=plan, **kwargs)

    monkeypatch.setattr(loop, "run_one_iteration", recording)

    async def agent(kernel_path, history, session_sink):
        assert MERGE_HEADING not in history
        if "Yours is lane 2" in history:
            Path(kernel_path).write_text("def kernel():\n    return 20\n")
            _verified(session_sink, score=2.0, case_ms=0.5)
            return "lane 2 changed the kernel"
        return "lane 1 found nothing to change"

    asyncio.run(loop.run(lane_agents(agent, supervisor_fn=_unused_supervisor)))

    assert _decisions(workspace) == ["KEEP"]
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 20\n"
    # Its own measurement was taken while another lane worked, so the canonical one is taken afresh.
    assert measured == [{}]


def test_a_lane_that_changes_the_driver_does_not_reach_the_canonical_tree(tmp_path, monkeypatch):
    loop, workspace = _two_lane_loop(tmp_path, monkeypatch)
    _stub_measurement(monkeypatch, walls=[0.5])

    async def agent(kernel_path, history, session_sink):
        assert MERGE_HEADING not in history
        if "Yours is lane 1" in history:
            Path(kernel_path).write_text("def kernel():\n    return 10\n")
            Path(kernel_path).with_name("driver.py").write_text("print('always passes')\n")
            return "changed what measures it"
        Path(kernel_path).write_text("def kernel():\n    return 20\n")
        _verified(session_sink, score=2.0, case_ms=0.5)
        return "changed the kernel"

    asyncio.run(loop.run(lane_agents(agent, supervisor_fn=_unused_supervisor)))

    # The lane that touched the driver is out of the round; the other lane's change is the candidate.
    assert _decisions(workspace) == ["KEEP"]
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 20\n"
    assert (workspace / "driver.py").read_text() == "pass\n"
    lane = json.loads((ExperimentsLayout.of(workspace).round(1) / "lane_1" / "lane.json").read_text())
    assert lane["error"] == "it changes the measurement surface: driver.py"


def test_a_new_file_the_campaign_may_commit_travels_with_the_lanes_change(tmp_path, monkeypatch):
    loop, workspace = _two_lane_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])
    _stub_measurement(monkeypatch, walls=[0.5])

    async def agent(kernel_path, history, session_sink):
        assert MERGE_HEADING not in history
        if "Yours is lane 1" in history:
            return "lane 1 found nothing to change"
        root = Path(kernel_path).parent
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        (root / "configs").mkdir()
        (root / "configs" / "small.json").write_text('{"tile": 64}\n')
        (root / "notes.txt").write_text("not on the allowlist\n")
        return "tuned and wrote the table it reads"

    asyncio.run(loop.run(lane_agents(agent, supervisor_fn=_unused_supervisor)))

    assert _decisions(workspace) == ["KEEP"]
    committed = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"], cwd=workspace, capture_output=True, text=True, check=True
    )
    assert sorted(committed.stdout.split()) == ["configs/small.json", "kernel.py"]
    assert not (workspace / "notes.txt").exists()


def test_a_lane_result_is_usable_only_when_it_changed_something_and_nothing_disqualified_it(tmp_path):
    lane = LaneResult(LaneAssignment("1"), tmp_path, diff="--- a/kernel.py\n", sink={"gate_passed": True})

    assert lane.changed and not lane.verified
    assert replace(lane, sink={"gate_passed": True, "mean_case_speedup": 1.4}).verified
    assert not replace(lane, error="session lost").changed
    assert not replace(lane, sink={"integrity_violation": True}).changed
    assert not replace(lane, diff="  \n").changed
