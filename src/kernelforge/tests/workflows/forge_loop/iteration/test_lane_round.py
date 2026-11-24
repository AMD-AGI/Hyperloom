# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One round: its lanes work in copies of the workspace, and what they leave is merged into the one candidate measured."""

from __future__ import annotations

import asyncio
import json
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
        workspace_dir=str(workspace), parent_dir=str(tmp_path), lanes=[LaneAssignment("1"), LaneAssignment("2", "b")]
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
    lanes = LaneRound(workspace_dir=str(workspace), parent_dir=str(tmp_path), lanes=[LaneAssignment("1")])
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


# ── a round of one lane ───────────────────────────────────────────────────────


def test_a_round_of_one_lane_works_in_a_copy_and_its_change_is_measured_on_the_canonical_tree(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    _stub_measurement(monkeypatch, walls=[0.5])
    worked_in: list[Path] = []
    gains: list[bool] = []

    async def agent(kernel_path, history, session_sink):
        worked_in.append(Path(kernel_path))
        assert MERGE_HEADING not in history
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        session_sink["plan"] = "return the faster constant"
        return "changed the constant"

    def factory(seat, *, gain_required):
        gains.append(gain_required)
        return agent

    asyncio.run(loop.run(replace(lane_agents(agent, supervisor_fn=_unused_supervisor), agent_factory=factory)))

    assert len(worked_in) == 1
    assert workspace.resolve() not in worked_in[0].resolve().parents
    assert any(part.startswith(LANE_ROOT_PREFIX) for part in worked_in[0].parts)
    # Alone in its round, the lane's stop has to beat the current best: there is no merge to use anything less.
    assert gains == [True]
    assert _decisions(workspace) == ["KEEP"]
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 2\n"
    assert _lane_copies(workspace) == []
    lane = ExperimentsLayout.of(workspace).round(1) / "lane_1"
    assert "return 2" in (lane / "change.diff").read_text()
    assert json.loads((lane / "lane.json").read_text())["plan"] == "return the faster constant"


def test_the_campaigns_lane_factory_carries_the_scoring_state_in_and_the_lanes_result_out(tmp_path, monkeypatch):
    """The factory a campaign is started with stands between the loop and the session it builds.

    What the loop hands a session has to cross it in both directions: the scoring state the lane's stop check
    measures against, and the sink the round reads the lane's verdict and score from.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times={"case": 4.0})
    _stub_measurement(monkeypatch, walls=[0.5])
    handed: list[GateScoring] = []

    class LaneAgent:
        """Stand in for the Implementer session the factory builds for one lane."""

        async def __call__(self, kernel_path, history, session_sink, gate_scoring):
            handed.append(gate_scoring)
            Path(kernel_path).write_text("def kernel():\n    return 2\n")
            _verified(session_sink, score=2.0, case_ms=2.0)
            return "changed the constant"

        def close(self):
            """Nothing is held between sessions."""

    factory = _make_lane_agent_factory(
        make_agent=lambda _config, _setup, usage=None: LaneAgent(),
        config=Config(workspace=str(workspace), agent_backend="claude"),
        binding=LaneBinding(
            workspace_dir=str(workspace), driver=loop.ic.driver_script, source_files=[loop.ic.kernel_file]
        ),
        setup=ImplementerSetup(ImplementerTask(program_md="Optimize the kernel.")),
    )

    asyncio.run(
        loop.run(replace(lane_agents(_no_change_agent, supervisor_fn=_unused_supervisor), agent_factory=factory))
    )

    [scoring] = handed
    assert scoring.baseline_case_times == {"case": 4.0}
    assert scoring.best_mean_case_speedup == 1.0
    lane = json.loads((ExperimentsLayout.of(workspace).round(1) / "lane_1" / "lane.json").read_text())
    assert (lane["verified"], lane["mean_case_speedup"], lane["case_times"]) == (True, 2.0, {"case": 2.0})
    assert _decisions(workspace) == ["KEEP"]


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
    gains: list[bool] = []

    async def agent(kernel_path, history, session_sink, gate_scoring):
        if MERGE_HEADING in history:
            merges.append(
                {"history": history, "tree": Path(kernel_path).read_text(), "bar": gate_scoring.best_mean_case_speedup}
            )
            Path(kernel_path).write_text("def kernel():\n    return 30\n")
            session_sink["plan"] = "dispatch between the two lanes"
            return "lane 2 is in; lane 1 is in for the small cases"
        lane = 2 if "Yours is lane 2" in history else 1
        lane_histories[lane] = history
        Path(kernel_path).write_text(f"def kernel():\n    return {10 * lane}\n")
        session_sink["plan"] = f"lane {lane}"
        _verified(session_sink, score=1.0 + 0.5 * lane, case_ms=1.0 / lane)
        return f"lane {lane} specialised its cases"

    def factory(seat, *, gain_required):
        gains.append(gain_required)
        return agent

    asyncio.run(loop.run(replace(lane_agents(agent, supervisor_fn=_unused_supervisor), agent_factory=factory)))

    # Beside another lane, a correct candidate may stop: what it gained is the merge's to use.
    assert gains == [False, False]
    assert "## This lane's focus" in lane_histories[1] and "the small cases" in lane_histories[1]
    assert "- lane 2: the large cases" in lane_histories[1]
    (merge,) = merges
    # The canonical tree already held the round's best lane, and that lane's score is what the merge has to beat.
    assert merge["tree"] == "def kernel():\n    return 20\n"
    assert merge["bar"] == 2.0
    assert "This workspace already holds lane 2's result" in merge["history"]
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
            merges.append(
                {"history": history, "tree": Path(kernel_path).read_text(), "bar": gate_scoring.best_mean_case_speedup}
            )
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
    assert merge["bar"] == 1.0
    assert "This workspace holds the round's starting point" in merge["history"]
    assert "- lane 1: stop check passed, score 0.9" in merge["history"]
    assert "- lane 2: unverified." in merge["history"]
    assert _decisions(workspace) == ["KEEP"]


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
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    async def agent(kernel_path, _history, session_sink):
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        Path(kernel_path).with_name("driver.py").write_text("print('always passes')\n")
        return "changed what measures it"

    asyncio.run(loop.run(lane_agents(agent, supervisor_fn=_unused_supervisor)))

    assert _decisions(workspace) == ["AGENT_ERROR"]
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 1\n"
    assert (workspace / "driver.py").read_text() == "pass\n"
    lane = json.loads((ExperimentsLayout.of(workspace).round(1) / "lane_1" / "lane.json").read_text())
    assert lane["error"] == "it changes the measurement surface: driver.py"


def test_a_new_file_the_campaign_may_commit_travels_with_the_lanes_change(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])
    _stub_measurement(monkeypatch, walls=[0.5])

    async def agent(kernel_path, _history, session_sink):
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
