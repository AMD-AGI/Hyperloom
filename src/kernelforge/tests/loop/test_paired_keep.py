# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Paired KEEP: the scoring rule, the pairs, the incumbent's tree, and the A-vs-A noise floor."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.evaluation.benchmark.paired import measure_pairs
from kernelforge.evaluation.benchmark.runner import BenchRun
from kernelforge.evaluation.scoring import (
    noise_floor_from_null,
    paired_inconclusive,
    paired_keep_floor,
    paired_lower_bound,
    passes_paired_keep,
)
from kernelforge.loop.runner import NOISE_FLOOR_FILE, IterationLoop
from kernelforge.loop.worktrees import DetachedWorktree, incumbent_tree_path, tree_copy_refusal

DRIVER = """import pathlib, sys
here = pathlib.Path(__file__).resolve().parent
if (here / "crash").exists():
    sys.exit(1)
speed = float((here / "speed.txt").read_text())
cases = (here / "cases.txt").read_text().split() if (here / "cases.txt").exists() else ["m=8", "m=16"]
if "--bench-mode" in sys.argv:
    for case in cases:
        print(f"case_ms: {case} {speed:.6f}")
    print(f"median_ms: {speed * len(cases):.6f}")
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def _workspace(tmp_path: Path, *, incumbent_speed: float, candidate_speed: float) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "driver.py").write_text(DRIVER)
    (workspace / "speed.txt").write_text(str(incumbent_speed))
    _git(workspace, "init", "-q")
    _git(workspace, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(workspace, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "incumbent")
    (workspace / "speed.txt").write_text(str(candidate_speed))
    return workspace


def test_the_paired_rule() -> None:
    assert paired_lower_bound([1.0]) is None
    assert paired_lower_bound([1.05, 1.05, 1.05]) == pytest.approx(1.05)
    assert passes_paired_keep([1.05, 1.05, 1.05], floor=0.01)
    assert not passes_paired_keep([1.005, 1.005, 1.005], floor=0.01)
    assert paired_inconclusive([1.00, 1.04, 1.08], floor=0.01)
    assert not paired_inconclusive([1.05, 1.05, 1.05], floor=0.01)
    assert paired_keep_floor(None) == pytest.approx(0.001)
    assert paired_keep_floor(0.012) == pytest.approx(0.012)
    assert noise_floor_from_null([1.0, 1.0, 1.0]) == 0.0
    assert noise_floor_from_null([0.99, 1.01, 1.0, 1.0, 1.0]) > 0.0


def _arms(tmp_path: Path) -> tuple[BenchRun, BenchRun]:
    arms = []
    for name, speed in (("cand", 1.0), ("inc", 1.2)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "driver.py").write_text(DRIVER)
        (folder / "speed.txt").write_text(str(speed))
        arms.append(BenchRun(str(folder / "driver.py"), driver_args=[], cwd=str(folder)))
    return arms[0], arms[1]


async def test_pairs_run_in_a_seeded_order_and_score_incumbent_over_candidate(tmp_path: Path) -> None:
    candidate, incumbent = _arms(tmp_path)
    first = await measure_pairs(candidate, incumbent, pairs=4, rng=random.Random("seed"), timeout_sec=60)
    again = await measure_pairs(candidate, incumbent, pairs=4, rng=random.Random("seed"), timeout_sec=60)
    assert [pair.order for pair in first.pairs] == [pair.order for pair in again.pairs]
    assert first.scores == pytest.approx([1.2] * 4)
    assert first.to_dict()["mean"] == pytest.approx(1.2)


async def test_arms_that_time_different_cases_are_not_compared(tmp_path: Path) -> None:
    candidate, incumbent = _arms(tmp_path)
    (Path(incumbent.cwd) / "cases.txt").write_text("m=8 m=16 m=32")
    measured = await measure_pairs(candidate, incumbent, pairs=3, rng=random.Random(1), timeout_sec=60)
    assert measured.pairs == () and "different cases" in measured.failure


async def test_a_failing_arm_is_named(tmp_path: Path) -> None:
    candidate, incumbent = _arms(tmp_path)
    (Path(incumbent.cwd) / "crash").write_text("")
    measured = await measure_pairs(candidate, incumbent, pairs=3, rng=random.Random(1), timeout_sec=60)
    assert measured.failure.startswith("incumbent arm: BENCH CRASHED")


def test_the_incumbent_tree_follows_head_and_undoes_edits(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, incumbent_speed=1.2, candidate_speed=1.0)
    first = _git(workspace, "rev-parse", "HEAD").strip()
    tree = DetachedWorktree(workspace, incumbent_tree_path(workspace))
    assert tree.path.parent == workspace.parent
    assert tree.ensure_at(first) == "created"
    assert (tree.path / "speed.txt").read_text() == "1.2"
    assert tree.ensure_at(first) == ""
    (tree.path / "speed.txt").write_text("9")
    assert tree.ensure_at(first) == "found changed and restored"
    assert (tree.path / "speed.txt").read_text() == "1.2"
    _git(workspace, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "kept")
    assert tree.ensure_at(_git(workspace, "rev-parse", "HEAD").strip()) == "moved to the new incumbent"
    assert (tree.path / "speed.txt").read_text() == "1.0"
    tree.remove()
    assert not tree.path.exists()


def test_a_kernel_only_the_workspace_can_run_is_refused(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, incumbent_speed=1.0, candidate_speed=1.0)
    driver = workspace / "driver.py"
    assert tree_copy_refusal(workspace=workspace, driver=driver, source_files=["kernel.py"]) == ""
    assert "C/C++" in tree_copy_refusal(workspace=workspace, driver=driver, source_files=["src/gemm.hip"])
    outside = tmp_path / "driver.py"
    outside.write_text(DRIVER)
    assert "outside the workspace" in tree_copy_refusal(workspace=workspace, driver=outside, source_files=[])
    (workspace / "untracked.py").write_text(DRIVER)
    assert "not committed" in tree_copy_refusal(workspace=workspace, driver=workspace / "untracked.py", source_files=[])


def _loop(workspace: Path) -> SimpleNamespace:
    root = workspace / "forge_experiments"
    root.mkdir(exist_ok=True)
    events: list[dict] = []
    loop = SimpleNamespace(
        ic=SimpleNamespace(
            workspace_dir=str(workspace),
            driver_script=str(workspace / "driver.py"),
            bench_repeat=1,
            bench_timeout_sec=60,
            paired_keep=True,
            null_control_pairs=3,
            null_control_every=0,
        ),
        run_state=SimpleNamespace(campaign_id="c"),
        state_store=SimpleNamespace(root=root, append_event=events.append),
        events=events,
        _git=lambda *args: _git(workspace, *args),
        _scored_cases=lambda: None,
        _observe_measurement=lambda started: None,
    )
    loop._paired_arms = lambda iteration: IterationLoop._paired_arms(loop, iteration)
    loop._noise_floor_record = lambda: IterationLoop._noise_floor_record(loop)
    loop._noise_floor = lambda: IterationLoop._noise_floor(loop)
    return loop


@pytest.mark.parametrize(("candidate_speed", "kept"), [(1.0, True), (1.2, False)])
async def test_a_candidate_is_kept_only_when_it_beats_the_incumbent_beside_it(
    tmp_path: Path, candidate_speed: float, kept: bool
) -> None:
    loop = _loop(_workspace(tmp_path, incumbent_speed=1.2, candidate_speed=candidate_speed))
    bench, paired = await IterationLoop._measure_paired(loop, 3)
    assert bench["success"] and len(bench["measurements"]) == 3
    assert bench["paired"]["scores"] == pytest.approx([1.2 / candidate_speed] * 3)
    assert IterationLoop._paired_keep_bar(loop, paired)["passes"] is kept
    assert loop.events[0]["type"] == "incumbent_tree_reset"


async def test_the_null_control_measures_the_incumbent_beside_itself_once(tmp_path: Path) -> None:
    loop = _loop(_workspace(tmp_path, incumbent_speed=1.2, candidate_speed=1.2))
    await IterationLoop._refresh_noise_floor_if_due(loop, 1)
    record = json.loads((loop.state_store.root / NOISE_FLOOR_FILE).read_text())
    assert record["value"] == 0.0 and record["scores"] == pytest.approx([1.0] * 3)
    assert loop.events[-1]["type"] == "null_control"
    count = len(loop.events)
    await IterationLoop._refresh_noise_floor_if_due(loop, 2)
    assert len(loop.events) == count


def test_the_driver_used_by_these_tests_runs(tmp_path: Path) -> None:
    candidate, _ = _arms(tmp_path)
    out = subprocess.run([sys.executable, candidate.driver_script, "--bench-mode"], capture_output=True, text=True)
    assert "case_ms: m=8 1.000000" in out.stdout
