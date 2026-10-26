# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Start worktrees, and lanes copied from different starts of one round."""

from __future__ import annotations

import subprocess
from pathlib import Path

from kernelforge.loop.fanout import LanePlan, campaign_device_lock_path, run_lanes
from kernelforge.loop.tree.worktrees import StartWorktrees, untracked_outside

DRIVER = "driver.py"


def _run(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _campaign(tmp_path: Path) -> tuple[Path, list[str]]:
    """A workspace on branch ``opt`` with two commits of ``kernel.py``; HEAD is the second."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _run(workspace, "init", "-q", "-b", "opt")
    _run(workspace, "config", "user.email", "t@t")
    _run(workspace, "config", "user.name", "t")
    (workspace / ".gitignore").write_text("build/\nforge_experiments/\n")
    (workspace / DRIVER).write_text("print('ok')\n")
    commits = []
    for value in (0, 1):
        (workspace / "kernel.py").write_text(f"VALUE = {value}\n")
        _run(workspace, "add", ".")
        _run(workspace, "commit", "-qm", f"v{value}")
        commits.append(_run(workspace, "rev-parse", "HEAD"))
    return workspace, commits


def test_a_start_worktree_is_a_clean_checkout_of_its_commit_beside_the_workspace(tmp_path):
    workspace, (first, _second) = _campaign(tmp_path)
    starts = StartWorktrees(workspace)

    path = starts.ensure(first)

    assert path.parent == workspace.parent / "workspace.tree-starts"
    assert (path / "kernel.py").read_text() == "VALUE = 0\n"
    assert _run(path, "rev-parse", "HEAD") == first
    assert _run(workspace, "rev-parse", "--abbrev-ref", "HEAD") == "opt"


def test_a_commit_made_in_a_start_worktree_belongs_to_the_campaign_repository(tmp_path):
    workspace, (first, _second) = _campaign(tmp_path)
    starts = StartWorktrees(workspace)
    path = starts.ensure(first)
    patch = "--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-VALUE = 0\n+VALUE = 7\n"

    assert starts.apply(path, patch)
    child = starts.commit(path, "iter-1-s1: seven")

    assert _run(workspace, "show", f"{child}:kernel.py") == "VALUE = 7"
    assert _run(workspace, "rev-parse", f"{child}^") == first
    assert (workspace / "kernel.py").read_text() == "VALUE = 1\n"


def test_a_patch_that_does_not_apply_leaves_the_worktree_untouched(tmp_path):
    workspace, (first, _second) = _campaign(tmp_path)
    starts = StartWorktrees(workspace)
    path = starts.ensure(first)

    applied = starts.apply(path, "--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-VALUE = 9\n+VALUE = 7\n")

    assert not applied
    assert _run(path, "status", "--porcelain") == ""


def test_reusing_a_worktree_discards_what_the_last_candidate_left_in_it(tmp_path):
    workspace, (first, _second) = _campaign(tmp_path)
    starts = StartWorktrees(workspace)
    path = starts.ensure(first)
    (path / "kernel.py").write_text("VALUE = 5\n")
    (path / "scratch.txt").write_text("left over\n")
    (path / "build").mkdir()
    (path / "build" / "kernel.so").write_text("binary\n")

    again = starts.ensure(first)

    assert again == path
    assert (path / "kernel.py").read_text() == "VALUE = 0\n"
    assert not (path / "scratch.txt").exists()
    assert (path / "build" / "kernel.so").exists()


def test_release_keeps_only_the_named_starts(tmp_path):
    workspace, (first, second) = _campaign(tmp_path)
    starts = StartWorktrees(workspace)
    kept, dropped = starts.ensure(first), starts.ensure(second)

    starts.release([first])

    assert kept.is_dir()
    assert not dropped.exists()
    assert str(dropped) not in _run(workspace, "worktree", "list")


def test_untracked_sources_a_worktree_would_miss_are_named(tmp_path):
    workspace, _commits = _campaign(tmp_path)
    (workspace / "forge_experiments").mkdir()
    (workspace / "forge_experiments" / "run_state.json").write_text("{}")
    (workspace / "data.bin").write_text("x")
    (workspace / "notes").mkdir()
    (workspace / "notes" / "a.txt").write_text("x")

    assert untracked_outside(workspace, ["notes"]) == ["data.bin"]


async def test_lanes_of_one_round_copy_their_own_starts_and_share_the_campaign_lock(tmp_path):
    workspace, (first, second) = _campaign(tmp_path)
    starts = StartWorktrees(workspace)
    seen: dict[str, str] = {}
    locks: set[str] = set()

    async def session(lane: LanePlan, lane_dir: Path, driver: Path) -> None:
        seen[lane.lane_id] = (lane_dir / "kernel.py").read_text()
        locks.add(next(line for line in driver.read_text().splitlines() if "forge-device-bench.lock" in line))
        (lane_dir / "kernel.py").write_text(f"VALUE = 1{lane.lane_id}\n")

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[
            LanePlan("1", "deepen", source_dir=str(starts.ensure(first))),
            LanePlan("2", "branch", source_dir=str(starts.ensure(second))),
        ],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER,
    )

    assert seen == {"1": "VALUE = 0\n", "2": "VALUE = 1\n"}
    assert "-VALUE = 0" in results[0].diff and "-VALUE = 1" in results[1].diff
    assert len(locks) == 1 and campaign_device_lock_path(workspace).name in locks.pop()
