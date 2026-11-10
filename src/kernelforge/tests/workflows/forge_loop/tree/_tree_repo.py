# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A real repository of kernel versions, and a tree-loop host that measures each version by its source.

Each constant ``NAME = value`` of ``kernel.py`` is the case ``name``, timed at ``value / 10`` ms against a 1 ms
baseline, so a version's case times follow from its source and a merge is measured as what was merged.
"""

from __future__ import annotations

import re
import statistics
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from kernelforge.workflows.forge_loop.campaign.experiments_layout import ExperimentsLayout
from kernelforge.workflows.forge_loop.iteration.fanout import LaneResult
from kernelforge.workflows.forge_loop.iteration.iteration_types import IterationResult
from kernelforge.workflows.forge_loop.iteration.merge_candidates import case_spreads
from kernelforge.workflows.forge_loop.policy.round_budget import FIRST_ROUND_MEASUREMENT_SEC
from kernelforge.workflows.forge_loop.tree.admission import TreeAdmissionMixin
from kernelforge.workflows.forge_loop.tree.merge import TreeMergeMixin
from kernelforge.workflows.forge_loop.tree.model import NodeSource, TreeNode
from kernelforge.workflows.forge_loop.tree.store import SearchTreeStore
from kernelforge.workflows.forge_loop.tree.worktrees import StartWorktrees

_CONSTANT = re.compile(r"^(\w+) = (\d+)$", re.MULTILINE)
# Three runs a little apart, so each case has the spread its noise is judged by.
_RUN_FACTORS = (1.0, 1.001, 0.999)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _source(values: dict[str, int]) -> str:
    # Far enough apart that a change to one constant is a hunk of its own, as in a real kernel.
    return "".join(f"{key} = {value}\n" + "#\n" * 8 for key, value in values.items())


def _repo(tmp_path: Path, values: dict[str, int]) -> tuple[Path, str]:
    """A repository whose one commit is ``values``, and that commit."""
    repo = tmp_path / "workspace"
    repo.mkdir()
    (repo / "kernel.py").write_text(_source(values))
    _git(repo, "init", "-q", "-b", "opt")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "root")
    return repo, _git(repo, "rev-parse", "HEAD")


def _commit(repo: Path, base: str, values: dict[str, int]) -> str:
    _git(repo, "checkout", "-q", "--detach", base)
    (repo / "kernel.py").write_text(_source(values))
    _git(repo, "commit", "-qam", str(values))
    return _git(repo, "rev-parse", "HEAD")


def _measurement(iteration: int, values: dict[str, int], *, correct: bool = True) -> IterationResult:
    times = {name.lower(): value / 10.0 for name, value in values.items()}
    runs = [
        {"success": True, "case_times": {case: ms * factor for case, ms in times.items()}} for factor in _RUN_FACTORS
    ]
    scores = [statistics.fmean(1.0 / ms for ms in run["case_times"].values()) for run in runs]
    detail = {"success": True, "measurement_mean_case_speedups": scores, "case_times": times, "measurements": runs}
    summary = "PASS" if correct else "FAIL: the output differs from the reference"
    return IterationResult(
        iteration, 0.0, correct, summary, mean_case_speedup=statistics.fmean(scores), bench_detail=detail
    )


def _node(commit: str, parent: str, direction: str, values: dict[str, int]) -> TreeNode:
    """A node holding ``values``, scored as the host would measure it."""
    detail = _measurement(1, values).bench_detail
    return TreeNode(
        commit=commit,
        parent=parent,
        direction=direction,
        slot_type=None,
        iteration=1,
        slot=1,
        samples=tuple(detail["measurement_mean_case_speedups"]),
        wall_ms=None,
        case_times=dict(detail["case_times"]),
        case_spreads=case_spreads(detail["measurements"]),
        snr_db=None,
        plan=f"plan of {commit[:4]}",
        source=NodeSource.ITERATION,
    )


def _values(text: str) -> dict[str, int]:
    return {name: int(value) for name, value in _CONSTANT.findall(text)}


class _HazardLog:
    """The device-hazard log, without the process inspection behind a real finding."""

    def __init__(self) -> None:
        self.live = None

    def record(self, *, iteration: int, detail: str, pids) -> None:
        self.live = SimpleNamespace(iteration=iteration, detail=detail, pids=set(pids), describe=lambda: detail)


class _Host(TreeMergeMixin, TreeAdmissionMixin):
    """What admission and merging need from the tree loop: measurements by source, and sessions that are scripted."""

    def __init__(self, repo: Path, *, remaining_sec: float = 12 * 3600.0) -> None:
        self.ic = SimpleNamespace(workspace_dir=str(repo))
        self._start_worktrees = StartWorktrees(repo)
        self._experiments = ExperimentsLayout(repo.parent / "experiments")
        self._tree_store = SearchTreeStore(repo.parent / "search_tree.json")
        self._lane_rejection = lambda _diff: ""
        self.device_hazard = _HazardLog()
        self.remaining_sec = remaining_sec
        self.measured: list[str] = []
        # Kernel sources that measure incorrect.
        self.incorrect: set[str] = set()
        # What each session was asked and on which profile, and what each one returns, in order.
        self.prompts: list[str] = []
        self.session_profiles: list[str] = []
        self.session_results: list[LaneResult] = []
        # The profiles whose startup check the next boundary is asked to run again.
        self.rechecked: list[str] = []

    def _time_remaining(self) -> float:
        return self.remaining_sec

    def _measurement_estimate_sec(self) -> float:
        return FIRST_ROUND_MEASUREMENT_SEC

    async def _measure_at(self, iteration: int, plan: str, worktree: Path) -> IterationResult:
        text = (worktree / "kernel.py").read_text()
        self.measured.append(text)
        return _measurement(iteration, _values(text), correct=text not in self.incorrect)

    async def _run_sessions(self, iteration: int, sessions) -> dict[str, LaneResult]:
        results = {}
        for session in sessions:
            self.prompts.append(session.prompt)
            self.session_profiles.append(session.agent.profile_id)
            lane = session.lane
            results[lane.lane_id] = replace(self.session_results.pop(0), lane_id=lane.lane_id, plan=lane.plan)
        return results

    def _check_profile_again(self, profile_id: str) -> None:
        self.rechecked.append(profile_id)
