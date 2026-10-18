# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hand-built search trees for the replay and prior tests."""

from __future__ import annotations

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.vocab import CommitOrigin, LineageKind
from hyperloom_multibranch.notebook.rsi.archive import MeasurementNoise, SearchTree, TreeBranch, TreeCommit
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions

HOUR = 3600.0
TASK = TaskConditions(
    labels={
        "gpu": "MI355X",
        "framework": "vllm",
        "architecture": "DeepseekV3ForCausalLM",
        "model": "deepseek-ai/DeepSeek-R1",
    },
    sizes={"tp": 8.0, "conc": 64.0, "isl": 1024.0, "osl": 1024.0},
    profile={"attention": 40.0, "gemm": 45.0, "other": 15.0},
)
LATE_BLOOMER = [(1800.0, 2.0), (3600.0, 2.3), (5400.0, 2.6), (7200.0, 12.0)]


def branch(name: str, gains: list[tuple[float, float]], **fields: object) -> TreeBranch:
    defaults: dict[str, object] = {
        "origin": LineageKind.OPEN,
        "parent_branch": INTEGRATION_BRANCH,
        "opened_s": 0.0,
        "closed_s": None,
        "fork_gain_pct": 0.0,
        "gpu_seconds": 4 * HOUR,
    }
    commits = tuple(TreeCommit(t_s, gain, CommitOrigin.COMMIT) for t_s, gain in gains)
    return TreeBranch(name=name, commits=commits, **{**defaults, **fields})


def tree(
    session_id: str, *branches: TreeBranch, task: TaskConditions = TASK, noise: MeasurementNoise | None = None
) -> SearchTree:
    integration = branch(INTEGRATION_BRANCH, [], origin=LineageKind.ROOT, parent_branch="", gpu_seconds=0.0)
    final = max((commit.gain_pct for item in branches for commit in item.commits), default=0.0)
    return SearchTree(session_id, "test", task, 4 * HOUR, final, (integration, *branches), noise or MeasurementNoise())
