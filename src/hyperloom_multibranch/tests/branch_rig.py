# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session's source base over one installed ``vllm`` package, its experiment repository with ``recipe`` as the root,
and one branch opened at that root in the blackboard, for the tests of a branch's specialist."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hyperloom.contracts.recipe import Recipe
from hyperloom.measure.compile import LocalBlobStore, PackageBase
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import LineageEvent
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH, LineageKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.specialist.worktree import SourceBase

#: What the installed package's one file holds at the root.
ROOT_CONFIG = b"max_num_seqs = 128\n"


@dataclass(frozen=True)
class BranchRig:
    base: SourceBase
    repo: GitExperimentRepo
    root: str
    blackboard: LocalBlackboard


async def branch_rig(tmp_path: Path, recipe: Recipe, branch: str) -> BranchRig:
    """``branch`` opened at the root ``recipe`` commits, and the branch's runtime handle on the blackboard."""

    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "config.py").write_bytes(ROOT_CONFIG)
    installed = PackageBase("vllm", package, None, None)
    base = SourceBase(tmp_path / "source", locator=lambda _layout: installed)
    base.build(["vllm"])
    repo = GitExperimentRepo(
        tmp_path / "experiments.git", LocalBlobStore(tmp_path / "blobs"), lambda _layout: installed
    )
    root = await repo.init_root(recipe)
    await repo.create_branch(branch, root)
    scheduler = LocalBlackboard(tmp_path / "blackboard", Writer(Role.SCHEDULER, "scheduler"))
    await scheduler.record_lineage(
        LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=root, recipe_digest=recipe.digest())
    )
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, branch, from_commit=root))
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.SPECIALIST_RUNTIME, branch, branch))
    return BranchRig(base, repo, root, blackboard)
