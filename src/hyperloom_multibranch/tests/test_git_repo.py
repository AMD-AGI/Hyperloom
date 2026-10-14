# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio

import pytest

from hyperloom.contracts.recipe import (
    ComponentPatches,
    ConfigDelta,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeDelta,
    apply,
)
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import BranchMoved, ExperimentRepo
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo

BASELINE = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128", envs={"VLLM_USE_V1": "1"}),
    components=(
        ComponentPatches(
            "vllm", "https://github.com/vllm-project/vllm", "1" * 40, (PatchRef("patch/vllm/0001.patch", "a" * 64),)
        ),
    ),
)
CHUNKED = apply(BASELINE, RecipeDelta(config=ConfigDelta(server_args="--enable-chunked-prefill")))


@pytest.fixture
async def repo(tmp_path) -> ExperimentRepo:
    return await GitExperimentRepo.create(tmp_path / "experiments.git", BASELINE)


async def test_the_root_commit_holds_the_baseline_on_the_integration_branch(repo):
    root = await repo.head(INTEGRATION_BRANCH)
    assert len(root) == 40
    assert await repo.read_recipe(root) == BASELINE


async def test_a_branch_moves_only_on_its_own_commits(repo):
    root = await repo.head(INTEGRATION_BRANCH)
    await repo.create_branch("s1", root)

    tip = await repo.commit("s1", root, CHUNKED, "enable chunked prefill")

    assert await repo.head("s1") == tip
    assert await repo.head(INTEGRATION_BRANCH) == root
    assert await repo.read_recipe(tip) == CHUNKED


async def test_a_commit_judged_against_a_parent_the_branch_left_is_refused(repo):
    root = await repo.head(INTEGRATION_BRANCH)
    await repo.create_branch("s1", root)
    tip = await repo.commit("s1", root, CHUNKED, "first")

    with pytest.raises(BranchMoved):
        await repo.commit("s1", root, BASELINE, "judged against the old head")
    assert await repo.head("s1") == tip


async def test_a_commit_is_taken_back_only_from_the_head_it_made(repo):
    root = await repo.head(INTEGRATION_BRANCH)
    await repo.create_branch("s1", root)
    tip = await repo.commit("s1", root, CHUNKED, "first")

    await repo.rewind("s1", tip, root)

    assert await repo.head("s1") == root
    with pytest.raises(BranchMoved):
        await repo.rewind("s1", tip, root)


async def test_two_writers_racing_on_one_branch_leave_one_commit(repo):
    root = await repo.head(INTEGRATION_BRANCH)
    await repo.create_branch("s1", root)

    results = await asyncio.gather(
        repo.commit("s1", root, CHUNKED, "one"), repo.commit("s1", root, BASELINE, "two"), return_exceptions=True
    )

    landed = [result for result in results if isinstance(result, str)]
    assert len(landed) == 1
    assert any(isinstance(result, BranchMoved) for result in results)
    assert await repo.head("s1") == landed[0]


async def test_the_repo_refuses_unknown_commits_and_branches(repo):
    root = await repo.head(INTEGRATION_BRANCH)
    with pytest.raises(KeyError):
        await repo.create_branch("s1", "f" * 40)
    with pytest.raises(KeyError):
        await repo.read_recipe("f" * 40)
    with pytest.raises(KeyError):
        await repo.head("s9")
    await repo.create_branch("s1", root)
    with pytest.raises(ValueError, match="already exists"):
        await repo.create_branch("s1", root)


async def test_an_inherited_git_dir_never_redirects_the_session_repository(tmp_path, monkeypatch):
    decoy = await GitExperimentRepo.create(tmp_path / "decoy.git", BASELINE)
    decoy_root = await decoy.head(INTEGRATION_BRANCH)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "decoy.git"))

    repo = await GitExperimentRepo.create(tmp_path / "experiments.git", BASELINE)
    await repo.create_branch("s1", await repo.head(INTEGRATION_BRANCH))

    assert await decoy.head(INTEGRATION_BRANCH) == decoy_root
    with pytest.raises(KeyError):
        await decoy.head("s1")


async def test_a_session_never_reuses_a_repository(tmp_path):
    await GitExperimentRepo.create(tmp_path / "experiments.git", BASELINE)
    with pytest.raises(FileExistsError):
        await GitExperimentRepo.create(tmp_path / "experiments.git", BASELINE)
