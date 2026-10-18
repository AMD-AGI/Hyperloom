# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from hyperloom.contracts.recipe import ConfigDelta, LaunchConfig, Recipe, RecipeDelta, apply
from hyperloom_multibranch.contracts.lifecycle import SpecialistLifecycle
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import BranchMoved, ExperimentRepo
from hyperloom_multibranch.tests.fakes import InMemoryRepo, RecordingLifecycle

BASELINE = Recipe("rocm/vllm@sha256:" + "c" * 64, LaunchConfig(server_args="--max-num-seqs 128"))
CHUNKED = apply(BASELINE, RecipeDelta(config=ConfigDelta(server_args="--enable-chunked-prefill")))


def _repo() -> ExperimentRepo:
    return InMemoryRepo(BASELINE)


async def test_a_branch_starts_at_its_commit_and_moves_only_on_its_own_commits():
    repo = _repo()
    root = await repo.head(INTEGRATION_BRANCH)
    await repo.create_branch("b1", root)
    assert await repo.head("b1") == root

    tip = await repo.commit("b1", root, CHUNKED, "enable chunked prefill")

    assert await repo.head("b1") == tip
    assert await repo.head(INTEGRATION_BRANCH) == root
    assert await repo.read_recipe(tip) == CHUNKED
    assert await repo.read_recipe(root) == BASELINE


async def test_a_commit_judged_against_a_parent_the_branch_left_is_refused():
    repo = _repo()
    root = await repo.head(INTEGRATION_BRANCH)
    await repo.create_branch("b1", root)
    await repo.commit("b1", root, CHUNKED, "first")

    with pytest.raises(BranchMoved):
        await repo.commit("b1", root, BASELINE, "judged against the old head")


async def test_the_repo_refuses_unknown_commits_and_branches():
    repo = _repo()
    with pytest.raises(KeyError):
        await repo.create_branch("b1", "no-such-commit")
    await repo.create_branch("b1", await repo.head(INTEGRATION_BRANCH))
    with pytest.raises(ValueError, match="already exists"):
        await repo.create_branch("b1", await repo.head(INTEGRATION_BRANCH))
    with pytest.raises(KeyError):
        await repo.commit("b2", "c0", BASELINE, "no such branch")


async def test_the_lifecycle_tracks_running_branches_and_their_k():
    recording = RecordingLifecycle()
    lifecycle: SpecialistLifecycle = recording
    await lifecycle.start("b1", "c0", brief="leads: fp8 kv cache")
    assert (recording.running, recording.briefs, recording.repairs, recording.k) == (
        {"b1": "c0"},
        {"b1": "leads: fp8 kv cache"},
        {"b1": None},
        {"b1": 1},
    )
    await lifecycle.set_k("b1", 4)
    assert recording.k == {"b1": 4}

    await lifecycle.stop("b1")

    assert recording.running == {}
    with pytest.raises(KeyError):
        await lifecycle.set_k("b1", 1)


@pytest.mark.parametrize("k", [0, -1])
async def test_a_branch_keeps_room_for_a_test(k):
    lifecycle = RecordingLifecycle()
    await lifecycle.start("b1", "c0")
    with pytest.raises(ValueError, match="stop the branch"):
        await lifecycle.set_k("b1", k)
