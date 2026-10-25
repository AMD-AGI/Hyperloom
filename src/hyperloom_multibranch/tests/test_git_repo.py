# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import subprocess

import pytest

from hyperloom.contracts.recipe import (
    ComponentChange,
    ComponentPatches,
    ConfigDelta,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeDelta,
    apply,
    with_unexpressed,
)
from hyperloom.measure.compile import BlobError, LocalBlobStore
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import BranchMoved, ExperimentRepo
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo

VLLM = "https://github.com/vllm-project/vllm"
BASELINE = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128", envs={"VLLM_USE_V1": "1"}),
    components=(ComponentPatches("vllm", VLLM, "1" * 40, (PatchRef("patch/vllm/0001.patch", "a" * 64),)),),
)
CHUNKING = RecipeDelta(config=ConfigDelta(server_args="--enable-chunked-prefill"))
CHUNKED = apply(BASELINE, CHUNKING)
PATCH = "--- a/vllm/config.py\n+++ b/vllm/config.py\n@@ -1 +1 @@\n-max_num_seqs = 128\n+max_num_seqs = 256\n"


async def _rooted(path, blobs) -> GitExperimentRepo:
    repo = GitExperimentRepo(path, blobs)
    await repo.init_root(BASELINE)
    return repo


def _parents(path, commit: str) -> list[str]:
    argv = ["git", "--git-dir", str(path), "rev-list", "--parents", "-n", "1", commit]
    return subprocess.run(argv, capture_output=True, text=True, check=True).stdout.split()[1:]


@pytest.fixture
def blobs(tmp_path) -> LocalBlobStore:
    return LocalBlobStore(tmp_path / "blobs")


@pytest.fixture
async def repo(tmp_path, blobs) -> ExperimentRepo:
    return await _rooted(tmp_path / "experiments.git", blobs)


async def test_the_root_commit_holds_the_baseline_on_the_integration_branch(tmp_path, blobs):
    repo = GitExperimentRepo(tmp_path / "experiments.git", blobs)

    root = await repo.init_root(BASELINE)

    assert await repo.head(INTEGRATION_BRANCH) == root
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


async def test_only_a_full_commit_id_names_a_commit(repo):
    root = await repo.head(INTEGRATION_BRANCH)
    for name in (INTEGRATION_BRANCH, root[:7], f"{root}~0"):
        with pytest.raises(KeyError):
            await repo.read_recipe(name)
        with pytest.raises(KeyError):
            await repo.create_branch("s1", name)


async def test_an_inherited_git_dir_never_redirects_the_session_repository(tmp_path, blobs, monkeypatch):
    decoy = await _rooted(tmp_path / "decoy.git", blobs)
    decoy_root = await decoy.head(INTEGRATION_BRANCH)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "decoy.git"))

    repo = await _rooted(tmp_path / "experiments.git", blobs)
    await repo.create_branch("s1", await repo.head(INTEGRATION_BRANCH))

    assert await decoy.head(INTEGRATION_BRANCH) == decoy_root
    with pytest.raises(KeyError):
        await decoy.head("s1")


async def test_a_session_never_reuses_a_repository(tmp_path, blobs):
    await _rooted(tmp_path / "experiments.git", blobs)
    with pytest.raises(FileExistsError):
        await GitExperimentRepo(tmp_path / "experiments.git", blobs).init_root(BASELINE)


async def test_an_integration_merge_lands_only_on_the_head_it_was_tested_on(tmp_path, repo):
    root = await repo.head(INTEGRATION_BRANCH)
    await repo.create_branch("s1", root)
    tip = await repo.commit("s1", root, CHUNKED, "enable chunked prefill")

    merged = await repo.commit_integration(root, CHUNKING, (tip,))

    assert await repo.head(INTEGRATION_BRANCH) == merged
    assert await repo.read_recipe(merged) == CHUNKED
    assert _parents(tmp_path / "experiments.git", merged) == [root, tip]
    with pytest.raises(BranchMoved):
        await repo.commit_integration(root, CHUNKING, (tip,))


async def test_an_integration_merge_keeps_what_either_side_leaves_unexpressed(repo):
    root = await repo.head(INTEGRATION_BRANCH)
    await repo.create_branch("s1", root)
    tip = await repo.commit("s1", root, with_unexpressed(CHUNKED, ("setup_commands",)), "on an incomplete root")

    merged = await repo.commit_integration(root, CHUNKING, (tip,))

    assert (await repo.read_recipe(merged)).unexpressed == ("setup_commands",)


async def test_a_stored_patch_reads_back_from_any_commit_and_an_unknown_one_is_a_key_error(repo, blobs):
    root = await repo.head(INTEGRATION_BRANCH)
    await repo.create_branch("s1", root)

    stored = await repo.store_patch("vllm", PATCH)
    tip = await repo.commit("s1", root, apply(BASELINE, _adding(stored)), "raise max_num_seqs in the config")

    assert stored.path.startswith("patch/vllm/")
    assert blobs.get(stored.sha256) == PATCH.encode()
    assert await repo.patch_text(tip, stored.path) == PATCH
    assert await repo.patch_text(root, stored.path) == PATCH
    for unknown in ("patch/vllm/0001-fix.patch", f"patch/vllm/{'0' * 64}.patch"):
        with pytest.raises(KeyError):
            await repo.patch_text(tip, unknown)


async def test_a_checkout_writes_every_file_the_recipe_names_at_its_path(tmp_path, repo, blobs):
    root = await repo.head(INTEGRATION_BRANCH)
    await repo.create_branch("s1", root)
    stored = await repo.store_patch("vllm", PATCH)
    kernel = KernelArtifact(KernelKind.GEMM, "moe", "kernel/gemm/moe.so", blobs.put(b"\x7fELF"))
    recipe = Recipe(BASELINE.base_image_digest, components=(ComponentPatches("vllm", VLLM, "1" * 40, (stored,)),))
    tip = await repo.commit("s1", root, apply(recipe, RecipeDelta(add_kernels=(kernel,))), "a tuned moe kernel")

    await repo.checkout(tip, tmp_path / "tree")

    assert (tmp_path / "tree" / stored.path).read_text() == PATCH
    assert (tmp_path / "tree" / "kernel/gemm/moe.so").read_bytes() == b"\x7fELF"
    with pytest.raises(BlobError):
        await repo.checkout(root, tmp_path / "root")


def _adding(patch: PatchRef) -> RecipeDelta:
    return RecipeDelta(components=(ComponentChange("vllm", add_patches=(patch,)),))
