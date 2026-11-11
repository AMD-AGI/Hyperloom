# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import os
import stat
import subprocess
from typing import NoReturn

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
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH
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


def _session(tmp_path, mode: int):
    session = tmp_path / "session"
    session.mkdir()
    session.chmod(mode)
    if mode & stat.S_ISGID and not session.stat().st_mode & stat.S_ISGID:
        pytest.skip("this filesystem does not keep the setgid bit for the test user")
    return session


def _mode(path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


@pytest.mark.parametrize("mask", [0o002, 0o022], ids=["umask-002", "umask-022"])
async def test_below_shared_storage_only_objects_and_refs_are_the_group_s(tmp_path, blobs, mask):
    """Whatever the umask: the repository ``3775`` (the group makes ``HEAD.lock`` and ``packed-refs.lock`` there but
    replaces nothing of its maker's), its config and ``HEAD`` ``0644``, ``info`` and ``hooks`` ``2755``, and the
    objects and refs, what the next account's commit writes, ``2775``."""
    path = _session(tmp_path, 0o2775) / "experiments.git"
    previous = os.umask(mask)
    try:
        repo = await _rooted(path, blobs)
        await repo.create_branch("b01", await repo.root())
    finally:
        os.umask(previous)

    assert _mode(path) == 0o3775
    assert {name: _mode(path / name) for name in ("config", "HEAD", "info", "hooks")} == {
        "config": 0o644,
        "HEAD": 0o644,
        "info": 0o2755,
        "hooks": 0o2755,
    }
    shared = [path / "objects", *(path / "objects").iterdir(), path / "refs", path / "refs" / "heads"]
    assert {_mode(entry) for entry in shared if entry.is_dir()} == {0o2775}
    assert await repo.read_recipe(await repo.root()) == BASELINE


@pytest.mark.parametrize("mask", [0o002, 0o077], ids=["umask-002", "umask-077"])
async def test_off_shared_storage_nothing_in_the_repository_is_the_group_s_to_write(tmp_path, blobs, mask):
    """Off shared storage the umask decides, but never gives the group write: the config is ``0644`` under ``002``."""
    path = _session(tmp_path, 0o755) / "experiments.git"
    previous = os.umask(mask)
    try:
        await _rooted(path, blobs)
    finally:
        os.umask(previous)

    assert _mode(path / "config") == 0o666 & ~(mask | 0o022)
    assert not _mode(path) & 0o022


async def test_a_shared_repository_whose_config_the_group_can_write_starts_no_git(tmp_path, blobs, monkeypatch):
    path = _session(tmp_path, 0o2775) / "experiments.git"
    repo = await _rooted(path, blobs)
    (path / "config").chmod(0o664)

    async def no_git(*argv: str, **kwargs: object) -> NoReturn:
        raise AssertionError(f"git started: {argv}")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_git)
    with pytest.raises(OSError, match="git's own state of a shared repository is refused"):
        await repo.head(INTEGRATION_BRANCH)


async def _no_git(*argv: str, **kwargs: object) -> NoReturn:
    raise AssertionError(f"git started: {argv}")


@pytest.mark.parametrize("swap", ["objects-0755", "a-link"])
async def test_below_shared_storage_a_repository_not_laid_out_shared_starts_no_git(tmp_path, blobs, monkeypatch, swap):
    """What another account of the group could put at the repository's path below shared storage: a repository whose
    objects are not the group's, or a link to one elsewhere. Its own objects and refs do not decide whether it is
    checked; where it lives does, and there it must be laid out shared."""
    session = _session(tmp_path, 0o2775)
    path = session / "experiments.git"
    repo = await _rooted(path, blobs)
    if swap == "objects-0755":
        (path / "objects").chmod(0o755)
    else:
        await _rooted(tmp_path / "elsewhere.git", blobs)
        path.rename(session / "moved.git")
        path.symlink_to(tmp_path / "elsewhere.git")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _no_git)
    with pytest.raises(OSError, match="below shared storage"):
        await repo.head(INTEGRATION_BRANCH)


async def test_below_shared_storage_what_git_makes_later_no_other_account_writes(tmp_path, blobs):
    """Under umask ``000`` too: the object directories a later commit makes are the group's, never every account's."""
    path = _session(tmp_path, 0o2775) / "experiments.git"
    repo = await _rooted(path, blobs)
    root = await repo.root()
    await repo.create_branch("s1", root)
    before = set((path / "objects").iterdir())
    previous = os.umask(0)
    try:
        await repo.commit("s1", root, CHUNKED, "made under umask 000")
    finally:
        os.umask(previous)

    made = [entry for entry in (path / "objects").iterdir() if entry not in before]
    assert made, "the commit made object directories of its own"
    assert {_mode(entry) for entry in made} == {0o2775}


async def test_every_driver_and_config_hook_the_config_names_is_blanked(tmp_path, blobs, monkeypatch):
    """As every other git the app runs: a filter, a diff driver and a hook the config defines run as nothing."""
    path = tmp_path / "experiments.git"
    repo = await _rooted(path, blobs)
    root = await repo.root()
    for key in ("filter.own.clean", "diff.own.textconv", "hook.own.command"):
        subprocess.run(["git", "--git-dir", str(path), "config", key, "touch ran"], check=True)
    spawned: list[tuple[str, ...]] = []
    spawn = asyncio.create_subprocess_exec

    async def recording(*argv: str, **kwargs: object) -> object:
        spawned.append(argv)
        return await spawn(*argv, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", recording)

    assert await repo.root() == root
    assert {"filter.own.clean=", "diff.own.textconv=", "hook.own.enabled=false"} <= set(spawned[-1])


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


async def test_a_branch_behind_the_integration_head_is_re_rooted_onto_it_and_its_old_head_kept(tmp_path, repo):
    root = await repo.head(INTEGRATION_BRANCH)
    await repo.create_branch("s1", root)
    await repo.create_branch("s0", root)
    own = await repo.commit("s1", root, apply(BASELINE, RecipeDelta(config=ConfigDelta(envs={"A": "1"}))), "own")
    merged = await repo.commit("s0", root, CHUNKED, "chunked")
    onto = await repo.commit_integration(root, CHUNKING, (merged,))

    await repo.reroot("s1", own, onto)

    assert await repo.head("s1") == onto
    refs = ["git", "--git-dir", str(tmp_path / "experiments.git"), "for-each-ref", "--format=%(objectname)"]
    assert subprocess.run([*refs, "refs/rerooted"], capture_output=True, text=True, check=True).stdout.split() == [own]
    with pytest.raises(BranchMoved):
        await repo.reroot("s1", own, onto)


async def test_a_branch_that_holds_the_integration_head_already_is_not_re_rooted(repo):
    root = await repo.head(INTEGRATION_BRANCH)
    await repo.create_branch("s1", root)
    tip = await repo.commit("s1", root, CHUNKED, "first")

    with pytest.raises(ValueError, match="holds"):
        await repo.reroot("s1", tip, root)
    assert await repo.head("s1") == tip


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


async def test_the_root_stays_the_baseline_commit_however_far_the_integration_branch_moves(tmp_path, blobs):
    repo = GitExperimentRepo(tmp_path / "experiments.git", blobs)
    root = await repo.init_root(BASELINE)
    await repo.create_branch("s1", root)
    tip = await repo.commit("s1", root, CHUNKED, "enable chunked prefill")
    merged = await repo.commit_integration(root, CHUNKING, (tip,))
    later = await repo.commit(INTEGRATION_BRANCH, merged, apply(CHUNKED, CHUNKING), "again")

    assert await repo.head(INTEGRATION_BRANCH) == later
    assert await repo.root() == root


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


class _EndingGit:
    """A git process that ends as soon as it is read, recording that it was."""

    returncode = 0
    read = False

    async def communicate(self, stdin: bytes | None) -> tuple[bytes, bytes]:
        self.read = True
        return b"c0", b""


async def test_a_git_call_cancelled_as_git_ends_is_still_cancelled(tmp_path, blobs, monkeypatch):
    """The cancellation lands after git ended but before the call has its output: it must not be dropped."""

    process = _EndingGit()

    async def spawn(*argv: str, **kwargs: object) -> _EndingGit:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    call = asyncio.ensure_future(GitExperimentRepo(tmp_path / "experiments.git", blobs).head(INTEGRATION_BRANCH))
    while not process.read:
        await asyncio.sleep(0)
    call.cancel()

    with pytest.raises(asyncio.CancelledError):
        await call
