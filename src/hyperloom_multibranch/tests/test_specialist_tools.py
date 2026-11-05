# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import subprocess
import time
from types import SimpleNamespace

import pytest

from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, Recipe, RecipeDelta, apply
from hyperloom.hosted.mcp_server import InvalidParamsError, ToolError
from hyperloom.measure.compile import LocalBlobStore, PackageBase
from hyperloom_multibranch.contracts.sources import NativeTree, SourceRoot
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.specialist import tools as tools_module
from hyperloom_multibranch.specialist.tools import register_worktree_tools, sync_to_commit
from hyperloom_multibranch.specialist import worktree as worktree_module
from hyperloom_multibranch.specialist.worktree import SourceBase

BRANCH = "s1"
ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)


class _Server:
    """Keeps the registered handlers and calls one as the service does, as the specialist of ``branch``."""

    def __init__(self) -> None:
        self.handlers = {}

    def register(self, name, *, description, schema, handler, scopes):
        self.handlers[name] = handler

    async def call(self, name: str, **arguments):
        return await self.handlers[name](arguments, SimpleNamespace(principal=SimpleNamespace(branch=BRANCH)))


@pytest.fixture
async def rig(tmp_path, monkeypatch):
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "config.py").write_bytes(b"max_num_seqs = 128\n")
    installed = PackageBase("vllm", package, None, None)
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: installed)
    base = SourceBase(tmp_path / "source")
    base.build(["vllm"])
    blobs = LocalBlobStore(tmp_path / "blobs")
    repo = GitExperimentRepo(tmp_path / "experiments.git", blobs, lambda _layout: installed)
    root = await repo.init_root(ROOT)
    await repo.create_branch(BRANCH, root)
    server = _Server()

    def worktrees(branch):
        return base.worktree(branch, tmp_path / "worktrees" / branch)

    monkeypatch.setattr(tools_module, "locate_installed_package", lambda _layout: installed)
    register_worktree_tools(server, repo=repo, worktrees=worktrees)
    return SimpleNamespace(
        server=server,
        repo=repo,
        blobs=blobs,
        root=root,
        worktrees=worktrees,
        config=worktrees(BRANCH).path / "vllm/config.py",
    )


async def test_a_change_made_after_a_sync_is_a_patch_ready_to_test(rig):
    synced = await rig.server.call("worktree_sync")
    rig.config.write_bytes(b"max_num_seqs = 256\n")

    taken = await rig.server.call("worktree_patch", component="vllm")

    assert synced["head"] == taken["parent_commit"] == rig.root
    (stored,) = apply(ROOT, RecipeDelta.from_document(taken["delta"])).component("vllm").patches
    assert stored.to_document() == taken["patch"]
    assert b"+max_num_seqs = 256" in rig.blobs.get(stored.sha256)


async def test_a_change_is_checked_on_the_cpu_before_it_is_tested(rig):
    await rig.server.call("worktree_sync")
    rig.config.write_bytes(b"max_num_seqs = (\n")

    with pytest.raises(ToolError, match="vllm/config.py does not compile"):
        await rig.server.call("worktree_check", component="vllm")
    rig.config.write_bytes(b"max_num_seqs = 256\n")
    checked = await rig.server.call("worktree_check", component="vllm")

    assert checked == {"component": "vllm", "compiled": 1, "imported": "vllm"}


async def test_a_patch_after_the_head_moved_waits_for_the_next_sync(rig):
    await rig.server.call("worktree_sync")
    rig.config.write_bytes(b"max_num_seqs = 256\n")
    taken = await rig.server.call("worktree_patch", component="vllm")
    landed = apply(ROOT, RecipeDelta.from_document(taken["delta"]))
    await rig.repo.commit(BRANCH, rig.root, landed, "max_num_seqs 256")

    with pytest.raises(ToolError, match="call worktree_sync"):
        await rig.server.call("worktree_patch", component="vllm")
    await rig.server.call("worktree_sync")
    assert rig.config.read_bytes() == b"max_num_seqs = 256\n"
    with pytest.raises(ToolError, match="no change to vllm"):
        await rig.server.call("worktree_patch", component="vllm")


async def test_a_component_the_worktree_holds_no_source_of_is_refused(rig):
    await rig.server.call("worktree_sync")

    for tool in ("worktree_check", "worktree_patch"):
        with pytest.raises(InvalidParamsError, match="no source of sglang"):
            await rig.server.call(tool, component="sglang")


def _installed_checkout(tmp_path, version_suffix: str = "") -> PackageBase:
    """A package installed in place from a git checkout, as the image's frameworks are; its version names the commit
    it was built from unless ``version_suffix`` says otherwise."""
    checkout = tmp_path / "checkout"
    package = checkout / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "config.py").write_bytes(b"max_num_seqs = 128\n")
    for argv in (
        ["init", "-q"],
        ["remote", "add", "origin", "https://github.com/vllm-project/vllm.git"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "built"],
    ):
        subprocess.run(["git", *argv], cwd=checkout, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, check=True, capture_output=True, text=True)
    version = version_suffix or f"0.11.0.dev1+g{commit.stdout.strip()[:10]}"
    return PackageBase("vllm", package, "vllm", version)


async def _new_component_rig(tmp_path, installed: PackageBase, monkeypatch):
    bare = Recipe(ROOT.base_image_digest, ROOT.launch)
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: installed)
    base = SourceBase(tmp_path / "source")
    base.build(["vllm"])
    blobs = LocalBlobStore(tmp_path / "blobs")
    repo = GitExperimentRepo(tmp_path / "experiments.git", blobs, lambda _layout: installed)
    root = await repo.init_root(bare)
    await repo.create_branch(BRANCH, root)
    server = _Server()

    def worktrees(branch):
        return base.worktree(branch, tmp_path / "worktrees" / branch)

    monkeypatch.setattr(tools_module, "locate_installed_package", lambda _layout: installed)
    register_worktree_tools(server, repo=repo, worktrees=worktrees)
    await server.call("worktree_sync")
    (worktrees(BRANCH).path / "vllm/config.py").write_bytes(b"max_num_seqs = 256\n")
    return server, bare


async def test_a_patch_to_a_component_new_on_the_head_names_the_installed_upstream_commit(tmp_path, monkeypatch):
    installed = _installed_checkout(tmp_path)
    server, bare = await _new_component_rig(tmp_path, installed, monkeypatch)
    built = subprocess.run(["git", "rev-parse", "HEAD"], cwd=installed.path, check=True, capture_output=True, text=True)

    taken = await server.call("worktree_patch", component="vllm")

    component = apply(bare, RecipeDelta.from_document(taken["delta"])).component("vllm")
    assert (component.upstream_repo, component.upstream_ref) == (
        "https://github.com/vllm-project/vllm.git",
        built.stdout.strip(),
    )
    assert "note" not in taken


async def test_a_new_component_whose_install_names_no_commit_is_said_so_with_the_patch(tmp_path, monkeypatch):
    server, _bare = await _new_component_rig(
        tmp_path, _installed_checkout(tmp_path, version_suffix="0.11.0"), monkeypatch
    )

    taken = await server.call("worktree_patch", component="vllm")

    change = RecipeDelta.from_document(taken["delta"]).components[0]
    assert change.upstream_ref is None and change.add_patches
    assert "set upstream_repo and upstream_ref" in taken["note"]


async def test_a_patch_to_a_component_with_a_native_tree_names_the_commit_that_tree_was_laid_out_from(
    tmp_path, monkeypatch
):
    installed = _installed_checkout(tmp_path, version_suffix="0.11.0")
    checkout = tmp_path / "sources" / "vllm" / ("2" * 40)
    (checkout / "csrc").mkdir(parents=True)
    (checkout / "csrc" / "cache.cu").write_bytes(b"int block = 16;\n")
    tree = NativeTree(SourceRoot("csrc", checkout / "csrc"), "https://github.com/vllm-project/vllm.git", "2" * 40)
    bare = Recipe(ROOT.base_image_digest, ROOT.launch)
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: installed)
    base = SourceBase(tmp_path / "source", native={"vllm": tree})
    base.build(["vllm"])
    repo = GitExperimentRepo(
        tmp_path / "experiments.git",
        LocalBlobStore(tmp_path / "blobs"),
        lambda _layout: installed,
        native={"vllm": tree},
    )
    await repo.create_branch(BRANCH, await repo.init_root(bare))
    server = _Server()

    def worktrees(branch):
        return base.worktree(branch, tmp_path / "worktrees" / branch)

    monkeypatch.setattr(tools_module, "locate_installed_package", lambda _layout: installed)
    register_worktree_tools(server, repo=repo, worktrees=worktrees)
    await server.call("worktree_sync")
    (worktrees(BRANCH).path / "csrc" / "cache.cu").write_bytes(b"int block = 32;\n")

    taken = await server.call("worktree_patch", component="vllm")

    component = apply(bare, RecipeDelta.from_document(taken["delta"])).component("vllm")
    assert (component.upstream_repo, component.upstream_ref) == ("https://github.com/vllm-project/vllm.git", "2" * 40)
    assert "note" not in taken
    assert b"b/csrc/cache.cu" in LocalBlobStore(tmp_path / "blobs").get(component.patches[0].sha256)


def _agent_commit(worktree, message: str) -> str:
    git = ["git", "-c", "user.name=agent", "-c", "user.email=agent@a"]
    subprocess.run([*git, "commit", "-qam", message], cwd=worktree, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=worktree, check=True, capture_output=True, text=True).stdout


async def test_a_change_the_specialist_committed_in_its_worktree_is_still_its_patch(rig):
    await rig.server.call("worktree_sync")
    rig.config.write_bytes(b"max_num_seqs = 256\n")
    _agent_commit(rig.config.parent.parent, "max_num_seqs 256")

    taken = await rig.server.call("worktree_patch", component="vllm")

    assert taken["parent_commit"] == rig.root
    (stored,) = apply(ROOT, RecipeDelta.from_document(taken["delta"])).component("vllm").patches
    assert b"+max_num_seqs = 256" in rig.blobs.get(stored.sha256)


async def test_a_sync_names_the_commits_of_the_specialist_s_it_drops(rig):
    await rig.server.call("worktree_sync")
    rig.config.write_bytes(b"max_num_seqs = 256\n")
    commit = _agent_commit(rig.config.parent.parent, "max_num_seqs 256")

    synced = await rig.server.call("worktree_sync")

    assert synced["dropped_commits"] == [commit.strip()]
    assert rig.config.read_bytes() == b"max_num_seqs = 128\n"
    assert (await rig.server.call("worktree_sync"))["dropped_commits"] == []


async def _two_heads(rig):
    """The branch's root and a head on it whose patch sets max_num_seqs to 256, with the worktree synced to the root."""
    await rig.server.call("worktree_sync")
    rig.config.write_bytes(b"max_num_seqs = 256\n")
    taken = await rig.server.call("worktree_patch", component="vllm")
    landed = await rig.repo.commit(BRANCH, rig.root, apply(ROOT, RecipeDelta.from_document(taken["delta"])), "256")
    return {rig.root: b"max_num_seqs = 128\n", landed: b"max_num_seqs = 256\n"}


async def test_concurrent_syncs_of_one_worktree_leave_it_at_one_head_with_its_patches(rig):
    files = await _two_heads(rig)
    worktree = rig.worktrees(BRANCH)

    outcomes = await asyncio.gather(
        *(sync_to_commit(rig.repo, head, worktree) for head in [*files] * 4), return_exceptions=True
    )

    assert [outcome for outcome in outcomes if isinstance(outcome, BaseException)] == []
    assert rig.config.read_bytes() == files[worktree.head()]
    assert worktree.diff("vllm") == ""


async def test_a_patch_taken_beside_a_sync_is_the_change_against_the_head_it_names(rig, monkeypatch):
    files = await _two_heads(rig)
    worktree = rig.worktrees(BRANCH)
    git = worktree_module._git

    def slow_git(cwd, *args, **kwargs):
        if args[0] == "add":
            time.sleep(0.2)
        return git(cwd, *args, **kwargs)

    monkeypatch.setattr(worktree_module, "_git", slow_git)
    taken = []
    for _round in range(3):
        await rig.server.call("worktree_sync")
        rig.config.write_bytes(b"max_num_seqs = 512\n")
        _, outcome = await asyncio.gather(
            sync_to_commit(rig.repo, rig.root, worktree),
            rig.server.call("worktree_patch", component="vllm"),
            return_exceptions=True,
        )
        if not isinstance(outcome, ToolError):
            taken.append(outcome)

    for patch in taken:
        (stored,) = RecipeDelta.from_document(patch["delta"]).components[0].add_patches
        text = rig.blobs.get(stored.sha256).decode()
        assert (f"-{files[patch['parent_commit']].decode()}", "+max_num_seqs = 512") == tuple(
            line + "\n" if line.startswith("-") else line for line in text.splitlines()[-2:]
        )
