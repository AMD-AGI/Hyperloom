# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, Recipe, RecipeDelta
from hyperloom.hosted.mcp_server import ToolError
from hyperloom.inference_optimizer.framework_paths import GENERIC_FRAMEWORK_ROOT_ENV
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.specialist.tools import register_worktree_tools
from hyperloom_multibranch.specialist.worktree import SourceBase

LLAMACPP = "llamacpp"
ORIGIN = "https://github.com/ggml-org/llama.cpp.git"
TRACKED = {
    "ggml/src/ggml-cuda/mmvq.cu": "int nwarps = 4;\n",
    "src/llama.cpp": "// libllama\n",
    "include/llama.h": "// header\n",
    "common/speculative.cpp": "// draft\n",
    "tools/server/server.cpp": "// server\n",
    "CMakeLists.txt": "project(llama.cpp)\n",
    "vendor/cpp-httplib/httplib.h": "// vendored\n",
    "tools/llama-bench/llama-bench.cpp": "// bench\n",
}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "llama.cpp"
    for name, text in TRACKED.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    _git(root, "init", "-q", "-b", "master")
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "pinned")
    _git(root, "remote", "add", "origin", ORIGIN)
    # What an operator's builds leave behind: untracked, never part of the commit a build compiles.
    for untracked in ("build-gfx1100/bin/llama-server", "ggml/build/ggml.o", "src/scratch.txt"):
        (root / untracked).parent.mkdir(parents=True, exist_ok=True)
        (root / untracked).write_text("built\n")
    monkeypatch.setenv(GENERIC_FRAMEWORK_ROOT_ENV, str(root))
    return root


def _files(tree: Path) -> list[str]:
    return sorted(
        path.relative_to(tree).as_posix()
        for path in tree.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(tree).parts
    )


def test_a_checkout_component_lays_out_the_tracked_files_of_its_trees_alone(tmp_path: Path, checkout: Path) -> None:
    base = SourceBase(tmp_path / "source")
    base.build([LLAMACPP])

    assert _files(base.path) == [
        "common/speculative.cpp",
        "ggml/src/ggml-cuda/mmvq.cu",
        "include/llama.h",
        "src/llama.cpp",
        "tools/server/server.cpp",
    ]


def test_a_worktree_change_to_any_tree_of_the_checkout_is_the_component_s_and_scratch_beside_it_is_not(
    tmp_path: Path, checkout: Path
) -> None:
    base = SourceBase(tmp_path / "source")
    base.build([LLAMACPP])
    worktree = base.worktree("s1", tmp_path / "s1")
    (worktree.path / "ggml/src/ggml-cuda/mmvq.cu").write_text("int nwarps = 2;\n")
    (worktree.path / "common/speculative.cpp").write_text("// draft more\n")
    (worktree.path / "notes.txt").write_text("scratch\n")

    patch = worktree.diff(LLAMACPP)

    assert "a/ggml/src/ggml-cuda/mmvq.cu" in patch and "a/common/speculative.cpp" in patch
    assert "notes.txt" not in patch
    assert worktree.holds(LLAMACPP)


async def test_the_experiment_repository_applies_a_checkout_component_s_patch_over_its_checkout(
    tmp_path: Path, checkout: Path
) -> None:
    repo = GitExperimentRepo(tmp_path / "experiments.git", LocalBlobStore(tmp_path / "blobs"))
    stack = ComponentPatches(LLAMACPP, ORIGIN, _git(checkout, "rev-parse", "HEAD"))

    assert await repo.apply_error(stack, _diff("common/speculative.cpp", "// draft", "// draft more")) is None
    error = await repo.apply_error(stack, _diff("common/speculative.cpp", "// gone", "// draft more"))
    assert error is not None and "does not apply" in error


def _diff(path: str, before: str, after: str) -> str:
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-{before}\n+{after}\n"


class _Server:
    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}

    def register(self, name, *, description, schema, handler, scopes):
        self.handlers[name] = handler

    async def call(self, name: str, **arguments: Any) -> Any:
        return await self.handlers[name](arguments, SimpleNamespace(principal=SimpleNamespace(branch="s1")))


async def _tools(tmp_path: Path) -> _Server:
    base = SourceBase(tmp_path / "source")
    base.build([LLAMACPP])
    repo = GitExperimentRepo(tmp_path / "experiments.git", LocalBlobStore(tmp_path / "blobs"))
    await repo.create_branch("s1", await repo.init_root(Recipe("ghcr.io/x@sha256:" + "c" * 64, LaunchConfig())))
    server = _Server()
    register_worktree_tools(server, repo=repo, worktrees=lambda branch: base.worktree(branch, tmp_path / branch))
    await server.call("worktree_sync")
    (tmp_path / "s1" / "ggml/src/ggml-cuda/mmvq.cu").write_text("int nwarps = 2;\n")
    return server


async def test_a_patch_to_a_checkout_built_framework_names_the_commit_its_checkout_is_at(
    tmp_path: Path, checkout: Path
) -> None:
    server = await _tools(tmp_path)

    taken = await server.call("worktree_patch", component=LLAMACPP)

    change = RecipeDelta.from_document(taken["delta"]).components[0]
    assert (change.upstream_repo, change.upstream_ref) == (ORIGIN, _git(checkout, "rev-parse", "HEAD"))
    assert "note" not in taken


async def test_a_checkout_built_framework_is_not_checked_by_import(tmp_path: Path, checkout: Path) -> None:
    server = await _tools(tmp_path)

    with pytest.raises(ToolError, match="compiled from its checkout, not imported"):
        await server.call("worktree_check", component=LLAMACPP)
