# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A patch to aiter that spans its package and its csrc, in the experiment repository."""

import difflib
import shutil
import subprocess

from hyperloom.contracts.recipe import ComponentChange, ComponentPatches, LaunchConfig, Recipe, RecipeDelta, apply
from hyperloom.measure.compile import LocalBlobStore, PackageBase
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo

IMAGE = "rocm/sglang@sha256:" + "c" * 64
AITER = "https://github.com/ROCm/aiter"
PIN = "1" * 40
BASE = Recipe(IMAGE, LaunchConfig(server_args="--tp 4"), components=(ComponentPatches("aiter", AITER, PIN),))
LINES = {name: str(value) for value, name in enumerate("abcdefgh", 1)}
OPS, KERNEL = "aiter/ops.py", "csrc/kernels/gemm.cu"


def _lines(**values: str) -> list[str]:
    return [f"{name} = {value}\n" for name, value in {**LINES, **values}.items()]


def _diff(path: str, before: list[str], after: list[str]) -> str:
    body = difflib.unified_diff(before, after, f"a/{path}", f"b/{path}")
    return f"diff --git a/{path} b/{path}\n" + "".join(body)


def _edit(ops: dict[str, str] | None = None, kernel: dict[str, str] | None = None) -> str:
    """One patch setting lines of the package's ``ops.py`` and of csrc's ``gemm.cu``, from the installed files."""
    parts = [(OPS, ops), (KERNEL, kernel)]
    return "".join(_diff(path, _lines(), _lines(**values)) for path, values in parts if values)


def _checkout(tmp_path) -> PackageBase:
    """A develop install: the package and its csrc beside it in one checkout."""
    checkout = tmp_path / "aiter-checkout"
    for path in (OPS, KERNEL):
        (checkout / path).parent.mkdir(parents=True, exist_ok=True)
        (checkout / path).write_text("".join(_lines()))
    return PackageBase("aiter", checkout / "aiter", None, None)


def _repository(tmp_path) -> GitExperimentRepo:
    installed = _checkout(tmp_path)
    return GitExperimentRepo(tmp_path / "experiments.git", LocalBlobStore(tmp_path / "blobs"), lambda _: installed)


def _changed(recipe: Recipe, *patches) -> Recipe:
    return apply(recipe, RecipeDelta(components=(ComponentChange("aiter", add_patches=patches),)))


async def _candidate(repo: GitExperimentRepo, branch: str, root: str, text: str) -> str:
    await repo.create_branch(branch, root)
    stored = await repo.store_patch("aiter", text)
    return await repo.commit(branch, root, _changed(await repo.read_recipe(root), stored), branch)


async def _merged_onto(tmp_path, head_patch: str, candidate_patch: str):
    repo = _repository(tmp_path)
    root = await repo.init_root(BASE)
    first = await _candidate(repo, "s1", root, head_patch)
    head = await repo.commit_integration(root, (await repo.merge(first, root)).delta, (first,))
    return repo, head, await repo.merge(await _candidate(repo, "s2", root, candidate_patch), head)


def _laid_out(tmp_path, stack: ComponentPatches) -> dict[str, str]:
    """The checkout with ``stack`` applied by ``git apply``, as a test's build of the upstream tree would take it."""
    tree = tmp_path / "applied"
    shutil.copytree(tmp_path / "aiter-checkout", tree)
    blobs = LocalBlobStore(tmp_path / "blobs")
    for patch in stack.patches:
        subprocess.run(["git", "apply", "-"], cwd=tree, input=blobs.get(patch.sha256), check=True)
    return {path: (tree / path).read_text() for path in (OPS, KERNEL)}


async def test_a_stored_patch_spanning_the_package_and_csrc_applies_as_one_stack(tmp_path):
    repo = _repository(tmp_path)
    stored = await repo.store_patch("aiter", _edit(ops={"c": "30"}, kernel={"c": "30"}))
    stack = ComponentPatches("aiter", AITER, PIN, (stored,))
    on_top = _diff(OPS, _lines(c="30"), _lines(c="30", e="50")) + _diff(KERNEL, _lines(c="30"), _lines(c="30", e="50"))

    assert stored.path.startswith("patch/aiter/")
    assert await repo.apply_error(stack, on_top) is None


async def test_a_csrc_hunk_that_does_not_apply_is_an_apply_error(tmp_path):
    repo = _repository(tmp_path)
    stored = await repo.store_patch("aiter", _edit(ops={"c": "30"}, kernel={"c": "30"}))
    stack = ComponentPatches("aiter", AITER, PIN, (stored,))

    error = await repo.apply_error(stack, _edit(kernel={"c": "31"}))

    assert error is not None and "does not apply" in error


async def test_a_file_outside_the_package_and_its_source_trees_is_an_apply_error(tmp_path):
    repo = _repository(tmp_path)
    stack = ComponentPatches("aiter", AITER, PIN)

    error = await repo.apply_error(stack, _diff("3rdparty/composable_kernel/include/ck.hpp", _lines(), _lines(c="30")))

    assert error is not None and "outside" in error


async def test_a_patch_spanning_the_package_and_csrc_merges_beside_the_head(tmp_path):
    repo, head, merge = await _merged_onto(
        tmp_path, _edit(ops={"a": "10"}, kernel={"a": "10"}), _edit(ops={"h": "80"}, kernel={"h": "80"})
    )

    assert merge.clean and merge.rebased_patches == ()
    stack = apply(await repo.read_recipe(head), merge.delta).component("aiter")
    assert stack is not None and len(stack.patches) == 2
    assert _laid_out(tmp_path, stack) == {path: "".join(_lines(a="10", h="80")) for path in (OPS, KERNEL)}


async def test_a_patch_spanning_the_package_and_csrc_is_rebased_three_ways(tmp_path):
    """The head moved a context line of both files; no line overlaps, so both files are merged and the patch rebased."""
    repo, head, merge = await _merged_onto(
        tmp_path, _edit(ops={"c": "30"}, kernel={"c": "30"}), _edit(ops={"e": "50"}, kernel={"e": "50"})
    )

    (rebased,) = merge.rebased_patches
    assert merge.clean and rebased.component == "aiter"
    stack = apply(await repo.read_recipe(head), merge.delta).component("aiter")
    assert stack is not None and stack.patches[-1] == rebased.patch
    assert _laid_out(tmp_path, stack) == {path: "".join(_lines(c="30", e="50")) for path in (OPS, KERNEL)}


async def test_a_csrc_line_both_sides_changed_stays_a_conflict_and_the_package_file_is_rebased(tmp_path):
    repo, head, merge = await _merged_onto(
        tmp_path, _edit(ops={"c": "30"}, kernel={"c": "30", "h": "80"}), _edit(ops={"e": "50"}, kernel={"c": "31"})
    )

    (conflict,) = merge.patch_conflicts
    (rebased,) = merge.rebased_patches
    assert conflict.component == "aiter" and "does not apply" in conflict.detail
    rest = await repo.patch_text(head, conflict.path)
    assert KERNEL in rest and OPS not in rest
    stack = apply(await repo.read_recipe(head), merge.delta).component("aiter")
    assert stack is not None and stack.patches[-1] == rebased.patch
    assert _laid_out(tmp_path, stack) == {OPS: "".join(_lines(c="30", e="50")), KERNEL: "".join(_lines(c="30", h="80"))}
