# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
from dataclasses import replace

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
    RecipeError,
    ValueChange,
    apply,
    diff,
)
from hyperloom.measure.compile import LocalBlobStore, PackageBase, PatchApplyError
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.repo.merge import carry

IMAGE = "rocm/vllm@sha256:" + "c" * 64
VLLM = "https://github.com/vllm-project/vllm"
TRITON = "https://github.com/triton-lang/triton"
PIN, MOVED, ELSEWHERE = "1" * 40, "2" * 40, "3" * 40
BASE = Recipe(
    IMAGE,
    LaunchConfig(server_args="--max-num-seqs 128 --enforce-eager", envs={"VLLM_USE_V1": "1"}),
    components=(ComponentPatches("vllm", VLLM, PIN),),
)


def _patch(name: str) -> PatchRef:
    return PatchRef(f"patch/vllm/{name}.patch", hashlib.sha256(name.encode()).hexdigest())


def _kernel(name: str) -> KernelArtifact:
    return KernelArtifact(KernelKind.GEMM, "moe", f"kernel/gemm/{name}.so", hashlib.sha256(name.encode()).hexdigest())


def _changed(recipe: Recipe, config: ConfigDelta | None = None, **change: object) -> Recipe:
    components = (ComponentChange("vllm", **change),) if change else ()
    return apply(recipe, RecipeDelta(config or ConfigDelta(), components))


def _applies(_component: ComponentPatches) -> None:
    return None


def _refuses(path: str, *, unless: str = ""):
    """A package ``path`` does not apply to, unless the stack also holds ``unless``."""

    def check(component: ComponentPatches) -> None:
        held = {patch.path for patch in component.patches}
        if path in held and unless not in held:
            raise PatchApplyError(f"{path} does not apply to the installed package: patch failed: vllm/config.py:1")

    return check


def test_the_candidates_settings_land_beside_what_the_head_changed():
    candidate = _changed(
        BASE,
        ConfigDelta(
            server_args="--max-num-seqs 256", remove_args=("--enforce-eager",), envs={"NCCL_MIN_NCHANNELS": "16"}
        ),
    )
    ours = _changed(BASE, ConfigDelta(server_args="--enable-chunked-prefill", unset_envs=("VLLM_USE_V1",)))

    merge = carry("i1", BASE, candidate, ours, _applies)

    merged = apply(ours, merge.delta)
    assert [change.name for change in diff(candidate, merged).args] == ["--enable-chunked-prefill"]
    assert [change.name for change in diff(candidate, merged).envs] == ["VLLM_USE_V1"]
    assert (merge.config_conflicts, merge.clean) == ((), True)


def test_a_setting_both_sides_changed_takes_the_candidates_value_and_is_listed():
    candidate = _changed(BASE, ConfigDelta(server_args="--max-num-seqs 256"))
    ours = _changed(BASE, ConfigDelta(server_args="--max-num-seqs 192"))

    merge = carry("i1", BASE, candidate, ours, _applies)

    assert diff(candidate, apply(ours, merge.delta)).is_empty
    assert merge.config_conflicts == (ValueChange("--max-num-seqs", "192", "256"),)


def test_the_candidates_patches_stack_after_the_heads():
    candidate, ours = _changed(BASE, add_patches=(_patch("theirs"),)), _changed(BASE, add_patches=(_patch("ours"),))

    merge = carry("i1", BASE, candidate, ours, _applies)

    assert apply(ours, merge.delta).component("vllm").patches == (_patch("ours"), _patch("theirs"))
    assert merge.clean


def test_a_candidate_patch_that_does_not_apply_after_the_heads_is_left_out_and_reported():
    candidate = _changed(BASE, ConfigDelta(server_args="--max-num-seqs 256"), add_patches=(_patch("theirs"),))
    ours = _changed(BASE, add_patches=(_patch("ours"),))

    merge = carry("i1", BASE, candidate, ours, _refuses("patch/vllm/theirs.patch"))

    (conflict,) = merge.patch_conflicts
    assert (conflict.component, conflict.path, merge.clean) == ("vllm", "patch/vllm/theirs.patch", False)
    assert "does not apply" in conflict.detail
    merged = apply(ours, merge.delta)
    assert (merged.component("vllm").patches, merged.launch.server_args) == (
        (_patch("ours"),),
        candidate.launch.server_args,
    )


def test_a_head_patch_the_candidate_breaks_is_dropped_and_reported():
    base = _changed(BASE, add_patches=(_patch("foundation"),))
    candidate = _changed(base, remove_patches=("patch/vllm/foundation.patch",))
    ours = _changed(base, add_patches=(_patch("built-on-it"),))
    check = _refuses("patch/vllm/built-on-it.patch", unless="patch/vllm/foundation.patch")

    merge = carry("i1", base, candidate, ours, check)

    assert [conflict.path for conflict in merge.patch_conflicts] == ["patch/vllm/built-on-it.patch"]
    assert apply(ours, merge.delta).component("vllm").patches == ()


def test_a_candidate_labelling_a_component_with_another_upstream_commit_still_merges_its_patches():
    candidate = _changed(BASE, upstream_ref=MOVED, add_patches=(_patch("theirs"),))
    ours = _changed(BASE, upstream_ref=ELSEWHERE, add_patches=(_patch("ours"),))

    merge = carry("i1", BASE, candidate, ours, _applies)

    merged = apply(ours, merge.delta).component("vllm")
    assert merge.clean and merged is not None
    assert merged.patches == (_patch("ours"), _patch("theirs"))
    assert merged.upstream_ref == ELSEWHERE


@pytest.mark.parametrize("patches", [(), (_patch("theirs"),)], ids=["without-patches", "with-patches"])
def test_a_component_the_candidate_adds_carries_its_upstream_repo_and_commit(patches):
    added = ComponentChange("triton", MOVED, TRITON, add_patches=patches)
    candidate = apply(BASE, RecipeDelta(components=(added,)))

    merge = carry("i1", BASE, candidate, BASE, _applies)

    merged = apply(BASE, merge.delta).component("triton")
    assert merged is not None
    assert (merged.upstream_repo, merged.upstream_ref, merged.patches) == (TRITON, MOVED, patches)


def test_a_kernel_slot_both_sides_filled_does_not_merge_and_a_new_one_lands():
    candidate = apply(BASE, RecipeDelta(add_kernels=(_kernel("theirs"),)))

    assert apply(BASE, carry("i1", BASE, candidate, BASE, _applies).delta).kernels == (_kernel("theirs"),)
    with pytest.raises(RecipeError, match="both sides replaced the gemm kernel for moe"):
        carry("i1", BASE, candidate, apply(BASE, RecipeDelta(add_kernels=(_kernel("ours"),))), _applies)


def test_a_candidate_on_another_base_image_does_not_merge():
    with pytest.raises(RecipeError, match="base image"):
        carry("i1", BASE, replace(BASE, base_image_digest="rocm/vllm@sha256:" + "d" * 64), BASE, _applies)


def _git_patch(file: str, old: str, new: str) -> str:
    return (
        f"diff --git a/vllm/{file} b/vllm/{file}\n--- a/vllm/{file}\n+++ b/vllm/{file}\n@@ -1 +1 @@\n-{old}\n+{new}\n"
    )


async def _candidate(repo: GitExperimentRepo, branch: str, root: str, text: str) -> str:
    await repo.create_branch(branch, root)
    stored = await repo.store_patch("vllm", text)
    return await repo.commit(branch, root, _changed(await repo.read_recipe(root), add_patches=(stored,)), branch)


async def test_the_repository_reports_what_does_not_apply_to_the_installed_package(tmp_path):
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "config.py").write_bytes(b"max_num_seqs = 128\n")
    (package / "scheduler.py").write_bytes(b"policy = 'fcfs'\n")
    installed = PackageBase("vllm", package, None, None)
    repo = GitExperimentRepo(
        tmp_path / "experiments.git", LocalBlobStore(tmp_path / "blobs"), lambda _layout: installed
    )
    root = await repo.init_root(BASE)
    first = await _candidate(repo, "s1", root, _git_patch("config.py", "max_num_seqs = 128", "max_num_seqs = 192"))
    head = await repo.commit_integration(root, (await repo.merge(first, root)).delta, (first,))

    clash = _git_patch("config.py", "max_num_seqs = 128", "max_num_seqs = 256")
    clashing = await repo.merge(await _candidate(repo, "s2", root, clash), head)
    elsewhere = _git_patch("scheduler.py", "policy = 'fcfs'", "policy = 'lpm'")
    beside = await repo.merge(await _candidate(repo, "s3", root, elsewhere), head)

    (conflict,) = clashing.patch_conflicts
    assert conflict.component == "vllm" and "does not apply" in conflict.detail
    assert [change.add_patches for change in clashing.delta.components] == [()]
    assert beside.clean and len(apply(await repo.read_recipe(head), beside.delta).component("vllm").patches) == 2
    with pytest.raises(ValueError, match="root commit"):
        await repo.merge(root, head)
    assert await repo.head(INTEGRATION_BRANCH) == head
