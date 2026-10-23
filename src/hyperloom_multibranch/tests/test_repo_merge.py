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
from hyperloom_multibranch.repo.merge import Stored, carry, wider_table

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


#: The bytes behind every kernel artifact these tests build, by sha256, as the session's blob store holds them.
TABLES: dict[str, bytes] = {}
HEADER = "cu_num,M,N,K,us"


def _kernel(name: str, *rows: str) -> KernelArtifact:
    """An artifact over a tuned table of ``rows``, or over bytes of its own when a test only needs two slot fillers."""

    data = "\n".join((HEADER, *rows)).encode() if rows else name.encode()
    sha256 = hashlib.sha256(data).hexdigest()
    TABLES[sha256] = data
    return KernelArtifact(KernelKind.GEMM, "moe", f"kernel/gemm/{name}.so", sha256)


def _table(artifact: KernelArtifact) -> bytes:
    return TABLES[artifact.sha256]


def _rows_only(name: str, *rows: str) -> KernelArtifact:
    """An artifact the toolbox put as rows to read over the image's default table, as `rows_only` records it."""

    load = {"via": "env_file", "env": "AITER_CONFIG_GEMM", "rows_only": True}
    return replace(_kernel(name, *rows), metadata={"load": load})


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

    merge = carry("i1", BASE, candidate, ours, Stored(_applies, _table))

    merged = apply(ours, merge.delta)
    assert [change.name for change in diff(candidate, merged).args] == ["--enable-chunked-prefill"]
    assert [change.name for change in diff(candidate, merged).envs] == ["VLLM_USE_V1"]
    assert (merge.config_conflicts, merge.clean) == ((), True)


def test_a_setting_both_sides_changed_takes_the_candidates_value_and_is_listed():
    candidate = _changed(BASE, ConfigDelta(server_args="--max-num-seqs 256"))
    ours = _changed(BASE, ConfigDelta(server_args="--max-num-seqs 192"))

    merge = carry("i1", BASE, candidate, ours, Stored(_applies, _table))

    assert diff(candidate, apply(ours, merge.delta)).is_empty
    assert merge.config_conflicts == (ValueChange("--max-num-seqs", "192", "256"),)


def test_the_candidates_patches_stack_after_the_heads():
    candidate, ours = _changed(BASE, add_patches=(_patch("theirs"),)), _changed(BASE, add_patches=(_patch("ours"),))

    merge = carry("i1", BASE, candidate, ours, Stored(_applies, _table))

    assert apply(ours, merge.delta).component("vllm").patches == (_patch("ours"), _patch("theirs"))
    assert merge.clean


def test_a_candidate_patch_that_does_not_apply_after_the_heads_is_left_out_and_reported():
    candidate = _changed(BASE, ConfigDelta(server_args="--max-num-seqs 256"), add_patches=(_patch("theirs"),))
    ours = _changed(BASE, add_patches=(_patch("ours"),))

    merge = carry("i1", BASE, candidate, ours, Stored(_refuses("patch/vllm/theirs.patch"), _table))

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

    merge = carry("i1", base, candidate, ours, Stored(check, _table))

    assert [conflict.path for conflict in merge.patch_conflicts] == ["patch/vllm/built-on-it.patch"]
    assert apply(ours, merge.delta).component("vllm").patches == ()


def test_a_candidate_labelling_a_component_with_another_upstream_commit_still_merges_its_patches():
    candidate = _changed(BASE, upstream_ref=MOVED, add_patches=(_patch("theirs"),))
    ours = _changed(BASE, upstream_ref=ELSEWHERE, add_patches=(_patch("ours"),))

    merge = carry("i1", BASE, candidate, ours, Stored(_applies, _table))

    merged = apply(ours, merge.delta).component("vllm")
    assert merge.clean and merged is not None
    assert merged.patches == (_patch("ours"), _patch("theirs"))
    assert merged.upstream_ref == ELSEWHERE


def test_a_component_the_candidate_adds_carries_its_upstream_repo_and_commit():
    added = ComponentChange("triton", MOVED, TRITON, add_patches=(_patch("theirs"),))
    candidate = apply(BASE, RecipeDelta(components=(added,)))

    merge = carry("i1", BASE, candidate, BASE, Stored(_applies, _table))

    merged = apply(BASE, merge.delta).component("triton")
    assert merged is not None
    assert (merged.upstream_repo, merged.upstream_ref, merged.patches) == (TRITON, MOVED, (_patch("theirs"),))


def test_a_component_the_candidate_records_without_patching_it_leaves_the_merge_empty():
    recorded = ComponentChange("triton", MOVED, TRITON)
    candidate = apply(BASE, RecipeDelta(components=(recorded,)))

    merge = carry("i1", BASE, candidate, BASE, Stored(_applies, _table))

    assert merge.delta.components == ()
    assert apply(BASE, merge.delta).digest() == BASE.digest()


def test_a_kernel_slot_both_sides_filled_does_not_merge_and_a_new_one_lands():
    candidate = apply(BASE, RecipeDelta(add_kernels=(_kernel("theirs"),)))

    assert apply(BASE, carry("i1", BASE, candidate, BASE, Stored(_applies, _table)).delta).kernels == (
        _kernel("theirs"),
    )
    with pytest.raises(RecipeError, match="both sides replaced the gemm kernel for moe"):
        carry("i1", BASE, candidate, apply(BASE, RecipeDelta(add_kernels=(_kernel("ours"),))), Stored(_applies, _table))


def test_the_candidates_table_lands_when_it_holds_every_row_of_the_head_s():
    """A branch that tuned the shapes the head's table holds and more: the wider table serves both."""

    wider = _kernel("b09", "304,1,4096,4096,23.1", "304,8,4096,4096,31.0", "304,16,4096,4096,44.2")
    head = apply(BASE, RecipeDelta(add_kernels=(_kernel("head", "304,1,4096,4096,23.1", "304,8,4096,4096,31.0"),)))
    candidate = apply(BASE, RecipeDelta(add_kernels=(wider,)))

    merge = carry("i1", BASE, candidate, head, Stored(_applies, _table))

    assert apply(head, merge.delta).kernels == (wider,)


def test_a_table_the_head_already_holds_every_row_of_leaves_the_head_s_in_place():
    narrower = _kernel("b02", "304,1,4096,4096,23.1")
    head = apply(BASE, RecipeDelta(add_kernels=(_kernel("head", "304,1,4096,4096,23.1", "304,8,4096,4096,31.0"),)))
    candidate = apply(BASE, RecipeDelta(add_kernels=(narrower,)))

    merge = carry("i1", BASE, candidate, head, Stored(_applies, _table))

    assert merge.delta.add_kernels == ()
    assert apply(head, merge.delta).kernels == head.kernels


def test_two_tables_with_a_row_the_other_lacks_do_not_merge():
    """The row neither side can prefer: the same shape timed differently, or two disjoint sets of shapes."""

    head = apply(BASE, RecipeDelta(add_kernels=(_kernel("head", "304,1,4096,4096,23.1"),)))
    candidate = apply(BASE, RecipeDelta(add_kernels=(_kernel("b06", "304,1,4096,4096,19.8", "304,8,4096,4096,31.0"),)))

    with pytest.raises(RecipeError, match="both sides replaced the gemm kernel for moe"):
        carry("i1", BASE, candidate, head, Stored(_applies, _table))


def test_tables_under_unlike_headers_do_not_merge():
    """Rows of one table say nothing about another's when the columns differ, whatever rows they share."""

    head = apply(BASE, RecipeDelta(add_kernels=(_kernel("head", "304,1,4096,4096,23.1"),)))
    other = KernelArtifact(KernelKind.GEMM, "moe", "kernel/gemm/other.so", hashlib.sha256(b"other").hexdigest())
    TABLES[other.sha256] = b"cu_num,M,N,K,gfx,us\n304,1,4096,4096,gfx950,23.1\n"

    with pytest.raises(RecipeError, match="both sides replaced the gemm kernel for moe"):
        carry("i1", BASE, apply(BASE, RecipeDelta(add_kernels=(other,))), head, Stored(_applies, _table))


def test_wider_table_names_the_artifact_that_serves_both_or_none():
    """What a replay asks before it swaps the head's table out: whose table loses nothing."""

    narrow = _kernel("b02", "304,1,4096,4096,23.1")
    wide = _kernel("b09", "304,1,4096,4096,23.1", "304,8,4096,4096,31.0")
    crossing = _kernel("b06", "304,8,4096,4096,29.4")

    assert wider_table(wide, narrow, _table) == wide
    assert wider_table(narrow, wide, _table) == wide
    assert wider_table(crossing, wide, _table) is None


def test_rows_put_over_the_image_s_default_table_are_never_the_narrower_side():
    """The rows a branch tuned name the shapes it overrides, not the shapes it serves, so no containment holds."""

    head = apply(BASE, RecipeDelta(add_kernels=(_kernel("head", "304,1,4096,4096,23.1", "304,8,4096,4096,31.0"),)))
    tuned = _rows_only("b02", "304,1,4096,4096,19.8")
    candidate = apply(BASE, RecipeDelta(add_kernels=(tuned,)))

    assert wider_table(tuned, head.kernels[0], _table) is None
    with pytest.raises(RecipeError, match="both sides replaced the gemm kernel for moe"):
        carry("i1", BASE, candidate, head, Stored(_applies, _table))


def test_a_candidate_on_another_base_image_does_not_merge():
    with pytest.raises(RecipeError, match="base image"):
        carry(
            "i1", BASE, replace(BASE, base_image_digest="rocm/vllm@sha256:" + "d" * 64), BASE, Stored(_applies, _table)
        )


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
