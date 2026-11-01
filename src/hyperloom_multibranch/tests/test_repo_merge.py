# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import difflib
import hashlib
import json
import os
import subprocess
from collections.abc import Collection
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import (
    REMOVED,
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
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    LocalBlobStore,
    PackageBase,
    PatchApplyError,
    apply_patch_stack,
    kernel_load_env,
    kernel_rows_only,
)
from hyperloom_multibranch.contracts.llm import CompletionReply
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import ReplacedPatch, RepoMerge, ResolvedPatch, UnitedTable
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.repo.merge import Stored, carry, stack_kernel_slot, wider_table
from hyperloom_multibranch.repo.rebase import Files
from hyperloom_multibranch.scheduling.integrator import LlmPatchResolver
from hyperloom_multibranch.toolbox.kernel_artifacts import mint_kernel_artifact, union_rows
from hyperloom_multibranch.toolbox.tests.kernel_rows_cc2bf0ef import (
    B01_HEAD,
    B01_NEW,
    B01_SHARED,
    B06_HEAD,
    B06_NEW,
    B06_SHARED,
    B09_NEW,
    ENV,
    SHARED,
    csv_text,
    install,
)

IMAGE = "rocm/vllm@sha256:" + "c" * 64
#: A commit made outside the repository's own code, which sets its identity itself, on a host with no git identity.
IDENTITY = {"NAME": "test", "EMAIL": "test@hyperloom.invalid"}
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


def _keep(data: bytes) -> str:
    sha256 = hashlib.sha256(data).hexdigest()
    TABLES[sha256] = data
    return sha256


def _aiter(data: bytes) -> KernelArtifact:
    """A table the toolbox put on the aiter variable of run ``cc2bf0ef``'s tables."""

    _keep(data)
    return mint_kernel_artifact(KernelKind.GEMM, "a8w8", ENV, data)


def _aiter_rows(data: bytes) -> KernelArtifact:
    """Rows the toolbox put over the image's default table of that variable."""

    _keep(data)
    return mint_kernel_artifact(KernelKind.GEMM, "a8w8", ENV, data, rows_only=True)


def _rows_only(name: str, *rows: str) -> KernelArtifact:
    """An artifact the toolbox put as rows to read over the image's default table, as `rows_only` records it."""

    load = {"via": "env_file", "env": "AITER_CONFIG_GEMM", "rows_only": True}
    return replace(_kernel(name, *rows), metadata={"load": load})


def _changed(recipe: Recipe, config: ConfigDelta | None = None, **change: object) -> Recipe:
    components = (ComponentChange("vllm", **change),) if change else ()
    return apply(recipe, RecipeDelta(config or ConfigDelta(), components))


def _applies(_component: ComponentPatches, _paths: Collection[str]) -> Files:
    return {}


def _store(component: str, text: str) -> PatchRef:
    raise AssertionError(f"nothing of {component} is rebased: {text}")


def _refuses(path: str, *, unless: str = ""):
    """A package ``path`` does not apply to, unless the stack also holds ``unless``."""

    def check(component: ComponentPatches, _paths: Collection[str]) -> Files:
        held = {patch.path for patch in component.patches}
        if path in held and unless not in held:
            raise PatchApplyError(f"{path} does not apply to the installed package: patch failed: vllm/config.py:1")
        return {}

    return check


def test_the_candidates_settings_land_beside_what_the_head_changed():
    candidate = _changed(
        BASE,
        ConfigDelta(
            server_args="--max-num-seqs 256", remove_args=("--enforce-eager",), envs={"NCCL_MIN_NCHANNELS": "16"}
        ),
    )
    ours = _changed(BASE, ConfigDelta(server_args="--enable-chunked-prefill", unset_envs=("VLLM_USE_V1",)))

    merge = carry("i1", BASE, candidate, ours, Stored(_applies, _table, _store, _keep))

    merged = apply(ours, merge.delta)
    assert [change.name for change in diff(candidate, merged).args] == ["--enable-chunked-prefill"]
    assert [change.name for change in diff(candidate, merged).envs] == ["VLLM_USE_V1"]
    assert (merge.config_conflicts, merge.clean) == ((), True)


def test_a_setting_both_sides_changed_takes_the_candidates_value_and_is_listed():
    candidate = _changed(BASE, ConfigDelta(server_args="--max-num-seqs 256"))
    ours = _changed(BASE, ConfigDelta(server_args="--max-num-seqs 192"))

    merge = carry("i1", BASE, candidate, ours, Stored(_applies, _table, _store, _keep))

    assert diff(candidate, apply(ours, merge.delta)).is_empty
    assert merge.config_conflicts == (ValueChange("--max-num-seqs", "192", "256"),)


def test_the_candidates_removals_from_the_resolved_launch_land_beside_the_heads_and_one_it_stopped_leaves():
    base = _changed(BASE, ConfigDelta(remove_args=("--resolved-a",)))
    candidate = _changed(base, ConfigDelta(launch_remove_args=("--resolved-b",)))
    ours = _changed(base, ConfigDelta(remove_args=("--enforce-eager", "--resolved-c")))

    merge = carry("i1", base, candidate, ours, Stored(_applies, _table, _store, _keep))

    merged = apply(ours, merge.delta)
    assert (merged.launch.server_args, merged.launch.remove_args) == (
        "--max-num-seqs 128",
        ("--resolved-b", "--resolved-c"),
    )
    assert (merge.config_conflicts, merge.clean) == ((), True)


def test_a_removal_from_the_resolved_launch_against_a_value_the_head_set_takes_the_candidates_and_is_listed():
    candidate = _changed(BASE, ConfigDelta(remove_args=("--enable-prefix-caching",)))
    ours = _changed(BASE, ConfigDelta(server_args="--enable-prefix-caching"))

    merge = carry("i1", BASE, candidate, ours, Stored(_applies, _table, _store, _keep))

    assert diff(candidate, apply(ours, merge.delta)).is_empty
    assert merge.config_conflicts == (ValueChange("--enable-prefix-caching", "", REMOVED),)


def test_the_candidates_patches_stack_after_the_heads():
    candidate, ours = _changed(BASE, add_patches=(_patch("theirs"),)), _changed(BASE, add_patches=(_patch("ours"),))

    merge = carry("i1", BASE, candidate, ours, Stored(_applies, _table, _store, _keep))

    assert apply(ours, merge.delta).component("vllm").patches == (_patch("ours"), _patch("theirs"))
    assert merge.clean


def test_a_candidate_patch_that_does_not_apply_after_the_heads_is_left_out_and_reported():
    candidate = _changed(BASE, ConfigDelta(server_args="--max-num-seqs 256"), add_patches=(_patch("theirs"),))
    ours = _changed(BASE, add_patches=(_patch("ours"),))

    merge = carry("i1", BASE, candidate, ours, Stored(_refuses("patch/vllm/theirs.patch"), _table, _store, _keep))

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

    merge = carry("i1", base, candidate, ours, Stored(check, _table, _store, _keep))

    assert [conflict.path for conflict in merge.patch_conflicts] == ["patch/vllm/built-on-it.patch"]
    assert apply(ours, merge.delta).component("vllm").patches == ()


def test_a_candidate_labelling_a_component_with_another_upstream_commit_still_merges_its_patches():
    candidate = _changed(BASE, upstream_ref=MOVED, add_patches=(_patch("theirs"),))
    ours = _changed(BASE, upstream_ref=ELSEWHERE, add_patches=(_patch("ours"),))

    merge = carry("i1", BASE, candidate, ours, Stored(_applies, _table, _store, _keep))

    merged = apply(ours, merge.delta).component("vllm")
    assert merge.clean and merged is not None
    assert merged.patches == (_patch("ours"), _patch("theirs"))
    assert merged.upstream_ref == ELSEWHERE


def test_a_component_the_candidate_adds_carries_its_upstream_repo_and_commit():
    added = ComponentChange("triton", MOVED, TRITON, add_patches=(_patch("theirs"),))
    candidate = apply(BASE, RecipeDelta(components=(added,)))

    merge = carry("i1", BASE, candidate, BASE, Stored(_applies, _table, _store, _keep))

    merged = apply(BASE, merge.delta).component("triton")
    assert merged is not None
    assert (merged.upstream_repo, merged.upstream_ref, merged.patches) == (TRITON, MOVED, (_patch("theirs"),))


def test_a_component_the_candidate_records_without_patching_it_leaves_the_merge_empty():
    recorded = ComponentChange("triton", MOVED, TRITON)
    candidate = apply(BASE, RecipeDelta(components=(recorded,)))

    merge = carry("i1", BASE, candidate, BASE, Stored(_applies, _table, _store, _keep))

    assert merge.delta.components == ()
    assert apply(BASE, merge.delta).digest() == BASE.digest()


def test_a_kernel_slot_both_sides_filled_does_not_merge_and_a_new_one_lands():
    candidate = apply(BASE, RecipeDelta(add_kernels=(_kernel("theirs"),)))

    assert apply(BASE, carry("i1", BASE, candidate, BASE, Stored(_applies, _table, _store, _keep)).delta).kernels == (
        _kernel("theirs"),
    )
    with pytest.raises(RecipeError, match="both sides replaced the gemm kernel for moe"):
        carry(
            "i1",
            BASE,
            candidate,
            apply(BASE, RecipeDelta(add_kernels=(_kernel("ours"),))),
            Stored(_applies, _table, _store, _keep),
        )


def test_the_candidates_table_lands_when_it_holds_every_row_of_the_head_s():
    """A branch that tuned the shapes the head's table holds and more: the wider table serves both."""

    wider = _kernel("b09", "304,1,4096,4096,23.1", "304,8,4096,4096,31.0", "304,16,4096,4096,44.2")
    head = apply(BASE, RecipeDelta(add_kernels=(_kernel("head", "304,1,4096,4096,23.1", "304,8,4096,4096,31.0"),)))
    candidate = apply(BASE, RecipeDelta(add_kernels=(wider,)))

    merge = carry("i1", BASE, candidate, head, Stored(_applies, _table, _store, _keep))

    assert apply(head, merge.delta).kernels == (wider,)


def test_a_table_the_head_already_holds_every_row_of_leaves_the_head_s_in_place():
    narrower = _kernel("b02", "304,1,4096,4096,23.1")
    head = apply(BASE, RecipeDelta(add_kernels=(_kernel("head", "304,1,4096,4096,23.1", "304,8,4096,4096,31.0"),)))
    candidate = apply(BASE, RecipeDelta(add_kernels=(narrower,)))

    merge = carry("i1", BASE, candidate, head, Stored(_applies, _table, _store, _keep))

    assert merge.delta.add_kernels == ()
    assert apply(head, merge.delta).kernels == head.kernels


def test_two_tables_with_a_row_the_other_lacks_do_not_merge():
    """Tables no test can load name no variable to read their shapes by, so their rows are never united."""

    head = apply(BASE, RecipeDelta(add_kernels=(_kernel("head", "304,1,4096,4096,23.1"),)))
    candidate = apply(BASE, RecipeDelta(add_kernels=(_kernel("b06", "304,1,4096,4096,19.8", "304,8,4096,4096,31.0"),)))

    with pytest.raises(RecipeError, match="both sides replaced the gemm kernel for moe"):
        carry("i1", BASE, candidate, head, Stored(_applies, _table, _store, _keep))


def test_tables_under_unlike_headers_do_not_merge():
    """Rows of one table say nothing about another's when the columns differ, whatever rows they share."""

    head = apply(BASE, RecipeDelta(add_kernels=(_kernel("head", "304,1,4096,4096,23.1"),)))
    other = KernelArtifact(KernelKind.GEMM, "moe", "kernel/gemm/other.so", hashlib.sha256(b"other").hexdigest())
    TABLES[other.sha256] = b"cu_num,M,N,K,gfx,us\n304,1,4096,4096,gfx950,23.1\n"

    with pytest.raises(RecipeError, match="both sides replaced the gemm kernel for moe"):
        carry("i1", BASE, apply(BASE, RecipeDelta(add_kernels=(other,))), head, Stored(_applies, _table, _store, _keep))


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
        carry("i1", BASE, candidate, head, Stored(_applies, _table, _store, _keep))


@pytest.fixture
def configs(tmp_path) -> Path:
    return install(tmp_path / "image")


def _filled(*artifacts: KernelArtifact) -> Recipe:
    return apply(BASE, RecipeDelta(add_kernels=artifacts))


def test_two_tables_each_with_rows_the_other_lacks_unite_into_one_table(configs):
    """Shapes only one side tuned, and shared shapes tuned to one row: every row of both serves in one table."""

    head, theirs = _aiter(csv_text(SHARED, B01_NEW)), _aiter(csv_text(SHARED[:3], B06_NEW))

    merge = carry("i1", BASE, _filled(theirs), _filled(head), Stored(_applies, _table, _store, _keep, configs))

    (united,) = merge.delta.add_kernels
    data = union_rows(TABLES[theirs.sha256], TABLES[head.sha256], ENV, configs=configs)
    assert (united.slot, united.sha256, TABLES[united.sha256]) == (head.slot, hashlib.sha256(data).hexdigest(), data)
    assert united.metadata == theirs.metadata and not kernel_rows_only(united)
    assert merge.united_tables == (UnitedTable(theirs, head, united),)
    assert apply(_filled(head), merge.delta).kernels == (united,)


def test_a_shape_both_sides_tuned_to_other_rows_keeps_the_head_s_row_and_unites_the_rest(configs):
    """The head's row was measured on integration; the candidate's other shapes still join it, and the shapes whose row
    the merge kept from the head are named."""

    head, theirs = _aiter(csv_text(SHARED, B01_HEAD)), _aiter(csv_text(B01_SHARED, B01_NEW))

    merge = carry("i1", BASE, _filled(theirs), _filled(head), Stored(_applies, _table, _store, _keep, configs))

    (united,) = merge.delta.add_kernels
    expected = union_rows(csv_text(B01_NEW), TABLES[head.sha256], ENV, configs=configs)
    assert TABLES[united.sha256] == expected
    (kept_from,) = merge.united_tables
    assert kept_from.ours == head and len(kept_from.kept_head_shapes) == 2
    named = {" ".join(f"{k}={v}" for k, v in shape.items()) for shape in kept_from.kept_head_shapes}
    assert any("M=64 N=34816 K=5120" in shape for shape in named)
    assert any("M=64 N=5120 K=17408" in shape for shape in named)


def test_two_rows_only_sets_unite_into_a_rows_only_set(configs):
    head, theirs = _aiter_rows(csv_text(B01_NEW)), _aiter_rows(csv_text(B06_NEW))

    merge = carry("i1", BASE, _filled(theirs), _filled(head), Stored(_applies, _table, _store, _keep, configs))

    (united,) = merge.delta.add_kernels
    data = union_rows(TABLES[head.sha256], TABLES[theirs.sha256], ENV, rows_only=(True, True), configs=configs)
    assert kernel_rows_only(united) and TABLES[united.sha256] == data


def test_rows_only_rows_over_a_whole_table_make_one_whole_table(configs):
    """Rows put over the image's table read the same over the head's whole table: the union is a whole table."""

    head, theirs = _aiter(csv_text(SHARED)), _aiter_rows(csv_text(B06_NEW))

    merge = carry("i1", BASE, _filled(theirs), _filled(head), Stored(_applies, _table, _store, _keep, configs))

    (united,) = merge.delta.add_kernels
    assert not kernel_rows_only(united)
    assert TABLES[united.sha256] == union_rows(csv_text(B06_NEW), TABLES[head.sha256], ENV, configs=configs)


def _aiter_for(target: str, data: bytes) -> KernelArtifact:
    """A table the toolbox put on the same aiter variable as :func:`_aiter`, tuned for another layer's shapes."""

    _keep(data)
    return mint_kernel_artifact(KernelKind.GEMM, target, ENV, data)


def _envs(recipe: Recipe) -> list[str]:
    return [kernel_load_env(artifact) for artifact in recipe.kernels]


def test_tables_for_two_targets_that_load_through_one_variable_unite_into_one_table(configs):
    """The service reads one table per variable, so a candidate's table for another layer cannot land beside the head's:
    the two unite, and the merged recipe sets the variable once."""

    head, theirs = _aiter(csv_text(SHARED, B01_NEW)), _aiter_for("qkv", csv_text(SHARED[:3], B06_NEW))

    merge = carry("i1", BASE, _filled(theirs), _filled(head), Stored(_applies, _table, _store, _keep, configs))

    merged = apply(_filled(head), merge.delta)
    (united,) = merged.kernels
    data = union_rows(TABLES[theirs.sha256], TABLES[head.sha256], ENV, configs=configs)
    assert (united.slot, TABLES[united.sha256], _envs(merged)) == (head.slot, data, [ENV])
    assert merge.united_tables == (UnitedTable(theirs, head, united),)


def test_a_wider_table_for_another_target_on_one_variable_replaces_the_head_s(configs):
    head, wide = _aiter(csv_text(SHARED)), _aiter_for("qkv", csv_text(SHARED, B09_NEW))

    merge = carry("i1", BASE, _filled(wide), _filled(head), Stored(_applies, _table, _store, _keep, configs))

    assert apply(_filled(head), merge.delta).kernels == (wide,)


def test_a_replay_stacks_a_kernel_slot_as_the_merge_fills_it(configs):
    """The wider table, else the union the merge makes, else ``None`` where the merge refuses the pair."""

    narrow, wide = _aiter(csv_text(SHARED)), _aiter(csv_text(SHARED, B09_NEW))
    head, theirs = _aiter(csv_text(SHARED, B01_HEAD)), _aiter(csv_text(B01_SHARED, B01_NEW))
    stored = Stored(_applies, _table, _store, _keep, configs)

    assert stack_kernel_slot(wide, narrow, stored) == stack_kernel_slot(narrow, wide, stored) == wide
    merged = carry("i1", BASE, _filled(theirs), _filled(head), stored).delta.add_kernels
    assert (stack_kernel_slot(theirs, head, stored),) == merged
    refused = _rows_only("b02", "304,1,4096,4096,19.8"), _kernel("head", "304,1,4096,4096,23.1")
    assert stack_kernel_slot(*refused, stored) is None


#: ``B01_HEAD``'s shapes at another time: neither the head's row nor the candidate's.
B01_BASE = tuple(row.replace(",0,20.0,", ",0,30.0,") for row in B01_HEAD)


def _named(shapes) -> set[str]:
    return {" ".join(f"{k}={shape[k]}" for k in "MNK") for shape in shapes}


def test_a_union_of_only_the_head_s_rows_carries_no_kernel(configs):
    """Every shape the candidate differs on the head also tuned: the head's table stays, and the kept shapes are named."""

    base = _aiter(csv_text(SHARED[:2]))
    head, theirs = _aiter(csv_text(SHARED, B01_HEAD)), _aiter(csv_text(SHARED[:2], B01_SHARED))

    merge = carry("i1", _filled(base), _filled(theirs), _filled(head), Stored(_applies, _table, _store, _keep, configs))

    assert merge.delta.add_kernels == ()
    assert apply(_filled(head), merge.delta).kernels == (head,)
    (united,) = merge.united_tables
    assert (united.theirs, united.ours, united.table) == (theirs, head, head)
    assert _named(united.kept_head_shapes) == {"M=64 N=34816 K=5120", "M=64 N=5120 K=17408"}


def _untimed(rows: tuple[str, ...]) -> tuple[str, ...]:
    """``rows`` with no time in their ``us`` cell."""
    return tuple(",".join(cell if index != 8 else "" for index, cell in enumerate(row.split(","))) for row in rows)


def test_a_candidate_s_retune_of_a_row_the_head_left_as_in_base_lands_where_a_row_has_no_time(configs):
    base = _aiter(csv_text(SHARED, B01_HEAD))
    head, theirs = _aiter(csv_text(SHARED, B01_HEAD, B09_NEW)), _aiter(csv_text(SHARED, _untimed(B01_SHARED)))

    merge = carry("i1", _filled(base), _filled(theirs), _filled(head), Stored(_applies, _table, _store, _keep, configs))

    (united,) = merge.delta.add_kernels
    assert TABLES[united.sha256] == union_rows(TABLES[theirs.sha256], csv_text(SHARED, B09_NEW), ENV, configs=configs)
    assert merge.united_tables == (UnitedTable(theirs, head, united),)


def test_a_head_row_left_as_in_base_stays_where_the_tuner_timed_it_faster(configs):
    """``B01_HEAD``'s rows are timed at 20 us, the candidate's retunes of them at 49 and 39."""

    base = _aiter(csv_text(SHARED, B01_HEAD))
    head, theirs = _aiter(csv_text(SHARED, B01_HEAD, B09_NEW)), _aiter(csv_text(SHARED, B01_SHARED))

    merge = carry("i1", _filled(base), _filled(theirs), _filled(head), Stored(_applies, _table, _store, _keep, configs))

    assert merge.delta.add_kernels == ()
    (united,) = merge.united_tables
    assert united.table == head
    assert _named(united.kept_head_shapes) == {"M=64 N=34816 K=5120", "M=64 N=5120 K=17408"}


def test_a_candidate_row_the_tuner_timed_faster_replaces_the_head_s_retune(configs):
    base = _aiter(csv_text(SHARED, B01_BASE))
    head, theirs = _aiter(csv_text(SHARED, B01_SHARED)), _aiter(csv_text(SHARED, B01_HEAD))

    merge = carry("i1", _filled(base), _filled(theirs), _filled(head), Stored(_applies, _table, _store, _keep, configs))

    (united,) = merge.delta.add_kernels
    assert TABLES[united.sha256] == union_rows(TABLES[theirs.sha256], csv_text(SHARED), ENV, configs=configs)
    assert merge.united_tables == (UnitedTable(theirs, head, united),)


def test_only_a_shape_both_sides_moved_from_base_keeps_the_head_s_row_and_is_named(configs):
    """The head retuned a shape the candidate left as in base, and both retuned another: the head's rows serve both,
    and only the second is named."""

    base = _aiter(csv_text(SHARED, B06_HEAD, B01_BASE))
    head = _aiter(csv_text(SHARED, B06_SHARED, B01_HEAD))
    theirs = _aiter(csv_text(SHARED, B06_HEAD, B01_SHARED, B01_NEW))

    merge = carry("i1", _filled(base), _filled(theirs), _filled(head), Stored(_applies, _table, _store, _keep, configs))

    (united,) = merge.delta.add_kernels
    assert TABLES[united.sha256] == union_rows(csv_text(B01_NEW), TABLES[head.sha256], ENV, configs=configs)
    (kept_from,) = merge.united_tables
    assert _named(kept_from.kept_head_shapes) == {"M=64 N=34816 K=5120", "M=64 N=5120 K=17408"}


def test_a_replay_keeps_the_head_s_table_where_the_union_is_its_rows(configs):
    head, theirs = _aiter(csv_text(SHARED, B01_HEAD)), _aiter(csv_text(SHARED[:2], B01_SHARED))

    assert stack_kernel_slot(theirs, head, Stored(_applies, _table, _store, _keep, configs)) == head


def test_a_candidate_on_another_base_image_does_not_merge():
    with pytest.raises(RecipeError, match="base image"):
        carry(
            "i1",
            BASE,
            replace(BASE, base_image_digest="rocm/vllm@sha256:" + "d" * 64),
            BASE,
            Stored(_applies, _table, _store, _keep),
        )


def _git_patch(file: str, old: str, new: str) -> str:
    return (
        f"diff --git a/vllm/{file} b/vllm/{file}\n--- a/vllm/{file}\n+++ b/vllm/{file}\n@@ -1 +1 @@\n-{old}\n+{new}\n"
    )


async def _candidate(repo: GitExperimentRepo, branch: str, root: str, text: str) -> str:
    await repo.create_branch(branch, root)
    stored = await repo.store_patch("vllm", text)
    return await repo.commit(branch, root, _changed(await repo.read_recipe(root), add_patches=(stored,)), branch)


def _installed(tmp_path, **files: str) -> PackageBase:
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    for name, text in files.items():
        (package / f"{name}.py").write_text(text)
    return PackageBase("vllm", package, None, None)


def _repository(tmp_path, installed: PackageBase) -> GitExperimentRepo:
    return GitExperimentRepo(tmp_path / "experiments.git", LocalBlobStore(tmp_path / "blobs"), lambda _: installed)


async def test_the_repository_reports_what_does_not_apply_to_the_installed_package(tmp_path):
    repo = _repository(tmp_path, _installed(tmp_path, config="max_num_seqs = 128\n", scheduler="policy = 'fcfs'\n"))
    root = await repo.init_root(BASE)
    first = await _candidate(repo, "s1", root, _git_patch("config.py", "max_num_seqs = 128", "max_num_seqs = 192"))
    head = await repo.commit_integration(root, (await repo.merge(first, root)).delta, (first,))

    clash = _git_patch("config.py", "max_num_seqs = 128", "max_num_seqs = 256")
    clashing = await repo.merge(await _candidate(repo, "s2", root, clash), head, frozenset({first}))
    elsewhere = _git_patch("scheduler.py", "policy = 'fcfs'", "policy = 'lpm'")
    beside = await repo.merge(await _candidate(repo, "s3", root, elsewhere), head)

    (conflict,) = clashing.patch_conflicts
    assert conflict.component == "vllm" and "does not apply" in conflict.detail
    assert [change.add_patches for change in clashing.delta.components] == [()]
    assert beside.clean and len(apply(await repo.read_recipe(head), beside.delta).component("vllm").patches) == 2
    assert (clashing.rebased_patches, beside.rebased_patches) == ((), ())
    assert await repo.head(INTEGRATION_BRANCH) == head


CONFIG = {name: str(value) for value, name in enumerate("abcdefgh", 1)}


def _config(**values: str) -> list[str]:
    return [f"{name} = {value}\n" for name, value in {**CONFIG, **values}.items()]


def _edit(**values: str) -> str:
    """A patch of the installed ``config.py`` setting each named line, with three lines of context as git writes."""

    body = difflib.unified_diff(_config(), _config(**values), "a/vllm/config.py", "b/vllm/config.py")
    return "diff --git a/vllm/config.py b/vllm/config.py\n" + "".join(body)


async def _merged_onto(tmp_path, head_patch: str, candidate_patch: str):
    """The candidate's merge onto an integration head that holds ``head_patch``, both branched from one root."""

    installed = _installed(tmp_path, config="".join(_config()))
    repo = _repository(tmp_path, installed)
    root = await repo.init_root(BASE)
    first = await _candidate(repo, "s1", root, head_patch)
    head = await repo.commit_integration(root, (await repo.merge(first, root)).delta, (first,))
    return repo, installed, head, await repo.merge(await _candidate(repo, "s2", root, candidate_patch), head)


async def test_the_repository_keeps_a_united_table_a_checkout_writes(tmp_path, configs):
    blobs = LocalBlobStore(tmp_path / "blobs")
    repo = GitExperimentRepo(tmp_path / "experiments.git", blobs, aiter_configs=configs)
    root = await repo.init_root(BASE)
    commits = []
    tables = [csv_text(SHARED, B01_NEW), csv_text(SHARED[:3], B06_NEW)]
    for branch, data in zip(("s1", "s2"), tables, strict=True):
        blobs.put(data)
        await repo.create_branch(branch, root)
        commits.append(await repo.commit(branch, root, _filled(_aiter(data)), branch))
    head = await repo.commit_integration(root, (await repo.merge(commits[0], root)).delta, commits[:1])

    merge = await repo.merge(commits[1], head)

    (united,) = merge.united_tables
    merged = await repo.commit_integration(head, merge.delta, commits[1:])
    await repo.checkout(merged, tmp_path / "checkout")
    assert (await repo.read_recipe(merged)).kernels == (united.table,)
    assert (tmp_path / "checkout" / united.table.ref).read_bytes() == union_rows(*tables, ENV, configs=configs)


async def test_a_patch_whose_context_the_head_moved_is_rebased_onto_it(tmp_path):
    """Nearby lines: the candidate's context holds a line the head changed, so it does not apply, but no line overlaps."""

    repo, installed, head, merge = await _merged_onto(tmp_path, _edit(c="30"), _edit(e="50"))

    (rebased,) = merge.rebased_patches
    assert merge.clean and rebased.component == "vllm"
    stack = (await repo.read_recipe(await repo.commit_integration(head, merge.delta, ()))).component("vllm")
    assert stack is not None and stack.patches[-1] == rebased.patch
    assert rebased.path not in {patch.path for patch in stack.patches}
    blobs = LocalBlobStore(tmp_path / "blobs")
    (entry,) = apply_patch_stack(stack, COMPONENT_LAYOUTS["vllm"], installed, blobs).entries
    assert blobs.get(entry.blob).decode() == "".join(_config(c="30", e="50"))


async def test_a_line_both_sides_changed_stays_a_conflict(tmp_path):
    """The head's patch also changes a line the candidate's leaves alone, so the candidate's does not replace it."""

    *_, merge = await _merged_onto(tmp_path, _edit(c="30", h="80"), _edit(c="31", e="50"))

    (conflict,) = merge.patch_conflicts
    assert "does not apply" in conflict.detail and merge.rebased_patches == ()
    assert [change.add_patches for change in merge.delta.components] == [()]


async def test_a_change_the_head_already_holds_is_not_carried_twice(tmp_path):
    *_, merge = await _merged_onto(tmp_path, _edit(c="30", e="50"), _edit(e="50"))

    assert merge.clean and merge.rebased_patches == ()
    assert [change.add_patches for change in merge.delta.components] == [()]


def _on(lines: list[str], before: list[str] | None = None) -> str:
    """A patch of the installed ``config.py`` from ``before`` (the installed file) to ``lines``."""

    body = difflib.unified_diff(before or _config(), lines, "a/vllm/config.py", "b/vllm/config.py")
    return "diff --git a/vllm/config.py b/vllm/config.py\n" + "".join(body)


async def _replaced_onto(tmp_path, head_patches: list[str], candidate_patch: str, *, protect: bool = False):
    """The candidate's merge from the root onto a head that merged each of ``head_patches`` in turn, each branched off
    the head before it; ``protect`` passes the first of those merges as a foundation fix."""

    installed = _installed(tmp_path, config="".join(_config()))
    repo = _repository(tmp_path, installed)
    head = root = await repo.init_root(BASE)
    merged = []
    for number, text in enumerate(head_patches):
        merged.append(await _candidate(repo, f"h{number}", head, text))
        head = await repo.commit_integration(head, (await repo.merge(merged[-1], head)).delta, merged[-1:])
    candidate = await _candidate(repo, "s2", root, candidate_patch)
    merge = await repo.merge(candidate, head, frozenset(merged[:1]) if protect else frozenset())
    return repo, installed, head, merge


def _paths(repo: GitExperimentRepo, recipe: Recipe) -> list[str]:
    stack = recipe.component("vllm")
    return [] if stack is None else [patch.path for patch in stack.patches]


async def test_a_head_patch_the_candidate_rewrote_in_full_is_replaced_by_the_candidate_s(tmp_path):
    """b19 in miniature: the head merged X after the candidate branched off; P sets the same lines otherwise."""

    repo, installed, head, merge = await _replaced_onto(tmp_path, [_edit(c="30", d="40")], _edit(c="31", d="41"))

    (x,) = _paths(repo, await repo.read_recipe(head))
    (replaced,) = merge.replaced_patches
    assert merge.clean and (replaced.component, replaced.removed) == ("vllm", (x,))
    (change,) = merge.delta.components
    assert (change.remove_patches, change.add_patches) == ((x,), (replaced.by,))
    stack = apply(await repo.read_recipe(head), merge.delta).component("vllm")
    blobs = LocalBlobStore(tmp_path / "blobs")
    (entry,) = apply_patch_stack(stack, COMPONENT_LAYOUTS["vllm"], installed, blobs).entries
    assert blobs.get(entry.blob).decode() == "".join(_config(c="31", d="41"))


async def test_a_head_patch_with_a_hunk_the_candidate_leaves_alone_is_not_replaced(tmp_path):
    *_, merge = await _replaced_onto(tmp_path, [_edit(a="10", g="70")], _edit(a="11"))

    (conflict,) = merge.patch_conflicts
    assert "does not apply" in conflict.detail and merge.replaced_patches == ()
    assert [(change.add_patches, change.remove_patches) for change in merge.delta.components] == [((), ())]


async def test_a_foundation_fix_on_the_head_is_never_replaced(tmp_path):
    *_, merge = await _replaced_onto(tmp_path, [_edit(c="30", d="40")], _edit(c="31", d="41"), protect=True)

    assert len(merge.patch_conflicts) == 1 and merge.replaced_patches == ()
    assert [change.remove_patches for change in merge.delta.components] == [()]


async def test_a_head_patch_that_only_applies_on_top_of_the_replaced_one_goes_with_it(tmp_path):
    on_x = _on(_config(c="30", d="40"), _config(c="30"))
    repo, _, head, merge = await _replaced_onto(tmp_path, [_edit(c="30"), on_x], _edit(c="31", d="41"))

    x, dependent = _paths(repo, await repo.read_recipe(head))
    (replaced,) = merge.replaced_patches
    assert merge.clean and replaced.removed == (x, dependent)
    assert _paths(repo, apply(await repo.read_recipe(head), merge.delta)) == [replaced.by.path]


async def test_a_head_patch_beside_the_replaced_one_stays(tmp_path):
    beside = _on(_config(c="30", h="80"), _config(c="30"))
    repo, _, head, merge = await _replaced_onto(tmp_path, [_edit(c="30"), beside], _edit(c="31"))

    x, kept = _paths(repo, await repo.read_recipe(head))
    (replaced,) = merge.replaced_patches
    assert merge.clean and replaced.removed == (x,)
    assert _paths(repo, apply(await repo.read_recipe(head), merge.delta)) == [kept, replaced.by.path]


def test_a_merge_replaces_no_head_patch_unless_the_tier_settles_one():
    replaced = ReplacedPatch("vllm", ("patch/vllm/old.patch",), _patch("new"))

    assert RepoMerge(PIN, RecipeDelta()).replaced_patches == ()
    assert replace(RepoMerge(PIN, RecipeDelta()), replaced_patches=(replaced,)).replaced_patches == (replaced,)


def _blob_names(tmp_path) -> set[str]:
    return {path.name for path in (tmp_path / "blobs").iterdir()}


async def _applied_on(tmp_path) -> tuple[GitExperimentRepo, ComponentPatches]:
    """A repository over an installed ``config.py`` and a stack whose one patch sets ``c``."""

    repo = _repository(tmp_path, _installed(tmp_path, config="".join(_config())))
    stored = await repo.store_patch("vllm", _edit(c="30"))
    return repo, ComponentPatches("vllm", VLLM, PIN, (stored,))


async def test_a_patch_that_applies_on_top_of_the_stack_has_no_apply_error(tmp_path):
    repo, stack = await _applied_on(tmp_path)
    before = _blob_names(tmp_path)

    body = difflib.unified_diff(_config(c="30"), _config(c="30", e="50"), "a/vllm/config.py", "b/vllm/config.py")
    on_top = "diff --git a/vllm/config.py b/vllm/config.py\n" + "".join(body)

    assert await repo.apply_error(stack, on_top) is None
    assert await repo.apply_error(stack, _edit(e="50")) is not None
    assert _blob_names(tmp_path) == before


async def test_a_patch_that_conflicts_with_the_stack_returns_why(tmp_path):
    repo, stack = await _applied_on(tmp_path)
    before = _blob_names(tmp_path)

    error = await repo.apply_error(stack, _edit(c="31"))

    assert error is not None and "does not apply to the installed package" in error
    assert _blob_names(tmp_path) == before


async def test_a_text_that_is_no_git_patch_is_an_apply_error(tmp_path):
    repo, stack = await _applied_on(tmp_path)
    before = _blob_names(tmp_path)

    error = await repo.apply_error(stack, "set c to 31 in config.py")

    assert error is not None and "not a git patch" in error
    assert _blob_names(tmp_path) == before


def test_a_merge_carries_no_resolved_patch_until_the_integrator_adds_one():
    resolved = ResolvedPatch("vllm", "patch/vllm/old.patch", _patch("new"), "claude-sonnet-5-5", "v1", 2)

    assert RepoMerge(PIN, RecipeDelta()).resolved_patches == ()
    assert (resolved.component, resolved.replaces, resolved.patch) == ("vllm", "patch/vllm/old.patch", _patch("new"))
    assert (resolved.model, resolved.prompt_version, resolved.attempts) == ("claude-sonnet-5-5", "v1", 2)
    assert replace(RepoMerge(PIN, RecipeDelta()), resolved_patches=(resolved,)).resolved_patches == (resolved,)


def _setting(**config: object) -> RecipeDelta:
    return RecipeDelta(ConfigDelta(**config))


async def _lineage(repo: GitExperimentRepo, branch: str, at: str, *deltas: RecipeDelta) -> list[str]:
    """Commits of ``deltas`` in turn on a new ``branch`` from ``at``, each on the one before."""

    await repo.create_branch(branch, at)
    commits, parent = [], at
    for number, delta in enumerate(deltas):
        parent = await repo.commit(branch, parent, apply(await repo.read_recipe(parent), delta), f"{branch}.{number}")
        commits.append(parent)
    return commits


async def _merged(repo: GitExperimentRepo, candidate: str) -> str:
    head = await repo.head(INTEGRATION_BRANCH)
    return await repo.commit_integration(head, (await repo.merge(candidate, head)).delta, (candidate,))


@pytest.fixture
async def rooted(tmp_path) -> tuple[GitExperimentRepo, str]:
    repo = GitExperimentRepo(tmp_path / "experiments.git", LocalBlobStore(tmp_path / "blobs"))
    return repo, await repo.init_root(BASE)


async def test_an_unmerged_lineage_is_carried_from_the_merge_base(rooted):
    repo, root = rooted
    first, second = await _lineage(
        repo, "s1", root, _setting(server_args="--max-num-seqs 256"), _setting(envs={"VLLM_ROCM_USE_AITER": "1"})
    )
    (other,) = await _lineage(repo, "s2", root, _setting(server_args="--enable-chunked-prefill"))
    head = await _merged(repo, other)

    merge = await repo.merge(second, head)

    merged = apply(await repo.read_recipe(head), merge.delta)
    assert merge.clean and merge.carried == (first, second)
    assert "--max-num-seqs 256" in merged.launch.server_args and "--enable-chunked-prefill" in merged.launch.server_args
    assert merged.launch.envs["VLLM_ROCM_USE_AITER"] == "1"


async def test_a_lineage_whose_first_commit_was_merged_carries_only_the_second(rooted):
    repo, root = rooted
    first, second = await _lineage(
        repo, "s1", root, _setting(server_args="--max-num-seqs 256"), _setting(envs={"VLLM_ROCM_USE_AITER": "1"})
    )
    head = await _merged(repo, first)

    merge = await repo.merge(second, head)

    assert merge.carried == (second,)
    assert merge.delta == _setting(envs={"VLLM_ROCM_USE_AITER": "1"})


async def test_a_child_of_a_parent_never_merged_carries_the_parent(tmp_path):
    """b13's 65ee1103 in miniature: the child rewrites the line its parent changed, which the head never took."""

    repo = _repository(tmp_path, _installed(tmp_path, config="".join(_config())))
    root = await repo.init_root(BASE)
    parent_patch = await repo.store_patch("vllm", _edit(c="30"))
    child_patch = await repo.store_patch("vllm", _on(_config(c="31"), _config(c="30")))
    parent, child = await _lineage(repo, "s1", root, _adding_patch(parent_patch), _adding_patch(child_patch))
    (other,) = await _lineage(repo, "s2", root, _adding_patch(await repo.store_patch("vllm", _edit(h="80"))))
    head = await _merged(repo, other)

    merge = await repo.merge(child, head)

    assert merge.clean and merge.carried == (parent, child)
    stack = apply(await repo.read_recipe(head), merge.delta).component("vllm")
    assert stack is not None and [patch.path for patch in stack.patches][-2:] == [parent_patch.path, child_patch.path]


async def test_a_candidate_already_on_onto_carries_nothing(rooted):
    repo, root = rooted
    (first,) = await _lineage(repo, "s1", root, _setting(server_args="--max-num-seqs 256"))
    head = await _merged(repo, first)

    for candidate in (root, first, head):
        merge = await repo.merge(candidate, head)
        assert (merge.onto, merge.delta, merge.carried) == (head, RecipeDelta(), ())


async def test_a_criss_cross_history_does_not_merge(tmp_path, rooted):
    repo, root = rooted
    (left,) = await _lineage(repo, "s1", root, _setting(server_args="--max-num-seqs 256"))
    (right,) = await _lineage(repo, "s2", root, _setting(envs={"VLLM_ROCM_USE_AITER": "1"}))
    head = await repo.commit_integration(root, RecipeDelta(), (left, right))
    git = ["git", "--git-dir", str(tmp_path / "experiments.git")]
    tree = subprocess.run([*git, "rev-parse", f"{head}^{{tree}}"], capture_output=True, text=True, check=True)
    argv = [*git, "commit-tree", "-p", right, "-p", left, "-m", "crossed", tree.stdout.strip()]
    who = {f"GIT_{role}_{field}": value for role in ("AUTHOR", "COMMITTER") for field, value in IDENTITY.items()}
    crossed = subprocess.run(argv, capture_output=True, text=True, check=True, env={**os.environ, **who}).stdout.strip()

    with pytest.raises(RecipeError, match="merge bases") as raised:
        await repo.merge(crossed, head)
    assert left in str(raised.value) and right in str(raised.value)


async def test_a_reverse_merge_carries_every_merge_integration_took_since_the_branch_forked(rooted):
    """zheng's case: integration's head carried onto a leading branch that forked before three merges."""

    repo, root = rooted
    (lead,) = await _lineage(repo, "lead", root, _setting(server_args="--max-num-seqs 256"))
    taken = []
    for branch, delta in (
        ("s1", _setting(server_args="--enable-chunked-prefill")),
        ("s2", _setting(envs={"VLLM_ROCM_USE_AITER": "1"})),
        ("s3", _setting(remove_args=("--enforce-eager",))),
    ):
        (commit,) = await _lineage(repo, branch, await repo.head(INTEGRATION_BRANCH), delta)
        taken.append((commit, await _merged(repo, commit)))
    head = taken[-1][1]

    merge = await repo.merge(head, lead)

    merged = apply(await repo.read_recipe(lead), merge.delta)
    assert merge.clean and set(merge.carried) == {commit for pair in taken for commit in pair}
    assert merge.carried[-1] == head and lead not in merge.carried
    assert merged.launch.server_args.split() == ["--max-num-seqs", "256", "--enable-chunked-prefill"]
    assert merged.launch.envs == {"VLLM_USE_V1": "1", "VLLM_ROCM_USE_AITER": "1"}


def _adding_patch(patch: PatchRef) -> RecipeDelta:
    return RecipeDelta(components=(ComponentChange("vllm", add_patches=(patch,)),))


SCHEDULER = "policy = 'fcfs'\n"
#: c7eddf22's 3b275cb2 in miniature: one patch over two files, only one of which the head also changed.
WIDE = _edit(a="11") + _git_patch("scheduler.py", "policy = 'fcfs'", "policy = 'lpm'")


async def _wide_onto(tmp_path) -> tuple[GitExperimentRepo, PackageBase, str, str, RepoMerge]:
    installed = _installed(tmp_path, config="".join(_config()), scheduler=SCHEDULER)
    repo = _repository(tmp_path, installed)
    root = await repo.init_root(BASE)
    first = await _candidate(repo, "s1", root, _edit(a="10", g="70"))
    head = await repo.commit_integration(root, (await repo.merge(first, root)).delta, (first,))
    candidate = await _candidate(repo, "s2", root, WIDE)
    return repo, installed, head, candidate, await repo.merge(candidate, head)


def _built(stack: ComponentPatches, installed: PackageBase, tmp_path) -> dict[str, str]:
    blobs = LocalBlobStore(tmp_path / "blobs")
    entries = apply_patch_stack(stack, COMPONENT_LAYOUTS["vllm"], installed, blobs).entries
    return {entry.path: blobs.get(entry.blob).decode() for entry in entries}


async def test_only_the_files_of_a_patch_that_overlap_the_head_stay_a_conflict(tmp_path):
    repo, installed, head, candidate, merge = await _wide_onto(tmp_path)

    (wide,) = (await repo.read_recipe(candidate)).component("vllm").patches
    (rebased,) = merge.rebased_patches
    (conflict,) = merge.patch_conflicts
    blobs = LocalBlobStore(tmp_path / "blobs")
    assert rebased.path == wide.path and "scheduler.py" in blobs.get(rebased.patch.sha256).decode()
    assert "config.py" not in blobs.get(rebased.patch.sha256).decode()
    rest = await repo.patch_text(candidate, conflict.path)
    assert "config.py" in rest and "scheduler.py" not in rest and "does not apply" in conflict.detail
    (change,) = merge.delta.components
    assert (change.add_patches, change.remove_patches) == ((rebased.patch,), ())
    built = _built(apply(await repo.read_recipe(head), merge.delta).component("vllm"), installed, tmp_path)
    assert built == {"config.py": "".join(_config(a="10", g="70")), "scheduler.py": "policy = 'lpm'\n"}


async def test_the_resolver_is_asked_about_the_overlapping_files_alone(tmp_path):
    repo, installed, head, candidate, merge = await _wide_onto(tmp_path)
    asked: list[str] = []
    rewrite = _on(_config(a="11", g="70"), _config(a="10", g="70"))

    async def completion(_system: str, prompt: str) -> CompletionReply:
        asked.append(prompt)
        return CompletionReply(json.dumps({"patch": rewrite}), TokenCounts(1, 1), None)

    resolved = await LlmPatchResolver(completion, repo, "model").resolve(merge, candidate)

    (to_rewrite,) = [prompt.split("Why it does not apply")[0] for prompt in asked]
    assert "config.py" in to_rewrite and "scheduler.py" not in to_rewrite
    built = _built(apply(await repo.read_recipe(head), resolved.delta).component("vllm"), installed, tmp_path)
    assert built == {"config.py": "".join(_config(a="11", g="70")), "scheduler.py": "policy = 'lpm'\n"}
