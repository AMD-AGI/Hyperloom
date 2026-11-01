# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from hyperloom.contracts.recipe import (
    RECIPE_SCHEMA_VERSION,
    REMOVED,
    ArgsMode,
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
    apply,
    delta_between,
    diff,
    with_unexpressed,
)

IMAGE = "rocm/vllm@sha256:" + "c" * 64
SNAPSHOT = "sha256:" + "d" * 64
VLLM_AT = "1" * 40
VLLM_NEXT = "2" * 40
AITER_AT = "3" * 40
MOE_PATCH = PatchRef("patch/vllm/0001-moe-align.patch", "a" * 64)
ATTN_PATCH = PatchRef("patch/vllm/0002-attn-split.patch", "b" * 64)
GEMM = KernelArtifact(KernelKind.GEMM, "bf16 4096x14336", "kernel/gemm/tuned.csv", "e" * 64, {"speedup": 1.21})


def _recipe(**overrides: object) -> Recipe:
    fields: dict[str, object] = {
        "base_image_digest": IMAGE,
        "launch": LaunchConfig(
            server_args="--max-num-seqs 128 --enable-chunked-prefill", envs={"VLLM_USE_V1": "1", "HIP_FORCE": "0"}
        ),
        "components": (
            ComponentPatches("vllm", "https://github.com/vllm-project/vllm", VLLM_AT, (MOE_PATCH,)),
            ComponentPatches("aiter", "https://github.com/ROCm/aiter", AITER_AT),
        ),
        "kernels": (GEMM,),
        "runtime_snapshot": SNAPSHOT,
    }
    fields.update(overrides)
    return Recipe(**fields)


def test_a_recipe_document_round_trips_and_names_its_parts_like_the_recipe_kb() -> None:
    recipe = _recipe()

    document = recipe.to_document()

    assert Recipe.from_document(document) == recipe
    assert document["schema_version"] == RECIPE_SCHEMA_VERSION
    assert document["launch"] == {
        "extra_server_args": "--max-num-seqs 128 --enable-chunked-prefill",
        "extra_envs": {"HIP_FORCE": "0", "VLLM_USE_V1": "1"},
    }


def test_the_digest_names_the_content_whatever_order_the_mappings_were_built_in() -> None:
    first = _recipe()
    reordered = _recipe(
        launch=LaunchConfig(
            server_args="--max-num-seqs 128 --enable-chunked-prefill", envs={"HIP_FORCE": "0", "VLLM_USE_V1": "1"}
        )
    )

    assert first.digest() == reordered.digest()
    assert first.digest().startswith("sha256:")
    assert first.digest() != _recipe(runtime_snapshot=None).digest()


def test_a_document_of_another_schema_version_is_refused() -> None:
    document = {**_recipe().to_document(), "schema_version": RECIPE_SCHEMA_VERSION + 1}

    with pytest.raises(RecipeError, match="schema version"):
        Recipe.from_document(document)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: _recipe(base_image_digest="rocm/vllm:latest"), "base image digest"),
        (lambda: _recipe(runtime_snapshot="snapshot-1"), "runtime snapshot"),
        (lambda: ComponentPatches("vllm", "https://x", "main"), "upstream commit"),
        (
            lambda: _recipe(
                components=(
                    ComponentPatches("vllm", "https://x", VLLM_AT),
                    ComponentPatches("vllm", "https://y", AITER_AT),
                )
            ),
            "duplicate component names",
        ),
        (lambda: _recipe(kernels=(GEMM, replace(GEMM, ref="kernel/gemm/other.csv"))), "duplicate kernel targets"),
        (lambda: LaunchConfig(server_args="--port 8000; rm -rf /"), "shell control"),
        (lambda: LaunchConfig(server_args=" --max-num-seqs 64"), "trimmed"),
        (lambda: LaunchConfig(envs={"EXTRA_VLLM_ARGS": "--x"}), "framework arguments"),
        (lambda: LaunchConfig(envs={"1BAD": "x"}), "environment variable name"),
    ],
)
def test_a_recipe_that_breaks_the_contract_is_refused(build: Callable[[], object], message: str) -> None:
    with pytest.raises(RecipeError, match=message):
        build()


@pytest.mark.parametrize(
    ("path", "sha256"),
    [
        ("vllm/0001.patch", "a" * 64),
        ("patch/../etc/passwd", "a" * 64),
        ("patch/./0001.patch", "a" * 64),
        ("/patch/0001.patch", "a" * 64),
        ("patch/0001.patch", "A" * 64),
    ],
)
def test_a_patch_must_live_under_patch_and_carry_its_hash(path: str, sha256: str) -> None:
    with pytest.raises(RecipeError):
        PatchRef(path, sha256)


def test_two_patches_of_one_component_cannot_share_a_path() -> None:
    with pytest.raises(RecipeError, match="duplicate patches of vllm"):
        ComponentPatches("vllm", "https://x", VLLM_AT, (MOE_PATCH, replace(ATTN_PATCH, path=MOE_PATCH.path)))


def test_apply_composes_server_args_the_way_the_launcher_does() -> None:
    delta = RecipeDelta(
        config=ConfigDelta(
            server_args="--max-num-seqs 256",
            remove_args=("--enable-chunked-prefill",),
            envs={"VLLM_ROCM_USE_AITER": "1"},
            unset_envs=("HIP_FORCE",),
        )
    )

    launched = apply(_recipe(), delta).launch

    assert launched.server_args == "--max-num-seqs 256"
    assert dict(launched.envs) == {"VLLM_USE_V1": "1", "VLLM_ROCM_USE_AITER": "1"}


def test_apply_moves_components_stacks_patches_and_swaps_kernels() -> None:
    better_gemm = replace(GEMM, ref="kernel/gemm/tuned-v2.csv", sha256="f" * 64)
    delta = RecipeDelta(
        components=(
            ComponentChange("vllm", upstream_ref=VLLM_NEXT, add_patches=(ATTN_PATCH,)),
            ComponentChange("sglang", upstream_ref=VLLM_AT, upstream_repo="https://github.com/sgl-project/sglang"),
        ),
        add_kernels=(better_gemm,),
    )
    original = _recipe()

    changed = apply(original, delta)

    vllm = changed.component("vllm")
    assert vllm is not None
    assert (vllm.upstream_ref, vllm.patches) == (VLLM_NEXT, (MOE_PATCH, ATTN_PATCH))
    assert [component.component for component in changed.components] == ["vllm", "aiter", "sglang"]
    assert changed.kernels == (better_gemm,)
    assert original == _recipe()


def test_delta_between_makes_one_recipe_into_the_other() -> None:
    better_gemm = replace(GEMM, ref="kernel/gemm/tuned-v2.csv", sha256="f" * 64)
    fork = PatchRef("patch/sglang/0001-chunk.patch", "9" * 64)
    sglang = ComponentPatches("sglang", "https://github.com/sgl-project/sglang", VLLM_AT, (fork,))
    first = _recipe()
    second = _recipe(
        launch=LaunchConfig(server_args="--max-num-seqs 256", envs={"VLLM_USE_V1": "1", "VLLM_ROCM_USE_AITER": "1"}),
        components=(
            ComponentPatches("vllm", "https://github.com/vllm-project/vllm", VLLM_NEXT, (MOE_PATCH, ATTN_PATCH)),
            sglang,
        ),
        kernels=(better_gemm,),
    )

    delta = delta_between(first, second)
    made = apply(first, delta)

    assert (delta.config.remove_args, delta.config.unset_envs) == (("--enable-chunked-prefill",), ("HIP_FORCE",))
    assert made.launch == second.launch
    assert {component.component: component for component in made.components} == {
        "vllm": second.component("vllm"),
        "aiter": first.component("aiter"),
        "sglang": sglang,
    }
    assert made.kernels == (better_gemm,)
    assert delta_between(second, second) == RecipeDelta()


def test_delta_between_keeps_the_shared_head_of_a_patch_stack_and_drops_the_patches_the_other_lacks() -> None:
    reworked = replace(MOE_PATCH, sha256="8" * 64)
    stacked = _recipe(
        components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", VLLM_AT, (MOE_PATCH, ATTN_PATCH)),)
    )
    other = _recipe(
        components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", VLLM_AT, (reworked,)),)
    )
    bare = _recipe(components=())

    (change,) = delta_between(stacked, other).components
    (dropped,) = delta_between(stacked, bare).components

    assert (change.add_patches, change.remove_patches) == ((reworked,), (MOE_PATCH.path, ATTN_PATCH.path))
    assert apply(stacked, delta_between(stacked, other)).component("vllm").patches == (reworked,)
    assert apply(stacked, delta_between(stacked, bare)).component("vllm").patches == ()
    assert dropped.upstream_ref is None
    with pytest.raises(RecipeError, match="comes from"):
        delta_between(stacked, _recipe(components=(ComponentPatches("vllm", "https://fork", VLLM_AT),)))


def test_apply_removes_patches_and_kernel_artifacts() -> None:
    delta = RecipeDelta(
        components=(ComponentChange("vllm", remove_patches=(MOE_PATCH.path,)),), remove_kernels=(GEMM.ref,)
    )

    changed = apply(_recipe(), delta)

    assert changed.component("vllm").patches == ()
    assert changed.kernels == ()


@pytest.mark.parametrize(
    ("delta", "message"),
    [
        (RecipeDelta(components=(ComponentChange("vllm", remove_patches=(ATTN_PATCH.path,)),)), "no patches"),
        (RecipeDelta(components=(ComponentChange("vllm", add_patches=(MOE_PATCH,)),)), "already has patches"),
        (RecipeDelta(components=(ComponentChange("sglang", upstream_ref=VLLM_AT),)), "new component"),
        (RecipeDelta(components=(ComponentChange("vllm", upstream_repo="https://fork"),)), "already comes from"),
        (RecipeDelta(remove_kernels=("kernel/gemm/missing.csv",)), "no kernel artifacts"),
    ],
)
def test_apply_refuses_a_change_that_does_not_fit_the_recipe(delta: RecipeDelta, message: str) -> None:
    with pytest.raises(RecipeError, match=message):
        apply(_recipe(), delta)


def test_a_delta_document_keeps_the_recipe_delta_shape_and_its_fingerprint_names_the_change() -> None:
    delta = RecipeDelta(
        config=ConfigDelta(server_args="--max-num-seqs 256", args_mode=ArgsMode.REPLACE, envs={"B": "2", "A": "1"}),
        components=(ComponentChange("vllm", add_patches=(ATTN_PATCH,)),),
    )
    same = RecipeDelta(
        config=ConfigDelta(server_args="--max-num-seqs 256", args_mode=ArgsMode.REPLACE, envs={"A": "1", "B": "2"}),
        components=(ComponentChange("vllm", add_patches=(ATTN_PATCH,)),),
    )

    document = delta.to_document()

    assert set(document["config"]) == {"extra_server_args", "args_mode", "remove_args", "extra_envs", "unset_envs"}
    assert RecipeDelta.from_document(document) == delta
    assert delta.fingerprint() == same.fingerprint()
    assert delta.fingerprint() != RecipeDelta().fingerprint()


@pytest.mark.parametrize(
    "build",
    [
        lambda: ConfigDelta(envs={"A": "1"}, unset_envs=("A",)),
        lambda: ConfigDelta(remove_args=("max-num-seqs",)),
        lambda: ConfigDelta.from_document({"args_mode": "merge"}),
        lambda: KernelArtifact.from_document({**GEMM.to_document(), "kind": "attention"}),
        lambda: KernelArtifact(KernelKind.GEMM, "t", "kernel/x.bin", "e" * 64, {"bad": {1, 2}}),
        lambda: RecipeDelta(components=(ComponentChange("vllm"), ComponentChange("vllm"))),
        lambda: Recipe.from_document(
            {"schema_version": RECIPE_SCHEMA_VERSION, "base_image_digest": IMAGE, "kernels": "x"}
        ),
    ],
)
def test_malformed_deltas_and_documents_are_refused(build: Callable[[], object]) -> None:
    with pytest.raises(RecipeError):
        build()


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            {"config": {"server_args": {"enable-torch-compile": True}}},
            r"unknown key 'server_args' in a config delta; known: extra_server_args, args_mode, remove_args, "
            r"extra_envs, unset_envs",
        ),
        ({"configs": {}}, r"unknown key 'configs' in a recipe delta; known: config, components,"),
        ({"components": [{"component": "vllm", "add_patch": []}]}, r"unknown key 'add_patch' in a component change"),
        (
            {"components": [{"component": "vllm", "add_patches": [{**ATTN_PATCH.to_document(), "hash": "x"}]}]},
            r"unknown key 'hash' in a patch",
        ),
        ({"add_kernels": [{**GEMM.to_document(), "speedup": 1.2}]}, r"unknown key 'speedup' in a kernel artifact"),
    ],
)
def test_a_delta_document_with_a_key_its_part_does_not_know_is_refused(document: object, message: str) -> None:
    with pytest.raises(RecipeError, match=message):
        RecipeDelta.from_document(document)


def test_every_part_of_a_delta_document_round_trips() -> None:
    delta = RecipeDelta(
        config=ConfigDelta(
            server_args="--max-num-seqs 256", remove_args=("--enable-chunked-prefill",), envs={"A": "1"}
        ),
        components=(ComponentChange("vllm", VLLM_NEXT, add_patches=(ATTN_PATCH,), remove_patches=(MOE_PATCH.path,)),),
        add_kernels=(GEMM,),
        remove_kernels=("kernel/gemm/old.csv",),
    )

    assert RecipeDelta.from_document(delta.to_document()) == delta


def test_diff_names_config_conflicts_moved_refs_and_patch_overlap() -> None:
    first = _recipe()
    second = apply(
        first,
        RecipeDelta(
            config=ConfigDelta(server_args="--max-num-seqs 256 --kv-cache-dtype fp8", unset_envs=("HIP_FORCE",)),
            components=(ComponentChange("vllm", upstream_ref=VLLM_NEXT, add_patches=(ATTN_PATCH,)),),
        ),
    )

    changes = diff(first, second)

    assert {(change.name, change.before, change.after) for change in changes.args} == {
        ("--max-num-seqs", "128", "256"),
        ("--kv-cache-dtype", None, "fp8"),
    }
    assert [change.name for change in changes.config_conflicts] == ["--max-num-seqs"]
    assert [(change.name, change.before) for change in changes.envs] == [("HIP_FORCE", "0")]
    assert [(change.component, change.before, change.after) for change in changes.refs] == [
        ("vllm", VLLM_AT, VLLM_NEXT)
    ]
    assert changes.shared_patches == {MOE_PATCH.sha256}
    assert changes.patches_only_in_second == {ATTN_PATCH.sha256}
    assert changes.patch_overlap == pytest.approx(0.5)
    assert not changes.base_image_changed


def test_identical_recipes_differ_in_nothing_and_a_new_base_image_is_a_difference() -> None:
    recipe = _recipe()

    assert diff(recipe, recipe).is_empty
    assert diff(recipe, recipe).patch_overlap == 1.0
    assert diff(recipe, _recipe(runtime_snapshot=None)).base_image_changed
    assert diff(_recipe(components=()), _recipe(components=())).patch_overlap == 1.0


def test_a_recipe_that_names_nothing_unexpressed_keeps_its_document_and_digest() -> None:
    recipe = _recipe()

    assert recipe.complete
    assert "unexpressed" not in recipe.to_document()
    assert Recipe.from_document({**recipe.to_document(), "unexpressed": []}).digest() == recipe.digest()


def test_what_a_recipe_leaves_out_travels_in_its_document_digest_and_every_change_made_to_it() -> None:
    recipe = with_unexpressed(_recipe(), ["setup_commands", "patch:site-packages/vllm/x.py", "setup_commands"])
    changed = apply(recipe, RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 256")))

    assert not recipe.complete
    assert recipe.unexpressed == ("patch:site-packages/vllm/x.py", "setup_commands")
    assert Recipe.from_document(recipe.to_document()) == recipe
    assert recipe.digest() != _recipe().digest()
    assert changed.unexpressed == recipe.unexpressed


def test_a_merge_keeps_what_either_side_left_out_and_diff_names_the_difference() -> None:
    ours = with_unexpressed(_recipe(), ["args_mode"])
    theirs = with_unexpressed(_recipe(), ["unset_envs:HIP_FORCE"])

    merged = with_unexpressed(ours, theirs.unexpressed)
    changes = diff(_recipe(), merged)

    assert merged.unexpressed == ("args_mode", "unset_envs:HIP_FORCE")
    assert changes.unexpressed_only_in_second == {"args_mode", "unset_envs:HIP_FORCE"}
    assert not changes.unexpressed_only_in_first and not changes.is_empty


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (("pip_install",), "kind one of"),
        (("patch: spaced",), "kind one of"),
        (("setup_commands", "setup_commands"), "duplicate unexpressed"),
        (("setup_commands", "args_mode"), "sorted"),
    ],
)
def test_an_unexpressed_entry_of_no_known_kind_or_out_of_order_is_refused(
    entries: tuple[str, ...], message: str
) -> None:
    with pytest.raises(RecipeError, match=message):
        _recipe(unexpressed=entries)


#: An option of the launch the framework resolves (a Magpie AgentX recipe's), which a recipe's own arguments lack.
RESOLVED_OPTION = "--enable-linear-replayssm-spec"


def test_a_removal_of_an_option_the_recipe_s_own_arguments_lack_is_remembered_until_one_sets_it_again() -> None:
    root = _recipe(launch=LaunchConfig("--max-num-seqs 128"))

    off = apply(root, RecipeDelta(config=ConfigDelta(remove_args=(RESOLVED_OPTION,))))
    own = apply(root, RecipeDelta(config=ConfigDelta(remove_args=("--max-num-seqs",))))
    later = apply(off, RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 64")))
    back = apply(off, RecipeDelta(config=ConfigDelta(server_args=RESOLVED_OPTION)))

    assert (off.launch.server_args, off.launch.remove_args) == ("--max-num-seqs 128", (RESOLVED_OPTION,))
    assert (own.launch.server_args, own.launch.remove_args) == ("", ())
    assert later.launch.remove_args == (RESOLVED_OPTION,)
    assert back.launch.remove_args == ()


def test_a_launch_with_no_removals_keeps_its_document_and_one_with_them_reads_back() -> None:
    plain = _recipe()
    removing = _recipe(launch=replace(plain.launch, remove_args=("--a", "--b")))

    assert "remove_args" not in plain.launch.to_document()
    assert LaunchConfig.from_document(removing.launch.to_document()) == removing.launch
    assert removing.digest() != plain.digest()


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: LaunchConfig(remove_args=("--a=1",)), "bare option names"),
        (lambda: LaunchConfig(remove_args=("--a 1",)), "bare option names"),
        (lambda: LaunchConfig(remove_args=("a",)), "bare option names"),
        (lambda: LaunchConfig(remove_args=("--b", "--a")), "sorted and unique"),
        (lambda: LaunchConfig(remove_args=("--a", "--a")), "sorted and unique"),
        (lambda: LaunchConfig("--a 1 --b", remove_args=("--b",)), "both sets and removes --b"),
        (lambda: LaunchConfig.from_document({"extra_server_args": "", "remove": ["--a"]}), "unknown key 'remove'"),
        (lambda: ConfigDelta(launch_remove_args=("--b", "--a")), "sorted and unique"),
    ],
)
def test_a_launch_s_removals_have_one_canonical_form(build: Callable[[], object], message: str) -> None:
    with pytest.raises(RecipeError, match=message):
        build()


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(LaunchConfig("--a 1"), LaunchConfig("--a 1", remove_args=("--r",)), id="a new removal"),
        pytest.param(LaunchConfig("--a 1", remove_args=("--r",)), LaunchConfig("--a 1"), id="a removal undone"),
        pytest.param(LaunchConfig(remove_args=("--r",)), LaunchConfig("--r 2"), id="a removed option set again"),
        pytest.param(LaunchConfig("--r 2"), LaunchConfig(remove_args=("--r",)), id="an own option removed resolved"),
        pytest.param(
            LaunchConfig("--a 1", remove_args=("--q", "--r")),
            LaunchConfig("--b", remove_args=("--r", "--s")),
            id="removals both added and undone",
        ),
    ],
)
def test_delta_between_makes_one_launch_s_removals_into_the_other_s(before: LaunchConfig, after: LaunchConfig) -> None:
    first = _recipe(launch=before)
    second = _recipe(launch=after)

    delta = delta_between(first, second)
    made = apply(first, delta)

    assert made.launch == second.launch
    assert made.digest() == second.digest()
    assert RecipeDelta.from_document(delta.to_document()) == delta


def test_the_delta_between_two_recipes_carries_a_new_removal_and_the_diff_names_it() -> None:
    first = _recipe(launch=LaunchConfig("--max-num-seqs 128"))
    second = apply(first, RecipeDelta(config=ConfigDelta(remove_args=(RESOLVED_OPTION,))))

    delta = delta_between(first, second)

    assert (delta.config.remove_args, delta.config.launch_remove_args) == ((RESOLVED_OPTION,), None)
    assert "launch_remove_args" not in delta.config.to_document()
    assert apply(first, delta).launch == second.launch
    assert [(change.name, change.before, change.after) for change in diff(first, second).args] == [
        (RESOLVED_OPTION, None, REMOVED)
    ]
    assert not diff(first, second).is_empty


def test_a_delta_that_stops_removing_an_option_gives_the_launch_s_removals_whole() -> None:
    first = _recipe(launch=LaunchConfig("--max-num-seqs 128", remove_args=(RESOLVED_OPTION,)))
    second = _recipe(launch=LaunchConfig("--max-num-seqs 128"))

    delta = delta_between(first, second)

    assert delta.config.launch_remove_args == ()
    assert delta.config.to_document()["launch_remove_args"] == []
    assert [(change.name, change.before, change.after) for change in diff(first, second).args] == [
        (RESOLVED_OPTION, REMOVED, None)
    ]
