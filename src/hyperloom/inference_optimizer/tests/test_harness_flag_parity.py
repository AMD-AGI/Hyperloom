# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A variant is measured against its reference only when both run the same benchmark harness."""

from __future__ import annotations

import pytest

from hyperloom.common.failure_signature import SERVE_FLAG
from hyperloom.inference_optimizer.grid_server_args import (
    HarnessFlagChangeError,
    harness_flag_change_reason,
    refuse_harness_flag_change,
)
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors._grid_variant_filter import ServingStack, apply_compatibility_filter

NO_PC = "--no-enable-prefix-caching"
VLLM = ServingStack("vllm", "")


def _filter(variant: GridVariant, *, base: str) -> tuple[list[GridVariant], list[dict]]:
    return apply_compatibility_filter([variant], serving=VLLM, base_server_args=base)


@pytest.mark.parametrize(
    ("base", "variant"),
    [
        pytest.param(
            "--max-num-seqs 256",
            GridVariant("mnbt_16k_no_prefix_cache", f"--max-num-batched-tokens 16384 {NO_PC}"),
            id="variant-turns-prefix-caching-off",
        ),
        pytest.param(
            f"--max-num-seqs 256 {NO_PC}",
            GridVariant("prefix_cache_back_on", "--max-num-batched-tokens 16384", remove_args=[NO_PC]),
            id="variant-drops-the-operators-harness-flag",
        ),
        pytest.param(
            f"--max-num-seqs 256 {NO_PC}",
            GridVariant("snapshot_without_harness", "--max-num-seqs 512", remove_args=[NO_PC], args_mode="replace"),
            id="replace-snapshot-removing-the-operators-harness-flag",
        ),
    ],
)
def test_a_variant_running_another_harness_than_its_reference_is_refused(base, variant):
    kept, dropped = _filter(variant, base=base)

    assert kept == []
    assert [d["name"] for d in dropped] == [variant.name]
    assert dropped[0]["source"] == SERVE_FLAG
    assert NO_PC in dropped[0]["reason"]


@pytest.mark.parametrize(
    ("base", "variant"),
    [
        pytest.param(
            "--max-num-seqs 256",
            GridVariant("mnbt_16k", "--max-num-batched-tokens 16384"),
            id="neither-runs-the-harness-flag",
        ),
        pytest.param(
            f"--max-num-seqs 256 {NO_PC}",
            GridVariant("mnbt_16k", "--max-num-batched-tokens 16384"),
            id="variant-inherits-the-operators-harness-flag",
        ),
        pytest.param(
            f"--max-num-seqs 256 {NO_PC}",
            GridVariant("restated", f"--max-num-batched-tokens 16384 {NO_PC}"),
            id="variant-restates-the-operators-harness-flag",
        ),
        pytest.param(
            f"--max-num-seqs 256 {NO_PC}",
            GridVariant("snapshot", f"--max-num-seqs 512 {NO_PC}", args_mode="replace"),
            id="replace-snapshot-keeping-the-operators-harness-flag",
        ),
        pytest.param(
            f"--max-num-seqs 256 {NO_PC}",
            GridVariant("snapshot", "--max-num-seqs 512", args_mode="replace"),
            id="replace-snapshot-silent-on-the-harness-flag-still-launches-with-it",
        ),
    ],
)
def test_a_variant_running_its_references_harness_is_kept(base, variant):
    kept, dropped = _filter(variant, base=base)

    assert kept == [variant]
    assert dropped == []


@pytest.mark.parametrize(
    ("reference", "candidate", "differs"),
    [
        pytest.param("", "", False, id="both-empty"),
        pytest.param(None, NO_PC, True, id="candidate-only"),
        pytest.param(f"--max-num-seqs 64 {NO_PC}", "--max-num-seqs 64", True, id="reference-only"),
        pytest.param(f"{NO_PC} --max-num-seqs 64", f"--max-num-seqs 128 {NO_PC}", False, id="both-any-position"),
    ],
)
def test_a_harness_flag_exactly_one_launch_carries_is_refused_either_way(reference, candidate, differs):
    for ours, theirs in ((reference, candidate), (candidate, reference)):
        reason = harness_flag_change_reason(ours, theirs)
        assert (reason is not None) is differs
        if differs:
            assert NO_PC in reason
            with pytest.raises(HarnessFlagChangeError, match=NO_PC) as refused:
                refuse_harness_flag_change(ours, theirs)
            assert refused.value.error_class == SERVE_FLAG
        else:
            refuse_harness_flag_change(ours, theirs)


@pytest.mark.parametrize(
    ("reference", "candidate", "differs"),
    [
        pytest.param(NO_PC, f"{NO_PC} --enable-prefix-caching", True, id="candidate-re-enables-after-the-negative"),
        pytest.param("--enable-prefix-caching", f"--enable-prefix-caching {NO_PC}", True, id="candidate-disables-last"),
        pytest.param(NO_PC, f"--enable-prefix-caching {NO_PC}", False, id="negative-last-on-both"),
        pytest.param("", "--no_enable_prefix_caching", True, id="underscore-spelling"),
        pytest.param("", "--enable-prefix-caching", False, id="positive-is-the-default"),
    ],
)
def test_the_spelling_the_server_reads_last_decides_a_harness_flag(reference, candidate, differs):
    assert (harness_flag_change_reason(reference, candidate) is not None) is differs


@pytest.mark.parametrize(
    ("base", "variant"),
    [
        pytest.param(
            "--max-num-seqs 256",
            GridVariant("env_sets_carrier", "", extra_envs={"EXTRA_VLLM_ARGS": f"--max-num-seqs 256 {NO_PC}"}),
            id="variant-env-sets-the-args-carrier",
        ),
        pytest.param(
            f"--max-num-seqs 256 {NO_PC}",
            GridVariant("env_unsets_carrier", "--max-num-batched-tokens 16384", unset_envs=["EXTRA_VLLM_ARGS"]),
            id="variant-env-unsets-the-args-carrier",
        ),
        pytest.param(
            f"--max-num-seqs 256 {NO_PC}",
            GridVariant("re_enables", "--enable-prefix-caching"),
            id="variant-appends-the-positive-spelling",
        ),
    ],
)
def test_the_filter_reads_the_harness_the_launch_will_run(base, variant):
    kept, dropped = _filter(variant, base=base)

    assert kept == []
    assert dropped[0]["source"] == SERVE_FLAG and NO_PC in dropped[0]["reason"]


def _vllm_recipe(tmp_path, operator_args: str):
    import yaml

    from hyperloom.orchestrator.actions.executors._grid_runner import GridRecipe

    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        yaml.safe_dump({"benchmark": {"framework": "vllm", "envs": {"EXTRA_VLLM_ARGS": operator_args}}}),
        encoding="utf-8",
    )
    return GridRecipe(base_yaml_path=base_yaml)


def test_the_launch_refuses_a_variant_that_adds_a_harness_flag(tmp_path):
    """The launch is the last point every path crosses, integrate_patch included, which skips the filter."""
    from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

    variant = GridVariant("patch_no_prefix_cache", f"--max-num-batched-tokens 16384 {NO_PC}")
    with pytest.raises(HarnessFlagChangeError, match=NO_PC) as refused:
        _build_variant_yaml(_vllm_recipe(tmp_path, "--max-num-seqs 256"), variant, output_subdir=tmp_path / "v")
    assert refused.value.error_class == SERVE_FLAG


def test_the_launch_keeps_a_variant_inheriting_the_operators_harness_flag(tmp_path):
    import yaml

    from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

    variant = GridVariant("mnbt_16k", "--max-num-batched-tokens 16384")
    path = _build_variant_yaml(
        _vllm_recipe(tmp_path, f"--max-num-seqs 256 {NO_PC}"), variant, output_subdir=tmp_path / "v"
    )
    assert NO_PC in yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]


def test_an_agentx_candidate_adding_a_harness_flag_is_refused():
    from hyperloom.orchestrator.actions.executors.agentx_point import LaunchChanges, apply_launch_changes

    benchmark = {"agentx": {"enabled": True, "launch_overrides": {"version": 1, "append_args": ["--max-num-seqs=256"]}}}
    with pytest.raises(HarnessFlagChangeError, match=NO_PC):
        apply_launch_changes(benchmark, LaunchChanges(args=NO_PC))


def test_the_launch_refuses_a_harness_flag_set_through_the_variants_env(tmp_path):
    from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

    variant = GridVariant("env_sets_carrier", "", extra_envs={"EXTRA_VLLM_ARGS": f"--max-num-seqs 256 {NO_PC}"})
    with pytest.raises(HarnessFlagChangeError, match=NO_PC):
        _build_variant_yaml(_vllm_recipe(tmp_path, "--max-num-seqs 256"), variant, output_subdir=tmp_path / "v")


def test_a_repeated_harness_flag_in_the_composed_args_is_read_where_the_launch_dedupe_keeps_it(tmp_path):
    """The launch dedupes composed args to each flag's first position, so appending the negative again changes nothing."""
    from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

    reference = f"{NO_PC} --enable-prefix-caching"
    variant = GridVariant("restates_negative", f"--max-num-batched-tokens 16384 {NO_PC}")

    kept, dropped = _filter(variant, base=reference)
    assert kept == [variant] and dropped == []
    _build_variant_yaml(_vllm_recipe(tmp_path, reference), variant, output_subdir=tmp_path / "v")


def test_an_env_override_of_the_args_carrier_is_read_undeduped(tmp_path):
    """An env layer replaces the carrier after the launch's dedupe, so the server reads its last spelling as written."""
    from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

    carrier = f"{NO_PC} --enable-prefix-caching {NO_PC}"
    variant = GridVariant("env_repeats_negative", "", extra_envs={"EXTRA_VLLM_ARGS": carrier})

    kept, dropped = _filter(variant, base="")
    assert kept == [] and dropped[0]["source"] == SERVE_FLAG
    with pytest.raises(HarnessFlagChangeError, match=NO_PC):
        _build_variant_yaml(_vllm_recipe(tmp_path, ""), variant, output_subdir=tmp_path / "v")
