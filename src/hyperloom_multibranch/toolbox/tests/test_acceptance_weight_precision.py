# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance refuses a candidate whose launch changes the model's weight precision from the session's root recipe,
before anything compiles or queues; intermediates (the KV cache, attention, fusions, env vars) may change."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose
from hyperloom.contracts.recipe import ConfigDelta, LaunchConfig, Recipe, RecipeDelta
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, RequestRejected
from hyperloom_multibranch.toolbox.acceptance import DELTA_NOT_APPLICABLE_CODE, Acceptance, AcceptanceSetup
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakePlanner, FakeRepo, build, req
from hyperloom_multibranch.toolbox.weight_precision import weight_precision_changes

ROOT = "r0"
SAME_PRECISION = "the weight precision of the session cannot change"


def _recipe(server_args: str = "", **launch: object) -> Recipe:
    return Recipe(base_image_digest=IMAGE, launch=LaunchConfig(server_args=server_args, **launch))  # type: ignore[arg-type]


def _repo(root_args: str = "", parent_args: str | None = None) -> FakeRepo:
    """A repository whose root holds ``root_args`` and whose parent ``PARENT`` holds ``parent_args`` (the root's when
    ``None``)."""
    repo = FakeRepo(_recipe(root_args if parent_args is None else parent_args))
    repo.recipes[ROOT] = _recipe(root_args)
    repo.root_commit = ROOT
    return repo


def _delta(server_args: str = "", **config: object) -> RecipeDelta:
    return RecipeDelta(config=ConfigDelta(server_args=server_args, **config))  # type: ignore[arg-type]


async def _refusal(tmp_path: Path, repo: FakeRepo, change: RecipeDelta, **fields: object) -> RequestRejected | None:
    """The refusal of a request carrying ``change`` on ``PARENT``, ``None`` when it is accepted."""
    rig = build(tmp_path, repo=repo)
    try:
        await rig.toolbox.submit_test(req(change=change, **fields))
    except RequestRejected as refused:
        assert rig.planner.calls == [], "a refused request must not reach the compiler"
        assert rig.blackboard._state().requests == {}, "a refused request must write nothing to the blackboard"
        assert rig.toolbox._log.entries() == [], "a refused request must not queue"
        return refused
    finally:
        await rig.toolbox.close()
    return None


async def _refused_naming(tmp_path: Path, repo: FakeRepo, change: RecipeDelta, *changes: str) -> None:
    refused = await _refusal(tmp_path, repo, change)
    assert refused is not None, "a change of the weight precision was accepted"
    assert refused.code == DELTA_NOT_APPLICABLE_CODE
    assert SAME_PRECISION in str(refused)
    assert f"commit {ROOT}" in str(refused)
    for change_named in changes:
        assert change_named in str(refused)


def test_the_refusal_code_is_a_documented_rejection_code() -> None:
    assert DELTA_NOT_APPLICABLE_CODE in REJECTION_CODES


# --- each setting: added, changed, removed ---------------------------------------------------------------------------

#: One spelling of each setting with two values, and the name a change of it is reported under.
SETTINGS = [
    ("--quantization", "fp8", "fp8_per_channel", "--quantization"),
    ("-q", "fp8", "fp8_per_channel", "--quantization"),
    ("--quantization-config", '{"a":1}', '{"a":2}', "--quantization-config"),
    ("--quantization-param-path", "/scales/a.json", "/scales/b.json", "--quantization-param-path"),
    ("--online-quant-config", "/quant/a.yaml", "/quant/b.yaml", "--online-quant-config"),
    ("--online_quant_config", "/quant/a.yaml", "/quant/b.yaml", "--online-quant-config"),
    ("--dtype", "bfloat16", "float16", "--dtype"),
    ("--model", "/models/bf16", "/models/fp8", "the checkpoint"),
    ("--model-path", "/models/bf16", "/models/fp8", "the checkpoint"),
]


@pytest.mark.parametrize(("flag", "old", "new", "name"), SETTINGS)
async def test_a_candidate_that_adds_a_weight_precision_flag_is_refused(
    tmp_path: Path, flag: str, old: str, new: str, name: str
) -> None:
    await _refused_naming(tmp_path, _repo(), _delta(f"{flag} {new}"), f"{name}: (not set) -> {new}")


@pytest.mark.parametrize(("flag", "old", "new", "name"), SETTINGS)
async def test_a_candidate_that_changes_a_weight_precision_flag_is_refused(
    tmp_path: Path, flag: str, old: str, new: str, name: str
) -> None:
    await _refused_naming(tmp_path, _repo(f"{flag} {old}"), _delta(f"{flag} {new}"), f"{name}: {old} -> ", new)


@pytest.mark.parametrize(("flag", "old", "new", "name"), [s for s in SETTINGS if s[0].startswith("--")])
async def test_a_candidate_that_removes_a_weight_precision_flag_is_refused(
    tmp_path: Path, flag: str, old: str, new: str, name: str
) -> None:
    removal = _delta("--max-num-seqs 64", remove_args=(flag,))
    await _refused_naming(tmp_path, _repo(f"{flag} {old}"), removal, f"{name}: {old} -> (not set)")


async def test_a_candidate_that_removes_a_weight_flag_from_the_resolved_launch_is_refused(tmp_path: Path) -> None:
    removal = _delta("--max-num-seqs 64", remove_args=("--dtype",))
    await _refused_naming(tmp_path, _repo(), removal, "--dtype: (not set) -> <removed>")


async def test_every_changed_setting_is_named_with_its_old_and_new_value(tmp_path: Path) -> None:
    change = _delta("--quantization fp8 --dtype float16 --kv-cache-dtype fp8")
    await _refused_naming(
        tmp_path, _repo("--dtype bfloat16"), change, "--dtype: bfloat16 -> float16", "--quantization: (not set) -> fp8"
    )


# --- what is not a change ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("root_args", "server_args"),
    [
        ("--quantization fp8", "--quantization=fp8 --max-num-seqs 64"),
        ("--quantization=fp8", "--quantization fp8 --max-num-seqs 64"),
        ("-q fp8", "--quantization fp8"),
        ("--online_quant_config /quant/a.yaml", "--online-quant-config /quant/a.yaml"),
        ("--model /models/bf16", "--model-path /models/bf16"),
        ("--dtype bfloat16 --dtype bfloat16", "--max-num-seqs 64"),
        ('--quantization-config {"b":1,"a":[2,3]}', '--quantization-config \'{"a": [2, 3], "b": 1}\''),
        ("--quantization fp8 --dtype bfloat16", "--max-num-seqs 64"),
        ("", "--kv-cache-dtype fp8"),
        ("", "--attention-backend TRITON_ATTN"),
        ("--kv-cache-dtype auto", "--kv-cache-dtype fp8_e4m3"),
    ],
)
async def test_a_candidate_that_keeps_the_root_weight_precision_is_accepted(
    tmp_path: Path, root_args: str, server_args: str
) -> None:
    assert await _refusal(tmp_path, _repo(root_args), _delta(server_args)) is None


async def test_an_alias_swapped_on_the_way_from_the_root_is_not_a_change(tmp_path: Path) -> None:
    """The parent spells the root's setting another way; the candidate inherits that spelling."""
    repo = _repo("-q fp8 --online_quant_config /quant/a.yaml", "--quantization fp8 --online-quant-config /quant/a.yaml")
    assert await _refusal(tmp_path, repo, _delta("--max-num-seqs 64")) is None


async def test_env_vars_are_not_weight_precision(tmp_path: Path) -> None:
    change = _delta("--max-num-seqs 64", envs={"VLLM_ROCM_USE_AITER_FP8BMM": "1", "HSA_NO_SCRATCH_RECLAIM": "1"})
    assert await _refusal(tmp_path, _repo(), change) is None


async def test_a_quantization_config_given_another_value_is_refused(tmp_path: Path) -> None:
    root = '--quantization-config {"b":1,"a":[2,3]}'
    change = _delta('--quantization-config \'{"a": [2, 4], "b": 1}\'')
    await _refused_naming(
        tmp_path, _repo(root), change, '--quantization-config: {"a":[2,3],"b":1} -> {"a":[2,4],"b":1}'
    )


async def test_the_root_baseline_is_accepted_whatever_its_launch_repeats(tmp_path: Path) -> None:
    """The root gives ``--dtype`` twice; its own baseline test composes the launch as every candidate does."""
    repo = _repo("--dtype float16 --dtype bfloat16 --quantization fp8")
    assert await _refusal(tmp_path, repo, RecipeDelta(), purpose=Purpose.BASELINE) is None


# --- the root, not the parent --------------------------------------------------------------------------------------


async def test_a_parent_that_carries_a_change_does_not_pass_it_on(tmp_path: Path) -> None:
    repo = _repo("--dtype bfloat16", parent_args="--dtype bfloat16 --quantization fp8_per_channel")
    await _refused_naming(tmp_path, repo, _delta("--max-num-seqs 64"), "--quantization: (not set) -> fp8_per_channel")


async def test_a_candidate_that_restores_the_root_precision_over_its_parent_is_accepted(tmp_path: Path) -> None:
    repo = _repo("--dtype bfloat16", parent_args="--dtype bfloat16 --quantization fp8_per_channel")
    restore = _delta("--max-num-seqs 64", remove_args=("--quantization",))
    assert await _refusal(tmp_path, repo, restore) is None


@pytest.mark.parametrize("purpose", [Purpose.INTEGRATION, Purpose.FINAL])
async def test_a_rerun_of_a_parent_that_carries_a_change_is_refused(tmp_path: Path, purpose: Purpose) -> None:
    repo = _repo("", parent_args="--quantization fp8")
    refused = await _refusal(tmp_path, repo, RecipeDelta(), purpose=purpose)
    assert refused is not None and refused.code == DELTA_NOT_APPLICABLE_CODE


async def test_a_request_an_earlier_toolbox_accepted_is_not_withdrawn_by_the_rule() -> None:
    acceptance = Acceptance(AcceptanceSetup(_repo(), FakePlanner(), IMAGE), is_accepted=lambda _: True)
    request = req(change=_delta("--quantization fp8"))
    await acceptance.prepare(dataclasses.replace(request, request_id="r1"), replay=True)
    with pytest.raises(RequestRejected) as refused:
        await acceptance.prepare(request)
    assert refused.value.code == DELTA_NOT_APPLICABLE_CODE


# --- the comparison itself -----------------------------------------------------------------------------------------


def _changes(before: str, after: str) -> list[str]:
    return weight_precision_changes(LaunchConfig(server_args=before), LaunchConfig(server_args=after))


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("--quantization=fp8", "--quantization fp8", []),
        ("-q fp8", "--quantization fp8", []),
        ("-q fp8 --quantization fp8", "--quantization fp8", []),
        ("--dtype bfloat16 --dtype bfloat16", "--dtype bfloat16", []),
        ('--quantization-config {"b":1,"a":2}', '--quantization-config {"a":2,"b":1}', []),
        ("--max-num-seqs 8 --kv-cache-dtype fp8", "--kv-cache-dtype auto", []),
        ("", "--quantization", ["--quantization: (not set) -> (on)"]),
        ("--dtype bfloat16", "--dtype bfloat16 --dtype float16", ["--dtype: bfloat16 -> bfloat16 | float16"]),
        ("--model /a", "--model-path /b", ["the checkpoint: /a -> /b"]),
        ("--quantization-config a.json", "--quantization-config b.json", ["--quantization-config: a.json -> b.json"]),
        ("--quantization-config [1,2]", "--quantization-config [2,1]", ["--quantization-config: [1,2] -> [2,1]"]),
        ("--dtype -1", "--dtype 1", ["--dtype: -1 -> 1"]),
    ],
)
def test_the_comparison_reads_flags_as_arbor_does(before: str, after: str, expected: list[str]) -> None:
    assert _changes(before, after) == expected


def test_a_removal_from_the_resolved_launch_is_a_value_of_its_own() -> None:
    removed = LaunchConfig(remove_args=("--quantization",))
    assert weight_precision_changes(LaunchConfig(), removed) == ["--quantization: (not set) -> <removed>"]
    assert weight_precision_changes(removed, removed) == []
