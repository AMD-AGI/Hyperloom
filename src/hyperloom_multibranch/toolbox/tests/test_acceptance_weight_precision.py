# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance refuses a candidate whose launch changes the model's weight precision from the session's root recipe,
before anything compiles or queues; intermediates (the KV cache, attention, fusions, env vars) may change."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose
from hyperloom.contracts.recipe import ConfigDelta, LaunchConfig, Recipe, RecipeDelta, apply
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, RequestRejected
from hyperloom_multibranch.toolbox.acceptance import DELTA_NOT_APPLICABLE_CODE, Acceptance, AcceptanceSetup
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, PARENT, FakePlanner, FakeRepo, build, req
from hyperloom_multibranch.toolbox.weight_precision import (
    SPEED_ONLY_LOAD_FORMATS,
    ambiguous_words,
    weight_precision_changes,
)

ROOT = "r0"
SAME_PRECISION = "the weight precision of the session cannot change"


def _recipe(server_args: str = "", **launch: object) -> Recipe:
    launch_config = LaunchConfig(server_args=server_args, **launch)  # type: ignore[arg-type]
    return Recipe(base_image_digest=IMAGE, launch=launch_config)


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
    ("--revision", "abc123", "def456", "the checkpoint's revision"),
    ("--config", "/cfg/a.yaml", "/cfg/b.yaml", "--config"),
    ("--modelopt-quant", "fp8", "nvfp4", "--modelopt-quant"),
    ("--torchao-config", "int4wo-128", "int8wo", "--torchao-config"),
    ("--quantize-and-serve", "true", "false", "--quantize-and-serve"),
    ("--hf-config-path", "/cfg/a", "/cfg/b", "--hf-config-path"),
    ("--decrypted-config-file", "/cfg/a.json", "/cfg/b.json", "--decrypted-config-file"),
    ("--load-format", "flash_rl", "bitsandbytes", "--load-format"),
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


# --- a flag in every form its framework reads as it ------------------------------------------------------------------

#: A word the frameworks read as a weight-precision flag without spelling it, on a root that sets nothing, and the
#: change it is refused for.
OTHER_FORMS = [
    # vLLM merges ``--flag.key value`` into ``--flag``'s JSON value; its documentation quantizes linear weights so.
    ("--quantization-config.linear fp8_per_channel", "--quantization-config: (not set) -> .linear fp8_per_channel"),
    ("--quantization-config.linear=fp8_per_channel", "--quantization-config: (not set) -> .linear fp8_per_channel"),
    # vLLM reads ``_`` in a long flag's name as ``-``.
    ('--quantization_config {"linear":"fp8_per_channel"}', '--quantization-config: (not set) -> {"linear":'),
    ("--quantization_param_path /scales/x.json", "--quantization-param-path: (not set) -> /scales/x.json"),
    ("--model_path /models/fp8", "the checkpoint: (not set) -> /models/fp8"),
    ("--quantization_config.linear fp8_per_channel", "--quantization-config: (not set) -> .linear fp8_per_channel"),
    # argparse reads a value attached to a one-letter option as its value.
    ("-qfp8_per_channel", "--quantization: (not set) -> fp8_per_channel"),
    ("-q=fp8_per_channel", "--quantization: (not set) -> fp8_per_channel"),
    # argparse takes a prefix of an option for the option; neither framework turns that off.
    ("--dty float16", "--dtype: (not set) -> float16"),
    ("--dty=float16", "--dtype: (not set) -> float16"),
    ('--quantization-c {"linear":"fp8_per_channel"}', '--quantization-config: (not set) -> {"linear":'),
    ("--quantization-p /scales/x.json", "--quantization-param-path: (not set) -> /scales/x.json"),
    ("--model-pa /models/fp8", "the checkpoint: (not set) -> /models/fp8"),
    ("--online_q /quant/a.yaml", "--online-quant-config: (not set) -> /quant/a.yaml"),
    ("--quantization-c.linear fp8_per_channel", "--quantization-config: (not set) -> .linear fp8_per_channel"),
    # The same readings for the flags added with Arbor's copy of the list.
    ("--revision=abc123", "the checkpoint's revision: (not set) -> abc123"),
    ("--rev abc123", "the checkpoint's revision: (not set) -> abc123"),
    ("--revision.x abc123", "the checkpoint's revision: (not set) -> .x abc123"),
    ("--config=/cfg/b.yaml", "--config: (not set) -> /cfg/b.yaml"),
    ("--conf /cfg/b.yaml", "--config: (not set) -> /cfg/b.yaml"),
    ("--config.x /cfg/b.yaml", "--config: (not set) -> .x /cfg/b.yaml"),
    ("--modelopt-quant=fp8", "--modelopt-quant: (not set) -> fp8"),
    ("--modelopt_quant fp8", "--modelopt-quant: (not set) -> fp8"),
    ("--modelopt fp8", "--modelopt-quant: (not set) -> fp8"),
    ("--modelopt-quant.x fp8", "--modelopt-quant: (not set) -> .x fp8"),
    ("--torchao-config=int4wo-128", "--torchao-config: (not set) -> int4wo-128"),
    ("--torchao_config int4wo-128", "--torchao-config: (not set) -> int4wo-128"),
    ("--torchao int4wo-128", "--torchao-config: (not set) -> int4wo-128"),
    ("--torchao-config.x int4wo-128", "--torchao-config: (not set) -> .x int4wo-128"),
    ("--quantize-and-serve", "--quantize-and-serve: (not set) -> (on)"),
    ("--quantize-and-serve=true", "--quantize-and-serve: (not set) -> true"),
    ("--quantize_and_serve", "--quantize-and-serve: (not set) -> (on)"),
    ("--quantize", "--quantize-and-serve: (not set) -> (on)"),
    ("--quantize-and-serve.x 1", "--quantize-and-serve: (not set) -> .x 1"),
    # And for the files the model's config is read from.
    ("--hf-config-path=/cfg/b", "--hf-config-path: (not set) -> /cfg/b"),
    ("--hf_config_path /cfg/b", "--hf-config-path: (not set) -> /cfg/b"),
    ("--hf-config /cfg/b", "--hf-config-path: (not set) -> /cfg/b"),
    ("--hf-config-path.x /cfg/b", "--hf-config-path: (not set) -> .x /cfg/b"),
    ("--decrypted-config-file=/cfg/b.json", "--decrypted-config-file: (not set) -> /cfg/b.json"),
    ("--decrypted_config_file /cfg/b.json", "--decrypted-config-file: (not set) -> /cfg/b.json"),
    ("--decrypted-config /cfg/b.json", "--decrypted-config-file: (not set) -> /cfg/b.json"),
    ("--decrypted-config-file.x /cfg/b.json", "--decrypted-config-file: (not set) -> .x /cfg/b.json"),
]


@pytest.mark.parametrize(("server_args", "change"), OTHER_FORMS)
async def test_a_weight_precision_flag_in_another_form_its_framework_reads_is_refused(
    tmp_path: Path, server_args: str, change: str
) -> None:
    await _refused_naming(tmp_path, _repo(), _delta(server_args), change)


@pytest.mark.parametrize(
    ("root_args", "server_args", "change"),
    [
        # The last value given wins in argparse: these launches run ptpc_fp8 and float16, however the root's flag reads.
        ("--quantization fp8", "-qptpc_fp8", "--quantization: fp8 -> fp8 | ptpc_fp8"),
        ("--dtype bfloat16", "--dty float16", "--dtype: bfloat16 -> bfloat16 | float16"),
        (
            '--quantization-config {"linear":"fp8"}',
            "--quantization-config.linear mxfp8",
            '--quantization-config: {"linear":"fp8"} -> .linear mxfp8 | {"linear":"fp8"}',
        ),
        (
            "--quantization-config.linear fp8_per_channel",
            "--quantization-config.embedding fp8_per_channel",
            "--quantization-config: .linear fp8_per_channel -> .embedding fp8_per_channel | .linear fp8_per_channel",
        ),
    ],
)
async def test_another_form_of_a_flag_the_root_sets_is_refused(
    tmp_path: Path, root_args: str, server_args: str, change: str
) -> None:
    await _refused_naming(tmp_path, _repo(root_args), _delta(server_args), change)


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
        ("-qfp8", "--quantization fp8"),
        ('--quantization_config {"linear":"fp8"}', '--quantization-config {"linear":"fp8"}'),
        ("--dty bfloat16", "--dtype bfloat16"),
        ("--model_path /models/bf16", "--model /models/bf16"),
        ("--quantization-config.linear fp8", "--quantization-config.linear=fp8 --max-num-seqs 64"),
        ("--model /models/bf16", "--model-impl transformers --max-model-len 8192"),
        ("--model /models/bf16", '--model-loader-extra-config {"a":1}'),
        ("", "-tp 8 -O3 --max-num-seqs 64"),
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
        (
            "--dtype bfloat16",
            "--dtype bfloat16 --dtype float16",
            ["--dtype: bfloat16 -> bfloat16 | float16 (float16 in effect)"],
        ),
        ("--model /a", "--model-path /b", ["the checkpoint: /a -> /b"]),
        ("--quantization-config a.json", "--quantization-config b.json", ["--quantization-config: a.json -> b.json"]),
        ("--quantization-config [1,2]", "--quantization-config [2,1]", ["--quantization-config: [1,2] -> [2,1]"]),
        ("--dtype -1", "--dtype 1", ["--dtype: -1 -> 1"]),
        ("--quantization", "--quantization=", ["--quantization: (on) -> ''"]),
        # A prefix that several settings' spellings begin is each of them: argparse refuses it where the framework
        # has several of those options, and reads it as the one where it has one.
        (
            "",
            "--quantization- x",
            ["--quantization-config: (not set) -> x", "--quantization-param-path: (not set) -> x"],
        ),
        ("", "--mod /m", ["--modelopt-quant: (not set) -> /m", "the checkpoint: (not set) -> /m"]),
        ("", "-- --max-num-seqs 8", []),
    ],
)
def test_the_comparison_reads_flags_as_arbor_does(before: str, after: str, expected: list[str]) -> None:
    assert _changes(before, after) == expected


def test_a_removal_from_the_resolved_launch_is_a_value_of_its_own() -> None:
    removed = LaunchConfig(remove_args=("--quantization",))
    assert weight_precision_changes(LaunchConfig(), removed) == ["--quantization: (not set) -> <removed>"]
    assert weight_precision_changes(removed, removed) == []


def test_a_removal_is_read_in_every_form_of_its_flag() -> None:
    removed = LaunchConfig(remove_args=("--quantization_config",))
    assert weight_precision_changes(LaunchConfig(), removed) == ["--quantization-config: (not set) -> <removed>"]


# --- the override flags: what they set under a weight-precision key -------------------------------------------------

HF = "--hf-overrides (compression_config, dtype, quantization_config, torch_dtype)"
JSON_ARGS = "--json-model-override-args (compression_config, dtype, quantization_config, torch_dtype)"

#: A candidate launch that sets a weight-precision key through an override flag, on a root that sets none, and the
#: change it is refused for.
OVERRIDES_ADDED = [
    ('--hf-overrides \'{"torch_dtype": "float16"}\'', f'{HF}: (not set) -> torch_dtype="float16"'),
    ('--hf-overrides \'{"dtype": "float16"}\'', f'{HF}: (not set) -> dtype="float16"'),
    (
        '--hf-overrides \'{"quantization_config": {"quant_method": "fp8"}}\'',
        f'{HF}: (not set) -> quantization_config={{"quant_method":"fp8"}}',
    ),
    # compressed-tensors' key, which vLLM and SGLang read when quantization_config is absent.
    (
        '--hf-overrides \'{"compression_config": {"format": "float-quantized"}}\'',
        f'{HF}: (not set) -> compression_config={{"format":"float-quantized"}}',
    ),
    (
        '--json-model-override-args \'{"compression_config": {"format": "float-quantized"}}\'',
        f'{JSON_ARGS}: (not set) -> compression_config={{"format":"float-quantized"}}',
    ),
    # A null counts too: it drops a compressed-tensors checkpoint's quantization config.
    ("--hf-overrides '{\"compression_config\": null}'", f"{HF}: (not set) -> compression_config=null"),
    (
        '--hf-overrides \'{"text_config": {"compression_config": {"format": "float-quantized"}}}\'',
        f'{HF}: (not set) -> text_config.compression_config={{"format":"float-quantized"}}',
    ),
    (
        "--hf-overrides.compression_config.format float-quantized",
        f'{HF}: (not set) -> compression_config={{"format":"float-quantized"}}',
    ),
    (
        "--hf-overrides.text_config.compression_config.format float-quantized",
        f'{HF}: (not set) -> text_config.compression_config={{"format":"float-quantized"}}',
    ),
    # At any depth of the value, through objects and arrays.
    (
        '--json-model-override-args \'{"text_config": {"quantization_config": {}}}\'',
        f"{JSON_ARGS}: (not set) -> text_config.quantization_config={{}}",
    ),
    (
        '--json-model-override-args \'{"layers": [{"torch_dtype": "float16"}]}\'',
        f'{JSON_ARGS}: (not set) -> layers[0].torch_dtype="float16"',
    ),
    # vLLM's dotted form: its path ends at a key or passes through one, or its JSON value holds one.
    ("--hf-overrides.torch_dtype float16", f'{HF}: (not set) -> torch_dtype="float16"'),
    ("--hf-overrides.torch_dtype=float16", f'{HF}: (not set) -> torch_dtype="float16"'),
    ("--hf_overrides.torch_dtype float16", f'{HF}: (not set) -> torch_dtype="float16"'),
    ("--hf-overrides.text_config.torch_dtype float16", f'{HF}: (not set) -> text_config.torch_dtype="float16"'),
    (
        "--hf-overrides.quantization_config.quant_method fp8",
        f'{HF}: (not set) -> quantization_config={{"quant_method":"fp8"}}',
    ),
    ('--hf-overrides.text_config \'{"dtype": "float16"}\'', f'{HF}: (not set) -> text_config.dtype="float16"'),
    ("--hf-overrides.text_config.dtype+ float16", f'{HF}: (not set) -> text_config.dtype=["float16"]'),
    # A prefix of an override flag is that flag.
    ('--hf-over \'{"torch_dtype": "float16"}\'', f'{HF}: (not set) -> torch_dtype="float16"'),
    ('--json-model \'{"dtype": "float16"}\'', f'{JSON_ARGS}: (not set) -> dtype="float16"'),
    # A value that is not JSON counts by its text when the text names a key.
    ("--hf-overrides torch_dtype=float16", f"{HF}: (not set) -> (unparsed) torch_dtype=float16"),
    (
        "--json-model-override-args {torch_dtype=float16}",
        f"{JSON_ARGS}: (not set) -> (unparsed) {{torch_dtype=float16}}",
    ),
    ("--hf-overrides compression_config=null", f"{HF}: (not set) -> (unparsed) compression_config=null"),
]


@pytest.mark.parametrize(("server_args", "change"), OVERRIDES_ADDED)
async def test_an_override_that_sets_a_weight_precision_key_is_refused(
    tmp_path: Path, server_args: str, change: str
) -> None:
    await _refused_naming(tmp_path, _repo(), _delta(server_args), change)


@pytest.mark.parametrize(
    ("root_args", "server_args", "change"),
    [
        (
            '--hf-overrides \'{"torch_dtype": "bfloat16"}\'',
            '--hf-overrides \'{"torch_dtype": "float16"}\'',
            f'{HF}: torch_dtype="bfloat16" -> torch_dtype="float16"',
        ),
        (
            '--hf-overrides \'{"torch_dtype": "bfloat16", "max_position_embeddings": 32768}\'',
            "--hf-overrides '{\"max_position_embeddings\": 65536}'",
            f'{HF}: torch_dtype="bfloat16" -> (not set)',
        ),
        # The same value at another path is another setting.
        (
            '--hf-overrides \'{"text_config": {"torch_dtype": "bfloat16"}}\'',
            '--hf-overrides \'{"torch_dtype": "bfloat16"}\'',
            f'{HF}: text_config.torch_dtype="bfloat16" -> torch_dtype="bfloat16"',
        ),
        (
            '--json-model-override-args \'{"quantization_config": {"quant_method": "fp8", "block": [128, 128]}}\'',
            '--json-model-override-args \'{"quantization_config": {"quant_method": "fp8", "block": [64, 64]}}\'',
            f'{JSON_ARGS}: quantization_config={{"block":[128,128],"quant_method":"fp8"}} -> quantization_config=',
        ),
        (
            "--hf-overrides.torch_dtype bfloat16",
            "--hf-overrides.torch_dtype float16",
            f'{HF}: torch_dtype="bfloat16" -> torch_dtype="float16"',
        ),
    ],
)
async def test_an_override_that_changes_what_the_root_sets_under_a_key_is_refused(
    tmp_path: Path, root_args: str, server_args: str, change: str
) -> None:
    await _refused_naming(tmp_path, _repo(root_args), _delta(server_args), change)


async def test_an_override_flag_removed_from_the_resolved_launch_is_refused(tmp_path: Path) -> None:
    """What the resolved launch's override sets cannot be read, so neither can what its removal changes."""
    removal = _delta("--max-num-seqs 64", remove_args=("--hf-overrides",))
    await _refused_naming(tmp_path, _repo(), removal, f"{HF}: (not set) -> <removed>")


@pytest.mark.parametrize(
    ("root_args", "server_args"),
    [
        ("", "--hf-overrides '{\"max_position_embeddings\": 65536}'"),
        ("", "--hf-overrides.rope_scaling.factor 2"),
        ("", "--hf-overrides.text_config.max_position_embeddings 4096"),
        ("", '--json-model-override-args \'{"rope_scaling": {"type": "yarn", "factor": 4}}\''),
        # A JSON key that only contains a key's name is not that key; only a value that is not JSON counts by its text.
        ("", '--hf-overrides \'{"kv_cache_dtype": "fp8"}\''),
        ("", "--hf-overrides max_position_embeddings=65536"),
        # vLLM reads a dotted value that is not JSON as its text: these set a string under a key that is none of them.
        ("", "--hf-overrides.text_config torch_dtype=float16"),
        ("", "--hf-overrides.max_position_embeddings dtype"),
        (
            '--hf-overrides \'{"torch_dtype": "bfloat16", "max_position_embeddings": 32768}\'',
            '--hf-overrides \'{"max_position_embeddings": 65536, "torch_dtype": "bfloat16"}\'',
        ),
        ('--hf-overrides \'{"torch_dtype": "bfloat16"}\'', "--hf-overrides.torch_dtype bfloat16"),
    ],
)
async def test_an_override_that_keeps_what_the_root_sets_under_the_keys_is_accepted(
    tmp_path: Path, root_args: str, server_args: str
) -> None:
    assert await _refusal(tmp_path, _repo(root_args), _delta(server_args)) is None


async def test_removing_an_override_that_sets_no_key_is_accepted(tmp_path: Path) -> None:
    repo = _repo("--hf-overrides '{\"max_position_embeddings\": 32768}'")
    assert await _refusal(tmp_path, repo, _delta("--max-num-seqs 64", remove_args=("--hf-overrides",))) is None


# --- an override flag given more than once ---------------------------------------------------------------------------

ROOT_DTYPE = '--hf-overrides \'{"torch_dtype": "bfloat16"}\''
DROPPED = 'torch_dtype="bfloat16" -> (no key) | torch_dtype="bfloat16"'


@pytest.mark.parametrize(
    ("root_args", "server_args", "change"),
    [
        # argparse keeps the last value given, and the composed launch keeps both spellings: each of these serves
        # without the root's dtype override.
        (ROOT_DTYPE, "--hf_overrides '{\"max_position_embeddings\": 4096}'", f"{HF}: {DROPPED}"),
        (ROOT_DTYPE, "--hf-over '{\"max_position_embeddings\": 4096}'", f"{HF}: {DROPPED}"),
        (
            '--json-model-override-args \'{"torch_dtype": "bfloat16"}\'',
            "--json-model '{\"max_position_embeddings\": 4096}'",
            f"{JSON_ARGS}: {DROPPED}",
        ),
        # vLLM gives its merged dotted occurrences after all the others.
        (ROOT_DTYPE, "--hf-overrides.max_position_embeddings 4096", f"{HF}: {DROPPED}"),
        # It merges the dotted occurrences of each spelling apart, and gives them in turn.
        ("--hf-overrides.torch_dtype bfloat16", "--hf-over.max_position_embeddings 4096", f"{HF}: {DROPPED}"),
        # An occurrence that sets no key is a value even where vLLM keeps another one: here the dotted one.
        (
            "--hf-overrides.torch_dtype bfloat16",
            "--hf_overrides '{\"max_position_embeddings\": 4096}'",
            f"{HF}: {DROPPED}",
        ),
        # Merged, the dotted occurrences set another quantization config.
        (
            "--hf-overrides.quantization_config.quant_method fp8",
            "--hf-overrides.quantization_config.activation_scheme dynamic",
            f'{HF}: quantization_config={{"quant_method":"fp8"}} -> '
            'quantization_config={"activation_scheme":"dynamic","quant_method":"fp8"}',
        ),
    ],
)
async def test_an_occurrence_that_can_drop_what_the_root_sets_under_a_key_is_refused(
    tmp_path: Path, root_args: str, server_args: str, change: str
) -> None:
    await _refused_naming(tmp_path, _repo(root_args), _delta(server_args), change)


@pytest.mark.parametrize(
    ("root_args", "server_args"),
    [
        # vLLM merges the dotted occurrences of one spelling, ``_`` read as ``-``, into one value.
        ("--hf-overrides.torch_dtype bfloat16", "--hf-overrides.max_position_embeddings 4096"),
        ("--hf-overrides.torch_dtype bfloat16", "--hf_overrides.max_position_embeddings 4096"),
        # No occurrence sets a key.
        ("--hf-overrides '{\"max_position_embeddings\": 32768}'", "--hf_overrides '{\"rope_theta\": 1000000}'"),
        ("--hf-overrides '{\"max_position_embeddings\": 32768}'", "--hf-overrides.rope_theta 1000000"),
        # Every occurrence sets the root's value under the keys.
        (ROOT_DTYPE, '--hf_overrides \'{"torch_dtype": "bfloat16", "max_position_embeddings": 4096}\''),
    ],
)
async def test_occurrences_that_keep_what_the_root_sets_under_the_keys_are_accepted(
    tmp_path: Path, root_args: str, server_args: str
) -> None:
    assert await _refusal(tmp_path, _repo(root_args), _delta(server_args)) is None


def test_dotted_occurrences_merge_as_vllm_merges_them() -> None:
    assert _changes("--hf-overrides.a.dtype+ x", "--hf-overrides.a.dtype+ x --hf-overrides.a.dtype+=y") == [
        f'{HF}: a.dtype=["x"] -> a.dtype=["x","y"]'
    ]
    nested = '--hf-overrides \'{"text_config": {"torch_dtype": "bfloat16"}}\''
    assert _changes(nested, "--hf-overrides.text_config.torch_dtype bfloat16") == []
    assert _changes(nested, "--hf-overrides.text_config.torch_dtype bfloat16 --hf-overrides.text_config.x 1") == []


async def test_a_later_dotted_key_replaces_an_earlier_one_as_vllm_merges_them(tmp_path: Path) -> None:
    """The candidate's launch gives the root's ``bfloat16`` and then ``float16`` under one key, under two spellings
    that vLLM reads as one flag: it merges them into one value of ``--hf-overrides`` holding the later, float16."""
    repo = _repo("--hf-overrides.torch_dtype bfloat16")
    change = _delta("--hf_overrides.torch_dtype float16")
    await _refused_naming(tmp_path, repo, change, f'{HF}: torch_dtype="bfloat16" -> torch_dtype="float16"')


async def test_a_later_dotted_key_that_gives_the_root_value_is_accepted(tmp_path: Path) -> None:
    """The launch composed is ``--hf-overrides.torch_dtype float16 --hf_overrides.torch_dtype bfloat16``: vLLM serves
    the later, the root's bfloat16."""
    repo = _repo("--hf-overrides.torch_dtype bfloat16")
    change = _delta("--hf-overrides.torch_dtype float16 --hf_overrides.torch_dtype bfloat16")
    assert await _refusal(tmp_path, repo, change) is None


# --- two values of a setting: the one in effect ----------------------------------------------------------------------

#: A root whose dtype override vLLM serves after an occurrence that sets no key.
KEYED_LAST = '--hf_overrides \'{"max_position_embeddings": 4096}\' --hf-overrides \'{"torch_dtype": "bfloat16"}\''
#: A root that gives two dtype overrides, bfloat16 last.
TWO_DTYPES = '--hf-overrides \'{"torch_dtype": "float16"}\' --hf_overrides \'{"torch_dtype": "bfloat16"}\''
#: The dtype override those roots serve.
BF16_IN_EFFECT = '(torch_dtype="bfloat16" in effect)'


#: Each candidate below also changes something besides the order: an explore whose recipe differs from its parent's
#: only in the order of its flags is refused before this check, as one that changes nothing.
@pytest.mark.parametrize(
    ("root_args", "change", "named"),
    [
        # The candidate drops the occurrence that sets no key and gives one again after the dtype override: its launch
        # gives the root's values and serves no dtype override.
        (
            KEYED_LAST,
            _delta("--hf_overrides '{\"max_position_embeddings\": 8192}'", remove_args=("--hf_overrides",)),
            f'{HF}: (no key) | torch_dtype="bfloat16" {BF16_IN_EFFECT} -> '
            '(no key) | torch_dtype="bfloat16" ((no key) in effect)',
        ),
        # It drops the float16 override and gives it again last (with another key): float16 is served.
        (
            TWO_DTYPES,
            _delta(
                '--hf-overrides \'{"torch_dtype": "float16", "max_position_embeddings": 4096}\'',
                remove_args=("--hf-overrides",),
            ),
            f'{HF}: torch_dtype="bfloat16" | torch_dtype="float16" {BF16_IN_EFFECT} -> '
            'torch_dtype="bfloat16" | torch_dtype="float16" (torch_dtype="float16" in effect)',
        ),
        # argparse keeps a plain flag's last value too.
        (
            "--dtype float16 --dty bfloat16",
            _delta("--dtype float16 --max-num-seqs 64", remove_args=("--dtype",)),
            "--dtype: bfloat16 | float16 (bfloat16 in effect) -> bfloat16 | float16 (float16 in effect)",
        ),
        # The root's launch, composed, keeps the last of its two; the candidate gives the other after it.
        (
            "--dtype float16 --dtype bfloat16",
            _delta("--dty float16"),
            "--dtype: bfloat16 -> bfloat16 | float16 (float16 in effect)",
        ),
        # vLLM gives each spelling's merged dotted occurrences in the order the spellings first come: float16 last.
        (
            TWO_DTYPES,
            _delta("--hf-overrides.torch_dtype bfloat16 --hf-over.torch_dtype float16"),
            f'{HF}: torch_dtype="bfloat16" | torch_dtype="float16" {BF16_IN_EFFECT} -> '
            'torch_dtype="bfloat16" | torch_dtype="float16" (torch_dtype="float16" in effect)',
        ),
    ],
)
async def test_a_candidate_that_changes_which_value_takes_effect_is_refused(
    tmp_path: Path, root_args: str, change: RecipeDelta, named: str
) -> None:
    await _refused_naming(tmp_path, _repo(root_args), change, named)


@pytest.mark.parametrize(
    ("root_args", "change"),
    [
        # The occurrence that sets no key is given again where it was: the dtype override still comes last.
        (
            '--hf-overrides \'{"max_position_embeddings": 4096}\' --hf_overrides \'{"torch_dtype": "bfloat16"}\'',
            _delta("--hf-overrides '{\"max_position_embeddings\": 8192}'"),
        ),
        # vLLM gives a dotted occurrence after the others wherever the launch gives it: float16 both times.
        (
            '--hf-overrides.torch_dtype float16 --hf_overrides \'{"torch_dtype": "bfloat16"}\'',
            _delta("--hf-overrides.torch_dtype float16 --max-num-seqs 64", remove_args=("--hf-overrides.torch_dtype",)),
        ),
        # The root serves its dotted float16; the candidate gives float16 as a plain occurrence after bfloat16.
        (
            '--hf-overrides \'{"torch_dtype": "bfloat16"}\' --hf-overrides.torch_dtype float16',
            _delta('--hf_overrides \'{"torch_dtype": "float16"}\'', remove_args=("--hf-overrides.torch_dtype",)),
        ),
        (KEYED_LAST, _delta("--max-num-seqs 64")),
        ("--dtype float16 --dty bfloat16", _delta("--max-num-seqs 64")),
        # The spellings' dotted occurrences come in the other order: bfloat16 last, as the root serves it.
        (TWO_DTYPES, _delta("--hf-over.torch_dtype float16 --hf-overrides.torch_dtype bfloat16")),
    ],
)
async def test_a_candidate_that_keeps_the_value_in_effect_is_accepted(
    tmp_path: Path, root_args: str, change: RecipeDelta
) -> None:
    assert await _refusal(tmp_path, _repo(root_args), change) is None


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (
            "--dtype float16 --dtype bfloat16",
            "--dtype float16 --dtype bfloat16 --dty float16",
            ["--dtype: bfloat16 | float16 (bfloat16 in effect) -> bfloat16 | float16 (float16 in effect)"],
        ),
        (
            "--dtype float16 --dtype bfloat16",
            "--dtype bfloat16 --dtype float16",
            ["--dtype: bfloat16 | float16 (bfloat16 in effect) -> bfloat16 | float16 (float16 in effect)"],
        ),
        ("--dtype float16 --dtype bfloat16", "--dty float16 --dtype bfloat16", []),
        # A value given twice is one value: its place is not compared.
        ("--dtype bfloat16 --dty bfloat16", "--dty bfloat16 --dtype bfloat16", []),
        # A value that begins with a dot is a value, not a dotted key.
        (
            "--model ./a --model ./b",
            "--model ./b --model ./a",
            ["the checkpoint: ./a | ./b (./b in effect) -> ./a | ./b (./a in effect)"],
        ),
        # vLLM merges a spelling's dotted occurrences into one value, the later key replacing the earlier, and gives
        # it after the others.
        (
            "--quantization-config.a 1 --quantization-config.b 2",
            "--quantization-config.b 2 --quantization-config.a 1",
            [],
        ),
        (
            "--quantization-config.linear fp8 --quantization-config.linear mxfp8",
            "--quantization-config.linear mxfp8 --quantization-config.linear fp8",
            [
                '--quantization-config: .linear fp8 | .linear mxfp8 ({"linear":"mxfp8"} in effect) -> '
                '.linear fp8 | .linear mxfp8 ({"linear":"fp8"} in effect)'
            ],
        ),
        (
            '--quantization-config.linear fp8 --quantization-config {"linear":"mxfp8"}',
            '--quantization-config {"linear":"mxfp8"} --quantization-config.linear fp8',
            [],
        ),
        (
            "--hf-overrides.torch_dtype bfloat16",
            "--hf-overrides.torch_dtype float16 --hf_overrides.torch_dtype bfloat16",
            [],
        ),
        # The dotted float16 is served, though given before; then the plain bfloat16, given last.
        (
            '--hf-overrides {"torch_dtype":"bfloat16"} --hf-overrides.torch_dtype float16',
            '--hf_overrides {"torch_dtype":"float16"} --hf-overrides {"torch_dtype":"bfloat16"}',
            [
                f'{HF}: torch_dtype="bfloat16" | torch_dtype="float16" (torch_dtype="float16" in effect) -> '
                'torch_dtype="bfloat16" | torch_dtype="float16" (torch_dtype="bfloat16" in effect)'
            ],
        ),
    ],
)
def test_two_values_of_a_setting_compare_by_the_one_in_effect(before: str, after: str, expected: list[str]) -> None:
    assert _changes(before, after) == expected


def test_a_removal_from_the_resolved_launch_comes_before_the_launch_own_flags() -> None:
    """The launch's own flags come after the resolved launch's, whatever it removes from that: float16 is served, then
    bfloat16."""
    before = LaunchConfig(server_args="--dty bfloat16 --dtyp float16", remove_args=("--dtype",))
    after = LaunchConfig(server_args="--dtyp float16 --dty bfloat16", remove_args=("--dtype",))
    assert weight_precision_changes(before, after) == [
        "--dtype: <removed> | bfloat16 | float16 (float16 in effect) -> "
        "<removed> | bfloat16 | float16 (bfloat16 in effect)"
    ]


# --- vLLM's dotted flag takes the next word as its value, whatever it is ---------------------------------------------


def test_a_flag_word_a_dotted_flag_takes_as_its_value_is_not_that_flag() -> None:
    """vLLM sets ``foo`` to ``-qfp8`` and serves no quantization; acceptance refuses a candidate that adds such a pair
    (below), so only a root's own is read so."""
    for server_args in (
        "--hf-overrides.foo -qfp8",
        "--hf-overrides.foo -q=fp8",
        "--additional-config.foo -qfp8 --max-num-seqs 64",
        # A one-dash flag with a dot is dotted too (vLLM's ``-cc`` is its ``--compilation-config``).
        "-cc.custom_ops+ -qfp8",
    ):
        assert _changes("--quantization fp8", server_args) == ["--quantization: fp8 -> (not set)"], server_args


@pytest.mark.parametrize(
    ("root_args", "change"),
    [
        # The dotted flag takes ``bar``; ``-qfp8`` is the root's quantization.
        ("--quantization fp8", _delta("--hf-overrides.foo bar -qfp8", remove_args=("--quantization",))),
        # A word after the dotted flag's value is read with the flag before the dotted one, as argparse reads what vLLM
        # leaves of the launch: ``--revision 3``.
        ("--revision 3", _delta("--revision --hf-overrides.foo x 3")),
    ],
)
async def test_the_words_after_a_dotted_flag_value_are_read_as_vllm_leaves_them(
    tmp_path: Path, root_args: str, change: RecipeDelta
) -> None:
    assert await _refusal(tmp_path, _repo(root_args), change) is None


async def test_a_word_after_a_dotted_flag_value_that_changes_the_flag_before_it_is_refused(tmp_path: Path) -> None:
    change = _delta("--revision --hf-overrides.foo x 4")
    await _refused_naming(tmp_path, _repo("--revision 3"), change, "the checkpoint's revision: 3 -> 4")


def test_a_dotted_flag_written_with_its_value_takes_no_other_word() -> None:
    """The composed launch writes ``--flag.key=value`` as two words, so only the comparison sees this form."""
    assert _changes("--quantization fp8", "--hf-overrides.foo=bar -qfp8") == []
    assert _changes("--quantization fp8", "--hf-overrides.foo -qfp8") == ["--quantization: fp8 -> (not set)"]


# --- -O<rest> is --optimization-level <rest> -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        # vLLM reads ``-O3.0`` as ``--optimization-level 3.0``, never as a dotted flag that takes the next word.
        ("", "-O3.0 --quantization fp8", ["--quantization: (not set) -> fp8"]),
        ("", "-O.x -qfp8", ["--quantization: (not set) -> fp8"]),
        ("", "-O=3.0 --dtype float16", ["--dtype: (not set) -> float16"]),
        # The words after ``-O<rest>`` are values of ``--optimization-level``, not of the flag before it.
        ("--dtype 16", "--dtype -O3.0 x 16", ["--dtype: 16 -> (on)"]),
        ("--dtype float16", "-O3 --dtype float16", []),
        ("--dtype float16", "-O 3 --dtype float16", []),
    ],
)
def test_a_word_that_begins_with_dash_o_is_the_optimization_level(before: str, after: str, expected: list[str]) -> None:
    assert _changes(before, after) == expected


# --- a launch the frameworks read in two ways ------------------------------------------------------------------------

AMBIGUOUS = "ambiguous between frameworks"


def _composed(repo: FakeRepo, change: RecipeDelta) -> tuple[LaunchConfig, LaunchConfig]:
    """The root's launch and the candidate's, as acceptance composes them."""
    return apply(repo.recipes[ROOT], RecipeDelta()).launch, apply(repo.recipes[PARENT], change).launch


async def _refused_as_ambiguous(
    tmp_path: Path, repo: FakeRepo, change: RecipeDelta, pair: str, **fields: object
) -> None:
    refused = await _refusal(tmp_path, repo, change, **fields)
    assert refused is not None, "a launch the frameworks read in two ways was accepted"
    assert refused.code == DELTA_NOT_APPLICABLE_CODE
    assert AMBIGUOUS in str(refused)
    assert "(--flag.key=value)" in str(refused)
    assert repr(pair) in str(refused), str(refused)


#: A candidate that adds a word that begins with ``-`` and holds a ``.`` but no ``=``, right before a word that begins
#: with ``-``, on a root that does not give the pair, and the pair. The comparison alone accepts each: it reads the pair
#: one way (a dotted flag takes the next word, a number is a value, ``-O3.0`` is ``--optimization-level 3.0``), and a
#: framework that reads it the other way serves another precision.
READ_TWO_WAYS = [
    ("", _delta("--hf-overrides.foo -qfp8"), "--hf-overrides.foo -qfp8"),
    ("", _delta("-cc.custom_ops+ -qfp8"), "-cc.custom_ops+ -qfp8"),
    ("--quantization fp8", _delta("-O3.0 -qfp8", remove_args=("--quantization",)), "-O3.0 -qfp8"),
    ("--dtype float16", _delta("-0.5 --dtype float16", remove_args=("--dtype",)), "-0.5 --dtype"),
    ("--dtype float16", _delta("-0.0 --dtype float16", remove_args=("--dtype",)), "-0.0 --dtype"),
    ("--dtype float16", _delta("-.5 --dtype float16", remove_args=("--dtype",)), "-.5 --dtype"),
    ("--dtype float16", _delta("-1. --dtype float16", remove_args=("--dtype",)), "-1. --dtype"),
    ("--quantization fp8", _delta("--served-model-name a -0.5 -qfp8", remove_args=("--quantization",)), "-0.5 -qfp8"),
    # The composed launch writes ``--flag=-0.5`` as ``--flag -0.5``.
    ("--dtype float16", _delta("--some-flag=-0.5 --dtype float16", remove_args=("--dtype",)), "-0.5 --dtype"),
    # The root gives the word with another word after it, or as its last.
    (
        "--served-model-name a -0.5 --max-num-seqs 32",
        _delta("--enable-x", remove_args=("--max-num-seqs",)),
        "-0.5 --enable-x",
    ),
    ("--some-flag -0.5", _delta("--max-num-seqs 64"), "-0.5 --max-num-seqs"),
    # A pair counts each time it is given: the candidate gives the root's twice.
    ("-cc.custom_ops+ -qfp8", _delta("-cc.custom_ops+ -qfp8"), "-cc.custom_ops+ -qfp8"),
]


@pytest.mark.parametrize(("root_args", "change", "pair"), READ_TWO_WAYS)
async def test_a_candidate_that_adds_a_word_read_two_ways_is_refused(
    tmp_path: Path, root_args: str, change: RecipeDelta, pair: str
) -> None:
    repo = _repo(root_args)
    assert weight_precision_changes(*_composed(repo, change)) == [], "the comparison alone refuses this candidate"
    await _refused_as_ambiguous(tmp_path, repo, change, pair)


@pytest.mark.parametrize(
    ("root_args", "change", "pair", "purpose"),
    [
        ("--max-num-seqs 64", _delta("-O3.0 --quantization fp8"), "-O3.0 --quantization", Purpose.EXPLORE),
        ("--max-num-seqs 64", _delta("-O3.0 -qfp8"), "-O3.0 -qfp8", Purpose.EXPLORE),
        ("--max-num-seqs 64", _delta("-O3.0 --dtype float16"), "-O3.0 --dtype", Purpose.EXPLORE),
        (
            "--max-num-seqs 64",
            _delta('-O2.0 --hf-overrides {"torch_dtype":"float16"}'),
            "-O2.0 --hf-overrides",
            Purpose.EXPLORE,
        ),
        (
            "--quantization fp8 --served-model-name a",
            _delta("-0.5 -qfp8", remove_args=("--quantization",)),
            "-0.5 -qfp8",
            Purpose.EXPLORE,
        ),
        (
            "--dtype float16 --served-model-name a",
            _delta("-0.5 --dtype float16", remove_args=("--dtype",)),
            "-0.5 --dtype",
            Purpose.EXPLORE,
        ),
        # A dotted flag that takes the flag after it, whatever the comparison makes of the rest.
        (
            "--quantization fp8",
            _delta("--hf-overrides.foo -qfp8", remove_args=("--quantization",)),
            "--hf-overrides.foo -qfp8",
            Purpose.EXPLORE,
        ),
        (
            "--quantization fp8",
            _delta("--hf-overrides.foo -q=fp8", remove_args=("--quantization",)),
            "--hf-overrides.foo -q=fp8",
            Purpose.EXPLORE,
        ),
        (
            "--quantization fp8",
            _delta("--additional-config.foo -qfp8 --max-num-seqs 64", remove_args=("--quantization",)),
            "--additional-config.foo -qfp8",
            Purpose.EXPLORE,
        ),
        ("", _delta("--hf-overrides.foo --quantization fp8"), "--hf-overrides.foo --quantization", Purpose.EXPLORE),
    ],
)
async def test_the_launches_found_read_two_ways_are_refused(
    tmp_path: Path, root_args: str, change: RecipeDelta, pair: str, purpose: Purpose
) -> None:
    await _refused_as_ambiguous(tmp_path, _repo(root_args), change, pair, purpose=purpose)


@pytest.mark.parametrize(
    ("root_args", "change", "pair"),
    [
        ("--max-num-seqs 64", _delta("-O3.0 --quantization fp8"), "-O3.0 --quantization"),
        (
            "--quantization fp8 --served-model-name a",
            _delta("-0.5 -qfp8", remove_args=("--quantization",)),
            "-0.5 -qfp8",
        ),
    ],
)
async def test_a_confirm_that_adds_a_word_read_two_ways_is_refused(
    root_args: str, change: RecipeDelta, pair: str
) -> None:
    acceptance = Acceptance(AcceptanceSetup(_repo(root_args), FakePlanner(), IMAGE), is_accepted=lambda _: True)
    with pytest.raises(RequestRejected) as refused:
        await acceptance.prepare(req(purpose=Purpose.CONFIRM, retest_of="r1", change=change))
    assert refused.value.code == DELTA_NOT_APPLICABLE_CODE
    assert AMBIGUOUS in str(refused.value)
    assert repr(pair) in str(refused.value)


@pytest.mark.parametrize("purpose", [Purpose.INTEGRATION, Purpose.FINAL])
async def test_a_rerun_of_a_parent_that_adds_a_word_read_two_ways_is_refused(tmp_path: Path, purpose: Purpose) -> None:
    repo = _repo("--max-num-seqs 64 -qfp8", parent_args="--max-num-seqs 64 -0.5 -qfp8")
    await _refused_as_ambiguous(tmp_path, repo, RecipeDelta(), "-0.5 -qfp8", purpose=purpose)


@pytest.mark.parametrize(
    ("root_args", "change"),
    [
        # A dotted value written with ``=`` takes no other word: the composed launch gives ``bar`` after the flag.
        ("--quantization fp8", _delta("--hf-overrides.foo=bar -qfp8", remove_args=("--quantization",))),
        # A one-dash word keeps its ``=``.
        ("--quantization fp8", _delta("-cc.custom_ops+=none -qfp8", remove_args=("--quantization",))),
        ("", _delta("-cc.custom_ops+=none")),
        ("", _delta("-tp 8 -O3 --max-num-seqs 64")),
        ("", _delta("-O 3 --max-num-seqs 64")),
        ("", _delta("-O=3.0 --max-num-seqs 64")),
        ("", _delta('--compilation-config \'{"cudagraph_mode": "FULL_AND_PIECEWISE", "x": -0.5}\' --max-num-seqs 64')),
        ("", _delta("--hf-overrides.rope_scaling.factor 2.5 --max-num-seqs 64")),
        ("", _delta("--profiler-config.ignore_frontend True --max-num-seqs 64")),
        # A number given to a flag, last or with no word that begins with ``-`` after it.
        ("--dtype float16", _delta("--max-num-seqs 64 --some-flag -0.5")),
        ("--dtype float16", _delta("--some-flag -1 --max-num-seqs 64")),
        ("--dtype float16", _delta("--some-flag -0.5 1 --max-num-seqs 64")),
        # The root's own pairs: the session's launch.
        ("--served-model-name a -0.5 --dtype float16", _delta("--max-num-seqs 64")),
        ("--hf-overrides.foo -qfp8", _delta("--max-num-seqs 64")),
        ("-O3.0 --quantization fp8", _delta("--max-num-seqs 64")),
        ("-0.5 --max-num-seqs 32", _delta("-0.5 --max-num-seqs 64")),
    ],
)
async def test_a_word_read_one_way_or_the_root_own_is_accepted(
    tmp_path: Path, root_args: str, change: RecipeDelta
) -> None:
    assert await _refusal(tmp_path, _repo(root_args), change) is None


@pytest.mark.parametrize("purpose", [Purpose.BASELINE, Purpose.INTEGRATION])
async def test_the_root_launch_read_two_ways_is_the_session_own(tmp_path: Path, purpose: Purpose) -> None:
    repo = _repo("--hf-overrides.foo -qfp8 --served-model-name a -0.5 --dtype float16 -O3.0 -qfp8")
    assert await _refusal(tmp_path, repo, RecipeDelta(), purpose=purpose) is None


def test_the_pairs_a_launch_adds_are_named_in_its_order() -> None:
    root = LaunchConfig(server_args="--a -0.5 --b x")
    assert ambiguous_words(root, root) == []
    after = LaunchConfig(server_args='-1.5 --dtype float16 --a -0.5 --b x --hf-overrides.foo -qfp8 --c {"d":-0.5}')
    assert ambiguous_words(root, after) == ["-1.5 --dtype", "--hf-overrides.foo -qfp8"]
    assert ambiguous_words(after, root) == []


# --- options of the frameworks that a spelling only begins, or that only begin one ---------------------------------


@pytest.mark.parametrize(
    "server_args",
    [
        # vLLM's serve
        "--config-format hf",
        "--model-impl transformers",
        "--model-class-overrides {}",
        "--code-revision abc",
        "--tokenizer-revision abc",
        "--hf-token x",
        # SGLang's server
        "--model-checksum x",
        "--model-config-parser x",
        "--modelexpress-config x",
        "--modelopt-checkpoint-restore-path /c",
        "--modelopt-checkpoint-save-path /c",
        "--modelopt-export-path /c",
        "--hf-chat-template-name x",
        "--torch-compile-max-bs 8",
        # SGLang's diffusion server takes ``--mode`` for its ``--performance-mode``, though it begins ``--model``.
        "--mode speed",
    ],
)
def test_an_option_of_the_frameworks_is_read_as_itself(server_args: str) -> None:
    assert _changes("", server_args) == []


# --- the load format: a change only among the speed-only ones is none -----------------------------------------------

#: Load formats that load something other than the checkpoint's own tensors as stored, or nothing: SGLang's
#: ``flash_rl`` serves a BF16 checkpoint in FP8, ``bitsandbytes`` in NF4, ``gguf`` reads GGUF's quantized blocks,
#: ``dummy`` random weights; the rest read another file than the checkpoint, through another loader or a plugin.
NOT_SPEED_ONLY = [
    "flash_rl",
    "bitsandbytes",
    "gguf",
    "dummy",
    "tensorizer",
    "sharded_state",
    "presharded",
    "mistral",
    "layered",
    "remote",
    "remote_instance",
    "jax",
    "extensible",
    "private",
    "runai_streamer",
    "runai_streamer_sharded",
    "expert_pack",
    "instanttensor",
    "ipc_cache",
    "modelexpress",
    "hf",
    "my_plugin_loader",
]


async def test_a_flash_rl_load_of_a_bf16_root_is_refused(tmp_path: Path) -> None:
    repo = _repo("--model-path /models/bf16 --dtype bfloat16")
    change = _delta("--load-format flash_rl")
    await _refused_naming(tmp_path, repo, change, "--load-format: (not set) -> flash_rl")


@pytest.mark.parametrize("load_format", NOT_SPEED_ONLY)
@pytest.mark.parametrize("root_args", ["", "--load-format auto", "--load-format safetensors"])
async def test_a_load_format_that_is_not_speed_only_is_refused(
    tmp_path: Path, root_args: str, load_format: str
) -> None:
    await _refused_naming(tmp_path, _repo(root_args), _delta(f"--load-format {load_format}"), "--load-format: ")


SPEED_ONLY_CHANGES = [
    ("", "--load-format safetensors"),
    ("", "--load-format auto"),
    ("--load-format safetensors", "--load-format pt"),
    ("--load-format pt", "--load-format auto"),
    ("--load-format auto", "--load-format fastsafetensors"),
    ("--load-format fastsafetensors", "--load-format safetensors"),
    ("--load-format safetensors", "--load-format npcache"),
    ("--load-format=pt", "--load_format SafeTensors"),
    ("", "--load-form=FASTSAFETENSORS"),
]


@pytest.mark.parametrize(("root_args", "server_args"), SPEED_ONLY_CHANGES)
async def test_a_change_only_among_speed_only_load_formats_is_accepted(
    tmp_path: Path, root_args: str, server_args: str
) -> None:
    assert await _refusal(tmp_path, _repo(root_args), _delta(server_args)) is None


@pytest.mark.parametrize(
    "server_args", ["--load-format safetensors", "--load-format pt --load-format auto", "--max-num-seqs 64"]
)
def test_a_load_format_given_or_dropped_only_among_speed_only_ones_is_no_change(server_args: str) -> None:
    for root_args in ("", "--load-format auto", "--load-format=fastsafetensors", "--load-format npcache"):
        assert _changes(root_args, server_args) == [], (root_args, server_args)
        assert _changes(server_args, root_args) == [], (server_args, root_args)


async def test_a_root_that_loads_with_flash_rl_and_a_candidate_that_keeps_it_is_accepted(tmp_path: Path) -> None:
    repo = _repo("--load-format flash_rl")
    assert await _refusal(tmp_path, repo, _delta("--load-format flash_rl --max-num-seqs 64")) is None
    assert await _refusal(tmp_path, repo, _delta("--max-num-seqs 64")) is None


@pytest.mark.parametrize(
    ("change", "shown"),
    [
        (_delta("--max-num-seqs 64", remove_args=("--load-format",)), "--load-format: flash_rl -> (not set)"),
        (_delta("--load-format auto"), "--load-format: flash_rl -> auto"),
        (_delta("--load-format=safetensors"), "--load-format: flash_rl -> safetensors"),
    ],
)
async def test_a_candidate_that_drops_a_root_flash_rl_load_is_refused(
    tmp_path: Path, change: RecipeDelta, shown: str
) -> None:
    await _refused_naming(tmp_path, _repo("--load-format flash_rl"), change, shown)


def test_a_flash_rl_load_dropped_from_a_launch_is_a_change() -> None:
    assert _changes("--load-format flash_rl", "") == ["--load-format: flash_rl -> (not set)"]
    assert _changes("--load-format flash_rl", "--load-format auto") == ["--load-format: flash_rl -> auto"]


@pytest.mark.parametrize(
    ("server_args", "shown"),
    [
        ("--load-format flash_rl", "flash_rl"),
        ("--load-format=flash_rl", "flash_rl"),
        ("--load_format flash_rl", "flash_rl"),
        ("--load-format FLASH_RL", "FLASH_RL"),
        ("--load-form flash_rl", "flash_rl"),
        ("--load-f=gguf", "gguf"),
        ("--load-format.x safetensors", ".x safetensors"),
        ("--load-format", "(on)"),
        ("--load-format safetensors --load-format dummy", "dummy | safetensors (dummy in effect)"),
        ("--load-format pt --max-num-seqs 64 --load_format=bitsandbytes", "bitsandbytes | pt (bitsandbytes in effect)"),
    ],
)
def test_the_load_format_in_every_form_its_framework_reads(server_args: str, shown: str) -> None:
    assert _changes("", server_args) == [f"--load-format: (not set) -> {shown}"]


def test_the_speed_only_load_formats_are_the_default_loader_ones() -> None:
    assert SPEED_ONLY_LOAD_FORMATS == {"auto", "pt", "safetensors", "npcache", "fastsafetensors"}
    assert not SPEED_ONLY_LOAD_FORMATS & set(NOT_SPEED_ONLY)


# --- Arbor's cases (arbor/tests/test_weight_precision.py) ------------------------------------------------------------

BASE = "--tensor-parallel-size 1 --max-num-seqs 64 --kv-cache-dtype auto"


def test_arbor_a_weight_quantization_added_is_a_change() -> None:
    assert _changes(BASE, BASE + " --quantization fp8_per_channel") == ["--quantization: (not set) -> fp8_per_channel"]
    assert _changes(BASE, BASE + " -q=mxfp4") == ["--quantization: (not set) -> mxfp4"]


def test_arbor_removing_changing_or_reconfiguring_weight_precision_is_a_change() -> None:
    quantized = BASE + " --quantization fp8"
    assert _changes(quantized, BASE)
    assert _changes(quantized, BASE + " --quantization mxfp8")
    assert _changes(BASE, BASE + " --quantization-config '{\"a\": 1}'")
    assert _changes(BASE, BASE + ' --online_quant_config \'{"global_quant_config": "ptpc_fp8"}\'')
    assert _changes(BASE, BASE + " --dtype float16")
    assert _changes(BASE + " --model-path /models/a-bf16", BASE + " --model-path /models/a-fp8")


def test_arbor_intermediates_and_unchanged_weight_flags_are_not() -> None:
    assert _changes(BASE, BASE.replace("--kv-cache-dtype auto", "--kv-cache-dtype fp8")) == []
    assert _changes(BASE, BASE + " --enable-aiter-allreduce-fusion --attention-backend aiter") == []
    assert _changes("--quantization fp8 " + BASE, BASE + " --quantization=fp8 --max-num-seqs 128") == []


def test_arbor_the_spellings_a_parser_reads_as_a_listed_flag() -> None:
    assert _changes(BASE, BASE + " -qfp8_per_channel") == ["--quantization: (not set) -> fp8_per_channel"]
    assert _changes(BASE, BASE + " --dty float16") == ["--dtype: (not set) -> float16"]
    assert _changes(BASE, BASE + " --quantization_config '{\"a\": 1}'") == [
        '--quantization-config: (not set) -> {"a":1}'
    ]
    assert _changes(BASE, BASE + " --quantization-config.linear fp8_per_channel") == [
        "--quantization-config: (not set) -> .linear fp8_per_channel"
    ]
    assert {change.split(":")[0] for change in _changes(BASE, BASE + " --quant fp8")} == {
        "--quantization",
        "--quantization-config",
        "--quantization-param-path",
        "--quantize-and-serve",
    }


def test_arbor_one_setting_under_two_spellings_is_one_setting() -> None:
    root = BASE + " --quantization fp8"
    assert _changes(root, BASE + " -q fp8 --quantization fp8") == []
    assert _changes(root, BASE + " --quantization=fp8") == []


def test_arbor_the_flags_toolbox_added_and_the_overrides() -> None:
    assert _changes(BASE, BASE + " --revision abc123") == ["the checkpoint's revision: (not set) -> abc123"]
    for flag in ("--modelopt-quant fp8", "--torchao-config int4wo-128", "--quantize-and-serve", "--config c.yaml"):
        assert _changes(BASE, f"{BASE} {flag}"), flag
    assert _changes(BASE, BASE + " --hf-overrides '{\"max_position_embeddings\": 65536}'") == []
    assert _changes(BASE, BASE + ' --hf-overrides \'{"torch_dtype": "float16"}\'') == [
        f'{HF}: (not set) -> torch_dtype="float16"'
    ]
    assert _changes(BASE, BASE + ' --json-model-override-args \'{"text_config": {"quantization_config": {}}}\'')
    assert _changes(BASE, BASE + " --hf-overrides.torch_dtype float16")
    assert _changes(BASE, BASE + " --hf-overrides.rope_scaling.factor 2") == []
