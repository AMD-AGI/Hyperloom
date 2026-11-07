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
    ("--revision", "abc123", "def456", "the checkpoint's revision"),
    ("--config", "/cfg/a.yaml", "/cfg/b.yaml", "--config"),
    ("--modelopt-quant", "fp8", "nvfp4", "--modelopt-quant"),
    ("--torchao-config", "int4wo-128", "int8wo", "--torchao-config"),
    ("--quantize-and-serve", "true", "false", "--quantize-and-serve"),
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
        ("--dtype bfloat16", "--dtype bfloat16 --dtype float16", ["--dtype: bfloat16 -> bfloat16 | float16"]),
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

HF = "--hf-overrides (dtype, quantization_config, torch_dtype)"
JSON_ARGS = "--json-model-override-args (dtype, quantization_config, torch_dtype)"

#: A candidate launch that sets a weight-precision key through an override flag, on a root that sets none, and the
#: change it is refused for.
OVERRIDES_ADDED = [
    ('--hf-overrides \'{"torch_dtype": "float16"}\'', f'{HF}: (not set) -> torch_dtype="float16"'),
    ('--hf-overrides \'{"dtype": "float16"}\'', f'{HF}: (not set) -> dtype="float16"'),
    (
        '--hf-overrides \'{"quantization_config": {"quant_method": "fp8"}}\'',
        f'{HF}: (not set) -> quantization_config={{"quant_method":"fp8"}}',
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
    (
        "--hf-overrides.text_config torch_dtype=float16",
        f"{HF}: (not set) -> (unparsed) .text_config torch_dtype=float16",
    ),
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
        "--hf-config-path /c",
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
