# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A candidate may not change the draft or the simulated acceptance of a base launch that simulates acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors.tests.test_native_launch_replay import native_receipt
from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors._grid_runner import GridRecipe, _build_variant_yaml, run_grid
from hyperloom.orchestrator.actions.executors._grid_variant_filter import ServingStack, apply_compatibility_filter
from hyperloom.orchestrator.actions.executors._launch_evidence import persist_launch_evidence
from hyperloom.orchestrator.actions.executors._workload_envs import (
    BenchmarkTarget,
    LaunchLayer,
    materialize_benchmark_config,
)
from hyperloom.common.failure_signature import SERVE_FLAG
from hyperloom.orchestrator.actions.executors._spec_decode_pin import (
    SpeculativeDecodingPinnedError,
    refuse_unpinned_speculative_change,
    speculative_pin_reason,
)


def _spec_config(**overrides) -> str:
    config = {
        "method": "eagle3",
        "model": "/models/MiniMax-M3-eagle3",
        "num_speculative_tokens": 3,
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": 2.78,
        **overrides,
    }
    return "--speculative-config " + json.dumps({k: v for k, v in config.items() if v is not None})


VLLM_SYNTHETIC = _spec_config() + " --max-num-seqs 128"
DSPARK_SYNTHETIC = "--speculative-config " + json.dumps(
    {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": 3.51,
    }
)
SGLANG_EAGLE = "--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6"
SGLANG_SYNTHETIC_ENV = {"SGLANG_SIMULATE_ACC_LEN": "3.61"}
ATOM_SYNTHETIC = "--method mtp --num-speculative-tokens 3 --spec-decode-acceptance-length 2.5"
NATIVE_PREFIX = ["python3", "-m", "sglang.launch_server", "--model-path", "Qwen/Qwen3-0.6B"]
DSPARK_RECIPE_ARGS = [
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-dspark-block-size",
    "4",
    "--speculative-num-draft-tokens",
    "4",
]


@pytest.mark.parametrize(
    ("base_args", "base_envs", "candidate_args", "candidate_envs"),
    [
        pytest.param(
            VLLM_SYNTHETIC,
            {},
            VLLM_SYNTHETIC + " " + _spec_config(num_speculative_tokens=5, synthetic_acceptance_length=3.21),
            {},
            id="vllm-draft-length-and-acceptance",
        ),
        pytest.param(VLLM_SYNTHETIC, {}, _spec_config(synthetic_acceptance_length=4.0), {}, id="vllm-acceptance-only"),
        pytest.param(VLLM_SYNTHETIC, {}, _spec_config(num_speculative_tokens=4), {}, id="vllm-stale-acceptance"),
        pytest.param(VLLM_SYNTHETIC, {}, _spec_config(method="mtp"), {}, id="vllm-method"),
        pytest.param(
            VLLM_SYNTHETIC,
            {},
            '--speculative-config {"attention_backend":"TRITON_ATTN"}',
            {},
            id="vllm-config-that-drops-the-draft",
        ),
        pytest.param(VLLM_SYNTHETIC, {}, "--max-num-seqs 128", {}, id="vllm-config-removed"),
        pytest.param(
            DSPARK_SYNTHETIC,
            {},
            DSPARK_SYNTHETIC.replace('"num_speculative_tokens": 5', '"num_speculative_tokens": 6'),
            {},
            id="dspark-draft-length",
        ),
        pytest.param(
            SGLANG_EAGLE,
            SGLANG_SYNTHETIC_ENV,
            SGLANG_EAGLE + " --speculative-num-steps 3",
            SGLANG_SYNTHETIC_ENV,
            id="sglang-steps",
        ),
        pytest.param(
            SGLANG_EAGLE, SGLANG_SYNTHETIC_ENV, SGLANG_EAGLE, {"SGLANG_SIMULATE_ACC_LEN": "2.99"}, id="sglang-acc-len"
        ),
        pytest.param(SGLANG_EAGLE, SGLANG_SYNTHETIC_ENV, SGLANG_EAGLE, {}, id="sglang-acc-len-unset"),
        pytest.param(
            SGLANG_EAGLE,
            SGLANG_SYNTHETIC_ENV,
            SGLANG_EAGLE,
            {**SGLANG_SYNTHETIC_ENV, "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token"},
            id="sglang-acceptance-env-added",
        ),
        pytest.param(
            ATOM_SYNTHETIC, {}, ATOM_SYNTHETIC + " --spec-decode-acceptance-length 3.0", {}, id="atom-acceptance-flag"
        ),
        pytest.param(
            VLLM_SYNTHETIC,
            {},
            VLLM_SYNTHETIC + " --speculative-config.num_speculative_tokens 5",
            {},
            id="vllm-dotted-key-over-the-json",
        ),
        pytest.param(
            VLLM_SYNTHETIC,
            {},
            _spec_config(enable_adaptive_verification=True),
            {},
            id="vllm-adaptive-verification-on",
        ),
        pytest.param(
            _spec_config(enable_adaptive_verification=False),
            {},
            VLLM_SYNTHETIC + " --speculative-config.enable_adaptive_verification true",
            {},
            id="vllm-adaptive-verification-flipped",
        ),
    ],
)
def test_a_candidate_changing_the_pinned_draft_is_refused(base_args, base_envs, candidate_args, candidate_envs):
    with pytest.raises(SpeculativeDecodingPinnedError, match=SERVE_FLAG) as refused:
        refuse_unpinned_speculative_change(base_args, base_envs, candidate_args, candidate_envs)
    assert refused.value.error_class == SERVE_FLAG


@pytest.mark.parametrize(
    ("base_args", "base_envs", "candidate_args", "candidate_envs"),
    [
        pytest.param(
            VLLM_SYNTHETIC,
            {},
            VLLM_SYNTHETIC + " --max-num-seqs 256 --gpu-memory-utilization 0.92",
            {"VLLM_ROCM_USE_AITER": "1"},
            id="vllm-other-knobs",
        ),
        pytest.param(
            VLLM_SYNTHETIC,
            {},
            _spec_config(attention_backend="TRITON_ATTN"),
            {},
            id="vllm-non-pinned-config-key",
        ),
        pytest.param(
            SGLANG_EAGLE,
            SGLANG_SYNTHETIC_ENV,
            SGLANG_EAGLE + " --cuda-graph-max-bs 128",
            {**SGLANG_SYNTHETIC_ENV, "SGLANG_USE_AITER": "1"},
            id="sglang-other-knobs",
        ),
        pytest.param(
            SGLANG_EAGLE,
            {"SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token"},
            SGLANG_EAGLE + " --speculative-num-steps 3",
            {"SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token"},
            id="glm-real-draft-token",
        ),
        pytest.param(
            _spec_config(rejection_sample_method="block", synthetic_acceptance_length=None),
            {},
            _spec_config(rejection_sample_method="block", synthetic_acceptance_length=None, num_speculative_tokens=5),
            {},
            id="kimi-block-verify",
        ),
        pytest.param(
            "--max-num-seqs 128",
            {},
            _spec_config(rejection_sample_method="block", synthetic_acceptance_length=None),
            {},
            id="no-speculative-decoding-adds-a-real-draft",
        ),
        pytest.param(
            VLLM_SYNTHETIC,
            {},
            VLLM_SYNTHETIC + " --speculative-config.num_speculative_tokens 3"
            " --speculative-config.synthetic_acceptance_length 2.780",
            {},
            id="vllm-dotted-keys-restating-the-json",
        ),
        pytest.param(
            SGLANG_EAGLE,
            SGLANG_SYNTHETIC_ENV,
            SGLANG_EAGLE,
            {"SGLANG_SIMULATE_ACC_LEN": "3.610"},
            id="sglang-acc-len-spelled-differently",
        ),
    ],
)
def test_a_candidate_that_keeps_the_pinned_draft_or_has_nothing_pinned_is_allowed(
    base_args, base_envs, candidate_args, candidate_envs
):
    refuse_unpinned_speculative_change(base_args, base_envs, candidate_args, candidate_envs)


@pytest.mark.parametrize(
    ("base_args", "base_envs", "candidate_args", "candidate_envs"),
    [
        pytest.param(
            _spec_config(rejection_sample_method="block", synthetic_acceptance_length=None),
            {},
            _spec_config(),
            {},
            id="vllm-block-verify-to-synthetic",
        ),
        pytest.param(
            SGLANG_EAGLE,
            {"SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token"},
            SGLANG_EAGLE,
            {"SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token", "SGLANG_SIMULATE_ACC_LEN": "4.0"},
            id="sglang-real-draft-to-simulated-length",
        ),
    ],
)
def test_a_candidate_may_not_declare_the_acceptance_a_real_draft_base_measures(
    base_args, base_envs, candidate_args, candidate_envs
):
    with pytest.raises(SpeculativeDecodingPinnedError, match="verifies a real draft"):
        refuse_unpinned_speculative_change(base_args, base_envs, candidate_args, candidate_envs)


@pytest.mark.parametrize(
    ("base_args", "base_envs", "candidate_args", "candidate_envs"),
    [
        pytest.param("--max-num-seqs 128", {}, _spec_config(), {}, id="vllm-enables-a-synthetic-draft"),
        pytest.param(
            "--max-num-seqs 128",
            {},
            "--max-num-seqs 128 --speculative-algorithm EAGLE --speculative-num-steps 3",
            {"SGLANG_SIMULATE_ACC_LEN": "2.78"},
            id="sglang-enables-a-simulated-draft",
        ),
    ],
)
def test_a_base_with_no_speculative_decoding_may_enable_it_with_a_simulated_acceptance(
    base_args, base_envs, candidate_args, candidate_envs
):
    """There is no acceptance on the base to compare against; the golden value is the agent's to look up."""
    refuse_unpinned_speculative_change(base_args, base_envs, candidate_args, candidate_envs)


def test_the_refusal_names_what_changed():
    with pytest.raises(SpeculativeDecodingPinnedError) as refused:
        refuse_unpinned_speculative_change(
            SGLANG_EAGLE, SGLANG_SYNTHETIC_ENV, SGLANG_EAGLE + " --speculative-num-steps 3", SGLANG_SYNTHETIC_ENV
        )
    assert "--speculative-num-steps ('5' -> '3')" in str(refused.value)


def _admission(variant: GridVariant, *, framework: str, base_args: str, base_envs: dict | None = None):
    """What a round's pre-flight filter makes of one proposal, on a base launch of ``base_args``/``base_envs``."""
    return apply_compatibility_filter(
        [variant],
        serving=ServingStack(framework, ""),
        base_server_args=base_args,
        base_envs=base_envs or {},
    )


@pytest.mark.parametrize(
    ("framework", "base_args", "base_envs", "variant"),
    [
        pytest.param(
            "vllm",
            VLLM_SYNTHETIC,
            {},
            GridVariant("v", extra_server_args=_spec_config(num_speculative_tokens=5)),
            id="changes-the-draft",
        ),
        pytest.param(
            "vllm",
            VLLM_SYNTHETIC,
            {},
            GridVariant("v", extra_server_args=_spec_config(synthetic_acceptance_length=2.0), args_mode="replace"),
            id="replace-restates-another-acceptance",
        ),
        pytest.param(
            "vllm", VLLM_SYNTHETIC, {}, GridVariant("v", remove_args=["--speculative-config"]), id="removes-the-draft"
        ),
        pytest.param(
            "sglang",
            SGLANG_EAGLE,
            SGLANG_SYNTHETIC_ENV,
            GridVariant("v", extra_server_args="--speculative-num-steps 3"),
            id="changes-steps",
        ),
        pytest.param(
            "sglang",
            SGLANG_EAGLE,
            SGLANG_SYNTHETIC_ENV,
            GridVariant("v", unset_envs=["SGLANG_SIMULATE_ACC_LEN"]),
            id="unsets-acc-len",
        ),
        pytest.param(
            "sglang",
            SGLANG_EAGLE,
            SGLANG_SYNTHETIC_ENV,
            GridVariant("v", extra_envs={"SGLANG_SIMULATE_ACC_LEN": "2.99"}),
            id="retunes-acc-len",
        ),
        pytest.param(
            "sglang",
            SGLANG_EAGLE,
            {"SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token"},
            GridVariant("v", extra_envs={"SGLANG_SIMULATE_ACC_LEN": "4.0"}),
            id="declares-the-acceptance-a-real-draft-measures",
        ),
    ],
)
def test_a_round_does_not_admit_a_variant_that_moves_a_lever_the_base_pins(framework, base_args, base_envs, variant):
    """The proposal is refused where it is admitted, so the round keeps the slot rather than losing it at launch."""
    kept, dropped = _admission(variant, framework=framework, base_args=base_args, base_envs=base_envs)
    assert not kept
    assert [(d["name"], d["source"]) for d in dropped] == [("v", SERVE_FLAG)]
    assert SERVE_FLAG in dropped[0]["reason"]


@pytest.mark.parametrize(
    ("framework", "base_args", "base_envs", "variant"),
    [
        pytest.param(
            "vllm", VLLM_SYNTHETIC, {}, GridVariant("v", extra_server_args="--max-num-seqs 256"), id="vllm-other-knobs"
        ),
        pytest.param(
            "vllm",
            VLLM_SYNTHETIC,
            {},
            GridVariant("v", extra_server_args="--max-num-seqs 64", args_mode="replace"),
            id="replace-keeps-the-base-draft",
        ),
        pytest.param(
            "sglang",
            SGLANG_EAGLE,
            SGLANG_SYNTHETIC_ENV,
            GridVariant("v", extra_envs={"SGLANG_USE_AITER": "1"}),
            id="sglang-other-knobs",
        ),
        pytest.param(
            "sglang",
            SGLANG_EAGLE,
            {"SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token"},
            GridVariant("v", extra_server_args="--speculative-num-steps 3"),
            id="real-draft-base-leaves-the-draft-open",
        ),
    ],
)
def test_a_round_admits_a_variant_that_leaves_the_pin_where_the_base_set_it(framework, base_args, base_envs, variant):
    kept, dropped = _admission(variant, framework=framework, base_args=base_args, base_envs=base_envs)
    assert [v.name for v in kept] == ["v"] and dropped == []


def test_admission_and_the_launch_state_one_rule():
    """The filter asks the rule of a proposal and the launch states it as a refusal: same words, same verdict."""
    moved = GridVariant("v", extra_server_args="--speculative-num-steps 3")
    _, dropped = _admission(moved, framework="sglang", base_args=SGLANG_EAGLE, base_envs=SGLANG_SYNTHETIC_ENV)
    with pytest.raises(SpeculativeDecodingPinnedError) as refused:
        refuse_unpinned_speculative_change(
            SGLANG_EAGLE, SGLANG_SYNTHETIC_ENV, SGLANG_EAGLE + " --speculative-num-steps 3", SGLANG_SYNTHETIC_ENV
        )
    assert dropped[0]["reason"] == str(refused.value)
    assert (
        speculative_pin_reason(
            SGLANG_EAGLE, SGLANG_SYNTHETIC_ENV, SGLANG_EAGLE + " --cuda-graph-max-bs 128", SGLANG_SYNTHETIC_ENV
        )
        is None
    )


def test_a_round_admits_against_the_environment_its_base_launches_with():
    """A name the accepted stack unsets is not part of the base a proposal is judged against."""
    from hyperloom.orchestrator.actions.executors.explore import _ExploreRound

    rnd = _ExploreRound.__new__(_ExploreRound)
    rnd.grid = [GridVariant("drop_simulated_acceptance", unset_envs=["SGLANG_SIMULATE_ACC_LEN"])]
    rnd.params = {}
    rnd.ss = None
    rnd.framework = "sglang"
    rnd.target = BenchmarkTarget(model_path="", gpu_type="mi355x")
    rnd.yaml_envs = dict(SGLANG_SYNTHETIC_ENV)
    rnd.envs = {}
    rnd.unset_envs = ["SGLANG_SIMULATE_ACC_LEN"]
    rnd.base_args = SGLANG_EAGLE
    rnd.base_remove_args = []
    rnd.inherited_args = ""
    rnd.args_mode = "append"
    rnd.skipped = []
    rnd._drop_leading_noops = lambda: None
    rnd._shape_runnable()
    assert [gv.name for gv in rnd.runnable] == ["drop_simulated_acceptance"]
    assert rnd.skipped == []


def _base_yaml(tmp_path: Path, framework: str, args: str, envs: dict[str, str]) -> Path:
    args_env = {"vllm": "EXTRA_VLLM_ARGS", "sglang": "EXTRA_SGLANG_ARGS"}[framework]
    path = tmp_path / "base.yaml"
    body = {
        "framework": framework,
        "model": str(tmp_path / "model"),
        "run_mode": "local",
        "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256, args_env: args, **envs},
    }
    path.write_text(yaml.safe_dump({"benchmark": body}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _no_launcher_checkout(monkeypatch):
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SERVER_ARGS", raising=False)


def _build(tmp_path: Path, base: Path, variant: GridVariant, stack: LaunchLayer = LaunchLayer()) -> dict:
    out = _build_variant_yaml(GridRecipe(base_yaml_path=base, base=stack), variant, output_subdir=tmp_path / "v")
    return yaml.safe_load(out.read_text(encoding="utf-8"))["benchmark"]["envs"]


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            GridVariant("v", extra_server_args=_spec_config(num_speculative_tokens=5)), id="changes-the-draft"
        ),
        pytest.param(
            GridVariant("v", extra_server_args="--max-num-seqs 64", args_mode="replace"), id="replace-drops-it"
        ),
        pytest.param(GridVariant("v", remove_args=["--speculative-config"]), id="removes-it"),
    ],
)
def test_building_a_variant_refuses_a_change_to_the_inherited_synthetic_draft(tmp_path, variant):
    base = _base_yaml(tmp_path, "vllm", VLLM_SYNTHETIC, {})
    with pytest.raises(SpeculativeDecodingPinnedError):
        _build(tmp_path, base, variant)


def test_building_a_variant_keeps_other_knobs_open_on_a_synthetic_base(tmp_path):
    base = _base_yaml(tmp_path, "vllm", VLLM_SYNTHETIC, {})
    envs = _build(tmp_path, base, GridVariant("v", extra_server_args="--max-num-seqs 256"))
    assert "--max-num-seqs 256" in envs["EXTRA_VLLM_ARGS"]
    assert '"synthetic_acceptance_length":2.78' in envs["EXTRA_VLLM_ARGS"]


def test_building_a_variant_pins_the_draft_a_replacing_stack_carries(tmp_path):
    base = _base_yaml(tmp_path, "vllm", "", {})
    stack = LaunchLayer(extra_server_args=VLLM_SYNTHETIC, args_mode="replace")
    _build(tmp_path, base, GridVariant("ok", extra_server_args="--max-num-seqs 256", args_mode="replace"), stack)
    with pytest.raises(SpeculativeDecodingPinnedError):
        _build(
            tmp_path,
            base,
            GridVariant("v", extra_server_args=_spec_config(synthetic_acceptance_length=3.0), args_mode="replace"),
            stack,
        )


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(GridVariant("v", unset_envs=["SGLANG_SIMULATE_ACC_LEN"]), id="unsets-acc-len"),
        pytest.param(GridVariant("v", extra_server_args="--speculative-num-steps 3"), id="changes-steps"),
    ],
)
def test_building_a_variant_refuses_a_change_to_the_simulated_acceptance_env(tmp_path, variant):
    base = _base_yaml(tmp_path, "sglang", SGLANG_EAGLE, SGLANG_SYNTHETIC_ENV)
    with pytest.raises(SpeculativeDecodingPinnedError):
        _build(tmp_path, base, variant)


def test_building_a_variant_leaves_a_real_draft_base_open(tmp_path):
    base = _base_yaml(tmp_path, "sglang", SGLANG_EAGLE, {"SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token"})
    envs = _build(tmp_path, base, GridVariant("v", extra_server_args="--speculative-num-steps 3"))
    assert "--speculative-num-steps 3" in envs["EXTRA_SGLANG_ARGS"]
    assert "--speculative-num-steps 5" not in envs["EXTRA_SGLANG_ARGS"]


@pytest.mark.asyncio
async def test_run_grid_records_the_refusal_without_launching(tmp_path, monkeypatch):
    monkeypatch.setattr(gr, "run_grid_suite_round", lambda *_a, **_k: pytest.fail("a refused variant was launched"))
    base = _base_yaml(tmp_path, "vllm", VLLM_SYNTHETIC, {})
    results = await run_grid(
        grid=[GridVariant("more-draft", extra_server_args=_spec_config(num_speculative_tokens=5))],
        output_root=tmp_path / "out",
        recipe=GridRecipe(base_yaml_path=base),
    )
    assert [(r.status, r.error_class) for r in results] == [("failed", SERVE_FLAG)]
    assert "num_speculative_tokens" in results[0].error
    assert not results[0].launch_evidence


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("candidate", "key"),
    [
        pytest.param(_spec_config(num_speculative_tokens=5), "num_speculative_tokens", id="draft-length"),
        pytest.param(
            "--speculative-config.enable_adaptive_verification true",
            "enable_adaptive_verification",
            id="adaptive-verification",
        ),
    ],
)
async def test_a_variant_touching_golden_fails_as_serve_flag_naming_the_key(tmp_path, monkeypatch, candidate, key):
    monkeypatch.setattr(gr, "run_grid_suite_round", lambda *_a, **_k: pytest.fail("a refused variant was launched"))
    results = await run_grid(
        grid=[GridVariant("touches-golden", extra_server_args=candidate)],
        output_root=tmp_path / "out",
        recipe=GridRecipe(base_yaml_path=_base_yaml(tmp_path, "vllm", VLLM_SYNTHETIC, {})),
    )
    assert [(r.status, r.error_class) for r in results] == [("failed", SERVE_FLAG)]
    assert f"{SERVE_FLAG}: golden-owned --speculative-config.{key}" in results[0].error


def _native_recipe(
    tmp_path: Path,
    recipe_args: list[str] | None = None,
    recipe_envs: dict[str, str | None] | None = None,
) -> Path:
    """A native recipe as a verified session leaves it: the resolved workload and the launch receipt beside it.

    A native launch spells its server args in its own resolved argv and its settings in the environment it exports
    itself, not in ``launch_overrides`` or the workload's ``envs``, so the receipt is the only place a layer
    composing on top can read the draft and the simulated acceptance the recipe already pinned.
    """
    source = tmp_path / "workload.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "model": "Qwen/Qwen3-0.6B",
                    "framework": "sglang",
                    "precision": "bf16",
                    "runner_type": "mi355x",
                    "agentx": {"enabled": True, "mode": "canonical"},
                    "envs": {"TP": "1", "EP_SIZE": "1", "CONC": "8"},
                }
            }
        ),
        encoding="utf-8",
    )
    config = materialize_benchmark_config(source, tmp_path / "recipe")
    argv = NATIVE_PREFIX + list(recipe_args or [])
    persist_launch_evidence(native_receipt(config, argv, dict(recipe_envs or {})), slot=config.parent)
    return config


def _build_native(tmp_path: Path, config: Path, variant: GridVariant, stack: LaunchLayer = LaunchLayer()) -> dict:
    out = _build_variant_yaml(
        GridRecipe(base_yaml_path=config, base=stack), variant, output_subdir=tmp_path / variant.name
    )
    return yaml.safe_load(out.read_text(encoding="utf-8"))["benchmark"]["agentx"]["launch_overrides"]


def test_building_a_native_variant_keeps_other_knobs_open(tmp_path):
    config = _native_recipe(tmp_path, DSPARK_RECIPE_ARGS, SGLANG_SYNTHETIC_ENV)
    launch = _build_native(tmp_path, config, GridVariant("v", extra_server_args="--mem-fraction-static 0.85"))
    assert launch["append_args"] == ["--mem-fraction-static", "0.85"]


def test_a_variable_the_recipe_cleared_is_not_a_setting_it_pinned(tmp_path):
    """A ``None`` in the receipt records a name the launch cleared, so this recipe verifies a real draft."""
    config = _native_recipe(tmp_path, DSPARK_RECIPE_ARGS, {"SGLANG_SIMULATE_ACC_LEN": None})
    launch = _build_native(tmp_path, config, GridVariant("v", extra_server_args="--speculative-num-draft-tokens 6"))
    assert launch["append_args"] == ["--speculative-num-draft-tokens", "6"]


def test_building_a_native_variant_pins_the_draft_the_accepted_stack_carries(tmp_path):
    """The session's own accepted layer is what the pin is measured against, not a candidate to refuse."""
    config = _native_recipe(tmp_path)
    stack = LaunchLayer(extra_server_args=SGLANG_EAGLE, extra_envs=SGLANG_SYNTHETIC_ENV)
    launch = _build_native(tmp_path, config, GridVariant("ok", extra_server_args="--mem-fraction-static 0.85"), stack)
    assert "--speculative-num-steps" in launch["append_args"]
    with pytest.raises(SpeculativeDecodingPinnedError):
        _build_native(tmp_path, config, GridVariant("v", extra_server_args="--speculative-num-steps 3"), stack)


def test_resolving_the_session_launch_may_restate_the_recipes_draft(tmp_path):
    """Materializing the accepted launch admits it as it is; only a candidate composed on top is measured."""
    config = _native_recipe(tmp_path, DSPARK_RECIPE_ARGS, SGLANG_SYNTHETIC_ENV)
    resolved = materialize_benchmark_config(
        config, tmp_path / "accepted", launch=LaunchLayer(extra_server_args="--speculative-num-draft-tokens 6")
    )
    launch = yaml.safe_load(resolved.read_text(encoding="utf-8"))["benchmark"]["agentx"]["launch_overrides"]
    assert launch["append_args"] == ["--speculative-num-draft-tokens", "6"]
