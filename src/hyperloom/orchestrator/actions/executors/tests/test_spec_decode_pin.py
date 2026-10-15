# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A candidate may not change the draft or the simulated acceptance of a base launch that simulates acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml, run_grid
from hyperloom.orchestrator.actions.executors._spec_decode_pin import (
    SPECULATIVE_DECODING_PINNED,
    SpeculativeDecodingPinnedError,
    refuse_unpinned_speculative_change,
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
    ],
)
def test_a_candidate_changing_the_pinned_draft_is_refused(base_args, base_envs, candidate_args, candidate_envs):
    with pytest.raises(SpeculativeDecodingPinnedError, match=SPECULATIVE_DECODING_PINNED) as refused:
        refuse_unpinned_speculative_change(base_args, base_envs, candidate_args, candidate_envs)
    assert refused.value.error_class == SPECULATIVE_DECODING_PINNED


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


def _build(tmp_path: Path, base: Path, variant: GridVariant, **base_layers) -> dict:
    out = _build_variant_yaml(
        base, base_layers.pop("base_extra_args", ""), variant, output_subdir=tmp_path / "v", **base_layers
    )
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
    stack = {"base_extra_args": VLLM_SYNTHETIC, "base_args_mode": "replace"}
    _build(tmp_path, base, GridVariant("ok", extra_server_args="--max-num-seqs 256", args_mode="replace"), **stack)
    with pytest.raises(SpeculativeDecodingPinnedError):
        _build(
            tmp_path,
            base,
            GridVariant("v", extra_server_args=_spec_config(synthetic_acceptance_length=3.0), args_mode="replace"),
            **stack,
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
    def _never_launched(**_kwargs):
        raise AssertionError("a refused variant must not reach Magpie")

    monkeypatch.setattr(gr, "_run_magpie", _never_launched)
    base = _base_yaml(tmp_path, "vllm", VLLM_SYNTHETIC, {})
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("more-draft", extra_server_args=_spec_config(num_speculative_tokens=5))],
        output_root=tmp_path / "out",
    )
    assert [(r.status, r.error_class) for r in results] == [("failed", SPECULATIVE_DECODING_PINNED)]
    assert "num_speculative_tokens" in results[0].error
    assert not results[0].launch_evidence
