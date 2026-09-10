# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""sglang ``--moe-runner-backend`` tests.

Hyperloom does NOT force a ``--moe-runner-backend`` for MoE sglang models on
AMD: sglang's own ``auto`` resolution already follows ``SGLANG_USE_AITER``
correctly on current sglang/ROCm images. What remains is the aiter-only quant
scheme detection (``moe_runner_requires_aiter``), used elsewhere to strip an
*inherited* ``--moe-runner-backend`` that would crash such a checkpoint (grid
variants, baseline retries), and ``materialize_config_with_envs`` never adding
the flag on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.cli import model_gate as cli_model_gate
from hyperloom.orchestrator.actions.executors._workload_envs import (
    _remove_moe_runner_backend_arg,
    materialize_config_with_envs,
)

_AMD = "mi300x"


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Neutralise host GPU autodetect + env so AMD-gating is deterministic."""
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    # Pin autodetect OFF so non-AMD test cases never see real hardware.
    monkeypatch.setattr(cli_model_gate, "_autodetect_gpu_type", lambda: None)
    for key in (
        "CONC",
        "ISL",
        "OSL",
        "MAX_MODEL_LEN",
        "TP",
        "RANDOM_RANGE_RATIO",
        "ROCR_VISIBLE_DEVICES",
        "PRECISION",
        "RUN_EVAL",
        "FRAMEWORK",
    ):
        monkeypatch.delenv(key, raising=False)


def _write_model_config(dir_path: Path, config: dict) -> str:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(dir_path)


@pytest.fixture
def moe_model(tmp_path) -> str:
    """A Qwen3-MoE-style checkpoint dir (declares an expert count)."""
    return _write_model_config(
        tmp_path / "Qwen-Qwen3-30B-A3B",
        {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "num_experts": 128,
        },
    )


@pytest.fixture
def dense_model(tmp_path) -> str:
    """A dense (non-MoE) checkpoint dir."""
    return _write_model_config(
        tmp_path / "Qwen-Qwen3-8B",
        {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
    )


def _mx_fp4_spec(**overrides) -> dict:
    """A Quark spec dict that satisfies sglang's ``_is_mx_fp4``."""
    spec = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
        "is_dynamic": False,
    }
    spec.update(overrides)
    return spec


def _mx_fp4_entry(**overrides) -> dict:
    return {
        "weight": _mx_fp4_spec(),
        "input_tensors": _mx_fp4_spec(is_dynamic=True),
        **overrides,
    }


@pytest.fixture
def quark_mxfp4_moe_model(tmp_path) -> str:
    """A Quark-PTQ MXFP4 (W4A4) MoE checkpoint dir."""
    return _write_model_config(
        tmp_path / "Qwen3.5-397B-A17B-MXFP4",
        {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "num_experts": 128,
            "quantization_config": {
                "quant_method": "quark",
                "global_quant_config": _mx_fp4_entry(),
            },
        },
    )


# _model_is_moe detection
@pytest.mark.parametrize(
    "config",
    [
        {"num_experts": 128},
        {"num_local_experts": 8},
        {"n_routed_experts": 64},
        {"moe_intermediate_size": 768},
        {"model_type": "qwen3_moe"},
        {"architectures": ["Qwen3MoeForCausalLM"]},
        {"text_config": {"num_experts": 16}},
    ],
)
def test_model_is_moe_true(tmp_path, config):
    path = _write_model_config(tmp_path / "m", config)
    assert cli_model_gate._model_is_moe(path) is True


@pytest.mark.parametrize(
    "config",
    [
        {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
        {"num_experts": 1},  # single "expert" is not MoE
        {"num_experts": True},  # bool must not count as an int expert count
        {},
    ],
)
def test_model_is_moe_false(tmp_path, config):
    path = _write_model_config(tmp_path / "m", config)
    assert cli_model_gate._model_is_moe(path) is False


def test_model_is_moe_missing_config_is_false(tmp_path):
    assert cli_model_gate._model_is_moe(str(tmp_path / "does-not-exist")) is False


# _model_moe_runner_requires_aiter detection
@pytest.mark.parametrize(
    "quant_config",
    [
        {"quant_method": "quark", "global_quant_config": _mx_fp4_entry()},
        # The MoE layers alone carry MX-FP4; sglang resolves per-layer first.
        {
            "quant_method": "quark",
            "global_quant_config": {"weight": {"dtype": "fp8_e4m3", "qscheme": "per_tensor"}},
            "layer_quant_config": {"*mlp.experts*": _mx_fp4_entry()},
        },
        {"quant_method": "quark", "layer_type_quant_config": {"FusedMoE": _mx_fp4_entry()}},
    ],
)
def test_moe_runner_requires_aiter_true(tmp_path, quant_config):
    path = _write_model_config(
        tmp_path / "m",
        {"num_experts": 128, "quantization_config": quant_config},
    )
    assert cli_model_gate._model_moe_runner_requires_aiter(path) is True


@pytest.mark.parametrize(
    "quant_config",
    [
        None,  # unquantized MoE
        {"quant_method": "fp8"},
        # nvfp4 is fp4 too, but not per_group/e8m0 microscaling.
        {
            "quant_method": "quark",
            "global_quant_config": _mx_fp4_entry(weight=_mx_fp4_spec(qscheme="per_tensor", scale_format="float32")),
        },
        # fp4 with a non-MX group size.
        {"quant_method": "quark", "global_quant_config": _mx_fp4_entry(weight=_mx_fp4_spec(group_size=16))},
        # Keep exact parity with sglang: its _is_mx_fp4 compares against integer 32, so a string value is not a valid
        # MX-FP4 config either.
        {"quant_method": "quark", "global_quant_config": _mx_fp4_entry(weight=_mx_fp4_spec(group_size="32"))},
        # Statically quantized activations are not the W4A4 dynamic scheme.
        {
            "quant_method": "quark",
            "global_quant_config": _mx_fp4_entry(input_tensors=_mx_fp4_spec(is_dynamic=False)),
        },
        # Same shape, but a quant method sglang routes elsewhere.
        {"quant_method": "modelopt", "global_quant_config": _mx_fp4_entry()},
        {"quant_method": "quark", "global_quant_config": {"weight": {"dtype": "fp8_e4m3"}}},
    ],
)
def test_moe_runner_requires_aiter_false(tmp_path, quant_config):
    config: dict = {"num_experts": 128}
    if quant_config is not None:
        config["quantization_config"] = quant_config
    path = _write_model_config(tmp_path / "m", config)
    assert cli_model_gate._model_moe_runner_requires_aiter(path) is False


def test_moe_runner_requires_aiter_reads_nested_text_config(tmp_path):
    path = _write_model_config(
        tmp_path / "m",
        {
            "text_config": {
                "num_experts": 128,
                "quantization_config": {"quant_method": "quark", "global_quant_config": _mx_fp4_entry()},
            }
        },
    )
    assert cli_model_gate._model_moe_runner_requires_aiter(path) is True


def test_moe_runner_requires_aiter_missing_config_is_false(tmp_path):
    assert cli_model_gate._model_moe_runner_requires_aiter(str(tmp_path / "nope")) is False


# materialize_config_with_envs (the production choke point)
def _write_yaml(path: Path, *, model: str, framework: str = "sglang") -> None:
    cfg = {
        "benchmark": {
            "framework": framework,
            "model": model,
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _materialize_envs(
    tmp_path: Path,
    *,
    model: str,
    framework: str = "sglang",
    extra_server_args: str = "",
    drop_moe_runner_backend: bool = False,
) -> dict:
    base = tmp_path / "base.yaml"
    _write_yaml(base, model=model, framework=framework)
    out = tmp_path / "out"
    out.mkdir()
    materialized = materialize_config_with_envs(
        base,
        out,
        extra_server_args=extra_server_args,
        drop_moe_runner_backend=drop_moe_runner_backend,
    )
    return yaml.safe_load(materialized.read_text())["benchmark"]["envs"]


def test_materialize_never_injects_moe_runner_backend_on_amd(tmp_path, moe_model, monkeypatch):
    # sglang's own --moe-runner-backend auto already picks the right backend;
    # Hyperloom leaves it alone even for a MoE model on AMD/ROCm.
    monkeypatch.setenv("GPU_TYPE", _AMD)
    envs = _materialize_envs(tmp_path, model=moe_model)
    assert "--moe-runner-backend" not in envs.get("EXTRA_SGLANG_ARGS", "")


def test_materialize_noop_for_dense_model_on_amd(tmp_path, dense_model, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", _AMD)
    envs = _materialize_envs(tmp_path, model=dense_model)
    assert "--moe-runner-backend" not in envs.get("EXTRA_SGLANG_ARGS", "")


def test_materialize_noop_for_quark_mxfp4_moe_on_amd(tmp_path, quark_mxfp4_moe_model, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", _AMD)
    envs = _materialize_envs(tmp_path, model=quark_mxfp4_moe_model)
    assert "--moe-runner-backend" not in envs.get("EXTRA_SGLANG_ARGS", "")


def test_materialize_drop_flag_is_noop_without_pin(tmp_path, moe_model, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", _AMD)
    envs = _materialize_envs(tmp_path, model=moe_model, drop_moe_runner_backend=True)
    assert "--moe-runner-backend" not in envs.get("EXTRA_SGLANG_ARGS", "")


def test_materialize_drop_strips_inherited_backend(tmp_path, moe_model, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", _AMD)
    envs = _materialize_envs(
        tmp_path,
        model=moe_model,
        extra_server_args="--moe-runner-backend ck --foo 1",
        drop_moe_runner_backend=True,
    )
    sglang_args = envs.get("EXTRA_SGLANG_ARGS", "")
    assert "--moe-runner-backend" not in sglang_args
    assert "--foo 1" in sglang_args


def test_materialize_does_not_double_user_backend(tmp_path, moe_model, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", _AMD)
    envs = _materialize_envs(
        tmp_path,
        model=moe_model,
        extra_server_args="--moe-runner-backend ck",
    )
    sglang_args = envs["EXTRA_SGLANG_ARGS"]
    assert sglang_args.count("--moe-runner-backend") == 1
    assert "ck" in sglang_args
    assert "triton" not in sglang_args


@pytest.mark.parametrize(
    "args, expected",
    [
        ("--foo 1 --moe-runner-backend triton --bar 2", "--foo 1 --bar 2"),
        ("--foo 1 --moe-runner-backend=triton", "--foo 1"),
        ("--moe-runner-backend aiter", ""),
        # Value-less flag: strip it rather than let it survive a retry.
        ("--foo 1 --moe-runner-backend", "--foo 1"),
        ("--moe-runner-backend", ""),
        ("--moe-runner-backend --foo 1", "--foo 1"),
    ],
)
def test_remove_moe_runner_backend_arg(args, expected):
    assert _remove_moe_runner_backend_arg(args) == expected


# aiter CK fused-MoE shape support (gates the forge fmoe_ck tuner)


@pytest.mark.parametrize(
    "tp,supported",
    [(1, True), (2, True), (4, False), (8, False), (16, False)],
)
def test_aiter_ck_fused_moe_needs_128_aligned_partition(tmp_path, tp, supported) -> None:
    """Qwen3-30B-A3B's 768 shards below the CK kernel's 128 alignment past TP 2."""
    model = _write_model_config(
        tmp_path / "Qwen-Qwen3-30B-A3B",
        {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "num_experts": 128,
            "moe_intermediate_size": 768,
        },
    )
    assert cli_model_gate.model_supports_aiter_ck_fused_moe(model, tp) is supported


def test_aiter_ck_fused_moe_support_defaults_open(tmp_path, dense_model) -> None:
    """Dense models and unreadable configs leave the choice to sglang."""
    # Never reaches the MoE kernel, so nothing to gate.
    assert cli_model_gate.model_supports_aiter_ck_fused_moe(dense_model, 8) is True
    # No config to judge by: do not skip work on a guess.
    assert cli_model_gate.model_supports_aiter_ck_fused_moe(str(tmp_path / "absent"), 8) is True
    # MoE without a declared intermediate size is equally undecidable.
    moe_no_size = _write_model_config(
        tmp_path / "moe-no-size",
        {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "num_experts": 128},
    )
    assert cli_model_gate.model_supports_aiter_ck_fused_moe(moe_no_size, 8) is True
