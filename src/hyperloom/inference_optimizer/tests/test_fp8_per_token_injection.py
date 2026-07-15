# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""sglang ``SGLANG_USE_AITER_FP8_PER_TOKEN`` injection tests.

Hyperloom injects the env from its env-materialization choke point, strictly
scoped to sglang + fp8 + gfx942 + per-channel weight/per-token dynamic act,
never clobbering an operator-set value. Exercised at both the pure
config-detection layer and the ``materialize_config_with_envs`` layer.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.cli import model_gate as cli_model_gate
from hyperloom.inference_optimizer.model_config_utils import _fp8_is_per_channel_per_token
from hyperloom.orchestrator.actions.executors._workload_envs import (
    materialize_config_with_envs,
)

_AMD = "mi300x"
_ENV = "SGLANG_USE_AITER_FP8_PER_TOKEN"


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Neutralise host GPU autodetect + env so AMD-gating is deterministic."""
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
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
        _ENV,
    ):
        monkeypatch.delenv(key, raising=False)


def _write_model_config(dir_path: Path, config: dict) -> str:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(dir_path)


def _write_safetensors(path: Path, tensors: dict[str, list[int]]) -> None:
    """Write a minimal valid ``.safetensors`` file (header + zeroed data).

    ``tensors`` maps tensor name -> shape. Only the header (which carries the
    shape the gate inspects) needs to be correct; payload bytes are zeros.
    """
    header: dict[str, object] = {}
    blob = b""
    offset = 0
    for name, shape in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * 4  # F32 scales
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        blob += b"\x00" * nbytes
        offset += nbytes
    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(blob)


def _write_fp8_weights(dir_path: Path, *, per_channel: bool) -> None:
    """Drop a safetensors shard carrying a representative ``weight_scale`` tensor.

    Per-channel -> shape ``[out_features, 1]`` (numel > 1); per-tensor -> scalar
    ``[]`` (numel == 1).
    """
    shape = [16, 1] if per_channel else []
    _write_safetensors(
        dir_path / "model.safetensors",
        {
            "model.layers.0.mlp.down_proj.weight": [16, 16],
            "model.layers.0.mlp.down_proj.weight_scale": shape,
        },
    )


def _fp8_quant_config(**overrides) -> dict:
    """A standard HF FP8 quantization_config (per-channel/per-token dynamic)."""
    qc = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3"}
    qc.update(overrides)
    return qc


@pytest.fixture
def fp8_dynamic_model(tmp_path) -> str:
    """Dense FP8 checkpoint with per-channel weight + per-token dynamic act."""
    path = _write_model_config(
        tmp_path / "org-fp8-dynamic",
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "quantization_config": _fp8_quant_config(),
        },
    )
    _write_fp8_weights(Path(path), per_channel=True)
    return path


# ── pure config-detection layer ────────────────────────────────────────────
def test_detect_true_for_fp8_dynamic_no_block(fp8_dynamic_model):
    assert _fp8_is_per_channel_per_token(fp8_dynamic_model) is True


def test_detect_true_when_activation_scheme_absent(tmp_path):
    # Absent activation_scheme defaults to dynamic in sglang's Fp8Config.
    path = _write_model_config(
        tmp_path / "m",
        {"quantization_config": {"quant_method": "fp8", "fmt": "e4m3"}},
    )
    _write_fp8_weights(Path(path), per_channel=True)
    assert _fp8_is_per_channel_per_token(path) is True


def test_detect_false_for_per_tensor_weight(tmp_path):
    # Per-tensor weight already serves the fast fused path; the gate must decline.
    path = _write_model_config(
        tmp_path / "per-tensor",
        {"quantization_config": _fp8_quant_config()},
    )
    _write_fp8_weights(Path(path), per_channel=False)
    assert _fp8_is_per_channel_per_token(path) is False


def test_detect_false_when_weight_granularity_undeterminable(tmp_path):
    # No safetensors to confirm per-channel weights: default-safe -> decline.
    path = _write_model_config(
        tmp_path / "no-weights",
        {"quantization_config": _fp8_quant_config()},
    )
    assert _fp8_is_per_channel_per_token(path) is False


def test_detect_false_for_static_activation(tmp_path):
    # Static activation is per-tensor; forcing the env would regress it.
    path = _write_model_config(
        tmp_path / "m",
        {"quantization_config": _fp8_quant_config(activation_scheme="static")},
    )
    assert _fp8_is_per_channel_per_token(path) is False


def test_detect_false_for_block_scale(tmp_path):
    # Block-scale FP8 takes the w8a8_block_fp8_linear path; never touch it.
    path = _write_model_config(
        tmp_path / "m",
        {"quantization_config": _fp8_quant_config(weight_block_size=[128, 128])},
    )
    assert _fp8_is_per_channel_per_token(path) is False


def test_detect_false_for_non_fp8_quant_method(tmp_path):
    path = _write_model_config(
        tmp_path / "m",
        {"quantization_config": {"quant_method": "compressed-tensors"}},
    )
    assert _fp8_is_per_channel_per_token(path) is False


def test_detect_false_for_unquantized_model(tmp_path):
    path = _write_model_config(
        tmp_path / "m",
        {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
    )
    assert _fp8_is_per_channel_per_token(path) is False


def test_detect_false_for_missing_config(tmp_path):
    assert _fp8_is_per_channel_per_token(str(tmp_path / "nope")) is False


# ── materialize_config_with_envs (production choke point) ───────────────────
def _write_yaml(path: Path, *, model: str, framework: str, precision: str) -> None:
    cfg = {
        "benchmark": {
            "framework": framework,
            "model": model,
            "precision": precision,
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
    precision: str = "fp8",
    gpu_type: str | None = _AMD,
    extra_envs: dict | None = None,
) -> dict:
    base = tmp_path / "base.yaml"
    _write_yaml(base, model=model, framework=framework, precision=precision)
    out = tmp_path / "out"
    out.mkdir()
    materialized = materialize_config_with_envs(
        base,
        out,
        model_path=model,
        gpu_type=gpu_type,
        extra_envs=extra_envs,
    )
    return yaml.safe_load(materialized.read_text())["benchmark"]["envs"]


def test_materialize_injects_for_sglang_fp8_dynamic_on_gfx942(
    tmp_path, fp8_dynamic_model
):
    """The fast-path env must be materialized for this workload."""
    envs = _materialize_envs(tmp_path, model=fp8_dynamic_model)
    assert envs.get(_ENV) == "1"


def test_materialize_noop_for_non_sglang(tmp_path, fp8_dynamic_model):
    envs = _materialize_envs(tmp_path, model=fp8_dynamic_model, framework="vllm")
    assert _ENV not in envs


def test_materialize_noop_for_non_fp8_precision(tmp_path, fp8_dynamic_model):
    envs = _materialize_envs(tmp_path, model=fp8_dynamic_model, precision="bf16")
    assert _ENV not in envs


def test_materialize_noop_on_non_gfx942_gpu(tmp_path, fp8_dynamic_model):
    # MI355X is gfx950 and ships a different kernel; do not inject.
    envs = _materialize_envs(tmp_path, model=fp8_dynamic_model, gpu_type="mi355x")
    assert _ENV not in envs


def test_materialize_noop_for_block_scale_fp8(tmp_path):
    model = _write_model_config(
        tmp_path / "blockscale",
        {"quantization_config": _fp8_quant_config(weight_block_size=[128, 128])},
    )
    envs = _materialize_envs(tmp_path, model=model)
    assert _ENV not in envs


def test_materialize_noop_for_static_fp8(tmp_path):
    model = _write_model_config(
        tmp_path / "static",
        {"quantization_config": _fp8_quant_config(activation_scheme="static")},
    )
    envs = _materialize_envs(tmp_path, model=model)
    assert _ENV not in envs


def test_materialize_noop_for_per_tensor_weight(tmp_path):
    # A per-tensor fp8 + dynamic checkpoint must NOT get the env materialized.
    model = _write_model_config(
        tmp_path / "per-tensor",
        {"quantization_config": _fp8_quant_config()},
    )
    _write_fp8_weights(Path(model), per_channel=False)
    envs = _materialize_envs(tmp_path, model=model)
    assert _ENV not in envs


def test_materialize_does_not_clobber_operator_value(tmp_path, fp8_dynamic_model):
    envs = _materialize_envs(
        tmp_path,
        model=fp8_dynamic_model,
        extra_envs={_ENV: "0"},
    )
    assert envs.get(_ENV) == "0"


def test_materialize_injects_for_gfx942_sibling_mi325x(tmp_path, fp8_dynamic_model):
    envs = _materialize_envs(tmp_path, model=fp8_dynamic_model, gpu_type="mi325x")
    assert envs.get(_ENV) == "1"
