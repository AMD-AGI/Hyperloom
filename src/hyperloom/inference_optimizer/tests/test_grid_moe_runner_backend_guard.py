# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Grid variants must not inherit a MoE runner backend the model cannot serve.

The grid never injects ``--moe-runner-backend`` itself, but it inherits one
from the baseline recipe it is seeded with (or from an authored variant). On an
aiter-only quant scheme that flag is a guaranteed first-forward-pass crash, and
the grid has no retry to salvage it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml


def _write_model_config(dir_path: Path, config: dict) -> str:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(dir_path)


def _mx_fp4_entry() -> dict:
    spec = {"dtype": "fp4", "qscheme": "per_group", "group_size": 32, "scale_format": "e8m0"}
    return {
        "weight": {**spec, "is_dynamic": False},
        "input_tensors": {**spec, "is_dynamic": True},
    }


@pytest.fixture
def quark_mxfp4_moe_model(tmp_path) -> str:
    return _write_model_config(
        tmp_path / "Qwen3.5-397B-A17B-MXFP4",
        {
            "model_type": "qwen3_5_moe",
            "num_experts": 128,
            "quantization_config": {"quant_method": "quark", "global_quant_config": _mx_fp4_entry()},
        },
    )


@pytest.fixture
def plain_moe_model(tmp_path) -> str:
    return _write_model_config(
        tmp_path / "Qwen3-30B-A3B",
        {"model_type": "qwen3_moe", "num_experts": 128},
    )


def _base_yaml(path: Path, model: str) -> Path:
    cfg = {
        "benchmark": {
            "framework": "sglang",
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
    return path


def _variant_args(tmp_path: Path, *, model: str, base_extra_args: str, variant_args: str = "") -> str:
    base = _base_yaml(tmp_path / "base.yaml", model)
    out = _build_variant_yaml(
        base,
        base_extra_args,
        GridVariant(name="v00", extra_server_args=variant_args, extra_envs={}),
        output_subdir=tmp_path / "v00",
        model_path=model,
        gpu_type="mi300x",
    )
    cfg = yaml.safe_load(out.read_text())
    return str(cfg["benchmark"]["envs"].get("EXTRA_SGLANG_ARGS", ""))


def test_inherited_backend_dropped_for_aiter_only_model(tmp_path, quark_mxfp4_moe_model):
    args = _variant_args(
        tmp_path,
        model=quark_mxfp4_moe_model,
        base_extra_args="--moe-runner-backend triton --mem-fraction-static 0.9",
    )
    assert "--moe-runner-backend" not in args
    assert "--mem-fraction-static 0.9" in args


def test_variant_authored_backend_dropped_for_aiter_only_model(tmp_path, quark_mxfp4_moe_model):
    args = _variant_args(
        tmp_path,
        model=quark_mxfp4_moe_model,
        base_extra_args="",
        variant_args="--moe-runner-backend triton",
    )
    assert "--moe-runner-backend" not in args


def test_backend_kept_for_ordinary_moe_model(tmp_path, plain_moe_model):
    args = _variant_args(
        tmp_path,
        model=plain_moe_model,
        base_extra_args="--moe-runner-backend triton",
    )
    assert "--moe-runner-backend triton" in args
