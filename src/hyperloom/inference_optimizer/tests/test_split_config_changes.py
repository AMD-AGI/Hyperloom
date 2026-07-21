# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for split_config_changes: server-flag vs env routing."""

from __future__ import annotations

import yaml
from pathlib import Path

from hyperloom.orchestrator.actions.executors._grid_server_args import split_config_changes
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    _build_variant_yaml,
)


# ---------------------------------------------------------------------------
# Unit tests for split_config_changes
# ---------------------------------------------------------------------------

def test_dash_prefixed_keys_become_server_args():
    args, envs = split_config_changes({"--tokenizer-mode": "deepseek_v4"})
    assert "deepseek_v4" in args
    assert "--tokenizer-mode" in args
    assert "tokenizer" not in envs


def test_env_keys_stay_in_envs():
    args, envs = split_config_changes({"VLLM_ROCM_USE_AITER": "1"})
    assert envs == {"VLLM_ROCM_USE_AITER": "1"}
    assert args == ""


def test_mixed_dict_splits_correctly():
    args, envs = split_config_changes({
        "--tokenizer-mode": "deepseek_v4",
        "VLLM_ROCM_USE_AITER": "1",
    })
    assert "--tokenizer-mode" in args
    assert "deepseek_v4" in args
    assert envs == {"VLLM_ROCM_USE_AITER": "1"}


def test_bare_flag_no_value():
    args, envs = split_config_changes({"--enable-mtp": ""})
    assert "--enable-mtp" in args
    assert envs == {}


def test_empty_input():
    args, envs = split_config_changes({})
    assert args == ""
    assert envs == {}


def test_legacy_flag_not_lost():
    """A --key is rebuilt into server_args, not silently dropped."""
    args, envs = split_config_changes({"--speculative-num-steps": "3"})
    assert "3" in args
    assert "--speculative-num-steps" in args
    assert not any(k.startswith("-") for k in envs)


# ---------------------------------------------------------------------------
# Integration test: flag reaches materialized YAML benchmark.envs
# ---------------------------------------------------------------------------

def _minimal_base_yaml(tmp_path: Path, framework: str = "vllm") -> Path:
    cfg = {
        "benchmark": {
            "framework": framework,
            "model": "/model",
            "envs": {},
        }
    }
    p = tmp_path / "base.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_tokenizer_flag_reaches_extra_vllm_args_in_yaml(tmp_path):
    """--tokenizer-mode deepseek_v4 must appear in EXTRA_VLLM_ARGS in the
    materialized YAML, not be silently dropped by valid_env_key."""
    base_yaml = _minimal_base_yaml(tmp_path, framework="vllm")

    # Simulate the split that now happens in _bench_patch/_confirm_stack_rebench
    from hyperloom.orchestrator.actions.executors._grid_server_args import (
        merge_server_args,
        split_config_changes,
    )
    config_changes = {
        "--tokenizer-mode": "deepseek_v4",
        "VLLM_ROCM_USE_AITER": "1",
    }
    cc_args, cc_envs = split_config_changes(config_changes)
    variant = GridVariant(
        name="test-tokenizer",
        extra_server_args=merge_server_args("", cc_args),
        extra_envs=cc_envs,
    )

    out_yaml = _build_variant_yaml(
        base_yaml_path=base_yaml,
        base_extra_args="",
        variant=variant,
        output_subdir=tmp_path / "out",
    )

    with out_yaml.open(encoding="utf-8") as f:
        materialized = yaml.safe_load(f)

    envs = materialized["benchmark"]["envs"]
    extra_vllm = envs.get("EXTRA_VLLM_ARGS", "")
    assert "--tokenizer-mode" in extra_vllm, f"flag not in EXTRA_VLLM_ARGS: {envs}"
    assert "deepseek_v4" in extra_vllm, f"value not in EXTRA_VLLM_ARGS: {envs}"
    assert envs.get("VLLM_ROCM_USE_AITER") == "1", f"env var missing: {envs}"


def test_env_key_not_in_server_args_yaml(tmp_path):
    """Pure env keys must NOT appear in EXTRA_VLLM_ARGS."""
    base_yaml = _minimal_base_yaml(tmp_path, framework="vllm")

    from hyperloom.orchestrator.actions.executors._grid_server_args import split_config_changes
    _, cc_envs = split_config_changes({"MY_ENV": "value"})
    variant = GridVariant(name="test-env", extra_server_args="", extra_envs=cc_envs)

    out_yaml = _build_variant_yaml(
        base_yaml_path=base_yaml,
        base_extra_args="",
        variant=variant,
        output_subdir=tmp_path / "out2",
    )

    with out_yaml.open(encoding="utf-8") as f:
        materialized = yaml.safe_load(f)

    envs = materialized["benchmark"]["envs"]
    extra_vllm = envs.get("EXTRA_VLLM_ARGS", "")
    assert "MY_ENV" not in extra_vllm
    assert envs.get("MY_ENV") == "value"
