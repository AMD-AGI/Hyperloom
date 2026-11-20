# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Focused coverage for CLI bootstrap helpers."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.inference_optimizer.cli import bootstrap as cb

from .test_cli_bootstrap import _args, _neutralize_seed_io


def test_seed_uses_explicit_workload_flags_over_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Regression for #903: explicit workload flags must reach SharedState."""
    _neutralize_seed_io(monkeypatch, gpu_count=8, lane_ceiling=16)
    monkeypatch.setenv("ISL", "256")
    monkeypatch.setenv("OSL", "256")
    monkeypatch.setenv("CONC", "8")

    state = cb._seed_shared_state(
        tmp_path,
        _args(model="/models/Qwen2.5-7B", isl=512, osl=512, conc=64, tp=1, ep=1),
        session_id="s-explicit",
        framework="vllm",
        settled=cb.SettledLaunch(compute_partition={}, gpu_power_settings={}, base=None),
    )

    assert state.isl == 512
    assert state.osl == 512
    assert state.conc == 64
    assert state.tp == 1
    assert state.ep == 1


def test_seed_applies_new_workload_defaults_when_unset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Regression for #903: when workload flags are unset (argparse default ``None``) and no inherited env is present, SharedState must seed the new fallback defaults (ISL/OSL=1024, CONC=64, TP/EP=1)."""
    _neutralize_seed_io(monkeypatch, gpu_count=8, lane_ceiling=16)
    for key in ("ISL", "OSL", "CONC", "TP", "EP"):
        monkeypatch.delenv(key, raising=False)

    state = cb._seed_shared_state(
        tmp_path,
        _args(
            model="/models/Qwen2.5-7B",
            isl=None,
            osl=None,
            conc=None,
            tp=None,
            ep=None,
        ),
        session_id="s-default",
        framework="vllm",
        settled=cb.SettledLaunch(compute_partition={}, gpu_power_settings={}, base=None),
    )

    assert state.isl == 1024
    assert state.osl == 1024
    assert state.conc == 64
    assert state.tp == 1
    assert state.ep == 1


def test_seed_keeps_the_model_arch_of_an_hf_cache_snapshot_launch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Regression for #930: an HF cache ``models--org--repo/snapshots/<hash>`` launch collapses the model identity to
    the hash, so the stale guard accepts the session's profile only by recovering the repo name from the raw ``--model``
    path."""
    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
    from hyperloom.common import visible_devices
    from hyperloom.orchestrator.policy import gate as policy

    monkeypatch.setattr(visible_devices, "detect_gpu_count", lambda: 8)
    monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 16)
    profile = {"model_name": "Qwen2.5-7B-Instruct", "attention": "GQA"}
    (tmp_path / "model_arch.json").write_text(json.dumps(profile), encoding="utf-8")

    raw = (
        "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
        "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    )
    state = cb._seed_shared_state(
        tmp_path,
        _args(model=raw),
        session_id="s-raw",
        framework="vllm",
        settled=cb.SettledLaunch(compute_partition={}, gpu_power_settings={}, base=None),
    )

    assert state.model_arch == profile


# _detect_checkpoint_precision tests


def test_detect_checkpoint_precision_bf16_from_torch_dtype(tmp_path):
    from hyperloom.inference_optimizer.cli.launch_shape import _detect_checkpoint_precision

    (tmp_path / "config.json").write_text('{"torch_dtype": "bfloat16"}', encoding="utf-8")
    assert _detect_checkpoint_precision(str(tmp_path)) == "bf16"


def test_detect_checkpoint_precision_fp8_from_quant_method(tmp_path):
    from hyperloom.inference_optimizer.cli.launch_shape import _detect_checkpoint_precision

    (tmp_path / "config.json").write_text('{"quantization_config": {"quant_method": "fp8"}}', encoding="utf-8")
    assert _detect_checkpoint_precision(str(tmp_path)) == "fp8"


def test_detect_checkpoint_precision_fallback_on_missing():
    from hyperloom.inference_optimizer.cli.launch_shape import _detect_checkpoint_precision

    assert _detect_checkpoint_precision("/nonexistent/model/path") == ""
    assert _detect_checkpoint_precision(None) == ""


def test_detect_checkpoint_precision_propagates_summary_failure(monkeypatch):
    import pytest

    from hyperloom.inference_optimizer import model_config_utils
    from hyperloom.inference_optimizer.cli.launch_shape import _detect_checkpoint_precision

    def summary_fails(_path):
        raise RuntimeError("summary bug")

    monkeypatch.setattr(model_config_utils, "summarize_model_config", summary_fails)

    with pytest.raises(RuntimeError):
        _detect_checkpoint_precision("/models/any")


def test_detect_checkpoint_precision_float8_e4m3fn(tmp_path):
    from hyperloom.inference_optimizer.cli.launch_shape import _detect_checkpoint_precision

    (tmp_path / "config.json").write_text('{"torch_dtype": "float8_e4m3fn"}', encoding="utf-8")
    assert _detect_checkpoint_precision(str(tmp_path)) == "fp8"


def test_resolve_precision_auto_detected(tmp_path, monkeypatch):
    import argparse

    from hyperloom.inference_optimizer.cli.launch_shape import _resolve_precision

    monkeypatch.setenv("PRECISION", "")
    (tmp_path / "config.json").write_text('{"torch_dtype": "bfloat16"}', encoding="utf-8")
    args = argparse.Namespace(precision=None, model=str(tmp_path), tp=None, ep=None, conc=None)
    _resolve_precision(args)
    assert args.precision == "bf16"


def test_resolve_precision_explicit_kept_with_warning(tmp_path, capsys):
    import argparse

    from hyperloom.inference_optimizer.cli.launch_shape import _resolve_precision

    (tmp_path / "config.json").write_text('{"torch_dtype": "bfloat16"}', encoding="utf-8")
    args = argparse.Namespace(precision="fp8", model=str(tmp_path), tp=None, ep=None, conc=None)
    _resolve_precision(args)
    assert args.precision == "fp8"
    captured = capsys.readouterr()
    assert "WARN" in captured.err and "bf16" in captured.err
