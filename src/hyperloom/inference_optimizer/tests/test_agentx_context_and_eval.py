# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX context-window and eval-opt-out wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyperloom.inference_optimizer.cli import _resolve_run_max_model_len
from hyperloom.inference_optimizer.grid_server_args import (
    inject_sglang_context_length,
)

_NATIVE = 262144


def _model_dir(tmp_path: Path, max_pos: int | None = _NATIVE) -> str:
    d = tmp_path / "model"
    d.mkdir()
    cfg: dict[str, object] = {"model_type": "qwen3_5_moe_text"}
    if max_pos is not None:
        cfg["max_position_embeddings"] = max_pos
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(d)


def _args(model: str) -> argparse.Namespace:
    return argparse.Namespace(max_model_len=None, isl=1024, osl=1024, model=model)


def _clear(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    monkeypatch.delenv("MAX_MODEL_LEN", raising=False)


# --- MAX_MODEL_LEN resolution -------------------------------------------------


def test_max_model_len_synthetic_path_unchanged(tmp_path, monkeypatch):
    """AgentX OFF keeps the ISL+OSL+headroom derivation exactly as before."""
    _clear(monkeypatch)
    val, src = _resolve_run_max_model_len(_args(_model_dir(tmp_path)))
    assert (val, src) == (1024 + 1024 + 4096, "auto")


def test_long_context_model_does_not_warn(native_benchmark, tmp_path, monkeypatch, capsys):
    """A model that fits the corpus must stay quiet."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    _resolve_run_max_model_len(_args(_model_dir(tmp_path)))  # 262144 > peak
    assert "below the replay corpus peak" not in capsys.readouterr().err


def test_short_context_model_is_silent_on_the_synthetic_path(tmp_path, monkeypatch, capsys):
    """The corpus does not exist off the AgentX path; warning there is noise."""
    _clear(monkeypatch)
    _resolve_run_max_model_len(_args(_model_dir(tmp_path, max_pos=4096)))
    assert "below the replay corpus peak" not in capsys.readouterr().err


def test_max_model_len_agentx_uses_native_context(native_benchmark, tmp_path, monkeypatch):
    """AgentX ON resolves the model's own window instead of the synthetic shape."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    val, src = _resolve_run_max_model_len(_args(_model_dir(tmp_path)))
    assert (val, src) == (_NATIVE, "agentx-native-context")


def test_max_model_len_operator_override_still_wins(native_benchmark, tmp_path, monkeypatch):
    """An explicit ``$MAX_MODEL_LEN`` outranks the AgentX default."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    monkeypatch.setenv("MAX_MODEL_LEN", "131072")
    val, src = _resolve_run_max_model_len(_args(_model_dir(tmp_path)))
    assert (val, src) == (131072, "$MAX_MODEL_LEN")


def test_max_model_len_agentx_falls_back_when_config_unreadable(native_benchmark, tmp_path, monkeypatch, capsys):
    """An uncached model must not hard-fail -- but nothing re-resolves it later."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    val, src = _resolve_run_max_model_len(_args(_model_dir(tmp_path, max_pos=None)))
    assert (val, src) == (1024 + 1024 + 4096, "auto")
    err = capsys.readouterr().err
    assert "planning estimate" in err
    assert "below the replay corpus peak" not in err


# --- sglang --context-length --------------------------------------------------


def test_sglang_context_synthetic_path_unchanged(tmp_path, monkeypatch):
    """AgentX OFF still clamps to max(ISL+OSL+headroom, floor)."""
    _clear(monkeypatch)
    out = inject_sglang_context_length("", "sglang", _model_dir(tmp_path), 1024, 1024, max_model_len=6144)
    assert out.strip() == "--context-length 6144"


def test_sglang_context_agentx_skips_isl_osl_cap(native_benchmark, tmp_path, monkeypatch):
    """AgentX ON sizes the window off the model, not the placeholder ISL/OSL."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    out = inject_sglang_context_length("", "sglang", _model_dir(tmp_path), 1024, 1024, max_model_len=_NATIVE)
    assert out.strip() == f"--context-length {_NATIVE}"


def test_sglang_context_agentx_still_honours_explicit_ceiling(native_benchmark, tmp_path, monkeypatch):
    """An explicit MAX_MODEL_LEN ceiling keeps clamping under AgentX."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    out = inject_sglang_context_length("", "sglang", _model_dir(tmp_path), 1024, 1024, max_model_len=131072)
    assert out.strip() == "--context-length 131072"


def test_sglang_context_non_sglang_untouched(native_benchmark, tmp_path, monkeypatch):
    """vLLM runs never receive --context-length, AgentX or not."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    assert inject_sglang_context_length("", "vllm", _model_dir(tmp_path), 1024, 1024) == ""


# --- eval opt-out -------------------------------------------------------------


def test_agentx_runs_its_own_eval_not_lmeval(native_benchmark, monkeypatch):
    """AgentX eval is a post-hoc error-rate gate, not an lm-eval opt-out, so ``eval_disabled`` tracks --no-eval."""
    from hyperloom.inference_optimizer.cli import bootstrap

    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    assert bootstrap._agentx_enabled() is False

    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    assert bootstrap._agentx_enabled() is True

    # eval_disabled must NOT be forced on by AgentX any more.
    import inspect

    src = inspect.getsource(bootstrap)
    assert "_agentx_enabled()" not in src.split("eval_disabled=")[1].split("\n")[0]
