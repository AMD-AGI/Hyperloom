# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX context-window and eval-opt-out wiring.

Three invariants, each guarding a defect that produced a *plausible* number
rather than an error:

- ``MAX_MODEL_LEN`` under AgentX must come from the model's own context window,
  not from ``ISL+OSL+headroom``. The synthetic derivation yields 6144 at the
  1024/1024 defaults, which caps a corpus whose traces reach ~1M tokens.
- sglang's ``--context-length`` must skip the same ISL/OSL ceiling; it is a
  second, independent path to the same truncation.
- AgentX must be a *deliberate* eval opt-out, so the baseline's
  missing-accuracy guard does not stamp a good AgentX baseline as a failure.

The synthetic (non-AgentX) path must be byte-for-byte unaffected by all three.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyperloom.inference_optimizer.cli import _resolve_run_max_model_len
from hyperloom.orchestrator.actions.executors._grid_server_args import (
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
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("MAX_MODEL_LEN", raising=False)


# --- MAX_MODEL_LEN resolution -------------------------------------------------


def test_max_model_len_synthetic_path_unchanged(tmp_path, monkeypatch):
    """AgentX OFF keeps the ISL+OSL+headroom derivation exactly as before."""
    _clear(monkeypatch)
    val, src = _resolve_run_max_model_len(_args(_model_dir(tmp_path)))
    assert (val, src) == (1024 + 1024 + 4096, "auto")


def test_max_model_len_agentx_uses_native_context(tmp_path, monkeypatch):
    """AgentX ON resolves the model's own window instead of the synthetic shape."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    val, src = _resolve_run_max_model_len(_args(_model_dir(tmp_path)))
    assert (val, src) == (_NATIVE, "agentx-native-context")


def test_max_model_len_operator_override_still_wins(tmp_path, monkeypatch):
    """An explicit ``$MAX_MODEL_LEN`` outranks the AgentX default."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("MAX_MODEL_LEN", "131072")
    val, src = _resolve_run_max_model_len(_args(_model_dir(tmp_path)))
    assert (val, src) == (131072, "$MAX_MODEL_LEN")


def test_max_model_len_agentx_falls_back_when_config_unreadable(tmp_path, monkeypatch):
    """An uncached model must not hard-fail; the client re-resolves later."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    val, src = _resolve_run_max_model_len(_args(_model_dir(tmp_path, max_pos=None)))
    assert (val, src) == (1024 + 1024 + 4096, "auto")


# --- sglang --context-length --------------------------------------------------


def test_sglang_context_synthetic_path_unchanged(tmp_path, monkeypatch):
    """AgentX OFF still clamps to max(ISL+OSL+headroom, floor)."""
    _clear(monkeypatch)
    out = inject_sglang_context_length("", "sglang", _model_dir(tmp_path), 1024, 1024, max_model_len=6144)
    assert out.strip() == "--context-length 6144"


def test_sglang_context_agentx_skips_isl_osl_cap(tmp_path, monkeypatch):
    """AgentX ON sizes the window off the model, not the placeholder ISL/OSL."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    out = inject_sglang_context_length(
        "", "sglang", _model_dir(tmp_path), 1024, 1024, max_model_len=_NATIVE
    )
    assert out.strip() == f"--context-length {_NATIVE}"


def test_sglang_context_agentx_still_honours_explicit_ceiling(tmp_path, monkeypatch):
    """An explicit MAX_MODEL_LEN ceiling keeps clamping under AgentX."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    out = inject_sglang_context_length(
        "", "sglang", _model_dir(tmp_path), 1024, 1024, max_model_len=131072
    )
    assert out.strip() == "--context-length 131072"


def test_sglang_context_non_sglang_untouched(tmp_path, monkeypatch):
    """vLLM runs never receive --context-length, AgentX or not."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    assert inject_sglang_context_length("", "vllm", _model_dir(tmp_path), 1024, 1024) == ""


# --- eval opt-out -------------------------------------------------------------


def test_agentx_is_a_deliberate_eval_optout(monkeypatch):
    """``eval_disabled`` must be set by AgentX, not only by ``--no-eval``.

    ``baseline._maybe_stop_on_missing_baseline_accuracy`` rejects an incidental
    ``RUN_EVAL=false`` as an excuse for a missing accuracy, and reads only
    ``state.eval_disabled``. Without this wiring an AgentX baseline is stamped
    as an eval failure, never anchors ``baseline_tput``, and every variant's
    gain comes back None.
    """
    from hyperloom.inference_optimizer.cli import bootstrap

    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    assert bootstrap._agentx_enabled() is False

    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    assert bootstrap._agentx_enabled() is True

    # The seeding expression must OR the two opt-outs together.
    import inspect

    src = inspect.getsource(bootstrap)
    assert 'eval_disabled=bool(getattr(args, "no_eval", False)) or _agentx_enabled(),' in src
