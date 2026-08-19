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


def test_short_context_model_warns_before_the_round_starts(tmp_path, monkeypatch, capsys):
    """A model too small for the corpus must say so up front.

    The run still proceeds -- it is the honest one to make, and a model that
    cannot hold the corpus cannot hold a leaderboard row either. But left
    unannounced the failure surfaces roughly an hour in as a
    ``--failed-request-threshold`` abort, with nothing pointing at the cause.
    """
    from hyperloom.inference_optimizer.cli import AGENTX_CAPPED_CORPUS_PEAK_TOKENS

    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    short = AGENTX_CAPPED_CORPUS_PEAK_TOKENS // 8
    val, src = _resolve_run_max_model_len(_args(_model_dir(tmp_path, max_pos=short)))
    assert (val, src) == (short, "agentx-native-context")  # still runs
    err = capsys.readouterr().err
    assert "below the replay corpus peak" in err
    assert str(AGENTX_CAPPED_CORPUS_PEAK_TOKENS) in err


def test_explicit_flag_below_the_corpus_peak_still_warns(tmp_path, monkeypatch, capsys):
    """``--max-model-len 8192`` is the likeliest way to get this wrong.

    The warning used to live inside the auto-resolve branch, so it stayed silent
    for exactly the two configurations that override the native context by hand.
    """
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    args = _args(_model_dir(tmp_path))
    args.max_model_len = 8192
    val, src = _resolve_run_max_model_len(args)
    assert (val, src) == (8192, "--max-model-len")
    assert "below the replay corpus peak" in capsys.readouterr().err


def test_stale_env_below_the_corpus_peak_still_warns(tmp_path, monkeypatch, capsys):
    """Same for an inherited ``$MAX_MODEL_LEN`` left over from a synthetic run."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("MAX_MODEL_LEN", "6144")
    val, src = _resolve_run_max_model_len(_args(_model_dir(tmp_path)))
    assert (val, src) == (6144, "$MAX_MODEL_LEN")
    assert "below the replay corpus peak" in capsys.readouterr().err


def test_long_context_model_does_not_warn(tmp_path, monkeypatch, capsys):
    """A model that fits the corpus must stay quiet."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    _resolve_run_max_model_len(_args(_model_dir(tmp_path)))  # 262144 > peak
    assert "below the replay corpus peak" not in capsys.readouterr().err


def test_short_context_model_is_silent_on_the_synthetic_path(tmp_path, monkeypatch, capsys):
    """The corpus does not exist off the AgentX path; warning there is noise."""
    _clear(monkeypatch)
    _resolve_run_max_model_len(_args(_model_dir(tmp_path, max_pos=4096)))
    assert "below the replay corpus peak" not in capsys.readouterr().err


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


def test_max_model_len_agentx_falls_back_when_config_unreadable(tmp_path, monkeypatch, capsys):
    """An uncached model must not hard-fail -- but nothing re-resolves it later.

    The client deliberately emits no context cap and the server phase is a bare
    delegation, so this fallback really is the width the server gets. It is a
    synthetic-shape derivation that does not describe the agentic workload, so
    the corpus-fit warning has to fire on it too.
    """
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    val, src = _resolve_run_max_model_len(_args(_model_dir(tmp_path, max_pos=None)))
    assert (val, src) == (1024 + 1024 + 4096, "auto")
    err = capsys.readouterr().err
    assert "could not be read" in err
    assert "below the replay corpus peak" in err  # 6144 << the corpus peak


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
