# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve representative decode shapes from a model's ``config.json``.

Kernel-level validation needs realistic tensor shapes (hidden size, head count,
head dim, intermediate size, ...) for the fused op chain. These come from the HF
``config.json`` plus a decode batch size (from the trace or a default), NOT from
booting the model -- keeping validation cheap and e2e-free.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any


def load_model_config(model_path: str | Path) -> dict[str, Any]:
    """Load ``config.json`` from a model directory (or a direct file path)."""
    p = Path(model_path)
    cfg_path = p if p.is_file() else p / "config.json"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _first(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-null config key among ``keys``."""
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k]
    # Some models nest under ``text_config`` / ``language_config``.
    for nest in ("text_config", "language_config"):
        sub = cfg.get(nest)
        if isinstance(sub, dict):
            for k in keys:
                if sub.get(k) is not None:
                    return sub[k]
    return default


def resolve_decode_shapes(model_path: str | Path, *, decode_batch: int = 16) -> dict[str, Any]:
    """Derive representative decode shapes from a model config.

    Args:
        model_path: Model directory (containing ``config.json``) or file path.
        decode_batch: Number of concurrent decode tokens (T). Defaults to 16.

    Returns:
        A best-effort shape dict. Missing fields are omitted rather than guessed;
        ``model_type`` is always present ("" if unknown) so downstream can branch.
    """
    cfg = load_model_config(model_path)
    hidden = _first(cfg, "hidden_size", "d_model", "n_embd")
    n_heads = _first(cfg, "num_attention_heads", "n_head")
    n_kv = _first(cfg, "num_key_value_heads", "num_kv_heads", default=n_heads)
    head_dim = _first(cfg, "head_dim")
    if head_dim is None and hidden and n_heads:
        try:
            head_dim = int(hidden) // int(n_heads)
        except (TypeError, ValueError, ZeroDivisionError):
            head_dim = None
    inter = _first(cfg, "intermediate_size", "ffn_dim", "n_inner")

    shapes: dict[str, Any] = {
        "model_type": str(cfg.get("model_type") or ""),
        "decode_batch": int(decode_batch),
        "T": int(decode_batch),
    }
    for key, val in (
        ("hidden_size", hidden),
        ("num_attention_heads", n_heads),
        ("num_key_value_heads", n_kv),
        ("head_dim", head_dim),
        ("intermediate_size", inter),
        ("num_hidden_layers", _first(cfg, "num_hidden_layers", "n_layer")),
        ("rms_norm_eps", _first(cfg, "rms_norm_eps", "norm_eps", "layer_norm_eps")),
    ):
        if val is not None:
            shapes[key] = val
    if n_heads and n_kv:
        with contextlib.suppress(TypeError, ValueError, ZeroDivisionError):
            shapes["gqa_groups"] = int(n_heads) // int(n_kv)
    return shapes
