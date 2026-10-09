# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve representative decode shapes from a model's ``config.json``."""

from __future__ import annotations

import contextlib
import json
import re
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


def resolve_decode_shapes(
    model_path: str | Path,
    *,
    decode_batch: int = 16,
    attn_tp_size: int = 1,
) -> dict[str, Any]:
    """Derive representative decode shapes from a model config.

    ``attn_tp_size`` is the *attention* TP shard (often 1 under DP-attention),
    not the serving ``--tp`` used for MoE / smoke. It only scales
    ``n_local_heads`` / ``n_local_groups``.
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
    o_groups = _first(cfg, "o_groups", "n_groups")
    qk_rope = _first(cfg, "qk_rope_head_dim", "rope_head_dim")
    o_lora = _first(cfg, "o_lora_rank")
    try:
        tp = max(1, int(attn_tp_size))
    except (TypeError, ValueError):
        tp = 1

    shapes: dict[str, Any] = {
        "model_type": str(cfg.get("model_type") or ""),
        "decode_batch": int(decode_batch),
        "T": int(decode_batch),
        "attn_tp_size": tp,
    }
    for key, val in (
        ("hidden_size", hidden),
        ("num_attention_heads", n_heads),
        ("num_key_value_heads", n_kv),
        ("head_dim", head_dim),
        ("intermediate_size", inter),
        ("num_hidden_layers", _first(cfg, "num_hidden_layers", "n_layer")),
        ("rms_norm_eps", _first(cfg, "rms_norm_eps", "norm_eps", "layer_norm_eps")),
        ("o_groups", o_groups),
        ("qk_rope_head_dim", qk_rope),
        ("o_lora_rank", o_lora),
    ):
        if val is not None:
            shapes[key] = val
    if n_heads and n_kv:
        with contextlib.suppress(TypeError, ValueError, ZeroDivisionError):
            shapes["gqa_groups"] = int(n_heads) // int(n_kv)
    if n_heads is not None:
        with contextlib.suppress(TypeError, ValueError, ZeroDivisionError):
            shapes["n_local_heads"] = int(n_heads) // tp
    if o_groups is not None:
        with contextlib.suppress(TypeError, ValueError, ZeroDivisionError):
            shapes["n_local_groups"] = int(o_groups) // tp
        # Agents previously mistook gqa_groups / num_attention_heads for the
        # wo_a group axis. Stamp an explicit reminder into the recipe prompt --
        # only for models that actually have the axis, since the whole shapes
        # dict is rendered into the authoring prompt.
        shapes["group_axis_note"] = (
            "For DeepSeek-V4 wo_a / mxfp8 group-quant paths, G is "
            "n_local_groups (from o_groups // attn_tp_size), NOT gqa_groups "
            "and NOT num_attention_heads. Attention output is "
            "[T, n_local_heads, head_dim]; H and G are often unequal — do not "
            "require H == G."
        )
    return shapes


_ASSIGN_G = re.compile(
    r"""(?mx)
    ^\s*G\s*=\s*
    (?P<val>\d+)
    \s*(?:\#.*)?$
    """
)
_ASSIGN_G_FROM_SHAPES = re.compile(
    r"""(?mx)
    ^\s*G\s*=\s*.*
    (?:gqa_groups|num_attention_heads|n_heads)
    """
)


def harness_group_dim_mismatch(harness_text: str, shapes: dict[str, Any]) -> str:
    """Why a harness used the wrong group axis, or ``\"\"`` when it looks right.

    Catches the DSv4 failure mode where the author sets ``G = gqa_groups`` /
    ``num_attention_heads`` (e.g. 128) while the call site uses
    ``n_local_groups`` from ``o_groups`` (e.g. 16).
    """
    want = shapes.get("n_local_groups")
    if want is None:
        return ""
    try:
        want_i = int(want)
    except (TypeError, ValueError):
        return ""

    text = harness_text or ""
    if _ASSIGN_G_FROM_SHAPES.search(text):
        return (
            f"harness derives G from gqa_groups/num_attention_heads; "
            f"use n_local_groups={want_i} (from o_groups) instead"
        )

    assigned: list[int] = []
    for match in _ASSIGN_G.finditer(text):
        with contextlib.suppress(TypeError, ValueError):
            assigned.append(int(match.group("val")))
    if not assigned:
        return ""

    wrong_axes: set[int] = set()
    for key in ("gqa_groups", "num_attention_heads", "n_local_heads"):
        val = shapes.get(key)
        if val is None:
            continue
        with contextlib.suppress(TypeError, ValueError):
            wrong_axes.add(int(val))
    wrong_axes.discard(want_i)

    if want_i in assigned:
        return ""
    bad = [g for g in assigned if g in wrong_axes]
    if not bad:
        return ""
    return (
        f"harness sets G={bad[0]} but n_local_groups={want_i} "
        f"(o_groups={shapes.get('o_groups')}); do not use gqa_groups / "
        f"num_attention_heads as G"
    )
