# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Weight formats a quantization toolkit nests inside a checkpoint's ``quantization_config``.

Quark and compressed-tensors put the toolkit's name in ``quant_method`` and the formats one level down: Quark on a
whole-checkpoint ``global_quant_config`` that ``layer_quant_config`` overrides per layer pattern and
``layer_type_quant_config`` per module class, compressed-tensors on ``config_groups`` that each name the layers they
target. Stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# sglang resolves a Quark layer's config from these, most specific first.
QUARK_LAYER_CONFIG_KEYS = ("layer_quant_config", "layer_type_quant_config")


@dataclass(frozen=True)
class WeightDeclaration:
    """One nested weight spec and the layers it applies to."""

    targets: str  # Layer pattern(s) or module class the spec applies to; empty for a whole-checkpoint scope.
    dtype: str  # Lower-cased ``dtype``/``type`` tag; empty when the spec names none.
    bits: float  # ``num_bits``/``bits``; 0.0 when absent or unusable.


def _declaration(targets: str, scope: Any) -> WeightDeclaration | None:
    if not isinstance(scope, dict):
        return None
    spec = scope.get("weight") or scope.get("weights")
    if not isinstance(spec, dict):
        return None
    try:
        bits = float(spec.get("num_bits") or spec.get("bits") or 0)
    except (TypeError, ValueError):
        bits = 0.0
    dtype = str(spec.get("dtype") or spec.get("type") or "").strip().lower()
    return WeightDeclaration(targets=targets, dtype=dtype, bits=max(bits, 0.0))


def weight_declarations(quant_cfg: Any) -> tuple[list[WeightDeclaration], list[WeightDeclaration]]:
    """``(whole-checkpoint, per-layer)`` weight declarations nested in ``quant_cfg``, empty when it nests none.

    Quark's per-layer declarations come in the order sglang resolves them, most specific first.
    """
    if not isinstance(quant_cfg, dict):
        return [], []
    whole = [
        decl for decl in (_declaration("", quant_cfg.get("global_quant_config")), _declaration("", quant_cfg)) if decl
    ]
    per_layer: list[WeightDeclaration] = []
    for key in QUARK_LAYER_CONFIG_KEYS:
        layer_cfg = quant_cfg.get(key)
        if isinstance(layer_cfg, dict):
            per_layer.extend(decl for target, scope in layer_cfg.items() if (decl := _declaration(str(target), scope)))
    groups = quant_cfg.get("config_groups")
    if isinstance(groups, dict):
        for name, scope in groups.items():
            targets = scope.get("targets") if isinstance(scope, dict) else None
            label = ",".join(str(t) for t in targets) if isinstance(targets, list) and targets else str(name)
            if decl := _declaration(label, scope):
                per_layer.append(decl)
    return whole, per_layer
