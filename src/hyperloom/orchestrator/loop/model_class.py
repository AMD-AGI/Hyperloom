# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic model_class inference from a local checkpoint's config and path."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.model_config_utils import resolve_local_model_dir

log = logging.getLogger(__name__)

# A multimodal checkpoint keeps the language model one level down, so the expert counts and the LM architecture live
# there rather than at the top.
_NESTED_LM_CONFIG_KEYS: tuple[str, ...] = ("text_config", "llm_config", "language_config")
_ARCHITECTURE_TYPE_KEYS: tuple[str, ...] = ("model_type", "attention_type", "attn_type")
_EXPERT_COUNT_KEYS: tuple[str, ...] = (
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
    "moe_num_experts",
)
_MOE_KEYWORDS: tuple[str, ...] = (
    "moe",
    "mixtral",
    "deepseek-v2",
    "deepseek-v3",
    "deepseek-r1",
    "kimi",
    "glm-5",
    "glm5",
)
_MLA_KEYWORDS: tuple[str, ...] = (
    "mla",
    "multi-head latent",
    "deepseek",
    "kimi",
    "glm-5",
    "glm5",
)
_NSA_KEYWORDS: tuple[str, ...] = (
    "nsa",
    "native sparse attention",
    "glm-5",
    "glm5",
)


def _read_model_config(raw_path: str) -> dict[str, Any]:
    """``config.json`` of a local checkpoint; empty when it is absent or unreadable."""
    if not raw_path:
        return {}
    # ``model_path`` may be an HF repo id; resolve to the local weights dir so the config-based classification
    # works (the raw string still feeds the keyword fallback below).
    resolved = resolve_local_model_dir(raw_path)
    cfg = (resolved / "config.json") if resolved is not None else Path(raw_path) / "config.json"
    try:
        if not cfg.is_file():
            return {}
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.debug("model_class inference: failed to read %s", cfg, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _config_scopes(raw_path: str) -> list[dict[str, Any]]:
    """The config payloads to classify from, outer first: a VL wrapper would otherwise classify as dense."""
    payload = _read_model_config(raw_path)
    scopes = [payload]
    for nested_key in _NESTED_LM_CONFIG_KEYS:
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            scopes.append(nested)
    return scopes


def _architecture_text(raw_path: str, scopes: list[dict[str, Any]]) -> str:
    """Lower-cased blob of the path and every architecture/type field, for keyword matching."""
    text_parts: list[str] = [raw_path.lower()]
    for scope in scopes:
        arch = scope.get("architectures")
        if isinstance(arch, list):
            text_parts.extend(str(x).lower() for x in arch if x)
        elif arch:
            text_parts.append(str(arch).lower())
        for key in _ARCHITECTURE_TYPE_KEYS:
            if scope.get(key):
                text_parts.append(str(scope[key]).lower())
    return " ".join(text_parts)


def _has_positive_count(scopes: list[dict[str, Any]], keys: tuple[str, ...]) -> bool:
    """True when any scope carries one of ``keys`` as a positive integer count."""
    for scope in scopes:
        for key in keys:
            val = scope.get(key)
            if isinstance(val, bool):
                continue
            try:
                if val is not None and int(val) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def infer_model_class_from_config(model_path: str) -> str:
    """Infer a deterministic model_class from local model metadata."""
    raw_path = (model_path or "").strip()
    scopes = _config_scopes(raw_path)
    text = _architecture_text(raw_path, scopes)

    is_moe = _has_positive_count(scopes, _EXPERT_COUNT_KEYS) or any(k in text for k in _MOE_KEYWORDS)
    is_mla = any(k in text for k in _MLA_KEYWORDS)
    is_nsa = any(k in text for k in _NSA_KEYWORDS)
    if is_moe and is_mla and is_nsa:
        return "moe_mla_nsa"
    if is_moe and is_mla:
        return "moe_mla"
    if is_moe:
        return "moe_swa"
    return "dense"
