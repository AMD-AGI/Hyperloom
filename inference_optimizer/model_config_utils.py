# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared model-config helpers (config.json parsing + arch/type detection).

Leaf module: depends only on the standard library so both ``cli`` and the
orchestrator executors can import it without a circular dependency.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path


def _load_model_config_dict(model_path: str) -> dict | None:
    """Best-effort parse of ``<model_path>/config.json`` into a dict; returns ``None`` on any failure.

    Args:
        model_path: Filesystem path to the model directory.

    Returns:
        The parsed ``config.json`` dict, or ``None`` when the path is empty,
        the file is missing/unreadable, the JSON is invalid, or it is not a
        dict.
    """
    if not model_path:
        return None
    cfg_path = Path(model_path) / "config.json"
    try:
        raw = cfg_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logging.warning("model_config_unreadable: %s (%s)", cfg_path, exc)
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logging.warning("model_config_invalid_json: %s (%s)", cfg_path, exc)
        return None
    if not isinstance(data, dict):
        logging.warning(
            "model_config_not_a_dict: %s (got %s)",
            cfg_path,
            type(data).__name__,
        )
        return None
    return data


def _config_architectures(config: dict) -> list[str]:
    """Normalise ``config["architectures"]`` to a list of non-empty strings (scalar wrapped; absent -> []).

    Args:
        config: A parsed model config dict.

    Returns:
        The non-empty architecture strings; a lone scalar is wrapped and a
        missing key yields ``[]``.
    """
    arches_raw = config.get("architectures")
    if isinstance(arches_raw, list):
        return [str(a).strip() for a in arches_raw if str(a or "").strip()]
    if isinstance(arches_raw, str) and arches_raw.strip():
        return [arches_raw.strip()]
    return []


# Gemma2 forward builds ``normalizer = torch.tensor(...)`` (a host scalar) on
# every call. The TraceLens kernel_shape_profiler patch activates inside the
# CUDA-graph capture critical section, so that host construct runs during HIP
# stream capture and raises ``hipErrorStreamCaptureUnsupported`` -> capture
# fails -> roofline produces no ceiling. Callers skip shape-discovery for
# Gemma2 to keep CUDA graph while avoiding the crash.
# Single source of truth: ``cli`` reuses these for its preflight checks too.
GEMMA2_MODEL_TYPE = "gemma2"
GEMMA2_ARCHITECTURES = frozenset({"gemma2forcausallm"})


# ``gemma`` then an optional single separator then ``2`` as a standalone token
# (start/separator on the left, separator/end on the right). Matches gemma2 /
# gemma-2 / gemma_2 but not gemma3, gemma25, or notgemma2.
_GEMMA2_PATH_RE = re.compile(r"(?:^|[-_.])gemma[-_.]?2(?:[-_.]|$)")


def _path_looks_like_gemma2(model_path: str) -> bool:
    """Heuristic Gemma2 detection from the path when config.json is absent.

    Word-boundary match on the directory name (gemma2 / gemma-2 / gemma_2),
    so a not-yet-materialized Hub-id style path still gets the workaround
    without false-positives on names like notgemma2 / gemma25.

    Args:
        model_path: Filesystem or Hub-id style path to the model.

    Returns:
        ``True`` when the path's final component looks like a Gemma2 name.
    """
    if not model_path:
        return False
    return _GEMMA2_PATH_RE.search(Path(model_path).name.lower()) is not None


def _config_gemma2_scopes(data: dict) -> list[dict]:
    """Return [top-level, text_config?] scopes for Gemma2 inspection.

    Args:
        data: A parsed model config dict.

    Returns:
        The top-level config plus its nested ``text_config`` when that is a
        dict.
    """
    scopes = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append(nested)
    return scopes


def _config_is_gemma2(data: dict) -> bool:
    """True when a parsed config dict declares Gemma2 (top level or text_config).

    Args:
        data: A parsed model config dict.

    Returns:
        ``True`` when any scope declares the Gemma2 ``model_type`` or
        architecture.
    """
    for cfg in _config_gemma2_scopes(data):
        if str(cfg.get("model_type") or "").strip().lower() == GEMMA2_MODEL_TYPE:
            return True
        if any(a.lower() in GEMMA2_ARCHITECTURES for a in _config_architectures(cfg)):
            return True
    return False


def _config_has_model_identity(data: dict) -> bool:
    """True when the config carries any recognizable model_type/architectures.

    Used to decide whether a non-Gemma2 verdict is trustworthy: a config that
    clearly identifies another model (e.g. llama) must NOT fall back to the path
    heuristic, while an empty/residual config (``{}``, no model_type) should.

    Args:
        data: A parsed model config dict.

    Returns:
        ``True`` when any scope carries a non-empty ``model_type`` or
        ``architectures``.
    """
    for cfg in _config_gemma2_scopes(data):
        if str(cfg.get("model_type") or "").strip():
            return True
        if _config_architectures(cfg):
            return True
    return False


_MLA_KEYS = ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "q_lora_rank")
_MOE_EXPERT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")


def _merge_config_scopes(data: dict) -> dict:
    """Flatten nested ``text_config`` over the top level (nested wins for shape fields).

    Multimodal wrappers describe the benchmarkable decoder under
    ``text_config``; merge so structural fields resolve from there.
    """
    merged = dict(data)
    nested = data.get("text_config")
    if isinstance(nested, dict):
        for k, v in nested.items():
            if v not in (None, ""):
                merged[k] = v
    return merged


def _derive_attention_type(cfg: dict) -> str:
    """Infer attention variant (MLA/MQA/GQA/MHA) from head/lora config fields."""
    if any(cfg.get(k) for k in _MLA_KEYS):
        return "MLA"
    heads = int(cfg.get("num_attention_heads") or 0)
    kv = cfg.get("num_key_value_heads")
    kv = int(kv if kv is not None else heads or 0)
    if heads <= 0 or kv <= 0:
        return ""
    if kv == 1:
        return "MQA"
    if kv < heads:
        return "GQA"
    return "MHA"


def _derive_quantization(cfg: dict) -> str:
    """Return the weight quant method (e.g. ``fp8``) or '' when unquantized."""
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict):
        return str(qc.get("quant_method") or "").strip()
    return ""


def summarize_model_config(model_path: str) -> dict:
    """Best-effort structured summary of a model's ``config.json`` ({} on failure).

    Reads core shape/quant fields plus inferred ``attention_type`` and
    ``is_moe`` so session state can carry model basics without a framework.
    """
    data = _load_model_config_dict(model_path)
    if data is None:
        return {}
    cfg = _merge_config_scopes(data)
    out: dict = {}

    model_type = str(data.get("model_type") or cfg.get("model_type") or "").strip()
    if model_type:
        out["model_type"] = model_type
    arches = _config_architectures(data) or _config_architectures(cfg)
    if arches:
        out["architectures"] = arches

    heads = int(cfg.get("num_attention_heads") or 0)
    kv_raw = cfg.get("num_key_value_heads")
    kv = int(kv_raw if kv_raw is not None else heads or 0)
    head_dim = cfg.get("head_dim")
    hidden = cfg.get("hidden_size")
    if not head_dim and hidden and heads:
        head_dim = int(hidden) // int(heads)

    attn = _derive_attention_type(cfg)
    if attn:
        out["attention_type"] = attn
    if heads:
        out["num_attention_heads"] = heads
    if kv:
        out["num_key_value_heads"] = kv
    if head_dim:
        out["head_dim"] = int(head_dim)

    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size", "max_position_embeddings"):
        val = cfg.get(key)
        if val not in (None, ""):
            out[key] = int(val)

    num_experts = 0
    for k in _MOE_EXPERT_KEYS:
        if cfg.get(k):
            num_experts = int(cfg.get(k) or 0)
            break
    experts_per_tok = int(cfg.get("num_experts_per_tok") or 0)
    out["is_moe"] = num_experts > 0
    if num_experts > 0:
        out["num_experts"] = num_experts
    if experts_per_tok > 0:
        out["num_experts_per_tok"] = experts_per_tok

    quant = _derive_quantization(cfg)
    if quant:
        out["quantization"] = quant
    for key in ("torch_dtype", "kv_cache_dtype"):
        val = str(cfg.get(key) or "").strip()
        if val:
            out[key] = val
    return out


def _model_is_gemma2(model_path: str) -> bool:
    """Best-effort detect a Gemma2 model from config.json (top level or text_config).

    Falls back to a path heuristic when config.json is missing/unreadable OR
    present-but-unidentifiable (empty dict / no model_type/architectures). A
    config that clearly identifies a non-Gemma2 model is trusted as-is.

    Args:
        model_path: Filesystem path to the model directory.

    Returns:
        ``True`` when the model is detected as Gemma2 via config.json or the
        path heuristic.
    """
    data = _load_model_config_dict(model_path)
    if data is not None:
        if _config_is_gemma2(data):
            return True
        if _config_has_model_identity(data):
            return False
    return _path_looks_like_gemma2(model_path)
