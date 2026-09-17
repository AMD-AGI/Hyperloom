# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared model-config helpers (config.json parsing + arch/type detection)."""

from __future__ import annotations

import json
import logging
import re
import struct
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_int

# Single source of truth for --model (path OR HF repo id) -> local dir.
from hyperloom.common.model_paths import resolve_local_model_dir  # noqa: F401


_MAXPOS_CONFIG_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",  # HuggingFace tokenizer_config field; used by some custom models (e.g. kimi_linear)
)

# Quark PTQ MX-FP4 (W4A4) MoE is implemented in sglang only on its aiter MoE runner; every other backend leaves the
# scheme without a ``runner`` attribute and the server dies on the first forward pass.
_NATIVE_MOE_RUNNER_QUANT_METHODS = frozenset({"quark"})

# MX group size, mirroring sglang's ``QuarkConfig._is_mx_fp4`` validation.
_MX_FP4_GROUP_SIZE = 32

# sglang resolves a layer's quant config from these, most specific first.
_QUARK_LAYER_CONFIG_KEYS = ("layer_quant_config", "layer_type_quant_config")


def _load_model_config_dict(model_path: str) -> dict | None:
    """Best-effort parse of ``<model_path>/config.json`` into a dict; returns ``None`` on any failure."""
    if not model_path:
        return None
    # --model may be an HF repo id rather than a local dir; resolve it (a real dir is returned unchanged) so
    # config-derived metadata isn't silently empty for repo-id launches.
    base = resolve_local_model_dir(model_path) or Path(model_path)
    cfg_path = base / "config.json"
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


def _load_model_max_position_embeddings(model_path: str) -> int | None:
    """Best-effort read of max sequence length from config.json (first positive among known keys, incl. nested ``text_config``), or None."""
    if not model_path:
        return None
    cfg_path = (resolve_local_model_dir(model_path) or Path(model_path)) / "config.json"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    candidates = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for cfg in candidates:
        for key in _MAXPOS_CONFIG_KEYS:
            val = cfg.get(key)
            if isinstance(val, bool):
                continue
            if isinstance(val, int) and val > 0:
                return val
    return None


def _model_has_dual_chunk_attention(model_path: str) -> bool:
    """Best-effort detect a ``dual_chunk_attention_config`` in config.json."""
    data = _load_model_config_dict(model_path)
    if data is None:
        return False
    if data.get("dual_chunk_attention_config"):
        return True
    nested = data.get("text_config")
    return isinstance(nested, dict) and bool(nested.get("dual_chunk_attention_config"))


def _is_quark_mx_fp4_entry(entry: Any) -> bool:
    """Whether one Quark layer-config entry is the MX-FP4 (W4A4) scheme."""
    if not isinstance(entry, dict):
        return False
    weight = entry.get("weight")
    inputs = entry.get("input_tensors")
    if not isinstance(weight, dict) or not isinstance(inputs, dict):
        return False
    for spec in (weight, inputs):
        if spec.get("dtype") != "fp4" or spec.get("qscheme") != "per_group":
            return False
        if spec.get("group_size") != _MX_FP4_GROUP_SIZE:
            return False
        if spec.get("scale_format") != "e8m0":
            return False
    return weight.get("is_dynamic") is not True and inputs.get("is_dynamic") is not False


def _model_moe_runner_requires_aiter(model_path: str) -> bool:
    """Best-effort detect a MoE quant scheme that only the aiter runner serves."""
    if not model_path:
        return False
    data = _load_model_config_dict(model_path)
    if data is None:
        return False
    candidates = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for cfg in candidates:
        qc = cfg.get("quantization_config")
        if not isinstance(qc, dict):
            continue
        if str(qc.get("quant_method") or "").strip().lower() not in _NATIVE_MOE_RUNNER_QUANT_METHODS:
            continue
        entries: list[Any] = [qc.get("global_quant_config")]
        for key in _QUARK_LAYER_CONFIG_KEYS:
            per_layer = qc.get(key)
            if isinstance(per_layer, dict):
                entries.extend(per_layer.values())
        if any(_is_quark_mx_fp4_entry(entry) for entry in entries):
            return True
    return False


def _model_declared_quant_method(model_path: str) -> str:
    """Return the checkpoint's declared ``quant_method``, lowercased."""
    if not model_path:
        return ""
    data = _load_model_config_dict(model_path)
    if data is None:
        return ""
    candidates = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for cfg in candidates:
        qc = cfg.get("quantization_config")
        if isinstance(qc, dict):
            method = str(qc.get("quant_method") or "").strip().lower()
            if method:
                return method
    return ""


def _config_architectures(config: dict) -> list[str]:
    """Normalise ``config[\"architectures\"]`` to a list of non-empty strings (scalar wrapped; absent -> [])."""
    arches_raw = config.get("architectures")
    if isinstance(arches_raw, list):
        return [str(a).strip() for a in arches_raw if str(a or "").strip()]
    if isinstance(arches_raw, str) and arches_raw.strip():
        return [arches_raw.strip()]
    return []


# Gemma2 breaks the TraceLens shape-discovery patch under CUDA-graph capture, so callers skip shape-discovery for
# Gemma2.
GEMMA2_MODEL_TYPE = "gemma2"
GEMMA2_ARCHITECTURES = frozenset({"gemma2forcausallm"})


# Matches gemma2 / gemma-2 / gemma_2 but not gemma3, gemma25, or notgemma2.
_GEMMA2_PATH_RE = re.compile(r"(?:^|[-_.])gemma[-_.]?2(?:[-_.]|$)")


def _path_looks_like_gemma2(model_path: str) -> bool:
    """Heuristic Gemma2 detection from the path when config.json is absent."""
    if not model_path:
        return False
    return _GEMMA2_PATH_RE.search(Path(model_path).name.lower()) is not None


def _config_gemma2_scopes(data: dict) -> list[dict]:
    """Return [top-level, text_config?] scopes for Gemma2 inspection."""
    scopes = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append(nested)
    return scopes


def _config_is_gemma2(data: dict) -> bool:
    """True when a parsed config dict declares Gemma2 (top level or text_config)."""
    for cfg in _config_gemma2_scopes(data):
        if str(cfg.get("model_type") or "").strip().lower() == GEMMA2_MODEL_TYPE:
            return True
        if any(a.lower() in GEMMA2_ARCHITECTURES for a in _config_architectures(cfg)):
            return True
    return False


def _config_has_model_identity(data: dict) -> bool:
    """True when the config carries any recognizable model_type/architectures."""
    for cfg in _config_gemma2_scopes(data):
        if str(cfg.get("model_type") or "").strip():
            return True
        if _config_architectures(cfg):
            return True
    return False


# Standard HF FP8 quant_method handled by sglang's Fp8LinearMethod.
_FP8_QUANT_METHOD = "fp8"
# Sanity cap for the safetensors JSON header length.
_SAFETENSORS_HEADER_MAX_BYTES = 100 * 1024 * 1024


def _read_safetensors_header(path: Path) -> dict | None:
    """Parse the JSON header of a ``.safetensors`` file without loading tensor data."""
    try:
        with path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                return None
            (header_len,) = struct.unpack("<Q", raw_len)
            if header_len <= 0 or header_len > _SAFETENSORS_HEADER_MAX_BYTES:
                return None
            header_bytes = fh.read(header_len)
            if len(header_bytes) != header_len:
                return None
        header = json.loads(header_bytes)
    except (OSError, ValueError, struct.error) as exc:
        logging.warning("safetensors_header_unreadable: %s (%s)", path, exc)
        return None
    return header if isinstance(header, dict) else None


def _fp8_weight_scale_is_per_channel(model_path: str) -> bool | None:
    """Classify a serialized FP8 checkpoint's weight-scale granularity."""
    if not model_path:
        return None
    # --model may be an HF repo id; resolve to the local weights dir so the safetensors scan works for repo-id
    # launches.
    base = resolve_local_model_dir(model_path) or Path(model_path)
    files = sorted(base.glob("*.safetensors"))
    if not files:
        return None
    for fpath in files:
        header = _read_safetensors_header(fpath)
        if not header:
            continue
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            # Skip block-scale ``weight_scale_inv``; only per-channel/per-tensor ``weight_scale`` is relevant here.
            if "weight_scale" not in name or "weight_scale_inv" in name:
                continue
            shape = meta.get("shape")
            if not isinstance(shape, list):
                continue
            numel = 1
            for dim in shape:
                if isinstance(dim, int):
                    numel *= dim
            return numel > 1
    return None


def _fp8_is_per_channel_per_token(model_path: str) -> bool:
    """True when a serialized FP8 checkpoint uses per-channel weight + per-token (dynamic) activation."""
    data = _load_model_config_dict(model_path)
    if not isinstance(data, dict):
        return False
    qc = data.get("quantization_config")
    if not isinstance(qc, dict):
        return False
    if str(qc.get("quant_method") or "").strip().lower() != _FP8_QUANT_METHOD:
        return False
    # Block-scale FP8 is served by a different kernel path; never touch it.
    if qc.get("weight_block_size") is not None:
        return False
    # Only dynamic (per-token) activation hits the fast path.
    activation = str(qc.get("activation_scheme") or "").strip().lower()
    if activation not in ("", "dynamic"):
        return False
    # Only confirmed per-channel weights benefit; undeterminable -> decline.
    return _fp8_weight_scale_is_per_channel(model_path) is True


def _fp8_is_block_scale(model_path: str) -> bool:
    """True when a serialized FP8 checkpoint uses block-scale quantization."""
    data = _load_model_config_dict(model_path)
    if not isinstance(data, dict):
        return False
    qc = data.get("quantization_config")
    if not isinstance(qc, dict):
        return False
    if str(qc.get("quant_method") or "").strip().lower() != _FP8_QUANT_METHOD:
        return False
    # Require a non-empty weight_block_size.
    return bool(qc.get("weight_block_size"))


_MLA_KEYS = ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "q_lora_rank")
_MOE_EXPERT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")
_SHARED_EXPERT_KEYS = ("n_shared_experts", "num_shared_experts", "moe_num_shared_experts")
# Nested text-tower config keys used by multimodal wrappers (priority order).
_TEXT_SCOPE_KEYS = ("text_config", "llm_config", "language_config")
# Base-family tokens for derived/hybrid model_types (longest first).
_FAMILY_TOKENS = ("qwen3", "qwen2", "deepseek", "llama", "gemma", "mistral", "phi", "glm")


def _merge_config_scopes(data: dict) -> dict:
    """Flatten nested text-tower config(s) over the top level (nested wins)."""
    merged = dict(data)
    seen: set[str] = set()
    for scope_key in _TEXT_SCOPE_KEYS:
        nested = data.get(scope_key)
        if not isinstance(nested, dict):
            continue
        for k, v in nested.items():
            if v in (None, ""):
                continue
            if k in seen:
                continue
            merged[k] = v
            seen.add(k)
    return merged


def _derive_attention_type(cfg: dict) -> str:
    """Infer attention variant (MLA/MQA/GQA/MHA) from head/lora config fields."""
    if any(cfg.get(k) for k in _MLA_KEYS):
        return "MLA"
    heads = to_int(cfg.get("num_attention_heads")) or 0
    kv_raw = cfg.get("num_key_value_heads")
    kv = to_int(kv_raw) if kv_raw is not None else heads
    kv = kv or 0
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


def _derive_model_family(model_type: str, model_path: str) -> str:
    """Infer the base model family with generation (e.g. qwen3, deepseek_v3)."""
    mt = str(model_type or "").strip().lower()
    name = Path(model_path or "").name.lower()

    # DeepSeek: keep major version. Check v3 before v2 (deepseek_v32 has both).
    if mt.startswith("deepseek"):
        if "v4" in mt:
            return "deepseek_v4"
        if "v3" in mt or mt == "deepseek":
            return "deepseek_v3"
        if "v2" in mt:
            return "deepseek_v2"
        return "deepseek"
    # Qwen: collapse the generation's variants.
    if mt.startswith("qwen"):
        if mt.startswith("qwen3"):
            return "qwen3"
        if mt.startswith("qwen2"):
            return "qwen2"
        if mt.startswith("qwen1") or "qwen1.5" in name:
            return "qwen1.5"
        return "qwen"
    # Gemma generations.
    for gen in ("gemma4", "gemma3", "gemma2"):
        if mt.startswith(gen):
            return gen
    if mt == "gemma":
        return "gemma"
    # Mistral vs Mixtral kept distinct.
    if mt.startswith("mixtral"):
        return "mixtral"
    if mt.startswith("mistral"):
        return "mistral"
    # Llama: model_type is bare 'llama'; derive generation from name.
    if mt == "llama" or mt.startswith("llama"):
        if mt == "llama4" or "llama-4" in name or "llama4" in name:
            return "llama4"
        if "llama-3" in name or "llama3" in name or "llama_3" in name:
            return "llama3"
        if "llama-2" in name or "llama2" in name or "llama_2" in name:
            return "llama2"
        return "llama"
    # MiniMax / Nemotron / InternVL families collapse sub-variants.
    if mt.startswith("minimax"):
        return "minimax"
    if mt.startswith("nemotron"):
        return "nemotron"
    if mt.startswith("internvl"):
        return "internvl"
    if mt.startswith("glm"):
        return "glm4" if "4" in mt else "glm"
    if mt.startswith("phi"):
        return "phi3" if mt.startswith("phi3") else "phi"
    if not mt:
        return ""
    # Derived/hybrid types: map to the base family token in the model_type.
    for tok in _FAMILY_TOKENS:
        if tok in mt:
            return tok
    # Generic fallback: family prefix before the first separator.
    return mt.split("_")[0]


def summarize_model_config(model_path: str) -> dict:
    """Best-effort structured summary of a model's ``config.json`` ({} on failure)."""
    data = _load_model_config_dict(model_path)
    if data is None:
        return {}
    cfg = _merge_config_scopes(data)
    out: dict = {}

    # Prefer the merged (nested text-tower wins) model_type so multimodal wrappers report the real decoder rather than
    # the wrapper shell.
    model_type = str(cfg.get("model_type") or data.get("model_type") or "").strip()
    if model_type:
        out["model_type"] = model_type
    arches = _config_architectures(data) or _config_architectures(cfg)
    if arches:
        out["architectures"] = arches

    family = _derive_model_family(model_type, model_path)
    if family:
        out["model_family"] = family

    heads = to_int(cfg.get("num_attention_heads")) or 0
    kv_raw = cfg.get("num_key_value_heads")
    kv = (to_int(kv_raw) if kv_raw is not None else heads) or 0
    head_dim = to_int(cfg.get("head_dim"))
    hidden = to_int(cfg.get("hidden_size"))
    if not head_dim and hidden and heads:
        head_dim = hidden // heads

    attn = _derive_attention_type(cfg)
    if attn:
        out["attention_type"] = attn
    if heads:
        out["num_attention_heads"] = heads
    if kv:
        out["num_key_value_heads"] = kv
    if head_dim:
        out["head_dim"] = head_dim

    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size", "max_position_embeddings"):
        val = to_int(cfg.get(key))
        if val is not None:
            out[key] = val

    num_experts = 0
    for k in _MOE_EXPERT_KEYS:
        ne = to_int(cfg.get(k))
        if ne:
            num_experts = ne
            break
    experts_per_tok = to_int(cfg.get("num_experts_per_tok")) or 0
    out["is_moe"] = num_experts > 0
    if num_experts > 0:
        out["num_experts"] = num_experts
    if experts_per_tok > 0:
        out["num_experts_per_tok"] = experts_per_tok

    # Shared-expert detection: only emit when is_moe is also true to avoid false positives on non-MoE models that
    # happen to carry shared-looking keys.
    if out["is_moe"]:
        num_shared = 0
        for k in _SHARED_EXPERT_KEYS:
            ns = to_int(cfg.get(k))
            if ns:
                num_shared = ns
                break
        shared_evidence = (
            num_shared > 0 or bool(cfg.get("shared_expert_intermediate_size")) or bool(cfg.get("shared_experts"))
        )
        if shared_evidence:
            out["has_shared_expert"] = True
            if num_shared > 0:
                out["num_shared_experts"] = num_shared

    quant = _derive_quantization(cfg)
    if quant:
        out["quantization"] = quant
    for key in ("torch_dtype", "kv_cache_dtype"):
        val = str(cfg.get(key) or "").strip()
        if val:
            out[key] = val
    return out


def _model_is_gemma2(model_path: str) -> bool:
    """Best-effort detect a Gemma2 model from config.json (top level or text_config)."""
    data = _load_model_config_dict(model_path)
    if data is not None:
        if _config_is_gemma2(data):
            return True
        if _config_has_model_identity(data):
            return False
    return _path_looks_like_gemma2(model_path)


def _sparse_kv_block_size(model_path: str) -> int | None:
    """Return the KV-cache block size a sparse-attention model requires, or None."""
    data = _load_model_config_dict(model_path)
    if not isinstance(data, dict):
        return None
    sparse = _merge_config_scopes(data).get("sparse_attention_config")
    if not isinstance(sparse, dict):
        return None
    return to_int(sparse.get("sparse_block_size"))
