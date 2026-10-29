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

from common.checkpoint_quant import QUARK_LAYER_CONFIG_KEYS, checkpoint_format
from common.coerce import optional_positive_int, to_int
from hyperloom.common.model_paths import resolve_local_model_dir


log = logging.getLogger(__name__)


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


def load_model_config_dict(model_path: str) -> dict | None:
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
    except (OSError, UnicodeDecodeError) as exc:
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


def config_scopes(data: dict) -> list[dict]:
    """Return the [top-level, text_config?] scopes a config.json is inspected over."""
    scopes = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append(nested)
    return scopes


def _load_model_max_position_embeddings(model_path: str) -> int | None:
    """Best-effort read of max sequence length from config.json (first positive among known keys, incl. nested ``text_config``), or None."""
    data = load_model_config_dict(model_path)
    if data is None:
        return None
    for cfg in config_scopes(data):
        for key in _MAXPOS_CONFIG_KEYS:
            val = cfg.get(key)
            if isinstance(val, bool):
                continue
            if isinstance(val, int) and val > 0:
                return val
    return None


def _model_has_dual_chunk_attention(model_path: str) -> bool:
    """Best-effort detect a ``dual_chunk_attention_config`` in config.json."""
    data = load_model_config_dict(model_path)
    if data is None:
        return False
    return any(cfg.get("dual_chunk_attention_config") for cfg in config_scopes(data))


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
    data = load_model_config_dict(model_path)
    if data is None:
        return False
    for cfg in config_scopes(data):
        qc = cfg.get("quantization_config")
        if not isinstance(qc, dict):
            continue
        if str(qc.get("quant_method") or "").strip().lower() not in _NATIVE_MOE_RUNNER_QUANT_METHODS:
            continue
        entries: list[Any] = [qc.get("global_quant_config")]
        for key in QUARK_LAYER_CONFIG_KEYS:
            per_layer = qc.get(key)
            if isinstance(per_layer, dict):
                entries.extend(per_layer.values())
        if any(_is_quark_mx_fp4_entry(entry) for entry in entries):
            return True
    return False


def _model_is_moe(model_path: str) -> bool:
    """Best-effort detect a Mixture-of-Experts model from config.json."""
    data = load_model_config_dict(model_path)
    if data is None:
        return False
    expert_keys = ("num_experts", "num_local_experts", "n_routed_experts")
    for cfg in config_scopes(data):
        for key in expert_keys:
            val = cfg.get(key)
            if isinstance(val, bool):
                continue
            if isinstance(val, int) and val > 1:
                return True
        if cfg.get("moe_intermediate_size"):
            return True
        if "moe" in str(cfg.get("model_type") or "").lower():
            return True
        if any("moe" in arch.lower() for arch in _config_architectures(cfg)):
            return True
    return False


def model_supports_aiter_ck_fused_moe(model_path: str, tp: int) -> bool:
    """Whether aiter's CK fused-MoE can serve this checkpoint at this TP."""
    if not _model_is_moe(model_path):
        return True
    data = load_model_config_dict(model_path)
    if data is None:
        return True
    for cfg in config_scopes(data):
        size = cfg.get("moe_intermediate_size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            continue
        shards = max(1, int(tp or 1))
        return (size // shards) % 128 == 0
    return True


def _model_declared_quant_method(model_path: str) -> str:
    """Return the checkpoint's declared ``quant_method``, lowercased."""
    if not model_path:
        return ""
    data = load_model_config_dict(model_path)
    if data is None:
        return ""
    for cfg in config_scopes(data):
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


def _config_is_gemma2(data: dict) -> bool:
    """True when a parsed config dict declares Gemma2 (top level or text_config)."""
    for cfg in config_scopes(data):
        if str(cfg.get("model_type") or "").strip().lower() == GEMMA2_MODEL_TYPE:
            return True
        if any(a.lower() in GEMMA2_ARCHITECTURES for a in _config_architectures(cfg)):
            return True
    return False


def _config_has_model_identity(data: dict) -> bool:
    """True when the config carries any recognizable model_type/architectures."""
    for cfg in config_scopes(data):
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
    data = load_model_config_dict(model_path)
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


_MLA_KEYS = ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "q_lora_rank")
_MOE_EXPERT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")
_SHARED_EXPERT_KEYS = ("n_shared_experts", "num_shared_experts", "moe_num_shared_experts")
# Each family names its multi-token-prediction layer count differently (DeepSeek/GLM, Qwen3.5, MiniMax).
_MTP_LAYER_KEYS = ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_modules")
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
    """Return the checkpoint's weight format (e.g. ``fp8``, ``fp4``) or '' when unquantized.

    Quark and compressed-tensors put their own name in ``quant_method`` and the format one level down, so a nested
    format answers before the method does.
    """
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict):
        return checkpoint_format(qc) or str(qc.get("quant_method") or "").strip()
    return ""


def _deepseek_family(mt: str, name: str) -> str:
    """DeepSeek keeps its major version; v3 is checked before v2 because deepseek_v32 carries both."""
    if "v4" in mt:
        return "deepseek_v4"
    if "v3" in mt or mt == "deepseek":
        return "deepseek_v3"
    if "v2" in mt:
        return "deepseek_v2"
    return "deepseek"


def _qwen_family(mt: str, name: str) -> str:
    """Qwen collapses each generation's variants onto the generation."""
    if mt.startswith("qwen3"):
        return "qwen3"
    if mt.startswith("qwen2"):
        return "qwen2"
    if mt.startswith("qwen1") or "qwen1.5" in name:
        return "qwen1.5"
    return "qwen"


def _gemma_family(mt: str, name: str) -> str:
    """Gemma generations are already in the model_type; a generation with no rule here declines."""
    for generation in ("gemma4", "gemma3", "gemma2"):
        if mt.startswith(generation):
            return generation
    return "gemma" if mt == "gemma" else ""


def _llama_family(mt: str, name: str) -> str:
    """Llama's model_type is the bare family, so the generation comes from the directory name."""
    if mt == "llama4" or "llama-4" in name or "llama4" in name:
        return "llama4"
    if "llama-3" in name or "llama3" in name or "llama_3" in name:
        return "llama3"
    if "llama-2" in name or "llama2" in name or "llama_2" in name:
        return "llama2"
    return "llama"


def _glm_family(mt: str, name: str) -> str:
    """GLM4 is the only generation this tree distinguishes."""
    return "glm4" if "4" in mt else "glm"


def _phi_family(mt: str, name: str) -> str:
    """Phi3 is the only generation this tree distinguishes."""
    return "phi3" if mt.startswith("phi3") else "phi"


# Families whose generation has to be derived, keyed on the model_type prefix that selects the rule.
# A resolver returns "" to decline, leaving the generic fallbacks below to answer.
_FAMILY_RESOLVERS = (
    ("deepseek", _deepseek_family),
    ("qwen", _qwen_family),
    ("gemma", _gemma_family),
    ("llama", _llama_family),
    ("glm", _glm_family),
    ("phi", _phi_family),
)

# Families that collapse every sub-variant onto one name. Mistral and Mixtral are kept distinct.
_COLLAPSED_FAMILY_PREFIXES = ("mixtral", "mistral", "minimax", "nemotron", "internvl")


def _derive_model_family(model_type: str, model_path: str) -> str:
    """Infer the base model family with generation (e.g. qwen3, deepseek_v3)."""
    mt = str(model_type or "").strip().lower()
    name = Path(model_path or "").name.lower()
    for prefix, resolve in _FAMILY_RESOLVERS:
        if mt.startswith(prefix):
            family = resolve(mt, name)
            if family:
                return family
    for prefix in _COLLAPSED_FAMILY_PREFIXES:
        if mt.startswith(prefix):
            return prefix
    if not mt:
        return ""
    # Derived/hybrid types: map to the base family token in the model_type.
    for token in _FAMILY_TOKENS:
        if token in mt:
            return token
    # Generic fallback: family prefix before the first separator.
    return mt.split("_")[0]


def _attention_shape(cfg: dict) -> dict:
    """The attention head counts and head dimension this config states or implies."""
    heads = to_int(cfg.get("num_attention_heads")) or 0
    kv_raw = cfg.get("num_key_value_heads")
    kv = (to_int(kv_raw) if kv_raw is not None else heads) or 0
    head_dim = to_int(cfg.get("head_dim"))
    hidden = to_int(cfg.get("hidden_size"))
    if not head_dim and hidden and heads:
        head_dim = hidden // heads
    shape: dict = {}
    attn = _derive_attention_type(cfg)
    if attn:
        shape["attention_type"] = attn
    if heads:
        shape["num_attention_heads"] = heads
    if kv:
        shape["num_key_value_heads"] = kv
    if head_dim:
        shape["head_dim"] = head_dim
    return shape


def _first_int_under(cfg: dict, keys) -> int:
    """The first non-zero integer any of these keys carries, or 0."""
    for key in keys:
        value = to_int(cfg.get(key))
        if value:
            return value
    return 0


def _moe_shape(cfg: dict) -> dict:
    """Whether this config describes an MoE model, and the expert counts if it does.

    Shared-expert detection is reported only alongside ``is_moe``, so a dense
    model carrying a shared-looking key is not read as having one.
    """
    num_experts = _first_int_under(cfg, _MOE_EXPERT_KEYS)
    experts_per_tok = to_int(cfg.get("num_experts_per_tok")) or 0
    shape: dict = {"is_moe": num_experts > 0}
    if num_experts > 0:
        shape["num_experts"] = num_experts
    if experts_per_tok > 0:
        shape["num_experts_per_tok"] = experts_per_tok
    if not shape["is_moe"]:
        return shape
    num_shared = _first_int_under(cfg, _SHARED_EXPERT_KEYS)
    shared_evidence = (
        num_shared > 0 or bool(cfg.get("shared_expert_intermediate_size")) or bool(cfg.get("shared_experts"))
    )
    if shared_evidence:
        shape["has_shared_expert"] = True
        if num_shared > 0:
            shape["num_shared_experts"] = num_shared
    return shape


def summarize_model_config(model_path: str) -> dict:
    """Best-effort structured summary of a model's ``config.json`` ({} on failure)."""
    data = load_model_config_dict(model_path)
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

    out.update(_attention_shape(cfg))

    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size", "max_position_embeddings"):
        value = to_int(cfg.get(key))
        if value is not None:
            out[key] = value

    out.update(_moe_shape(cfg))
    # Always written, so an absent key marks a summary that predates the field, not a model without MTP.
    out["mtp_layers"] = _first_int_under(cfg, _MTP_LAYER_KEYS)

    quant = _derive_quantization(cfg)
    if quant:
        out["quantization"] = quant
    for key in ("torch_dtype", "kv_cache_dtype"):
        value = str(cfg.get(key) or "").strip()
        if value:
            out[key] = value
    return out


def model_is_gemma2(model_path: str) -> bool:
    """Best-effort detect a Gemma2 model from config.json (top level or text_config)."""
    data = load_model_config_dict(model_path)
    if data is not None:
        if _config_is_gemma2(data):
            return True
        if _config_has_model_identity(data):
            return False
    matched = _path_looks_like_gemma2(model_path)
    if matched and data is None:
        log.warning(
            "Gemma2 detected via path heuristic (no readable config.json at %r); shape-discovery skip may be imprecise.",
            model_path,
        )
    return matched


def _sparse_kv_block_size(model_path: str) -> int | None:
    """Return the KV-cache block size a sparse-attention model requires, or None."""
    data = load_model_config_dict(model_path)
    if not isinstance(data, dict):
        return None
    sparse = _merge_config_scopes(data).get("sparse_attention_config")
    if not isinstance(sparse, dict):
        return None
    return to_int(sparse.get("sparse_block_size"))


# Path/architecture keyword fallbacks for model_class inference, used when the config carries no expert counts.
_MODEL_CLASS_TEXT_KEYS = ("model_type", "attention_type", "attn_type")
_MOE_KEYWORDS = ("moe", "mixtral", "deepseek-v2", "deepseek-v3", "deepseek-r1", "kimi", "glm-5", "glm5")
_MLA_KEYWORDS = ("mla", "multi-head latent", "deepseek", "kimi", "glm-5", "glm5")
_NSA_KEYWORDS = ("nsa", "native sparse attention", "glm-5", "glm5")


def infer_model_class(model_path: str) -> str:
    """Infer a deterministic model_class from local model metadata."""
    raw_path = model_path.strip()
    config = load_model_config_dict(raw_path) or {}
    # A multimodal checkpoint keeps the language model one level down, so the expert counts and the LM architecture
    # live there rather than at the top; a VL wrapper would otherwise classify as dense.
    scopes = [config, *(config[k] for k in _TEXT_SCOPE_KEYS if isinstance(config.get(k), dict))]
    expert_keys = (*_MOE_EXPERT_KEYS, "moe_num_experts")
    has_experts = any(optional_positive_int(s.get(k)) for s in scopes for k in expert_keys)
    text = " ".join(
        [raw_path.lower()]
        + [a.lower() for s in scopes for a in _config_architectures(s)]
        + [str(s[k]).lower() for s in scopes for k in _MODEL_CLASS_TEXT_KEYS if s.get(k)]
    )
    is_moe = has_experts or any(k in text for k in _MOE_KEYWORDS)
    if is_moe and any(k in text for k in _MLA_KEYWORDS):
        return "moe_mla_nsa" if any(k in text for k in _NSA_KEYWORDS) else "moe_mla"
    return "moe_swa" if is_moe else "dense"
