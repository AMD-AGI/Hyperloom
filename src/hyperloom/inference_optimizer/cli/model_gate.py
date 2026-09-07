# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Model / GPU gate for the CLI: GPU-type resolution, arch / config loading, unsupported-model detection, and the
pre-flight gates that run before a session is born.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import gpu_types as _gpu_types
from ...common.timeutil import now_iso
from ..model_config_utils import (  # noqa: F401 - re-exported for callers/tests
    GEMMA2_ARCHITECTURES as _GEMMA2_ARCHITECTURES,
    _config_architectures,
    _load_model_config_dict,
    resolve_local_model_dir,
)

# Re-exported from model_config_utils for callers/tests.
__all__ = ["_GEMMA2_ARCHITECTURES", "_config_architectures", "_load_model_config_dict"]

log = logging.getLogger(__name__)

# GPU-type resolution.
_AMD_GPU_TYPES = _gpu_types._AMD_GPU_TYPES
_GFX_TO_RUNNER = _gpu_types._GFX_TO_RUNNER
_gpu_runner_type = _gpu_types._gpu_runner_type
_resolve_gpu_type = _gpu_types._resolve_gpu_type


def _autodetect_gpu_type() -> str | None:
    """Return mi300x|mi308x|mi325x|mi355x or None if undetectable (rocm-smi then torch gcnArchName, best-effort)."""
    return _gpu_types._autodetect_gpu_type()


def _resolve_amd_gpu_type(explicit: str | None = None) -> str | None:
    """Resolve the current AMD GPU type, or None when not on AMD/unknown."""
    explicit_norm = str(explicit or "").strip().lower()
    if explicit_norm:
        return explicit_norm if explicit_norm in _AMD_GPU_TYPES else None
    env_norm = os.environ.get("GPU_TYPE", "").strip().lower()
    if env_norm:
        return env_norm if env_norm in _AMD_GPU_TYPES else None
    detected = (_autodetect_gpu_type() or "").strip().lower()
    return detected if detected in _AMD_GPU_TYPES else None


_SUPPORTED_ARCH_MARKERS = (
    "ForCausalLM",
    "LMHeadModel",
    "ForCausalLMWithValueHead",
)

_SUPPORTED_MODEL_TYPES = frozenset(
    {
        "llama",
        "mistral",
        "mixtral",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "gemma",
        "gemma2",
        "phi",
        "phi3",
        "phimoe",
        "starcoder2",
        "codellama",
        "deepseek_v2",
        "deepseek_v3",
        "falcon",
        "gpt_neox",
        "gpt2",
        "opt",
        "bloom",
        "internlm",
        "internlm2",
        "yi",
        "baichuan",
        "chatglm",
        "glm",
        "glm4",
        "command-r",
        "cohere",
        "cohere2",
        "dbrx",
        "mpt",
        "olmo",
        "olmo2",
        "jamba",
        "arctic",
        "exaone",
        "granite",
        "granitemoeshared",
        "stablelm",
        "persimmon",
    }
)

_UNSUPPORTED_MODEL_TYPES = frozenset(
    {
        # RWKV6/Qwen2 hybrid identifiable by model_type alone in some checkpoints.
        "rwkv6qwen2",
        "gemma3",
        "mllama",
        "llava",
        "llava_next",
        "qwen2_vl",
        "qwen2_5_vl",
        "idefics",
        "idefics2",
        "idefics3",
        "paligemma",
        "pixtral",
        "internvl_chat",
        "phi3_v",
    }
)

_UNSUPPORTED_ARCHITECTURES = frozenset(
    {
        # RWKV6/Qwen2 hybrid linear-attention arch: fails ModelConfig validation at boot.
        "RWKV6Qwen2ForCausalLM",
        "Gemma3ForConditionalGeneration",
        "InternVLChatModel",
        "Phi3VForCausalLM",
        "LlavaForConditionalGeneration",
        "LlavaNextForConditionalGeneration",
        "MllamaForConditionalGeneration",
        "PaliGemmaForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Idefics2ForConditionalGeneration",
        "Idefics3ForConditionalGeneration",
        "PixtralForConditionalGeneration",
    }
)

_UNSUPPORTED_CONFIG_KEYS = (
    "vision_config",
    "image_token_id",
    "image_token_index",
    "mm_config",
    "multi_modal_config",
    "vision_tower",
    "vision_tower_cfg",
    "image_processor_type",
    "projector_config",
    "mm_projector_type",
)

_TEXT_DECODER_CONFIG_KEYS = (
    "text_config",
    "language_config",
    "llm_config",
)

_VERDICT_TEXT_COERCIBLE = "text_coercible"

_VERDICT_VISION_ONLY = "vision_only"

_TEXT_COERCIBLE_MODEL_TYPES = frozenset(
    {
        "kimi_k25",
        "qwen3_5_moe",
    }
)

_MAXPOS_CONFIG_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",  # HuggingFace tokenizer_config field; used by some custom models (e.g. kimi_linear)
)

_ROPE_CONFIG_KEYS = ("rope_scaling", "rope_parameters", "rope_theta")

# minimax_m1: its lightning-attention kernel needs 128KB LDS but MI300X's per-CU shared-memory limit is 64KB → "out of
# resource: shared memory" at engine init.
_AMD_UNSUPPORTED_MODEL_TYPES = frozenset({"deepseek_v32", "minimax_m1"})

_AMD_UNSUPPORTED_ARCHITECTURES = frozenset(
    {
        "deepseekv32forcausallm",
        "minimaxm1forcausallm",
    }
)

_UNREGISTERED_CUSTOM_CONFIG_TYPES = frozenset({"kimi_k2"})

# Architectures Transformers/sglang's ModelConfig does not recognize at all (hardware-agnostic): ModelConfig
# validation raises a ValidationError in engine init regardless of GPU vendor.
_UNRECOGNIZED_MODEL_TYPES = frozenset(
    {
        "glm4_moe_lite",
        "mimo_v2_flash",
    }
)
_UNRECOGNIZED_ARCHITECTURES = frozenset(
    {
        "glm4moeliteforcausallm",
        "mimov2flashforcausallm",
    }
)
# Some model_type values only appear inside nested decoder configs carried by a wrapper, so these are checked only
# against the nested text_config scope. ministral3: Mistral3 multimodal wrapper (vLLM registry raises
# KeyError('ministral3') for text_config.model_type).
_NESTED_ONLY_UNRECOGNIZED_MODEL_TYPES = frozenset(
    {
        "ministral3",
    }
)

_PHI3_ROPE_TYPES = frozenset({"su", "longrope"})
_STRICT_BOOL_CONFIG_KEYS = ("use_cache",)

_AMD_UNSUPPORTED_QUANT_ALGOS = frozenset({"nvfp4", "fp4"})

_AMD_UNSUPPORTED_QUANT_METHODS = frozenset({"bitsandbytes", "bnb"})

# Quark PTQ MX-FP4 (W4A4) MoE is implemented in sglang only on its aiter MoE runner; every other backend leaves the
# scheme without a ``runner`` attribute and the server dies on the first forward pass.
_NATIVE_MOE_RUNNER_QUANT_METHODS = frozenset({"quark"})

# MX group size, mirroring sglang's ``QuarkConfig._is_mx_fp4`` validation.
_MX_FP4_GROUP_SIZE = 32

# sglang resolves a layer's quant config from these, most specific first.
_QUARK_LAYER_CONFIG_KEYS = ("layer_quant_config", "layer_type_quant_config")

# Quant methods with a real vLLM/sglang loader.
_SUPPORTED_QUANT_METHODS = frozenset(
    {
        "fp8",
        "mxfp8",
        "mxfp4",
        "nvfp4",
        "blockwise_int8",
        "modelopt",
        "modelopt_fp8",
        "modelopt_fp4",
        "modelopt_mixed",
        "w8a8_int8",
        "w8a8_fp8",
        "w4afp8",
        "awq",
        "awq_marlin",
        "gptq",
        "gptq_marlin",
        "moe_wna16",
        "compressed-tensors",
        "compressed_tensors",
        "qoq",
        "petit_nvfp4",
        "fbgemm_fp8",
        "quark",
        "quark_int4fp8_moe",
        "auto-round",
        "modelslim",
        "bitsandbytes",
        "bnb",
        "gguf",
        "torchao",
    }
)
# MLX mx.quantize uses a ``mode: affine/mlx`` block and emits per-tensor ``.biases`` / ``.scales`` weights (plural —
# distinct from a standard ``.bias``).
_MLX_QUANT_MODES = frozenset({"affine", "mlx"})


def _read_preseeded_model_arch(arch_path: Path) -> str | None:
    """Return the pre-seeded ``$HYPERLOOM_MODEL_ARCH_FILE`` text, or ``None``."""
    src = (os.environ.get("HYPERLOOM_MODEL_ARCH_FILE") or "").strip()
    if not src:
        return None
    try:
        text = Path(src).expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        logging.warning("model_arch_preseed_unreadable: %s (%s)", src, exc)
        return None
    try:
        arch_path.parent.mkdir(parents=True, exist_ok=True)
        arch_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logging.warning("model_arch_preseed_copy_failed: %s -> %s (%s)", src, arch_path, exc)
    logging.info("model_arch_preseeded_from: %s", src)
    return text


def _load_model_arch(
    workspace_root: Path,
    model_name: str,
    launched_model: str = "",
) -> dict:
    """Best-effort loader for the advisory ``<workspace_root>/model_arch.json`` profile (prompts only)."""
    from hyperloom.common.model_paths import model_identities_match

    arch_path = workspace_root / "model_arch.json"
    try:
        raw = arch_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raw = _read_preseeded_model_arch(arch_path)
        if raw is None:
            return {}
    except OSError as exc:
        logging.warning("model_arch_unreadable: %s (%s)", arch_path, exc)
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logging.warning("model_arch_invalid_json: %s (%s)", arch_path, exc)
        return {}
    if not isinstance(data, dict):
        logging.warning("model_arch_not_a_dict: %s (got %s)", arch_path, type(data).__name__)
        return {}
    declared = str(data.get("model_name") or "").strip()
    if not declared:
        logging.warning("model_arch_missing_model_name: %s (cannot verify freshness)", arch_path)
        return {}
    if not model_identities_match(declared, model_name, launched_model):
        logging.warning(
            "model_arch_stale_or_mismatch: %s declares model_name=%r but "
            "launching model_name=%r (--model=%r) — ignoring",
            arch_path,
            declared,
            model_name,
            launched_model,
        )
        return {}
    return data


def _load_model_config_tags(model_path: str) -> dict:
    """Best-effort loader for KB architecture-identity tags (``architectures`` + ``model_type``) from config.json."""
    data = _load_model_config_dict(model_path)
    if data is None:
        return {}
    out: dict = {}
    arches = _config_architectures(data)
    if arches:
        out["architectures"] = arches
    model_type = str(data.get("model_type") or "").strip()
    if model_type:
        out["model_type"] = model_type
    return out


def _arch_is_supported_text_generation(arch: str) -> bool:
    """True when an architecture class name denotes a supported text-generation"""
    a = (arch or "").strip()
    if not a:
        return False
    return any(marker in a for marker in _SUPPORTED_ARCH_MARKERS)


def _config_declares_text_decoder(config: dict, architectures: list[str], model_type_l: str) -> bool:
    """True when config positively identifies a usable text decoder."""
    if model_type_l in _TEXT_COERCIBLE_MODEL_TYPES:
        return True
    if any(_arch_is_supported_text_generation(a) for a in architectures):
        return True

    for key in _TEXT_DECODER_CONFIG_KEYS:
        nested = config.get(key)
        if not isinstance(nested, dict):
            continue
        nested_architectures = _config_architectures(nested)
        if any(_arch_is_supported_text_generation(a) for a in nested_architectures):
            return True

        nested_model_type = str(nested.get("model_type") or "").strip().lower()
        if nested_model_type in _SUPPORTED_MODEL_TYPES or nested_model_type in _TEXT_COERCIBLE_MODEL_TYPES:
            return True

        # Some multimodal configs expose a text_config with decoder dimensions but an unseen model_type; scoped to a
        # named text block, so this does not widen fallback for a top-level mislabeled VLM.
        has_vocab = isinstance(nested.get("vocab_size"), int) and nested["vocab_size"] > 0
        has_decoder_shape = any(
            isinstance(nested.get(field), int) and nested[field] > 0
            for field in ("hidden_size", "num_hidden_layers", "intermediate_size")
        )
        if has_vocab and has_decoder_shape:
            return True

    return False


def _detect_unsupported_model(model_path: str) -> dict | None:
    """Best-effort classify a model's text-serving viability."""
    config = _load_model_config_dict(model_path)
    if config is None:
        return None
    architectures = _config_architectures(config)
    # Wrapper models may nest the real arch under text_config; merge so the unsupported-arch blocklist still matches.
    nested = config.get("text_config")
    if isinstance(nested, dict):
        for a in _config_architectures(nested):
            if a not in architectures:
                architectures.append(a)
    model_type = str(config.get("model_type") or "").strip()
    model_type_l = model_type.lower()
    nested_model_type = ""
    nested_model_type_l = ""
    if isinstance(nested, dict):
        nested_model_type = str(nested.get("model_type") or "").strip()
        nested_model_type_l = nested_model_type.lower()

    # Registry/config incompatibilities are handled by the model-config gate so they get the precise
    # model_config_incompatible stop reason.
    if _detect_unrecognized_architecture(config) is not None:
        return None

    # Hard denylist wins first: explicit VLM arch / model_type is vision_only even if it also carries a ForCausalLM
    # marker.
    for arch in architectures:
        if arch in _UNSUPPORTED_ARCHITECTURES:
            return {
                "architecture": arch,
                "model_type": model_type,
                "signal": f"unsupported architecture '{arch}'",
                "verdict": _VERDICT_VISION_ONLY,
            }
    if model_type_l in _UNSUPPORTED_MODEL_TYPES:
        return {
            "architecture": architectures[0] if architectures else "",
            "model_type": model_type,
            "signal": f"unsupported model_type '{model_type}'",
            "verdict": _VERDICT_VISION_ONLY,
        }
    if nested_model_type_l in _UNSUPPORTED_MODEL_TYPES:
        return {
            "architecture": architectures[0] if architectures else "",
            "model_type": model_type,
            "signal": f"unsupported text_config.model_type '{nested_model_type}'",
            "verdict": _VERDICT_VISION_ONLY,
        }

    # A multimodal config key is only a degrade signal, not a hard block: if a text decoder exists we coerce to the
    # text path with a warning.
    _has_text_decoder = _config_declares_text_decoder(config, architectures, model_type_l)
    for key in _UNSUPPORTED_CONFIG_KEYS:
        if key in config:
            verdict = _VERDICT_TEXT_COERCIBLE if _has_text_decoder else _VERDICT_VISION_ONLY
            return {
                "architecture": architectures[0] if architectures else "",
                "model_type": model_type,
                "signal": f"multimodal config key '{key}'",
                "verdict": verdict,
            }

    if any(_arch_is_supported_text_generation(a) for a in architectures):
        return None

    if model_type_l in _SUPPORTED_MODEL_TYPES:
        return None

    if architectures:
        return {
            "architecture": architectures[0],
            "model_type": model_type,
            "signal": (
                f"architecture '{architectures[0]}' does not match any "
                f"supported text-generation pattern "
                f"({', '.join(_SUPPORTED_ARCH_MARKERS)})"
            ),
            "verdict": _VERDICT_VISION_ONLY,
        }

    if model_type:
        return {
            "architecture": "",
            "model_type": model_type,
            "signal": (f"model_type '{model_type}' is not in the supported text-generation allowlist"),
            "verdict": _VERDICT_VISION_ONLY,
        }

    return {
        "architecture": "",
        "model_type": "",
        "signal": "config.json has neither architectures nor model_type",
        "verdict": _VERDICT_VISION_ONLY,
    }


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


def _model_is_moe(model_path: str) -> bool:
    """Best-effort detect a Mixture-of-Experts model from config.json."""
    data = _load_model_config_dict(model_path)
    if data is None:
        return False
    candidates = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    expert_keys = ("num_experts", "num_local_experts", "n_routed_experts")
    for cfg in candidates:
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


def model_supports_aiter_ck_fused_moe(model_path: str, tp: int) -> bool:
    """Whether aiter's CK fused-MoE can serve this checkpoint at this TP."""
    if not _model_is_moe(model_path):
        return True
    data = _load_model_config_dict(model_path)
    if data is None:
        return True
    candidates = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for cfg in candidates:
        size = cfg.get("moe_intermediate_size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            continue
        shards = max(1, int(tp or 1))
        return (size // shards) % 128 == 0
    return True


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


def _detect_amd_unsupported_quant(model_path: str) -> str | None:
    """Return a reason when the model ships a quant format unsupported on ROCm."""
    if not model_path:
        return None
    cfg = _load_model_config_dict(model_path) or {}
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict):
        method = str(qc.get("quant_method") or "").strip().lower()
        if method in _AMD_UNSUPPORTED_QUANT_METHODS:
            return (
                f"quantization_config.quant_method '{method}' ships CUDA-only "
                f"kernels with no ROCm equivalent; it crashes in engine init "
                f"on AMD."
            )
    # NVIDIA ModelOpt writes a separate hf_quant_config.json, not config.json.
    hq_path = Path(model_path) / "hf_quant_config.json"
    if hq_path.is_file():
        try:
            hq = json.loads(hq_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            hq = None
        if isinstance(hq, dict):
            producer = (
                str(
                    (hq.get("producer") or {}).get("name") or "",
                )
                .strip()
                .lower()
            )
            algo = (
                str(
                    (hq.get("quantization") or {}).get("quant_algo") or "",
                )
                .strip()
                .lower()
            )
            if producer == "modelopt" and algo:
                return (
                    f"NVIDIA ModelOpt '{algo.upper()}' quantization "
                    f"(hf_quant_config.json) uses vendor-specific scale packing "
                    f"with no sglang ROCm loader (e.g. 'modelopt_fp8 ... not "
                    f"supported in ROCm'); use an AMD-native (Quark) checkpoint."
                )
            if algo in _AMD_UNSUPPORTED_QUANT_ALGOS:
                return (
                    f"'{algo.upper()}' quantization needs NVIDIA Blackwell hardware; no AMD/ROCm runtime path exists."
                )
    return None


def _detect_mlx_quant_weights(model_path: str) -> str | None:
    """Detect MLX (mx.quantize) checkpoints by their ``.biases``/``.scales``"""
    idx = (resolve_local_model_dir(model_path) or Path(model_path)) / "model.safetensors.index.json"
    if not idx.is_file():
        return None
    try:
        wm = (json.loads(idx.read_text(encoding="utf-8")) or {}).get(
            "weight_map",
        ) or {}
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if any(k.endswith(".biases") or k.endswith(".scales") for k in wm):
        return (
            "checkpoint ships MLX mx.quantize weights (per-tensor '.biases'/"
            "'.scales'); no vLLM/sglang loader handles this private format, so "
            "weights fail to map in engine init (JANG/MTPLX class)."
        )
    return None


def _detect_gguf_only_checkpoint(model_path: str) -> str | None:
    """Detect a GGUF-only checkpoint with no HF-loadable weight files."""
    d = Path(model_path)
    if not d.is_dir() or not any(d.glob("*.gguf")):
        return None
    has_hf_weights = (
        any(d.glob("model*.safetensors"))
        or any(d.glob("*.safetensors.index.json"))
        or any(d.glob("pytorch_model*.bin"))
    )
    if has_hf_weights:
        return None
    return (
        "checkpoint ships only GGUF weights (llama.cpp, e.g. TQ3_4S ternary) "
        "with no HF safetensors/pytorch_model.bin weights; the default "
        "vLLM/sglang loader finds no model weights and fails in engine init."
    )


def _detect_private_quant(model_path: str, data: dict) -> str | None:
    """Reject private/third-party quantization with no vLLM/sglang loader."""
    qc = data.get("quantization_config")
    declared_supported = False
    if isinstance(qc, dict):
        raw_method = qc.get("quant_method")
        method = str(raw_method or "").strip().lower()
        if "quant_method" in qc and not method:
            return (
                "quantization_config.quant_method is empty; sglang/vLLM treats "
                "the checkpoint as quantized but cannot select a loader and "
                "fails engine init with \"Unknown quantization method: ''\"."
            )
        if method and method not in _SUPPORTED_QUANT_METHODS:
            return (
                f"quantization_config.quant_method '{method}' is a private/"
                f"third-party format with no vLLM/sglang loader; it fails in "
                f"engine init (e.g. 'Unknown quantization method')."
            )
        wfmt = str(qc.get("weight_format") or "").strip().lower()
        if "mxtq" in wfmt or "mxtq" in str(qc.get("method") or "").lower():
            return (
                "quantization_config 'mxtq' weight format (MLX/JANGTQ) has no "
                "vLLM/sglang loader; weights fail to map in engine init."
            )
        if str(qc.get("mode") or "").strip().lower() in _MLX_QUANT_MODES and not method:
            return (
                "quantization_config 'mode: affine/mlx' with no quant_method is "
                "an MLX (mx.quantize) checkpoint with no vLLM/sglang loader."
            )
        # quantization_config carries real quant params (bits/group_size/...) but declares no quant_method: sglang
        # can't pick a loader and raises "Unknown quantization method: ''" in engine init.
        if not method and any(qc.get(k) is not None for k in ("bits", "group_size", "weight_format", "weight_bits")):
            return (
                "quantization_config declares quant params (e.g. bits/group_size) "
                "but no quant_method; sglang/vLLM cannot select a loader and "
                "fails engine init with \"Unknown quantization method: ''\"."
            )
        declared_supported = bool(method)
    # A declared supported quant_method (awq/gptq/compressed-tensors/...) legitimately ships '.scales'/'.biases'; the
    # MLX weight-index tell only applies to checkpoints with NO quant_method declared.
    if not declared_supported:
        mlx_reason = _detect_mlx_quant_weights(model_path)
        if mlx_reason is not None:
            return mlx_reason
    return _detect_gguf_only_checkpoint(model_path)


def _detect_phi3_rope_scaling_incompatible(data: dict) -> str | None:
    """Return a reason when a Phi-3 su/longrope config crashes Phi3Config validation."""
    model_type = str(data.get("model_type") or "").strip().lower()
    arches = {a.lower() for a in _config_architectures(data)}
    if model_type != "phi3" and "phi3forcausallm" not in arches:
        return None
    rope = data.get("rope_scaling")
    if not isinstance(rope, dict):
        return None
    rope_type = str(rope.get("type") or "").strip().lower()
    if rope_type not in _PHI3_ROPE_TYPES:
        return None
    # The crash only triggers when a top-level rope_theta exists: transformers folds it into rope_scaling, giving 4
    # keys instead of the required 3.
    if data.get("rope_theta") is None:
        return None
    return (
        f"config.json is a Phi-3 model with rope_scaling.type='{rope_type}' "
        f"and a top-level rope_theta={data['rope_theta']}; "
        "Phi3Config._rope_scaling_validation requires a 3-key rope_scaling, but "
        "transformers folds the top-level rope_theta into it at load, so the "
        "validator sees 4 keys and raises ValueError at "
        "AutoConfig.from_pretrained — before --json-model-override-args can "
        "apply, so the engine crashes in init."
    )


def _detect_gemma2_missing_hidden_act(data: dict) -> str | None:
    """Return a reason when a Gemma2 config omits hidden_act."""
    model_type = str(data.get("model_type") or "").strip().lower()
    arches = {a.lower() for a in _config_architectures(data)}
    if model_type != "gemma2" and not (arches & _GEMMA2_ARCHITECTURES):
        return None
    scopes = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append(nested)
    if any(s.get("hidden_act") for s in scopes):
        return None
    return (
        "config.json is a Gemma2 model but lacks hidden_act (only "
        "hidden_activation may be present); sglang's gemma2 runtime reads "
        "config.hidden_act unconditionally and crashes with AttributeError "
        "in engine init."
    )


def _detect_diffusers_pipeline_model(model_path: str) -> str | None:
    """Return a reason when the directory is a Diffusers pipeline, not an LLM."""
    idx = (resolve_local_model_dir(model_path) or Path(model_path)) / "model_index.json"
    if not idx.is_file():
        return None
    try:
        data = json.loads(idx.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    class_name = str(data.get("_class_name") or "").strip()
    if class_name.endswith("Pipeline") or class_name in {
        "FluxPipeline",
        "StableDiffusionPipeline",
        "DiffusionPipeline",
    }:
        return (
            f"model_index.json declares Diffusers pipeline '{class_name or '?'}', "
            "not a decoder-only causal LM; Hyperloom text-generation benchmarks "
            "cannot serve this model."
        )
    return None


def _detect_null_strict_bool_config(data: dict) -> str | None:
    """Return a reason for config fields that strict HF validators require bool."""
    scopes = [("config", data)]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append(("text_config", nested))
    for scope_name, scope in scopes:
        for key in _STRICT_BOOL_CONFIG_KEYS:
            if key in scope and scope.get(key) is None:
                return (
                    f"{scope_name}.{key} is null, but HuggingFace/vLLM strict "
                    "config validation expects a bool and raises "
                    "StrictDataclassFieldValidationError before server init."
                )
    return None


# Local tokenizer artifacts sglang/HF need to build a real tokenizer.
_TOKENIZER_ARTIFACT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "spiece.model",
)


def _detect_unrecognized_architecture(data: dict) -> str | None:
    """Return a reason when the architecture is unknown to Transformers/sglang."""
    scopes = [(data, False)]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append((nested, True))
    for scope, is_nested in scopes:
        model_type = str(scope.get("model_type") or "").strip().lower()
        arches = {a.lower() for a in _config_architectures(scope)}
        unrecognized_types = _UNRECOGNIZED_MODEL_TYPES
        if is_nested:
            unrecognized_types = _UNRECOGNIZED_MODEL_TYPES | _NESTED_ONLY_UNRECOGNIZED_MODEL_TYPES
        if model_type in unrecognized_types or arches & _UNRECOGNIZED_ARCHITECTURES:
            label = model_type or (next(iter(arches), "") if arches else "?")
            return (
                f"model type '{label}' is not recognized by Transformers/"
                f"sglang/vLLM's ModelConfig or model registry; engine init "
                f"raises a validation/registry error. Needs a framework "
                f"upgrade or a registered architecture mapping."
            )
    return None


_VOCAB_WEIGHT_NAMES = (
    "embed_tokens.weight",
    "wte.weight",
    "word_embeddings.weight",
    "lm_head.weight",
)
_FULL_BASE_WEIGHT_NAMES = (
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "transformer.wte.weight",
    "wte.weight",
    "word_embeddings.weight",
    "lm_head.weight",
)
_PEFT_ADAPTER_WEIGHT_MARKERS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A",
    ".lora_embedding_B",
    ".modules_to_save.",
    ".base_layer.",
)
_SAFETENSORS_HEADER_LIMIT = 64 * 1024 * 1024


def _read_safetensors_header(path: Path) -> dict | None:
    """Read only the safetensors JSON header; never materialize tensor data."""
    try:
        with path.open("rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                return None
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len <= 0 or header_len > _SAFETENSORS_HEADER_LIMIT:
                return None
            header = json.loads(f.read(header_len))
    except (OSError, json.JSONDecodeError, ValueError, struct.error):
        return None
    return header if isinstance(header, dict) else None


def _detect_vocab_weight_shape_mismatch(model_path: str, data: dict) -> str | None:
    """Return a reason when the checkpoint has FEWER vocab rows than config."""
    expected = data.get("vocab_size")
    nested = data.get("text_config")
    if not isinstance(expected, int) and isinstance(nested, dict):
        expected = nested.get("vocab_size")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        return None

    mdir = Path(model_path)
    for st_path in sorted(mdir.glob("*.safetensors")):
        header = _read_safetensors_header(st_path)
        if not header:
            continue
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            if not any(name.endswith(suffix) for suffix in _VOCAB_WEIGHT_NAMES):
                continue
            shape = meta.get("shape")
            if not (isinstance(shape, list) and shape and isinstance(shape[0], int) and not isinstance(shape[0], bool)):
                continue
            actual = shape[0]
            # Only block when the checkpoint has FEWER vocab rows than the config declares (a broken checkpoint).
            if actual < expected:
                return (
                    f"config.json vocab_size={expected} but {st_path.name}:"
                    f"{name} has only {actual} vocab rows ({expected - actual} "
                    f"short); the checkpoint cannot serve the full vocab and "
                    f"weight loading will fail."
                )
    return None


def _detect_peft_adapter_only_checkpoint(model_path: str, data: dict) -> str | None:
    """Return a reason when a checkpoint looks like an unmerged PEFT adapter."""
    mdir = Path(model_path)
    idx = mdir / "model.safetensors.index.json"
    if not idx.is_file():
        return None
    try:
        weight_map = (json.loads(idx.read_text(encoding="utf-8")) or {}).get(
            "weight_map",
        ) or {}
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(weight_map, dict) or not weight_map:
        return None

    keys = {str(k) for k in weight_map}
    has_adapter_tensors = any(marker in key for key in keys for marker in _PEFT_ADAPTER_WEIGHT_MARKERS)
    has_adapter_manifest = (mdir / "adapter_config.json").is_file() or any(
        "adapter" in str(v).lower() for v in weight_map.values()
    )
    if not has_adapter_tensors and not has_adapter_manifest:
        return None

    has_base_weights = any(key.endswith(suffix) for key in keys for suffix in _FULL_BASE_WEIGHT_NAMES)
    if has_base_weights:
        return None

    return (
        "checkpoint appears to be an unmerged PEFT/LoRA adapter: "
        "model.safetensors.index.json contains adapter/base_layer tensor names "
        "but no full base embedding or lm_head weights. The default "
        "vLLM/sglang loader cannot reconstruct missing base tensors such as "
        "base_model.model.lm_head.base_layer.weight; merge the adapter into the "
        "base model before running Hyperloom."
    )


def _detect_missing_tokenizer_files(model_path: str, data: dict) -> str | None:
    """Return a reason when a local checkpoint ships no tokenizer artifacts."""
    auto_map = data.get("auto_map")
    if isinstance(auto_map, dict) and auto_map.get("AutoTokenizer"):
        return None
    mdir = Path(model_path)
    if any((mdir / f).is_file() for f in _TOKENIZER_ARTIFACT_FILES):
        return None
    return (
        "model directory ships weights + config but no tokenizer artifacts "
        f"({', '.join(_TOKENIZER_ARTIFACT_FILES)}); sglang loads a degraded "
        "fallback tokenizer whose warmup encodes an empty prompt, producing an "
        "empty (M=0) batch that crashes the aiter rotary_embedding kernel with "
        "SIGFPE on MI300X (Gensyn-Swarm fine-tune class)."
    )


def _detect_mistral_common_tokenizer_gap(model_path: str, data: dict) -> str | None:
    """Return a reason for Mistral checkpoints missing Mistral tokenizer files."""
    model_type = str(data.get("model_type") or "").strip().lower()
    arches = {str(a or "").strip() for a in _config_architectures(data)}
    if model_type != "mistral" and "MistralForCausalLM" not in arches:
        return None

    auto_map = data.get("auto_map")
    if isinstance(auto_map, dict) and auto_map.get("AutoTokenizer"):
        return None

    mdir = Path(model_path)
    if not (mdir / "tokenizer.json").is_file():
        return None
    mistral_files = (
        "tokenizer.model",
        "tokenizer.model.v3",
        "tekken.json",
        "tokenizer_config.json",
    )
    if any((mdir / f).is_file() for f in mistral_files):
        return None
    return (
        "Mistral checkpoint ships tokenizer.json but none of the tokenizer "
        "metadata/files accepted by Transformers MistralCommonBackend "
        f"({', '.join(mistral_files)}); sglang server init fails with "
        '"No tokenizer file found".'
    )


def _detect_llama_sentencepiece_metadata_gap(model_path: str, data: dict) -> str | None:
    """Return a reason for Llama checkpoints with bare SentencePiece tokenizer."""
    model_type = str(data.get("model_type") or "").strip().lower()
    arches = {str(a or "").strip() for a in _config_architectures(data)}
    if model_type != "llama" and "LlamaForCausalLM" not in arches:
        return None

    auto_map = data.get("auto_map")
    if isinstance(auto_map, dict) and auto_map.get("AutoTokenizer"):
        return None

    mdir = Path(model_path)
    if not (mdir / "tokenizer.model").is_file():
        return None
    if (mdir / "tokenizer_config.json").is_file() or (mdir / "tokenizer.json").is_file():
        return None
    return (
        "Llama checkpoint ships tokenizer.model but lacks tokenizer_config.json "
        "or tokenizer.json; sglang falls back through a local-path tokenizer "
        "resolution path that raises HFValidationError before server init."
    )


def _framework_is_scriptable(framework: str | None) -> bool:
    """True when ``framework`` is a scriptable diffusion runtime (e.g. xDiT)."""
    try:
        from .. import framework_registry as _fr

        return _fr.is_scriptable(framework)
    except Exception:  # noqa: BLE001 — registry import must never block the gate
        return str(framework or "").strip().lower() == "xdit"


def _detect_amd_unsupported_architecture(data: dict) -> str | None:
    """Return a reason when the architecture has no AMD/ROCm runtime path."""
    model_type = str(data.get("model_type") or "").strip().lower()
    arches = {a.lower() for a in _config_architectures(data)}
    if model_type in _AMD_UNSUPPORTED_MODEL_TYPES or arches & _AMD_UNSUPPORTED_ARCHITECTURES:
        label = model_type or (next(iter(arches), "") if arches else "?")
        return (
            f"model architecture '{label}' has no AMD/ROCm runtime path "
            f"(needs a vendor engine on NVIDIA Hopper/Blackwell, e.g. "
            f"DeepSeek Sparse Attention); it crashes in engine init on "
            f"this hardware."
        )
    return None


def _detect_rope_without_max_position(data: dict) -> str | None:
    """Return a reason when a RoPE block ships with no max-position field."""
    scopes = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append(nested)
    has_rope = any(s.get(k) for s in scopes for k in _ROPE_CONFIG_KEYS)
    has_maxpos = any(
        isinstance(s.get(k), int) and not isinstance(s.get(k), bool) and s.get(k) > 0
        for s in scopes
        for k in _MAXPOS_CONFIG_KEYS
    )
    if has_rope and not has_maxpos:
        return (
            "config.json declares a RoPE block "
            f"({', '.join(_ROPE_CONFIG_KEYS)}) but no max-position field "
            f"({', '.join(_MAXPOS_CONFIG_KEYS)}); transformers/vLLM rope "
            "init dereferences a missing max_position_embeddings and crashes "
            "in engine init (DeepSeek-V3.2-Exp class)."
        )
    return None


def _detect_unregistered_custom_autoconfig(data: dict) -> str | None:
    """Return a reason for a custom AutoConfig with an unregistered model_type."""
    auto_map = data.get("auto_map")
    model_type = str(data.get("model_type") or "").strip().lower()
    if isinstance(auto_map, dict) and auto_map.get("AutoConfig") and model_type in _UNREGISTERED_CUSTOM_CONFIG_TYPES:
        return (
            f"model_type '{model_type}' ships a custom AutoConfig "
            f"({auto_map['AutoConfig']}) but is not registered in sglang/"
            f"vLLM's config mapping; the engine falls back to "
            f"PreTrainedConfig which lacks key attributes "
            f"(max_position_embeddings) and crashes in init."
        )
    return None


def _detect_amd_dual_chunk_attention(model_path: str) -> str | None:
    """Return a reason when a model needs the AMD-unsupported dual-chunk backend."""
    if not _model_has_dual_chunk_attention(model_path):
        return None
    return (
        "model declares dual_chunk_attention_config but sglang requires "
        "the dual_chunk_flash_attn backend which only builds on sm90+ "
        "(NVIDIA Hopper); no compatible backend exists for AMD/ROCm."
    )


@dataclass(frozen=True)
class DetectorSpec:
    """One entry in the model-config compatibility waterfall."""

    name: str
    fn: Callable[..., str | None] | tuple[Callable[..., str | None], ...]
    args: tuple[str, ...] = ()
    skip_when_scriptable: bool = False
    amd_only: bool = False


def _run_compat_detector(
    spec: DetectorSpec,
    *,
    model_path: str,
    data: dict,
    gpu_type: str | None,
) -> str | None:
    """Invoke a spec's detector sub-chain, returning the first non-None reason."""
    # Resolve a HF repo-id to its local cache dir ONCE so every disk-reading detector (hf_quant_config.json,
    # safetensors shards, tokenizer files, PEFT adapters, ...) sees a real directory.
    resolved_mp = str(resolve_local_model_dir(model_path) or model_path)
    available = {"model_path": resolved_mp, "data": data, "gpu_type": gpu_type}
    call_args = tuple(available[name] for name in spec.args)
    fns = spec.fn if isinstance(spec.fn, tuple) else (spec.fn,)
    for fn in fns:
        reason = fn(*call_args)
        if reason is not None:
            return reason
    return None


# The model-config compatibility waterfall as an ordered table.
_COMPAT_DETECTORS: tuple[DetectorSpec, ...] = (
    DetectorSpec(  # 3
        "amd_unsupported_quant",
        _detect_amd_unsupported_quant,
        args=("model_path",),
        amd_only=True,
    ),
    DetectorSpec(  # 4
        "amd_unsupported_architecture",
        _detect_amd_unsupported_architecture,
        args=("data",),
        amd_only=True,
    ),
    DetectorSpec(  # 5
        "null_strict_bool",
        _detect_null_strict_bool_config,
        args=("data",),
    ),
    DetectorSpec(  # 6
        "rope_without_max_position",
        _detect_rope_without_max_position,
        args=("data",),
    ),
    DetectorSpec(  # 7
        "phi3_rope_scaling",
        _detect_phi3_rope_scaling_incompatible,
        args=("data",),
    ),
    DetectorSpec(  # 8
        "gemma2_hidden_act",
        _detect_gemma2_missing_hidden_act,
        args=("data",),
    ),
    DetectorSpec(  # 9
        "unrecognized_architecture",
        _detect_unrecognized_architecture,
        args=("data",),
    ),
    DetectorSpec(  # 10
        "private_quant",
        _detect_private_quant,
        args=("model_path", "data"),
    ),
    DetectorSpec(  # 11
        "peft_adapter_only",
        _detect_peft_adapter_only_checkpoint,
        args=("model_path", "data"),
    ),
    DetectorSpec(  # 12
        "vocab_weight_shape",
        _detect_vocab_weight_shape_mismatch,
        args=("model_path", "data"),
    ),
    DetectorSpec(  # 13 — tokenizer-artifact sub-chain (text-server only)
        "tokenizer_artifacts",
        (
            _detect_missing_tokenizer_files,
            _detect_mistral_common_tokenizer_gap,
            _detect_llama_sentencepiece_metadata_gap,
        ),
        args=("model_path", "data"),
        skip_when_scriptable=True,
    ),
    DetectorSpec(  # 14
        "unregistered_custom_autoconfig",
        _detect_unregistered_custom_autoconfig,
        args=("data",),
    ),
    DetectorSpec(  # 15
        "amd_dual_chunk_attention",
        _detect_amd_dual_chunk_attention,
        args=("model_path",),
        amd_only=True,
    ),
)


def _detect_incompatible_model_config(
    model_path: str,
    gpu_type: str | None = None,
    framework: str | None = None,
) -> str | None:
    """Detect a statically-knowable model-config incompatibility."""
    if not model_path:
        return None
    is_scriptable_fw = _framework_is_scriptable(framework)
    # Step 1: diffusers pipeline gate (before the config-absent short-circuit).
    if not is_scriptable_fw:
        pipeline_reason = _detect_diffusers_pipeline_model(model_path)
        if pipeline_reason is not None:
            return pipeline_reason
    # Step 2: config.json absent (soft-degrade) / present-but-corrupt (block).
    cfg_path = (resolve_local_model_dir(model_path) or Path(model_path)) / "config.json"
    if not cfg_path.is_file():
        return None
    data = _load_model_config_dict(model_path)
    if data is None:
        return (
            f"config.json at {cfg_path} is present but unparseable "
            f"(corrupt JSON or not a JSON object); the framework would crash "
            f"at config load."
        )
    # Steps 3-15: run the ordered registry, first non-None reason wins.
    is_amd = bool(_resolve_amd_gpu_type(gpu_type))
    for spec in _COMPAT_DETECTORS:
        if spec.amd_only and not is_amd:
            continue
        if spec.skip_when_scriptable and is_scriptable_fw:
            continue
        reason = _run_compat_detector(
            spec,
            model_path=model_path,
            data=data,
            gpu_type=gpu_type,
        )
        if reason is not None:
            return reason
    return None


# Pre-flight gates: validate the requested context window + model-config compatibility before a run is born.
_CONTEXT_HEADROOM_ENV = "HYPERLOOM_CONTEXT_HEADROOM_TOKENS"

_CONTEXT_HEADROOM_DEFAULT = 512

_MAX_MODEL_LEN_HEADROOM = 4096

_MODEL_GATE_ORDER = (
    "unsupported_model_arch",
    "model_config_compat",
    "context_window",
)
_MODEL_GATE_EVENT_ATTR = "_sbd_v6_model_gate_event"


def _model_gate_workload(args: argparse.Namespace) -> dict[str, Any]:
    model_path = str(getattr(args, "model", "") or "")
    return {
        "model_path": model_path,
        "model_name": str(getattr(args, "model_display_name", "") or "")
        or (Path(model_path).name if model_path else ""),
        "framework": str(getattr(args, "framework", "") or os.environ.get("FRAMEWORK", "")),
        "gpu_type": str(getattr(args, "gpu_type", "") or os.environ.get("TARGET_GPU_TYPE", "")),
        "isl": int(getattr(args, "isl", 0) or 0),
        "osl": int(getattr(args, "osl", 0) or 0),
        "allow_mm_text_fallback": bool(getattr(args, "allow_mm_text_fallback", True)),
        "headroom_tokens": _context_headroom_tokens(),
        "headroom_env": _CONTEXT_HEADROOM_ENV,
    }


def _new_model_gate_event(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "type": "model_gate",
        "kind": "model_gate",
        "status": "succeeded",
        "start_time": now_iso(timespec="seconds"),
        "end_time": "",
        "ext": {
            "run_kind": "fresh",
            "skip_reason": None,
            "failed_gate_id": None,
            "workload": _model_gate_workload(args),
            "checks": [],
            "degraded": {"active": False, "warnings": []},
        },
    }


def _load_model_gate_event(args: argparse.Namespace, session_dir: Path) -> dict[str, Any]:
    from ..session.sbd_v6 import read_timeline_event_for_update

    event = getattr(args, _MODEL_GATE_EVENT_ATTR, None)
    if not isinstance(event, dict):
        event = read_timeline_event_for_update(session_dir, "model_gate")
    if event is None or str(event.get("type") or "") != "model_gate":
        event = _new_model_gate_event(args)
    setattr(args, _MODEL_GATE_EVENT_ATTR, event)
    event.setdefault("kind", "model_gate")
    event.setdefault("status", "succeeded")
    event.setdefault("start_time", now_iso(timespec="seconds"))
    event.setdefault("end_time", "")
    ext = event.get("ext")
    if not isinstance(ext, dict):
        ext = {}
        event["ext"] = ext
    ext.setdefault("run_kind", "fresh")
    ext.setdefault("skip_reason", None)
    ext.setdefault("failed_gate_id", None)
    ext["workload"] = _model_gate_workload(args)
    checks = ext.get("checks")
    ext["checks"] = [row for row in checks if isinstance(row, dict)] if isinstance(checks, list) else []
    degraded = ext.get("degraded")
    if not isinstance(degraded, dict):
        degraded = {}
        ext["degraded"] = degraded
    degraded["active"] = bool(degraded.get("active"))
    warnings = degraded.get("warnings")
    degraded["warnings"] = [row for row in warnings if isinstance(row, dict)] if isinstance(warnings, list) else []
    return event


def _write_model_gate_event(session_dir: Path, event: dict[str, Any]) -> bool:
    from ..session.sbd_v6 import record_write_warning, write_timeline_event_at

    try:
        write_timeline_event_at(session_dir, event)
    except Exception as exc:  # noqa: BLE001 — observability must never change gate behavior
        log.warning("failed to persist SBD V6 model-gate event", exc_info=True)
        if not record_write_warning(session_dir, component="model_gate.event", exc=exc):
            log.debug("failed to persist SBD V6 model-gate write warning", exc_info=True)
        return False
    return True


def _record_model_gate_warning(session_dir: Path, *, component: str, exc: BaseException) -> None:
    """Best-effort retain a model-gate observability failure for export."""
    from ..session.sbd_v6 import record_write_warning

    if not record_write_warning(session_dir, component=component, exc=exc):
        log.debug("failed to persist SBD V6 model-gate warning", exc_info=True)


def _model_gate_status(
    checks: list[dict[str, Any]],
    *,
    skip_reason: str | None = None,
) -> str:
    statuses = {str(check.get("status") or "") for check in checks}
    if "failed" in statuses:
        return "failed"
    if "warned" in statuses or "unknown" in statuses:
        return "degraded"
    if skip_reason:
        return "skipped"
    return "succeeded"


def _model_gate_check_order(check: dict[str, Any]) -> int:
    try:
        return int(check.get("order") or 0)
    except (TypeError, ValueError):
        return len(_MODEL_GATE_ORDER) + 1


def _record_model_gate_check(
    args: argparse.Namespace,
    session_dir: Path,
    check: dict[str, Any],
    *,
    failure: dict[str, Any] | None = None,
    degraded_warning: dict[str, Any] | None = None,
) -> None:
    try:
        event = _load_model_gate_event(args, session_dir)
        ext = event["ext"]
        checks = [
            row for row in ext.get("checks", []) if isinstance(row, dict) and row.get("gate_id") != check.get("gate_id")
        ]
        checks.append(check)
        checks.sort(key=_model_gate_check_order)
        if failure is not None:
            failed_order = int(check.get("order") or 0)
            present = {str(row.get("gate_id") or "") for row in checks}
            for order, gate_id in enumerate(_MODEL_GATE_ORDER, start=1):
                if order > failed_order and gate_id not in present:
                    checks.append(
                        {
                            "gate_id": gate_id,
                            "order": order,
                            "status": "skipped",
                            "skip_reason": "prior_gate_failed",
                            "detail": {},
                        }
                    )
            checks.sort(key=_model_gate_check_order)
            ext["failed_gate_id"] = str(check.get("gate_id") or "")
            ext["failure"] = failure
            event["end_time"] = now_iso(timespec="seconds")
        if degraded_warning is not None:
            degraded = ext.setdefault("degraded", {"active": False, "warnings": []})
            degraded["active"] = True
            degraded.setdefault("warnings", []).append(degraded_warning)
        ext["checks"] = checks
        event["status"] = _model_gate_status(
            checks,
            skip_reason=str(ext.get("skip_reason") or "") or None,
        )
        _write_model_gate_event(session_dir, event)
    except Exception as exc:  # noqa: BLE001 — V6 observability must never change gate behavior
        log.warning("failed to record SBD V6 model-gate check", exc_info=True)
        _record_model_gate_warning(session_dir, component="model_gate.check", exc=exc)


def _start_model_gate(args: argparse.Namespace, session_dir: Path) -> None:
    """Create the model-gate event before the first check executes."""
    try:
        event = _new_model_gate_event(args)
        setattr(args, _MODEL_GATE_EVENT_ATTR, event)
        _write_model_gate_event(session_dir, event)
    except Exception as exc:  # noqa: BLE001 — V6 observability must never change launch behavior
        log.warning("failed to initialize SBD V6 model-gate event", exc_info=True)
        _record_model_gate_warning(session_dir, component="model_gate.start", exc=exc)


def _finish_model_gate(args: argparse.Namespace, session_dir: Path) -> None:
    """Finalize a successfully completed three-check model-gate chain."""
    try:
        event = _load_model_gate_event(args, session_dir)
        ext = event["ext"]
        event["status"] = _model_gate_status(
            ext["checks"],
            skip_reason=str(ext.get("skip_reason") or "") or None,
        )
        event["end_time"] = now_iso(timespec="seconds")
        _write_model_gate_event(session_dir, event)
    except Exception as exc:  # noqa: BLE001 — V6 observability must never change launch behavior
        log.warning("failed to finalize SBD V6 model-gate event", exc_info=True)
        _record_model_gate_warning(session_dir, component="model_gate.finish", exc=exc)


def _record_resumed_model_gate(
    args: argparse.Namespace,
    session_dir: Path,
    *,
    workload_overrides: Mapping[str, Any] | None = None,
) -> None:
    """Persist the explicit V6 skip required by the resume path."""
    try:
        timestamp = now_iso(timespec="seconds")
        event = _new_model_gate_event(args)
        event["status"] = "skipped"
        event["start_time"] = timestamp
        event["end_time"] = timestamp
        event["ext"]["run_kind"] = "resume"
        event["ext"]["skip_reason"] = "resume"
        if workload_overrides:
            event["ext"]["workload"].update(workload_overrides)
        event["ext"]["checks"] = [
            {
                "gate_id": gate_id,
                "order": order,
                "status": "skipped",
                "skip_reason": "resume",
                "detail": {},
            }
            for order, gate_id in enumerate(_MODEL_GATE_ORDER, start=1)
        ]
        setattr(args, _MODEL_GATE_EVENT_ATTR, event)
        _write_model_gate_event(session_dir, event)
    except Exception as exc:  # noqa: BLE001 — V6 observability must never change resume behavior
        log.warning("failed to record resumed SBD V6 model-gate event", exc_info=True)
        _record_model_gate_warning(session_dir, component="model_gate.resume", exc=exc)


def _write_model_gate_breakdown(
    session_dir: Path,
    *,
    failure_label: str,
) -> None:
    """Write the fail-fast SBD once without masking the gate failure."""
    try:
        from ..breakdown import write_breakdown_json

        write_breakdown_json(session_dir)
    except Exception as exc:  # noqa: BLE001 — never mask the gate failure
        print(
            f"WARNING: failed to write session_breakdown.json on {failure_label} fail-fast: {exc!r}",
            file=sys.stderr,
        )
        _record_model_gate_warning(session_dir, component=f"model_gate.{failure_label}.breakdown", exc=exc)


def _context_headroom_tokens() -> int:
    """Resolve the context headroom (tokens); env override, else default."""
    raw = os.environ.get(_CONTEXT_HEADROOM_ENV, "").strip()
    if not raw:
        return _CONTEXT_HEADROOM_DEFAULT
    try:
        val = int(raw)
    except ValueError:
        return _CONTEXT_HEADROOM_DEFAULT
    return val if val >= 0 else _CONTEXT_HEADROOM_DEFAULT


def _resolve_max_model_len(isl: int, osl: int, model_path: str) -> int:
    """Resolve ``MAX_MODEL_LEN`` = ISL+OSL+headroom, clamped to ``max_position_embeddings`` (never stretch context)."""
    desired = int(isl) + int(osl) + _MAX_MODEL_LEN_HEADROOM
    maxpos = _load_model_max_position_embeddings(model_path)
    if maxpos:
        return min(desired, maxpos)
    return desired


def _emit_breakdown_to_langfuse(session_dir: Path) -> None:
    """Best-effort: push the just-written ``session_breakdown.json`` to Langfuse."""
    try:
        from ..breakdown import patch_breakdown_langfuse
        from hyperloom.orchestrator.trace.langfuse_emitter import (
            flush_session,
            record_session_breakdown,
        )

        flush_session(session_dir)
        patch_breakdown_langfuse(session_dir)
        record_session_breakdown(session_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort; never mask the reason
        print(
            f"WARNING: failed to emit session_breakdown to Langfuse on fail-fast: {exc!r}",
            file=sys.stderr,
        )


def _preflight_context_window(args: argparse.Namespace, session_dir: Path) -> bool:
    """Fail fast when ``max_position_embeddings < ISL+OSL+headroom`` (no --context-length stretch by policy)."""
    isl = int(getattr(args, "isl", 0) or 0)
    osl = int(getattr(args, "osl", 0) or 0)
    if isl <= 0 or osl <= 0:
        _record_model_gate_check(
            args,
            session_dir,
            {
                "gate_id": "context_window",
                "order": 3,
                "status": "skipped",
                "skip_reason": "isl_osl_unset",
                "detail": {
                    "isl": isl,
                    "osl": osl,
                    "headroom": _context_headroom_tokens(),
                    "required": None,
                    "max_position_embeddings": None,
                    "fits": None,
                    "policy": "no_context_length_override",
                },
            },
        )
        return False
    maxpos = _load_model_max_position_embeddings(str(getattr(args, "model", "") or ""))
    if not maxpos:
        headroom = _context_headroom_tokens()
        _record_model_gate_check(
            args,
            session_dir,
            {
                "gate_id": "context_window",
                "order": 3,
                "status": "skipped",
                "skip_reason": "max_position_unknown",
                "detail": {
                    "isl": isl,
                    "osl": osl,
                    "headroom": headroom,
                    "required": isl + osl + headroom,
                    "max_position_embeddings": None,
                    "fits": None,
                    "policy": "no_context_length_override",
                },
            },
        )
        return False
    headroom = _context_headroom_tokens()
    required = isl + osl + headroom
    if maxpos >= required:
        _record_model_gate_check(
            args,
            session_dir,
            {
                "gate_id": "context_window",
                "order": 3,
                "status": "passed",
                "skip_reason": None,
                "detail": {
                    "isl": isl,
                    "osl": osl,
                    "headroom": headroom,
                    "required": required,
                    "max_position_embeddings": maxpos,
                    "fits": True,
                    "policy": "no_context_length_override",
                },
            },
        )
        return False

    reason = (
        f"model max_position_embeddings={maxpos} < required {required} "
        f"(ISL={isl} + OSL={osl} + headroom={headroom}). The workload exceeds "
        f"the model context window; every request would 400. Refusing to run "
        f"(no --context-length override by policy). Lower ISL/OSL for this "
        f"model, or lower {_CONTEXT_HEADROOM_ENV} if the headroom is too "
        f"conservative (it is added to `required`, so raising it makes "
        f"admission stricter, not looser)."
    )
    # Persist the stop reason so CI / the robustness monitor read it from state.json.
    try:
        from hyperloom.orchestrator.state.shared_state import SharedState
        from hyperloom.orchestrator.actions.executors.report import (
            _build_summary_dict,
            _format_md,
        )
        from ..session.session_paths import reports_dir

        state = SharedState.load_or_init(session_dir)
        # Validated writer keeps the vocab-closed invariant Inv-8.3.
        state.set_stop_reason("model_context_window_too_small")
        state.closing_phase = True
        state.save(session_dir)
        summary = _build_summary_dict(state, {}, [], external_baseline=None)
        summary["stop_detail"] = reason
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "final.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (rdir / "final.md").write_text(_format_md(summary), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — don't mask the reason on a writer bug
        print(
            f"WARNING: failed to persist context-window stop report: {exc!r}",
            file=sys.stderr,
        )
    _record_model_gate_check(
        args,
        session_dir,
        {
            "gate_id": "context_window",
            "order": 3,
            "status": "failed",
            "skip_reason": None,
            "detail": {
                "isl": isl,
                "osl": osl,
                "headroom": headroom,
                "required": required,
                "max_position_embeddings": maxpos,
                "fits": False,
                "policy": "no_context_length_override",
            },
        },
        failure={
            "gate_id": "context_window",
            "stop_reason": "model_context_window_too_small",
            "exit_code": 2,
            "message": reason,
            "artifacts": {
                "final_json": "reports/final.json" if (session_dir / "reports" / "final.json").is_file() else None,
            },
        },
    )
    # Delivery-artifact parity: emit session_breakdown.json here too since fail-fast exits before coordinator.run()'s
    # finally.
    _write_model_gate_breakdown(session_dir, failure_label="context")
    # Langfuse parity: this gate exits before coordinator.run()'s finally, so push the breakdown to Langfuse here too.
    _emit_breakdown_to_langfuse(session_dir)
    print(f"ERROR: {reason}", file=sys.stderr)
    return True


def _preflight_model_config_compat(
    args: argparse.Namespace,
    session_dir: Path,
) -> bool:
    """Fail fast when the model config is statically known to be incompatible."""
    model = str(getattr(args, "model", "") or "")
    framework = (str(getattr(args, "framework", "") or "") or os.environ.get("FRAMEWORK", "")).strip().lower() or None
    detail = _detect_incompatible_model_config(
        model,
        str(getattr(args, "gpu_type", "") or "") or None,
        framework=framework,
    )
    if detail is None:
        model_dir = resolve_local_model_dir(model) or Path(model)
        config_path = model_dir / "config.json"
        absent = not config_path.is_file()
        _record_model_gate_check(
            args,
            session_dir,
            {
                "gate_id": "model_config_compat",
                "order": 2,
                "status": "skipped" if absent else "passed",
                "skip_reason": "config_absent_soft_pass" if absent else None,
                "detail": {
                    "config_path": str(config_path) if config_path.is_file() else None,
                    "incompatible": False,
                    "reason": None,
                    "detector": None,
                },
            },
        )
        return False
    name = Path(model).name or model
    reason = (
        f"Model '{name}' has an incompatible config: {detail} Refusing to run "
        f"before the heavy server bring-up. Upgrade the framework/transformers "
        f"to a version that supports this model, or skip it on this hardware."
    )
    try:
        from hyperloom.orchestrator.state.shared_state import SharedState
        from hyperloom.orchestrator.actions.executors.report import (
            _build_summary_dict,
            _format_md,
        )
        from ..session.session_paths import reports_dir

        state = SharedState.load_or_init(session_dir)
        state.set_stop_reason("model_config_incompatible")
        state.closing_phase = True
        state.save(session_dir)
        summary = _build_summary_dict(state, {}, [], external_baseline=None)
        summary["stop_detail"] = reason
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "final.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (rdir / "final.md").write_text(_format_md(summary), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — don't mask the reason on a writer bug
        print(
            f"WARNING: failed to persist model-config stop report: {exc!r}",
            file=sys.stderr,
        )
    model_dir = resolve_local_model_dir(model) or Path(model)
    config_path = model_dir / "config.json"
    _record_model_gate_check(
        args,
        session_dir,
        {
            "gate_id": "model_config_compat",
            "order": 2,
            "status": "failed",
            "skip_reason": None,
            "detail": {
                "config_path": str(config_path) if config_path.is_file() else None,
                "incompatible": True,
                "reason": detail,
                "detector": None,
            },
        },
        failure={
            "gate_id": "model_config_compat",
            "stop_reason": "model_config_incompatible",
            "exit_code": 2,
            "message": reason,
            "artifacts": {
                "final_json": "reports/final.json" if (session_dir / "reports" / "final.json").is_file() else None,
            },
        },
    )
    _write_model_gate_breakdown(session_dir, failure_label="config")
    # Langfuse parity: this gate exits before coordinator.run()'s finally, so push the breakdown to Langfuse here too.
    _emit_breakdown_to_langfuse(session_dir)
    print(f"ERROR: {reason}", file=sys.stderr)
    return True


def _preflight_unsupported_model_arch(
    args: argparse.Namespace,
    session_dir: Path,
) -> bool:
    """Gate multimodal/vision models before expensive bring-up."""
    # Scriptable diffusion frameworks (xDiT) are server-less image workloads, not decoder-only causal LMs.
    framework = getattr(args, "framework", "") or ""
    try:
        from . import framework_registry as _fr

        is_scriptable = _fr.is_scriptable(framework)
    except Exception:  # noqa: BLE001 — registry import must never block the gate
        is_scriptable = str(framework).strip().lower() == "xdit"
    if is_scriptable:
        _record_model_gate_check(
            args,
            session_dir,
            {
                "gate_id": "unsupported_model_arch",
                "order": 1,
                "status": "skipped",
                "skip_reason": "scriptable_framework",
                "verdict": None,
                "detail": {
                    "architecture": None,
                    "model_type": None,
                    "signal": None,
                    "allow_mm_text_fallback": bool(getattr(args, "allow_mm_text_fallback", True)),
                    "action": "proceed",
                },
            },
        )
        return False

    model = str(getattr(args, "model", "") or "")
    hit = _detect_unsupported_model(model)
    if hit is None:
        config = _load_model_config_dict(model)
        architectures = _config_architectures(config) if isinstance(config, dict) else []
        _record_model_gate_check(
            args,
            session_dir,
            {
                "gate_id": "unsupported_model_arch",
                "order": 1,
                "status": "passed" if isinstance(config, dict) else "unknown",
                "skip_reason": None,
                "verdict": "plain_text" if isinstance(config, dict) else None,
                "detail": {
                    "architecture": architectures[0] if architectures else None,
                    "model_type": str(config.get("model_type") or "") if isinstance(config, dict) else None,
                    "signal": None,
                    "allow_mm_text_fallback": bool(getattr(args, "allow_mm_text_fallback", True)),
                    "action": "proceed",
                },
            },
        )
        return False

    name = Path(model).name or model
    arch = hit.get("architecture") or "<unknown>"
    mt = hit.get("model_type") or "<unknown>"
    verdict = str(hit.get("verdict") or _VERDICT_VISION_ONLY)
    allow_fallback = bool(getattr(args, "allow_mm_text_fallback", True))

    if verdict == _VERDICT_TEXT_COERCIBLE and allow_fallback:
        warning = (
            f"DEGRADED MODE: model '{name}' carries a multimodal signal "
            f"({hit.get('signal', 'multimodal config')}; architecture '{arch}', "
            f"model_type '{mt}') but exposes a text-generation path. Hyperloom "
            f"is running it on the TEXT path only — any image/audio inputs are "
            f"ignored, so benchmark numbers reflect the text decoder alone. "
            f"Pass --no-allow-mm-text-fallback to fail-fast instead."
        )
        print(f"WARNING: {warning}", file=sys.stderr)
        log.warning(warning)
        try:
            from hyperloom.orchestrator.state.shared_state import SharedState

            state = SharedState.load_or_init(session_dir)
            state.degraded_mode = True
            state.model_warnings = list(state.model_warnings or []) + [
                {
                    "kind": "multimodal_text_fallback",
                    "model_name": name,
                    "architecture": arch,
                    "model_type": mt,
                    "signal": str(hit.get("signal") or ""),
                    "detail": warning,
                }
            ]
            state.save(session_dir)
        except Exception as exc:  # noqa: BLE001 — never block the run on advisory write
            print(
                f"WARNING: failed to persist degraded-mode marker: {exc!r}",
                file=sys.stderr,
            )
        _record_model_gate_check(
            args,
            session_dir,
            {
                "gate_id": "unsupported_model_arch",
                "order": 1,
                "status": "warned",
                "skip_reason": None,
                "verdict": verdict,
                "detail": {
                    "architecture": arch,
                    "model_type": mt,
                    "signal": str(hit.get("signal") or ""),
                    "allow_mm_text_fallback": allow_fallback,
                    "action": "proceed",
                },
            },
            degraded_warning={
                "kind": "multimodal_text_fallback",
                "architecture": arch,
                "model_type": mt,
                "signal": str(hit.get("signal") or ""),
            },
        )
        return False

    reason = (
        f"Unsupported model '{name}': architecture '{arch}' (model_type "
        f"'{mt}') is not a supported text-generation model. Hyperloom only "
        f"supports decoder-only causal LM models (architectures containing "
        f"ForCausalLM or LMHeadModel). Rejected because: "
        f"{hit.get('signal', 'unknown architecture')}. Submit a "
        f"text-generation checkpoint instead."
    )
    # Persist the stop reason so CI / the robustness monitor read it from state.json.
    try:
        from hyperloom.orchestrator.state.shared_state import SharedState
        from hyperloom.orchestrator.actions.executors.report import (
            _build_summary_dict,
            _format_md,
        )
        from ..session.session_paths import reports_dir

        state = SharedState.load_or_init(session_dir)
        # Validated writer keeps the vocab-closed invariant Inv-8.3.
        state.set_stop_reason("unsupported_model_arch")
        state.closing_phase = True
        state.save(session_dir)
        summary = _build_summary_dict(state, {}, [], external_baseline=None)
        summary["stop_detail"] = reason
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "final.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (rdir / "final.md").write_text(_format_md(summary), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — don't mask the reason on a writer bug
        print(
            f"WARNING: failed to persist unsupported-model stop report: {exc!r}",
            file=sys.stderr,
        )
    _record_model_gate_check(
        args,
        session_dir,
        {
            "gate_id": "unsupported_model_arch",
            "order": 1,
            "status": "failed",
            "skip_reason": None,
            "verdict": verdict,
            "detail": {
                "architecture": arch,
                "model_type": mt,
                "signal": str(hit.get("signal") or ""),
                "allow_mm_text_fallback": allow_fallback,
                "action": "fail_fast",
            },
        },
        failure={
            "gate_id": "unsupported_model_arch",
            "stop_reason": "unsupported_model_arch",
            "exit_code": 2,
            "message": reason,
            "artifacts": {
                "final_json": "reports/final.json" if (session_dir / "reports" / "final.json").is_file() else None,
            },
        },
    )
    # Delivery-artifact parity: emit session_breakdown.json here too since fail-fast exits before coordinator.run()'s
    # finally.
    _write_model_gate_breakdown(session_dir, failure_label="unsupported-model")
    # Langfuse parity: this gate exits before coordinator.run()'s finally, so push the breakdown to Langfuse here too.
    _emit_breakdown_to_langfuse(session_dir)
    print(f"ERROR: {reason}", file=sys.stderr)
    return True
