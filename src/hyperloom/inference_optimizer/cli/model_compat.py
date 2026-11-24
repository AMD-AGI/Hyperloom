# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Static model-config compatibility detectors and the ordered waterfall the model-config gate runs."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import framework_registry
from .. import gpu_types as _gpu_types
from ...common.model_paths import resolve_local_model_dir
from ..model_config_utils import (
    load_model_config_dict,
    _model_has_dual_chunk_attention,
)


_AMD_UNSUPPORTED_QUANT_ALGOS = frozenset({"nvfp4", "fp4"})


_AMD_UNSUPPORTED_QUANT_METHODS = frozenset({"bitsandbytes", "bnb"})

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


def _detect_amd_unsupported_quant(model_path: str) -> str | None:
    """Return a reason when the model ships a quant format unsupported on ROCm."""
    if not model_path:
        return None
    cfg = load_model_config_dict(model_path) or {}
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
    """Detect MLX (mx.quantize) checkpoints by their ``.biases``/``.scales`` tensors in the safetensors index."""
    idx = (resolve_local_model_dir(model_path) or Path(model_path)) / "model.safetensors.index.json"
    if not idx.is_file():
        return None
    try:
        wm = (json.loads(idx.read_text(encoding="utf-8")) or {}).get(
            "weight_map",
        ) or {}
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if any(k.endswith((".biases", ".scales")) for k in wm):
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


# Local tokenizer artifacts sglang/HF need to build a real tokenizer.
_TOKENIZER_ARTIFACT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "spiece.model",
)


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
        f"model directory has no tokenizer artifacts ({', '.join(_TOKENIZER_ARTIFACT_FILES)}) "
        "and config.json declares no AutoTokenizer."
    )


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
    fn: Callable[..., str | None]
    args: tuple[str, ...] = ()
    skip_when_scriptable: bool = False
    amd_only: bool = False


def _run_compat_detector(
    spec: DetectorSpec,
    *,
    model_path: str,
    data: dict,
) -> str | None:
    """Invoke a spec's detector with the arguments it names."""
    # Resolve a HF repo-id to its local cache dir ONCE so every disk-reading detector (hf_quant_config.json,
    # safetensors shards, tokenizer files, PEFT adapters, ...) sees a real directory.
    resolved_mp = str(resolve_local_model_dir(model_path) or model_path)
    available = {"model_path": resolved_mp, "data": data}
    return spec.fn(*(available[name] for name in spec.args))


# The model-config compatibility waterfall as an ordered table.
_COMPAT_DETECTORS: tuple[DetectorSpec, ...] = (
    DetectorSpec(
        "amd_unsupported_quant",
        _detect_amd_unsupported_quant,
        args=("model_path",),
        amd_only=True,
    ),
    DetectorSpec(
        "private_quant",
        _detect_private_quant,
        args=("model_path", "data"),
    ),
    DetectorSpec(
        "peft_adapter_only",
        _detect_peft_adapter_only_checkpoint,
        args=("model_path", "data"),
    ),
    DetectorSpec(
        "vocab_weight_shape",
        _detect_vocab_weight_shape_mismatch,
        args=("model_path", "data"),
    ),
    DetectorSpec(
        "tokenizer_artifacts",
        _detect_missing_tokenizer_files,
        args=("model_path", "data"),
        skip_when_scriptable=True,
    ),
    DetectorSpec(
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
    is_scriptable_fw = framework_registry.is_scriptable(framework)
    # Diffusers pipeline gate (before the config-absent short-circuit).
    if not is_scriptable_fw:
        pipeline_reason = _detect_diffusers_pipeline_model(model_path)
        if pipeline_reason is not None:
            return pipeline_reason
    # config.json absent (soft-degrade) / present-but-corrupt (block).
    cfg_path = (resolve_local_model_dir(model_path) or Path(model_path)) / "config.json"
    if not cfg_path.is_file():
        return None
    data = load_model_config_dict(model_path)
    if data is None:
        return (
            f"config.json at {cfg_path} is present but unparseable "
            f"(corrupt JSON or not a JSON object); the framework would crash "
            f"at config load."
        )
    # The first detector in table order that returns a reason wins.
    is_amd = bool(_gpu_types._resolve_amd_gpu_type(gpu_type))
    for spec in _COMPAT_DETECTORS:
        if spec.amd_only and not is_amd:
            continue
        if spec.skip_when_scriptable and is_scriptable_fw:
            continue
        reason = _run_compat_detector(
            spec,
            model_path=model_path,
            data=data,
        )
        if reason is not None:
            return reason
    return None
