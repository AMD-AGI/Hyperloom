# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HF checkpoint metadata (weight bytes, dtype, MoE and KV shape) read for the roofline ceiling."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.checkpoint_quant import WeightDeclaration, weight_declarations
from hyperloom.common.model_paths import resolve_local_model_dir
from hyperloom.inference_optimizer.model_config_utils import load_model_config_dict, _merge_config_scopes

log = logging.getLogger(__name__)

#: HF ``torch_dtype`` / precision tag → bytes per element (fallback when safetensors index absent).
_DTYPE_BYTES: dict[str, float] = {
    "float32": 4.0,
    "fp32": 4.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float16": 2.0,
    "fp16": 2.0,
    "float8_e4m3fn": 1.0,
    "float8_e5m2": 1.0,
    "fp8": 1.0,
    "mxfp8": 1.0,
    # 4-bit (block-scaled).
    "float4": 0.5,
    "fp4": 0.5,
    "mxfp4": 0.5,
}


def resolve_dtype_bytes(tag: str | None) -> float:
    """HF/precision tag → bytes per element; bf16 (2.0) on miss."""
    if not tag:
        return 2.0
    return _DTYPE_BYTES.get(str(tag).strip().lower(), 2.0)


#: Map a server-arg ``--quantization`` value to weight bytes-per-element.
#: Only weight-quantization methods are listed; activation/KV dtype is
#: tracked separately and stays >= bf16.
QUANT_WEIGHT_BYTES: dict[str, float] = {
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "mxfp8": 1.0,
    "fp4": 0.5,
    "mxfp4": 0.5,
    "nvfp4": 0.5,
    "int8": 1.0,
    "w8a8_int8": 1.0,
    "int4": 0.5,
    "awq": 0.5,
    "gptq": 0.5,
}


def _resolve_quant_config_weight_bytes(quant_cfg: Any) -> float:
    """On-disk weight bytes-per-element declared by an HF ``quantization_config``."""
    if not isinstance(quant_cfg, dict):
        return 0.0
    method = str(quant_cfg.get("quant_method", "")).strip().lower()
    if method in _DTYPE_BYTES:
        return _DTYPE_BYTES[method]
    if method in QUANT_WEIGHT_BYTES:
        return QUANT_WEIGHT_BYTES[method]

    def _bits_to_bytes(bits: Any) -> float:
        """``num_bits`` → bytes-per-element, ``0.0`` when unusable."""
        try:
            b = float(bits)
        except (TypeError, ValueError):
            return 0.0
        return b / 8.0 if b > 0 else 0.0

    def _declared_bytes(decl: WeightDeclaration) -> float:
        """Weight bytes-per-element one nested declaration names, ``0.0`` if none."""
        # Both tables, exactly as the flat ``quant_method`` path above does.
        if decl.dtype in _DTYPE_BYTES:
            return _DTYPE_BYTES[decl.dtype]
        if decl.dtype in QUANT_WEIGHT_BYTES:
            return QUANT_WEIGHT_BYTES[decl.dtype]
        return decl.bits / 8.0

    whole, per_layer = weight_declarations(quant_cfg)
    for decl in whole:
        by = _declared_bytes(decl)
        if by > 0:
            return by

    # The per-group containers are different: one entry per layer pattern, and they need not agree -- a Quark MoE
    # checkpoint routinely stores the routed experts at fp4 and the attention projections at fp8.
    tallies: dict[float, int] = {}
    for decl in per_layer:
        by = _declared_bytes(decl)
        if by > 0:
            tallies[by] = tallies.get(by, 0) + 1
    if tallies:
        return max(tallies.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return _bits_to_bytes(quant_cfg.get("bits") or quant_cfg.get("w_bit"))


def active_bytes(non_expert_bytes: int, expert_bytes: int, num_experts: int, experts_per_tok: int) -> int:
    """Weight bytes one token reads: every non-expert weight plus its share of the routed experts."""
    return non_expert_bytes + (int(expert_bytes * experts_per_tok / num_experts) if num_experts else 0)


@dataclass(frozen=True)
class ModelMeta:
    """HF subset needed for the decode roofline ceiling."""

    weight_bytes: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    weight_dtype_bytes: float
    # Bytes per element of the routed-expert (FFN) weights: the checkpoint's precision, or the runtime one once
    # apply_runtime_dtype has carried experts that share the general precision along with it.
    expert_weight_dtype_bytes: float
    active_weight_bytes: int = 0
    # MoE expert decomposition (0 for dense).
    num_experts: int = 0
    experts_per_tok: int = 0
    expert_weight_bytes: int = 0
    # Extra HF config fields for per-op PerfModel breakdown (0 = unavailable).
    hidden_size: int = 0
    intermediate_size: int = 0
    # Per-expert FFN dim for MoE models (0 = dense/unknown; falls back to intermediate_size).
    moe_intermediate_size: int = 0
    # Routed-expert input dim.
    moe_hidden_size: int = 0
    vocab_size: int = 0
    num_attention_heads: int = 0
    # How the decoder stack splits between routed-expert FFN layers and plain dense FFN layers.
    moe_layers: int = 0
    dense_ffn_layers: int = 0


def _read_total_size(model_path: Path) -> int | None:
    """Read ``metadata.total_size`` (bytes) from the safetensors index (byte-exact)."""
    idx = model_path / "model.safetensors.index.json"
    if idx.is_file():
        try:
            meta = json.loads(idx.read_text(encoding="utf-8")).get("metadata") or {}
            size = meta.get("total_size")
            if size:
                return int(size)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            # Malformed size metadata; fall back to the file-size scan.
            pass
    return _sum_weight_file_sizes(model_path, "*.safetensors") or _sum_weight_file_sizes(model_path, "*.bin")


def _sum_weight_file_sizes(model_path: Path, pattern: str) -> int | None:
    """Fallback weight size from local weight shards matching ``pattern``."""
    try:
        total = sum(p.stat().st_size for p in model_path.glob(pattern) if p.is_file())
    except OSError:
        return None
    return int(total) if total > 0 else None


HIDDEN_SIZE_KEY = "hidden_size"
NUM_HIDDEN_LAYERS_KEY = "num_hidden_layers"


def config_int(cfg: dict[str, Any], *keys: str) -> int:
    """The first set value among the alias ``keys``, as an ``int``; ``0`` when none is set."""
    for key in keys:
        value = cfg.get(key)
        if value:
            return int(value)
    return 0


def _derive_experts_per_tok(cfg: dict[str, Any]) -> int:
    """Routed experts activated per token, across the naming variants in use."""
    return config_int(cfg, "num_experts_per_tok", "num_experts_per_token", "top_k_experts", "moe_topk")


def _derive_moe_hidden_size(cfg: dict[str, Any]) -> int:
    """The routed experts' input dimension, which need not be ``hidden_size``."""
    return config_int(cfg, "routed_expert_hidden_size", "moe_hidden_size", HIDDEN_SIZE_KEY)


def _derive_kv_heads(cfg: dict[str, Any]) -> int:
    """GQA-aware: ``num_key_value_heads`` if present, else ``num_attention_heads``."""
    kv = cfg.get("num_key_value_heads")
    if kv is None:
        kv = cfg.get("num_attention_heads")
    return int(kv or 0)


def _derive_head_dim(cfg: dict[str, Any]) -> int:
    """``head_dim`` directly, or ``hidden_size / num_attention_heads``."""
    head_dim = cfg.get("head_dim")
    if head_dim:
        return int(head_dim)
    hidden = cfg.get(HIDDEN_SIZE_KEY)
    heads = cfg.get("num_attention_heads")
    if hidden and heads:
        return int(hidden) // int(heads)
    return 0


def _derive_moe_layer_counts(cfg: dict[str, Any], num_layers: int, num_experts: int) -> tuple[int, int]:
    """Split the decoder stack into ``(moe_layers, dense_ffn_layers)``."""
    if num_layers <= 0:
        return 0, 0
    if num_experts <= 0:
        return 0, num_layers

    freq = cfg.get("moe_layer_freq")
    if isinstance(freq, (list, tuple)) and len(freq) == num_layers:
        moe = sum(1 for v in freq if v)
        return moe, num_layers - moe

    first_dense = config_int(cfg, "first_k_dense_replace")
    first_dense = max(0, min(first_dense, num_layers))

    # The stride is phased differently by the two upstream implementations, and both phase it on the ABSOLUTE layer
    # index -- not on the offset from the dense prefix.
    stride, phase = 1, "deepseek"
    for raw, kind in ((freq, "deepseek"), (cfg.get("decoder_sparse_step"), "qwen")):
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int) and raw > 0:
            stride, phase = raw, kind
            break

    dense_only = {int(i) for i in (cfg.get("mlp_only_layers") or []) if isinstance(i, int)}
    offset = 1 if phase == "qwen" else 0
    moe = sum(1 for i in range(first_dense, num_layers) if (i + offset) % stride == 0 and i not in dense_only)
    return moe, num_layers - moe


def _compute_expert_decomposition(
    cfg: dict[str, Any],
    *,
    weight_bytes: int,
    expert_bpe: float,
) -> tuple[int, int, int, int]:
    """MoE decomposition for the batch-aware roofline; returns ``(active_weight_bytes, total_expert_bytes, num_experts, experts_per_tok)``. Safe-degrades to ``(weight_bytes, 0, 0, 0)``. Handles num_experts / n_routed_experts / num_local_experts aliases, the per-token aliases in :func:`_derive_experts_per_tok`, and the latent-MoE expert width in :func:`_derive_moe_hidden_size`."""
    num_experts = config_int(cfg, "num_experts", "n_routed_experts", "num_local_experts")
    experts_per_tok = _derive_experts_per_tok(cfg)
    if num_experts <= 0 or experts_per_tok <= 0:
        return int(weight_bytes), 0, 0, 0
    hidden_size = _derive_moe_hidden_size(cfg)
    num_layers = config_int(cfg, NUM_HIDDEN_LAYERS_KEY)
    moe_inter = config_int(cfg, "moe_intermediate_size", "intermediate_size")
    if hidden_size <= 0 or num_layers <= 0 or moe_inter <= 0 or expert_bpe <= 0:
        return int(weight_bytes), 0, 0, 0
    # Only the MoE layers hold expert weights.
    moe_layers, _dense_ffn_layers = _derive_moe_layer_counts(cfg, num_layers, num_experts)
    if moe_layers <= 0:
        return int(weight_bytes), 0, 0, 0
    expert_bytes_per_layer = num_experts * 3 * hidden_size * moe_inter * expert_bpe
    total_expert_bytes = int(moe_layers * expert_bytes_per_layer)
    if total_expert_bytes <= 0 or total_expert_bytes >= int(weight_bytes):
        # Degrading a MoE model to dense drops the largest op from the breakdown entirely, so say so: the usual cause
        # is an over-large ``expert_bpe`` (a quantized checkpoint misread as bf16), not a genuinely dense model.
        log.warning(
            "MoE decomposition degraded to dense: computed expert bytes %.4g >= checkpoint bytes %.4g "
            "(num_experts=%d, moe_intermediate_size=%d, expert_bytes_per_element=%.4g)",
            total_expert_bytes,
            float(weight_bytes),
            num_experts,
            moe_inter,
            expert_bpe,
        )
        return int(weight_bytes), 0, 0, 0
    return (
        active_bytes(int(weight_bytes) - total_expert_bytes, total_expert_bytes, num_experts, experts_per_tok),
        total_expert_bytes,
        num_experts,
        experts_per_tok,
    )


def load_model_meta(
    model_path: str | Path,
    *,
    precision_hint: str = "",
) -> ModelMeta | None:
    """Read ``weight_bytes`` + KV-cache shape from a local HF model dir (``None`` when unreadable). Weight-dtype priority: quantization_config.quant_method > torch_dtype > dtype > precision_hint."""
    if not model_path:
        return None
    p = Path(model_path).expanduser()
    if not p.is_dir():
        # ``model_path`` may be an HF repo id (the CLI ``--model`` is persisted verbatim); resolve it to the engine's
        # local HF cache dir via the shared resolver so the decode ceiling isn't null for repo-id launches.
        resolved = resolve_local_model_dir(model_path)
        if resolved is None:
            return None
        p = resolved
    data = load_model_config_dict(str(p))
    if data is None:
        return None
    # Multimodal wrappers (e.g. Kimi-K2 kimi_k25) nest the real decoder shape / MoE config under
    # text_config/llm_config/language_config; flatten it (nested wins) so num_experts / hidden_size /
    # moe_intermediate_size / kv-heads reach the ceiling readers instead of degrading to a dense full-weight roofline
    # (num_experts=0 -> active_weight_bytes = full weight_bytes; PerfModel falls back to the legacy top-down formula).
    cfg = _merge_config_scopes(data)
    weight_bytes = _read_total_size(p)
    if not weight_bytes:
        return None
    quant_tag = ""
    quant_cfg = cfg.get("quantization_config")
    if isinstance(quant_cfg, dict):
        quant_tag = str(quant_cfg.get("quant_method", "")).strip().lower()
    # A declared quantization wins outright: ``quant_method`` alone is not enough (Quark says ``"quark"``, precision
    # lives on the nested weight spec), and falling through to the checkpoint ``dtype`` would report bf16 for a 4-bit
    # checkpoint.
    quant_bytes = _resolve_quant_config_weight_bytes(quant_cfg)
    if quant_bytes > 0:
        dtype_bytes = quant_bytes
    else:
        dtype_bytes = resolve_dtype_bytes(quant_tag or cfg.get("torch_dtype") or cfg.get("dtype") or precision_hint)
    # Routed experts may be stored at a distinct precision (DeepSeek-V4 ``expert_dtype: fp4`` under fp8 attention).
    expert_dtype_raw = str(cfg.get("expert_dtype") or "").strip()
    expert_dtype_bytes = resolve_dtype_bytes(expert_dtype_raw) if expert_dtype_raw else dtype_bytes
    active_weight_bytes, total_expert_bytes, num_experts, experts_per_tok = _compute_expert_decomposition(
        cfg,
        weight_bytes=weight_bytes,
        expert_bpe=expert_dtype_bytes,
    )
    num_layers = config_int(cfg, NUM_HIDDEN_LAYERS_KEY)
    n_moe_layers, n_dense_ffn_layers = _derive_moe_layer_counts(cfg, num_layers, num_experts)
    intermediate_size = config_int(cfg, "intermediate_size")
    moe_intermediate_size = config_int(cfg, "moe_intermediate_size")
    if num_experts > 0 and moe_intermediate_size <= 0:
        moe_intermediate_size = intermediate_size
    return ModelMeta(
        weight_bytes=weight_bytes,
        num_layers=num_layers,
        num_kv_heads=_derive_kv_heads(cfg),
        head_dim=_derive_head_dim(cfg),
        weight_dtype_bytes=dtype_bytes,
        expert_weight_dtype_bytes=expert_dtype_bytes,
        active_weight_bytes=active_weight_bytes,
        num_experts=num_experts,
        experts_per_tok=experts_per_tok,
        expert_weight_bytes=total_expert_bytes,
        hidden_size=config_int(cfg, HIDDEN_SIZE_KEY),
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        moe_hidden_size=(_derive_moe_hidden_size(cfg) if num_experts > 0 else 0),
        vocab_size=config_int(cfg, "vocab_size"),
        num_attention_heads=config_int(cfg, "num_attention_heads"),
        moe_layers=n_moe_layers,
        dense_ffn_layers=n_dense_ffn_layers,
    )
