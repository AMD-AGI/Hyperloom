"""Theoretical peak ``output_throughput`` ceiling (decode memory roofline).

Hardware ceiling for the ``output_throughput`` (tok/s) field that
``baseline`` / ``profile`` / ``sweep`` executors already write into the
result dict and that ``SharedState._KEY_METRIC_MAP`` already pins as the
canonical comparable metric.

Formula (decode-only, memory-bound; same form across
arXiv 2402.16363 §LLM Inference Roofline, arXiv 2602.11506 RooflineBench,
and the Berkeley Williams roofline applied to LLM serving):

    peak_output_tok_per_sec
      = (HBM_BW_per_gpu × num_gpus)
        / (weight_bytes / batch + kv_bytes_per_token × kv_seq_len)

We deliberately do NOT model prefill: ``output_throughput`` only counts
decode tokens, so a decode-only ceiling stays language-faithful with
the implemented benchmark metric (see ``shared_state._KEY_METRIC_MAP``).
The ``batch = max(concurrency, 1)`` term captures shared-weight reuse
under continuous batching; it is the B1 variant the operator picked
over plain single-stream ceiling.

Outputs are an ``upper bound``: real serving never hits this because
of comm overhead, kernel efficiency < 100% of peak, and KV-cache
fragmentation. Down-stream renderers should label the value
accordingly (e.g. ``Theoretical Peak — single-stream-reuse ceiling``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Hardware specs (AMD MI300/MI355 line only).
# ---------------------------------------------------------------------------
#: GPU per-chip peak specs. ``hbm_bw_gbps`` is the vendor-quoted HBM
#: bandwidth (theoretical peak across all stacks at advertised
#: clocks); ``hbm_gb`` is total HBM capacity per chip. Keys must match
#: ``SharedState.gpu_type`` vocabulary (lowercase, no whitespace) —
#: see ``inference_optimizer/cli.py`` ``_autodetect_gpu_type`` for the
#: canonical list.
#:
#: CAVEAT — vendor-quoted peak vs sustained achievable. The numbers
#: below are the **upper-bound** marketing figures. In practice, the
#: bandwidth a single decode kernel can sustain over a steady-state
#: workload is typically **70–90% of the quoted peak** because of:
#:   * memory controller scheduling overhead at small batch sizes
#:   * cache-line / read-burst granularity that wastes a few percent
#:     when the access pattern is not fully coalesced
#:   * thermal / power throttling under sustained load
#:   * (multi-GPU only) XGMI / Infinity Fabric arbitration cost on
#:     all-gather / all-reduce when TP > 1
#: Using the vendor peak here is deliberate: ``compute_theoretical_
#: peak_output_tok_per_sec`` returns a **ceiling** that real
#: ``output_throughput`` always stays under. Dashboards label the
#: result accordingly (``Within roofline %`` = measured / ceiling;
#: 100% would mean "matching the theoretical peak", which is
#: physically unreachable but a useful single anchor for baseline /
#: optimized comparison).
#:
#: Sources (vendor datasheets / official product briefs):
#:   MI300X: 192 GB HBM3,  5.3 TB/s
#:   MI325X: 256 GB HBM3e, 6.0 TB/s
#:   MI355X: 288 GB HBM3e, 8.0 TB/s
HW_SPECS: dict[str, dict[str, float]] = {
    "mi300x": {"hbm_gb": 192.0, "hbm_bw_gbps": 5300.0},
    "mi325x": {"hbm_gb": 256.0, "hbm_bw_gbps": 6000.0},
    "mi355x": {"hbm_gb": 288.0, "hbm_bw_gbps": 8000.0},
}


# ---------------------------------------------------------------------------
# Dtype bytes lookup.
# ---------------------------------------------------------------------------
#: HF ``torch_dtype`` / precision tag → bytes per element. Used only as
#: a fallback when ``model.safetensors.index.json`` is not available:
#: when the index file is present, ``weight_bytes = total_size`` already
#: reflects on-disk quantization byte-exact, so dtype is sidestepped.
_DTYPE_BYTES: dict[str, float] = {
    "float32": 4.0, "fp32": 4.0,
    "bfloat16": 2.0, "bf16": 2.0,
    "float16": 2.0, "fp16": 2.0,
    "float8_e4m3fn": 1.0, "float8_e5m2": 1.0, "fp8": 1.0,
    "float4": 0.5, "fp4": 0.5,
}


def _resolve_dtype_bytes(tag: str | None) -> float:
    """HF/precision tag → bytes per element; bf16 (2.0) on miss."""
    if not tag:
        return 2.0
    return _DTYPE_BYTES.get(str(tag).strip().lower(), 2.0)


# ---------------------------------------------------------------------------
# Model metadata extraction.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelMeta:
    """HF subset needed for the decode roofline ceiling.

    ``active_weight_bytes`` (MoE-aware, optional): bytes of model weights
    actually fetched from HBM per generated token. For dense models this
    equals ``weight_bytes``; for Mixture-of-Experts models (e.g.
    Qwen3-30B-A3B, DeepSeek-V3, Mixtral) only a routed subset of expert
    weights fires per token so the divisor of the roofline must shrink.
    Defaults to ``0`` for backward-compat with callers that construct
    ``ModelMeta`` directly without MoE knowledge; the compute helpers
    treat ``0`` as "fall back to weight_bytes" so dense behaviour is
    preserved. ``load_model_meta`` always sets a concrete value.
    """

    weight_bytes: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    weight_dtype_bytes: float
    active_weight_bytes: int = 0


def _read_total_size(model_path: Path) -> int | None:
    """Read ``metadata.total_size`` (bytes) from the safetensors index.

    This is byte-exact on-disk weight size and already reflects
    quantization (FP8/FP4/INT4 weights produce a smaller value).
    """
    idx = model_path / "model.safetensors.index.json"
    if not idx.is_file():
        return None
    try:
        meta = json.loads(idx.read_text(encoding="utf-8")).get("metadata") or {}
        size = meta.get("total_size")
        return int(size) if size else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _read_hf_config(model_path: Path) -> dict[str, Any] | None:
    cfg = model_path / "config.json"
    if not cfg.is_file():
        return None
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _derive_kv_heads(cfg: dict[str, Any]) -> int:
    """GQA-aware: ``num_key_value_heads`` if present, else MHA fallback
    to ``num_attention_heads``."""
    kv = cfg.get("num_key_value_heads")
    if kv is None:
        kv = cfg.get("num_attention_heads")
    return int(kv or 0)


def _derive_head_dim(cfg: dict[str, Any]) -> int:
    """HF either exposes ``head_dim`` directly or implies it via
    ``hidden_size / num_attention_heads``."""
    head_dim = cfg.get("head_dim")
    if head_dim:
        return int(head_dim)
    hidden = cfg.get("hidden_size")
    heads = cfg.get("num_attention_heads")
    if hidden and heads:
        return int(hidden) // int(heads)
    return 0


def _compute_active_weight_bytes(
    cfg: dict[str, Any],
    *,
    weight_bytes: int,
    dtype_bytes: float,
) -> int:
    """MoE-aware estimate of bytes of weight actually fetched per token.

    Background: a vanilla Mixture-of-Experts decoder routes each token
    to ``num_experts_per_tok`` of ``num_experts`` routed experts; all
    other weights (attention / norms / embeddings / router / optional
    shared experts) run on every token. Using the full safetensors
    ``total_size`` as the divisor of the memory roofline therefore
    over-counts the per-token weight IO by ~10× on Qwen3-30B-A3B,
    DeepSeek-V3, Mixtral, etc., and drives the ceiling below the
    measured throughput (``within_roofline_pct > 100%``).

    Geometry-based estimate, derived from HF config:

      expert_bytes_per_layer = num_experts
                              × 3 × hidden_size × moe_intermediate_size
                              × dtype_bytes
      total_expert_bytes     = num_hidden_layers × expert_bytes_per_layer
      non_expert_bytes       = weight_bytes − total_expert_bytes
      active_expert_bytes    = (num_experts_per_tok / num_experts)
                              × total_expert_bytes
      active_weight_bytes    = non_expert_bytes + active_expert_bytes

    The ``3 ×`` factor matches the gated-MLP convention used by Qwen3
    / Llama-style experts (gate + up + down projections). DeepSeek's
    optional shared experts are conservatively treated as part of
    ``non_expert_bytes`` via the subtraction: any weight not accounted
    for by the routed experts (including shared experts) stays in the
    always-active pool, which is the safe direction (slightly larger
    divisor → slightly lower ceiling, never inflates).

    Safe degrade — returns ``int(weight_bytes)`` unchanged when:
      * any MoE field is missing or non-positive  → treat as dense
      * computed ``total_expert_bytes >= weight_bytes`` → config /
        safetensors mismatch (quantized experts, accounting drift);
        stay safe and keep the dense-equivalent divisor
    """
    num_experts = int(cfg.get("num_experts") or 0)
    experts_per_tok = int(cfg.get("num_experts_per_tok") or 0)
    if num_experts <= 0 or experts_per_tok <= 0:
        return int(weight_bytes)
    hidden_size = int(cfg.get("hidden_size") or 0)
    num_layers = int(cfg.get("num_hidden_layers") or 0)
    moe_inter = int(
        cfg.get("moe_intermediate_size")
        or cfg.get("intermediate_size")
        or 0
    )
    if hidden_size <= 0 or num_layers <= 0 or moe_inter <= 0 or dtype_bytes <= 0:
        return int(weight_bytes)
    expert_bytes_per_layer = (
        num_experts * 3 * hidden_size * moe_inter * dtype_bytes
    )
    total_expert_bytes = int(num_layers * expert_bytes_per_layer)
    if total_expert_bytes <= 0 or total_expert_bytes >= int(weight_bytes):
        return int(weight_bytes)
    non_expert_bytes = int(weight_bytes) - total_expert_bytes
    active_expert_bytes = int(
        (experts_per_tok / num_experts) * total_expert_bytes
    )
    return non_expert_bytes + active_expert_bytes


def load_model_meta(
    model_path: str | Path,
    *,
    precision_hint: str = "",
) -> ModelMeta | None:
    """Read ``weight_bytes`` + KV-cache shape from a local HF model dir.

    Returns ``None`` when ``config.json`` or ``model.safetensors.index.json``
    is unreadable — caller falls back to a ``0.0`` ceiling so the
    dashboard renders ``"—"`` instead of a misleading value.

    ``precision_hint`` is the operator-declared precision (passed via
    ``state.precision``); consulted only when ``config.json`` omits
    ``torch_dtype``.
    """
    if not model_path:
        return None
    p = Path(model_path).expanduser()
    if not p.is_dir():
        return None
    cfg = _read_hf_config(p)
    if cfg is None:
        return None
    weight_bytes = _read_total_size(p)
    if not weight_bytes:
        return None
    dtype_bytes = _resolve_dtype_bytes(
        cfg.get("torch_dtype") or precision_hint
    )
    active_weight_bytes = _compute_active_weight_bytes(
        cfg, weight_bytes=weight_bytes, dtype_bytes=dtype_bytes,
    )
    return ModelMeta(
        weight_bytes=weight_bytes,
        num_layers=int(cfg.get("num_hidden_layers") or 0),
        num_kv_heads=_derive_kv_heads(cfg),
        head_dim=_derive_head_dim(cfg),
        weight_dtype_bytes=dtype_bytes,
        active_weight_bytes=active_weight_bytes,
    )


# ---------------------------------------------------------------------------
# Peak throughput formula.
# ---------------------------------------------------------------------------
def compute_kv_bytes_per_token(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype_bytes: float,
) -> int:
    """KV cache footprint per generated token, summed over all layers.

    ``2`` factors covers K + V tensors. Caller multiplies by
    ``kv_seq_len`` to get the per-token HBM read volume.
    """
    return int(2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes)


def compute_theoretical_peak_output_tok_per_sec(
    *,
    gpu_type: str,
    num_gpus: int,
    weight_bytes: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype_bytes: float,
    isl: int,
    osl: int,
    concurrency: int,
    active_weight_bytes: int = 0,
) -> float:
    """Decode-only memory-bound ceiling for ``output_throughput``.

    Returns ``0.0`` when ``gpu_type`` is not in :data:`HW_SPECS` or any
    divisor degenerates so the caller can render a uniform ``"—"``
    placeholder. Never raises.

    Inputs map to ``SharedState`` fields:
      ``gpu_type``           -> ``state.gpu_type``
      ``num_gpus``           -> ``state.tp`` (tensor-parallel per replica)
      ``weight_bytes``       -> ``ModelMeta.weight_bytes``
      ``active_weight_bytes``-> ``ModelMeta.active_weight_bytes`` (MoE)
      ``concurrency``        -> ``state.conc`` (continuous-batching width)
      ``isl`` / ``osl``      -> ``state.isl`` / ``state.osl``

    ``active_weight_bytes`` (optional, defaults to 0) shrinks the
    per-token weight IO term for MoE models: the divisor uses
    ``active_weight_bytes`` instead of ``weight_bytes`` when the former
    is positive. Dense models (or callers without MoE knowledge) leave
    it at 0 and get the original ``weight_bytes`` divisor — preserves
    backward compatibility.
    """
    spec = HW_SPECS.get((gpu_type or "").strip().lower())
    if spec is None:
        return 0.0
    bw_total_bytes_per_sec = spec["hbm_bw_gbps"] * 1e9 * max(num_gpus, 1)
    batch = max(concurrency, 1)
    kv_bytes = compute_kv_bytes_per_token(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_dtype_bytes=kv_dtype_bytes,
    )
    # Average KV-cache length during decode: read isl + already-decoded
    # tokens, averaged over the osl decode steps.
    kv_seq_len = max(int(isl) + int(osl) // 2, 1)
    effective_weight = (
        int(active_weight_bytes)
        if active_weight_bytes and active_weight_bytes > 0
        else int(weight_bytes)
    )
    bytes_per_token_total = (
        effective_weight / batch + kv_bytes * kv_seq_len
    )
    if bytes_per_token_total <= 0:
        return 0.0
    return bw_total_bytes_per_sec / bytes_per_token_total


def compute_peak_from_state(state: Any) -> float:
    """Convenience wrapper reading directly from a ``SharedState``-like object.

    Returns ``0.0`` when ``model_path`` / ``gpu_type`` / any required
    field is unset or unreadable. Never raises so the caller can stamp
    the value into history entries unconditionally.
    """
    meta = load_model_meta(
        getattr(state, "model_path", ""),
        precision_hint=str(getattr(state, "precision", "") or ""),
    )
    if meta is None:
        return 0.0
    return compute_theoretical_peak_output_tok_per_sec(
        gpu_type=str(getattr(state, "gpu_type", "") or ""),
        num_gpus=int(getattr(state, "tp", 0) or 0),
        weight_bytes=meta.weight_bytes,
        active_weight_bytes=meta.active_weight_bytes,
        num_layers=meta.num_layers,
        num_kv_heads=meta.num_kv_heads,
        head_dim=meta.head_dim,
        kv_dtype_bytes=meta.weight_dtype_bytes,
        isl=int(getattr(state, "isl", 0) or 0),
        osl=int(getattr(state, "osl", 0) or 0),
        concurrency=int(getattr(state, "conc", 0) or 0),
    )
