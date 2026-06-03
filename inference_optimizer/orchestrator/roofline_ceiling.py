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
#: ``peak_tflops`` per chip is the **dense** matrix peak (no structured
#: sparsity); LLM inference matmul is not 2:4 structured-sparse, so the
#: sparse-doubled marketing figures must not be used here. Keys are the
#: same precision aliases ``_DTYPE_BYTES`` accepts, so the resolver can
#: share the precision-normalisation logic. Missing key ⇒ 0.0, which
#: degrades T_cmp to "unknown" and lets the roofline fall back to the
#: pure T_mem ceiling (see ``compute_compute_bound_ceiling_tok_per_sec``).
#:
#: Sources (vendor datasheets / official product briefs):
#:   MI300X: 192 GB HBM3,  5.3 TB/s; BF16/FP16 1307.4, FP8 2614.9 TFLOPS
#:   MI325X: 256 GB HBM3e, 6.0 TB/s; same compute as MI300X (CDNA3, 304 CU)
#:   MI355X: 288 GB HBM3e, 8.0 TB/s; CDNA4 ≈ 2× BF16/FP16, new MXFP4 path
_MI300X_PEAK_TFLOPS: dict[str, float] = {
    "bf16": 1307.4, "bfloat16": 1307.4, "fp16": 1307.4, "float16": 1307.4,
    "fp8": 2614.9, "float8_e4m3fn": 2614.9, "float8_e5m2": 2614.9,
    "fp32": 163.4, "float32": 163.4,
}
_MI355X_PEAK_TFLOPS: dict[str, float] = {
    "bf16": 2516.6, "bfloat16": 2516.6, "fp16": 2516.6, "float16": 2516.6,
    "fp8": 5033.2, "float8_e4m3fn": 5033.2, "float8_e5m2": 5033.2,
    "mxfp4": 10066.4, "fp4": 10066.4, "float4": 10066.4,
}
HW_SPECS: dict[str, dict[str, Any]] = {
    "mi300x": {
        "hbm_gb": 192.0, "hbm_bw_gbps": 5300.0,
        "peak_tflops": _MI300X_PEAK_TFLOPS,
    },
    "mi325x": {
        "hbm_gb": 256.0, "hbm_bw_gbps": 6000.0,
        "peak_tflops": _MI300X_PEAK_TFLOPS,
    },
    "mi355x": {
        "hbm_gb": 288.0, "hbm_bw_gbps": 8000.0,
        "peak_tflops": _MI355X_PEAK_TFLOPS,
    },
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
    # mxfp4 / OCP-FP4 are 4-bit (block-scaled); align with HW_SPECS keys.
    "float4": 0.5, "fp4": 0.5, "mxfp4": 0.5,
}


def _resolve_dtype_bytes(tag: str | None) -> float:
    """HF/precision tag → bytes per element; bf16 (2.0) on miss.

    Args:
        tag (str | None): HF ``torch_dtype`` / precision tag.

    Returns:
        float: Bytes per element, defaulting to ``2.0`` on an unknown tag.
    """
    if not tag:
        return 2.0
    return _DTYPE_BYTES.get(str(tag).strip().lower(), 2.0)


def _resolve_peak_tflops(gpu_type: str | None, precision_tag: str | None) -> float:
    """``(gpu, precision)`` → vendor dense peak TFLOPS; 0.0 on miss.

    Returning 0.0 is a deliberate safe-degrade signal: callers treat it
    as "T_cmp unavailable" and fall back to the pure T_mem ceiling
    (``bound_kind="memory"``), which keeps backward compatibility with
    sessions whose precision/GPU pair is not in ``HW_SPECS`` yet.

    Args:
        gpu_type (str | None): GPU type key (e.g. ``"mi300x"``).
        precision_tag (str | None): Precision tag (e.g. ``"bf16"``).

    Returns:
        float: Vendor dense peak TFLOPS, or ``0.0`` when the pair is unknown.
    """
    spec = HW_SPECS.get((gpu_type or "").strip().lower())
    if spec is None:
        return 0.0
    table = spec.get("peak_tflops")
    if not isinstance(table, dict) or not precision_tag:
        return 0.0
    return float(table.get(str(precision_tag).strip().lower(), 0.0))


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
    # MoE expert decomposition (0 for dense). Enables batch-aware expert
    # saturation in the peak formula: at batch B the union of activated
    # experts approaches all of them, so the weight-read term must grow
    # from ``active`` (B=1) toward the full ``weight_bytes`` (high B).
    num_experts: int = 0
    experts_per_tok: int = 0
    expert_weight_bytes: int = 0


def _read_total_size(model_path: Path) -> int | None:
    """Read ``metadata.total_size`` (bytes) from the safetensors index.

    This is byte-exact on-disk weight size and already reflects
    quantization (FP8/FP4/INT4 weights produce a smaller value).

    Args:
        model_path (Path): Local HF model directory.

    Returns:
        int | None: Total weight bytes, or ``None`` when the index file is
            absent or unreadable.
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
    """Read and parse ``config.json`` from a local HF model directory.

    Args:
        model_path (Path): Local HF model directory.

    Returns:
        dict[str, Any] | None: The parsed config, or ``None`` when the file is
            absent or unreadable.
    """
    cfg = model_path / "config.json"
    if not cfg.is_file():
        return None
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _derive_kv_heads(cfg: dict[str, Any]) -> int:
    """GQA-aware: ``num_key_value_heads`` if present, else MHA fallback
    to ``num_attention_heads``.

    Args:
        cfg (dict[str, Any]): Parsed HF ``config.json``.

    Returns:
        int: The number of KV heads, or ``0`` when neither field is present.
    """
    kv = cfg.get("num_key_value_heads")
    if kv is None:
        kv = cfg.get("num_attention_heads")
    return int(kv or 0)


def _derive_head_dim(cfg: dict[str, Any]) -> int:
    """HF either exposes ``head_dim`` directly or implies it via
    ``hidden_size / num_attention_heads``.

    Args:
        cfg (dict[str, Any]): Parsed HF ``config.json``.

    Returns:
        int: The attention head dimension, or ``0`` when it cannot be derived.
    """
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

    Args:
        cfg (dict[str, Any]): Parsed HF ``config.json``.
        weight_bytes (int): Total on-disk weight bytes.
        dtype_bytes (float): Bytes per weight element.

    Returns:
        int: Estimated per-token active weight bytes; equals ``weight_bytes``
            for dense or unknown-geometry models.
    """
    active, _total_expert, _ne, _ept = _compute_expert_decomposition(
        cfg, weight_bytes=weight_bytes, dtype_bytes=dtype_bytes,
    )
    return active


def _compute_expert_decomposition(
    cfg: dict[str, Any],
    *,
    weight_bytes: int,
    dtype_bytes: float,
) -> tuple[int, int, int, int]:
    """MoE decomposition for the batch-aware roofline.

    Returns ``(active_weight_bytes, total_expert_bytes, num_experts,
    experts_per_tok)``. ``active_weight_bytes`` is the B=1 per-token weight
    IO (non-expert + routed fraction of experts). ``total_expert_bytes`` is
    the full routed-expert pool, used by the peak formula to grow the
    weight term from ``active`` (B=1) toward ``weight_bytes`` as the batch
    saturates the activated-expert union.

    Safe degrade (dense / unknown geometry): returns
    ``(weight_bytes, 0, 0, 0)`` so callers fall back to the dense
    ``weight_bytes`` divisor and skip expert saturation.

    HF config key aliases handled here so the MoE detection stays
    architecture-agnostic:
      * ``num_experts`` (Qwen3 MoE, Mixtral, …)
      * ``n_routed_experts`` (DeepSeek V3 / GigaChat — V3-derived models)
      * ``num_local_experts`` (gpt-oss family — GptOssForCausalLM).
    Shared experts (``n_shared_experts``) are intentionally not folded
    in: they are always-active and the safetensors total already lumps
    them into ``non_expert_bytes`` via ``weight_bytes - routed_pool``,
    so ``active = non_expert + (k/n)*routed`` correctly counts them
    as always-active without double-charging the routed top-k pool.

    Args:
        cfg (dict[str, Any]): Parsed HF ``config.json``.
        weight_bytes (int): Total on-disk weight bytes.
        dtype_bytes (float): Bytes per weight element.

    Returns:
        tuple[int, int, int, int]: ``(active_weight_bytes, total_expert_bytes,
            num_experts, experts_per_tok)``; dense/unknown geometry yields
            ``(weight_bytes, 0, 0, 0)``.
    """
    num_experts = int(
        cfg.get("num_experts")
        or cfg.get("n_routed_experts")  # DeepSeek V3 alias
        or cfg.get("num_local_experts")  # gpt-oss alias
        or 0
    )
    experts_per_tok = int(cfg.get("num_experts_per_tok") or 0)
    if num_experts <= 0 or experts_per_tok <= 0:
        return int(weight_bytes), 0, 0, 0
    hidden_size = int(cfg.get("hidden_size") or 0)
    num_layers = int(cfg.get("num_hidden_layers") or 0)
    moe_inter = int(
        cfg.get("moe_intermediate_size")
        or cfg.get("intermediate_size")
        or 0
    )
    if hidden_size <= 0 or num_layers <= 0 or moe_inter <= 0 or dtype_bytes <= 0:
        return int(weight_bytes), 0, 0, 0
    expert_bytes_per_layer = (
        num_experts * 3 * hidden_size * moe_inter * dtype_bytes
    )
    total_expert_bytes = int(num_layers * expert_bytes_per_layer)
    if total_expert_bytes <= 0 or total_expert_bytes >= int(weight_bytes):
        return int(weight_bytes), 0, 0, 0
    non_expert_bytes = int(weight_bytes) - total_expert_bytes
    active_expert_bytes = int(
        (experts_per_tok / num_experts) * total_expert_bytes
    )
    return (
        non_expert_bytes + active_expert_bytes,
        total_expert_bytes,
        num_experts,
        experts_per_tok,
    )


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

    Weight-dtype resolution (priority, first non-empty wins):
      1. ``quantization_config.quant_method`` — fp8 / fp4 / mxfp4
         block-scaled weights (DeepSeek V3, GigaChat, etc.). The HF
         ``torch_dtype`` / ``dtype`` fields under these configs refer
         to *activation* dtype (typically bf16) and would over-count
         the per-param byte size by 2× if used naively.
      2. ``torch_dtype`` — HF-standard.
      3. ``dtype`` — DeepSeek-style alias used by some V3-derived
         configs (still activation dtype, but a closer hint than
         the operator-supplied fallback when ``torch_dtype`` is
         omitted).
      4. ``precision_hint`` from the CLI.

    Args:
        model_path (str | Path): Local HF model directory.
        precision_hint (str): Operator-declared precision, consulted only when
            the config omits a dtype. Defaults to ``""``.

    Returns:
        ModelMeta | None: The extracted metadata, or ``None`` when the config
            or safetensors index is missing/unreadable.
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
    quant_tag = ""
    quant_cfg = cfg.get("quantization_config")
    if isinstance(quant_cfg, dict):
        quant_tag = str(quant_cfg.get("quant_method", "")).strip().lower()
    dtype_bytes = _resolve_dtype_bytes(
        quant_tag
        or cfg.get("torch_dtype")
        or cfg.get("dtype")
        or precision_hint
    )
    active_weight_bytes, total_expert_bytes, num_experts, experts_per_tok = (
        _compute_expert_decomposition(
            cfg, weight_bytes=weight_bytes, dtype_bytes=dtype_bytes,
        )
    )
    return ModelMeta(
        weight_bytes=weight_bytes,
        num_layers=int(cfg.get("num_hidden_layers") or 0),
        num_kv_heads=_derive_kv_heads(cfg),
        head_dim=_derive_head_dim(cfg),
        weight_dtype_bytes=dtype_bytes,
        active_weight_bytes=active_weight_bytes,
        num_experts=num_experts,
        experts_per_tok=experts_per_tok,
        expert_weight_bytes=total_expert_bytes,
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

    Args:
        num_layers (int): Number of transformer layers.
        num_kv_heads (int): Number of KV heads.
        head_dim (int): Attention head dimension.
        kv_dtype_bytes (float): Bytes per KV-cache element.

    Returns:
        int: KV-cache bytes read per generated token across all layers.
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
    num_experts: int = 0,
    experts_per_tok: int = 0,
    expert_weight_bytes: int = 0,
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

    Args:
        gpu_type (str): GPU type key (e.g. ``"mi300x"``).
        num_gpus (int): Tensor-parallel GPU count per replica.
        weight_bytes (int): Total model weight bytes.
        num_layers (int): Number of transformer layers.
        num_kv_heads (int): Number of KV heads.
        head_dim (int): Attention head dimension.
        kv_dtype_bytes (float): Bytes per KV-cache element.
        isl (int): Input sequence length.
        osl (int): Output sequence length.
        concurrency (int): Continuous-batching width.
        active_weight_bytes (int): MoE B=1 per-token active weight bytes.
            Defaults to ``0`` (dense).
        num_experts (int): Total routed experts. Defaults to ``0``.
        experts_per_tok (int): Experts activated per token. Defaults to ``0``.
        expert_weight_bytes (int): Total routed-expert pool bytes. Defaults to
            ``0``.

    Returns:
        float: The decode memory-bound ceiling in tok/s, or ``0.0`` when the
            GPU is unsupported or a divisor degenerates.
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
    # Per-decode-step weight IO. For MoE, the union of experts activated
    # across the ``batch`` tokens saturates toward all of them as batch
    # grows: at B=1 only ``experts_per_tok/num_experts`` fire (==
    # active_weight_bytes); by B≈num_experts/experts_per_tok essentially
    # every expert is read each step (== weight_bytes). Model that with
    # ``activated_fraction = min(1, B * experts_per_tok / num_experts)``;
    # using a constant ``active_weight_bytes`` here would over-amortize the
    # expert weights at high batch and inflate the ceiling. Dense models
    # (num_experts==0) keep the full ``weight_bytes`` read each step.
    if (
        num_experts > 0
        and experts_per_tok > 0
        and expert_weight_bytes > 0
    ):
        non_expert_bytes = max(int(weight_bytes) - int(expert_weight_bytes), 0)
        activated_fraction = min(
            1.0, batch * experts_per_tok / num_experts
        )
        effective_weight = (
            non_expert_bytes + activated_fraction * int(expert_weight_bytes)
        )
    else:
        effective_weight = float(
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


def compute_compute_bound_ceiling_tok_per_sec(
    *,
    gpu_type: str,
    num_gpus: int,
    precision_tag: str,
    active_weight_bytes: int,
    weight_bytes: int,
    weight_dtype_bytes: float,
) -> float:
    """Decode-only compute-bound ceiling for ``output_throughput``.

    Implements the second half of the two-sided roofline:

        T_cmp = (F_peak * G * dtype_bytes) / (2 * active_weight_bytes_B1)

    Derivation: each generated token does ≈ 2 FLOP per active parameter
    (one multiply + one add inside every MFMA), so per-token FLOPs =
    2 * active_params = 2 * active_weight_bytes_B1 / dtype_bytes. Total
    compute throughput = F_peak * G / FLOPs_per_token, which rearranges
    to the formula above. Note the divisor uses ``active_weight_bytes``
    at **B=1** (per-token active parameters) — NOT the batch-saturated
    ``effective_weight`` used by T_mem. They look like the same symbol
    in the textbook roofline but model different physical quantities:
    T_mem amortises HBM weight reads across the batch (so MoE expert
    union saturates with B), while T_cmp counts arithmetic per token
    (every token still routes through only top-k experts, regardless
    of batch). Mixing them up over-amortises compute at high batch and
    under-estimates T_cmp by ~10x on MoE models.

    Returns 0.0 when any of ``gpu_type`` / precision / active_weight_bytes
    / dtype_bytes is missing or zero so the caller treats T_cmp as
    "unavailable" and the roofline degrades to the pure T_mem ceiling.

    Args:
        gpu_type (str): GPU type key (e.g. ``"mi300x"``).
        num_gpus (int): Tensor-parallel GPU count per replica.
        precision_tag (str): Precision tag (e.g. ``"bf16"``).
        active_weight_bytes (int): B=1 per-token active weight bytes.
        weight_bytes (int): Dense total weight bytes (fallback divisor).
        weight_dtype_bytes (float): Bytes per weight element.

    Returns:
        float: The decode compute-bound ceiling in tok/s, or ``0.0`` when any
            required input is missing or zero.
    """
    peak_tflops = _resolve_peak_tflops(gpu_type, precision_tag)
    if peak_tflops <= 0 or weight_dtype_bytes <= 0:
        return 0.0
    # ``active_weight_bytes`` is the B=1 per-token figure populated by
    # ``load_model_meta`` (dense models: == weight_bytes; MoE: routed-k
    # share of expert pool + non-expert blocks). Fall back to the dense
    # weight_bytes only when active is missing/0 — never use a batch-
    # saturated effective weight here (see docstring).
    active_b1 = (
        int(active_weight_bytes)
        if active_weight_bytes and active_weight_bytes > 0
        else int(weight_bytes)
    )
    if active_b1 <= 0:
        return 0.0
    peak_flops_total = peak_tflops * 1e12 * max(num_gpus, 1)
    flops_per_token = 2.0 * active_b1 / float(weight_dtype_bytes)
    if flops_per_token <= 0:
        return 0.0
    return peak_flops_total / flops_per_token


def _read_baseline_yaml_conc(state: Any) -> int:
    """Read ``benchmark.envs.CONC`` from the materialized baseline yaml.

    This is the ground-truth concurrency the Magpie subprocess actually
    ran with. Returns ``0`` when the file / field is unreadable.

    Args:
        state (Any): The SharedState-like object carrying ``last_baseline``.

    Returns:
        int: The parsed concurrency, or ``0`` when unreadable/non-positive.
    """
    last_bl = getattr(state, "last_baseline", None) or {}
    if not isinstance(last_bl, dict):
        return 0
    extras = last_bl.get("extras") or {}
    cfg_path = extras.get("materialized_config") if isinstance(extras, dict) else ""
    if not cfg_path:
        return 0
    try:
        import yaml as _yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
        envs = (cfg.get("benchmark") or {}).get("envs") or {}
        raw = envs.get("CONC")
        if raw is not None:
            parsed = int(raw)
            if parsed > 0:
                return parsed
    except Exception:  # noqa: BLE001 — yaml IO / parse is best-effort
        pass
    return 0


def _resolve_effective_concurrency(state: Any) -> int:
    """Resolve the concurrency the actual benchmark ran with.

    Priority (authoritative first):

      1. ``state.last_baseline.extras.materialized_config`` →
         ``benchmark.envs.CONC`` in the on-disk baseline yaml. This is the
         ground-truth value the Magpie subprocess actually ran with, so it
         is authoritative whenever the file is readable. It is checked
         FIRST because ``state.conc`` has a default-vs-stale pitfall: when
         an operator launches without ``--conc`` / ``$CONC``, ``state.conc``
         stays at the SharedState dataclass default (typically 8) while the
         yaml carries the real ``CONC: 64`` — reading ``state.conc`` first
         (the old behaviour) produced an 8x under-counted ceiling.
      2. ``state.conc`` (SharedState field) when the yaml is unavailable.
      3. ``1`` as the ultimate fallback so the formula divisor never
         degenerates (matches the single-stream interpretation).

    Returns ``int`` >= 1.

    Args:
        state (Any): The SharedState-like object carrying baseline / conc.

    Returns:
        int: The effective concurrency, always ``>= 1``.
    """
    yaml_conc = _read_baseline_yaml_conc(state)
    if yaml_conc > 0:
        return yaml_conc
    conc = int(getattr(state, "conc", 0) or 0)
    if conc > 0:
        return conc
    return 1


@dataclass(frozen=True)
class RooflineBreakdown:
    """Two-sided decode roofline ceiling.

    Captures the result of ``T_peak = min(T_mem, T_cmp)`` together with
    the side that dominated so reports can label which bound is active.

    ``bound_kind`` values:
      * ``"memory"``  — T_mem ≤ T_cmp (or T_cmp unavailable/degenerate).
      * ``"compute"`` — T_cmp < T_mem (compute-bound; rare in decode but
        possible at very small B with large F_peak relative to BW).
      * ``"unknown"`` — both ceilings degenerate (e.g. unsupported gpu,
        missing model HF config); ``peak_tok_per_sec == 0``.
    """
    mem_tok_per_sec: float
    cmp_tok_per_sec: float
    peak_tok_per_sec: float
    bound_kind: str


_EMPTY_BREAKDOWN = RooflineBreakdown(0.0, 0.0, 0.0, "unknown")


def compute_roofline_breakdown_from_state(state: Any) -> RooflineBreakdown:
    """Compute T_mem + T_cmp + min(T_mem, T_cmp) + bound_kind in one shot.

    This is the primary entry point introduced by the two-sided roofline
    formula change. ``compute_peak_from_state`` is kept as a thin scalar
    wrapper for callers that only need ``T_peak``.

    Safe degrade: never raises; returns ``_EMPTY_BREAKDOWN`` on any
    missing-field path so the caller can stamp the result into history
    snapshots unconditionally.

    Args:
        state (Any): The SharedState-like object carrying model path, GPU,
            precision, tp, isl, osl and concurrency fields.

    Returns:
        RooflineBreakdown: The T_mem / T_cmp / peak / bound-kind breakdown;
            ``_EMPTY_BREAKDOWN`` when inputs are missing.
    """
    meta = load_model_meta(
        getattr(state, "model_path", ""),
        precision_hint=str(getattr(state, "precision", "") or ""),
    )
    if meta is None:
        return _EMPTY_BREAKDOWN
    gpu_type = str(getattr(state, "gpu_type", "") or "")
    num_gpus = int(getattr(state, "tp", 0) or 0)
    concurrency = _resolve_effective_concurrency(state)
    precision_tag = (
        str(getattr(state, "precision", "") or "")
        or "bf16"  # mirror _resolve_dtype_bytes default
    )
    mem = compute_theoretical_peak_output_tok_per_sec(
        gpu_type=gpu_type,
        num_gpus=num_gpus,
        weight_bytes=meta.weight_bytes,
        active_weight_bytes=meta.active_weight_bytes,
        num_experts=meta.num_experts,
        experts_per_tok=meta.experts_per_tok,
        expert_weight_bytes=meta.expert_weight_bytes,
        num_layers=meta.num_layers,
        num_kv_heads=meta.num_kv_heads,
        head_dim=meta.head_dim,
        kv_dtype_bytes=meta.weight_dtype_bytes,
        isl=int(getattr(state, "isl", 0) or 0),
        osl=int(getattr(state, "osl", 0) or 0),
        concurrency=concurrency,
    )
    cmp = compute_compute_bound_ceiling_tok_per_sec(
        gpu_type=gpu_type,
        num_gpus=num_gpus,
        precision_tag=precision_tag,
        active_weight_bytes=meta.active_weight_bytes,
        weight_bytes=meta.weight_bytes,
        weight_dtype_bytes=meta.weight_dtype_bytes,
    )
    if mem <= 0 and cmp <= 0:
        return _EMPTY_BREAKDOWN
    if cmp <= 0:
        # T_cmp unknown (precision not in HW_SPECS); degrade to T_mem
        # ceiling and mark memory-bound (matches pre-PR behaviour).
        return RooflineBreakdown(mem, 0.0, mem, "memory")
    if mem <= 0:
        return RooflineBreakdown(0.0, cmp, cmp, "compute")
    if cmp < mem:
        return RooflineBreakdown(mem, cmp, cmp, "compute")
    return RooflineBreakdown(mem, cmp, mem, "memory")


def compute_peak_from_state(state: Any) -> float:
    """Convenience scalar wrapper for ``T_peak`` only.

    Kept for backward compatibility (shared_state, tests, report
    docstring references). New code should call
    ``compute_roofline_breakdown_from_state`` directly to get T_mem /
    T_cmp / bound_kind alongside the min.

    Args:
        state (Any): The SharedState-like object.

    Returns:
        float: The ``T_peak`` ceiling in tok/s (``0.0`` when unavailable).
    """
    return compute_roofline_breakdown_from_state(state).peak_tok_per_sec
