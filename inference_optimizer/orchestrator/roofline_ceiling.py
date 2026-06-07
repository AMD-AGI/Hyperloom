# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Theoretical peak ``output_throughput`` ceiling (decode memory roofline).

Formula (decode-only, memory-bound)::

    peak_output_tok_per_sec
      = (HBM_BW_per_gpu × num_gpus)
        / (weight_bytes / batch + kv_bytes_per_token × kv_seq_len)

Prefill is not modelled; ``batch = max(concurrency, 1)``. Outputs are an upper bound.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Hardware specs (AMD MI300/MI355 line only).
#: GPU per-chip peak specs (keys match ``SharedState.gpu_type``, lowercase). ``hbm_bw_gbps`` is vendor peak (strict ceiling); ``peak_tflops`` is DENSE peak (missing key ⇒ 0.0 falls back to T_mem). Vendor datasheets.
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


# Dtype bytes lookup.
#: HF ``torch_dtype`` / precision tag → bytes per element (fallback when safetensors index absent).
_DTYPE_BYTES: dict[str, float] = {
    "float32": 4.0, "fp32": 4.0,
    "bfloat16": 2.0, "bf16": 2.0,
    "float16": 2.0, "fp16": 2.0,
    "float8_e4m3fn": 1.0, "float8_e5m2": 1.0, "fp8": 1.0,
    # mxfp4 / OCP-FP4 are 4-bit (block-scaled); align with HW_SPECS keys.
    "float4": 0.5, "fp4": 0.5, "mxfp4": 0.5,
}


def _resolve_dtype_bytes(tag: str | None) -> float:
    """HF/precision tag → bytes per element; bf16 (2.0) on miss."""
    if not tag:
        return 2.0
    return _DTYPE_BYTES.get(str(tag).strip().lower(), 2.0)


def _resolve_peak_tflops(gpu_type: str | None, precision_tag: str | None) -> float:
    """``(gpu, precision)`` → vendor dense peak TFLOPS; 0.0 on miss (safe-degrade signal → T_cmp unavailable, fall back to T_mem)."""
    spec = HW_SPECS.get((gpu_type or "").strip().lower())
    if spec is None:
        return 0.0
    table = spec.get("peak_tflops")
    if not isinstance(table, dict) or not precision_tag:
        return 0.0
    return float(table.get(str(precision_tag).strip().lower(), 0.0))


# Model metadata extraction.
@dataclass(frozen=True)
class ModelMeta:
    """HF subset needed for the decode roofline ceiling. ``active_weight_bytes`` is per-token MoE weight IO (0 ⇒ fall back to ``weight_bytes``)."""

    weight_bytes: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    weight_dtype_bytes: float
    active_weight_bytes: int = 0
    # MoE expert decomposition (0 for dense); enables batch-aware expert saturation in the peak formula.
    num_experts: int = 0
    experts_per_tok: int = 0
    expert_weight_bytes: int = 0


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
            pass
    return (
        _sum_weight_file_sizes(model_path, "*.safetensors")
        or _sum_weight_file_sizes(model_path, "*.bin")
    )


def _sum_weight_file_sizes(model_path: Path, pattern: str) -> int | None:
    """Fallback weight size from local weight shards matching ``pattern``."""
    try:
        total = sum(
            p.stat().st_size
            for p in model_path.glob(pattern)
            if p.is_file()
        )
    except OSError:
        return None
    return int(total) if total > 0 else None


def _read_hf_config(model_path: Path) -> dict[str, Any] | None:
    cfg = model_path / "config.json"
    if not cfg.is_file():
        return None
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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
    """MoE-aware estimate of weight bytes fetched per token (geometry-based, avoids the ~10× over-count from the safetensors total; safe-degrades to ``weight_bytes``)."""
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
    """MoE decomposition for the batch-aware roofline; returns ``(active_weight_bytes, total_expert_bytes, num_experts, experts_per_tok)``. Safe-degrades to ``(weight_bytes, 0, 0, 0)``. Handles num_experts / n_routed_experts / num_local_experts aliases."""
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
    """Read ``weight_bytes`` + KV-cache shape from a local HF model dir (``None`` when unreadable). Weight-dtype priority: quantization_config.quant_method > torch_dtype > dtype > precision_hint."""
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


# Peak throughput formula.
def compute_kv_bytes_per_token(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype_bytes: float,
) -> int:
    """KV cache footprint per generated token, summed over all layers (the ``2`` covers K + V)."""
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
    """Decode-only memory-bound ceiling for ``output_throughput`` (returns 0.0, never raises, on unknown gpu_type / degenerate divisor). ``active_weight_bytes`` shrinks per-token IO for MoE."""
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
    # Average KV-cache length during decode (isl + half of osl).
    kv_seq_len = max(int(isl) + int(osl) // 2, 1)
    # Per-decode-step weight IO; MoE activated-expert union saturates with batch via ``activated_fraction = min(1, B * experts_per_tok / num_experts)``.
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

        T_cmp = (F_peak * G * dtype_bytes) / (2 * active_weight_bytes_B1)

    Divisor uses ``active_weight_bytes`` at B=1 (NOT batch-saturated). Returns 0.0 on missing input (degrade to T_mem).
    """
    peak_tflops = _resolve_peak_tflops(gpu_type, precision_tag)
    if peak_tflops <= 0 or weight_dtype_bytes <= 0:
        return 0.0
    # B=1 per-token figure; fall back to dense weight_bytes only when active is missing/0 (never a batch-saturated weight here).
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
    """Read ``benchmark.envs.CONC`` from the materialized baseline yaml (ground-truth concurrency; ``0`` when unreadable)."""
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
    """Resolve the concurrency the actual benchmark ran with (returns int >= 1; on-disk baseline yaml CONC wins, since ``state.conc`` can stay stale and under-count the ceiling 8x)."""
    yaml_conc = _read_baseline_yaml_conc(state)
    if yaml_conc > 0:
        return yaml_conc
    conc = int(getattr(state, "conc", 0) or 0)
    if conc > 0:
        return conc
    return 1


@dataclass(frozen=True)
class RooflineBreakdown:
    """Two-sided decode roofline ceiling: ``T_peak = min(T_mem, T_cmp)``. ``bound_kind`` ∈ {memory, compute, unknown}."""
    mem_tok_per_sec: float
    cmp_tok_per_sec: float
    peak_tok_per_sec: float
    bound_kind: str


_EMPTY_BREAKDOWN = RooflineBreakdown(0.0, 0.0, 0.0, "unknown")


def compute_roofline_breakdown_from_state(state: Any) -> RooflineBreakdown:
    """Compute T_mem + T_cmp + min + bound_kind in one shot (never raises; ``_EMPTY_BREAKDOWN`` on missing fields)."""
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
        # T_cmp unknown (precision not in HW_SPECS); degrade to T_mem ceiling.
        return RooflineBreakdown(mem, 0.0, mem, "memory")
    if mem <= 0:
        return RooflineBreakdown(0.0, cmp, cmp, "compute")
    if cmp < mem:
        return RooflineBreakdown(mem, cmp, cmp, "compute")
    return RooflineBreakdown(mem, cmp, mem, "memory")


def compute_peak_from_state(state: Any) -> float:
    """Convenience scalar wrapper for ``T_peak`` only (kept for backward compat; prefer ``compute_roofline_breakdown_from_state``)."""
    return compute_roofline_breakdown_from_state(state).peak_tok_per_sec
