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
import os
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
    "mi308x": {
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
    """HF/precision tag → bytes per element; bf16 (2.0) on miss.

    Args:
        tag (str | None): HF ``torch_dtype`` / precision tag.

    Returns:
        float: Bytes per element, defaulting to ``2.0`` on an unknown tag.
    """
    if not tag:
        return 2.0
    return _DTYPE_BYTES.get(str(tag).strip().lower(), 2.0)


# ---------------------------------------------------------------------------
# Runtime dtype / quantization resolution.
# ---------------------------------------------------------------------------
#: Map a server-arg ``--quantization`` value to weight bytes-per-element.
#: Only weight-quantization methods are listed; activation/KV dtype is
#: tracked separately (``runtime_activation_dtype``) and stays >= bf16.
_QUANT_WEIGHT_BYTES: dict[str, float] = {
    "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0,
    "fp4": 0.5, "mxfp4": 0.5, "nvfp4": 0.5,
    "int8": 1.0, "w8a8_int8": 1.0,
    "int4": 0.5, "awq": 0.5, "gptq": 0.5,
}


def _parse_server_arg(args: str, flag: str) -> str:
    """Return the value following ``--flag`` (space or ``=`` form), else ``""``.

    Tolerant of both ``--quantization fp8`` and ``--quantization=fp8``.
    When the flag appears multiple times, the last value wins so an
    optimized overlay can override the baseline args.
    """
    if not args:
        return ""
    toks = str(args).replace("=", " ").split()
    value = ""
    for i, tok in enumerate(toks):
        if tok == flag and i + 1 < len(toks):
            value = toks[i + 1].strip()
    return value


#: Magpie ``benchmark.envs`` keys that carry the runtime server args, one
#: per framework. The baseline yaml only ever sets the one matching its
#: framework, so reading all three and concatenating is safe.
_RUNTIME_SERVER_ARG_ENV_KEYS = (
    "EXTRA_SGLANG_ARGS", "EXTRA_VLLM_ARGS", "EXTRA_ATOM_ARGS",
)


@dataclass(frozen=True)
class RuntimeWorkload:
    """Runtime workload config used by the roofline ceiling.

    Baseline materialized yaml is the base source of truth; optimized
    current_best contributes only overlay server args.
    """

    model_path: str
    gpu_type: str
    precision: str
    framework: str
    tp: int
    concurrency: int
    isl: int
    osl: int
    server_args: str


def _read_baseline_yaml_benchmark(state: Any) -> dict[str, Any]:
    """Read ``benchmark`` from the materialized baseline yaml."""
    last_bl = getattr(state, "last_baseline", None) or {}
    if not isinstance(last_bl, dict):
        return {}
    extras = last_bl.get("extras") or {}
    cfg_path = extras.get("materialized_config") if isinstance(extras, dict) else ""
    if not cfg_path:
        return {}
    try:
        import yaml as _yaml  # type: ignore[reportMissingModuleSource]
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
        benchmark = cfg.get("benchmark") or {}
        return benchmark if isinstance(benchmark, dict) else {}
    except Exception:  # noqa: BLE001 — yaml IO / parse is best-effort
        return {}


def _benchmark_envs(benchmark: dict[str, Any]) -> dict[str, Any]:
    envs = benchmark.get("envs") or {}
    return envs if isinstance(envs, dict) else {}


def _env_int(envs: dict[str, Any], key: str) -> int:
    raw = envs.get(key)
    if raw is None:
        return 0
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else 0
    except (TypeError, ValueError):
        return 0


def _server_args_from_envs(envs: dict[str, Any]) -> str:
    parts = [
        str(envs[k]).strip()
        for k in _RUNTIME_SERVER_ARG_ENV_KEYS
        if isinstance(envs.get(k), str) and envs[k].strip()
    ]
    return " ".join(parts)


def _read_baseline_yaml_server_args(state: Any) -> str:
    """Read the runtime server args from the materialized baseline yaml.

    The baseline ``current_best`` snapshot carries no ``extra_server_args``
    (the flags live only in ``benchmark.envs.EXTRA_*_ARGS`` of the on-disk
    yaml), so a baseline-only run with ``--quantization fp8`` in the yaml
    would otherwise be invisible to dtype resolution. Returns ``""`` when
    the file / fields are unreadable.
    """
    return _server_args_from_envs(
        _benchmark_envs(_read_baseline_yaml_benchmark(state))
    )


def _server_args_env_override(entry: Any) -> str:
    """Return framework server args pinned via ``extra_envs``."""
    if not isinstance(entry, dict):
        return ""
    envs = entry.get("extra_envs") or {}
    if isinstance(envs, dict):
        return _server_args_from_envs(envs)
    return ""


def _server_args_payload(entry: Any) -> str:
    """Return framework-neutral overlay server args from an entry."""
    if not isinstance(entry, dict):
        return ""
    for key in ("candidate_extra_server_args", "extra_server_args", "extra_args"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _server_args_from(entry: Any) -> str:
    """Extract the final server-args string from one state entry."""
    return _server_args_env_override(entry) or _server_args_payload(entry)


def _achieved_arm_source(state: Any) -> str:
    """Which arm the roofline ``achieved`` throughput comes from.

    Mirrors the snapshot writer (``current_best.tput > 0`` ⇒ optimized,
    else baseline) so the ceiling's dtype is resolved from the SAME run
    its measured throughput came from. Returns ``"current_best"`` or
    ``"baseline"``.
    """
    cb = getattr(state, "current_best", None)
    if isinstance(cb, dict):
        t = cb.get("tput")
        if isinstance(t, (int, float)) and t > 0:
            return "current_best"
    return "baseline"


def _collect_runtime_server_args(state: Any) -> str:
    """Server args for the arm the ceiling is actually compared against.

    Selects a SINGLE source aligned with ``achieved`` (see
    ``_achieved_arm_source``). Baseline materialized yaml is the base
    config; current_best contributes only overlay server args when the
    measured throughput comes from the optimized arm.
    """
    base_args = (
        _read_baseline_yaml_server_args(state)
        or _server_args_from(getattr(state, "last_baseline", None))
    )
    if _achieved_arm_source(state) == "current_best":
        current_best = getattr(state, "current_best", None)
        override = _server_args_env_override(current_best)
        if override:
            return override
        overlay = _server_args_payload(current_best)
        return " ".join(p for p in (base_args, overlay) if p)
    return base_args


def _runtime_gpu_type(state: Any, benchmark: dict[str, Any]) -> str:
    """Resolve real hardware for roofline; runner_type is only script routing."""
    return str(
        getattr(state, "gpu_type", "")
        or os.environ.get("TARGET_GPU_TYPE", "")
        or benchmark.get("runner_type")
        or ""
    )


def resolve_runtime_workload(state: Any) -> RuntimeWorkload:
    """Resolve runtime workload fields from baseline yaml plus overlay args."""
    benchmark = _read_baseline_yaml_benchmark(state)
    envs = _benchmark_envs(benchmark)

    def _state_int(name: str) -> int:
        try:
            parsed = int(getattr(state, name, 0) or 0)
            return parsed if parsed > 0 else 0
        except (TypeError, ValueError):
            return 0

    return RuntimeWorkload(
        model_path=str(benchmark.get("model") or getattr(state, "model_path", "") or ""),
        gpu_type=_runtime_gpu_type(state, benchmark),
        precision=str(benchmark.get("precision") or getattr(state, "precision", "") or ""),
        framework=str(benchmark.get("framework") or getattr(state, "framework", "") or ""),
        tp=_env_int(envs, "TP") or _state_int("tp"),
        concurrency=_env_int(envs, "CONC") or _state_int("conc") or 1,
        isl=_env_int(envs, "ISL") or _state_int("isl"),
        osl=_env_int(envs, "OSL") or _state_int("osl"),
        server_args=_collect_runtime_server_args(state),
    )


@dataclass(frozen=True)
class RuntimeDtype:
    """Resolved runtime precision provenance for the roofline ceiling.

    ``weight_dtype_bytes`` drives the per-token weight IO term; it reflects
    the dtype weights are *actually read in* at runtime (e.g. fp8 when the
    server ran ``--quantization fp8``), not the on-disk ``torch_dtype``.
    ``activation_dtype_bytes`` is the activation/KV dtype and stays >= 2B.

    ``compute_precision_tag`` is the precision key for the compute-peak
    TFLOPS lookup (``fp8`` / ``bf16`` / ...). It differs from
    ``weight_dtype_tag`` for pre-quantized checkpoints whose provenance
    label is ``quantization_config`` but whose GEMM peak is fp8.
    """

    weight_dtype_bytes: float
    activation_dtype_bytes: float
    weight_dtype_tag: str
    quantization: str
    source: str
    compute_precision_tag: str = ""


def resolve_runtime_dtype(state: Any, meta: "ModelMeta") -> RuntimeDtype:
    """Resolve the runtime weight/activation dtype from the actual run.

    Priority (first decisive signal wins):
      1. ``--quantization`` in the recorded server args (the run truly
         quantized the weights, e.g. dense fp8 over a float32 checkpoint).
      2. Model ``quantization_config`` already reflected in ``meta``
         (on-disk weights are pre-quantized; ``weight_dtype_bytes`` < 2).
      3. ``--dtype`` server arg (sets weight+activation when not quantized).
      4. Config ``torch_dtype`` already in ``meta``, floored at bf16.

    Workload ``precision`` is deliberately NOT used to drive the weight dtype:
    a workload tagged ``precision=fp8`` whose run did NOT pass
    ``--quantization`` actually serves bf16/fp16 weights (the server only
    downcasts float32→fp16), so trusting the tag would over-shrink the
    weight IO term and under-report baseline within%.

    Weight dtype is floored at bf16 (2B) in the fallback: servers downcast
    float32 checkpoints to fp16, never keep 4B at runtime, and never go
    sub-bf16 without an explicit quantization signal. Activation dtype
    follows ``--dtype`` when present, else bf16, also floored at 2B.
    """
    runtime = resolve_runtime_workload(state)
    args = runtime.server_args
    quant = _parse_server_arg(args, "--quantization").lower()
    dtype_arg = _parse_server_arg(args, "--dtype").lower()

    act_bytes = max(_resolve_dtype_bytes(dtype_arg) if dtype_arg else 2.0, 2.0)

    if quant and quant in _QUANT_WEIGHT_BYTES:
        wb = _QUANT_WEIGHT_BYTES[quant]
        return RuntimeDtype(
            weight_dtype_bytes=wb,
            activation_dtype_bytes=act_bytes,
            weight_dtype_tag=quant,
            quantization=quant,
            source="server_args_quantization",
            compute_precision_tag=_compute_tag_for_bytes(wb),
        )
    # On-disk weights already sub-bf16 → pre-quantized checkpoint (MoE fp8).
    meta_w_bytes = float(getattr(meta, "weight_dtype_bytes", 0.0) or 0.0)
    if 0 < meta_w_bytes < 2.0:
        wb = meta_w_bytes
        return RuntimeDtype(
            weight_dtype_bytes=wb,
            activation_dtype_bytes=act_bytes,
            weight_dtype_tag="quantization_config",
            quantization="quantization_config",
            source="quantization_config",
            compute_precision_tag=_compute_tag_for_bytes(wb),
        )
    if dtype_arg:
        b = _resolve_dtype_bytes(dtype_arg)
        return RuntimeDtype(
            weight_dtype_bytes=b,
            activation_dtype_bytes=act_bytes,
            weight_dtype_tag=dtype_arg,
            quantization="none",
            source="server_args_dtype",
            compute_precision_tag=_compute_tag_for_bytes(b),
        )
    # No weight-quantization signal: serve at the checkpoint dtype, floored
    # at bf16 (fp32 checkpoints are downcast to fp16 at runtime).
    cfg_b = min(meta_w_bytes, 2.0) if meta_w_bytes > 0 else 2.0
    return RuntimeDtype(
        weight_dtype_bytes=cfg_b,
        activation_dtype_bytes=act_bytes,
        weight_dtype_tag="config_torch_dtype",
        quantization="none",
        source="config_torch_dtype",
        compute_precision_tag=_compute_tag_for_bytes(cfg_b),
    )


def _compute_tag_for_bytes(weight_bytes: float) -> str:
    """Map weight bytes-per-element to a HW_SPECS compute precision key."""
    if weight_bytes <= 0.5:
        return "fp4"
    if weight_bytes <= 1.0:
        return "fp8"
    if weight_bytes <= 2.0:
        return "bf16"
    return "fp32"


def apply_runtime_dtype(meta: "ModelMeta", rt: RuntimeDtype) -> "ModelMeta":
    """Rescale ``meta`` weight bytes to the runtime weight dtype.

    The on-disk ``weight_bytes`` reflects the checkpoint dtype (e.g.
    float32). When the run reads weights at a different dtype (fp8), the
    per-token weight IO must scale by ``runtime_bpe / checkpoint_bpe``.
    No-op (scale == 1.0) when the checkpoint already matches runtime
    (pre-quantized MoE fp8), so it is safe to call unconditionally.
    """
    import dataclasses as _dc

    # Safe degrade for non-dataclass / fake meta (test doubles): return as-is.
    if not _dc.is_dataclass(meta):
        return meta
    cfg_b = float(getattr(meta, "weight_dtype_bytes", 0.0) or 0.0)
    rt_b = float(rt.weight_dtype_bytes or 0.0)
    if cfg_b <= 0 or rt_b <= 0 or abs(cfg_b - rt_b) < 1e-9:
        return _dc.replace(meta, weight_dtype_bytes=rt_b or cfg_b)
    scale = rt_b / cfg_b
    return _dc.replace(
        meta,
        weight_dtype_bytes=rt_b,
        weight_bytes=int(meta.weight_bytes * scale),
        active_weight_bytes=int(meta.active_weight_bytes * scale),
        expert_weight_bytes=int(meta.expert_weight_bytes * scale),
    )


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
    """HF subset needed for the decode roofline ceiling. ``active_weight_bytes`` is per-token MoE weight IO (0 ⇒ fall back to ``weight_bytes``; ``load_model_meta`` always sets it). ``hidden_size``/``intermediate_size``/``vocab_size``/``num_attention_heads`` drive the PerfModel per-op breakdown (default 0)."""

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
    # Extra HF config fields for per-op PerfModel breakdown (0 = unavailable).
    hidden_size: int = 0
    intermediate_size: int = 0
    # Per-expert FFN dim for MoE models (e.g. Qwen3-MoE, DeepSeek-V3).
    # 0 means dense or unknown; PerfModel uses intermediate_size as fallback.
    moe_intermediate_size: int = 0
    vocab_size: int = 0
    num_attention_heads: int = 0


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
    intermediate_size = int(cfg.get("intermediate_size") or 0)
    moe_intermediate_size = int(cfg.get("moe_intermediate_size") or 0)
    if num_experts > 0 and moe_intermediate_size <= 0:
        moe_intermediate_size = intermediate_size
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
        hidden_size=int(cfg.get("hidden_size") or 0),
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        vocab_size=int(cfg.get("vocab_size") or 0),
        num_attention_heads=int(cfg.get("num_attention_heads") or 0),
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
    # Per-decode-step weight IO; MoE activated-expert union uses the coupon form ``activated_fraction = 1-(1-k/n)^B`` (valid upper bound everywhere) instead of the linear ``min(1, B*k/n)`` bound that over-counts at mid batch. Dense (num_experts==0) reads full ``weight_bytes`` each step.
    if (
        num_experts > 0
        and experts_per_tok > 0
        and expert_weight_bytes > 0
    ):
        non_expert_bytes = max(int(weight_bytes) - int(expert_weight_bytes), 0)
        activated_fraction = 1.0 - (
            1.0 - experts_per_tok / num_experts
        ) ** batch
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
    return _env_int(
        _benchmark_envs(_read_baseline_yaml_benchmark(state)),
        "CONC",
    )


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
    """Decode roofline ceiling plus memory/compute side projections (PerfModel peak sums per-op max(t_mem,t_cmp), may differ from min(T_mem,T_cmp)). ``bound_kind`` ∈ {memory, compute, unknown} (unknown ⇒ peak 0)."""
    mem_tok_per_sec: float
    cmp_tok_per_sec: float
    peak_tok_per_sec: float
    bound_kind: str


_EMPTY_BREAKDOWN = RooflineBreakdown(0.0, 0.0, 0.0, "unknown")


def _activation_kv_dtype_bytes(meta: ModelMeta) -> float:
    return max(float(meta.weight_dtype_bytes or 2.0), 2.0)


def compute_roofline_breakdown_from_state(state: Any) -> RooflineBreakdown:
    """Primary decode ceiling + T_mem/T_cmp side projections. Prefers the bottom-up PerfModel (``compute_roofline_from_perfmodel``) when model config is complete, else the legacy top-down aggregate. Never raises; returns ``_EMPTY_BREAKDOWN`` on missing fields."""
    runtime = resolve_runtime_workload(state)
    meta = load_model_meta(
        runtime.model_path,
        precision_hint=runtime.precision,
    )
    if meta is None:
        return _EMPTY_BREAKDOWN
    gpu_type = runtime.gpu_type
    num_gpus = runtime.tp
    concurrency = runtime.concurrency
    # Rescale weights to the dtype the run actually read (e.g. fp8 over a
    # float32 checkpoint) so the ceiling reflects runtime, not on-disk size.
    rt = resolve_runtime_dtype(state, meta)
    meta = apply_runtime_dtype(meta, rt)
    precision_tag = (
        rt.compute_precision_tag
        or runtime.precision
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
        kv_dtype_bytes=_activation_kv_dtype_bytes(meta),
        isl=runtime.isl,
        osl=runtime.osl,
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
        # T_cmp unknown (precision not in HW_SPECS); degrade to T_mem ceiling, memory-bound.
        legacy = RooflineBreakdown(mem, 0.0, mem, "memory")
    elif mem <= 0:
        legacy = RooflineBreakdown(0.0, cmp, cmp, "compute")
    elif cmp < mem:
        legacy = RooflineBreakdown(mem, cmp, cmp, "compute")
    else:
        legacy = RooflineBreakdown(mem, cmp, mem, "memory")

    # Prefer the bottom-up PerfModel peak (MoE FFN uses the coupon expert-activation count, tight at every batch); legacy is the fallback.
    try:
        pm_bd = compute_roofline_from_perfmodel(
            meta=meta,
            gpu_type=gpu_type,
            concurrency=concurrency,
            isl=runtime.isl,
            osl=runtime.osl,
            num_gpus=num_gpus,
            precision_tag=precision_tag,
        )
        if pm_bd is not None and pm_bd.decode_tok_per_s > 0:
            return RooflineBreakdown(
                mem_tok_per_sec=pm_bd.decode_mem_tok_per_s,
                cmp_tok_per_sec=pm_bd.decode_cmp_tok_per_s,
                peak_tok_per_sec=pm_bd.decode_tok_per_s,
                bound_kind=pm_bd.bound_kind,
            )
    except Exception:  # noqa: BLE001 — PerfModel is best-effort
        pass

    return legacy


def compute_peak_from_state(state: Any) -> float:
    """Convenience scalar wrapper for ``T_peak`` only (kept for backward compat; prefer ``compute_roofline_breakdown_from_state``)."""
    return compute_roofline_breakdown_from_state(state).peak_tok_per_sec


# ---------------------------------------------------------------------------
# Phase 2: TraceLens PerfModel per-op breakdown (bottom-up).
# ---------------------------------------------------------------------------

#: Max-achievable (sustained) TFLOPS from TraceLens arch JSON files.
#: These use the same HBM bandwidth as HW_SPECS but replace vendor-quoted
#: dense TFLOPS with the empirically-measured maximum throughput used by
#: the TraceLens team for per-op arithmetic-intensity analysis.
#:
#: Sources: TraceLens/AgenticMode/Standalone/utils/arch/MI{300,325,355}X.json
_MI300X_ACHIEVABLE_TFLOPS: dict[str, float] = {
    "bf16": 708.0, "bfloat16": 708.0, "fp16": 654.0, "float16": 654.0,
    "fp8": 1273.0, "float8_e4m3fn": 1273.0, "float8_e5m2": 1273.0,
    "fp32": 163.0, "float32": 163.0,
}
_MI325X_ACHIEVABLE_TFLOPS: dict[str, float] = {
    "bf16": 843.0, "bfloat16": 843.0, "fp16": 794.0, "float16": 794.0,
    "fp8": 1519.0, "float8_e4m3fn": 1519.0, "float8_e5m2": 1519.0,
    "fp32": 194.0, "float32": 194.0,
}
_MI355X_ACHIEVABLE_TFLOPS: dict[str, float] = {
    "bf16": 1686.0, "bfloat16": 1686.0, "fp16": 1686.0, "float16": 1686.0,
    "fp8": 3567.0, "float8_e4m3fn": 3567.0, "float8_e5m2": 3567.0,
    "mxfp4": 5663.0, "fp4": 5663.0, "float4": 5663.0,
    "fp32": 137.0, "float32": 137.0,
}

HW_SPECS_ACHIEVABLE: dict[str, dict[str, Any]] = {
    "mi300x": {
        "hbm_bw_gbps": 5300.0,
        "hbm_gb": 192.0,
        "peak_tflops": _MI300X_ACHIEVABLE_TFLOPS,
    },
    "mi325x": {
        "hbm_bw_gbps": 6000.0,
        "hbm_gb": 256.0,
        "peak_tflops": _MI325X_ACHIEVABLE_TFLOPS,
    },
    "mi355x": {
        "hbm_bw_gbps": 8000.0,
        "hbm_gb": 288.0,
        "peak_tflops": _MI355X_ACHIEVABLE_TFLOPS,
    },
}


def _resolve_achievable_tflops(gpu_type: str | None, precision_tag: str | None) -> float:
    """Max-achievable TFLOPS from ``HW_SPECS_ACHIEVABLE``; 0.0 on miss."""
    spec = HW_SPECS_ACHIEVABLE.get((gpu_type or "").strip().lower())
    if spec is None:
        return 0.0
    table = spec.get("peak_tflops")
    if not isinstance(table, dict) or not precision_tag:
        return 0.0
    return float(table.get(str(precision_tag).strip().lower(), 0.0))


import dataclasses as _dc


def _fused_moe_flops(M: int, K: int, N: int, topk: int) -> float:
    """FLOPs for one gated SwiGLU MoE forward (gate+up+down projections).

    Mirrors TraceLens.PerfModel.extensions.moe_perf_model_extensions
    .FusedMoE.flops_func with gated=True (all LLM MoE uses SwiGLU):
      gate+up : 2 * M * K * N * topk * 2
      down    : 2 * M * K * N * topk
      aggregation: M * K * (2 * topk - 1)
    """
    return 2.0 * M * K * N * topk * 2 + 2.0 * M * K * N * topk + M * K * (2 * topk - 1)


def _fused_moe_bytes(
    M: int, K: int, N: int, num_experts: int, topk: int,
    weight_bpe: float, act_bpe: float | None = None,
) -> float:
    """HBM bytes for one gated SwiGLU MoE forward using the coupon-collector
    active-expert count, inlined from TraceLens FusedMoE.bytes_func.

    Mirrors the TraceLens separation of input/weight/output dtypes:
      input  : M * K * act_bpe
      fc1 (gate+up weights): E_active * N * K * weight_bpe * 2
      fc2 (down weights)   : E_active * N * K * weight_bpe
      output : M * K * act_bpe

    act_bpe defaults to weight_bpe when not provided. For FP8/FP4-weight
    models pass act_bpe=2.0 (bf16 activations) to match TraceLens semantics
    where input_bpe != weight_bpe.
    """
    if act_bpe is None:
        act_bpe = weight_bpe
    e_active = num_experts * (1.0 - ((num_experts - topk) / num_experts) ** M)
    return (
        M * K * act_bpe                      # input read
        + e_active * N * K * weight_bpe * 2  # gate + up weight reads
        + e_active * N * K * weight_bpe       # down weight reads
        + M * K * act_bpe                     # output write
    )


@_dc.dataclass(frozen=True)
class OpBreakdown:
    """Per-operator roofline result from TraceLens PerfModel."""
    name: str
    flops: float          # total FLOPs (across all layers)
    bytes_moved: float    # total bytes (across all layers)
    ai: float             # arithmetic intensity = flops / bytes_moved
    time_s: float         # roofline time (seconds, across all layers)
    bound: str            # "compute" | "memory"
    pct_time: float       # fraction of total forward time (0–1)


@_dc.dataclass(frozen=True)
class PerfModelBreakdown:
    """Bottom-up per-op roofline breakdown via TraceLens PerfModel.

    ``decode_tok_per_s`` and ``prefill_tok_per_s`` are derived from the
    sum of per-op times over one forward pass at the given batch size.
    ``decode_mem_tok_per_s`` / ``decode_cmp_tok_per_s`` expose the same
    bottom-up formulas under memory-only and compute-only assumptions.

    ``ops`` lists every GEMM / SDPA, one row per logical operator
    (already summed over the layer repetitions encoded in ``rep``).
    ``bound_kind`` reflects the decode-forward dominant bound.
    """
    decode_tok_per_s: float
    prefill_tok_per_s: float
    decode_mem_tok_per_s: float
    decode_cmp_tok_per_s: float
    ops: list[OpBreakdown]
    bound_kind: str     # "compute" | "memory" | "unknown"
    hbm_bw_gbps: float
    peak_achievable_tflops: float


def _gemm_flops(M: int, N: int, K: int) -> float:
    """FLOPs for a bias-free matrix multiply (2*M*N*K).

    Mirrors TraceLens.PerfModel.perf_model.GEMM.flops_func (no bias path
    needed here since LLM linear layers are weight-only without bias in
    the roofline model).
    """
    return 2.0 * M * N * K


def _gemm_bytes(
    M: int, N: int, K: int, weight_bpe: float, act_bpe: float | None = None,
) -> float:
    """HBM bytes for a bias-free matmul: read(act MK + weight KN) + write(act MN).

    Mirrors TraceLens.PerfModel.perf_model.GEMM.bytes_func with bpe_mat1=act_bpe
    (input activation), bpe_mat2=weight_bpe (weight), bpe_output=act_bpe (output).
    act_bpe defaults to weight_bpe; for FP8/FP4-weight models pass act_bpe=2.0
    (bf16 activations) to correctly separate activation from weight bytes.
    """
    a = act_bpe if act_bpe is not None else weight_bpe
    return M * K * a + K * N * weight_bpe + M * N * a


def _sdpa_flops(
    B: int, N_Q: int, H_Q: int, N_KV: int, H_KV: int,
    d_h_qk: int, d_h_v: int, causal: bool,
) -> float:
    """FLOPs for scaled dot-product attention.

    Mirrors TraceLens.PerfModel.perf_model.SDPA.flops_func:
      QK^T  : B * H_Q * 2 * N_Q * N_KV * d_h_qk
      PV    : B * H_Q * 2 * N_Q * d_h_v * N_KV
    Softmax FLOPs omitted (dominated by matmuls).
    Causal masking halves the work when N_Q == N_KV (prefill only).
    """
    flops_qk = B * H_Q * (2.0 * N_Q * N_KV * d_h_qk)
    flops_pv = B * H_Q * (2.0 * N_Q * d_h_v * N_KV)
    total = flops_qk + flops_pv
    if causal and N_Q == N_KV:
        total /= 2.0
    return total


def _sdpa_bytes(
    B: int, N_Q: int, H_Q: int, N_KV: int, H_KV: int,
    d_h_qk: int, d_h_v: int, causal: bool, bpe: float,
) -> float:
    """HBM bytes for SDPA: read Q, K, V + write output.

    Mirrors TraceLens.PerfModel.perf_model.SDPA.bytes_func.
    causal is accepted for API symmetry but does not change the I/O volume
    (KV is always fully read even under causal masking at the HBM level).
    """
    elems = (
        B * N_Q * H_Q * d_h_qk    # Q read
        + B * N_KV * H_KV * d_h_qk  # K read
        + B * N_KV * H_KV * d_h_v   # V read
        + B * N_Q * H_Q * d_h_v     # output write
    )
    return float(elems) * bpe


def compute_roofline_from_perfmodel(
    *,
    meta: "ModelMeta",
    gpu_type: str,
    concurrency: int,
    isl: int,
    osl: int,
    num_gpus: int = 1,
    precision_tag: str = "bf16",
) -> "PerfModelBreakdown | None":
    """Bottom-up decode + prefill roofline using inlined GEMM/SDPA formulas.

    Returns ``None`` when:
      * Model metadata is incomplete (hidden_size / num_attention_heads == 0)
      * GPU is not in ``HW_SPECS_ACHIEVABLE``

    The returned ceilings use **max-achievable** TFLOPS (from
    ``HW_SPECS_ACHIEVABLE`` backed by TraceLens arch JSONs) rather than
    vendor-quoted peaks, so ``within_roofline_pct`` values will be
    somewhat higher than with the legacy formula.  Both flavours are
    upper bounds on real throughput; the achievable variant is the one
    used by TraceLens analysis reports for consistency.

    The GEMM / SDPA formulas are inlined here (no TraceLens import) and
    maintained independently.  They match TraceLens PerfModel exactly
    (verified by the PoC in roofline_perfmodel_poc.py: deviation < 1.6%).
    """
    if meta is None:
        return None
    if not meta.hidden_size or not meta.num_attention_heads:
        return None
    spec = HW_SPECS_ACHIEVABLE.get((gpu_type or "").strip().lower())
    if spec is None:
        return None

    bw_gbps = spec["hbm_bw_gbps"] * max(num_gpus, 1)
    bw_bps = bw_gbps * 1e9
    tag = (precision_tag or "bf16").strip().lower()
    f_peak_tflops = _resolve_achievable_tflops(gpu_type, tag) * max(num_gpus, 1)
    if f_peak_tflops <= 0:
        return None
    f_peak = f_peak_tflops * 1e12
    bpe = float(meta.weight_dtype_bytes or 2.0)
    # Activations (input/output) are at least bf16 even for quantized-weight models.
    # Mirrors TraceLens FusedMoE.bytes() where input_bpe defaults to 2 (bf16)
    # and weight_bpe defaults to 1 (FP8), keeping them separate.
    act_bpe = max(bpe, 2.0)

    hidden = meta.hidden_size
    ffn = meta.intermediate_size
    vocab = meta.vocab_size
    n_q_heads = meta.num_attention_heads
    n_kv_heads = meta.num_kv_heads
    hd = meta.head_dim or (hidden // n_q_heads if n_q_heads else 0)
    n_layers = meta.num_layers

    if not all((hidden, n_q_heads, n_kv_heads, hd, n_layers)):
        return None

    q_out = n_q_heads * hd
    kv_out = n_kv_heads * hd

    # ---- linear projections per layer ----
    # (name, K_in, N_out, repeat_per_forward)
    # Attention projections and lm_head use the standard GEMM formula.
    # MoE FFN is handled separately in _forward via _fused_moe_flops/bytes.
    linears: list[tuple[str, int, int, int]] = [
        ("q_proj",    hidden, q_out,  n_layers),
        ("k_proj",    hidden, kv_out, n_layers),
        ("v_proj",    hidden, kv_out, n_layers),
        ("o_proj",    q_out,  hidden, n_layers),
    ]
    # Dense FFN: add gate/up/down to linears (standard GEMM).
    # MoE FFN is added inside _forward with the batch-aware coupon formula.
    if ffn and not (meta.num_experts > 0 and meta.moe_intermediate_size > 0):
        linears += [
            ("gate_proj", hidden, ffn, n_layers),
            ("up_proj",   hidden, ffn, n_layers),
            ("down_proj", ffn, hidden, n_layers),
        ]
    if vocab:
        linears.append(("lm_head", hidden, vocab, 1))

    def _roofline_time(fl: float, by: float) -> tuple[float, str, float, float]:
        t_cmp = fl / f_peak if f_peak > 0 else 1e30
        t_mem = by / bw_bps if bw_bps > 0 else 1e30
        return max(t_cmp, t_mem), "compute" if t_cmp >= t_mem else "memory", t_mem, t_cmp

    def _forward(batch: int, s_q: int, s_kv: int) -> tuple[float, float, float, list[OpBreakdown]]:
        total_t = 0.0
        total_mem_t = 0.0
        total_cmp_t = 0.0
        op_rows: list[OpBreakdown] = []
        M = batch * s_q
        for name, K, N, rep in linears:
            fl = _gemm_flops(M, N, K)
            by = _gemm_bytes(M, N, K, bpe, act_bpe)
            t, side, t_mem, t_cmp = _roofline_time(fl, by)
            total_t += t * rep
            total_mem_t += t_mem * rep
            total_cmp_t += t_cmp * rep
            op_rows.append(OpBreakdown(
                name=name,
                flops=fl * rep,
                bytes_moved=by * rep,
                ai=(fl / by if by else 0.0),
                time_s=t * rep,
                bound=side,
                pct_time=0.0,  # filled after total is known
            ))
        # MoE FFN: TraceLens FusedMoE formula with batch-aware coupon
        # E_active = n*(1-(1-k/n)^M), which correctly accounts for the fact that
        # at high batch more expert weights are loaded (vs. topk fixed for B=1).
        if meta.num_experts > 0 and meta.moe_intermediate_size > 0:
            fl_moe = _fused_moe_flops(M, hidden, meta.moe_intermediate_size, meta.experts_per_tok)
            by_moe = _fused_moe_bytes(
                M, hidden, meta.moe_intermediate_size,
                meta.num_experts, meta.experts_per_tok, bpe, act_bpe,
            )
            t_moe, side_moe, t_mem_moe, t_cmp_moe = _roofline_time(fl_moe, by_moe)
            total_t += t_moe * n_layers
            total_mem_t += t_mem_moe * n_layers
            total_cmp_t += t_cmp_moe * n_layers
            op_rows.append(OpBreakdown(
                name="moe_fused",
                flops=fl_moe * n_layers,
                bytes_moved=by_moe * n_layers,
                ai=(fl_moe / by_moe if by_moe else 0.0),
                time_s=t_moe * n_layers,
                bound=side_moe,
                pct_time=0.0,
            ))
        # SDPA
        causal = (s_q == s_kv)
        fl_s = _sdpa_flops(batch, s_q, n_q_heads, s_kv, n_kv_heads, hd, hd, causal)
        by_s = _sdpa_bytes(batch, s_q, n_q_heads, s_kv, n_kv_heads, hd, hd, causal, act_bpe)
        t_s, side_s, t_mem_s, t_cmp_s = _roofline_time(fl_s, by_s)
        total_t += t_s * n_layers
        total_mem_t += t_mem_s * n_layers
        total_cmp_t += t_cmp_s * n_layers
        op_rows.append(OpBreakdown(
            name="sdpa",
            flops=fl_s * n_layers,
            bytes_moved=by_s * n_layers,
            ai=(fl_s / by_s if by_s else 0.0),
            time_s=t_s * n_layers,
            bound=side_s,
            pct_time=0.0,
        ))
        # fill pct_time
        if total_t > 0:
            op_rows = [
                _dc.replace(op, pct_time=op.time_s / total_t)
                for op in op_rows
            ]
        return total_t, total_mem_t, total_cmp_t, op_rows

    batch = max(concurrency, 1)
    kv_seq = max(int(isl) + int(osl) // 2, 1)
    t_dec, t_dec_mem, t_dec_cmp, ops_dec = _forward(batch, 1, kv_seq)
    t_pre, _, _, _ = _forward(1, max(int(isl), 1), max(int(isl), 1))

    decode_tok = batch / t_dec if t_dec > 0 else 0.0
    prefill_tok = max(int(isl), 1) / t_pre if t_pre > 0 else 0.0
    decode_mem_tok = batch / t_dec_mem if t_dec_mem > 0 else 0.0
    decode_cmp_tok = batch / t_dec_cmp if t_dec_cmp > 0 else 0.0

    # dominant bound in decode
    bound_kind = "memory" if t_dec_mem >= t_dec_cmp else "compute"
    if t_dec <= 0:
        bound_kind = "unknown"

    return PerfModelBreakdown(
        decode_tok_per_s=decode_tok,
        prefill_tok_per_s=prefill_tok,
        decode_mem_tok_per_s=decode_mem_tok,
        decode_cmp_tok_per_s=decode_cmp_tok,
        ops=ops_dec,
        bound_kind=bound_kind,
        hbm_bw_gbps=bw_gbps,
        peak_achievable_tflops=f_peak_tflops,
    )


def compute_roofline_breakdown_from_state_v2(state: Any) -> "tuple[RooflineBreakdown, PerfModelBreakdown | None]":
    """Return the primary roofline breakdown AND the per-op PerfModel breakdown.

    The first element is a ``RooflineBreakdown`` from
    ``compute_roofline_breakdown_from_state``: when the model config is
    complete, ``mem_tok_per_sec`` / ``cmp_tok_per_sec`` / ``peak_tok_per_sec``
    all come from the bottom-up PerfModel. Legacy aggregate values are fallback.

    The second element is the full per-op ``PerfModelBreakdown`` (``None`` when
    the GPU or model config is unsupported).  Its ``decode_tok_per_s`` equals
    ``breakdown.peak_tok_per_sec`` for supported configs.
    """
    legacy = compute_roofline_breakdown_from_state(state)
    runtime = resolve_runtime_workload(state)
    meta = load_model_meta(
        runtime.model_path,
        precision_hint=runtime.precision,
    )
    pm_bd: "PerfModelBreakdown | None" = None
    if meta is not None:
        try:
            rt = resolve_runtime_dtype(state, meta)
            meta = apply_runtime_dtype(meta, rt)
            pm_bd = compute_roofline_from_perfmodel(
                meta=meta,
                gpu_type=runtime.gpu_type,
                concurrency=runtime.concurrency,
                isl=runtime.isl,
                osl=runtime.osl,
                num_gpus=runtime.tp,
                precision_tag=rt.compute_precision_tag
                or runtime.precision
                or "bf16",
            )
        except Exception:  # noqa: BLE001 — best-effort, never raise
            pm_bd = None
    return legacy, pm_bd
