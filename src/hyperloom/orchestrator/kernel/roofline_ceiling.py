# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Analytic performance ceilings for the roofline snapshot.

Four capabilities live here:

1. The top-down memory-bound decode formula::

       peak_output_tok_per_sec
         = (HBM_BW_per_gpu × num_gpus)
           / (weight_bytes / batch + kv_bytes_per_token × kv_seq_len)

   This formula does not model prefill; ``batch = max(concurrency, 1)``.
2. The compute-bound ceiling (``compute_compute_bound_ceiling_tok_per_sec``)
   and the ``min(T_mem, T_cmp)`` selector (``select_peak_and_bound``).
3. The bottom-up per-op ``PerfModel`` breakdown, which does produce a
   ``prefill_tok_per_s`` figure.
4. The xDiT diffusion images/sec ceiling.

All outputs are upper bounds.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.model_config_utils import _merge_config_scopes

log = logging.getLogger(__name__)


#: GPU per-chip peak specs (keys match ``SharedState.gpu_type``, lowercase).
#: ``hbm_bw_gbps`` is vendor peak; ``peak_tflops`` is DENSE peak (missing key
#: ⇒ 0.0 falls back to T_mem).
_MI300X_PEAK_TFLOPS: dict[str, float] = {
    "bf16": 1307.4,
    "bfloat16": 1307.4,
    "fp16": 1307.4,
    "float16": 1307.4,
    "fp8": 2614.9,
    "float8_e4m3fn": 2614.9,
    "float8_e5m2": 2614.9,
    "fp32": 163.4,
    "float32": 163.4,
}
_MI355X_PEAK_TFLOPS: dict[str, float] = {
    "bf16": 2516.6,
    "bfloat16": 2516.6,
    "fp16": 2516.6,
    "float16": 2516.6,
    "fp8": 5033.2,
    "float8_e4m3fn": 5033.2,
    "float8_e5m2": 5033.2,
    "mxfp4": 10066.4,
    "fp4": 10066.4,
    "float4": 10066.4,
}
HW_SPECS: dict[str, dict[str, Any]] = {
    "mi300x": {
        "hbm_gb": 192.0,
        "hbm_bw_gbps": 5300.0,
        "peak_tflops": _MI300X_PEAK_TFLOPS,
    },
    "mi308x": {
        "hbm_gb": 192.0,
        "hbm_bw_gbps": 5300.0,
        "peak_tflops": _MI300X_PEAK_TFLOPS,
    },
    "mi325x": {
        "hbm_gb": 256.0,
        "hbm_bw_gbps": 6000.0,
        "peak_tflops": _MI300X_PEAK_TFLOPS,
    },
    "mi355x": {
        "hbm_gb": 288.0,
        "hbm_bw_gbps": 8000.0,
        "peak_tflops": _MI355X_PEAK_TFLOPS,
    },
}


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


#: Map a server-arg ``--quantization`` value to weight bytes-per-element.
#: Only weight-quantization methods are listed; activation/KV dtype is
#: tracked separately and stays >= bf16.
_QUANT_WEIGHT_BYTES: dict[str, float] = {
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
    """On-disk weight bytes-per-element declared by an HF ``quantization_config``.

    ``quant_method`` names the *precision* in the flat HF form (``fp8``,
    ``awq``) but names the *toolkit* in wrapper formats — Quark writes
    ``quant_method: "quark"`` and puts the real precision on
    ``global_quant_config.weight.dtype`` (``"fp4"`` for MXFP4). Reading only
    ``quant_method`` therefore misses the tag and silently falls back to the
    checkpoint ``dtype`` (bf16), overcounting weight IO 4x — which also trips
    the ``total_expert_bytes >= weight_bytes`` sanity guard in
    :func:`_compute_expert_decomposition` and degrades a MoE model to dense.

    Args:
        quant_cfg: The parsed ``quantization_config`` block, or any non-dict.

    Returns:
        Bytes per weight element, or ``0.0`` when nothing decisive is found.
    """
    if not isinstance(quant_cfg, dict):
        return 0.0
    method = str(quant_cfg.get("quant_method", "")).strip().lower()
    if method in _DTYPE_BYTES:
        return _DTYPE_BYTES[method]
    if method in _QUANT_WEIGHT_BYTES:
        return _QUANT_WEIGHT_BYTES[method]

    def _bits_to_bytes(bits: Any) -> float:
        """``num_bits`` → bytes-per-element, ``0.0`` when unusable."""
        try:
            b = float(bits)
        except (TypeError, ValueError):
            return 0.0
        return b / 8.0 if b > 0 else 0.0

    def _spec_bytes(scope: Any) -> float:
        """Weight bytes-per-element declared by one quant scope, ``0.0`` if none."""
        if not isinstance(scope, dict):
            return 0.0
        spec = scope.get("weight") or scope.get("weights")
        if not isinstance(spec, dict):
            return 0.0
        tag = str(spec.get("dtype") or spec.get("type") or "").strip().lower()
        if tag in _DTYPE_BYTES:
            return _DTYPE_BYTES[tag]
        return _bits_to_bytes(spec.get("num_bits") or spec.get("bits"))

    # Wrapper toolkits nest the precision on a weight spec: Quark under
    # global_quant_config, compressed-tensors under config_groups.*.weights.
    # These two scopes describe the checkpoint as a whole, so either is decisive.
    for scope in (quant_cfg.get("global_quant_config"), quant_cfg):
        whole = _spec_bytes(scope)
        if whole > 0:
            return whole

    # The per-group containers are different: one entry per layer pattern, and
    # they need not agree -- a Quark MoE checkpoint routinely stores the routed
    # experts at fp4 and the attention projections at fp8. Reading whichever
    # entry the dict happened to yield first declared that precision for the
    # entire model. Take the value the most groups agree on; ties resolve to the
    # wider one, because undercounting weight bytes inflates the roofline and
    # reports a real regression as "already at ceiling".
    tallies: dict[float, int] = {}
    for container in (quant_cfg.get("layer_quant_config"), quant_cfg.get("config_groups")):
        if not isinstance(container, dict):
            continue
        for scope in container.values():
            by = _spec_bytes(scope)
            if by > 0:
                tallies[by] = tallies.get(by, 0) + 1
    if tallies:
        return max(tallies.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return _bits_to_bytes(quant_cfg.get("bits") or quant_cfg.get("w_bit"))


def _parse_server_arg(args: str, flag: str) -> str:
    """Return the value following ``--flag`` (space or ``=`` form), else ``""``.

    Tolerant of both ``--quantization fp8`` and ``--quantization=fp8``.
    When the flag appears multiple times, the last value wins so an
    optimized overlay can override the baseline args.

    Args:
        args: The server-args string to scan.
        flag: The flag whose value to extract (e.g. ``--quantization``).

    Returns:
        The last value following the flag, or ``""`` when absent.
    """
    if not args:
        return ""
    toks = str(args).replace("=", " ").split()
    value = ""
    for i, tok in enumerate(toks):
        if tok == flag and i + 1 < len(toks):
            value = toks[i + 1].strip()
    return value


#: Magpie ``benchmark.envs`` keys that carry the runtime server args for the
#: text-serving frameworks. The baseline yaml only ever sets the one matching
#: its framework, so reading these and concatenating is safe.
_RUNTIME_SERVER_ARG_ENV_KEYS = (
    "EXTRA_SGLANG_ARGS",
    "EXTRA_VLLM_ARGS",
    "EXTRA_ATOM_ARGS",
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
    """Read ``benchmark`` from the materialized baseline yaml.

    Args:
        state: Shared run state carrying ``last_baseline`` provenance.

    Returns:
        The ``benchmark`` mapping, or ``{}`` when unreadable.
    """
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
    """Return the ``envs`` mapping from a benchmark record.

    Args:
        benchmark: Benchmark record.

    Returns:
        The ``envs`` dict, or ``{}`` when absent or not a dict.
    """
    envs = benchmark.get("envs") or {}
    return envs if isinstance(envs, dict) else {}


def _env_int(envs: dict[str, Any], key: str) -> int:
    """Read a positive integer from an env mapping.

    Args:
        envs: Environment mapping.
        key: Key to read.

    Returns:
        The parsed positive integer, or ``0`` when missing, non-positive, or
        unparseable.
    """
    raw = envs.get(key)
    if raw is None:
        return 0
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else 0
    except (TypeError, ValueError):
        return 0


def _server_args_from_envs(envs: dict[str, Any]) -> str:
    """Assemble server CLI args from recognized env keys.

    Args:
        envs: Environment mapping.

    Returns:
        A space-joined string of the non-empty server-arg env values.
    """
    parts = [
        str(envs[k]).strip() for k in _RUNTIME_SERVER_ARG_ENV_KEYS if isinstance(envs.get(k), str) and envs[k].strip()
    ]
    return " ".join(parts)


def _read_baseline_yaml_server_args(state: Any) -> str:
    """Read the runtime server args from the materialized baseline yaml.

    The baseline ``current_best`` snapshot carries no ``extra_server_args``
    (the flags live only in ``benchmark.envs.EXTRA_*_ARGS`` of the on-disk
    yaml), so a baseline-only run with ``--quantization fp8`` in the yaml
    would otherwise be invisible to dtype resolution. If the materialized YAML
    is unreadable, fall back to ``last_baseline`` payload/env args so prelude
    profiling does not silently drop operator server flags.

    Args:
        state: Shared run state carrying ``last_baseline`` provenance.

    Returns:
        The assembled baseline server args, or ``""`` when unreadable.
    """
    yaml_args = _server_args_from_envs(_benchmark_envs(_read_baseline_yaml_benchmark(state)))
    if yaml_args:
        return yaml_args
    return _server_args_from(getattr(state, "last_baseline", None))


def _server_args_env_override(entry: Any) -> str:
    """Return framework server args pinned via ``extra_envs``.

    Args:
        entry: A state entry that may carry an ``extra_envs`` mapping.

    Returns:
        The server args assembled from ``extra_envs``, or ``""``.
    """
    if not isinstance(entry, dict):
        return ""
    envs = entry.get("extra_envs") or {}
    if isinstance(envs, dict):
        return _server_args_from_envs(envs)
    return ""


def _server_args_payload(entry: Any) -> str:
    """Return framework-neutral overlay server args from an entry.

    Args:
        entry: A state entry that may carry overlay server-arg keys.

    Returns:
        The first non-empty overlay server-args string, or ``""``.
    """
    if not isinstance(entry, dict):
        return ""
    for key in ("candidate_extra_server_args", "extra_server_args", "extra_args"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _server_args_from(entry: Any) -> str:
    """Extract the final server-args string from one state entry.

    Args:
        entry: A state entry to read server args from.

    Returns:
        The env-override args if present, else the overlay payload args.
    """
    return _server_args_env_override(entry) or _server_args_payload(entry)


def _achieved_arm_source(state: Any) -> str:
    """Which arm the roofline ``achieved`` throughput comes from.

    Mirrors the snapshot writer (``current_best.tput > 0`` ⇒ optimized, else
    baseline) so the ceiling's dtype is resolved from the same run its measured
    throughput came from.

    Args:
        state: Shared run state carrying ``current_best``.

    Returns:
        ``"current_best"`` when the optimized arm has positive throughput,
        otherwise ``"baseline"``.
    """
    cb = getattr(state, "current_best", None)
    if isinstance(cb, dict):
        t = cb.get("tput")
        if isinstance(t, (int, float)) and t > 0:
            return "current_best"
    return "baseline"


def _collect_runtime_server_args(state: Any, *, arm: str | None = None) -> str:
    """Server args for the arm the ceiling is actually compared against.

    Selects a SINGLE source aligned with ``achieved`` (see
    ``_achieved_arm_source``). Baseline materialized yaml is the base
    config; current_best contributes only overlay server args when the
    measured throughput comes from the optimized arm. ``arm`` pins the
    source explicitly ("baseline" never overlays current_best), so a
    baseline snapshot's ceiling precision stays anchored to baseline.

    Args:
        state: Shared run state to read baseline / current_best args from.
        arm: Pins the source arm; ``None`` infers it via
            ``_achieved_arm_source``.

    Returns:
        The server-args string for the selected arm.
    """
    base_args = _read_baseline_yaml_server_args(state) or _server_args_from(getattr(state, "last_baseline", None))
    resolved_arm = arm or _achieved_arm_source(state)
    if resolved_arm == "current_best":
        current_best = getattr(state, "current_best", None)
        override = _server_args_env_override(current_best)
        if override:
            return override
        overlay = _server_args_payload(current_best)
        return " ".join(p for p in (base_args, overlay) if p)
    return base_args


def _runtime_gpu_type(state: Any, benchmark: dict[str, Any]) -> str:
    """Resolve real hardware for roofline; runner_type is only script routing.

    Args:
        state: Shared run state carrying ``gpu_type``.
        benchmark: Benchmark record (fallback ``runner_type``).

    Returns:
        The resolved GPU type string, or ``""`` when unknown.
    """
    return str(
        getattr(state, "gpu_type", "") or os.environ.get("TARGET_GPU_TYPE", "") or benchmark.get("runner_type") or ""
    )


def resolve_runtime_workload(
    state: Any,
    *,
    arm: str | None = None,
) -> RuntimeWorkload:
    """Resolve runtime workload fields from baseline yaml plus overlay args.

    Geometry (model/gpu/tp/conc/isl/osl) always comes from the baseline
    yaml. ``arm`` only pins which arm's server args feed precision
    resolution; ``"baseline"`` keeps the ceiling's dtype anchored to
    baseline even after current_best is promoted.

    Args:
        state: Shared run state to resolve workload fields from.
        arm: Pins which arm's server args feed precision resolution.

    Returns:
        The resolved ``RuntimeWorkload``.
    """
    benchmark = _read_baseline_yaml_benchmark(state)
    envs = _benchmark_envs(benchmark)

    def _state_int(name: str) -> int:
        """Read a positive integer attribute from ``state``.

        Args:
            name: Attribute name to read.

        Returns:
            The positive integer value, or ``0`` when missing, non-positive, or
            unparseable.
        """
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
        server_args=_collect_runtime_server_args(state, arm=arm),
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


def resolve_runtime_dtype(
    state: Any,
    meta: "ModelMeta",
    *,
    arm: str | None = None,
) -> RuntimeDtype:
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

    Args:
        state: Shared run state to read the runtime server args from.
        meta: Model metadata (its on-disk weight dtype is a fallback signal).
        arm: Pins which arm's server args feed precision resolution.

    Returns:
        The resolved ``RuntimeDtype`` provenance.
    """
    runtime = resolve_runtime_workload(state, arm=arm)
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
    # On-disk weights already sub-bf16 → pre-quantized checkpoint.
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
    # No weight-quantization signal: serve at the checkpoint dtype, floored at bf16.
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
    """Map weight bytes-per-element to a HW_SPECS compute precision key.

    Args:
        weight_bytes: Weight bytes-per-element.

    Returns:
        The matching precision key (``fp4`` / ``fp8`` / ``bf16`` / ``fp32``).
    """
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

    Args:
        meta: Model metadata whose weight byte fields are rescaled.
        rt: Resolved runtime dtype carrying the runtime weight bytes.

    Returns:
        A new ``ModelMeta`` rescaled to the runtime weight dtype (or ``meta``
        unchanged for non-dataclass test doubles).
    """
    import dataclasses as _dc

    # Safe degrade for non-dataclass / fake meta (test doubles).
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


def _resolve_tflops(specs: dict[str, dict[str, Any]], gpu_type: str | None, precision_tag: str | None) -> float:
    """``(gpu, precision)`` → TFLOPS from *specs*' ``peak_tflops`` table (case-insensitive); 0.0 on any miss."""
    spec = specs.get((gpu_type or "").strip().lower())
    if spec is None:
        return 0.0
    table = spec.get("peak_tflops")
    if not isinstance(table, dict) or not precision_tag:
        return 0.0
    return float(table.get(str(precision_tag).strip().lower(), 0.0))


def _resolve_peak_tflops(gpu_type: str | None, precision_tag: str | None) -> float:
    """``(gpu, precision)`` → vendor dense peak TFLOPS; 0.0 on miss (safe-degrade signal → T_cmp unavailable, fall back to T_mem)."""
    return _resolve_tflops(HW_SPECS, gpu_type, precision_tag)


@dataclass(frozen=True)
class ModelMeta:
    """HF subset needed for the decode roofline ceiling.

    ``active_weight_bytes`` is per-token MoE weight IO (0 ⇒ fall back to
    ``weight_bytes``). ``hidden_size`` / ``intermediate_size`` / ``vocab_size`` /
    ``num_attention_heads`` drive the PerfModel per-op breakdown (default 0).
    """

    weight_bytes: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    weight_dtype_bytes: float
    active_weight_bytes: int = 0
    # MoE expert decomposition (0 for dense).
    num_experts: int = 0
    experts_per_tok: int = 0
    expert_weight_bytes: int = 0
    # Per-element bytes for the expert (routed FFN) weights. Some MoE checkpoints
    # store experts at a *different* precision than the rest of the model (e.g.
    # DeepSeek-V4 ``expert_dtype: fp4`` while ``quant_method: fp8``). 0 ⇒ same as
    # ``weight_dtype_bytes`` (dense or uniform-precision MoE).
    expert_weight_dtype_bytes: float = 0.0
    # Extra HF config fields for per-op PerfModel breakdown (0 = unavailable).
    hidden_size: int = 0
    intermediate_size: int = 0
    # Per-expert FFN dim for MoE models (0 = dense/unknown; falls back to intermediate_size).
    moe_intermediate_size: int = 0
    # Routed-expert input dim. Latent-MoE decoders (Kimi-K3
    # ``routed_expert_hidden_size``) run the experts in a narrower space than
    # the residual stream. 0 ⇒ same as ``hidden_size``.
    moe_hidden_size: int = 0
    vocab_size: int = 0
    num_attention_heads: int = 0
    # How the decoder stack splits between routed-expert FFN layers and plain
    # dense FFN layers. Real MoE checkpoints are not MoE all the way down:
    # DeepSeek/GLM run the first ``first_k_dense_replace`` layers as dense FFN,
    # and Qwen can mark individual layers dense via ``mlp_only_layers``.
    # Multiplying the MoE op by every layer overstates the largest term in the
    # breakdown, and dropping the dense FFN entirely loses a real one.
    # Both default to 0, meaning "not derived": the PerfModel then falls back to
    # the previous all-or-nothing behaviour, so a hand-built ``ModelMeta`` keeps
    # working. ``load_model_meta`` fills them from the HF config.
    moe_layers: int = 0
    dense_ffn_layers: int = 0


def _read_total_size(model_path: Path) -> int | None:
    """Read ``metadata.total_size`` (bytes) from the safetensors index (byte-exact).

    Args:
        model_path: Local HF model directory.

    Returns:
        The total weight size in bytes, or ``None`` when unavailable.
    """
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
    """Fallback weight size from local weight shards matching ``pattern``.

    Args:
        model_path: Local HF model directory.
        pattern: Glob pattern for weight shards (e.g. ``*.safetensors``).

    Returns:
        The summed shard size in bytes, or ``None`` when zero/unreadable.
    """
    try:
        total = sum(p.stat().st_size for p in model_path.glob(pattern) if p.is_file())
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
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    # Multimodal wrappers (e.g. Kimi-K2 kimi_k25) nest the real decoder shape /
    # MoE config under text_config/llm_config/language_config; flatten it (nested
    # wins) so num_experts / hidden_size / moe_intermediate_size / kv-heads reach
    # the ceiling readers instead of degrading to a dense full-weight roofline
    # (num_experts=0 -> active_weight_bytes = full weight_bytes; PerfModel falls
    # back to the legacy top-down formula).
    return _merge_config_scopes(data)


def _derive_experts_per_tok(cfg: dict[str, Any]) -> int:
    """Routed experts activated per token, across the naming variants in use.

    ``num_experts_per_tok`` is the common HF spelling; Gemma-4 writes
    ``top_k_experts`` and Kimi-K3 writes ``num_experts_per_token``. Missing the
    alias reads 0, which safe-degrades the whole MoE decomposition to dense.

    Args:
        cfg: Parsed HF ``config.json``.

    Returns:
        The per-token routed-expert count, or ``0`` when no alias is present.
    """
    for key in ("num_experts_per_tok", "num_experts_per_token", "top_k_experts", "moe_topk"):
        value = cfg.get(key)
        if value:
            return int(value)
    return 0


def _derive_moe_hidden_size(cfg: dict[str, Any]) -> int:
    """The routed experts' input dimension, which need not be ``hidden_size``.

    Latent-MoE decoders (Kimi-K3) project into a narrower space before the
    experts and declare it as ``routed_expert_hidden_size``; sizing the experts
    at the model ``hidden_size`` overcounts their weights (2x for Kimi-K3),
    which trips the ``total_expert_bytes >= weight_bytes`` safe-degrade.

    Args:
        cfg: Parsed HF ``config.json``.

    Returns:
        The expert input dimension, falling back to ``hidden_size``.
    """
    return int(cfg.get("routed_expert_hidden_size") or cfg.get("moe_hidden_size") or cfg.get("hidden_size") or 0)


def _derive_kv_heads(cfg: dict[str, Any]) -> int:
    """GQA-aware: ``num_key_value_heads`` if present, else ``num_attention_heads``.

    Args:
        cfg: Parsed HF ``config.json``.

    Returns:
        The number of KV heads, or ``0`` when neither field is present.
    """
    kv = cfg.get("num_key_value_heads")
    if kv is None:
        kv = cfg.get("num_attention_heads")
    return int(kv or 0)


def _derive_head_dim(cfg: dict[str, Any]) -> int:
    """``head_dim`` directly, or ``hidden_size / num_attention_heads``.

    Args:
        cfg: Parsed HF ``config.json``.

    Returns:
        The per-head dimension, or ``0`` when it cannot be derived.
    """
    head_dim = cfg.get("head_dim")
    if head_dim:
        return int(head_dim)
    hidden = cfg.get("hidden_size")
    heads = cfg.get("num_attention_heads")
    if hidden and heads:
        return int(hidden) // int(heads)
    return 0


def _derive_moe_layer_counts(cfg: dict[str, Any], num_layers: int, num_experts: int) -> tuple[int, int]:
    """Split the decoder stack into ``(moe_layers, dense_ffn_layers)``.

    A MoE checkpoint is rarely MoE in every layer. The three mechanisms in the
    wild:

    * ``first_k_dense_replace`` (DeepSeek, GLM, Kimi) -- the first K layers run
      a plain dense FFN. GLM-5.3-Flash sets 1, DeepSeek-V3 sets 3.
    * ``moe_layer_freq`` -- either a per-layer 0/1 list, or an int stride
      applied after the dense prefix. ``decoder_sparse_step`` is Qwen's name
      for the same stride.
    * ``mlp_only_layers`` (Qwen) -- explicit indices that stay dense.

    Anything unrecognised degrades to "every layer after the dense prefix is
    MoE", which is the common case and matches the previous behaviour when the
    prefix is 0.

    Args:
        cfg: Parsed HF ``config.json``.
        num_layers: ``num_hidden_layers``.
        num_experts: Routed expert count; ``<= 0`` means the model is dense.

    Returns:
        ``(moe_layers, dense_ffn_layers)``, summing to ``num_layers``.
    """
    if num_layers <= 0:
        return 0, 0
    if num_experts <= 0:
        return 0, num_layers

    freq = cfg.get("moe_layer_freq")
    if isinstance(freq, (list, tuple)) and len(freq) == num_layers:
        moe = sum(1 for v in freq if v)
        return moe, num_layers - moe

    first_dense = int(cfg.get("first_k_dense_replace") or 0)
    first_dense = max(0, min(first_dense, num_layers))

    # The stride is phased differently by the two upstream implementations, and
    # both phase it on the ABSOLUTE layer index -- not on the offset from the
    # dense prefix. DeepSeek/GLM:
    #     layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
    # Qwen3-MoE:
    #     layer_idx not in mlp_only_layers and (layer_idx + 1) % decoder_sparse_step == 0
    # Re-basing to ``(i - first_dense) % stride`` agreed with neither: for
    # DeepSeek-V3 (first_k_dense_replace=3) any stride > 1 selected a layer set
    # shifted by 3, and for Qwen it was off by one in the other direction.
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
    dtype_bytes: float,
    expert_dtype_bytes: float = 0.0,
) -> tuple[int, int, int, int]:
    """MoE decomposition for the batch-aware roofline; returns ``(active_weight_bytes, total_expert_bytes, num_experts, experts_per_tok)``. Safe-degrades to ``(weight_bytes, 0, 0, 0)``. Handles num_experts / n_routed_experts / num_local_experts aliases, the per-token aliases in :func:`_derive_experts_per_tok`, and the latent-MoE expert width in :func:`_derive_moe_hidden_size`.

    ``expert_dtype_bytes`` sizes the routed-expert weights when they are stored
    at a different precision than the rest of the model (e.g. DeepSeek-V4
    ``expert_dtype: fp4`` under ``quant_method: fp8``); ``0`` falls back to
    ``dtype_bytes``. Using the global dtype here over-counts expert bytes for
    such checkpoints, tripping the ``total_expert_bytes >= weight_bytes``
    safe-degrade and silently dropping the entire MoE from the ceiling.

    Args:
        cfg: Parsed HF ``config.json``.
        weight_bytes: Total weight bytes (the safe-degrade fallback).
        dtype_bytes: Weight bytes-per-element (non-expert / fallback).
        expert_dtype_bytes: Bytes-per-element for the routed-expert weights;
            ``0`` falls back to ``dtype_bytes``.

    Returns:
        A tuple of ``(active_weight_bytes, total_expert_bytes, num_experts,
        experts_per_tok)``, degrading to ``(weight_bytes, 0, 0, 0)``.
    """
    num_experts = int(cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg.get("num_local_experts") or 0)
    experts_per_tok = _derive_experts_per_tok(cfg)
    if num_experts <= 0 or experts_per_tok <= 0:
        return int(weight_bytes), 0, 0, 0
    hidden_size = _derive_moe_hidden_size(cfg)
    num_layers = int(cfg.get("num_hidden_layers") or 0)
    moe_inter = int(cfg.get("moe_intermediate_size") or cfg.get("intermediate_size") or 0)
    expert_bpe = expert_dtype_bytes if expert_dtype_bytes > 0 else dtype_bytes
    if hidden_size <= 0 or num_layers <= 0 or moe_inter <= 0 or expert_bpe <= 0:
        return int(weight_bytes), 0, 0, 0
    # Only the MoE layers hold expert weights. Charging every layer for them
    # inflates ``total_expert_bytes``, which both overstates the per-token
    # active bytes and pushes borderline checkpoints over the
    # ``>= weight_bytes`` safe-degrade that drops the MoE from the ceiling
    # altogether.
    moe_layers, _dense_ffn_layers = _derive_moe_layer_counts(cfg, num_layers, num_experts)
    if moe_layers <= 0:
        return int(weight_bytes), 0, 0, 0
    expert_bytes_per_layer = num_experts * 3 * hidden_size * moe_inter * expert_bpe
    total_expert_bytes = int(moe_layers * expert_bytes_per_layer)
    if total_expert_bytes <= 0 or total_expert_bytes >= int(weight_bytes):
        # Degrading a MoE model to dense drops the largest op from the
        # breakdown entirely, so say so: the usual cause is an over-large
        # ``expert_bpe`` (a quantized checkpoint misread as bf16), not a
        # genuinely dense model.
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
    non_expert_bytes = int(weight_bytes) - total_expert_bytes
    active_expert_bytes = int((experts_per_tok / num_experts) * total_expert_bytes)
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
    """Read ``weight_bytes`` + KV-cache shape from a local HF model dir (``None`` when unreadable). Weight-dtype priority: quantization_config.quant_method > torch_dtype > dtype > precision_hint.

    Args:
        model_path: Local HF model directory.
        precision_hint: Fallback precision tag when config lacks a dtype.

    Returns:
        The populated ``ModelMeta``, or ``None`` when the dir/config/weights
        are unreadable.
    """
    if not model_path:
        return None
    p = Path(model_path).expanduser()
    if not p.is_dir():
        # ``model_path`` may be an HF repo id (the CLI ``--model`` is persisted
        # verbatim); resolve it to the engine's local HF cache dir via the shared
        # resolver so the decode ceiling isn't null for repo-id launches. Lazy
        # import avoids an import-time orchestrator -> inference_optimizer cycle;
        # ``None`` preserves the prior "unreadable -> None" behavior.
        from hyperloom.inference_optimizer.model_config_utils import (
            resolve_local_model_dir,
        )

        resolved = resolve_local_model_dir(model_path)
        if resolved is None:
            return None
        p = resolved
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
    # A declared quantization wins outright: ``quant_method`` alone is not
    # enough (Quark says ``"quark"``, precision lives on the nested weight
    # spec), and falling through to the checkpoint ``dtype`` would report
    # bf16 for a 4-bit checkpoint.
    quant_bytes = _resolve_quant_config_weight_bytes(quant_cfg)
    if quant_bytes > 0:
        dtype_bytes = quant_bytes
    else:
        dtype_bytes = _resolve_dtype_bytes(quant_tag or cfg.get("torch_dtype") or cfg.get("dtype") or precision_hint)
    # Routed experts may be stored at a distinct precision (DeepSeek-V4
    # ``expert_dtype: fp4`` under fp8 attention). Fall back to the global dtype
    # when the field is absent (uniform-precision MoE / dense).
    expert_dtype_raw = str(cfg.get("expert_dtype") or "").strip()
    expert_dtype_bytes = _resolve_dtype_bytes(expert_dtype_raw) if expert_dtype_raw else dtype_bytes
    active_weight_bytes, total_expert_bytes, num_experts, experts_per_tok = _compute_expert_decomposition(
        cfg,
        weight_bytes=weight_bytes,
        dtype_bytes=dtype_bytes,
        expert_dtype_bytes=expert_dtype_bytes,
    )
    n_moe_layers, n_dense_ffn_layers = _derive_moe_layer_counts(
        cfg, int(cfg.get("num_hidden_layers") or 0), num_experts
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
        expert_weight_dtype_bytes=(expert_dtype_bytes if num_experts > 0 else 0.0),
        hidden_size=int(cfg.get("hidden_size") or 0),
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        moe_hidden_size=(_derive_moe_hidden_size(cfg) if num_experts > 0 else 0),
        vocab_size=int(cfg.get("vocab_size") or 0),
        num_attention_heads=int(cfg.get("num_attention_heads") or 0),
        moe_layers=n_moe_layers,
        dense_ffn_layers=n_dense_ffn_layers,
    )


def compute_kv_bytes_per_token(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype_bytes: float,
) -> int:
    """KV cache footprint per generated token, summed over all layers (the ``2`` covers K + V).

    Args:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimension.
        kv_dtype_bytes: Bytes per KV element.

    Returns:
        The KV-cache bytes per generated token.
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
    """Decode-only memory-bound ceiling for ``output_throughput`` (returns 0.0, never raises, on unknown gpu_type / degenerate divisor). ``active_weight_bytes`` shrinks per-token IO for MoE.

    Args:
        gpu_type: GPU type key for the HBM bandwidth lookup.
        num_gpus: Number of GPUs (tensor-parallel degree).
        weight_bytes: Total weight bytes.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimension.
        kv_dtype_bytes: Bytes per KV element.
        isl: Input sequence length.
        osl: Output sequence length.
        concurrency: Decode batch size (floored at 1).
        active_weight_bytes: Per-token active weight bytes for MoE (0 = dense).
        num_experts: Total experts (0 = dense).
        experts_per_tok: Experts activated per token.
        expert_weight_bytes: Total expert weight bytes.

    Returns:
        The memory-bound decode throughput ceiling, or ``0.0`` on unknown
        GPU type or a degenerate divisor.
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
    # Average KV-cache length during decode (isl + half of osl).
    kv_seq_len = max(int(isl) + int(osl) // 2, 1)
    # Per-decode-step weight IO; MoE uses the coupon activated-expert fraction
    # ``1-(1-k/n)^B``; dense reads full ``weight_bytes`` each step.
    if num_experts > 0 and experts_per_tok > 0 and expert_weight_bytes > 0:
        non_expert_bytes = max(int(weight_bytes) - int(expert_weight_bytes), 0)
        activated_fraction = 1.0 - (1.0 - experts_per_tok / num_experts) ** batch
        effective_weight = non_expert_bytes + activated_fraction * int(expert_weight_bytes)
    else:
        effective_weight = float(
            int(active_weight_bytes) if active_weight_bytes and active_weight_bytes > 0 else int(weight_bytes)
        )
    bytes_per_token_total = effective_weight / batch + kv_bytes * kv_seq_len
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

    ``F_peak`` is the max-achievable (sustained) TFLOPS (same convention as
    :func:`compute_roofline_from_perfmodel`); the vendor dense peak is only a
    coverage-gap fallback when the achievable table lacks the (gpu, precision).

    Args:
        gpu_type: GPU type key for the peak TFLOPS lookup.
        num_gpus: Number of GPUs (tensor-parallel degree).
        precision_tag: Precision key for the peak TFLOPS lookup.
        active_weight_bytes: Per-token active weight bytes at B=1.
        weight_bytes: Total weight bytes (fallback when active is missing).
        weight_dtype_bytes: Weight bytes-per-element.

    Returns:
        The compute-bound decode throughput ceiling, or ``0.0`` on missing
        input.
    """
    peak_tflops = _resolve_achievable_tflops(gpu_type, precision_tag) or _resolve_peak_tflops(gpu_type, precision_tag)
    if peak_tflops <= 0 or weight_dtype_bytes <= 0:
        return 0.0
    # B=1 per-token figure; fall back to dense weight_bytes when active is missing.
    active_b1 = int(active_weight_bytes) if active_weight_bytes and active_weight_bytes > 0 else int(weight_bytes)
    if active_b1 <= 0:
        return 0.0
    peak_flops_total = peak_tflops * 1e12 * max(num_gpus, 1)
    flops_per_token = 2.0 * active_b1 / float(weight_dtype_bytes)
    if flops_per_token <= 0:
        return 0.0
    return peak_flops_total / flops_per_token


def _read_baseline_yaml_conc(state: Any) -> int:
    """Read ``benchmark.envs.CONC`` from the materialized baseline yaml (ground-truth concurrency; ``0`` when unreadable).

    Args:
        state: Shared run state carrying ``last_baseline`` provenance.

    Returns:
        The baseline concurrency, or ``0`` when unreadable.
    """
    return _env_int(
        _benchmark_envs(_read_baseline_yaml_benchmark(state)),
        "CONC",
    )


@dataclass(frozen=True)
class RooflineBreakdown:
    """Decode roofline ceiling plus memory/compute side projections (PerfModel peak sums per-op max(t_mem,t_cmp), may differ from min(T_mem,T_cmp)). ``bound_kind`` ∈ {memory, compute, unknown} (unknown ⇒ peak 0)."""

    mem_tok_per_sec: float
    cmp_tok_per_sec: float
    peak_tok_per_sec: float
    bound_kind: str


_EMPTY_BREAKDOWN = RooflineBreakdown(0.0, 0.0, 0.0, "unknown")


def select_peak_and_bound(t_mem: float, t_cmp: float) -> tuple[float, str]:
    """Pick the dominant (lower) ceiling and its label from the memory- and
    compute-bound projections.

    Both non-positive ⇒ ``(0.0, "unknown")``; only one positive ⇒ that one;
    else the smaller wins (a tie goes to memory).

    Args:
        t_mem: Memory-bound throughput ceiling (tok/s); ``<=0`` means unknown.
        t_cmp: Compute-bound throughput ceiling (tok/s); ``<=0`` means unknown.

    Returns:
        ``(peak_tok_per_sec, bound_kind)`` with ``bound_kind`` ∈
        {``"memory"``, ``"compute"``, ``"unknown"``}.
    """
    if t_mem <= 0 and t_cmp <= 0:
        return 0.0, "unknown"
    if t_cmp <= 0:
        return t_mem, "memory"
    if t_mem <= 0:
        return t_cmp, "compute"
    if t_cmp < t_mem:
        return t_cmp, "compute"
    return t_mem, "memory"


def _activation_kv_dtype_bytes(meta: ModelMeta) -> float:
    """Return the per-element byte size for activation/KV tensors.

    Args:
        meta: Model metadata.

    Returns:
        The dtype byte width used for activation and KV-cache sizing.
    """
    return max(float(meta.weight_dtype_bytes or 2.0), 2.0)


def _read_diffusion_num_steps(state: Any) -> int:
    """Read the denoising step count from the baseline yaml.

    Reads ``XDIT_NUM_STEPS`` (xDiT) with a ``CUSTOM_NUM_STEPS`` fallback so an
    operator-supplied diffusion workload can feed the roofline too.

    Args:
        state: Shared run state carrying the materialized baseline yaml.

    Returns:
        The positive step count, or ``0`` when unavailable.
    """
    envs = _benchmark_envs(_read_baseline_yaml_benchmark(state))
    return _env_int(envs, "XDIT_NUM_STEPS") or _env_int(envs, "CUSTOM_NUM_STEPS")


def _read_diffusion_resolution(state: Any) -> tuple[int, int]:
    """Read the image/frame resolution from the baseline yaml.

    Reads ``XDIT_HEIGHT``/``XDIT_WIDTH`` (xDiT) with ``CUSTOM_HEIGHT``/
    ``CUSTOM_WIDTH`` fallback. Needed for models (e.g. FLUX,
    SD3) whose transformer config carries no ``sample_size`` -- the DiT sequence
    length is set by the runtime resolution.

    Args:
        state: Shared run state carrying the materialized baseline yaml.

    Returns:
        ``(height, width)`` in pixels; ``(0, 0)`` when unavailable.
    """
    try:
        envs = _benchmark_envs(_read_baseline_yaml_benchmark(state))
        height = _env_int(envs, "XDIT_HEIGHT") or _env_int(envs, "CUSTOM_HEIGHT")
        width = _env_int(envs, "XDIT_WIDTH") or _env_int(envs, "CUSTOM_WIDTH")
        return height, width
    except (AttributeError, TypeError, ValueError):
        return 0, 0


def _read_vae_geometry(model_path: str) -> tuple[int, int]:
    """Read ``(vae_scale_factor, latent_channels)`` from ``<model>/vae/config.json``.

    The VAE spatial downscale is ``2 ** (len(block_out_channels) - 1)`` (one
    stride-2 stage per extra block); ``latent_channels`` is the VAE latent depth.
    Both feed the DiT sequence-length derivation for sample_size-less configs.

    Args:
        model_path: Local diffusers model directory.

    Returns:
        ``(vae_scale_factor, latent_channels)``; ``vae_scale_factor`` defaults to
        8 (the standard SD/FLUX VAE) and ``latent_channels`` to 0 when unreadable.
    """
    try:
        cfg = json.loads((Path(model_path) / "vae" / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return 8, 0
    if not isinstance(cfg, dict):
        return 8, 0
    boc = cfg.get("block_out_channels")
    vae_scale = 2 ** (len(boc) - 1) if isinstance(boc, list) and boc else 8
    latent_channels = int(cfg.get("latent_channels") or 0)
    return vae_scale, latent_channels


def compute_diffusion_mem_img_per_sec(*, gpu_type: str, num_gpus: int, weight_bytes: int, num_steps: int) -> float:
    """Memory-roofline ceiling for diffusion image throughput (images/sec).

    Each denoising step must read the full model weights at least once, so the
    per-step time is lower-bounded by ``weight_bytes / (HBM_BW * num_gpus)`` and
    one image takes ``num_steps`` such steps. This is a strict upper bound on
    images/sec in the memory-bound regime. The compute-bound ceiling (which
    typically dominates for diffusion) needs per-op FLOPs modeling and is added
    separately once it can be validated against a real xDiT trace.

    Args:
        gpu_type: GPU type key for the HBM bandwidth lookup.
        num_gpus: Tensor/sequence-parallel degree (floored at 1).
        weight_bytes: Total model weight bytes (read once per step).
        num_steps: Number of denoising steps per image.

    Returns:
        The memory-bound images/sec ceiling, or ``0.0`` on unknown GPU type or
        degenerate input.
    """
    spec = HW_SPECS.get((gpu_type or "").strip().lower())
    if spec is None:
        return 0.0
    bw = spec["hbm_bw_gbps"] * 1e9 * max(num_gpus, 1)
    if weight_bytes <= 0 or num_steps <= 0 or bw <= 0:
        return 0.0
    per_step_s = weight_bytes / bw
    total_s = num_steps * per_step_s
    return 1.0 / total_s if total_s > 0 else 0.0


def _read_diffusion_dit_meta(model_path: str, *, height: int = 0, width: int = 0) -> tuple[int, int, int, int] | None:
    """DiT transformer shape from ``<model>/transformer/config.json`` for the
    compute-bound diffusion ceiling.

    Estimates DiT-ONLY params, NOT the total ``weight_bytes`` (which also holds
    the text encoder + VAE that do NOT run per denoising step). Per standard
    transformer block the count is ``12 * H**2`` (4H^2 attention + 8H^2 MLP);
    FLUX-style dual-stream blocks (``num_layers``) run separate image+text
    projections and are counted ``2x``, single-stream blocks (``num_single_layers``)
    ``1x`` -- so ``params = 12 * H**2 * (2 * num_layers + num_single_layers)``.

    Latent token count (the DiT sequence length):
      - ``sample_size`` present (Sana/PixArt/DiT):
        ``(sample_size / patch_h) * (sample_size / patch_w)``, where a
        2-element ``patch_size`` is ``(h, w)`` and need not be square.
      - else (FLUX/SD3): from the runtime resolution -- one token per
        ``vae_scale_factor * transformer_pack`` pixels a side, where the pack is
        inferred from ``in_channels / vae_latent_channels`` (FLUX folds a 2x2
        patch into channels), all read from the VAE + transformer configs.

    Args:
        model_path: Local diffusers model directory.
        height: Image height in pixels (used only when ``sample_size`` is absent).
        width: Image width in pixels (used only when ``sample_size`` is absent).

    Returns:
        ``(dit_params, latent_tokens, num_layers, hidden_size)`` where
        ``num_layers`` is the TOTAL block count (dual + single stream, for the
        attention term).

        ``latent_tokens == 0`` is an in-band sentinel for "the weights resolved
        but the sequence length did not", which today means a 3-element (video)
        ``patch_size`` this function has no frame axis for. Callers must keep
        using the other three fields in that case -- returning ``None`` instead
        would cost the DiT-only weight count, and
        ``_compute_diffusion_breakdown_from_state`` answers a missing one by
        falling its per-step memory bound back to the whole checkpoint, text
        encoder and VAE included. Only the compute half is unavailable.

        ``None`` when the config is unreadable, is missing ``num_layers`` or the
        hidden size, carries a non-numeric value where a number is required, or
        offers no way to size the latent grid (caller degrades to memory-only).
    """
    try:
        cfg = json.loads((Path(model_path) / "transformer" / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(cfg, dict):
        return None
    # Every field below is attacker-shaped JSON, so a non-numeric value must
    # degrade to "unreadable" rather than escape as TypeError past the caller.
    try:
        num_layers = int(cfg.get("num_layers") or cfg.get("num_hidden_layers") or 0)
        num_single_layers = int(cfg.get("num_single_layers") or 0)
        hidden = int(cfg.get("hidden_size") or cfg.get("inner_dim") or 0)
        if hidden <= 0:
            heads = int(cfg.get("num_attention_heads") or 0)
            head_dim = int(cfg.get("attention_head_dim") or cfg.get("head_dim") or 0)
            hidden = heads * head_dim
        _ps = cfg.get("patch_size")
        # A 3-element patch is (temporal, h, w): a video denoiser, whose token
        # count needs the frame axis nothing in this module models.
        video_patch = isinstance(_ps, (list, tuple)) and len(_ps) >= 3
        if isinstance(_ps, (list, tuple)):
            # A 2-element patch is (h, w) and need not be square, so keep both.
            patch_h = int((_ps[0] if _ps else 0) or 1) or 1
            patch_w = int((_ps[-1] if _ps else 0) or 1) or 1
        else:
            patch_h = patch_w = int(_ps or 1) or 1
        sample = int(cfg.get("sample_size") or 0)
    except (TypeError, ValueError):
        return None
    if num_layers <= 0 or hidden <= 0:
        return None

    if video_patch:
        # Only the compute side abstains. The weights are still resolvable, so the
        # caller keeps a DiT-only memory bound instead of the whole checkpoint's.
        latent_tokens = 0
    elif sample > 0:
        latent_tokens = (sample // patch_h) * (sample // patch_w)
    else:
        latent_tokens = _diffusion_latent_tokens_from_resolution(model_path, cfg, height, width)
    if latent_tokens <= 0 and not video_patch:
        return None

    total_layers = num_layers + num_single_layers
    param_layer_units = 2 * num_layers + num_single_layers if num_single_layers > 0 else num_layers
    dit_params = 12 * param_layer_units * hidden * hidden
    if dit_params <= 0:
        return None
    return dit_params, latent_tokens, total_layers, hidden


def _diffusion_latent_tokens_from_resolution(
    model_path: str, transformer_cfg: dict[str, Any], height: int, width: int
) -> int:
    """Latent-grid token count for a sample_size-less DiT from the runtime resolution.

    ``tokens = (H / downscale) * (W / downscale)`` where ``downscale =
    vae_scale_factor * transformer_pack``. The transformer pack is inferred from
    ``in_channels / vae_latent_channels`` (FLUX packs a 2x2 patch into channels),
    defaulting to 2 (the FLUX/SD3 convention) when it cannot be derived.

    Args:
        model_path: Local diffusers model directory (for the VAE geometry).
        transformer_cfg: Parsed transformer ``config.json``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        The latent token count, or ``0`` when the resolution is unavailable.
    """
    if height <= 0 or width <= 0:
        return 0
    vae_scale, latent_channels = _read_vae_geometry(model_path)
    in_ch = int(transformer_cfg.get("in_channels") or 0)
    pack = 2
    if in_ch > 0 and latent_channels > 0 and in_ch % latent_channels == 0:
        ratio = in_ch // latent_channels
        root = int(round(ratio**0.5))
        if root >= 1 and root * root == ratio:
            pack = root
    downscale = max(vae_scale, 1) * max(pack, 1)
    if downscale <= 0:
        return 0
    return (height // downscale) * (width // downscale)


def compute_diffusion_compute_img_per_sec(
    *,
    gpu_type: str,
    num_gpus: int,
    precision_tag: str,
    dit_params: int,
    latent_tokens: int,
    num_layers: int,
    hidden_size: int,
    num_steps: int,
) -> float:
    """Compute-roofline ceiling for diffusion image throughput (images/sec).

    Per denoising step the DiT does ``2 * dit_params * T`` linear FLOPs (the
    2N-per-token rule over ``T`` latent tokens) plus ``4 * L * T**2 * H``
    attention-score FLOPs (QK^T + A·V, weightless); one image is ``num_steps``
    such steps. ``F_peak`` is the max-achievable (sustained) TFLOPS — the SAME
    convention as the memory ceiling and the LLM ceiling (vendor peak only as a
    coverage-gap fallback). VAE decode (one-time) is intentionally excluded: a
    small, documented under-count that keeps this a valid upper bound.

    Args:
        gpu_type: GPU type key for the peak TFLOPS lookup.
        num_gpus: Tensor/sequence-parallel degree (floored at 1).
        precision_tag: Precision key for the peak TFLOPS lookup.
        dit_params: DiT-only parameter count (see :func:`_read_diffusion_dit_meta`).
        latent_tokens: Latent tokens processed per denoising step.
        num_layers: DiT transformer layers (for the attention-score term).
        hidden_size: DiT model dim (for the attention-score term).
        num_steps: Denoising steps per image.

    Returns:
        The compute-bound images/sec ceiling, or ``0.0`` on degenerate input.
    """
    peak_tflops = _resolve_achievable_tflops(gpu_type, precision_tag) or _resolve_peak_tflops(gpu_type, precision_tag)
    if peak_tflops <= 0 or dit_params <= 0 or latent_tokens <= 0 or num_steps <= 0:
        return 0.0
    linear = 2.0 * dit_params * latent_tokens
    attn = 4.0 * max(num_layers, 0) * (latent_tokens**2) * max(hidden_size, 0)
    flops_per_image = num_steps * (linear + attn)
    if flops_per_image <= 0:
        return 0.0
    peak_flops = peak_tflops * 1e12 * max(num_gpus, 1)
    return peak_flops / flops_per_image


def _compute_diffusion_breakdown_from_state(state: Any, runtime: RuntimeWorkload) -> RooflineBreakdown:
    """Diffusion (xDiT) roofline breakdown in images/sec.

    Ceiling = ``min(memory, compute)`` (the binding side), like the LLM path.
    Memory: per-step full-weight read. Compute: a config-analytical DiT FLOP
    model (see :func:`compute_diffusion_compute_img_per_sec`) — route-independent,
    so it tightens the (previously memory-only, too-loose) ceiling for every xDiT
    run. Degrades to memory-only when the DiT transformer config is unreadable.
    Values ride the ``*_tok_per_sec`` fields; the img/s unit is tagged at the
    snapshot layer (``throughput_unit``).

    Args:
        state: Shared run state (for the denoising step count).
        runtime: Resolved runtime workload (model_path / gpu_type / tp).

    Returns:
        The diffusion ``RooflineBreakdown``, or ``_EMPTY_BREAKDOWN`` when the
        model weights or step count are unavailable.
    """
    num_steps = _read_diffusion_num_steps(state)
    if num_steps <= 0:
        return _EMPTY_BREAKDOWN
    # DiT geometry drives both the compute FLOP model and per-step memory IO.
    # Read the runtime resolution so sample_size-less configs can size their grid.
    height, width = _read_diffusion_resolution(state)
    # ``model_path`` may be an HF repo id; resolve to the local diffusers dir so
    # the DiT/VAE config reads work (load_model_meta re-resolves internally, so
    # passing the resolved dir is harmless).
    from hyperloom.inference_optimizer.model_config_utils import (
        resolve_local_model_dir,
    )

    _resolved = resolve_local_model_dir(runtime.model_path)
    model_dir = str(_resolved) if _resolved is not None else runtime.model_path
    dit = _read_diffusion_dit_meta(model_dir, height=height, width=width)

    # Need at least one weight source: DiT geometry OR the full-checkpoint size;
    # bail only when both are missing.
    meta = load_model_meta(model_dir, precision_hint=runtime.precision)
    meta_bytes = int(meta.weight_bytes) if (meta is not None and meta.weight_bytes > 0) else 0
    if dit is None and meta_bytes <= 0:
        return _EMPTY_BREAKDOWN

    # Per step only the DiT runs, so per-step memory IO is the DiT-only weight
    # bytes; fall back to the full checkpoint when DiT geometry is unavailable.
    cmp_img_s = 0.0
    mem_bytes = meta_bytes
    if dit is not None:
        dit_params, latent_tokens, num_layers, hidden = dit
        dit_weight_bytes = int(dit_params * _resolve_dtype_bytes(runtime.precision or "bf16"))
        if dit_weight_bytes > 0:
            mem_bytes = dit_weight_bytes
        # ``latent_tokens == 0`` means the geometry read resolved the weights but
        # not the sequence length, so the memory bound above still holds.
        if latent_tokens > 0:
            cmp_img_s = compute_diffusion_compute_img_per_sec(
                gpu_type=runtime.gpu_type,
                num_gpus=runtime.tp,
                precision_tag=runtime.precision or "bf16",
                dit_params=dit_params,
                latent_tokens=latent_tokens,
                num_layers=num_layers,
                hidden_size=hidden,
                num_steps=num_steps,
            )
    mem_img_s = compute_diffusion_mem_img_per_sec(
        gpu_type=runtime.gpu_type,
        num_gpus=runtime.tp,
        weight_bytes=mem_bytes,
        num_steps=num_steps,
    )
    if mem_img_s <= 0 and cmp_img_s <= 0:
        return _EMPTY_BREAKDOWN
    peak, bound = select_peak_and_bound(mem_img_s, cmp_img_s)
    return RooflineBreakdown(mem_img_s, cmp_img_s, peak, bound)


def compute_roofline_breakdown_from_state(
    state: Any,
    *,
    arm: str | None = None,
) -> RooflineBreakdown:
    """Primary decode ceiling + T_mem/T_cmp side projections.

    Prefers the bottom-up PerfModel (``compute_roofline_from_perfmodel``) when
    the model config is complete, else the legacy top-down aggregate. Never
    raises; returns ``_EMPTY_BREAKDOWN`` on missing fields. ``arm`` pins
    precision to a specific arm.

    Args:
        state: Shared run state to resolve the workload and dtype from.
        arm: Pins precision to a specific arm; ``None`` infers it.

    Returns:
        The decode ``RooflineBreakdown`` (``_EMPTY_BREAKDOWN`` on missing
        fields).
    """
    runtime = resolve_runtime_workload(state, arm=arm)
    # Diffusion (xDiT) uses a distinct images/sec ceiling.
    if (runtime.framework or "").strip().lower() == "xdit":
        return _compute_diffusion_breakdown_from_state(state, runtime)
    meta = load_model_meta(
        runtime.model_path,
        precision_hint=runtime.precision,
    )
    if meta is None:
        return _EMPTY_BREAKDOWN
    gpu_type = runtime.gpu_type
    num_gpus = runtime.tp
    concurrency = runtime.concurrency
    # Rescale weights to the dtype the run actually read.
    rt = resolve_runtime_dtype(state, meta, arm=arm)
    meta = apply_runtime_dtype(meta, rt)
    precision_tag = rt.compute_precision_tag or runtime.precision or "bf16"
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
    peak, bound_kind = select_peak_and_bound(mem, cmp)
    legacy = RooflineBreakdown(mem, cmp, peak, bound_kind)

    # Prefer the bottom-up PerfModel peak; legacy is the fallback.
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


def compute_peak_from_state(state: Any, *, arm: str | None = None) -> float:
    """Convenience scalar wrapper for ``T_peak`` only (kept for backward compat; prefer ``compute_roofline_breakdown_from_state``). ``arm`` pins precision to a specific arm.

    Args:
        state: Shared run state to compute the ceiling from.
        arm: Pins precision to a specific arm; ``None`` infers it.

    Returns:
        The peak decode throughput (``peak_tok_per_sec``).
    """
    return compute_roofline_breakdown_from_state(state, arm=arm).peak_tok_per_sec


def read_baseline_server_args(state: Any) -> str:
    """Public accessor for the baseline arm's runtime server args.

    Stable entry point for callers (e.g. Coordinator profile injection) that
    must read baseline's own flags without depending on the private
    ``_read_baseline_yaml_server_args`` helper.

    Args:
        state: Shared run state carrying ``last_baseline`` provenance.

    Returns:
        The baseline arm's runtime server args, or ``""`` when unreadable.
    """
    return _read_baseline_yaml_server_args(state)


# TraceLens PerfModel per-op (bottom-up) roofline breakdown.

#: Max-achievable (sustained) TFLOPS from TraceLens arch JSON files (same HBM
#: bandwidth as HW_SPECS, empirically-measured max throughput instead of vendor
#: dense TFLOPS).
_MI300X_ACHIEVABLE_TFLOPS: dict[str, float] = {
    "bf16": 708.0,
    "bfloat16": 708.0,
    "fp16": 654.0,
    "float16": 654.0,
    "fp8": 1273.0,
    "float8_e4m3fn": 1273.0,
    "float8_e5m2": 1273.0,
    "fp32": 163.0,
    "float32": 163.0,
}
_MI325X_ACHIEVABLE_TFLOPS: dict[str, float] = {
    "bf16": 843.0,
    "bfloat16": 843.0,
    "fp16": 794.0,
    "float16": 794.0,
    "fp8": 1519.0,
    "float8_e4m3fn": 1519.0,
    "float8_e5m2": 1519.0,
    "fp32": 194.0,
    "float32": 194.0,
}
_MI355X_ACHIEVABLE_TFLOPS: dict[str, float] = {
    "bf16": 1686.0,
    "bfloat16": 1686.0,
    "fp16": 1686.0,
    "float16": 1686.0,
    "fp8": 3567.0,
    "float8_e4m3fn": 3567.0,
    "float8_e5m2": 3567.0,
    "mxfp4": 5663.0,
    "fp4": 5663.0,
    "float4": 5663.0,
    "fp32": 137.0,
    "float32": 137.0,
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
    return _resolve_tflops(HW_SPECS_ACHIEVABLE, gpu_type, precision_tag)


def resolve_compute_peak_provenance(gpu_type: str | None, precision_tag: str | None) -> dict[str, Any]:
    """Provenance for the compute-peak TFLOPS used by every compute ceiling.

    The unified convention is max-achievable (sustained) TFLOPS; the vendor
    dense peak is only a coverage-gap fallback. Surfacing convention + value +
    source keeps within%/gap interpretable.

    Args:
        gpu_type: GPU type key for the peak lookup.
        precision_tag: Precision key for the peak lookup.

    Returns:
        ``{compute_peak_convention, compute_peak_tflops, compute_peak_source}``.
    """
    ach = _resolve_achievable_tflops(gpu_type, precision_tag)
    if ach > 0:
        return {
            "compute_peak_convention": "achievable",
            "compute_peak_tflops": ach,
            "compute_peak_source": "TraceLens arch JSON (max-achievable sustained)",
        }
    vendor = _resolve_peak_tflops(gpu_type, precision_tag)
    return {
        "compute_peak_convention": "vendor" if vendor > 0 else "unknown",
        "compute_peak_tflops": vendor,
        "compute_peak_source": ("vendor dense peak (achievable-table miss fallback)" if vendor > 0 else "unavailable"),
    }


import dataclasses as _dc


def _fused_moe_flops(M: int, K: int, N: int, topk: int) -> float:
    """FLOPs for one gated SwiGLU MoE forward (gate+up+down projections).

    Mirrors TraceLens.PerfModel.extensions.moe_perf_model_extensions
    .FusedMoE.flops_func with gated=True (all LLM MoE uses SwiGLU):
      gate+up : 2 * M * K * N * topk * 2
      down    : 2 * M * K * N * topk
      aggregation: M * K * (2 * topk - 1)

    Args:
        M: Number of tokens (batch * seq).
        K: Hidden size.
        N: Per-expert FFN intermediate size.
        topk: Experts activated per token.

    Returns:
        Total FLOPs for the gated SwiGLU MoE forward.
    """
    return 2.0 * M * K * N * topk * 2 + 2.0 * M * K * N * topk + M * K * (2 * topk - 1)


def _fused_moe_bytes(
    M: int,
    K: int,
    N: int,
    num_experts: int,
    topk: int,
    weight_bpe: float,
    act_bpe: float | None = None,
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

    Args:
        M: Number of tokens (batch * seq).
        K: Hidden size.
        N: Per-expert FFN intermediate size.
        num_experts: Total experts.
        topk: Experts activated per token.
        weight_bpe: Weight bytes-per-element.
        act_bpe: Activation bytes-per-element; defaults to ``weight_bpe``.

    Returns:
        Total HBM bytes for the gated SwiGLU MoE forward.
    """
    if act_bpe is None:
        act_bpe = weight_bpe
    e_active = num_experts * (1.0 - ((num_experts - topk) / num_experts) ** M)
    return (
        M * K * act_bpe  # input read
        + e_active * N * K * weight_bpe * 2  # gate + up weight reads
        + e_active * N * K * weight_bpe  # down weight reads
        + M * K * act_bpe  # output write
    )


@_dc.dataclass(frozen=True)
class OpBreakdown:
    """Per-operator roofline result from TraceLens PerfModel."""

    name: str
    flops: float  # total FLOPs across all layers
    bytes_moved: float  # total bytes across all layers
    ai: float  # arithmetic intensity = flops / bytes_moved
    time_s: float  # roofline time (s) across all layers
    bound: str  # "compute" | "memory"
    pct_time: float  # fraction of total forward time (0–1)


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
    bound_kind: str  # "compute" | "memory" | "unknown"
    hbm_bw_gbps: float
    peak_achievable_tflops: float


def _gemm_flops(M: int, N: int, K: int) -> float:
    """FLOPs for a bias-free matrix multiply (2*M*N*K).

    Mirrors TraceLens.PerfModel.perf_model.GEMM.flops_func (no bias path
    needed here since LLM linear layers are weight-only without bias in
    the roofline model).

    Args:
        M: Rows of the output (tokens).
        N: Output features.
        K: Inner/contraction dimension.

    Returns:
        Total FLOPs for the matrix multiply.
    """
    return 2.0 * M * N * K


def _gemm_bytes(
    M: int,
    N: int,
    K: int,
    weight_bpe: float,
    act_bpe: float | None = None,
) -> float:
    """HBM bytes for a bias-free matmul: read(act MK + weight KN) + write(act MN).

    Mirrors TraceLens.PerfModel.perf_model.GEMM.bytes_func with bpe_mat1=act_bpe
    (input activation), bpe_mat2=weight_bpe (weight), bpe_output=act_bpe (output).
    act_bpe defaults to weight_bpe; for FP8/FP4-weight models pass act_bpe=2.0
    (bf16 activations) to correctly separate activation from weight bytes.

    Args:
        M: Rows of the output (tokens).
        N: Output features.
        K: Inner/contraction dimension.
        weight_bpe: Weight bytes-per-element.
        act_bpe: Activation bytes-per-element; defaults to ``weight_bpe``.

    Returns:
        Total HBM bytes for the matrix multiply.
    """
    a = act_bpe if act_bpe is not None else weight_bpe
    return M * K * a + K * N * weight_bpe + M * N * a


def _sdpa_flops(
    B: int,
    N_Q: int,
    H_Q: int,
    N_KV: int,
    H_KV: int,
    d_h_qk: int,
    d_h_v: int,
    causal: bool,
) -> float:
    """FLOPs for scaled dot-product attention.

    Mirrors TraceLens.PerfModel.perf_model.SDPA.flops_func:
      QK^T  : B * H_Q * 2 * N_Q * N_KV * d_h_qk
      PV    : B * H_Q * 2 * N_Q * d_h_v * N_KV
    Softmax FLOPs omitted (dominated by matmuls).
    Causal masking halves the work when N_Q == N_KV (prefill only).

    Args:
        B: Batch size.
        N_Q: Query sequence length.
        H_Q: Number of query heads.
        N_KV: Key/value sequence length.
        H_KV: Number of KV heads.
        d_h_qk: Per-head dimension for Q/K.
        d_h_v: Per-head dimension for V.
        causal: Whether causal masking applies.

    Returns:
        Total FLOPs for scaled dot-product attention.
    """
    flops_qk = B * H_Q * (2.0 * N_Q * N_KV * d_h_qk)
    flops_pv = B * H_Q * (2.0 * N_Q * d_h_v * N_KV)
    total = flops_qk + flops_pv
    if causal and N_Q == N_KV:
        total /= 2.0
    return total


def _sdpa_bytes(
    B: int,
    N_Q: int,
    H_Q: int,
    N_KV: int,
    H_KV: int,
    d_h_qk: int,
    d_h_v: int,
    causal: bool,
    bpe: float,
) -> float:
    """HBM bytes for SDPA: read Q, K, V + write output.

    Mirrors TraceLens.PerfModel.perf_model.SDPA.bytes_func.
    causal is accepted for API symmetry but does not change the I/O volume
    (KV is always fully read even under causal masking at the HBM level).

    Args:
        B: Batch size.
        N_Q: Query sequence length.
        H_Q: Number of query heads.
        N_KV: Key/value sequence length.
        H_KV: Number of KV heads.
        d_h_qk: Per-head dimension for Q/K.
        d_h_v: Per-head dimension for V.
        causal: Accepted for API symmetry; does not change I/O volume.
        bpe: Bytes per element for the tensors.

    Returns:
        Total HBM bytes for scaled dot-product attention.
    """
    elems = (
        B * N_Q * H_Q * d_h_qk  # Q read
        + B * N_KV * H_KV * d_h_qk  # K read
        + B * N_KV * H_KV * d_h_v  # V read
        + B * N_Q * H_Q * d_h_v  # output write
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

    The returned ceilings use max-achievable TFLOPS (from
    ``HW_SPECS_ACHIEVABLE``) rather than vendor-quoted peaks. The GEMM / SDPA
    formulas are inlined here (no TraceLens import) and mirror TraceLens PerfModel.

    Args:
        meta: Model metadata; complete config required (else ``None``).
        gpu_type: GPU type key for the achievable-spec lookup.
        concurrency: Decode batch size.
        isl: Input sequence length.
        osl: Output sequence length.
        num_gpus: Number of GPUs (tensor-parallel degree).
        precision_tag: Precision key for the achievable TFLOPS lookup.

    Returns:
        The per-op ``PerfModelBreakdown``, or ``None`` when model metadata is
        incomplete or the GPU is unsupported.
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
    # Routed-expert weights may be a distinct precision (e.g. DeepSeek-V4 fp4
    # experts under fp8 attention); size the MoE reads with it. 0 ⇒ same as bpe.
    expert_bpe = float(meta.expert_weight_dtype_bytes or bpe)
    # Activations (input/output) are at least bf16 even for quantized-weight models.
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

    # Linear projections per layer: (name, K_in, N_out, repeat_per_forward).
    # MoE FFN is handled separately in _forward.
    linears: list[tuple[str, int, int, int]] = [
        ("q_proj", hidden, q_out, n_layers),
        ("k_proj", hidden, kv_out, n_layers),
        ("v_proj", hidden, kv_out, n_layers),
        ("o_proj", q_out, hidden, n_layers),
    ]
    is_moe = meta.num_experts > 0 and meta.moe_intermediate_size > 0
    # A MoE checkpoint still runs a dense FFN in its ``first_k_dense_replace``
    # prefix, so the two are not mutually exclusive: gate/up/down repeat over
    # the dense layers and the MoE op over the MoE layers. Attention and SDPA
    # stay at n_layers -- every layer has those.
    if is_moe:
        n_moe_layers = meta.moe_layers or n_layers
        n_dense_ffn_layers = meta.dense_ffn_layers
    else:
        n_moe_layers = 0
        n_dense_ffn_layers = meta.dense_ffn_layers or n_layers
    if ffn and n_dense_ffn_layers > 0:
        linears += [
            ("gate_proj", hidden, ffn, n_dense_ffn_layers),
            ("up_proj", hidden, ffn, n_dense_ffn_layers),
            ("down_proj", ffn, hidden, n_dense_ffn_layers),
        ]
    if vocab:
        linears.append(("lm_head", hidden, vocab, 1))

    def _roofline_time(fl: float, by: float) -> tuple[float, str, float, float]:
        """Roofline time for one op given its FLOPs and byte traffic.

        Args:
            fl: Floating-point operations for the op.
            by: Bytes moved for the op.

        Returns:
            A tuple ``(time, bound_kind, t_mem, t_cmp)`` where ``time`` is the
            roofline max of compute and memory time and ``bound_kind`` is
            ``"compute"`` or ``"memory"``.
        """
        t_cmp = fl / f_peak if f_peak > 0 else 1e30
        t_mem = by / bw_bps if bw_bps > 0 else 1e30
        return max(t_cmp, t_mem), "compute" if t_cmp >= t_mem else "memory", t_mem, t_cmp

    def _forward(batch: int, s_q: int, s_kv: int) -> tuple[float, float, float, list[OpBreakdown]]:
        """Roofline-model one forward pass for the given shapes.

        Args:
            batch: Batch size (concurrency).
            s_q: Query sequence length.
            s_kv: Key/value sequence length.

        Returns:
            A tuple ``(total_time, total_mem_time, total_cmp_time, ops)`` where
            ``ops`` is the per-op breakdown list.
        """
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
            op_rows.append(
                OpBreakdown(
                    name=name,
                    flops=fl * rep,
                    bytes_moved=by * rep,
                    ai=(fl / by if by else 0.0),
                    time_s=t * rep,
                    bound=side,
                    pct_time=0.0,  # filled after total is known
                )
            )
        # MoE FFN: FusedMoE formula with batch-aware coupon E_active.
        if is_moe and n_moe_layers > 0:
            # Latent-MoE decoders run the experts below the residual width.
            moe_hidden = meta.moe_hidden_size or hidden
            fl_moe = _fused_moe_flops(M, moe_hidden, meta.moe_intermediate_size, meta.experts_per_tok)
            by_moe = _fused_moe_bytes(
                M,
                moe_hidden,
                meta.moe_intermediate_size,
                meta.num_experts,
                meta.experts_per_tok,
                expert_bpe,
                act_bpe,
            )
            t_moe, side_moe, t_mem_moe, t_cmp_moe = _roofline_time(fl_moe, by_moe)
            total_t += t_moe * n_moe_layers
            total_mem_t += t_mem_moe * n_moe_layers
            total_cmp_t += t_cmp_moe * n_moe_layers
            op_rows.append(
                OpBreakdown(
                    name="moe_fused",
                    flops=fl_moe * n_moe_layers,
                    bytes_moved=by_moe * n_moe_layers,
                    ai=(fl_moe / by_moe if by_moe else 0.0),
                    time_s=t_moe * n_moe_layers,
                    bound=side_moe,
                    pct_time=0.0,
                )
            )
        # SDPA
        causal = s_q == s_kv
        fl_s = _sdpa_flops(batch, s_q, n_q_heads, s_kv, n_kv_heads, hd, hd, causal)
        by_s = _sdpa_bytes(batch, s_q, n_q_heads, s_kv, n_kv_heads, hd, hd, causal, act_bpe)
        t_s, side_s, t_mem_s, t_cmp_s = _roofline_time(fl_s, by_s)
        total_t += t_s * n_layers
        total_mem_t += t_mem_s * n_layers
        total_cmp_t += t_cmp_s * n_layers
        op_rows.append(
            OpBreakdown(
                name="sdpa",
                flops=fl_s * n_layers,
                bytes_moved=by_s * n_layers,
                ai=(fl_s / by_s if by_s else 0.0),
                time_s=t_s * n_layers,
                bound=side_s,
                pct_time=0.0,
            )
        )
        # fill pct_time
        if total_t > 0:
            op_rows = [_dc.replace(op, pct_time=op.time_s / total_t) for op in op_rows]
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
