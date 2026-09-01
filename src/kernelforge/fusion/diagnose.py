# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 1: diagnose whether a decode trace is launch-bound (a fusion candidate).

Self-contained (no Hyperloom / KB dependency): reads a Chrome/kineto torch-profiler
trace, categorizes each GPU kernel by name (model-agnostic ROCm/HIP + PyTorch
naming rules), and decides whether the launch-bound op categories dominate enough
GPU-busy time -- while the GPU idles most of the wall -- to be worth fusing.

The diagnosis encodes the reusable lever behind the proven ZAYA/LFM2 wins: the
decode path is dominated by many tiny fp32 elementwise/reduce/cast/norm kernels
(launch/dispatch bound), so collapsing those chains into single kernels is the win.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

from .calibration import (
    DEFAULT_MIN_PREDICTED_GAIN,
    predict_cuda_graph_on_gain,
)
from .models import Diagnosis

# Launch-bound categories: tiny fp32 ops whose per-launch overhead dominates a
# dispatch-bound decode path. Fusing these is the lever. Kept in sync with the
# category names emitted by :func:`categorize_kernel_name`.
LAUNCH_BOUND_CATEGORIES: frozenset[str] = frozenset(
    {
        "elementwise",
        "copy",
        "reduce",
        "cast",
        "rmsnorm",
        "layernorm",
        "rope",
        "add",
        "mul",
        "activation",
    }
)

# Calibration finding (5 measured models, kernel/docs/fusion_calibration.json):
# launch_bound_share is a POOR discriminator of real cg-ON gain -- GraniteMoE has
# the BEST measured gain (+5.32%) yet a LOW share (0.17), because its big MoE
# GEMM/expert kernels dilute the launch-bound share. A share>=0.25 gate wrongly
# rejected it, so the share gate is only a soft "some launch-bound present" floor.
#
# busy_fraction_of_wall separated those 5 models cleanly (with gain: 0.12-0.29;
# without: 0.44-0.54) and was once the primary gate. It no longer rejects anything:
# GEMM-bound Qwen3-14B and 32B sit above 0.45 yet measured +6.2% and +3.1% E2E from
# decode fusions, so the threshold was discarding real opportunities sight-unseen.
# It is now only a ranking annotation surfaced in Diagnosis.reason.
DEFAULT_MIN_LAUNCH_BOUND_SHARE = 0.10
DEFAULT_MAX_BUSY_WALL = 0.45

# Ordered (first-match-wins) kernel-name -> category rules. Compute-bound buckets
# (gemm/attention/conv/moe) are matched first so a fused kernel whose name also
# mentions a launch-bound op (e.g. ``add_rmsnorm``) is not misfiled.
#
# The original alternations were written against torch-eager naming
# (``CUDAFunctor_add``, ``at::native::...``), where ``\bmul\b``/``\badd\b`` fire.
# AITER/vLLM fused kernels are snake_case, and ``_`` is a regex word char, so
# ``_act_mul_``, ``_fused_rms_``, ``_..._quant_kernel`` matched nothing and fell to
# ``other``. On Qwen3-14B-FP8 that buried 11.9% of GPU time (act_mul+rms+quant) and
# pushed launch_bound_share to 0.083, below the 0.10 floor -- the FP8 variant of a
# model whose BF16 form is a measured +6.2% fusion win (see 276aacf6). The
# The same word-boundary flaw ran the other way in the pre-existing ``gemm``
# rule: ``\bgemm\b`` matched none of ``_batched_gemm_a8w8_...``, ``bf16gemm_...``,
# ``deepgemm``, or ck_tile's ``QuantGemmKernel``, so a bare ``gemm`` replaces it.
# That makes ``gemm`` overlap ``moe``, whose kernels are GEMMs the table reports
# separately, so ``moe`` now precedes it -- and ``gemm`` precedes the quant rules,
# without which ``_batched_gemm_..._quant_kernel`` (3.7% of GPU time on a
# GLM-5.2-MXFP4 trace) is filed as ``cast`` and wrongly counted as launch-bound.
_KERNEL_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    # MoE first: its kernels are GEMMs too, and the table reports them separately.
    ("moe", r"fused_moe|mfma_moe|moe_align|moe_sum|moe_reduction|_routing|expert|grouped_topk"),
    ("gemm", r"cijk_|tensile|gemm|matmul|_bhs_"),
    ("attention", r"paged_attention|flash|fmha|\bmha\b|attention|attn_|_fwd_kernel|_fwd_grouped|mla_|_mla"),
    ("conv", r"conv1d|_conv_|\bconv\b"),
    ("rmsnorm", r"rmsnorm|rms_norm|_rms_"),
    ("layernorm", r"layernorm|layer_norm"),
    ("rope", r"rotary|\brope\b"),
    ("activation", r"silu|gelu|\brelu\b|sigmoid|activation|act_mul|_act_"),
    ("cast", r"tofloat|tohalf|_cast|convert|dtype_|scaled_quant|_quant_kernel"),
    ("copy", r"kvcache|memcpy|\bcopy\b|indexselect|index_select|gather|scatter"),
    ("reduce", r"reduce|rocprim|trampoline|\bsum\b|\bmean\b"),
    ("sample", r"sample"),
    # Match multiply BEFORE the generic add/binaryfunctor rule: a BinaryFunctor
    # doing a multiply (e.g. SwiGLU's ``silu(gate) * up``) would otherwise be
    # misfiled as ``add`` and lost from the ``mul`` trigger of the swiglu pattern.
    ("mul", r"\bmul\b|multiply|cudafunctor_mul|binaryfunctor.*mul|mul.*binaryfunctor"),
    ("add", r"cudafunctor_add|cudafunctoronself_add|binaryfunctor|\badd\b"),
    ("elementwise", r"elementwise|multi_tensor|arange|clamp|\bfill\b|gpu_index|triton|kv_indices"),
)


def categorize_kernel_name(name: str) -> str:
    """Map a GPU kernel name to a coarse op category (model-agnostic)."""
    s = (name or "").lower()
    for category, pattern in _KERNEL_CATEGORY_RULES:
        if re.search(pattern, s):
            return category
    return "other"


# ``elementwise`` is the catch-all bucket, and its pattern keys on kernel-naming
# artifacts (``triton``, ``gpu_index``, ``kv_indices``) rather than on a named
# op. Prose describing a Triton fusion mentions "triton" almost every time, so
# including it here would tag nearly every description and destroy the very
# distinctions this function exists to draw.
_PROSE_EXCLUDED_CATEGORIES: frozenset[str] = frozenset({"elementwise"})


def categories_in_text(text: str) -> list[str]:
    """Every op category a free-text description mentions, sorted and de-duped.

    :func:`categorize_kernel_name` classifies ONE kernel and stops at the first
    match. A fusion is defined by the SET of ops it folds together, so this
    collects every category the text mentions instead.

    The vocabulary is fixed and model-agnostic, which is what makes it usable as
    an identity: two independent runs describing the same fusion in different
    words still produce the same set.
    """
    s = (text or "").lower()
    return sorted(
        category
        for category, pattern in _KERNEL_CATEGORY_RULES
        if category not in _PROSE_EXCLUDED_CATEGORIES and re.search(pattern, s)
    )


def load_op_busy_from_kineto_trace(
    path: str | Path,
) -> tuple[dict[str, float], float | None, float]:
    """Extract per-category busy shares from a kineto/torch-profiler trace.

    Reads GPU kernel events (``cat == "kernel"``) from a ``*.trace.json[.gz]``
    (produced by e.g. ``sglang.bench_one_batch --profile --profile-stage decode``,
    run with CUDA graphs disabled so individual kernels are visible), categorizes
    each by name, and returns ``(category->share, busy_of_wall, kernels_total)``.

    Malformed/missing traces yield ``({}, None, 0.0)`` so callers skip cleanly.
    """
    p = Path(path)
    try:
        if p.suffix == ".gz" or p.name.endswith(".json.gz"):
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, None, 0.0

    events = data.get("traceEvents") if isinstance(data, dict) else None
    if not isinstance(events, list):
        return {}, None, 0.0

    busy_by_cat: dict[str, float] = {}
    total = 0.0
    n_kernels = 0
    first_ts: float | None = None
    last_end: float | None = None
    for ev in events:
        if not isinstance(ev, dict) or ev.get("cat") != "kernel":
            continue
        try:
            dur_f = float(ev.get("dur"))
        except (TypeError, ValueError):
            continue
        if dur_f <= 0:
            continue
        cat = categorize_kernel_name(str(ev.get("name") or ""))
        busy_by_cat[cat] = busy_by_cat.get(cat, 0.0) + dur_f
        total += dur_f
        n_kernels += 1
        try:
            ts_f = float(ev.get("ts"))
        except (TypeError, ValueError):
            ts_f = None
        if ts_f is not None:
            first_ts = ts_f if first_ts is None else min(first_ts, ts_f)
            last_end = ts_f + dur_f if last_end is None else max(last_end, ts_f + dur_f)

    if total <= 0:
        return {}, None, 0.0
    shares = {k: v / total for k, v in busy_by_cat.items()}
    busy_of_wall: float | None = None
    if first_ts is not None and last_end is not None and last_end > first_ts:
        # Clamp to [0, 1]: summing per-kernel durations overcounts busy time when
        # kernels overlap across concurrent streams, which could otherwise push a
        # launch-bound decode past the compute-bound gate and hide the opportunity.
        busy_of_wall = min(1.0, total / (last_end - first_ts))
    return shares, busy_of_wall, float(n_kernels)


# dtype -> bytes/element, keyed by the strings kineto writes into an op's
# ``args["Input type"]`` (torch scalar-type names + a few C++ aliases). Unknown /
# non-tensor entries contribute 0 bytes (they are scalars or metadata).
_DTYPE_BYTES: dict[str, int] = {
    "float": 4,
    "float32": 4,
    "f32": 4,
    "double": 8,
    "float64": 8,
    "half": 2,
    "float16": 2,
    "f16": 2,
    "c10::half": 2,
    "bfloat16": 2,
    "c10::bfloat16": 2,
    "bf16": 2,
    "long": 8,
    "int64": 8,
    "long int": 8,
    "int": 4,
    "int32": 4,
    "short": 2,
    "int16": 2,
    "char": 1,
    "int8": 1,
    "signed char": 1,
    "unsigned char": 1,
    "byte": 1,
    "uint8": 1,
    "bool": 1,
    "c10::float8_e4m3fn": 1,
    "c10::float8_e5m2": 1,
    "float8": 1,
    "fp8": 1,
}


def _dtype_bytes(dtype: str) -> int:
    """Bytes/element for a kineto ``Input type`` string; 0 when unknown/non-tensor."""
    return _DTYPE_BYTES.get(str(dtype or "").strip().lower(), 0)


def _tensor_bytes(dims: Any, dtype: str) -> float:
    """Bytes of one tensor arg = prod(dims) * dtype_size. 0 for scalars/unknown."""
    esize = _dtype_bytes(dtype)
    if esize <= 0 or not isinstance(dims, (list, tuple)) or not dims:
        return 0.0
    n = 1
    for d in dims:
        try:
            n *= int(d)
        except (TypeError, ValueError):
            return 0.0
    return float(n) * esize


def load_op_bytes_from_kineto_trace(path: str | Path) -> dict[str, float]:
    """MEASURED per-category memory-traffic shares from a kineto trace's op shapes.

    The GPU ``kernel`` events carry no shapes, but the CPU ``cpu_op`` events do
    (``args["Input Dims"]`` + ``args["Input type"]``, plus ``Output dims`` /
    ``Output type`` when present). For each op we sum input+output tensor bytes
    (the simplest correct memory-traffic proxy), categorize it with the SAME
    :func:`categorize_kernel_name` used for launch shares, and return a
    ``category -> fraction-of-total-bytes`` map.

    This is the memory channel that complements the launch-time discount: under
    CUDA-graph-ON the launch overhead is already gone, so the surviving fusion
    headroom is the HBM round-trips saved, which is proportional to these bytes.

    Returns ``{}`` when the trace is unreadable OR carries no op shape info (older
    /graph-on traces) -- callers then fall back to the launch-share discount.
    """
    p = Path(path)
    try:
        if p.suffix == ".gz" or p.name.endswith(".json.gz"):
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    events = data.get("traceEvents") if isinstance(data, dict) else None
    if not isinstance(events, list):
        return {}

    bytes_by_cat: dict[str, float] = {}
    total = 0.0
    for ev in events:
        if not isinstance(ev, dict) or ev.get("cat") != "cpu_op":
            continue
        args = ev.get("args")
        if not isinstance(args, dict):
            continue
        in_dims = args.get("Input Dims") or args.get("Input dims")
        in_types = args.get("Input type") or args.get("Input types")
        b = 0.0
        if isinstance(in_dims, list) and isinstance(in_types, list):
            for dims, dt in zip(in_dims, in_types):
                b += _tensor_bytes(dims, dt)
        out_dims = args.get("Output dims") or args.get("Output Dims")
        out_types = args.get("Output type") or args.get("Output types")
        if isinstance(out_dims, list) and isinstance(out_types, list):
            for dims, dt in zip(out_dims, out_types):
                b += _tensor_bytes(dims, dt)
        if b <= 0:
            continue
        cat = categorize_kernel_name(str(ev.get("name") or ""))
        bytes_by_cat[cat] = bytes_by_cat.get(cat, 0.0) + b
        total += b

    if total <= 0:
        return {}
    return {k: v / total for k, v in bytes_by_cat.items()}


def diagnose_from_shares(
    category_shares: dict[str, float],
    *,
    busy_fraction_of_wall: float | None,
    kernels_per_step: float = 0.0,
    decode_batch: int = 16,
    category_bytes_share: dict[str, float] | None = None,
    min_launch_bound_share: float = DEFAULT_MIN_LAUNCH_BOUND_SHARE,
    max_busy_wall: float = DEFAULT_MAX_BUSY_WALL,
    min_predicted_gain: float = DEFAULT_MIN_PREDICTED_GAIN,
) -> Diagnosis:
    """Turn category busy-shares into a fusion-candidate verdict.

    The only hard entry gate is a soft launch-bound share FLOOR: some fusible ops
    must be present at all. Both ``busy_fraction_of_wall`` and
    ``predicted_e2e_gain`` are computed and annotated for ranking but are NOT
    vetoes, because both were shown to reject real wins. The share-derived
    prediction under-predicts low-share/high-gain MoE (GraniteMoE) and
    over-predicts high-share/no-gain cases; the busy-of-wall heuristic rejected
    GEMM-bound Qwen3-14B/32B, which measured +6.2% and +3.1% end to end. Some
    low-gain causes (a framework fused op being CUDA-only) are not statically
    visible at all. The downstream validate/loop measures the real speedup and is
    the true filter.
    """
    shares = {str(k).strip().lower(): float(v) for k, v in (category_shares or {}).items() if v is not None}
    bytes_share = {str(k).strip().lower(): float(v) for k, v in (category_bytes_share or {}).items() if v is not None}
    lb_share = sum(v for k, v in shares.items() if k in LAUNCH_BOUND_CATEGORIES)
    # When the trace exposed op shapes, ground the predicted cg-ON gain in the
    # MEASURED launch-bound memory-traffic share; otherwise fall back to the flat
    # launch-share discount (mem_share=None keeps the legacy behavior).
    lb_mem_share = sum(v for k, v in bytes_share.items() if k in LAUNCH_BOUND_CATEGORIES) if bytes_share else None
    predicted = predict_cuda_graph_on_gain(lb_share, decode_batch=decode_batch, mem_share=lb_mem_share)
    dominant = [
        k
        for k, _ in sorted(
            ((k, v) for k, v in shares.items() if k in LAUNCH_BOUND_CATEGORIES and v > 0),
            key=lambda kv: kv[1],
            reverse=True,
        )
    ]

    def _diag(is_candidate: bool, reason: str) -> Diagnosis:
        return Diagnosis(
            lb_share,
            busy_fraction_of_wall,
            dominant,
            kernels_per_step,
            shares,
            is_candidate,
            reason,
            predicted_e2e_gain=predicted,
            category_bytes_share=bytes_share,
        )

    if not shares:
        return _diag(False, "empty_trace")
    if lb_share < min_launch_bound_share:
        return _diag(
            False,
            f"launch_bound_share {lb_share:.3f} < {min_launch_bound_share} (compute/attention/moe dominated)",
        )
    # busy_fraction_of_wall is annotated but NOT a hard veto. The 0.45 threshold
    # was calibrated on 5 models, and measured counter-examples exist: GEMM-bound
    # Qwen3-14B and 32B are well above it yet still gained +6.2% and +3.1% end to
    # end from decode fusions. Rejecting on this alone discarded real wins before
    # anything was measured, so it is reported for ranking instead.
    busy_note = ""
    if busy_fraction_of_wall is not None and busy_fraction_of_wall > max_busy_wall:
        busy_note = (
            f" (GPU busy {busy_fraction_of_wall:.2f} > {max_busy_wall} of wall: "
            f"compute-bound, so expect a smaller share of time to be fusible; "
            f"annotated for ranking, validate/loop will confirm)"
        )
    # predicted_e2e_gain is annotated (for ranking / manifest) but NOT a hard veto;
    # the downstream validate/loop is the true gain filter. Surface a low prediction
    # in the reason string so it is visible without silently rejecting real wins.
    if predicted < min_predicted_gain:
        return _diag(
            True,
            f"launch_bound_dispatch_bound (predicted cg-ON gain {predicted:.3f} is "
            f"below {min_predicted_gain}; share-derived prediction is unreliable, "
            f"validate/loop will confirm){busy_note}",
        )
    return _diag(True, f"launch_bound_elementwise_dominant{busy_note}")


def diagnose_trace(
    trace_path: str | Path,
    *,
    decode_steps: int = 0,
    **kwargs: Any,
) -> Diagnosis:
    """Full stage-1 entry point: categorize a trace and return the verdict.

    Args:
        trace_path: Path to the kineto ``*.trace.json[.gz]``.
        decode_steps: Number of decode steps captured (to normalize
            kernels_per_step); ``0`` leaves it as the raw kernel count.
    """
    # Distinguish a missing/unreadable trace (user error) from a present trace
    # that is simply not launch-bound, so the verdict reason is actionable.
    if not Path(trace_path).is_file():
        return Diagnosis(0.0, None, [], 0.0, {}, False, f"trace_unreadable: file not found: {trace_path}")
    shares, busy, n_kernels = load_op_busy_from_kineto_trace(trace_path)
    bytes_share = load_op_bytes_from_kineto_trace(trace_path)
    kps = n_kernels / decode_steps if decode_steps > 0 else n_kernels
    return diagnose_from_shares(
        shares, busy_fraction_of_wall=busy, kernels_per_step=kps, category_bytes_share=bytes_share, **kwargs
    )
