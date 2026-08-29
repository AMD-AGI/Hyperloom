###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Producer for the model-agnostic ``TraceShapeManifest`` (P0-A / WP-1).

The manifest is the frozen contract consumed by the Trace->CSV tuning loop
(KernelForge ``kernelforge.gemm_tune``). Unlike the existing hot-kernel candidate
lists (which *collapse* dtype/shape variants and *discard* CUDA-graph capture
shards), this producer keeps a **variant-discriminating signature** per row and
weights each row by its steady-state replay time.

Design (frozen decisions, 2026-07-23):

* **graph_variant / node_ordinal** -- for a CUDA-graph-on run, a variant is one
  captured graph, identified by the ``bs_<batch>`` capture shard it was recorded
  in; ``node_ordinal`` is the launch order inside that shard. For an eager run
  (no capture) the variant is ``"eager"`` and ordinal is the steady-window launch
  order.
* **Dual coverage tag** -- every row carries ``is_gemm`` (all GEMM-family) and
  ``is_target_gemm`` (the tuner-addressable subset). Coverage/gating denominators
  use ``is_target_gemm``; reports may also use ``is_gemm`` for the global share.
* **capture_only weighting** -- capture shards only observe each internal kernel
  once, so capture-derived rows are marked ``capture_only=True`` and their
  ``cum_gpu_us`` is the single capture-time cost. The steady replay multiplier is
  recorded per variant in ``workload.variant_steady_replay`` (from the main
  trace's graph-launch count) and is left ``null`` when it cannot be attributed
  to a specific variant -- the producer never fabricates steady per-node time.

This module is pure: it operates on already-parsed ``analyze_trace`` result
dicts and returns a manifest dict. All file I/O and env gating live in the
calling tool (``bypass_trace_analysis.py``). It has no GPU/serving dependency
and is unit-testable with synthetic inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

#: Manifest schema version. Bump on any breaking change to row/field semantics.
SCHEMA_VERSION = 1

MANIFEST_KIND = "trace_shape_manifest"

#: Variant label used when a run has no CUDA-graph capture shards (eager mode).
EAGER_VARIANT = "eager"

# --- op taxonomy -------------------------------------------------------------
# Ordered (first match wins). Kept intentionally small and additive; the
# consumer refines tuner-addressability. Names are lower-cased before matching.
_OP_RULES: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("moe", re.compile(r"moe|fmoe|expert|grouped.*(gemm|mm)|group_gemm")),
    ("gemm", re.compile(r"gemm|matmul|hipblas|cutlass|tensile|\bmm\b|_mm_|linear|wgrad|dgemm|sgemm")),
    ("attention", re.compile(r"attention|flash|fmha|\battn\b|mla|paged")),
    ("rope", re.compile(r"rope|rotary")),
    ("norm", re.compile(r"rmsnorm|rms_norm|layernorm|layer_norm|norm")),
    (
        "elementwise",
        re.compile(r"elementwise|activation|silu|gelu|swiglu|\badd\b|\bmul\b|cast|copy|convert|quant|dequant"),
    ),
    ("reduce", re.compile(r"reduce|softmax|topk|argmax|sum\b")),
    ("comm", re.compile(r"all_reduce|allreduce|all_gather|allgather|reduce_scatter|nccl|rccl")),
)

#: GEMM-family ops -> ``is_gemm`` (broad coverage denominator).
_GEMM_FAMILY = frozenset({"gemm", "moe"})

#: dtype tokens a kernelforge.gemm_tune tuner can address today (best-effort match on
#: the Kineto ``Input type`` strings). Refined further on the consumer side.
_TUNER_DTYPE_RE = re.compile(
    r"bf16|bfloat16|fp16|float16|half|fp8|float8|e4m3|e5m2|fp4|float4|f8|f4",
    re.IGNORECASE,
)

#: Graph-replay launch marker in the main (graph-on) trace.
_GRAPH_LAUNCH_RE = re.compile(r"graphlaunch|graph_launch|hipgraphlaunch|cudagraphlaunch", re.IGNORECASE)


def classify_op(name: str, op_name: str = "") -> str:
    """Return the coarse op category for a kernel/op name pair.

    Matches the launching op name first (more semantic), then the device kernel
    name. Returns ``"other"`` when nothing matches.
    """
    hay = f"{op_name or ''} {name or ''}".lower()
    for label, rule in _OP_RULES:
        if rule.search(hay):
            return label
    return "other"


def _is_gemm(op: str) -> bool:
    """Whether an op category counts toward the broad ``is_gemm`` denominator."""
    return op in _GEMM_FAMILY


def _tuner_addressable(dtypes: Any) -> bool:
    """Best-effort check that at least one input dtype maps to a known tuner.

    ``dtypes`` is the Kineto ``Input type`` payload (list of strings or a
    string). Empty/unknown dtype -> not addressable (an unmatched shape is
    treated as uncovered, mirroring AITER exact-match semantics).
    """
    if not dtypes:
        return False
    if isinstance(dtypes, (list, tuple)):
        text = " ".join(str(d) for d in dtypes)
    else:
        text = str(dtypes)
    return _TUNER_DTYPE_RE.search(text) is not None


def _canon_dims(shapes: Any) -> dict[str, Any]:
    """Best-effort extraction of GEMM dims from Kineto ``Input Dims``.

    The first 2-D operand is the activation ``A = [M, K]``. The second 2-D
    operand is the weight, which may be stored either ``[N, K]`` (inference
    layout, e.g. ``aiter::gemm_a8w8_blockscale_ck``) or ``[K, N]`` (plain
    ``torch.mm``). We pick ``N`` as the weight dim that is *not* ``K`` (verified
    against ``A``), which is correct for both conventions; a square weight is
    unambiguous. Scale operands (``[M, K/128]``, ``[N/128, K/128]``) follow the
    weight in the operand list, so taking the first two 2-D operands avoids them.

    Leaves fields ``None`` and preserves the raw shapes on anything unexpected.
    Never raises.
    """
    dims: dict[str, Any] = {"M": None, "N": None, "K": None, "batch": None, "groups": None, "raw": shapes or []}
    try:
        mats = [s for s in (shapes or []) if isinstance(s, (list, tuple)) and len(s) >= 2]
        two_d = [s for s in mats if len(s) == 2]
        if two_d:
            a = two_d[0]
            dims["M"], dims["K"] = int(a[0]), int(a[1])
            k = dims["K"]
            if len(two_d) >= 2:
                b = [int(b0) for b0 in two_d[1][:2]]
                if b[1] == k:  # weight [N, K] (inference layout)
                    dims["N"] = b[0]
                elif b[0] == k:  # weight [K, N] (plain torch.mm)
                    dims["N"] = b[1]
                else:  # unknown layout: generic [K, N] fallback
                    dims["N"] = b[1]
        batched = [s for s in mats if len(s) >= 3]
        if batched:
            dims["batch"] = int(batched[0][0])
    except (TypeError, ValueError, IndexError):
        pass
    return dims


def _signature_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Return only the variant-discriminating fields used for the dedup key."""
    d = row.get("dims") or {}
    return {
        "graph_variant": row.get("graph_variant"),
        "node_ordinal": row.get("node_ordinal"),
        "op": row.get("op"),
        "backend": row.get("backend"),
        "in_dtype": row.get("in_dtype"),
        "out_dtype": row.get("out_dtype"),
        "quant": row.get("quant"),
        "M": d.get("M"),
        "N": d.get("N"),
        "K": d.get("K"),
        "batch": d.get("batch"),
        "groups": d.get("groups"),
        "transpose": row.get("transpose"),
        "layout": row.get("layout"),
        "block_size": row.get("block_size"),
        "scale_layout": row.get("scale_layout"),
        "epilogue": row.get("epilogue"),
        "phase": row.get("phase"),
        "bucket": row.get("bucket"),
    }


def signature_key(row: dict[str, Any]) -> str:
    """Deterministic sha256 over the discriminating tuple (see frozen schema).

    Two rows sharing the same math shape but differing on graph_variant / layout
    / quant / epilogue / phase / bucket get *different* keys and are never
    merged.
    """
    payload = json.dumps(_signature_fields(row), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dtype_tokens(dtypes: Any) -> tuple[str, str]:
    """Return (in_dtype, out_dtype) best-effort from a Kineto ``Input type``.

    First token is the input dtype; output dtype is unknown from inputs alone so
    it mirrors the input (best-effort) unless a distinct trailing token exists.
    """
    if not dtypes:
        return ("", "")
    if isinstance(dtypes, (list, tuple)):
        toks = [str(d) for d in dtypes if str(d)]
    else:
        toks = [t for t in str(dtypes).replace(",", " ").split() if t]
    if not toks:
        return ("", "")
    return (toks[0], toks[-1])


def build_row(
    launch: dict[str, Any],
    *,
    graph_variant: str,
    node_ordinal: int,
    phase: str,
    bucket: str,
    capture_only: bool,
) -> dict[str, Any]:
    """Build one manifest row from an enriched ``kernel_launches`` record.

    ``launch`` is expected to carry ``name``/``op_name``/``dur`` and, when the
    reader enrichment is present, ``shapes``/``dtypes``/``kernel_file``/
    ``kernel_backend``. Missing enrichment degrades to empty signature fields
    (row still produced, just coarser).
    """
    name = launch.get("name", "") or ""
    op_name = launch.get("op_name", "") or ""
    op = classify_op(name, op_name)
    dtypes = launch.get("dtypes")
    in_dtype, out_dtype = _dtype_tokens(dtypes)
    dims = _canon_dims(launch.get("shapes"))
    is_gemm = _is_gemm(op)
    is_target = is_gemm and _tuner_addressable(dtypes)
    dur = float(launch.get("dur", 0.0) or 0.0)
    row: dict[str, Any] = {
        "graph_variant": graph_variant,
        "node_ordinal": node_ordinal,
        "op": op,
        "backend": launch.get("kernel_backend", "") or "",
        "in_dtype": in_dtype,
        "out_dtype": out_dtype,
        "quant": launch.get("quant", "") or "",
        "dims": {k: dims.get(k) for k in ("M", "N", "K", "batch", "groups")},
        "transpose": launch.get("transpose"),
        "layout": launch.get("layout"),
        "block_size": launch.get("block_size"),
        "scale_layout": launch.get("scale_layout"),
        "epilogue": launch.get("epilogue"),
        "phase": phase,
        "bucket": bucket,
        # measurement
        "replay_count": 1,
        "cum_gpu_us": round(dur, 3),
        "mean_gpu_us": round(dur, 3),
        "is_gemm": is_gemm,
        "is_target_gemm": is_target,
        "capture_only": capture_only,
        # provenance / source
        "kernel_symbol": name,
        "source": {
            "kernel_symbol": name,
            "kernel_file": launch.get("kernel_file", "") or "",
            "library": launch.get("kernel_backend", "") or "",
            "op_name": op_name,
            "raw_shapes": (launch.get("shapes") or []),
            "raw_dtypes": dtypes or [],
        },
    }
    row["signature_key"] = signature_key(row)
    return row


def build_variant_rows(
    *,
    graph_variant: str,
    analysis: dict[str, Any],
    phase: str,
    bucket: str,
    capture_only: bool,
) -> list[dict[str, Any]]:
    """Build variant-discriminating rows from one trace ``analyze_trace`` result.

    Iterates the time-ordered ``kernel_launches`` (requires the reader was called
    with ``emit_launches=True``) and emits one row per launch, with
    ``node_ordinal`` assigned by launch order. Because ``node_ordinal``
    participates in ``signature_key``, rows are never merged and ``replay_count``
    is always 1; the steady replay multiplier is recorded separately in
    ``workload.variant_steady_replay``.
    """
    launches = sorted(
        (analysis.get("kernel_launches") or []),
        key=lambda r: float(r.get("ts", 0.0) or 0.0),
    )
    by_sig: dict[str, dict[str, Any]] = {}
    for ordinal, launch in enumerate(launches):
        row = build_row(
            launch,
            graph_variant=graph_variant,
            node_ordinal=ordinal,
            phase=phase,
            bucket=bucket,
            capture_only=capture_only,
        )
        key = row["signature_key"]
        existing = by_sig.get(key)
        if existing is None:
            by_sig[key] = row
        else:
            existing["replay_count"] += 1
            existing["cum_gpu_us"] = round(existing["cum_gpu_us"] + row["cum_gpu_us"], 3)
            existing["mean_gpu_us"] = round(existing["cum_gpu_us"] / existing["replay_count"], 3)
    return list(by_sig.values())


def count_graph_replays(main_analysis: dict[str, Any]) -> int:
    """Count CUDA-graph replay launches in the main trace (steady window).

    Uses the aggregated ``kernels`` list: sums ``count`` over kernel names that
    look like a graph-launch wrapper. Returns 0 when none are found (eager run or
    unwrapped trace).
    """
    total = 0
    for k in main_analysis.get("kernels") or []:
        if _GRAPH_LAUNCH_RE.search(str(k.get("name", "") or "")):
            total += int(k.get("count", 0) or 0)
    return total


def _phase_and_bucket(graph_variant: str, phase_hint: str) -> tuple[str, str]:
    """Derive (phase, bucket) labels for a variant.

    Bucket is the batch label carried by the variant (``bs_<batch>`` or the
    variant string itself); phase comes from the caller's hint (prefill/decode/
    mixed) since a single capture shard does not by itself distinguish phase.
    """
    bucket = graph_variant
    phase = phase_hint or "mixed"
    return phase, bucket


def _totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Sum per-iteration GPU time over all rows and the two GEMM subsets."""
    total_gpu = sum(float(r.get("cum_gpu_us", 0.0) or 0.0) for r in rows)
    total_gemm = sum(float(r.get("cum_gpu_us", 0.0) or 0.0) for r in rows if r.get("is_gemm"))
    total_target = sum(float(r.get("cum_gpu_us", 0.0) or 0.0) for r in rows if r.get("is_target_gemm"))
    return {
        "total_gpu_kernel_us": round(total_gpu, 3),
        "total_gemm_us": round(total_gemm, 3),
        "total_target_gemm_us": round(total_target, 3),
    }


def _manifest_hash(rows: list[dict[str, Any]]) -> str:
    """Stable sha256 over the canonicalized, key-sorted row set."""
    keys = sorted(r.get("signature_key", "") for r in rows)
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode("utf-8")).hexdigest()


def build_shape_manifest(
    *,
    main_analysis: dict[str, Any],
    capture_variants: list[tuple[str, dict[str, Any]]] | None,
    provenance: dict[str, Any] | None,
    main_trace_hash: str,
    capture_trace_hashes: dict[str, str] | None = None,
    variant_meta: dict[str, dict[str, Any]] | None = None,
    tracelens_revision: str | None = None,
    analysis_route: str = "bypass",
    generated_at: str = "",
    phase_hint: str = "mixed",
) -> dict[str, Any]:
    """Assemble the ``TraceShapeManifest`` (frozen schema v1).

    Args:
        main_analysis: ``analyze_trace`` result for the steady main trace (used
            for the graph-replay count and for the eager fallback rows).
        capture_variants: list of ``(variant_label, analyze_trace_result)`` for
            each CUDA-graph capture shard; empty/None -> eager fallback (rows are
            derived from ``main_analysis`` with variant ``"eager"``).
        provenance: provenance block (from the shared builder; a minimal stub is
            acceptable for the first cut -- missing fields default to null).
        main_trace_hash: sha256 of the steady main trace file.
        capture_trace_hashes: ``{variant_label: sha256}`` for capture shards.
        tracelens_revision: optional TraceLens revision string.
        analysis_route: which route produced the inputs ("bypass"/"tracelens").
        generated_at: caller-supplied UTC timestamp (kept out of this pure fn).
        phase_hint: prefill/decode/mixed hint applied to variant rows.

    Returns:
        The manifest dict (see module docstring for semantics).
    """
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    variant_steady_replay: dict[str, Any] = {}

    total_graph_replays = count_graph_replays(main_analysis)

    if capture_variants:
        for variant_label, analysis in capture_variants:
            phase, bucket = _phase_and_bucket(variant_label, phase_hint)
            rows.extend(
                build_variant_rows(
                    graph_variant=variant_label,
                    analysis=analysis,
                    phase=phase,
                    bucket=bucket,
                    capture_only=True,
                )
            )
        # Steady replay attribution: only unambiguous when a single variant
        # exists. With multiple variants we cannot split the main-trace graph
        # launches per bs bucket yet -> record null + a warning (to be hardened
        # by the engagement work-package). Never fabricate a per-variant count.
        if len(capture_variants) == 1:
            variant_steady_replay[capture_variants[0][0]] = total_graph_replays or None
        else:
            for variant_label, _ in capture_variants:
                variant_steady_replay[variant_label] = None
            warnings.append("multi_variant_replay_unresolved")
    else:
        # Eager fallback: the steady window already is one representative
        # iteration, so per-node cum_gpu_us is real steady time (capture_only
        # False) and no replay multiplier is needed.
        phase, bucket = _phase_and_bucket(EAGER_VARIANT, phase_hint)
        rows.extend(
            build_variant_rows(
                graph_variant=EAGER_VARIANT,
                analysis=main_analysis,
                phase=phase,
                bucket=bucket,
                capture_only=False,
            )
        )
        variant_steady_replay[EAGER_VARIANT] = 1

    totals = _totals(rows)
    workload = {
        **totals,
        "variant_steady_replay": variant_steady_replay,
        "total_graph_replays": total_graph_replays,
        "variant_meta": variant_meta or {},
    }

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": MANIFEST_KIND,
        "generated_from": {
            "tracelens_revision": tracelens_revision,
            "main_trace_hash": main_trace_hash,
            "capture_trace_hashes": capture_trace_hashes or {},
            "analysis_route": analysis_route,
            "steady_window": main_analysis.get("steady_window"),
            "aggregation_scope": main_analysis.get("aggregation_scope", ""),
            "generated_at": generated_at,
        },
        "provenance": provenance or {},
        "workload": workload,
        "rows": rows,
        "warnings": warnings,
    }
    manifest["manifest_hash"] = _manifest_hash(rows)
    return manifest
