###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Build downstream artifacts (candidates / summary / kernel_roofline) for the
bypass analysis backend from the classified device-kernel aggregates.

Primary ranking unit is the device kernel (full coverage; robust to cudagraph),
classified by :mod:`_bypass_classify`, enriched with a best-effort launching op
name. Schema mirrors the ``kernel_candidates.json`` / ``summary.json`` /
``kernel_roofline.json`` contract. Roofline hardware fields are estimated from
the analytical model.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from _bypass_benchmark_resolver import find_benchmark_files, repo_root_from_source
from _bypass_classify import classify_kernel
from _bypass_fusion import analyze_fusion
from _analysis_md import render_report
from _bypass_roofline import compute_roofline
from _kernel_category import canonical_category
from _bypass_source_resolver import (
    resolve_by_kernel_name,
    resolve_source,
    resolve_triton_py,
)
from _idle_gate import resolve_idle_pct_threshold
from _roofline_source import PLACEHOLDER as _RL_PLACEHOLDER
from _task_group_contract import (
    build_operator_identity,
    build_task_group_shape_cases,
    legacy_operator_identity_keys,
    operator_identity_key,
)
from hyperloom.common.kernel_shape_contract import (
    DISPATCHABLE_SHAPE_PROVENANCE as _DISPATCHABLE_SHAPE_PROVENANCE,
)

# Category-appropriate optimization guidance (structured, not LLM prose).
_ACTION_BY_CATEGORY: dict[str, str] = {
    "SDPA": "Profile the attention kernel for tile/occupancy; consider a fused/flash "
    "attention backend; increase decode batch to amortize KV reads.",
    "GEMM": "Tune GEMM tile size / precision and fuse the epilogue where possible; "
    "vendor-library GEMMs (Tensile/rocBLAS) are not rewritable — tune via library config.",
    "Quantization": "Fuse quantization into the adjacent GEMM epilogue and drop redundant per-tensor scaling passes.",
    "KVCacheStore": "Fuse the KV-cache write into attention to remove the separate reshape pass.",
    "Normalization": "Use a fused RMSNorm/LayerNorm and fold the residual/quant into the norm kernel.",
    "Convolution": "Pick an NHWC/implicit-GEMM conv algorithm for the shape; fuse "
    "bias/activation into the conv epilogue (VAE encode/decode).",
    "Elementwise": "Fuse elementwise chains to cut intermediate memory traffic.",
    "MoE": "Optimize expert GEMM and routing; fuse gate/up projections.",
    "MemCpy": "Reduce host/device copies; keep tensors resident on device.",
    "Others": "Profile the kernel for tile size and wave occupancy.",
}

_UNKNOWN_BOUND = "\u2014"  # em dash "unknown bound" marker.
_REUSABLE_BACKENDS = ["forge", "geak"]

# Bound-type display prefixes for the (deterministic) per-kernel suggestion.
_BOUND_PREFIX: dict[str, str] = {"compute_bound": "Compute-bound", "memory_bound": "Memory-bound"}


def _build_suggestion(category: str, bound_type: str) -> str:
    """Deterministic optimization hint from ``category`` + ``bound_type``.

    Category->text lookup optionally prefixed with the analytical bound; feeds
    the specialist prompt's ``action`` slot.
    """
    action = _ACTION_BY_CATEGORY.get(category, _ACTION_BY_CATEGORY["Others"])
    prefix = _BOUND_PREFIX.get(bound_type, "")
    return f"{prefix}: {action}" if prefix else action


# torch ``Input type`` token -> compact dtype suffix for the shape-string
# contract (e.g. ``(15360,2048) bf16``); unmapped/empty types emit a bare shape.
_DTYPE_SUFFIX: dict[str, str] = {
    # Suffixes MUST match the shared harness dtype_map + roofline peak table; a
    # compact "f16"/"f32" makes the harness emit an invalid ``torch.f16``.
    "c10::bfloat16": "bf16",
    "bfloat16": "bf16",
    "c10::half": "fp16",
    "half": "fp16",
    "float16": "fp16",
    "float": "fp32",
    "float32": "fp32",
    "double": "fp64",
    "float64": "fp64",
    "int": "i32",
    "int32": "i32",
    "long": "i64",
    "int64": "i64",
    "short": "i16",
    "int16": "i16",
    "char": "i8",
    "int8": "i8",
    "uint8": "u8",
    "bool": "bool",
}


def _format_operand_shape(dims: Any, dtype: Any) -> str | None:
    """Render one operand as a ``(d0,d1,...) <dtype>`` string (or ``None``).

    Scalar / empty / non-integer operands return ``None`` (dropped from the
    shape string), matching the downstream harness contract. A 1-D operand keeps
    the trailing comma (``(d,)``) so it round-trips as a tuple.
    """
    if not isinstance(dims, (list, tuple)) or not dims:
        return None
    try:
        body = ",".join(str(int(d)) for d in dims)
    except (TypeError, ValueError):
        return None
    shape = f"({body},)" if len(dims) == 1 else f"({body})"
    suffix = _DTYPE_SUFFIX.get(str(dtype or "").strip().lower())
    return f"{shape} {suffix}" if suffix else shape


def _trace_shape_entries(op_shapes: Any, op_dtypes: Any, call_count: int) -> list[dict[str, Any]]:
    """Build the downstream ``input_shapes`` contract from Kineto Input Dims/type.

    Converts a call's per-arg dims (``op_shapes``) + dtypes (``op_dtypes``) into
    ``[{"call_num", "shape"}]`` where ``shape`` is the ``<br>``-joined operand
    strings the GEAK harness (``_build_configs`` / ``_parse_shape_string``) and
    TraceLens candidates consume. Returns ``[]`` when no operand is renderable.

    Args:
        op_shapes: List of per-arg dimension lists (Kineto ``Input Dims``).
        op_dtypes: List of per-arg dtype tokens (Kineto ``Input type``), aligned
            by argument index with ``op_shapes``.
        call_count: Number of launches (stamped as ``call_num``).

    Returns:
        A one-entry ``[{"call_num", "shape"}]`` list, or ``[]``.
    """
    dtypes = op_dtypes if isinstance(op_dtypes, (list, tuple)) else []
    operands: list[str] = []
    for i, dims in enumerate(op_shapes if isinstance(op_shapes, (list, tuple)) else []):
        rendered = _format_operand_shape(dims, dtypes[i] if i < len(dtypes) else "")
        if rendered:
            operands.append(rendered)
    if not operands:
        return []
    return [{"call_num": int(call_count) if call_count else 1, "shape": "<br>".join(operands)}]


# Triton autotune names embed the tile shape, e.g.
# ``_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_...``.
_TILE_NAME_RE = re.compile(r"BLOCK_SIZE_([A-Za-z]+)_(\d+)")


def _launch_grid_shape_entries(grid: Any, block: Any, call_count: int) -> list[dict[str, Any]]:
    """Build a shape entry from a kernel's launch geometry (grid/block).

    Fallback for kernels whose correlation->cpu_op chain is broken (Triton
    direct-launch, graph replay): the launch grid/block is coarse geometry, not
    dispatch-grade operand dims. Returns ``[]`` when no positive dimension is present.
    """

    def _dims(v: Any) -> list[int]:
        out: list[int] = []
        for d in v if isinstance(v, (list, tuple)) else []:
            try:
                n = int(d)
            except (TypeError, ValueError):
                return []
            out.append(n)
        return out

    g = _dims(grid)
    b = _dims(block)
    parts: list[str] = []
    if any(x > 0 for x in g):
        parts.append("grid=(" + ",".join(str(x) for x in g) + ")")
    if any(x > 0 for x in b):
        parts.append("block=(" + ",".join(str(x) for x in b) + ")")
    if not parts:
        return []
    return [{"call_num": int(call_count) if call_count else 1, "shape": "<br>".join(parts)}]


def _tile_name_shape_entries(kernel_name: str, call_count: int) -> list[dict[str, Any]]:
    """Extract a tile shape from a Triton autotune kernel name (``BLOCK_SIZE_*``).

    Lowest-priority fallback: yields a ``M32<br>N32<br>K256``-style tile shape
    when the name encodes it. Returns ``[]`` when no tile token is present.
    """
    matches = _TILE_NAME_RE.findall(kernel_name or "")
    if not matches:
        return []
    body = "<br>".join(f"{axis}{val}" for axis, val in matches)
    return [{"call_num": int(call_count) if call_count else 1, "shape": body}]


def _source_type_for_op(op_name: str) -> str:
    """Best-effort source-type guess from a launching op name.

    Args:
        op_name: Resolved launching op name (may be empty).

    Returns:
        ``"python"`` / ``"hip_cpp"`` / ``"unknown"``.
    """
    n = (op_name or "").lower()
    if not n:
        return "unknown"
    if n.startswith(("aten::", "vllm::", "vllm_aiter::", "_c_cache_ops::", "_rocm_c::")):
        return "python"
    if "aiter" in n or "ck" in n:
        return "hip_cpp"
    return "unknown"


# Native device-code extensions.
_NATIVE_SOURCE_EXTS = (
    ".cu",
    ".cuh",
    ".hip",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".hh",
    ".h",
    ".c",
)


def _build_task_groups(hot_kernels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group routable candidates that share an editable source into task groups.

    Only candidates that are both ``reusable_native_kernel`` and carry a resolved
    ``source_file`` participate: an unresolved (empty) source is not a shared
    function, so those stay standalone (per-kernel dispatch, unchanged). Every
    group is keyed by operation plus source: repeated shapes of one operator
    share a task, while different operators in one translation unit stay
    independent.

    Each group carries compact ``rows``:
    ``kernel_id`` / ``name`` / ``device_kernel_name`` / ``shapes`` / ``call_count``
    / ``duration_us`` / ``percent_of_total`` / ``gpu_pct`` / ``bound_type``. Groups
    are ranked by aggregate GPU time; the heaviest row is the primary.

    Args:
        hot_kernels: The candidate rows from :func:`build_candidates`.

    Returns:
        Ordered task-group dicts (``tg001`` first = heaviest), or ``[]`` when no
        candidate is routable-with-source.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for c in hot_kernels:
        if not c.get("reusable_native_kernel"):
            continue
        src = str(c.get("source_file") or "").strip()
        if not src:
            continue
        reported_source = src
        operation = str(c.get("name") or "")
        function_name = operation
        legacy_function_name = str(c.get("device_kernel_name") or operation)
        source_kind = "native" if src.lower().endswith(_NATIVE_SOURCE_EXTS) else "py"
        identity = build_operator_identity(
            source_kind=source_kind,
            source_path=reported_source,
            operation=operation,
            function_name=function_name,
        )
        src = str(identity["source_path"])
        operation_key = str(identity["operation"])
        key = operator_identity_key(
            source_kind=source_kind,
            source_path=reported_source,
            operation=operation,
            function_name=function_name,
        )
        legacy_keys = legacy_operator_identity_keys(
            source_kind=source_kind,
            source_path=reported_source,
            operation=operation,
            function_name=legacy_function_name,
        )
        shapes = c.get("input_shapes") or []
        row = {
            "kernel_id": c.get("kernel_id", ""),
            "name": c.get("name", ""),
            "device_kernel_name": c.get("device_kernel_name", ""),
            "shapes": shapes,
            "input_shapes": shapes,
            "input_dtypes": c.get("input_dtypes") or [],
            "output_shapes": c.get("output_shapes") or [],
            "output_dtypes": c.get("output_dtypes") or [],
            "raw_arg_spec": c.get("raw_arg_spec") or {},
            "call_count": c.get("call_count", 0),
            "duration_us": c.get("duration_us", 0.0),
            "percent_of_total": c.get("percent_of_total", 0.0),
            "gpu_pct": c.get("gpu_pct", 0.0),
            "bound_type": c.get("bound_type", ""),
        }
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = {
                "task_group_id": "",
                "task_group_key": key,
                "operator_identity": identity,
                "identity_route": "bypass",
                "legacy_task_group_keys": legacy_keys,
                "operation": operation,
                "operation_key": operation_key,
                "source_path": src,
                "kernel_ids": [],
                "primary_kernel_id": "",
                "rows": [],
                "aggregate_duration_us": 0.0,
                "aggregate_call_count": 0,
                "aggregate_gpu_pct": 0.0,
                "source": "bypass",
            }
        else:
            bucket["legacy_task_group_keys"] = list(
                dict.fromkeys(
                    [
                        *(bucket.get("legacy_task_group_keys") or []),
                        *legacy_keys,
                    ]
                )
            )
        if row["kernel_id"] and row["kernel_id"] not in bucket["kernel_ids"]:
            bucket["kernel_ids"].append(row["kernel_id"])
        bucket["rows"].append(row)
        bucket["aggregate_duration_us"] += float(row["duration_us"] or 0.0)
        bucket["aggregate_call_count"] += int(row["call_count"] or 0)
        bucket["aggregate_gpu_pct"] += float(row["gpu_pct"] or 0.0)

    ordered = sorted(buckets.values(), key=lambda g: g["aggregate_duration_us"], reverse=True)
    for idx, group in enumerate(ordered, start=1):
        group["task_group_id"] = f"tg{idx:03d}"
        group["rows"].sort(key=lambda r: float(r.get("duration_us") or 0.0), reverse=True)
        if group["rows"]:
            group["primary_kernel_id"] = group["rows"][0]["kernel_id"]
        group["aggregate_duration_us"] = round(group["aggregate_duration_us"], 3)
        group["aggregate_gpu_pct"] = round(group["aggregate_gpu_pct"], 3)
        group["shape_cases"] = build_task_group_shape_cases(group)
    return ordered


def _source_type_from_path(path: str) -> str:
    """Derive source type from a resolved source file's extension.

    Args:
        path: Resolved editable source path.

    Returns:
        ``"hip_cpp"`` for native device code, ``"python"`` for a ``.py`` source,
        or ``""`` when the extension is unrecognized (caller falls back to the
        op-name heuristic).
    """
    low = (path or "").lower()
    if low.endswith(_NATIVE_SOURCE_EXTS):
        return "hip_cpp"
    if low.endswith(".py"):
        return "python"
    return ""


def _implementation_contract(
    *,
    op_name: str,
    source_file: str,
    trace_kernel_kind: str,
) -> tuple[str, str]:
    """Derive source type and kind only from explicit trace/source evidence."""
    traced_kind = str(trace_kernel_kind or "").strip().lower().replace("-", "_")
    normalized_op = str(op_name or "").strip().lower()
    if "flydsl" in traced_kind or (normalized_op.startswith("pseudo_op::") and "flydsl" in normalized_op):
        return "flydsl", "flydsl"
    if "triton" in traced_kind:
        return "triton", "triton"
    source_type = _source_type_from_path(source_file) or _source_type_for_op(op_name)
    if source_type in {"triton", "flydsl", "ck"}:
        return source_type, source_type
    return source_type, ""


def _short_name(kernel_name: str) -> str:
    """Shorten a mangled device-kernel name for candidate identity display."""
    n = kernel_name or ""
    # Strip C++ template/mangling tails.
    n = re.sub(r"<.*$", "", n)
    n = n.strip()
    return n[:80] if n else "unknown_kernel"


def partition_kernels(
    hot_kernels: list[dict[str, Any]],
    is_routable: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a hot-kernel list into ``(routable, skipped)`` one way, everywhere.

    The three ``skipped_kernels`` producers used to each build the list by hand,
    with subtly different comprehensions; a reader could not trust that
    ``hot_kernels == routable + skipped`` held. This is the single partition path
    they now share.

    The ``routability`` test is the *caller's* -- the strict dispatch predicate
    (reusable + resolved source + dispatch-grade shape) and the coarse
    reusability predicate mean different things and are contracted separately, so
    each site passes its own ``is_routable``. What is unified is the *shape* of
    the split: ``skipped`` is always the exact ``kernel_id`` complement of
    ``routable`` within ``hot_kernels``, so the two lists partition the input with
    no overlap and no leakage by construction.

    Args:
        hot_kernels: The full ranked hotspot set; each row carries ``kernel_id``.
        is_routable: Predicate deciding whether a row belongs in ``routable``.

    Returns:
        ``(routable_kernels, skipped_kernels)`` -- a partition of ``hot_kernels``.
    """
    routable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in hot_kernels:
        if not isinstance(row, dict):
            continue
        (routable if is_routable(row) else skipped).append(row)
    return routable, skipped


def build_candidates(
    analyze_out: dict[str, Any],
    *,
    framework: str,
    target_platform: str,
    top_k: int = 15,
    discover_benchmarks: bool = False,
) -> dict[str, Any]:
    """Turn classified top device kernels into the candidate payload.

    Args:
        analyze_out: Result of :func:`_bypass_trace_reader.analyze_trace`.
        framework: Serving framework tag.
        target_platform: GPU platform tag.
        top_k: Max number of hot-kernel candidates to emit.

    Returns:
        A dict with ``hot_kernels`` (the FULL ranked hotspot set), plus
        ``routable_kernels`` / ``skipped_kernels`` (a partition of ``hot_kernels``:
        routable = reusable-with-resolved-source = dispatchable) and
        ``task_groups``.
    """
    kernels = analyze_out.get("kernels") or []
    hot_kernels: list[dict[str, Any]] = []
    for idx, k in enumerate(kernels[: top_k if top_k and top_k > 0 else len(kernels)], start=1):
        kname = k.get("name", "") or ""
        op_name = k.get("op_name", "") or ""
        kc = classify_kernel(kname, op_name=op_name)
        kernel_id = f"k{idx:03d}"
        display = op_name or _short_name(kname)

        # Source resolution. Priority: (1) a Triton .py kernel_file from the
        # trace's cpu_op args, with the exact @triton.jit def line pinned via AST
        # (resolve_triton_py, method "trace_kernel_file[_ast]"); (2) native
        # .cu/.hip active-finder lookup by device kernel symbol (resolve_source ->
        # source_resolver, method "symbol_index"); (3) repo-scan by device kernel
        # name; else unresolved. There is no static op_to_source map.
        source_file, source_line, source_method = resolve_triton_py(
            k.get("op_kernel_file", "") or "",
            k.get("op_kernel_backend", "") or "",
            symbol=kname,
        )
        # Native lookup is driven by the device symbol (kname), not op_name:
        # the finder ignores op_name, and device kernels with no correlated
        # cpu_op (common under HIP graphs) arrive with op_name="". Gating on
        # kname keeps those resolvable.
        #
        # finder_patchable carries the finder's authoritative patchability
        # verdict onto the candidate (op_to_source_* below); it stays None for a
        # Triton/repo-scan hit, which the classifier handles instead.
        finder_patchable: bool | None = None
        finder_status = ""
        finder_reason = ""
        if not source_file and kname:
            source_file, method = resolve_source(op_name, framework=framework, device_kernel_name=kname)
            if source_file:
                source_method = method
                finder_patchable = True
                finder_status = "resolved"
            elif method == "non_patchable":
                # A positive "known not rewritable" verdict (e.g. a CK template
                # instantiation), NOT a miss. Record it and DO NOT fall through
                # to the repo-scan tier, which could otherwise override the
                # finder with a coincidental kernel-name hit.
                source_method = method
                finder_patchable = False
                finder_status = "non_rewritable"
                finder_reason = (
                    "non-patchable kernel (symbol-detected: CK template / no single editable __global__ source)"
                )
        # Repo-scan is the last resort, and only for a genuine miss -- never when
        # the finder already returned an authoritative non_patchable verdict.
        if not source_file and kname and source_method != "non_patchable":
            source_file, method = resolve_by_kernel_name(kname)
            if source_file:
                source_method = method

        # Shape resolution waterfall (provenance records the source):
        #   1. torch_trace      -- this kernel's own cpu_op Input Dims (precise)
        #   2. capture_backfill -- same-name kernel's capture-time shape
        #   3. launch_grid      -- this kernel's launch grid/block geometry
        #   4. tile_name        -- BLOCK_SIZE_* tile embedded in the kernel name
        #   5. unresolved       -- none of the above
        _count = k.get("count") or 0
        op_shapes = k.get("op_shapes") or []
        op_dtypes = k.get("op_dtypes") or []
        shape_entries = _trace_shape_entries(op_shapes, op_dtypes, _count)
        shape_provenance = "torch_trace" if shape_entries else ""
        if not shape_entries and not k.get("backfill_ambiguous"):
            bf_shapes = k.get("backfill_shapes") or []
            bf_dtypes = k.get("backfill_dtypes") or []
            shape_entries = _trace_shape_entries(bf_shapes, bf_dtypes, _count)
            if shape_entries:
                op_dtypes = bf_dtypes
                shape_provenance = "capture_backfill"
        if not shape_entries:
            shape_entries = _launch_grid_shape_entries(k.get("launch_grid"), k.get("launch_block"), _count)
            if shape_entries:
                shape_provenance = "launch_grid"
        if not shape_entries:
            shape_entries = _tile_name_shape_entries(kname, _count)
            if shape_entries:
                shape_provenance = "tile_name"
        if not shape_provenance:
            shape_provenance = "unresolved"

        # Benchmark discovery is opt-in; a routable kernel's on-disk
        # test/benchmark can seed downstream harness generation.
        bench_files: list[str] = []
        kernel_repo = ""
        if discover_benchmarks and kc.reusable and source_file:
            kernel_repo = repo_root_from_source(source_file)
            bench_files = find_benchmark_files(op_name, source_file)
        source_type, kernel_kind = _implementation_contract(
            op_name=op_name,
            source_file=source_file,
            trace_kernel_kind=str(k.get("op_kernel_backend") or ""),
        )

        cand: dict[str, Any] = {
            "kernel_id": kernel_id,
            "name": display,
            "kernel_category": kc.category,
            "device_kernel_name": kname[:120],
            "device_kernel_names": [kname[:120]],
            "duration_us": k.get("gpu_time_us", 0.0),
            "gpu_pct": k.get("gpu_pct", 0.0),
            "percent_of_total": k.get("gpu_pct", 0.0),
            "call_count": k.get("count", 0),
            "bound_type": _UNKNOWN_BOUND,
            "efficiency_percent": 0.0,
            "roofline_attainment_pct": None,
            "flops_per_byte": None,
            "arithmetic_intensity": None,
            "compute_utilization_pct": None,
            "bandwidth_utilization_pct": None,
            "rocprof_roofline": None,
            # Placeholder roofline: bound_type/AI/util above are structural
            # defaults, NOT measured. ``roofline_source`` tracks derivation:
            # placeholder -> analytical -> rocprof.
            "roofline_measured": False,
            "roofline_source": _RL_PLACEHOLDER,
            "library": "",
            "backend": framework,
            "framework": framework,
            "source_file": source_file,
            # AST-pinned @triton.jit def line for .py sources (None when unpinned
            # or for native .cu kernels resolved by the symbol index).
            "source_line": source_line,
            "source_resolution_method": source_method,
            "source_type": source_type,
            "kernel_kind": kernel_kind,
            "reusable_native_kernel": kc.reusable,
            # Non-reusable keeps the classifier reason; a reusable kernel with a
            # finder non_patchable verdict reports that verdict; a reusable
            # kernel with no resolved source is simply not dispatchable.
            "skip_reason": (
                kc.skip_reason
                if not kc.reusable
                else (
                    ""
                    if source_file
                    else (f"source: {finder_reason}" if finder_patchable is False else "source file not resolved")
                )
            ),
            "recommended_backends": list(_REUSABLE_BACKENDS) if kc.reusable else [],
            # Seeds for the GEAK harness + rocprof enrichment (only when
            # discover_benchmarks is set).
            "benchmark_files": bench_files,
            "kernel_repo": kernel_repo,
            # ``shapes`` / ``input_shapes`` use the downstream contract form
            # (the kernel-opt gate + GEAK harness require this format).
            "shapes": shape_entries,
            "input_shapes": shape_entries,
            "input_dtypes": op_dtypes,
            "shape_provenance": shape_provenance,
            # launch_grid / tile_name shapes are geometry, so the kernel-opt gate
            # rejects them. Mark whether the shape actually satisfies dispatch so
            # the report and routing do not claim a geometry-only kernel is ready.
            "shape_dispatchable": shape_provenance in _DISPATCHABLE_SHAPE_PROVENANCE,
        }
        # Carry the finder's authoritative patchability verdict (method
        # "symbol_index") so the downstream gate honors it instead of
        # re-deriving from heuristics, and so a non_patchable verdict is not
        # silently indistinguishable from an unresolved miss.
        if finder_patchable is not None:
            cand["op_to_source_status"] = finder_status
            cand["op_to_source_patchable"] = finder_patchable
            cand["op_to_source_reason"] = finder_reason
        # Analytical roofline: derive bound_type / AI / efficiency from captured
        # shapes + measured time for EVERY estimable kernel (rocprof enrichment
        # later refines it to a measured roofline). Only precise operand shapes
        # (torch_trace / capture_backfill) feed the AI math; launch-grid and
        # tile-name fallbacks are geometry, not operand dims, so they would
        # poison the roofline -- skip analytical estimation for them.
        _roofline_shape = (
            shape_entries[0]["shape"] if shape_entries and shape_provenance in _DISPATCHABLE_SHAPE_PROVENANCE else ""
        )
        rl = compute_roofline(
            category=kc.category,
            shape_str=_roofline_shape,
            gpu_time_us=float(cand["duration_us"] or 0.0),
            call_count=int(cand["call_count"] or 1),
            gpu_type=target_platform,
        )
        if rl:
            cand.update(rl)
        # A reusable kernel with a resolved source but only a geometry shape
        # (launch_grid / tile_name) is visible but not auto-dispatchable: the
        # kernel-opt gate requires operand dims. Surface that in skip_reason so
        # it is not silently presented as ready to optimize.
        if kc.reusable and source_file and not cand["shape_dispatchable"]:
            cand["skip_reason"] = f"shape not dispatchable (provenance={shape_provenance}); need operand dims"
        # Optimization ROI = GPU-time share x headroom (1 - efficiency); with no
        # analytical efficiency, headroom=1 so it degrades to gpu_pct.
        eff = cand.get("efficiency_percent")
        eff = float(eff) if isinstance(eff, (int, float)) else 0.0
        headroom = 1.0 - min(max(eff, 0.0), 100.0) / 100.0
        cand["optimization_priority"] = round(float(cand.get("gpu_pct") or 0.0) * headroom, 4)
        # Deterministic per-kernel hint for the specialist prompt's action slot.
        suggestion = _build_suggestion(kc.category, str(cand.get("bound_type") or ""))
        cand["suggestion"] = suggestion
        cand["recommended_actions"] = [suggestion]
        hot_kernels.append(cand)

    # Group repeated shapes of one operator and editable source into one task;
    # stamp that shared contract onto each member.
    task_groups = _build_task_groups(hot_kernels)
    kid_to_group = {kid: g for g in task_groups for kid in g["kernel_ids"]}
    for c in hot_kernels:
        g = kid_to_group.get(c["kernel_id"])
        if g is not None:
            c["task_group"] = g

    # 1-based rank by optimization ROI, stamped WITHOUT reordering hot_kernels
    # (that list stays gpu_pct-sorted).
    for rank, c in enumerate(
        sorted(hot_kernels, key=lambda x: x.get("optimization_priority") or 0.0, reverse=True), start=1
    ):
        c["priority_rank"] = rank

    # ``routable_kernels`` = the subset actually dispatchable to kernel-opt:
    # reusable + resolved source + a dispatch-grade operand shape. A geometry-only
    # (launch_grid/tile_name) shape is rejected by the kernel-opt gate, so those
    # kernels stay in ``skipped_kernels`` and never re-enter the untried queue.
    # ``hot_kernels`` stays the FULL ranked hotspot set.
    # Complement within ``hot_kernels`` (via the shared partition helper) so the
    # contract ``hot_kernels == routable_kernels + skipped_kernels`` holds. This
    # site's routability is the strict dispatch predicate: reusable + resolved
    # source + a dispatch-grade operand shape.
    routable_kernels, skipped_kernels = partition_kernels(
        hot_kernels,
        lambda c: bool(c.get("reusable_native_kernel") and c.get("source_file") and c.get("shape_dispatchable")),
    )
    return {
        "source": "bypass",
        "framework": framework,
        "target_platform": target_platform,
        "aggregation_scope": analyze_out.get("aggregation_scope", "full_trace"),
        "hot_kernels": hot_kernels,
        "routable_kernels": routable_kernels,
        "skipped_kernels": skipped_kernels,
        "task_groups": task_groups,
    }


def build_summary(
    candidates: dict[str, Any],
    *,
    framework: str,
    target_platform: str,
    generated_at: str,
    trace_health_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the routed-vs-skipped audit ``summary.json`` payload.

    Args:
        candidates: Output of :func:`build_candidates`.
        framework: Serving framework tag.
        target_platform: GPU platform tag.
        generated_at: ISO timestamp string.
        trace_health_warnings: Optional health warnings to record.

    Returns:
        The ``summary.json`` payload dict.
    """

    # ``tasks`` / ``skipped`` reuse the SAME split build_candidates computed so
    # summary.json never disagrees with kernel_candidates.json.
    def _audit_row(c: dict[str, Any]) -> dict[str, Any]:
        return {
            "kernel_id": c["kernel_id"],
            "name": c["name"],
            "kernel_category": c["kernel_category"],
            "duration_us": c["duration_us"],
            "gpu_pct": c["gpu_pct"],
            "call_count": c["call_count"],
            "source_type": c["source_type"],
            "reusable_native_kernel": c["reusable_native_kernel"],
        }

    tasks = []
    for c in candidates.get("routable_kernels") or []:
        row = _audit_row(c)
        row["recommended_backends"] = c["recommended_backends"]
        tasks.append(row)
    skipped = []
    for c in candidates.get("skipped_kernels") or []:
        row = _audit_row(c)
        row["skip_reason"] = c["skip_reason"]
        skipped.append(row)
    # Reusable kernels that resolved a source but were held out of ``tasks``
    # because their shape is geometry-only (launch_grid/tile_name). Surfaced for
    # audit so the count reconciles with the "auto-dispatchable" report wording.
    geometry_only_skipped_count = sum(
        1
        for c in candidates.get("skipped_kernels") or []
        if c.get("reusable_native_kernel") and c.get("source_file") and not c.get("shape_dispatchable")
    )
    # Compact task-group projection for the audit view.
    group_entries = [
        {
            "task_group_id": g.get("task_group_id", ""),
            "operation": g.get("operation", ""),
            "source_path": g.get("source_path", ""),
            "primary_kernel_id": g.get("primary_kernel_id", ""),
            "kernel_ids": g.get("kernel_ids", []),
            "row_count": len(g.get("rows") or []),
            "aggregate_duration_us": g.get("aggregate_duration_us", 0.0),
            "aggregate_gpu_pct": g.get("aggregate_gpu_pct", 0.0),
        }
        for g in (candidates.get("task_groups") or [])
    ]
    return {
        "source": "bypass",
        "framework": framework,
        "target_platform": target_platform,
        "generated_at": generated_at,
        "aggregation_scope": candidates.get("aggregation_scope", "full_trace"),
        "tasks": tasks,
        "skipped": skipped,
        "task_groups": group_entries,
        "task_count": len(tasks),
        # ``tasks`` already excludes geometry-only kernels, so dispatchable_count
        # == task_count; kept explicit so the audit view states dispatchability.
        "dispatchable_count": len(tasks),
        "geometry_only_skipped_count": geometry_only_skipped_count,
        "skipped_count": len(skipped),
        "task_group_count": len(group_entries),
        "trace_health_warnings": list(trace_health_warnings or []),
    }


def _category_rollup(analyze_out: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate GPU time by category over *all* device kernels.

    Uses the full kernel list (not just top-K) so category shares are complete;
    requires the reader to have been called with ``top_k=0``.

    Returns:
        Category rows sorted by GPU time desc, each with category / gpu_ms /
        gpu_pct / kernel_count.
    """
    kernels = analyze_out.get("kernels") or []
    total_us = sum(float(k.get("gpu_time_us") or 0.0) for k in kernels) or 1.0
    cat_us: dict[str, float] = defaultdict(float)
    cat_cnt: dict[str, int] = defaultdict(int)
    for k in kernels:
        kc = classify_kernel(k.get("name", "") or "", op_name=k.get("op_name", "") or "")
        us = float(k.get("gpu_time_us") or 0.0)
        cat_us[kc.category] += us
        cat_cnt[kc.category] += int(k.get("count") or 0)
    rows = [
        {
            "category": cat,
            "gpu_ms": round(us / 1000.0, 3),
            "gpu_pct": round(us / total_us * 100.0, 2),
            "kernel_count": cat_cnt[cat],
        }
        for cat, us in cat_us.items()
    ]
    rows.sort(key=lambda r: r["gpu_ms"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Structured CSV export (deterministic, code-generated).
# ---------------------------------------------------------------------------
#: Stable column order for the per-kernel metrics CSV.
_METRICS_COLUMNS: list[str] = [
    "priority_rank",
    "optimization_priority",
    "kernel_id",
    "name",
    "kernel_category",
    "device_kernel_name",
    "duration_us",
    "gpu_pct",
    "call_count",
    "bound_type",
    "arithmetic_intensity",
    "efficiency_percent",
    "compute_utilization_pct",
    "bandwidth_utilization_pct",
    "roofline_source",
    "roofline_measured",
    "reusable_native_kernel",
    "source_file",
    "source_type",
    "recommended_backends",
    "benchmark_files_count",
    "skip_reason",
    "representative_shape",
    "input_dtypes",
    "shape_provenance",
    "suggestion",
]

_SUMMARY_COLUMNS: list[str] = [
    "kernel_category",
    "kernel_count",
    "total_gpu_pct",
    "total_duration_us",
    "mean_efficiency_percent",
    "dominant_bound_type",
    "routable_count",
]


def _join_list(value: Any) -> str:
    """Render a list cell as a ``;``-joined string (CSV-cell friendly)."""
    return ";".join(str(x) for x in value) if isinstance(value, list) else ""


def build_metrics_rows(candidates: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ALL hot kernels (routable + skipped) into per-kernel metric rows.

    Args:
        candidates: Output of :func:`build_candidates`.

    Returns:
        One flat dict per hot kernel keyed by :data:`_METRICS_COLUMNS`.
    """
    rows: list[dict[str, Any]] = []
    for c in candidates.get("hot_kernels") or []:
        shapes = c.get("shapes") or []
        rep = shapes[0].get("shape", "") if shapes and isinstance(shapes[0], dict) else ""
        rows.append(
            {
                "priority_rank": c.get("priority_rank", ""),
                "optimization_priority": c.get("optimization_priority", ""),
                "kernel_id": c.get("kernel_id", ""),
                "name": c.get("name", ""),
                "kernel_category": c.get("kernel_category", ""),
                "device_kernel_name": c.get("device_kernel_name", ""),
                "duration_us": c.get("duration_us", ""),
                "gpu_pct": c.get("gpu_pct", ""),
                "call_count": c.get("call_count", ""),
                "bound_type": c.get("bound_type", ""),
                "arithmetic_intensity": c.get("arithmetic_intensity", ""),
                "efficiency_percent": c.get("efficiency_percent", ""),
                "compute_utilization_pct": c.get("compute_utilization_pct", ""),
                "bandwidth_utilization_pct": c.get("bandwidth_utilization_pct", ""),
                "roofline_source": c.get("roofline_source", ""),
                "roofline_measured": c.get("roofline_measured", ""),
                "reusable_native_kernel": c.get("reusable_native_kernel", ""),
                "source_file": c.get("source_file", ""),
                "source_type": c.get("source_type", ""),
                "recommended_backends": _join_list(c.get("recommended_backends")),
                "benchmark_files_count": len(c.get("benchmark_files") or []),
                "skip_reason": c.get("skip_reason", ""),
                "representative_shape": rep,
                "input_dtypes": _join_list(c.get("input_dtypes")),
                "shape_provenance": c.get("shape_provenance", ""),
                "suggestion": c.get("suggestion", ""),
            }
        )
    return rows


def build_category_summary(candidates: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate hot kernels by category (the CSV 'summary' view).

    Args:
        candidates: Output of :func:`build_candidates`.

    Returns:
        One row per category (keyed by :data:`_SUMMARY_COLUMNS`), GPU%-descending.
    """
    agg: dict[str, dict[str, Any]] = {}
    for c in candidates.get("hot_kernels") or []:
        cat = c.get("kernel_category") or "Others"
        a = agg.setdefault(
            cat,
            {
                "kernel_count": 0,
                "total_gpu_pct": 0.0,
                "total_duration_us": 0.0,
                "eff_sum": 0.0,
                "eff_n": 0,
                "bounds": Counter(),
                "routable": 0,
            },
        )
        a["kernel_count"] += 1
        a["total_gpu_pct"] += float(c.get("gpu_pct") or 0.0)
        a["total_duration_us"] += float(c.get("duration_us") or 0.0)
        eff = c.get("efficiency_percent")
        if isinstance(eff, (int, float)) and eff > 0:
            a["eff_sum"] += float(eff)
            a["eff_n"] += 1
        bt = c.get("bound_type") or ""
        if bt in ("compute_bound", "memory_bound"):
            a["bounds"][bt] += 1
        if c.get("reusable_native_kernel"):
            a["routable"] += 1
    rows = [
        {
            "kernel_category": cat,
            "kernel_count": a["kernel_count"],
            "total_gpu_pct": round(a["total_gpu_pct"], 4),
            "total_duration_us": round(a["total_duration_us"], 3),
            "mean_efficiency_percent": round(a["eff_sum"] / a["eff_n"], 3) if a["eff_n"] else "",
            "dominant_bound_type": a["bounds"].most_common(1)[0][0] if a["bounds"] else "",
            "routable_count": a["routable"],
        }
        for cat, a in agg.items()
    ]
    rows.sort(key=lambda r: r["total_gpu_pct"], reverse=True)
    return rows


def _rows_to_csv(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Serialize rows to CSV text (stdlib ``csv``; empty cell for None/missing)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    return buf.getvalue()


def build_metrics_csv(candidates: dict[str, Any]) -> str:
    """Per-kernel metrics as CSV text (all hot kernels; see :data:`_METRICS_COLUMNS`)."""
    return _rows_to_csv(_METRICS_COLUMNS, build_metrics_rows(candidates))


def build_category_summary_csv(candidates: dict[str, Any]) -> str:
    """Category-aggregated summary as CSV text (see :data:`_SUMMARY_COLUMNS`)."""
    return _rows_to_csv(_SUMMARY_COLUMNS, build_category_summary(candidates))


def render_analysis_md(
    candidates: dict[str, Any],
    analyze_out: dict[str, Any],
    *,
    model_name: str,
    framework: str,
    target_platform: str,
    throughput_unit: str = "tok/s",
    metrics_csv_path: str = "",
    summary_csv_path: str = "",
) -> str:
    """Render the human/downstream ``analysis.md`` report (bypass route).

    Structured (not LLM prose); mirrors the golden section layout but is not
    consumed by ``parse_analysis_md``.

    Args:
        candidates: Output of :func:`build_candidates`.
        analyze_out: Output of :func:`_bypass_trace_reader.analyze_trace`
            (must be produced with ``top_k=0`` for a complete category rollup).
        model_name: Model identifier for the title.
        framework: Serving framework tag.
        target_platform: GPU platform tag.
        throughput_unit: ``tok/s`` (text-gen) or ``img/s`` (xDiT).

    Returns:
        The full markdown report text.
    """
    timeline = analyze_out.get("timeline") or {}
    attribution = analyze_out.get("attribution") or {}
    hot = candidates.get("hot_kernels") or []
    rollup = _category_rollup(analyze_out)
    top_cat = rollup[0]["category"] if rollup else "n/a"
    scope = candidates.get("aggregation_scope", "full_trace")

    total_ms = timeline.get("total_time_ms")
    memcpy_ms = timeline.get("gpu_memcpy_ms")
    idle_pct = timeline.get("idle_pct")
    exec_summary = {
        "total_gpu_time_ms": total_ms,
        "gpu_busy_pct": timeline.get("busy_pct"),
        "gpu_idle_pct": idle_pct,
        "gpu_memcpy_ms": memcpy_ms,
        "top_bottleneck_category": top_cat,
        "attribution_pct": attribution.get("attributed_pct"),
    }
    # memcpy as % of total wall time for the System-Level Signals table.
    memcpy_pct: float | None = None
    if isinstance(memcpy_ms, (int, float)) and isinstance(total_ms, (int, float)) and total_ms:
        memcpy_pct = float(memcpy_ms) / float(total_ms) * 100.0
    system_signals = {
        "idle_pct": idle_pct,
        "exposed_comm_pct": None,  # bypass does not model exposed communication
        "exposed_memcpy_pct": memcpy_pct,
    }

    # Top Hot Kernels rows. Displayed Eff% is the binding-side roofline
    # attainment.
    hot_rows = [
        {
            "name": c.get("name"),
            "time_us": c.get("duration_us"),
            "gpu_pct": c.get("gpu_pct"),
            "efficiency_percent": c.get("roofline_attainment_pct"),
            "arithmetic_intensity": c.get("arithmetic_intensity"),
            "bound_type": c.get("bound_type"),
            "category": c.get("kernel_category"),
            "source_file": c.get("source_file"),
        }
        for c in hot
    ]

    # One P-item per routable candidate (ranked by optimization ROI). %E2E and
    # launcher Kernel Path are not modelled by bypass.
    routable = [c for c in hot if c.get("reusable_native_kernel")]
    # Auto-dispatchable == resolved source AND a dispatch-grade operand shape;
    # a geometry-only (launch_grid/tile_name) shape does not satisfy the gate.
    dispatchable = [c for c in routable if c.get("source_file") and c.get("shape_dispatchable")]
    p_items = []
    for i, c in enumerate(routable, start=1):
        shapes = c.get("input_shapes") or c.get("shapes") or []
        args = [str(s.get("shape")) for s in shapes if isinstance(s, dict) and s.get("shape")]
        p_items.append(
            {
                "rank": c.get("priority_rank") or i,
                "category": c.get("kernel_category"),
                "rows": [
                    {
                        "name": c.get("name"),
                        "time_us": c.get("duration_us"),
                        "gpu_pct": c.get("gpu_pct"),
                        "e2e_pct": None,
                        "call_count": c.get("call_count"),
                        "flops_per_byte": c.get("flops_per_byte"),
                        "efficiency_percent": c.get("roofline_attainment_pct"),
                        "bound_type": c.get("bound_type"),
                        "args": args,
                        "source_file": c.get("source_file"),
                        "kernel_path": None,
                    }
                ],
            }
        )

    provenance = (
        f"framework={framework or 'unknown'}, platform={target_platform or 'unknown'}, "
        f"throughput_unit={throughput_unit}, aggregation_scope={scope}. "
        f"Per-kernel roofline (bound/AI/efficiency) is computed analytically from captured "
        f"operand shapes + measured kernel time (roofline_source=analytical)."
    )

    extra = _render_bypass_extra_sections(
        candidates,
        analyze_out,
        hot=hot,
        rollup=rollup,
        routable=routable,
        dispatchable=dispatchable,
        timeline=timeline,
        attribution=attribution,
        framework=framework,
        target_platform=target_platform,
        throughput_unit=throughput_unit,
        scope=scope,
        metrics_csv_path=metrics_csv_path,
        summary_csv_path=summary_csv_path,
    )

    return render_report(
        route="bypass",
        model_name=model_name or "Workload",
        provenance_detail=provenance,
        exec_summary=exec_summary,
        system_signals=system_signals,
        idle_threshold=resolve_idle_pct_threshold(),
        hot_kernels=hot_rows,
        p_items=p_items,
        extra_sections=extra,
    )


def _render_bypass_extra_sections(
    candidates: dict[str, Any],
    analyze_out: dict[str, Any],
    *,
    hot: list[dict[str, Any]],
    rollup: list[dict[str, Any]],
    routable: list[dict[str, Any]],
    dispatchable: list[dict[str, Any]],
    timeline: dict[str, Any],
    attribution: dict[str, Any],
    framework: str,
    target_platform: str,
    throughput_unit: str,
    scope: str,
    metrics_csv_path: str,
    summary_csv_path: str,
) -> str:
    """Render the bypass-only richer sections appended after the shared spine.

    Covers category rollup, optimization-priority Top-N, per-candidate
    optimization prose, task groups, per-kernel detail, appendix, and CSV links.
    """
    L: list[str] = []

    # Top Operations (category rollup) + analytical bound distribution.
    L.append("## Top Operations")
    L.append("")
    if rollup:
        L.append("| Rank | Category | GPU % | Time (ms) | Kernels |")
        L.append("|------|----------|-------|-----------|---------|")
        for i, r in enumerate(rollup, start=1):
            L.append(
                f"| {i} | {canonical_category(r['category'])} | {r['gpu_pct']} | {r['gpu_ms']} | {r['kernel_count']} |"
            )
    else:
        L.append("_No GPU kernels found in trace._")
    L.append("")
    n_compute = sum(1 for c in hot if c.get("bound_type") == "compute_bound")
    n_memory = sum(1 for c in hot if c.get("bound_type") == "memory_bound")
    if n_compute or n_memory:
        L.append(f"_Analytical roofline bound: {n_compute} compute-bound, {n_memory} memory-bound hot kernel(s)._")
        L.append("")

    # Top 10 kernels by optimization ROI.
    L.append("## Top 10 Kernels by Optimization Priority")
    L.append("")
    ranked = sorted(hot, key=lambda c: c.get("optimization_priority") or 0.0, reverse=True)[:10]
    if ranked:
        L.append(
            "_Priority = GPU% x (1 - efficiency): high-impact, low-efficiency kernels "
            "first. Full per-kernel metrics in the CSV linked below._"
        )
        L.append("")
        L.append("| # | kernel_id | Name | Category | GPU% | Bound | AI | Eff% | Priority | Suggestion |")
        L.append("|---|-----------|------|----------|------|-------|----|----|---------|------------|")
        for i, c in enumerate(ranked, start=1):
            ai = c.get("arithmetic_intensity")
            ai_str = f"{float(ai):.3g}" if isinstance(ai, (int, float)) else "\u2014"
            eff = c.get("roofline_attainment_pct")
            eff_str = f"{float(eff):.1f}%" if isinstance(eff, (int, float)) else "\u2014"
            L.append(
                f"| {i} | `{c.get('kernel_id', '')}` | {c.get('name', '')} | {canonical_category(c.get('kernel_category', ''))} "
                f"| {float(c.get('gpu_pct') or 0.0):.2f}% | {c.get('bound_type', '')} | {ai_str} | {eff_str} "
                f"| {float(c.get('optimization_priority') or 0.0):.2f} | {c.get('suggestion', '')} |"
            )
    else:
        L.append("_No GPU kernels found in trace._")
    L.append("")

    # Compute Kernel Optimizations (per-candidate insight/action/source/impact).
    L.append("## Compute Kernel Optimizations")
    L.append("")
    if not routable:
        L.append("_No rewritable compute-kernel candidates identified._")
        L.append("")
    else:
        with_source = [c for c in routable if c.get("source_file")]
        L.append(
            f"_{len(dispatchable)} of {len(routable)} rewritable candidate(s) are auto-dispatchable "
            f"(resolved editable source AND a dispatch-grade operand shape); "
            f"{len(with_source) - len(dispatchable)} have a source but only a geometry shape, "
            f"and the rest need a source first._"
        )
        L.append("")
        for i, c in enumerate(routable, start=1):
            _cat = canonical_category(c["kernel_category"])
            L.append(f"### P{i}: {c['name']} ({_cat})")
            L.append("")
            L.append(
                f"**Insight**: {_cat} kernel consuming "
                f"{c['gpu_pct']:.2f}% of GPU time across {c['call_count']} launches."
            )
            L.append("")
            L.append(f"**Action**: {_ACTION_BY_CATEGORY.get(c['kernel_category'], _ACTION_BY_CATEGORY['Others'])}")
            L.append("")
            src = c.get("source_file") or ""
            if src:
                tg = (c.get("task_group") or {}).get("task_group_id") or ""
                _prov = c.get("shape_provenance") or "unresolved"
                _dispatch_note = (
                    "dispatch-grade shape"
                    if c.get("shape_dispatchable")
                    else f"geometry-only shape ({_prov}) — NOT auto-dispatchable, needs operand dims"
                )
                L.append(
                    f"**Source**: `{src}` (via {c.get('source_resolution_method') or 'unknown'}); "
                    f"{_dispatch_note}" + (f"; task group `{tg}`" if tg else "") + "."
                )
            else:
                L.append(
                    "**Source**: unresolved — not auto-dispatchable (rewritable by classification, "
                    "but no editable source was located for its launching op)."
                )
            L.append("")
            bound = c.get("bound_type") or "\u2014"
            eff = c.get("roofline_attainment_pct")
            eff_str = f"{float(eff):.1f}%" if isinstance(eff, (int, float)) else "\u2014"
            L.append(
                f"**Impact**: {c['gpu_pct']:.2f}% of GPU time; bound={bound}, "
                f"attainment={eff_str}, priority={float(c.get('optimization_priority') or 0.0):.2f} "
                f"(roofline_source={c.get('roofline_source', 'placeholder')})."
            )
            L.append("")

    # Task Groups (source-function dispatch grouping).
    task_groups = candidates.get("task_groups") or []
    if task_groups:
        L.append("## Task Groups")
        L.append("")
        L.append(
            "_Repeated shapes of one rewritable operator collapse into a single dispatch with all observed cases._"
        )
        L.append("")
        L.append("| Group | Source | Kernels | GPU % | Time (ms) |")
        L.append("|-------|--------|---------|-------|-----------|")
        for g in task_groups:
            src_disp = (g.get("source_path", "") or "?").split("/")[-1] or "?"
            L.append(
                f"| {g.get('task_group_id', '')} | {src_disp} | {len(g.get('kernel_ids') or [])} "
                f"| {g.get('aggregate_gpu_pct', 0)} | {round(float(g.get('aggregate_duration_us', 0) or 0) / 1000.0, 3)} |"
            )
        L.append("")

    # Non-rewritable note.
    skipped = candidates.get("skipped_kernels") or []
    if skipped:
        L.append(
            f"_{len(skipped)} hot kernel(s) are non-rewritable "
            f"(vendor library / unresolved source) — see Detailed Analysis._"
        )
        L.append("")

    # Detailed Analysis (per hot kernel).
    L.append("## Detailed Analysis")
    L.append("")
    for c in hot:
        L.append(f"### {c['kernel_id']}: {c['name']} ({canonical_category(c['kernel_category'])})")
        L.append("")
        L.append(
            f"**Identification:** {c['gpu_pct']:.2f}% GPU time, {c['call_count']} launches, "
            f"reusable={c['reusable_native_kernel']}"
            + (f", skip_reason={c['skip_reason']}" if not c["reusable_native_kernel"] else "")
            + "."
        )
        L.append("")
        L.append(f"**Data:** device kernel `{c['device_kernel_name']}`; duration {c['duration_us'] / 1000.0:.2f} ms.")
        L.append("")
        src = c.get("source_file") or ""
        L.append(
            f"**Source:** {('`' + src + '`') if src else 'unresolved'} "
            f"(shape provenance: {c.get('shape_provenance', 'unresolved')})."
        )
        L.append("")
        _eff = c.get("roofline_attainment_pct")
        _eff_s = f"{float(_eff):.1f}%" if isinstance(_eff, (int, float)) else _UNKNOWN_BOUND
        _ai = c.get("arithmetic_intensity")
        _ai_s = f"{float(_ai):.3g}" if isinstance(_ai, (int, float)) else _UNKNOWN_BOUND
        _bound = c.get("bound_type") or _UNKNOWN_BOUND
        L.append(
            f"**Roofline:** bound={_bound}, AI={_ai_s}, "
            f"attainment={_eff_s}, priority={float(c.get('optimization_priority') or 0.0):.2f} "
            f"(roofline_source={c.get('roofline_source', 'placeholder')})."
        )
        L.append("")
        L.append(f"**Suggested action:** {c.get('suggestion', '')}")
        L.append("")

    # Appendix.
    L.append("## Appendix")
    L.append("")
    L.append(f"- Framework: {framework or 'unknown'}")
    L.append(f"- Platform: {target_platform or 'unknown'}")
    L.append(f"- Throughput unit: {throughput_unit}")
    L.append(f"- Aggregation scope: {scope}")
    L.append(f"- Events scanned: {analyze_out.get('event_total', 0)}")
    L.append(
        f"- Attribution: {attribution.get('attributed_kernels', 0)}/"
        f"{attribution.get('kernel_count', 0)} kernels linked to an op "
        f"({attribution.get('attributed_pct', 0)}% of GPU time)"
    )
    L.append("")

    # Structured CSV export (full data; code-generated, machine-readable).
    if metrics_csv_path or summary_csv_path:
        L.append("## Structured Metrics (CSV)")
        L.append("")
        L.append("_Code-generated (no LLM). The Top-10 table above is a preview; these CSVs carry the full data._")
        L.append("")
        if metrics_csv_path:
            L.append(f"- Per-kernel metrics (all hot kernels): `{metrics_csv_path}`")
        if summary_csv_path:
            L.append(f"- Category summary: `{summary_csv_path}`")
        L.append("")
    return "\n".join(L)


def build_workload_roofline_totals(
    analyze_out: dict[str, Any],
    *,
    target_platform: str,
) -> dict[str, Any]:
    """Aggregate the analytical roofline over ALL analyzed device kernels.

    The per-kernel candidate list is capped at ``top_k``; the WORKLOAD roofline
    must instead cover every device kernel so it is not truncated to the hottest
    few. Classifies + computes the analytical roofline inline for each kernel and
    accumulates the same totals shape as
    :func:`diffusion_roofline.aggregate_bypass_candidates`, weighting
    ``sigma_ideal`` by the binding-side attainment.

    Args:
        analyze_out: Result of :func:`_bypass_trace_reader.analyze_trace`.
        target_platform: GPU platform tag for the peak lookup.

    Returns:
        Workload totals keyed like ``aggregate_unified`` /
        ``aggregate_bypass_candidates``.
    """
    sigma_actual = 0.0
    sigma_ideal = 0.0
    compute_us = 0.0
    memory_us = 0.0
    no_model_us = 0.0
    for k in analyze_out.get("kernels") or []:
        dur = float(k.get("gpu_time_us") or 0.0)
        if dur <= 0:
            continue
        sigma_actual += dur
        kc = classify_kernel(k.get("name", "") or "", op_name=k.get("op_name", "") or "")
        _count = k.get("count") or 0
        # Match the candidate-side waterfall for dispatch-grade shapes: own
        # cpu_op dims, then the same-name capture-time backfill. Geometry
        # fallbacks (launch_grid/tile_name) are intentionally excluded so they
        # do not poison the analytical roofline.
        shape_entries = _trace_shape_entries(k.get("op_shapes") or [], k.get("op_dtypes") or [], _count)
        if not shape_entries:
            shape_entries = _trace_shape_entries(k.get("backfill_shapes") or [], k.get("backfill_dtypes") or [], _count)
        rl = (
            compute_roofline(
                category=kc.category,
                shape_str=shape_entries[0]["shape"] if shape_entries else "",
                gpu_time_us=dur,
                call_count=int(k.get("count") or 1),
                gpu_type=target_platform,
            )
            if shape_entries
            else None
        )
        attain = rl.get("roofline_attainment_pct") if rl else None
        if rl and isinstance(attain, (int, float)) and attain > 0:
            sigma_ideal += dur * (float(attain) / 100.0)
            bound = str(rl.get("bound_type") or "").upper()
            if "COMPUTE" in bound:
                compute_us += dur
            elif "MEMORY" in bound:
                memory_us += dur
            else:
                no_model_us += dur
        else:
            no_model_us += dur
    kernel_eff = (sigma_ideal / sigma_actual) if sigma_actual > 0 else 0.0
    return {
        "sigma_actual_kernel_us": round(sigma_actual, 3),
        "sigma_ideal_roofline_us": round(sigma_ideal, 3),
        "kernel_roofline_efficiency": kernel_eff,
        "compute_bound_us": round(compute_us, 3),
        "memory_bound_us": round(memory_us, 3),
        "no_perf_model_us": round(no_model_us, 3),
    }


def build_kernel_roofline(
    candidates: dict[str, Any],
    *,
    analysis_md_path: str,
    kernel_candidates_path: str,
) -> dict[str, Any]:
    """Build the per-kernel roofline sidecar payload.

    Hardware roofline fields are estimated from the analytical model.

    Args:
        candidates: Output of :func:`build_candidates`.
        analysis_md_path: Path to the written ``analysis.md``.
        kernel_candidates_path: Path to the written ``kernel_candidates.json``.

    Returns:
        The ``kernel_roofline.json`` payload dict.
    """
    rows = []
    for c in candidates.get("hot_kernels") or []:
        rows.append(
            {
                "kernel_id": c["kernel_id"],
                "name": c["name"],
                "kernel_category": c["kernel_category"],
                "duration_us": c["duration_us"],
                "gpu_pct": c["gpu_pct"],
                "call_count": c["call_count"],
                "bound_type": c["bound_type"],
                # bottleneck falls back to bound_type; roofline_name has no
                # analytical analogue so it stays null.
                "bottleneck": c.get("bottleneck") or c.get("bound_type"),
                "roofline_name": c.get("roofline_name"),
                "suggestion": c.get("suggestion") or "",
                "recommended_actions": list(c.get("recommended_actions") or []),
                "efficiency_percent": c["efficiency_percent"],
                # Binding-side attainment (compute util if compute-bound, else bw).
                "roofline_attainment_pct": c.get("roofline_attainment_pct"),
                "arithmetic_intensity": c["arithmetic_intensity"],
                "compute_utilization_pct": c["compute_utilization_pct"],
                "bandwidth_utilization_pct": c["bandwidth_utilization_pct"],
                "reusable_native_kernel": c["reusable_native_kernel"],
                "source_file": c["source_file"],
                "rocprof_roofline": c["rocprof_roofline"],
                "flops_per_byte": c.get("flops_per_byte"),
                # roofline_source is how the bound was derived.
                "roofline_measured": c.get("roofline_measured", False),
                "roofline_source": c.get("roofline_source", _RL_PLACEHOLDER),
            }
        )
    return {
        "source": "bypass",
        "analysis_md_path": analysis_md_path,
        "kernel_candidates_path": kernel_candidates_path,
        "kernels": rows,
    }


def build_fusion(analyze_out: dict[str, Any]) -> dict[str, Any]:
    """Build the kernel-fusion opportunity payload from the launch sequence.

    Classifies each time-ordered launch (device name + launching op) and finds
    fusable clusters + adjacent transitions (see :mod:`_bypass_fusion`). Returns
    an empty payload when the reader did not emit ``kernel_launches``.

    Args:
        analyze_out: Result of :func:`_bypass_trace_reader.analyze_trace`
            (with ``emit_launches=True``).

    Returns:
        The ``kernel_sequence`` payload dict.
    """
    launches = analyze_out.get("kernel_launches") or []
    # Memoize classification by (device name, op name): few distinct kernels but
    # many launches, so caching keeps this O(distinct) not O(launches x rules).
    _cat_cache: dict[tuple[str, str], str] = {}

    def _category(name: str, op_name: str) -> str:
        key = (name, op_name)
        cat = _cat_cache.get(key)
        if cat is None:
            cat = classify_kernel(name, op_name=op_name).category
            _cat_cache[key] = cat
        return cat

    categorized = [
        {
            "name": lc.get("name", ""),
            "op_name": lc.get("op_name", ""),
            "category": _category(lc.get("name", "") or "", lc.get("op_name", "") or ""),
            "ts": lc.get("ts", 0.0),
            "dur": lc.get("dur", 0.0),
        }
        for lc in launches
    ]
    payload = analyze_fusion(categorized)
    payload["source"] = "bypass"
    payload["aggregation_scope"] = analyze_out.get("aggregation_scope", "full_trace")
    return payload
