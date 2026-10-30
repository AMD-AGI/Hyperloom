# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shapes and dtypes a traced op was called with, and the CSV rows they are read from."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._literal_utils import LITERAL_EVAL_ERRORS as _LITERAL_EVAL_ERRORS, safe_literal_eval as _safe_literal_eval


def _coerce_count(value: Any) -> int | None:
    """Coerce a loosely-typed call-count value into a positive int.

    Handles ``None``/empty, numpy-repr strings (``np.float64(...)``), and
    float-like strings.

    Args:
        value (Any): The raw count value to coerce.

    Returns:
        int | None: A positive integer, or ``None`` when absent, unparseable,
            or non-positive.
    """
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.startswith("np.float64(") and text.endswith(")"):
        text = text[len("np.float64(") : -1]
    try:
        count = int(float(text))
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def _merge_shape_call(target: list[Any], shape: Any, call_num: int) -> None:
    """Merge a shape/call-count pair into an accumulator list in place.

    If an entry with the same shape already exists, its ``call_num`` is
    incremented; otherwise a new entry is appended.

    Args:
        target (list[Any]): Accumulator list of ``{"shape", "call_num"}`` dicts.
        shape (Any): The shape value to merge.
        call_num (int): Call count to add for this shape.
    """
    for entry in target:
        if isinstance(entry, dict) and entry.get("shape") == shape:
            entry["call_num"] = int(entry.get("call_num") or 0) + call_num
            return
    target.append({"call_num": call_num, "shape": shape})


def _shape_call_entries(shapes: Any, call_num: Any = None) -> list[dict[str, Any]]:
    """Normalize a shapes payload into merged ``{shape, call_num}`` entries.

    Args:
        shapes (Any): A list of shape values or ``{"shape", "call_num"}`` dicts.
        call_num (Any): Default call count applied when an entry lacks one.

    Returns:
        list[dict[str, Any]]: Deduplicated shape entries with summed call
            counts; empty when ``shapes`` is not a list.
    """
    if not isinstance(shapes, list):
        return []
    count = _coerce_count(call_num) or 1
    entries: list[dict[str, Any]] = []
    for shape in shapes:
        if isinstance(shape, dict):
            value = shape.get("shape")
            shape_count = _coerce_count(shape.get("call_num")) or count
        else:
            value = shape
            shape_count = count
        if value in (None, "", [], ()):
            continue
        _merge_shape_call(entries, value, shape_count)
    return entries


# Recognized dtype tokens appearing as the suffix on a TraceLens shape entry
# (e.g. ``(64,5120) bf16``); used to recognise a trailing dtype on a paren-less entry.
_KNOWN_DTYPE_TOKENS = frozenset(
    {
        "bf16",
        "fp16",
        "fp32",
        "f32",
        "fp8",
        "f8",
        "fp8_e4m3",
        "fp8_e5m2",
        "e4m3",
        "e5m2",
        "int8",
        "int4",
        "int32",
        "int64",
        "int",
        "uint8",
        "bool",
        "float",
        "double",
        "half",
    }
)


def _split_shape_dtype(entry: Any) -> tuple[str, str]:
    """Split a TraceLens shape entry ``"(64,5120) bf16"`` into ``("(64,5120)", "bf16")``.

    Separates the inline dtype so the per-arg dtype can be surfaced as a
    positionally aligned ``input_dtypes`` list. Non-string / dict entries return
    ``(str(value), "")``; an empty scalar ``()`` keeps an empty dtype so arity
    stays aligned with the call signature.
    """
    if isinstance(entry, dict):
        entry = entry.get("shape", "")
    s = str(entry).strip()
    if not s:
        return "", ""
    if s.startswith("("):
        close = s.find(")")
        if close != -1:
            return s[: close + 1].strip(), s[close + 1 :].strip()
        return s, ""
    parts = s.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].lower() in _KNOWN_DTYPE_TOKENS:
        return parts[0].strip(), parts[1].strip()
    return s, ""


def _dtypes_from_shapes(shapes: Any) -> list[str]:
    """Ordered per-arg dtype list parsed from a candidate's ``shapes`` strings.

    Positionally aligned with the shapes (one entry per arg, ``""`` when no dtype
    was captured). Returns ``[]`` when no dtype is present on any entry, so callers
    keep the existing empty-default behaviour when TraceLens captured none.
    """
    if not isinstance(shapes, list):
        return []
    dtypes = [_split_shape_dtype(s)[1] for s in shapes]
    return dtypes if any(dtypes) else []


# Trace-anchored shape capture for the fused-MoE expert kernel: its top-level
# kernel event carries no resolvable ``Input Dims``, so the candidate emits
# ``shapes: []`` and the dispatch gate rejects it. ``ops_unique_args.csv`` still
# carries the trace-recorded operands, so recover missing shapes from it by exact
# op-name match (or fused-MoE marker match).
_FUSED_MOE_KERNEL_MARKER = "invoke_fused_moe_kernel"

# torch ``Input type`` token -> compact dtype suffix for the shape-string contract (e.g. ``(15360,2048) bf16``);
# unmapped/empty types emit a bare shape.
_FP16 = "fp16"
_FP32 = "fp32"
_DTYPE_SUFFIX: dict[str, str] = {
    # Suffixes MUST match the shared harness dtype_map + roofline peak table; a compact "f16"/"f32" makes the harness
    # emit an invalid ``torch.f16``.
    "c10::bfloat16": "bf16",
    "bfloat16": "bf16",
    "c10::half": _FP16,
    "half": _FP16,
    "float16": _FP16,
    "float": _FP32,
    "float32": _FP32,
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


def format_operand_shape(dims: Any, dtype: Any) -> str | None:
    """Render one operand as a ``(d0,d1,...) <dtype>`` string, or ``None`` for scalar/empty operands."""
    if not isinstance(dims, (list, tuple)) or not dims:
        return None
    try:
        body = ",".join(str(int(d)) for d in dims)
    except (TypeError, ValueError):
        return None
    shape = f"({body},)" if len(dims) == 1 else f"({body})"
    suffix = _DTYPE_SUFFIX.get(str(dtype or "").strip().lower())
    return f"{shape} {suffix}" if suffix else shape


def _resolve_shapes_from_ops_unique_args_csv(
    perf_report_csv_dir: Path | str | None,
    row_matches: Callable[[str], bool],
) -> list[str]:
    """Recover operand shapes from matching ``ops_unique_args.csv`` rows.

    Parses ``Input Dims`` / ``Input type`` tuple-of-tuples into TraceLens-style
    shape strings (e.g. ``(15360,2048) bf16``), deduped in first-seen order.

    Args:
        perf_report_csv_dir: Directory containing ``ops_unique_args.csv``.
        row_matches: Predicate called with the normalized row name.

    Returns:
        The recovered operand shape strings, or ``[]`` when no trusted rows
        are available.
    """
    if not perf_report_csv_dir:
        return []
    csv_path = Path(perf_report_csv_dir) / "ops_unique_args.csv"
    if not csv_path.is_file():
        return []
    shapes: list[str] = []
    seen: set[str] = set()
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = str(row.get("name") or "").strip().lower()
                if not row_matches(name):
                    continue
                try:
                    dims = _safe_literal_eval(str(row.get("Input Dims") or "").strip() or "()")
                    dtypes = _safe_literal_eval(str(row.get("Input type") or "").strip() or "()")
                except _LITERAL_EVAL_ERRORS:
                    continue
                if not isinstance(dims, (list, tuple)):
                    continue
                if not isinstance(dtypes, (list, tuple)):
                    dtypes = ()
                for i, operand in enumerate(dims):
                    dtype = dtypes[i] if i < len(dtypes) else ""
                    s = format_operand_shape(operand, dtype)
                    if s and s not in seen:
                        seen.add(s)
                        shapes.append(s)
    except (OSError, csv.Error):
        return []
    return shapes


def _invocation_case_from_csv_row(row: dict[str, str]) -> dict[str, Any] | None:
    """Preserve one ops_unique_args row as an exact invocation case."""
    raw_dims = str(row.get("Input Dims") or "").strip()
    raw_types = str(row.get("Input type") or "").strip()
    raw_concrete = str(row.get("Concrete Inputs") or "").strip()
    try:
        dims = _safe_literal_eval(raw_dims or "()")
        dtypes = _safe_literal_eval(raw_types or "()")
    except _LITERAL_EVAL_ERRORS:
        return None
    if not isinstance(dims, (list, tuple)):
        return None
    if not isinstance(dtypes, (list, tuple)):
        dtypes = ()

    operands: list[str] = []
    tensor_dtypes: list[str] = []
    for index, operand in enumerate(dims):
        dtype = dtypes[index] if index < len(dtypes) else ""
        rendered = format_operand_shape(operand, dtype)
        if rendered:
            operands.append(rendered)
            tensor_dtypes.append(str(dtype or ""))

    raw_arg_spec = {
        "input_dims": raw_dims,
        "input_type": raw_types,
        "concrete_inputs": raw_concrete,
    }
    if not operands and not any(raw_arg_spec.values()):
        return None
    return {
        "operation": str(row.get("name") or ""),
        "input_shapes": ([{"call_num": 1, "shape": "<br>".join(operands)}] if operands else []),
        "input_dtypes": tensor_dtypes,
        "raw_arg_spec": raw_arg_spec,
    }


def _resolve_invocation_cases_from_ops_unique_args_csv(
    perf_report_csv_dir: Path | str | None,
    row_matches: Callable[[str], bool],
) -> list[dict[str, Any]]:
    """Return distinct CSV invocations without flattening argument boundaries."""
    if not perf_report_csv_dir:
        return []
    csv_path = Path(perf_report_csv_dir) / "ops_unique_args.csv"
    if not csv_path.is_file():
        return []

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with csv_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                name = str(row.get("name") or "").strip().lower()
                if not row_matches(name):
                    continue
                case = _invocation_case_from_csv_row(row)
                if case is None:
                    continue
                signature = json.dumps(
                    case,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                cases.append(case)
    except (OSError, csv.Error):
        return []
    return cases


def resolve_fused_moe_shapes_from_csv(
    perf_report_csv_dir: Path | str | None,
) -> list[str]:
    """Recover the fused-MoE expert-kernel operand shapes from ``ops_unique_args.csv``."""
    return _resolve_shapes_from_ops_unique_args_csv(
        perf_report_csv_dir,
        lambda name: _FUSED_MOE_KERNEL_MARKER in name,
    )


def resolve_fused_moe_invocation_cases_from_csv(
    perf_report_csv_dir: Path | str | None,
) -> list[dict[str, Any]]:
    """Recover every fused-MoE invocation row as an independent case."""
    return _resolve_invocation_cases_from_ops_unique_args_csv(
        perf_report_csv_dir,
        lambda name: _FUSED_MOE_KERNEL_MARKER in name,
    )


def resolve_shapes_from_csv_for_op(
    perf_report_csv_dir: Path | str | None,
    op_name: str,
) -> list[str]:
    """Recover operand shapes for a candidate by exact TraceLens op name."""
    target = str(op_name or "").strip().lower()
    if not target:
        return []
    return _resolve_shapes_from_ops_unique_args_csv(
        perf_report_csv_dir,
        lambda name: name == target,
    )


def resolve_invocation_cases_from_csv(
    perf_report_csv_dir: Path | str | None,
    op_name: str,
) -> list[dict[str, Any]]:
    """Recover every exact-name invocation row without merging its arguments."""
    target = str(op_name or "").strip().lower()
    if not target:
        return []
    return _resolve_invocation_cases_from_ops_unique_args_csv(
        perf_report_csv_dir,
        lambda name: name == target,
    )


def resolve_raw_arg_spec_from_csv(
    perf_report_csv_dir: Path | str | None,
    op_name: str,
) -> dict[str, str] | None:
    """Return the matched ``ops_unique_args.csv`` row's raw arg columns verbatim.

    Unlike :func:`resolve_shapes_from_csv_for_op` (which drops scalar/empty
    operands), this preserves the full ordered argument metadata exactly as the
    trace recorded it — ``Input Dims`` / ``Input type`` / ``Concrete Inputs`` —
    so the GEAK harness builder can reconstruct the real call signature
    (tensors + scalar args in order) instead of inferring it from tensor shapes
    alone. No parsing, no reshaping: the strings are forwarded as-is.

    Args:
        perf_report_csv_dir: Directory containing ``ops_unique_args.csv``.
        op_name: Exact TraceLens op name to match (case-insensitive).

    Returns:
        ``{"input_dims", "input_type", "concrete_inputs"}`` from the first
        matching row, or ``None`` when unavailable.
    """
    target = str(op_name or "").strip().lower()
    if not target or not perf_report_csv_dir:
        return None
    csv_path = Path(perf_report_csv_dir) / "ops_unique_args.csv"
    if not csv_path.is_file():
        return None
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("name") or "").strip().lower() != target:
                    continue
                spec = {
                    "input_dims": str(row.get("Input Dims") or "").strip(),
                    "input_type": str(row.get("Input type") or "").strip(),
                    "concrete_inputs": str(row.get("Concrete Inputs") or "").strip(),
                }
                return spec if any(spec.values()) else None
    except (OSError, csv.Error):
        return None
    return None
