# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Normalize grouped kernel workload cases for backend execution."""

from __future__ import annotations

import json
import re
from typing import Any


_SHAPE_RE = re.compile(r"\(([\d,\s]*)\)")
_NAMED_DIMENSIONS = ("M", "N", "K", "E", "TOPK")
CASE_SELECTOR_KEY = "CASE_ID"


def normalize_operation_key(operation: str) -> str:
    """Remove balanced C++ template arguments from an operation identity."""
    value = str(operation).strip()
    if "<" not in value:
        return value
    normalized: list[str] = []
    depth = 0
    for character in value:
        if character == "<":
            depth += 1
        elif character == ">":
            if depth > 0:
                depth -= 1
        elif depth == 0:
            normalized.append(character)
    result = "".join(normalized).strip()
    return result or value


def native_operation_key(operation: str) -> str:
    """Return a stable native operator identity across template instances."""
    normalized = str(operation or "").strip()
    normalized = re.sub(
        r"^[A-Za-z][A-Za-z0-9_]*->",
        "",
        normalized,
    ).strip()
    normalized = re.sub(
        r"^(?:void|bool|int|unsigned|long|short|char|float|double|size_t)\s+",
        "",
        normalized,
    ).strip()
    normalized = re.sub(
        r"\s*\([^()]*\bOp\)\s*$",
        "",
        normalized,
    ).strip()
    if normalized.endswith(".kd"):
        normalized = normalized[:-3].strip()
    normalized = normalize_operation_key(normalized)
    if not normalized.startswith(("_Z", "__Z")):
        return normalized

    mangled = normalized[1:] if normalized.startswith("__Z") else normalized
    index = 3 if mangled.startswith("_ZN") else 2
    components: list[str] = []
    while index < len(mangled):
        if not mangled[index].isdigit():
            break
        end = index
        while end < len(mangled) and mangled[end].isdigit():
            end += 1
        length = int(mangled[index:end])
        component = mangled[end : end + length]
        if len(component) != length:
            break
        components.append(component)
        index = end + length
    return "::".join(components) if components else normalized


def _shape_entries(row: dict[str, Any]) -> list[Any]:
    """Return one candidate row's shape entries without changing their order."""
    raw = row.get("input_shapes")
    if raw in (None, "", []):
        raw = row.get("shapes")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, (dict, str, tuple)):
        return [raw]
    return []


def tensor_dim_lists(row: dict[str, Any]) -> list[list[int]]:
    """Extract ordered tensor dimensions from supported TraceLens shape forms."""
    dimensions: list[list[int]] = []
    for entry in _shape_entries(row):
        value = entry.get("shape") if isinstance(entry, dict) else entry
        if isinstance(value, (list, tuple)) and value and all(isinstance(item, int) for item in value):
            dimensions.append([int(item) for item in value])
            continue
        if not isinstance(value, str):
            continue
        for match in _SHAPE_RE.finditer(value):
            parsed = [int(item) for item in match.group(1).split(",") if item.strip().isdigit()]
            if parsed:
                dimensions.append(parsed)
    return dimensions


def _gemm_dimensions(shapes: list[list[int]]) -> dict[str, int]:
    two_dimensional = [shape for shape in shapes if len(shape) == 2]
    for lhs in two_dimensional:
        for rhs in two_dimensional:
            if lhs is not rhs and lhs[1] == rhs[0]:
                return {"M": lhs[0], "K": lhs[1], "N": rhs[1]}
    for lhs in two_dimensional:
        for rhs in two_dimensional:
            if lhs is not rhs and lhs[1] == rhs[1]:
                return {"M": lhs[0], "K": lhs[1], "N": rhs[0]}
    if two_dimensional:
        return {"M": two_dimensional[0][0], "K": two_dimensional[0][1]}
    return {}


def _moe_dimensions(shapes: list[list[int]]) -> dict[str, int]:
    two_dimensional = [shape for shape in shapes if len(shape) == 2]
    three_dimensional = [shape for shape in shapes if len(shape) == 3]
    dimensions: dict[str, int] = {}
    hidden = max(two_dimensional, key=lambda shape: shape[1]) if two_dimensional else None
    if hidden is not None:
        dimensions["M"], dimensions["K"] = hidden[0], hidden[1]
        topk = next(
            (
                shape
                for shape in two_dimensional
                if shape is not hidden and shape[0] == hidden[0] and 0 < shape[1] <= 64
            ),
            None,
        )
        if topk is not None:
            dimensions["TOPK"] = topk[1]
    if three_dimensional:
        dimensions["E"] = three_dimensional[0][0]
        hidden_size = dimensions.get("K")
        output_size = None
        if hidden_size is not None:
            second_weight = next(
                (shape for shape in three_dimensional if shape[1] == hidden_size),
                None,
            )
            if second_weight is not None:
                output_size = second_weight[2]
            else:
                first_weight = next(
                    (shape for shape in three_dimensional if shape[2] == hidden_size),
                    None,
                )
                if first_weight is not None:
                    output_size = first_weight[1] // 2
        if output_size is not None:
            dimensions["N"] = output_size
    return dimensions


def named_dimensions(row: dict[str, Any]) -> dict[str, int]:
    """Derive backend-friendly dimensions while retaining a generic case ID."""
    operation = f"{row.get('operation') or ''} {row.get('name') or ''}".lower()
    shapes = tensor_dim_lists(row)
    if "moe" in operation:
        dimensions = _moe_dimensions(shapes)
    elif any(token in operation for token in ("gemm", "matmul", "_mm", "linear")):
        dimensions = _gemm_dimensions(shapes)
    else:
        dimensions = {}
    if dimensions:
        return dimensions

    entries = _shape_entries(row)
    first = entries[0] if entries else None
    if isinstance(first, dict):
        return {key: int(first[key]) for key in _NAMED_DIMENSIONS if isinstance(first.get(key), int)}
    return {}


def _case_signature(row: dict[str, Any]) -> str:
    """Return a deterministic identity for one exact invocation case."""
    signature_shapes = [
        {key: value for key, value in entry.items() if key not in {"call_num", "call_count"}}
        if isinstance(entry, dict)
        else entry
        for entry in _shape_entries(row)
    ]
    payload = {
        "operation": str(row.get("operation") or row.get("name") or ""),
        "input_shapes": signature_shapes,
        "input_dtypes": row.get("input_dtypes") or row.get("dtypes") or [],
        "output_shapes": row.get("output_shapes") or [],
        "output_dtypes": row.get("output_dtypes") or [],
        "raw_arg_spec": row.get("raw_arg_spec") or {},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _expanded_group_rows(group: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand per-candidate CSV invocation evidence into independent cases."""
    expanded: list[dict[str, Any]] = []
    for row in group.get("rows") or []:
        if not isinstance(row, dict):
            continue
        invocation_cases = row.get("invocation_cases")
        if not isinstance(invocation_cases, list) or not invocation_cases:
            expanded.append(row)
            continue
        for invocation_case in invocation_cases:
            if not isinstance(invocation_case, dict):
                continue
            merged = dict(row)
            for key in (
                "operation",
                "input_shapes",
                "input_dtypes",
                "output_shapes",
                "output_dtypes",
                "raw_arg_spec",
            ):
                if key in invocation_case:
                    merged[key] = invocation_case[key]
            merged["call_count"] = invocation_case.get("call_count", 0)
            expanded.append(merged)
    return expanded


def build_task_group_shape_cases(group: dict[str, Any]) -> list[dict[str, Any]]:
    """Build distinct, primary-first workload cases for one task group."""
    rows = _expanded_group_rows(group)
    primary_kernel_id = str(group.get("primary_kernel_id") or "")

    def _duration(row: dict[str, Any]) -> float:
        try:
            return float(row.get("duration_us") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    rows.sort(
        key=lambda row: (
            str(row.get("kernel_id") or "") != primary_kernel_id,
            -_duration(row),
        )
    )

    cases: list[dict[str, Any]] = []
    by_signature: dict[str, dict[str, Any]] = {}
    for row in rows:
        signature = _case_signature(row)
        kernel_id = str(row.get("kernel_id") or "")
        existing = by_signature.get(signature)
        if existing is not None:
            if kernel_id and kernel_id not in existing["kernel_ids"]:
                existing["kernel_ids"].append(kernel_id)
            try:
                existing["call_count"] += int(row.get("call_count") or 0)
            except (TypeError, ValueError):
                pass
            continue

        case: dict[str, Any] = {
            "kernel_ids": [kernel_id] if kernel_id else [],
            "operation": str(row.get("operation") or row.get("name") or ""),
            "input_shapes": _shape_entries(row),
            "input_dtypes": row.get("input_dtypes") or row.get("dtypes") or [],
            "output_shapes": row.get("output_shapes") or [],
            "output_dtypes": row.get("output_dtypes") or [],
            "raw_arg_spec": row.get("raw_arg_spec") or {},
        }
        try:
            case["call_count"] = int(row.get("call_count") or 0)
        except (TypeError, ValueError):
            case["call_count"] = 0
        case["_named_dimensions"] = named_dimensions(row)
        cases.append(case)
        by_signature[signature] = case

    for index, case in enumerate(cases, start=1):
        case_id = f"case_{index:03d}"
        dimensions = case.pop("_named_dimensions")
        case["case_id"] = case_id
        case["selector"] = {CASE_SELECTOR_KEY: case_id, **dimensions}
    return cases


def task_group_shape_cases(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized cases from a candidate's task group."""
    group = candidate.get("task_group")
    if not isinstance(group, dict):
        return []
    existing = group.get("shape_cases")
    if isinstance(existing, list) and all(
        isinstance(case, dict) and isinstance(case.get("selector"), dict) for case in existing
    ):
        return existing
    return build_task_group_shape_cases(group)


def forge_shapes_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Build Forge primary/minimal/validation selectors for a candidate."""
    cases = task_group_shape_cases(candidate)
    if cases:
        selectors = [dict(case["selector"]) for case in cases]
        primary = selectors[0]
        return {
            "primary": primary,
            "minimal": primary,
            "validation": selectors,
        }

    primary = named_dimensions(candidate)
    return {
        "primary": primary,
        "minimal": primary,
        "validation": [primary] if primary else [],
    }
