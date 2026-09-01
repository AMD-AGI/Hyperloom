# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Normalize grouped kernel workload cases for backend execution."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_SHAPE_RE = re.compile(r"\(([\d,\s]*)\)")
_NAMED_DIMENSIONS = ("M", "N", "K", "E", "TOPK")
CASE_SELECTOR_KEY = "CASE_ID"
OPERATOR_IDENTITY_VERSION = 3


def _strip_dispatch_decoration(operation: str) -> str:
    """Remove how-a-kernel-was-launched decoration from a traced operation name.

    The launch API, C return type and synthetic-op suffix describe the dispatch,
    not the operator, so keying on them splits one kernel into a task group per
    launch path and ports its source once per group.
    """
    value = str(operation or "").strip()
    value = re.sub(r"^[A-Za-z][A-Za-z0-9_]*->", "", value).strip()
    value = re.sub(
        r"^(?:void|bool|int|unsigned|long|short|char|float|double|size_t)\s+",
        "",
        value,
    ).strip()
    value = re.sub(r"\s*\([^()]*\bOp\)\s*$", "", value).strip()
    if value.endswith(".kd"):
        value = value[:-3].strip()
    return value


def _strip_template_arguments(value: str) -> str:
    """Remove balanced C++ template arguments from an already-clean symbol."""
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


def normalize_operation_key(operation: str) -> str:
    """Remove launch decoration and balanced C++ template arguments."""
    return _strip_template_arguments(_strip_dispatch_decoration(operation))


def logical_operator_name(candidate: dict[str, Any] | None) -> str:
    """Return the stable logical operation for Forge, launch API stripped.

    The trace names a candidate after both rows it occupies -- the CPU-side
    launch call and the device kernel -- so the same kernel reads
    ``hipModuleLaunchKernel->_gqa_sparse_fwd_kernel`` in one analysis and
    ``_gqa_sparse_fwd_kernel`` in another, depending on whether the profiler
    happened to record the parent that pairs them. Which of those a run sees is
    not a property of the kernel, and Forge keys its experience store on this
    name: two profiles of one configuration then write two identities, and the
    warm-start read of either reports no prior record.

    So the launch call comes off. ``native_operation_key`` already owns that
    normalization for the task-group identity, along with demangling and
    return-type removal, and both shapes of the name reduce to the same key
    under it. How the kernel was launched stays worth knowing -- it is what
    tells a shapeless candidate from a recorded one -- but it belongs beside the
    identity rather than inside it.
    """
    candidate = candidate or {}
    task_group = candidate.get("task_group")
    identity = task_group.get("operator_identity") if isinstance(task_group, dict) else None
    raw = (
        (identity.get("operation") if isinstance(identity, dict) else "")
        or candidate.get("operation")
        or candidate.get("name")
        or ""
    )
    normalized = native_operation_key(str(raw).strip())
    return re.sub(r"\s*::\s*", "::", normalized)


def native_operation_key(operation: str) -> str:
    """Return a stable native operator identity across template instances.

    Adds Itanium demangling to the shared normalization; every other step is
    language-independent and lives in :func:`normalize_operation_key`.
    """
    normalized = normalize_operation_key(operation)
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


def canonical_source_path(source_path: str) -> str:
    """Return one route-independent absolute source identity."""
    value = str(source_path or "").strip()
    if not value:
        return ""
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return str(Path(value).expanduser().absolute())


def build_operator_identity(
    *,
    source_kind: str,
    source_path: str,
    operation: str,
    function_name: str = "",
) -> dict[str, Any]:
    """Build the versioned operator identity shared by all TraceLens routes."""
    kind = "native" if str(source_kind).lower() == "native" else "py"
    operation_key = native_operation_key(operation) if kind == "native" else normalize_operation_key(operation)
    identity = {
        "version": OPERATOR_IDENTITY_VERSION,
        "source_kind": kind,
        "source_path": canonical_source_path(source_path),
        "operation": operation_key,
    }
    function_key = native_operation_key(function_name) if kind == "native" else str(function_name or "").strip()
    if function_key:
        identity["function"] = function_key
    return identity


def operator_identity_key(
    *,
    source_kind: str,
    source_path: str,
    operation: str,
    function_name: str = "",
) -> str:
    """Serialize a canonical operator identity as a stable ledger key."""
    return json.dumps(
        build_operator_identity(
            source_kind=source_kind,
            source_path=source_path,
            operation=operation,
            function_name=function_name,
        ),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def legacy_operator_identity_keys(
    *,
    source_kind: str,
    source_path: str,
    operation: str,
    function_name: str = "",
) -> list[str]:
    """Return both historical route-specific keys for state migration."""
    kind = "native" if str(source_kind).lower() == "native" else "py"
    raw_source = str(source_path or "").strip()
    sources = list(
        dict.fromkeys(
            [
                canonical_source_path(source_path),
                raw_source,
            ]
        )
    )
    # Reproduce v2 key shape (pre-decoration-strip) so warm-start can still match historical records.
    operation_key = (
        native_operation_key(operation) if kind == "native" else _strip_template_arguments(str(operation or "").strip())
    )
    function_keys = {
        (native_operation_key(function_name) if kind == "native" else str(function_name or "")),
        operation_key,
    }
    if kind == "native" and "::" in operation_key:
        function_keys.add(operation_key.rsplit("::", 1)[-1])
    function_keys.discard("")
    legacy = []
    legacy.append(
        operator_identity_key(
            source_kind=kind,
            source_path=source_path,
            operation=operation,
        )
    )
    for source in sources:
        legacy.append(
            json.dumps(
                (kind, source, operation_key),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        legacy.extend(
            json.dumps(
                (kind, operation_key, source, function_key),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for function_key in sorted(function_keys)
        )
    return list(dict.fromkeys(legacy))


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


def _is_attention_workload(row: dict[str, Any]) -> bool:
    """Classify attention generically from trace-visible semantic metadata."""
    labels = " ".join(
        str(row.get(key) or "")
        for key in (
            "operation",
            "name",
            "kernel_category",
            "tracelens_category",
        )
    ).lower()
    contract = row.get("kernel_contract")
    contract_kind = str(contract.get("kind") or "").lower() if isinstance(contract, dict) else ""
    return (
        "attention" in labels
        or "attn" in labels
        or contract_kind == "attention"
        or str(row.get("kernel_category") or "").strip().lower() == "sdpa"
    )


def _attention_dimensions(shapes: list[list[int]]) -> dict[str, int]:
    """Derive Q/K/V semantic dimensions from ordered rank-3 tensor shapes."""
    rank_three = [shape for shape in shapes if len(shape) == 3 and all(dimension > 0 for dimension in shape)]
    for index in range(len(rank_three) - 2):
        query, key, value = rank_three[index : index + 3]
        if key == value and query[0] == key[0] and query[2] == key[2] and query[1] >= key[1]:
            return {
                "QTOKENS": query[0],
                "QHEADS": query[1],
                "KVHEADS": key[1],
                "HEADSIZE": query[2],
            }
    return {}


def named_dimensions(row: dict[str, Any]) -> dict[str, int]:
    """Derive backend-friendly dimensions while retaining a generic case ID."""
    operation = f"{row.get('operation') or ''} {row.get('name') or ''}".lower()
    shapes = tensor_dim_lists(row)
    if _is_attention_workload(row):
        dimensions = _attention_dimensions(shapes)
    elif "moe" in operation:
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
                additional_call_count = int(row.get("call_count") or 0)
            except (TypeError, ValueError):
                additional_call_count = 0
            existing["call_count"] += additional_call_count
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
