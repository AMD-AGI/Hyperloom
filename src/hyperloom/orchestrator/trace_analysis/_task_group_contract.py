# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The operator identity a task group is keyed by, and the numbering of finished groups."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

OPERATOR_IDENTITY_VERSION = 3

# Native device-code extensions; these sources key a task group as ``native``.
NATIVE_SOURCE_SUFFIXES = (".cu", ".cuh", ".hip", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h", ".c")


def is_native_source(path: str) -> bool:
    """True for C/C++/HIP/CUDA source files."""
    return str(path).lower().endswith(NATIVE_SOURCE_SUFFIXES)


def _strip_dispatch_decoration(operation: str) -> str:
    """Remove how-a-kernel-was-launched decoration from a traced operation name."""
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


def native_operation_key(operation: str) -> str:
    """Return a stable native operator identity across template instances."""
    normalized = normalize_operation_key(operation)
    if not normalized.startswith(("_Z", "__Z")):
        return normalized

    mangled = normalized[1:] if normalized.startswith("__Z") else normalized
    nested = mangled.startswith("_ZN")
    index = 3 if nested else 2
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
        if not nested:
            break
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


def finalize_task_groups(groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Number task groups by descending aggregate time, then pick each group's primary row."""
    ordered = sorted(groups, key=lambda g: g["aggregate_duration_us"], reverse=True)
    for idx, group in enumerate(ordered, start=1):
        group["task_group_id"] = f"tg{idx:03d}"
        group["rows"].sort(key=lambda r: float(r.get("duration_us") or 0.0), reverse=True)
        primary = group["rows"][0]
        group["primary_kernel_id"] = str(primary.get("kernel_id") or primary.get("name") or "")
        group["aggregate_duration_us"] = round(group["aggregate_duration_us"], 3)
        group["aggregate_gpu_pct"] = round(group["aggregate_gpu_pct"], 3)
    return ordered
