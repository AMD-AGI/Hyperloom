# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve operation and framework identity from kernel source files."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from kernelforge.knowledge.kb_store.identity.implementation import canonical_owner_framework

log = logging.getLogger(__name__)

_UNKNOWN = "unknown"
_FRAMEWORKS = ("aiter", "sglang", "vllm")
_NO_FRAMEWORK_SENTINELS = {"standalone", "none", "unknown"}


def resolve_operation(kernel_source: str, kernel_path: str, target_functions: list[str] | None = None) -> str:
    """Return the operation identity (the entry function name, not the file name)."""

    def _pick(names: list[str]) -> str | None:
        preferred = [name for name in names if not name.lower().startswith(("launch", "main", "wrapper", "run_"))]
        if preferred:
            return preferred[0]
        return names[0] if names else None

    try:
        from kernelforge.mcp_server.tools.pmc import derive_kernel_names

        # Anchor source order is stable for the same file, so keep it (the first compute kernel is usually the primary
        # one, helpers come later).
        picked = _pick(derive_kernel_names(kernel_source or ""))
        if picked:
            return picked
    except Exception as exc:  # noqa: BLE001 - best-effort; fall back below
        log.debug("resolve_operation: derive_kernel_names failed: %r", exc)
    # Fallback: order-independent (sorted, de-duplicated) so producer/consumer converge even when their
    # target-function lists are ordered differently.
    candidates = sorted({function.strip() for function in (target_functions or []) if function and function.strip()})
    picked = _pick(candidates)
    if picked:
        return picked
    return Path(kernel_path).stem


def detect_backend_language(kernel_backend: str) -> str:
    """Derive the implementation language exclusively from the selected kernel backend."""
    language = str(kernel_backend or "").split("-", 1)[0].strip().lower()
    return language or _UNKNOWN


def detect_framework(kernel_path: str, framework_override: str = "") -> str:
    """Detect the framework that owns a kernel source path."""
    raw_framework = (framework_override or "").strip().lower()
    framework = canonical_owner_framework(raw_framework)
    if raw_framework:
        if raw_framework in _NO_FRAMEWORK_SENTINELS:
            return _UNKNOWN
        return framework
    path_parts = {canonical_owner_framework(part) for part in Path(kernel_path).parts}
    for framework_name in _FRAMEWORKS:
        if framework_name in path_parts:
            return framework_name
    return _UNKNOWN


def _read_text_safe(
    path: str,
    source_contents: dict[str, str] | None = None,
) -> str:
    if source_contents is not None:
        candidates = (str(path), str(Path(path).resolve()))
        for candidate in candidates:
            if candidate in source_contents:
                return source_contents[candidate]
    try:
        return Path(path).read_text(errors="replace")
    except Exception:  # noqa: BLE001 - best-effort
        return ""


def find_defining_source(
    operation: str,
    anchor_path: str,
    anchor_source: str,
    source_files: list[str] | None,
    *,
    source_contents: dict[str, str] | None = None,
) -> str:
    """Return the source text that defines an operation."""
    if not operation:
        return anchor_source or ""
    definition_pattern = re.compile(r"\bdef\s+" + re.escape(operation) + r"\b")
    global_pattern = re.compile(r"__global__[^\n]*\b" + re.escape(operation) + r"\b")
    if definition_pattern.search(anchor_source or "") or global_pattern.search(anchor_source or ""):
        return anchor_source or ""
    for source_file in source_files or []:
        text = _read_text_safe(source_file, source_contents)
        if text and (definition_pattern.search(text) or global_pattern.search(text)):
            return text
    return anchor_source or ""


def find_defining_path(
    operation: str,
    anchor_path: str,
    anchor_source: str,
    source_files: list[str] | None,
    *,
    source_contents: dict[str, str] | None = None,
) -> str:
    """Return the path of the file that defines an operation."""
    if not operation:
        return anchor_path
    definition_pattern = re.compile(r"\bdef\s+" + re.escape(operation) + r"\b")
    global_pattern = re.compile(r"__global__[^\n]*\b" + re.escape(operation) + r"\b")
    if definition_pattern.search(anchor_source or "") or global_pattern.search(anchor_source or ""):
        return anchor_path
    for source_file in source_files or []:
        text = _read_text_safe(source_file, source_contents)
        if text and (definition_pattern.search(text) or global_pattern.search(text)):
            return source_file
    return anchor_path


def infer_source_owner_framework(
    *,
    kernel_path: str,
    kernel_source: str,
    target_functions: list[str] | None = None,
    source_files: list[str] | None = None,
    framework_override: str = "",
    source_contents: dict[str, str] | None = None,
    concrete_operation: str = "",
) -> str:
    """Resolve the canonical framework that owns the concrete operation."""
    operation = concrete_operation or resolve_operation(
        kernel_source,
        kernel_path,
        target_functions=target_functions,
    )
    defining_path = find_defining_path(
        operation,
        kernel_path,
        kernel_source,
        source_files,
        source_contents=source_contents,
    )
    return detect_framework(
        defining_path,
        framework_override=framework_override,
    )


__all__ = [
    "detect_backend_language",
    "detect_framework",
    "find_defining_path",
    "find_defining_source",
    "infer_source_owner_framework",
    "resolve_operation",
]
