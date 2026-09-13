# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-name extraction from GPU kernel source."""

from __future__ import annotations

import re

# Decorator markers whose next `def <name>(` is a GPU kernel entry (FlyDSL/Triton).
_KERNEL_DECO_RE = re.compile(r"\.kernel\b|triton\.jit\b|\.jit\b", re.IGNORECASE)
_DEF_RE = re.compile(r"^\s*def\s+(\w+)\s*\(")
# HIP/CUDA __global__ entry points.
_GLOBAL_ATTR = r"(?:__launch_bounds__\s*\([^)]*\)|__attribute__\s*\(\([^)]*\)\)|\bstatic\b|\binline\b)"
_GLOBAL_RE = re.compile(
    rf"__global__\s+(?:{_GLOBAL_ATTR}\s+)*void\s+(?:{_GLOBAL_ATTR}\s*)*(\w+)\s*\(",
)


def derive_kernel_names(source: str) -> list[str]:
    """Best-effort list of GPU-kernel entry names declared in kernel source."""
    if not source:
        return []
    names: list[str] = []
    lines = source.splitlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("@") and _KERNEL_DECO_RE.search(s):
            for j in range(i + 1, min(i + 6, len(lines))):
                m = _DEF_RE.match(lines[j])
                if m:
                    names.append(m.group(1))
                    break
    for m in _GLOBAL_RE.finditer(source):
        names.append(m.group(1))
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        # Reject reserved-prefix tokens: no kernel is named `__...`, so such a capture is a compiler attribute that
        # leaked through.
        if n and not n.startswith("__") and n not in seen:
            seen.add(n)
            out.append(n)
    return out
