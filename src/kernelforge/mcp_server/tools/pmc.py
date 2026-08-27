# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-name extraction from GPU kernel source."""

from __future__ import annotations

import re

# Decorator markers whose next `def <name>(` is a GPU kernel entry (FlyDSL/Triton).
_KERNEL_DECO_RE = re.compile(r"\.kernel\b|triton\.jit\b|\.jit\b", re.IGNORECASE)
_DEF_RE = re.compile(r"^\s*def\s+(\w+)\s*\(")
# HIP/CUDA __global__ entry points. Attribute clauses may appear BEFORE or AFTER
# `void` — both orders are legal and both occur in real code
# (`__global__ void __launch_bounds__(512, 1) name(`) — and the name may sit on
# the next line. Matching only one order silently captures the attribute keyword
# as the kernel name, which then matches no dispatch at all.
# `static` / `inline` need word boundaries: without them the alternation also
# matches the prefix of an identifier, so `__global__ void inline_helper_kernel(`
# is read as the attribute `inline` followed by the name `_helper_kernel`, and
# that truncated name matches no dispatch.
_GLOBAL_ATTR = r"(?:__launch_bounds__\s*\([^)]*\)|__attribute__\s*\(\([^)]*\)\)|\bstatic\b|\binline\b)"
_GLOBAL_RE = re.compile(
    rf"__global__\s+(?:{_GLOBAL_ATTR}\s+)*void\s+(?:{_GLOBAL_ATTR}\s*)*(\w+)\s*\(",
)


def derive_kernel_names(source: str) -> list[str]:
    """Best-effort list of GPU-kernel entry names declared in kernel source.

    Recognizes FlyDSL/Triton (`@…​.kernel` / `@triton.jit` decorator then
    ``def name(``) and HIP/CUDA (``__global__ void name(``). The compiled
    dispatch name is derived from these (e.g. FlyDSL ``softmax_kernel`` ->
    dispatch ``softmax_kernel_0``), so matching dispatches by these substrings
    isolates the target kernel from reference/library dispatches. Order-preserving,
    de-duplicated. Returns [] when nothing matches (caller falls back to the
    framework-exclusion heuristic).
    """
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
        # Reject reserved-prefix tokens: no kernel is named `__...`, so such a
        # capture is a compiler attribute that leaked through. Keeping it would
        # put a hint in the allow-list that matches nothing, which reads as "the
        # target was not found" rather than as a parse failure.
        if n and not n.startswith("__") and n not in seen:
            seen.add(n)
            out.append(n)
    return out
