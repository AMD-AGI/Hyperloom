###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Source resolution shared by the compute and bypass analysis paths.

TraceLens owns path-finding: :func:`resolve_source_verdict` is the one call both
routes make, wrapping TraceLens' ``resolve_kernel_source`` and returning its
``ResolveResult`` straight through. Only the
Triton-vs-native routing decision is HL's, and it lives here so both callers
route identically. :func:`triton_def_line` is the AST reading of what a Triton
kernel definition looks like; ``source_type_for`` needs it to tell a Triton
``.py`` from any other Python file, and it is used on both routes.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from TraceLens.TraceUtils.kernel_source import ResolveResult, resolve_kernel_source

__all__ = ["ResolveResult", "resolve_source_verdict", "triton_def_line"]

#: Triton membership is keyed on the symbol, library, or launcher path, not the
#: launcher extension: every aiter native kernel dispatches through a ``.py`` too,
#: and forcing the Triton route on one bypasses the native Tensile/CK/MIOpen gate
#: and mislabels a precompiled kernel as patchable. aiter Gluon kernels carry the
#: marker only in their ``.../triton/...`` launcher path.
_TRITON_MARKER = "triton"


def resolve_source_verdict(
    symbol: str,
    *,
    kernel_file: str = "",
    op_name: str = "",
    library: str = "",
) -> ResolveResult:
    """Resolve one device symbol to its source via TraceLens.

    A genuine Triton kernel takes the Triton route with its launcher; a native
    kernel passes just the symbol so TraceLens' native gate can classify
    Tensile/CK/MIOpen instead of the Triton path accepting the ``.py`` dispatcher.
    """
    is_triton = (
        _TRITON_MARKER in symbol.lower() or _TRITON_MARKER in library.lower() or _TRITON_MARKER in kernel_file.lower()
    )
    return resolve_kernel_source(
        symbol,
        kernel_file=kernel_file if is_triton else "",
        is_triton=is_triton,
        op_name=op_name,
    )


# Triton decorators marking a device-kernel def (``@triton.jit`` / ``@jit`` and the
# autotune/heuristics wrappers that sit on top of a jit'd kernel).
_TRITON_DECORATORS = frozenset({"jit", "autotune", "heuristics"})


def _is_triton_kernel_def(node: ast.AST) -> bool:
    """Return whether an AST function node carries a Triton kernel decorator."""
    for dec in getattr(node, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = getattr(target, "attr", None) or getattr(target, "id", None)
        if name in _TRITON_DECORATORS:
            return True
    return False


def _normalize_symbol(symbol: str) -> str:
    """Reduce a device kernel symbol to a bare identifier core for matching."""
    core = re.sub(r"[^0-9A-Za-z_].*$", "", str(symbol or "").strip())
    core = re.sub(r"_+\d[\dA-Za-z]*$", "", core)  # drop trailing autotune/hash suffix
    return core.strip("_").lower()


def triton_def_line(py_path: str, *, func: str = "", symbol: str = "", require_name_match: bool = False) -> int | None:
    """Find a Triton kernel's ``def`` line in a ``.py`` via AST (no import)."""
    try:
        tree = ast.parse(Path(py_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None

    jit_defs: dict[str, int] = {}
    all_defs: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_defs.setdefault(node.name, node.lineno)
            if _is_triton_kernel_def(node):
                jit_defs.setdefault(node.name, node.lineno)

    if func and func in all_defs:
        return all_defs[func]

    core = _normalize_symbol(symbol)
    if core:
        for name, line in jit_defs.items():
            if name.lower() == core:
                return line
        for name, line in jit_defs.items():
            low = name.lower()
            if core in low or low in core:
                return line

    if not require_name_match and len(jit_defs) == 1:
        return next(iter(jit_defs.values()))
    return None
