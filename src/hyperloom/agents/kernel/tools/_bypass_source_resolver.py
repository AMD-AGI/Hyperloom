###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Op -> editable-source entry point for the bypass analysis backend.

Used by the bypass route (``HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass``) to populate
``source_file`` on hot-kernel candidates so the downstream kernel optimizer can
dispatch a rewrite (it filters out candidates with no ``source_file``).

Resolution runs entirely against the *currently installed* framework trees --
there is no static source mapping. Two complementary mechanisms are exposed:

* :func:`resolve_source` -- native (``.cu``/``.hip``) kernels: delegates to the
  active finder (:mod:`source_resolver`), which demangles the trace's device
  kernel symbol and looks it up in a live ``__global__`` index (method
  ``"symbol_index"``).
* :func:`resolve_triton_py` / :func:`triton_def_line` -- Triton ``.py`` kernels:
  resolved from the trace-provided ``kernel_file``, parsing launcher-form paths
  and pinning the exact ``@triton.jit`` def line via AST (no import of the kernel
  required).

This module also hosts the shared editability helpers
(:func:`is_editable_source`, :func:`editable_trace_source`) reused by the finder.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

# Editable source extensions: native device code plus repo-resident Triton .py.
_NATIVE_SOURCE_EXTS = (".cu", ".cuh", ".hip", ".h")

# Triton decorators marking a device-kernel def (``@triton.jit`` / ``@jit`` and
# the autotune/heuristics wrappers that sit on top of a jit'd kernel).
_TRITON_DECORATORS = frozenset({"jit", "autotune", "heuristics"})

# Launcher-path forms a trace ``kernel_file`` may carry instead of a bare path:
# ``<path>(<line>): <func>``, ``<path>:<line>:<func>``, or ``<path>#L<line>``.
_LAUNCHER_PATH_RE = re.compile(
    r"^(?P<path>.+?\.py)"
    r"(?:\((?P<pline>\d+)\)|[:#]L?(?P<cline>\d+))"
    r"(?::?\s*(?P<func>[A-Za-z_]\w*))?\s*$"
)


def is_editable_source(path: str | None, kernel_kind: str | None = None) -> bool:
    """Return whether ``path`` is a source we can route a kernel rewrite at.

    Editable == native device code (``.cu``/``.cuh``/``.hip``/``.h``) or a
    repo-resident Triton/TileLang ``.py``. Generated Triton is excluded
    (``triton_inductor_generated`` kind and any ``torchinductor`` / ``/tmp/``
    path).

    Args:
        path: Candidate source path (from a trace ``kernel_file`` or the finder).
        kernel_kind: Optional kernel-kind hint.

    Returns:
        ``True`` when the path is an editable source, else ``False``.
    """
    if not path:
        return False
    low = path.lower()
    if low.endswith(_NATIVE_SOURCE_EXTS):
        return True
    if low.endswith(".py"):
        if kernel_kind == "triton_inductor_generated":
            return False
        if "torchinductor" in path or path.startswith("/tmp/"):  # nosec B108 - marker for generated compiler artifacts.
            return False
        return True
    return False


def resolve_source(
    op_name: str,
    *,
    framework: str = "",
    device_kernel_name: str = "",
) -> tuple[str, str]:
    """Resolve a CPU op / device kernel to an editable source via the active finder.

    Delegates entirely to :func:`source_resolver.resolve_source`, which finds
    the kernel's source in the currently installed framework tree from the
    device kernel symbol. There is no static-mapping fallback: on any finder
    failure the op is simply reported unresolved.

    The finder is imported lazily so this module (whose
    :func:`is_editable_source` the finder reuses) has no import cycle.

    Args:
        op_name: The launching op name (e.g. ``_C::silu_and_mul``); carried for
            reporting, not used for lookup.
        framework: Serving framework hint used to rank candidate records.
        device_kernel_name: Device kernel symbol from the trace (authoritative).

    Returns:
        ``(source_file, method)`` on a hit (``method == "symbol_index"``), or
        ``("", "unresolved")`` / ``("", "non_patchable")`` otherwise.
    """
    try:
        try:  # package import (TraceLens route / tests)
            from . import source_resolver
        except ImportError:  # flat top-level import (bypass route)
            import source_resolver  # type: ignore[no-redef]

        return source_resolver.resolve_source(
            op_name, framework=framework, device_kernel_name=device_kernel_name
        )
    except (ImportError, OSError, ValueError):
        return "", "unresolved"


def editable_trace_source(kernel_file: str, kernel_kind: str = "") -> str:
    """Return a trace-provided Triton ``kernel_file`` iff it is an editable source.

    Kineto ``cpu_op`` args carry ``kernel_file`` for Triton kernels. A
    repo-resident ``.py`` is directly editable; inductor-generated / ``/tmp``
    Triton is not (filtered out here), so it returns ``""`` for those.

    Args:
        kernel_file: The ``kernel_file`` arg from a cpu_op event.
        kernel_kind: Optional kind hint.

    Returns:
        The editable source path, or ``""`` when unusable.
    """
    kf = str(kernel_file or "").strip()
    if not kf:
        return ""
    return kf if is_editable_source(kf, kernel_kind or None) else ""


def _parse_launcher_form(raw: str) -> tuple[str, int | None, str]:
    """Split a trace ``kernel_file`` into ``(py_path, line, func)``.

    Handles the plain-path case (no line/func) and the launcher forms
    ``a.py(12): foo`` / ``a.py:12:foo`` / ``a.py#L12``. Non-``.py`` inputs are
    returned unchanged with no line/func.

    Args:
        raw: The raw ``kernel_file`` string from the trace.

    Returns:
        ``(path, line_or_None, func_or_empty)``.
    """
    text = str(raw or "").strip()
    if not text:
        return "", None, ""
    match = _LAUNCHER_PATH_RE.match(text)
    if match:
        line_str = match.group("pline") or match.group("cline")
        return (
            match.group("path").strip(),
            int(line_str) if line_str else None,
            match.group("func") or "",
        )
    return text, None, ""


def _is_triton_kernel_def(node: ast.AST) -> bool:
    """Return whether an AST function node carries a Triton kernel decorator."""
    for dec in getattr(node, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = getattr(target, "attr", None) or getattr(target, "id", None)
        if name in _TRITON_DECORATORS:
            return True
    return False


def _normalize_symbol(symbol: str) -> str:
    """Reduce a device kernel symbol to a bare identifier core for matching.

    Triton device symbols often wrap the ``@triton.jit`` function name with a
    leading ``triton_``/``_`` prefix and a trailing autotune/hash suffix
    (e.g. ``_fwd_kernel_0d1d2``). Strip the common decorations so a fuzzy
    match against the def name has a chance.

    Args:
        symbol: The raw device kernel symbol from the trace.

    Returns:
        A lowercased identifier core (may be empty).
    """
    core = re.sub(r"[^0-9A-Za-z_].*$", "", str(symbol or "").strip())
    core = re.sub(r"_+\d[\dA-Za-z]*$", "", core)  # drop trailing autotune/hash suffix
    return core.strip("_").lower()


def triton_def_line(py_path: str, *, func: str = "", symbol: str = "") -> int | None:
    """Find a Triton kernel's ``def`` line in a ``.py`` via AST (no import).

    Matching precedence: (1) exact ``func`` name; (2) a ``@triton.jit`` def whose
    name matches the normalized device ``symbol`` (exact then substring); (3) the
    sole ``@triton.jit`` def in the file when unambiguous.

    Args:
        py_path: Absolute path to the ``.py`` source.
        func: Optional exact function name (e.g. from a launcher form).
        symbol: Optional device kernel symbol used for fuzzy matching.

    Returns:
        The 1-based ``def`` line, or ``None`` when the file is unreadable,
        unparseable, or no confident match is found.
    """
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

    if func:
        if func in all_defs:
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

    if len(jit_defs) == 1:
        return next(iter(jit_defs.values()))
    return None


def resolve_triton_py(
    kernel_file: str,
    kernel_kind: str = "",
    *,
    symbol: str = "",
) -> tuple[str, int | None, str]:
    """Resolve a trace ``kernel_file`` to an editable Triton ``.py`` plus def line.

    Extends :func:`editable_trace_source` with two AST-backed behaviours:
    launcher-form paths (``a.py:12:foo``) are parsed down to the bare ``.py``,
    and the exact ``@triton.jit`` def line is pinned via :func:`triton_def_line`.
    The AST step is a pure refinement: a resolved file is returned even when the
    def line cannot be pinned.

    Args:
        kernel_file: The ``kernel_file`` arg from a cpu_op event (bare path or
            launcher form).
        kernel_kind: Optional kind hint forwarded to :func:`is_editable_source`.
        symbol: Device kernel symbol used to disambiguate the def line.

    Returns:
        ``(source_file, line_or_None, method)`` where ``method`` is
        ``"trace_kernel_file_ast"`` (path + pinned line), ``"trace_kernel_file"``
        (path only), or ``"unresolved"`` (``source_file == ""``).
    """
    path, line, func = _parse_launcher_form(kernel_file)
    if not path:
        return "", None, "unresolved"
    source = editable_trace_source(path, kernel_kind)
    if not source:
        return "", None, "unresolved"
    ast_line: int | None = None
    if source.lower().endswith(".py") and os.path.isfile(source):
        ast_line = triton_def_line(source, func=func, symbol=symbol)
    def_line = ast_line if ast_line is not None else line
    method = "trace_kernel_file_ast" if ast_line is not None else "trace_kernel_file"
    return source, def_line, method


__all__ = [
    "resolve_source",
    "editable_trace_source",
    "resolve_triton_py",
    "triton_def_line",
    "is_editable_source",
]
