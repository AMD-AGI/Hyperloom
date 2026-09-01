###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Op -> editable-source resolver for the bypass analysis backend.

Used by the bypass route (``HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass``) to populate
``source_file`` on hot-kernel candidates so the downstream kernel optimizer can
dispatch a rewrite (it filters out candidates with no ``source_file``).

Resolution runs entirely against the *currently installed* framework trees --
there is no static op_to_source map. Three complementary mechanisms are exposed
(the bypass report tries them in this order):

* :func:`resolve_triton_py` -- Triton ``.py`` kernels: resolved from the
  trace-provided ``kernel_file``, pinning the exact ``@triton.jit`` def line via
  AST (no import of the kernel required).
* :func:`resolve_source` -- native (``.cu``/``.hip``) kernels: delegates to the
  active finder (:mod:`source_resolver`), which demangles the device kernel
  symbol and looks it up in a live ``__global__`` index (method
  ``"symbol_index"``).
* :func:`resolve_by_kernel_name` -- repo-scan fallback by demangled kernel name.

This module also hosts the shared editability helpers
(:func:`is_editable_source`, :func:`editable_trace_source`) reused by the finder.
It never imports TraceLens.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import logging
import os
import re
import time
from pathlib import Path

from hyperloom.common.env import is_truthy

log = logging.getLogger(__name__)

# Editable source extensions: native device code plus repo-resident Triton .py.
_NATIVE_SOURCE_EXTS = (".cu", ".cuh", ".hip", ".h")


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


def _exists(path: str) -> bool:
    """``os.path.exists`` guarded against odd paths (never raises)."""
    try:
        return bool(path) and os.path.exists(path)
    except OSError:
        return False


def resolve_source(
    op_name: str,
    *,
    framework: str = "",
    device_kernel_name: str = "",
) -> tuple[str, str]:
    """Resolve a native kernel to its live installed source via the active finder.

    This is the deterministic op->source tier for the bypass route: it delegates
    to :func:`source_resolver.resolve_source`, which demangles the device kernel
    symbol and looks it up in a live ``__global__`` index (method
    ``"symbol_index"``). There is no static op_to_source map; any import/lookup
    failure yields ``("", "unresolved")`` so the caller can fall back to the
    repo-scan tier.

    Args:
        op_name: The launching op name (carried for reporting, not lookup).
        framework: Serving framework hint used to rank multi-tree matches.
        device_kernel_name: Device kernel symbol from the trace (authoritative).

    Returns:
        ``(source_file, "symbol_index")`` on a hit, else ``("", "unresolved")``
        / ``("", "non_patchable")``.
    """
    if not device_kernel_name:
        return "", "unresolved"
    try:
        try:  # package import (TraceLens route / tests)
            from . import source_resolver
        except ImportError:  # flat top-level import (bypass route puts tools/ on sys.path)
            import source_resolver  # type: ignore[no-redef]

        return source_resolver.resolve_source(op_name, framework=framework, device_kernel_name=device_kernel_name)
    except (ImportError, OSError, ValueError) as exc:
        log.debug("bypass resolve_source failed for %r: %s", device_kernel_name, exc)
        return "", "unresolved"


# Triton kernel definition: @triton.jit then optional decorators then def NAME.
_TRITON_DEF_RE = re.compile(r"@triton\.jit[^\n]*\n(?:\s*@[^\n]*\n)*\s*def\s+(\w+)")
# Native kernel definition: __global__ with optional qualifiers then NAME.
_GLOBAL_DEF_RE = re.compile(
    r"__global__\s*(?:(?:void|static|inline|__forceinline__|"
    r"__launch_bounds__\s*\([^)]*\))\s*)*(\w+)\s*[\(<]"
)
# Directories/paths to skip while scanning source repos.
_SCAN_SKIP_MARKERS = ("/__pycache__", "/3rdparty/", "/example", "/test", "/jit/build/", "/.git/")
_TRITON_SCAN_EXTS = (".py",)
_NATIVE_SCAN_EXTS = (".cu", ".cuh", ".hip", ".h")


def _demangle_kernel_name(name: str) -> str | None:
    """Extract the bare function identifier from a device kernel name.

    Handles Itanium mangling (``_ZN<len><ns>...`` / ``_Z<len><name>...``) and
    plain C++/Triton names (strips ``void``, namespaces, and template/arg tails).
    """
    n = (name or "").strip()
    if not n:
        return None
    if n.startswith("_ZN") or n.startswith("_Z"):
        body = n[3:] if n.startswith("_ZN") else n[2:]
        tokens: list[str] = []
        i = 0
        while i < len(body) and body[i].isdigit():
            j = i
            while j < len(body) and body[j].isdigit():
                j += 1
            length = int(body[i:j])
            token = body[j : j + length]
            if not token:
                break
            tokens.append(token)
            i = j + length
        if tokens:
            return tokens[-1] if n.startswith("_ZN") else tokens[0]
        return None
    if n.startswith("void "):
        n = n[len("void ") :].strip()
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"<.*$", "", n)
    n = re.sub(r"\(.*$", "", n)
    n = n.strip()
    if "::" in n:
        n = n.rsplit("::", 1)[-1]
    return n or None


@functools.lru_cache(maxsize=1)
def _repo_scan_roots() -> tuple[str, ...]:
    """Discover live sglang/aiter source roots (and aiter csrc) without importing."""
    roots: list[str] = []
    seen: set[str] = set()
    for pkg in ("sglang", "aiter"):
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError, ModuleNotFoundError):
            continue
        if spec is None:
            continue
        for loc in list(getattr(spec, "submodule_search_locations", None) or []):
            for cand in (loc, os.path.join(os.path.dirname(loc), "csrc")):
                if cand not in seen and os.path.isdir(cand):
                    seen.add(cand)
                    roots.append(cand)
    return tuple(roots)


@functools.lru_cache(maxsize=1)
def _build_repo_kernel_index() -> dict[str, str]:
    """Map kernel function name -> source path by scanning repo roots once.

    A name that resolves to more than one distinct source path is ambiguous and
    mapped to ``""`` so :func:`resolve_by_kernel_name` refuses it rather than
    routing a rewrite at an arbitrary first-seen file.
    """
    index: dict[str, str] = {}
    roots = _repo_scan_roots()
    if not roots:
        return index
    t0 = time.monotonic()
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            if any(m in dirpath for m in _SCAN_SKIP_MARKERS):
                continue
            for fname in files:
                path = os.path.join(dirpath, fname)
                if any(m in path for m in _SCAN_SKIP_MARKERS):
                    continue
                low = fname.lower()
                if low.endswith(_TRITON_SCAN_EXTS):
                    pattern, marker = _TRITON_DEF_RE, "@triton.jit"
                elif low.endswith(_NATIVE_SCAN_EXTS):
                    pattern, marker = _GLOBAL_DEF_RE, "__global__"
                else:
                    continue
                try:
                    with open(path, encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                except OSError:
                    continue
                if marker not in text:
                    continue
                for match in pattern.finditer(text):
                    name = match.group(1)
                    prev = index.get(name)
                    if prev is None:
                        index[name] = path
                    elif prev and prev != path:
                        # Same kernel name in two files: ambiguous, do not guess.
                        log.info("repo scan: kernel name %r is ambiguous (%s vs %s)", name, prev, path)
                        index[name] = ""
    elapsed = time.monotonic() - t0
    log.info(
        "repo scan: indexed %d kernel name(s) from %d root(s) in %.2fs",
        len(index),
        len(roots),
        elapsed,
    )
    return index


def resolve_by_kernel_name(device_kernel_name: str) -> tuple[str, str]:
    """Resolve a device kernel name to an editable source via repo scan.

    Demangles the kernel name and looks it up in the repo kernel index, returning
    ``(path, "repo_scan")`` on an unambiguous editable on-disk hit, else
    ``("", "unresolved")`` (an empty index entry marks an ambiguous name).
    """
    if is_truthy(os.environ.get("HYPERLOOM_BYPASS_DISABLE_REPO_SCAN")):
        return "", "unresolved"
    bare = _demangle_kernel_name(device_kernel_name)
    if not bare:
        return "", "unresolved"
    path = _build_repo_kernel_index().get(bare)
    if path and is_editable_source(path) and _exists(path):
        return path, "repo_scan"
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


# ---------------------------------------------------------------------------
# Triton .py AST pinning: resolve the exact @triton.jit def line from the trace's
# kernel_file. Native .cu/.hip kernels are resolved by :func:`resolve_source`
# (the active finder); Triton .py kernels come from the trace directly, so the
# report tries this first, then the finder, then the repo scan.
# ---------------------------------------------------------------------------

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


def _parse_launcher_form(raw: str) -> tuple[str, int | None, str]:
    """Split a trace ``kernel_file`` into ``(py_path, line, func)``.

    Handles the plain-path case (no line/func) and the launcher forms
    ``a.py(12): foo`` / ``a.py:12:foo`` / ``a.py#L12``. Non-``.py`` inputs are
    returned unchanged with no line/func.
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
    (e.g. ``_fwd_kernel_0d1d2``). Strip the common decorations so a fuzzy match
    against the def name has a chance.
    """
    core = re.sub(r"[^0-9A-Za-z_].*$", "", str(symbol or "").strip())
    core = re.sub(r"_+\d[\dA-Za-z]*$", "", core)  # drop trailing autotune/hash suffix
    return core.strip("_").lower()


def triton_def_line(py_path: str, *, func: str = "", symbol: str = "", require_name_match: bool = False) -> int | None:
    """Find a Triton kernel's ``def`` line in a ``.py`` via AST (no import).

    Matching precedence: (1) exact ``func`` name; (2) a ``@triton.jit`` def whose
    name matches the normalized device ``symbol`` (exact then substring); (3) the
    sole ``@triton.jit`` def when unambiguous and ``require_name_match`` is ``False``.
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
    def line cannot be pinned. ``method`` is ``"trace_kernel_file_ast"`` (path +
    pinned line), ``"trace_kernel_file"`` (path only), or ``"unresolved"``.
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
    "resolve_by_kernel_name",
    "editable_trace_source",
    "resolve_triton_py",
    "triton_def_line",
    "is_editable_source",
]
