# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a kernel's source is looked for, and the keywords a search for it is built from."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from functools import lru_cache

from hyperloom.inference_optimizer.framework_paths import (
    FRAMEWORK_SOURCE_PACKAGES as _FRAMEWORK_SOURCE_PACKAGES,
    resolve_kernel_search_roots as _resolve_kernel_search_roots,
)

from ._task_group_contract import _strip_dispatch_decoration, _strip_template_arguments

log = logging.getLogger(__name__)


#: Last-resort checkout layouts for a host where nothing above is importable.
#: Kept small on purpose: a pinned path cannot follow a package across
#: container images or Python versions, and a list of them going stale in
#: silence is what emptied this tier and stalled kernel-opt entirely.
_FALLBACK_SEARCH_ROOTS: tuple[str, ...] = (
    "/sgl-workspace/aiter",
    "/sgl-workspace/sglang/sgl-kernel",
    "/sgl-workspace/sglang/python/sglang",
    "/sgl-workspace/vllm",
)


def _installed_package_dir(package: str) -> str:
    """Locate a package's directory without importing it.

    Args:
        package (str): Importable package name.

    Returns:
        str: The package directory, or ``""`` when it is not on this
            interpreter's path.
    """
    if not package or not package.isidentifier():
        return ""
    try:
        spec = importlib.util.find_spec(package)
    except (AttributeError, ImportError, ValueError):
        return ""
    if spec is None:
        return ""
    for location in list(getattr(spec, "submodule_search_locations", None) or []):
        candidate = str(location).rstrip("/")
        if candidate:
            return candidate
    origin = str(getattr(spec, "origin", "") or "")
    return os.path.dirname(origin) if origin else ""


@lru_cache(maxsize=1)
def kernel_search_roots() -> tuple[str, ...]:
    """Resolve the framework trees to grep for kernel source, at runtime.

    Prefers the orchestrator's centralised resolver so this tool agrees with
    PolicyGate and patch application on where framework source lives. Falls back
    to locating each known package itself, then to the pinned checkout layouts,
    when that resolver resolves nothing -- an empty answer from it used to end
    the search, which is the same silent outcome as having no roots at all.

    Non-existent roots are dropped: grepping them returns nothing and is
    indistinguishable from a kernel that genuinely has no source here.

    Cached for the duration of a run, since every candidate consults it and
    ``find_spec`` plus a directory probe per package is not free. A process that
    outlives one run -- the orchestrator imports this module -- calls
    :func:`refresh_kernel_search_roots` at run entry so a framework installed
    since the last one is still discoverable.

    Returns:
        tuple[str, ...]: Existing roots without trailing separators,
            de-duplicated in discovery order.
    """
    discovered: list[str] = list(_resolve_kernel_search_roots())
    # A chain, not an either/or. The centralised resolver is authoritative and
    # goes first, but it can return nothing: when it does, the alternative to
    # probing here is a run that greps no directory at all and reports every hot
    # kernel as non-routable. Its answer is kept whole when it has one, so this
    # cannot widen the roots a normal host searches -- it only decides between
    # local discovery and nothing.
    if not discovered:
        discovered.extend(
            location
            for location in (_installed_package_dir(package) for package in _FRAMEWORK_SOURCE_PACKAGES)
            if location
        )
        discovered.extend(_FALLBACK_SEARCH_ROOTS)
    roots: list[str] = []
    seen: set[str] = set()
    for root in discovered:
        normalized = str(root or "").rstrip("/")
        if not normalized or normalized in seen or not os.path.isdir(normalized):
            continue
        seen.add(normalized)
        roots.append(normalized)
    if not roots:
        log.warning(
            "no framework source root exists on this host (looked for %s); "
            "kernel source resolution will find nothing and every hot kernel "
            "will be reported as non-routable",
            ", ".join(_FRAMEWORK_SOURCE_PACKAGES),
        )
    return tuple(roots)


@lru_cache(maxsize=1)
def _harness_search_bases() -> tuple[str, ...]:
    """Directories a checkout-relative harness path may be joined onto.

    A hint reads ``aiter/op_tests/...``, so the base is whatever holds the
    ``aiter`` checkout. Each resolved search root contributes both itself and
    its parent, because a root is the package directory on a wheel install
    (``.../dist-packages/aiter``) and the checkout itself in a serving image
    (``/sgl-workspace/aiter``); one join is the right one and the other simply
    does not exist.

    Returns:
        tuple[str, ...]: Existing base directories, de-duplicated.
    """
    bases: list[str] = []
    seen: set[str] = set()
    for root in kernel_search_roots():
        trimmed = root.rstrip("/")
        for base in (os.path.dirname(trimmed), trimmed):
            if base and base not in seen and os.path.isdir(base):
                seen.add(base)
                bases.append(base)
    return tuple(bases)


def refresh_kernel_search_roots() -> tuple[str, ...]:
    """Re-run root discovery, dropping every answer derived from the old roots.

    :func:`_harness_search_bases` is cached for the same run and derived from
    these roots, so clearing one without the other reproduces the bug this
    function exists to fix, one step further on: the second analysis in a
    long-lived orchestrator would grep the freshly installed framework and still
    resolve its harnesses against the bases discovered when it was absent, so
    ``benchmark_files`` comes back empty and the invocation spec reaches the
    backend with no benchmark to run.

    Returns:
        tuple[str, ...]: The freshly discovered roots.
    """
    kernel_search_roots.cache_clear()
    _harness_search_bases.cache_clear()
    return kernel_search_roots()


_NAMESPACE_BLOCKLIST = {
    "aiter",
    "sglang",
    "vllm",
    "torch",
    "ck_tile",
    "ck",
    "pybind",
    "RankData",
    "RankSignals",
    "Signal",
    "module",
    "namespace",
}

_TYPE_BLOCKLIST = {
    "void",
    "int",
    "float",
    "char",
    "long",
    "short",
    "bool",
    "unsigned",
    "string",
}


def _normalize_profiler_op_name(name: str) -> str:
    """Strip graph-capture / synthetic wrappers; return the original if stripping empties it."""
    original = (name or "").strip()
    if not original:
        return ""
    return _strip_dispatch_decoration(original) or original


def _candidate_keywords(name: str) -> list[str]:
    """Pick stable search keywords from a kernel symbol.

    Prefers descriptive identifiers (e.g. cross_device_reduce_2stage, gemm_a16w16)
    over namespace/type tokens (aiter, vllm, RankData) that match too widely.

    Args:
        name (str): Kernel symbol/name (possibly Itanium-mangled).

    Returns:
        list[str]: Up to three descriptive search keywords, most-specific
            first; empty when nothing usable can be extracted.
    """
    cleaned = _normalize_profiler_op_name(name)
    if cleaned.startswith("_Z"):
        # Itanium ABI uses <len><name>; slice manually so consecutive segments
        # are parsed as separate identifiers.
        tokens = []
        pos = 0
        while pos < len(cleaned):
            m = re.match(r"(\d+)", cleaned[pos:])
            if not m:
                pos += 1
                continue
            length = int(m.group(1))
            start = pos + m.end()
            ident = cleaned[start : start + length]
            if ident and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", ident):
                tokens.append(ident)
                pos = start + length
            else:
                pos = start + 1
        if not tokens:
            tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{4,}", cleaned)
    else:
        cleaned = _strip_template_arguments(cleaned)
        if "::" in cleaned:
            cleaned = cleaned.split("::")[-1]
        tokens = [cleaned]
    seen: set[str] = set()
    raw: list[str] = []
    for tok in tokens:
        tok = tok.rstrip("_")
        if not tok or tok in seen:
            continue
        if tok in _TYPE_BLOCKLIST:
            continue
        if len(tok) < 5:
            continue
        seen.add(tok)
        raw.append(tok)
    if not raw:
        return []
    # Prefer multi-segment identifiers and drop namespace tokens that match too widely.
    descriptive = [t for t in raw if t not in _NAMESPACE_BLOCKLIST]
    if descriptive:
        descriptive.sort(key=lambda t: (-t.count("_"), -len(t)))
        return descriptive[:3]
    raw.sort(key=lambda t: (-t.count("_"), -len(t)))
    return raw[:3]


# Bare HIP/CUDA launch APIs. TraceLens emits these as standalone rows when it
# aggregates launches it could not attribute to a device kernel, but they are
# runtime entry points, not kernels. Grepping them matches wherever the API name
# merely appears -- e.g. aiter's hipify name-mapping table, which then gets
# routed to a backend as if it were rewritable kernel source.
_RUNTIME_API_NAMES: frozenset[str] = frozenset(
    {
        "cudagraphlaunch",
        "cudalaunchcooperativekernel",
        "cudalaunchkernel",
        "cudamodulelaunchkernel",
        "hipextlaunchkernel",
        "hipgraphlaunch",
        "hiplaunchcooperativekernel",
        "hiplaunchkernel",
        "hipmodulelaunchkernel",
    }
)


def is_runtime_api_name(name: str) -> bool:
    """Return whether ``name`` is a bare HIP/CUDA launch API rather than a kernel.

    Args:
        name (str): Kernel symbol/name, possibly profiler-wrapped.

    Returns:
        bool: ``True`` when the normalised name is a bare launch API.
    """
    return _normalize_profiler_op_name(name).strip().lower() in _RUNTIME_API_NAMES
