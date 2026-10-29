# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a traced kernel is: its shape, the source file behind it, and whether that source can be patched."""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

from hyperloom.common.outside_facts import AITER_META_DIR_ENV
from hyperloom.measure.jit.modules import image_aiter_meta
from hyperloom.inference_optimizer.framework_paths import (
    resolve_flydsl_source_roots as _resolve_flydsl_source_roots,
    resolve_known_source_prefixes as _resolve_known_source_prefixes,
)

from ._kernel_search import _candidate_keywords, is_runtime_api_name

# Vendor-operator-playbook registry: routes a closed-source vendor op (no
# rewritable device source) to a validated KernelForge task bundle instead of
# a source rewrite -- see KernelForge PR #88's mori dispatch/combine gap.
from ._vendor_operator_playbooks import match_vendor_operator_playbook

# source_type_for asks TraceLens whether a ``.py`` defines the traced Triton kernel.
try:
    from TraceLens.TraceUtils.kernel_source.triton_pin import triton_def_line as _triton_def_line
except ImportError:
    _triton_def_line = None  # type: ignore[assignment]


_log = logging.getLogger(__name__)
#: The directory of aiter's meta directory that holds its device sources.
_AITER_CSRC = "csrc"

RUNTIME_API_NAMES = {
    "hipeventsynchronize",
    "hipdevicesynchronize",
    "hipstreamsynchronize",
    "hipgraphlaunch",
    "hiplaunchkernel",
    "hipmodulelaunchkernel",
    "hipmemcpy",
    "hipmemset",
    "cudaeventsynchronize",
    "cudadevicesynchronize",
    "cudastreamsynchronize",
}


def is_kernel_event(event: dict[str, Any]) -> bool:
    """Apply a strict GPU-kernel filter to a trace event.

    Only ``cat == 'kernel'`` events qualify, excluding host-side sync/launch
    wrappers that would eclipse real kernels.

    Args:
        event: A single trace event dict.

    Returns:
        ``True`` if the event is a real GPU kernel.
    """
    cat = str(event.get("cat") or event.get("category") or "").lower()
    if cat != "kernel":
        return False
    name = str(event.get("name") or event.get("kernel_name") or "")
    if name.lower() in RUNTIME_API_NAMES:
        return False
    return True


def extract_shape(event: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a shape annotation from a trace event when present.

    Checks both ``event['args']`` and the event top level for the first of
    several known shape keys.

    Args:
        event (dict[str, Any]): A single trace event.

    Returns:
        dict[str, Any] | None: A single-key dict ``{shape_key: value}`` for
            the first shape field found, or ``None`` when none is present.
    """
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    for key in ("shape", "shapes", "input_shape", "trace_shapes"):
        if key in args:
            return {key: args[key]}
        if key in event:
            return {key: event[key]}
    return None


def extract_source_file(event: dict[str, Any]) -> str:
    """Extract a source-file path from a trace event when present.

    Checks both ``event['args']`` and the event top level for the first of
    several known path keys.

    Args:
        event (dict[str, Any]): A single trace event.

    Returns:
        str: The first non-empty source path found, or ``""`` when none.
    """
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    for key in ("source_file", "file", "filename", "path"):
        value = args.get(key) or event.get(key)
        if value:
            return str(value)
    return ""


_FLYDSL_SOURCE_MARKERS = (
    "import flydsl",
    "from flydsl",
    "@flyc.kernel",
    "@flyc.jit",
    "flydsl.compiler",
    "flydsl.expr",
)

_FLYDSL_SCAN_BYTES = 4096


def _looks_like_flydsl_source(source_file: str) -> bool:
    """Detect whether a source file is a FlyDSL kernel source.

    Content-sniffs the first 4 KiB for FlyDSL markers.

    Args:
        source_file: Path to the candidate ``.py`` source.

    Returns:
        ``True`` when FlyDSL markers are found in the file head.
    """
    if not source_file or not source_file.endswith(".py"):
        return False
    try:
        with open(source_file, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(_FLYDSL_SCAN_BYTES)
    except OSError:
        return False
    return any(marker in head for marker in _FLYDSL_SOURCE_MARKERS)


def _defines_traced_triton_kernel(name: str, source_file: str) -> bool:
    """Whether ``source_file`` defines the Triton kernel ``name`` was traced from.

    A device symbol carries no language and frameworks import Triton through a
    shim, so both signals the name-based check reads are absent. Resolve the
    symbol to a ``@triton.jit`` def instead.

    Args:
        name: Kernel/symbol name as the trace reports it.
        source_file: Resolved source path.

    Returns:
        ``True`` when a ``@triton.jit`` def in ``source_file`` matches ``name``.
    """
    if _triton_def_line is None or not source_file or not source_file.endswith(".py"):
        return False
    for keyword in _candidate_keywords(name):
        if _triton_def_line(source_file, symbol=keyword, require_name_match=True) is not None:
            return True
    return False


_FLYDSL_PSEUDO_OP_NAME_MARKERS = (
    "pseudo_op::moe_flydsl_",
    "pseudo_op::flydsl_",
)


def source_type_for(name: str, source_file: str) -> str:
    """Classify a kernel's source type from its name and source path.

    Recognizes FlyDSL pseudo-ops, runtime-generated kernels, HIP/C++,
    Triton, FlyDSL, plain Python, and vendor-binary backends.

    Args:
        name (str): Kernel symbol/name.
        source_file (str): Resolved source-file path (may be empty).

    Returns:
        str: One of ``flydsl``, ``runtime_generated``, ``hip_cpp``,
            ``triton``, ``python``, ``vendor_binary``, or ``unknown``.
    """
    lower_name = name.lower()
    # Synthetic flydsl pseudo-ops carry no source_file; match the name prefix.
    if any(marker in lower_name for marker in _FLYDSL_PSEUDO_OP_NAME_MARKERS):
        return "flydsl"
    if is_runtime_generated_kernel(name, source_file):
        return "runtime_generated"
    if source_file.endswith((".cu", ".cuh", ".hip", ".cpp", ".h", ".hpp")):
        return "hip_cpp"
    if "triton" in lower_name and source_file.endswith(".py"):
        return "triton"
    # Ahead of the Triton proof: a FlyDSL kernel may import Triton for its own
    # reference path, and the FlyDSL identity is the one its consumers act on.
    if _looks_like_flydsl_source(source_file):
        return "flydsl"
    if _defines_traced_triton_kernel(name, source_file):
        return "triton"
    if source_file.endswith(".py"):
        return "python"
    if "hipblas" in lower_name or "rocblas" in lower_name:
        return "vendor_binary"
    return "unknown"


_RUNTIME_GENERATED_SOURCE_MARKERS = (  # nosec B108 - marker strings, not filesystem writes.
    "/tmp/torchinductor",
    "/torchinductor_",
    "/.cache/torch/inductor",
    "/.triton/cache",
    "/triton/cache",
)

_COMPILE_GENERATED_NAME_MARKERS = (
    "triton_poi_",
    "triton_red_",
    "triton_tem_",
    "torchinductor",
    "inductor",
)


@functools.lru_cache(maxsize=1)
def _framework_source_roots() -> tuple[str, ...]:
    """Resolve framework install roots for reusable-source matching.

    Roots come from ``framework_paths.resolve_known_source_prefixes`` -- a
    kernel path can name a tree that exists on the serving pod and not here, so
    matching must not be filtered by what is on this host. A lower-case variant
    of each (e.g. ``/app/ATOM/atom/`` -> ``/app/atom/atom/``) is also emitted
    for case-insensitive matching.

    Returns:
        The framework install roots, including lower-case variants.
    """
    roots = _resolve_known_source_prefixes()
    out: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for variant in (root, root.lower()):
            if variant and variant not in seen:
                seen.add(variant)
                out.append(variant)
    return tuple(out)


@functools.lru_cache(maxsize=1)
def _aiter_csrc_root() -> str:
    """Resolve aiter's own device-source root from the installed package, where ``aiter/jit/core.py`` resolves it.

    aiter is found, not imported: importing ``aiter.jit.core`` probes the GPU with ``rocminfo``, opens every card the
    process can see (which a card census then reads as a foreign holder on other leases' cards) and raises on a host
    with none. Cached once per process.

    Returns:
        The aiter csrc root (e.g. ``.../aiter_meta/csrc/``), or ``""`` when aiter is not installed or has no ``csrc``.
    """
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        return ""
    package = Path(next(iter(spec.submodule_search_locations)))
    override = os.environ.get(AITER_META_DIR_ENV)
    meta = Path(override) if override else image_aiter_meta(package)
    if not (meta / _AITER_CSRC).is_dir():
        meta = package.parent
    csrc = meta / _AITER_CSRC
    return f"{csrc.resolve().as_posix()}/" if csrc.is_dir() else ""


def _flydsl_reusable_roots() -> tuple[str, ...]:
    """Resolve FlyDSL checkout root(s) for moe_flydsl pseudo-ops.

    ``$DSL2_ROOT`` / ``$FLYDSL_ROOT`` take precedence over the WekaFS default.
    Uncached, so an env override applies without a process restart.

    Returns:
        The lower-cased FlyDSL checkout roots.
    """
    return tuple(dict.fromkeys(r.lower() for r in _resolve_flydsl_source_roots()))


def _reusable_roots() -> tuple[str, ...]:
    """Combine all reusable-source roots used by the patchability gate.

    Returns:
        tuple[str, ...]: Discovered framework roots plus the aiter csrc root
            and FlyDSL checkout roots, deduplicated.
    """
    roots = _framework_source_roots()
    csrc = _aiter_csrc_root()
    if csrc and csrc not in roots:
        roots = roots + (csrc,)
    for fly in _flydsl_reusable_roots():
        if fly not in roots:
            roots = roots + (fly,)
    return roots


# Kernel-name substrings marking an op non-patchable: vendor BLAS, collectives, copies.
_NON_PATCHABLE_NAME_MARKERS: tuple[str, ...] = (
    "rocblas",
    "hipblas",
    "hipblaslt",
    "rocblaslt",
    "tensile",
    "miopen",
    "ck_kernels",
    "nccl",
    "rccl",
    "hipmemcpy",
    "__amd_rocclr_copybuffer",
    "aten::copy",
)

# Vendor BLAS / closed-source compute backends: a candidate whose runtime
# implementation is one of these has no rewritable source (the device body
# lives in a precompiled vendor binary). Matched against the ``library`` field.
_VENDOR_BACKEND_LIBRARIES: frozenset[str] = frozenset(
    {
        "tensile",
        "hipblas",
        "hipblaslt",
        "rocblas",
        "rocblaslt",
        "ck",
        "composable_kernel",
        "ck_kernels",
        "miopen",
    }
)

# torch dispatch interception shims: these forward a tensor op to a vendor
# backend and hold no rewritable device kernel, so a symbol resolving here is a
# dispatch stub. Matched as a POSIX path suffix against the resolved source_file.
_TORCH_DISPATCH_SHIM_SOURCES: tuple[str, ...] = ("vllm/model_executor/parameter.py",)


def is_torch_dispatch_shim_source(source_file: str) -> bool:
    """True when ``source_file`` is a known torch-dispatch interception shim.

    These ``__torch_function__`` / ``__torch_dispatch__`` files forward a
    tensor op to a vendor backend and hold no rewritable device kernel, so a
    symbol attributed here must be treated as a non-reusable dispatch stub
    (same handling as ``@compile_ops`` JIT stubs in ``aiter/ops/moe_op.py``).

    Args:
        source_file: The resolved source-file path.

    Returns:
        ``True`` when the file is a known torch-dispatch interception shim.
    """
    posix = str(source_file or "").replace("\\", "/")
    return any(posix.endswith(suffix) for suffix in _TORCH_DISPATCH_SHIM_SOURCES)


def is_runtime_generated_kernel(name: str, source_file: str) -> bool:
    """Detect torch.compile / Inductor / cache-generated kernels.

    These are not portable across serving runs.

    Args:
        name: The kernel symbol/name.
        source_file: The resolved source-file path.

    Returns:
        ``True`` for a runtime-generated kernel that is not a stable in-repo
        Triton source.
    """
    lower_name = (name or "").lower()
    lower_file = (source_file or "").lower()
    if any(marker in lower_file for marker in _RUNTIME_GENERATED_SOURCE_MARKERS):
        return True
    if any(marker in lower_name for marker in _COMPILE_GENERATED_NAME_MARKERS):
        # A stable in-repo Triton source can still be reusable.
        return not any(root in lower_file for root in _reusable_roots())
    return False


# TraceLens ``resolve_kernel_source`` methods that carry a patchability verdict
# on the ``op_to_source_*`` fields; classify_patchability honors these directly
# instead of re-running its heuristics. ``unresolved`` is excluded on purpose --
# it is a miss, not a verdict, so those rows fall through to the name/library
# heuristics below.
_RESOLVER_VERDICT_METHODS = frozenset(
    {
        "symbol_index",
        "triton_ast",
        "trace_kernel_file",
        "triton_symbol_index",
        "gate_non_patchable",
    }
)


def classify_patchability(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Return ``(reusable, skip_reason)`` for a hot-kernel candidate.

    Single source of truth for the kernel-opt routing gate. Also rejects
    vendor/collective/native-op name markers
    (:data:`_NON_PATCHABLE_NAME_MARKERS`) and library-less ``aten::*`` ops.

    Args:
        candidate: The hot-kernel candidate dict.

    Returns:
        A ``(reusable, skip_reason)`` tuple; ``skip_reason`` is empty when
        reusable, else a short audit explanation.
    """
    source_file = str(candidate.get("source_file") or "")
    name = str(candidate.get("name") or "")
    lower_name = name.lower()
    verdict = _resolver_verdict(candidate, source_file)
    if verdict is not None:
        return verdict
    # A vendor operator with a validated KernelForge task bundle is tuned through launch knobs, never rewritten,
    # so it passes before the source-file and vendor-binary rejections below.
    if match_vendor_operator_playbook(candidate) is not None:
        return True, ""
    if not source_file:
        # A bare launch API has no kernel body to rewrite, which is a different
        # situation from a kernel whose source we merely failed to locate. Say
        # so rather than sending a reader looking for a file that cannot exist.
        if is_runtime_api_name(name):
            return False, "launch API, not a kernel (no rewritable body)"
        return False, "source file not resolved"
    if candidate.get("source_type") == "vendor_binary":
        return False, "vendor binary (no rewritable source)"
    if is_torch_dispatch_shim_source(source_file):
        return False, (f"torch dispatch shim (no rewritable kernel body): {source_file}")
    for marker in _NON_PATCHABLE_NAME_MARKERS:
        if marker in lower_name:
            return False, (f"non-patchable kernel name marker '{marker}' in {name!r}")
    library = str(candidate.get("library") or "").strip().lower()
    if library in _VENDOR_BACKEND_LIBRARIES:
        return False, (
            f"vendor backend library {candidate.get('library')!r} (precompiled binary, no rewritable source)"
        )
    if name.startswith("aten::") and library in {"", "pytorch native"}:
        return False, (
            f"PyTorch native op {name!r} backed by "
            f"{candidate.get('library') or 'unknown'} library "
            "(typically Tensile / vendor backend)"
        )
    if is_runtime_generated_kernel(name, source_file):
        return False, (f"runtime-generated (torch.compile / Inductor cache): {source_file}")
    return _source_location_verdict(source_file, candidate.get("source_type"))


def _resolver_verdict(candidate: dict[str, Any], source_file: str) -> tuple[bool, str] | None:
    """Return TraceLens' resolver verdict as ``(reusable, skip_reason)``, or ``None`` when it carries none.

    The verdict is ground truth: a patchable entry skips the heuristics, and a non-patchable one (Tensile, MIOpen,
    inductor-Triton, a Composable Kernel template) reports its kind as the reason.
    """
    if candidate.get("source_resolution_method") not in _RESOLVER_VERDICT_METHODS:
        return None
    patchable = candidate.get("op_to_source_patchable")
    if patchable is True and source_file:
        return True, ""
    if patchable is False:
        reason = (
            str(candidate.get("op_to_source_reason") or "").strip()
            or str(candidate.get("kernel_kind") or "").strip()
            or "non-rewritable"
        )
        return False, f"source: {reason}"
    return None


def _source_location_verdict(source_file: str, source_type: Any) -> tuple[bool, str]:
    """Gate a resolved, non-vendor source on its framework root, file kind and ``source_type``."""
    lower_file = source_file.lower()
    if not any(root in lower_file for root in _reusable_roots()):
        return False, (f"source not under a reusable framework root: {source_file}")
    if "/csrc/cpp_itfs/" in lower_file and lower_file.endswith(".py"):
        return False, (
            f"cpp_itfs host launcher (device code is in sibling .cuh/.cpp.jinja, not this .py): {source_file}"
        )
    # aiter device-source promotion: a real .cu/.cuh/.hip kernel under /aiter/ is
    # patchable even when the classifier left source_type unknown, since aiter
    # JIT-compiles each op and forge edits it in place.
    if "/aiter/" in lower_file and lower_file.endswith((".cu", ".cuh", ".hip")):
        return True, ""
    if source_type not in {"hip_cpp", "triton", "python", "flydsl"}:
        return False, (f"source_type={source_type!r} not in {{hip_cpp, triton, python, flydsl}}")
    return True, ""


# Wrapper TUs that just dispatch to a precompiled .so/.co, detected by small
# file size + content signature (conservative so real small kernels survive).
_VENDOR_DISPATCH_SIGS = (
    "ctypes.CDLL",  # pure-Python wrapper around .so
    "torch.ops.aiter.",  # registered aten op forwarding
    "_C_aiter.",  # bound C extension forwarding
    "module_name = ",  # aiter jit module loaders
    "AITER_JIT_LOAD",  # aiter macro
    "hipModuleLoad",  # raw .co loader
    "AiterAsmKernel",  # ASM dispatch wrapper
)

_VENDOR_KEYWORD_NAMES = (
    "hipblaslt",
    "rocblaslt",
    "miopen",
    "ck_kernels",
)


def is_vendor_dispatch_wrapper(name: str, source_file: str) -> bool:
    """Detect a thin dispatch wrapper around a precompiled vendor binary.

    Uses a small file size plus content signatures so that real small kernels
    survive (conservative).

    Args:
        name: The kernel symbol/name.
        source_file: The resolved source-file path.

    Returns:
        ``True`` when the source is a dispatch wrapper around a ``.so``/``.co``
        with nothing to rewrite.
    """
    nm = (name or "").lower()
    if any(kw in nm for kw in _VENDOR_KEYWORD_NAMES):
        return True
    if not source_file:
        return False
    p = Path(source_file)
    try:
        if not p.is_file():
            return False
        # >16 KB is presumed a real device kernel; wrappers/shims sit well below.
        if p.stat().st_size > 16 * 1024:
            return False
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(sig in text for sig in _VENDOR_DISPATCH_SIGS)
