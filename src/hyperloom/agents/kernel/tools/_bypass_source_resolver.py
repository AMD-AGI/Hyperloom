###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Independent op -> editable-source resolver for the bypass analysis backend.

Used by the bypass route (``HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass``) to populate
``source_file`` on hot-kernel candidates so the downstream kernel optimizer can
dispatch a rewrite (it filters out candidates with no ``source_file``).

A compact, independent reimplementation of the op->source lookup that reads only
the shared data file
``src/hyperloom/agents/kernel/tools/data/op_to_source.json`` and never imports
TraceLens.

The dictionary maps a CPU op name to the device kernels seen per container
(``vllm`` / ``sglang``), each ``{device_kernel_name: {kernel_source_path,
kernel_kind, patchable}}`` plus a top-level ``kind`` (``single`` / ``dispatch`` /
``composite``). An op resolves to an *editable* source when its selected
container holds a ``patchable`` kernel whose source is native
(``.cu``/``.cuh``/``.hip``/``.h``) or a repo-resident (non-inductor, non-``/tmp``)
Triton ``.py``.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any

_OP_TO_SOURCE_JSON = Path(__file__).resolve().parent / "data" / "op_to_source.json"

# Steady-state phase suffix stamped onto some op names (e.g. "aten::mm (decode)").
_PHASE_SUFFIX_RE = re.compile(r"\s*\((?:prefill|decode|prefilldecode|mixed)\)\s*$")

# Editable source extensions: native device code plus repo-resident Triton .py.
_NATIVE_SOURCE_EXTS = (".cu", ".cuh", ".hip", ".h")
# dist-packages root; relative JSON paths are absolutized against it.
_PY_DIST_ROOT = "/usr/local/lib/python3.12/dist-packages/"


@functools.lru_cache(maxsize=1)
def _load_mapping() -> dict[str, Any]:
    """Load and cache the shared ``op_to_source.json`` (``{}`` on any failure)."""
    try:
        with open(_OP_TO_SOURCE_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


@functools.lru_cache(maxsize=1)
def _aiter_csrc_root() -> str:
    """Best-effort live aiter ``csrc/`` root via find-spec (no import), else ``""``.

    Discovering the live root lets the remap below recover CK/aiter native sources
    when aiter lives outside the JSON's build-time path. ``find_spec`` locates the
    package without importing it.
    """
    try:
        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError, ModuleNotFoundError):
        return ""
    if spec is None:
        return ""
    for loc in list(getattr(spec, "submodule_search_locations", None) or []):
        cand = os.path.join(loc, "csrc")
        if os.path.isdir(cand):
            return cand + "/"
    return ""


def _remap_aiter_meta(path: str) -> str:
    """Remap a build-time ``…/aiter_meta/csrc/`` path to the live aiter root.

    No-op when the path is not an ``aiter_meta`` path, the live root is unknown,
    or the remapped path does not exist on disk (keeps the original then).
    """
    if not path or "aiter_meta/csrc/" not in path:
        return path
    live = _aiter_csrc_root()
    if not live:
        return path
    tail = path[path.find("aiter_meta/csrc/") + len("aiter_meta/csrc/"):]
    remapped = live.rstrip("/") + "/" + tail
    return remapped if os.path.exists(remapped) else path


def _absolutize(path: str) -> str:
    """Absolutize a JSON source path (prepend dist-root if relative) + aiter remap."""
    if not path:
        return path
    abs_path = path if path.startswith("/") else _PY_DIST_ROOT + path
    return _remap_aiter_meta(abs_path)


def is_editable_source(path: str | None, kernel_kind: str | None = None) -> bool:
    """Return whether ``path`` is a source we can route a kernel rewrite at.

    Editable == native device code (``.cu``/``.cuh``/``.hip``/``.h``) or a
    repo-resident Triton/TileLang ``.py``. Generated Triton is excluded
    (``triton_inductor_generated`` kind and any ``torchinductor`` / ``/tmp/``
    path).

    Args:
        path: Candidate source path (from the JSON or a trace ``kernel_file``).
        kernel_kind: Optional kernel-kind hint from the JSON leaf.

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
        if "torchinductor" in path or path.startswith("/tmp/"):
            return False
        return True
    return False


def _exists(path: str) -> bool:
    """``os.path.exists`` guarded against odd paths (never raises)."""
    try:
        return bool(path) and os.path.exists(path)
    except OSError:
        return False


def _container_sources(container: dict[str, Any] | None) -> list[str]:
    """Editable, patchable, absolutized source paths for one container (deduped)."""
    out: list[str] = []
    seen: set[str] = set()
    for info in (container or {}).values():
        if not isinstance(info, dict) or not info.get("patchable"):
            continue
        raw = info.get("kernel_source_path")
        if not is_editable_source(raw, info.get("kernel_kind")):
            continue
        abs_path = _absolutize(str(raw))
        if abs_path in seen:
            continue
        seen.add(abs_path)
        out.append(abs_path)
    return out


def _select_container_sources(entry: dict[str, Any], framework: str) -> list[str]:
    """Pick the editable source list from the better container.

    Prefer whichever container is present on disk; otherwise honor the
    ``framework`` hint (only ``vllm`` / ``sglang`` recognized); else default to
    sglang, then vllm.
    """
    sgl = _container_sources(entry.get("sglang"))
    vll = _container_sources(entry.get("vllm"))
    if not (sgl or vll):
        return []
    sgl_present = any(_exists(p) for p in sgl)
    vll_present = any(_exists(p) for p in vll)
    if sgl_present and not vll_present:
        return sgl
    if vll_present and not sgl_present:
        return vll
    fw = (framework or "").strip().lower()
    if fw == "vllm" and vll:
        return vll
    if fw == "sglang" and sgl:
        return sgl
    return sgl or vll


def _dispatch_sources(entry: dict[str, Any], framework: str, device_kernel_name: str) -> list[str]:
    """Resolve a ``dispatch`` op: the one kernel whose name matches the trace.

    Falls back to container selection when the device kernel name is unknown or
    not found (a dictionary that types the op as dispatch but lacks that exact
    kernel still yields its editable sources).
    """
    if device_kernel_name:
        for cont_name in ("vllm", "sglang"):
            info = (entry.get(cont_name) or {}).get(device_kernel_name)
            if isinstance(info, dict) and info.get("patchable"):
                raw = info.get("kernel_source_path")
                if is_editable_source(raw, info.get("kernel_kind")):
                    return [_absolutize(str(raw))]
    return _select_container_sources(entry, framework)


def resolve_source(
    op_name: str,
    *,
    framework: str = "",
    device_kernel_name: str = "",
) -> tuple[str, str]:
    """Resolve a CPU op name to an editable source file.

    Args:
        op_name: The launching op name (e.g. ``_C::silu_and_mul``).
        framework: Serving framework hint for container selection.
        device_kernel_name: Device kernel name (used for ``dispatch`` ops).

    Returns:
        ``(source_file, "op_to_source")`` on a hit (an on-disk source is
        preferred when several editable sources exist), or ``("", "unresolved")``
        on a dictionary miss / no editable source.
    """
    mapping = _load_mapping()
    if not mapping or not op_name:
        return "", "unresolved"
    key = _PHASE_SUFFIX_RE.sub("", op_name)
    entry = mapping.get(key)
    if not isinstance(entry, dict):
        return "", "unresolved"
    kind = str(entry.get("kind") or "single")
    if kind == "dispatch":
        sources = _dispatch_sources(entry, framework, device_kernel_name)
    else:  # single / composite both dedup the selected container's editable src
        sources = _select_container_sources(entry, framework)
    if not sources:
        return "", "unresolved"
    for p in sources:
        if _exists(p):
            return p, "op_to_source"
    return sources[0], "op_to_source"


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


__all__ = ["resolve_source", "editable_trace_source", "is_editable_source"]
