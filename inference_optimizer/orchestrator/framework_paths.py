# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Framework source-root resolution for PolicyGate and flag discovery.

Containers may ship framework code under ``/sgl-workspace/{aiter,sglang,vllm}``
or ``/app/ATOM/atom`` (atom editable install), or under ``site-packages`` /
``dist-packages`` when only pip wheels are present. This module centralises
probe order so PolicyGate, AST discovery, install.sh, and
``apply_kernel_patch`` all agree.

Three frameworks are first-class today (alphabetical):

* atom    — ``/app/ATOM/atom/`` editable install layout, ``arg_utils.py``
  under ``model_engine/``.
* sglang  — ``/sgl-workspace/sglang/python/sglang/srt/server_args.py``.
* vllm    — ``/sgl-workspace/vllm/vllm/engine/arg_utils.py``.

aiter is included in the source-roots allowlist because it's a kernel
library shared across all three frameworks (atom imports it for fused
MoE / MLA paths the same way sglang/vllm do).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_DEFAULT_SGLANG_SERVER_ARGS = Path(
    "/sgl-workspace/sglang/python/sglang/srt/server_args.py"
)
_DEFAULT_VLLM_ARG_UTILS = Path(
    "/sgl-workspace/vllm/vllm/engine/arg_utils.py"
)
_DEFAULT_ATOM_ARG_UTILS = Path(
    "/app/ATOM/atom/model_engine/arg_utils.py"
)

_DEFAULT_SOURCE_ROOTS: tuple[str, ...] = (
    "/sgl-workspace/aiter/",
    "/sgl-workspace/sglang/",
    "/sgl-workspace/vllm/",
    # atom's editable-install layout. Production atom boxes also expose
    # atom under ``/opt/venv/lib/pythonX.Y/site-packages/atom/`` — that
    # path is picked up dynamically by ``probe_framework_source_roots_for_env``
    # via the VIRTUAL_ENV glob below, mirroring the sglang/vllm pattern.
    "/app/ATOM/atom/",
)

_FRAMEWORK_PACKAGES: tuple[str, ...] = ("aiter", "sglang", "vllm", "atom")

# Parents scanned for ``python*/{site,dist}-packages/<pkg>`` wheel layouts.
_INSTALL_GLOB_PARENTS: tuple[Path, ...] = (
    Path("/usr/local/lib"),
    Path("/opt/venv/lib"),
)

# aiter device sources often live in the sibling ``aiter_meta`` package.
_AITER_META_CSRC_ROOT = "/aiter_meta/csrc/"

# Minimal static fallbacks when importlib/glob find nothing (image defaults).
_STATIC_PATCH_FALLBACK_ROOTS: tuple[str, ...] = (
    "/opt/venv/lib/python3.10/site-packages/aiter/",
    "/opt/venv/lib/python3.10/site-packages/sglang/",
    "/opt/venv/lib/python3.10/site-packages/vllm/",
    "/opt/venv/lib/python3.10/site-packages/atom/",
    "/opt/venv/lib/python3.12/site-packages/aiter/",
    "/opt/venv/lib/python3.12/site-packages/sglang/",
    "/opt/venv/lib/python3.12/site-packages/vllm/",
    "/opt/venv/lib/python3.12/site-packages/atom/",
    "/usr/local/lib/python3.12/dist-packages/aiter/",
    "/usr/local/lib/python3.12/dist-packages/sglang/",
    "/usr/local/lib/python3.12/dist-packages/vllm/",
    "/usr/local/lib/python3.12/dist-packages/atom/",
    "/usr/local/lib/python3.10/dist-packages/aiter/",
    "/usr/local/lib/python3.10/dist-packages/sglang/",
    "/usr/local/lib/python3.10/dist-packages/vllm/",
    "/usr/local/lib/python3.10/dist-packages/atom/",
    "/app/ATOM/atom/",
    _AITER_META_CSRC_ROOT,
)


def _normalize_root(path: str) -> str:
    p = str(path or "").strip()
    if not p:
        return ""
    return p if p.endswith("/") else f"{p}/"


def _merge_roots(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for root in group:
            if root and root not in seen:
                seen.add(root)
                out.append(root)
    return tuple(out)


def _find_spec_origin(module_name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    origin = Path(spec.origin)
    if origin.name == "__init__.py":
        return origin.parent
    return origin.parent


def _glob_install_package_roots() -> tuple[str, ...]:
    """Discover framework package dirs under common lib layouts."""
    patterns = (
        "python*/dist-packages/aiter",
        "python*/dist-packages/sglang",
        "python*/dist-packages/vllm",
        "python*/dist-packages/atom",
        "python*/site-packages/aiter",
        "python*/site-packages/sglang",
        "python*/site-packages/vllm",
        "python*/site-packages/atom",
    )
    found: list[str] = []
    seen: set[str] = set()
    parents: list[Path] = list(_INSTALL_GLOB_PARENTS)
    prefix_lib = Path(sys.prefix) / "lib"
    if prefix_lib.is_dir() and prefix_lib not in parents:
        parents.append(prefix_lib)
    for parent in parents:
        if not parent.is_dir():
            continue
        for pattern in patterns:
            for match in sorted(parent.glob(pattern)):
                if not match.is_dir():
                    continue
                root = _normalize_root(str(match))
                if root and root not in seen:
                    seen.add(root)
                    found.append(root)
    return tuple(found)


def _discover_installed_framework_roots() -> tuple[str, ...]:
    """Runtime discovery via importlib and filesystem globs."""
    found: list[str] = []
    seen: set[str] = set()

    def add(path: str | Path) -> None:
        root = _normalize_root(str(path))
        if root and root not in seen:
            seen.add(root)
            found.append(root)

    for mod in _FRAMEWORK_PACKAGES:
        origin = _find_spec_origin(mod)
        if origin is not None:
            add(origin)

    venv = os.environ.get("VIRTUAL_ENV", "").strip()
    if venv:
        site = Path(venv) / "lib"
        if site.is_dir():
            for pattern in (
                "python*/site-packages/vllm",
                "python*/site-packages/sglang",
                "python*/site-packages/aiter",
                "python*/site-packages/atom",
            ):
                for match in sorted(site.glob(pattern)):
                    if match.is_dir():
                        add(match)

    for root in _glob_install_package_roots():
        add(root)

    return tuple(found)


def resolve_source_file_allowlist() -> tuple[str, ...]:
    """Return PolicyGate ``source_file`` allowlist roots (default ∪ discovered ∪ env)."""
    env = os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", "").strip()
    env_roots = tuple(
        _normalize_root(p) for p in env.split(":") if p.strip()
    ) if env else ()
    return _merge_roots(
        _DEFAULT_SOURCE_ROOTS,
        _discover_installed_framework_roots(),
        env_roots,
    )


def resolve_patch_target_roots() -> tuple[str, ...]:
    """Roots for substring matching in patch apply + kernel classifiers.

    Same as :func:`resolve_source_file_allowlist` plus static fallbacks for
    layouts that are not importable until first use (e.g. ``aiter_meta/csrc``).
    """
    return _merge_roots(
        resolve_source_file_allowlist(),
        _STATIC_PATCH_FALLBACK_ROOTS,
    )


def resolve_sglang_server_args_path() -> tuple[Path, str]:
    """Resolve SGLang server_args.py for AST discovery."""
    override = os.environ.get("INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS", "").strip()
    if override:
        p = Path(override)
        if p.is_file():
            return p, str(p)
        return p, f"INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS={override} not found"
    if _DEFAULT_SGLANG_SERVER_ARGS.is_file():
        return _DEFAULT_SGLANG_SERVER_ARGS, str(_DEFAULT_SGLANG_SERVER_ARGS)
    origin = _find_spec_origin("sglang")
    if origin is not None:
        candidate = origin / "srt" / "server_args.py"
        if candidate.is_file():
            return candidate, str(candidate)
        for alt in (
            origin / "python" / "sglang" / "srt" / "server_args.py",
            origin / "sglang" / "srt" / "server_args.py",
        ):
            if alt.is_file():
                return alt, str(alt)
    return _DEFAULT_SGLANG_SERVER_ARGS, (
        f"sglang server_args not found (checked {_DEFAULT_SGLANG_SERVER_ARGS})"
    )


def resolve_vllm_arg_utils_path() -> tuple[Path, str]:
    """Resolve vLLM arg_utils.py for AST discovery."""
    override = os.environ.get("INFERENCE_OPTIMIZER_VLLM_ARG_UTILS", "").strip()
    if override:
        p = Path(override)
        if p.is_file():
            return p, str(p)
        return p, f"INFERENCE_OPTIMIZER_VLLM_ARG_UTILS={override} not found"
    if _DEFAULT_VLLM_ARG_UTILS.is_file():
        return _DEFAULT_VLLM_ARG_UTILS, str(_DEFAULT_VLLM_ARG_UTILS)
    origin = _find_spec_origin("vllm")
    if origin is not None:
        candidate = origin / "engine" / "arg_utils.py"
        if candidate.is_file():
            return candidate, str(candidate)
        for alt in (
            origin / "vllm" / "engine" / "arg_utils.py",
        ):
            if alt.is_file():
                return alt, str(alt)
    return _DEFAULT_VLLM_ARG_UTILS, (
        f"vllm arg_utils not found (checked {_DEFAULT_VLLM_ARG_UTILS})"
    )


def resolve_atom_arg_utils_path() -> tuple[Path, str]:
    """Resolve atom ``model_engine/arg_utils.py`` for AST discovery.

    Symmetric with ``resolve_sglang_server_args_path()`` and
    ``resolve_vllm_arg_utils_path()``:

    1. Honour ``$INFERENCE_OPTIMIZER_ATOM_ARG_UTILS`` if set (operator
       override for non-default layouts).
    2. Check the default editable-install location
       ``/app/ATOM/atom/model_engine/arg_utils.py``.
    3. Fall back to ``importlib.util.find_spec("atom")`` and walk to
       ``<origin>/model_engine/arg_utils.py``.

    Returns ``(Path, str)`` where the str is the file path on success or
    a diagnostic message on failure (same contract as the sister
    helpers).
    """
    override = os.environ.get("INFERENCE_OPTIMIZER_ATOM_ARG_UTILS", "").strip()
    if override:
        p = Path(override)
        if p.is_file():
            return p, str(p)
        return p, f"INFERENCE_OPTIMIZER_ATOM_ARG_UTILS={override} not found"
    if _DEFAULT_ATOM_ARG_UTILS.is_file():
        return _DEFAULT_ATOM_ARG_UTILS, str(_DEFAULT_ATOM_ARG_UTILS)
    origin = _find_spec_origin("atom")
    if origin is not None:
        candidate = origin / "model_engine" / "arg_utils.py"
        if candidate.is_file():
            return candidate, str(candidate)
        for alt in (
            origin / "atom" / "model_engine" / "arg_utils.py",
        ):
            if alt.is_file():
                return alt, str(alt)
    return _DEFAULT_ATOM_ARG_UTILS, (
        f"atom arg_utils not found (checked {_DEFAULT_ATOM_ARG_UTILS})"
    )


def probe_framework_source_roots_for_env() -> str:
    """Colon-separated roots for ``INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS``."""
    found: list[str] = []
    for root in resolve_source_file_allowlist():
        p = Path(root.rstrip("/"))
        if p.is_dir():
            found.append(_normalize_root(str(p)))
    return ":".join(found)


# ---------------------------------------------------------------------------
# Discovery summary — operator-facing log helper.
# install.sh consumes this to emit a one-line ``sglang=ok atom=ok ...``
# summary after the colon-separated probe. Preflight greps the line, so
# keep its format stable.
# ---------------------------------------------------------------------------

# Buckets are ordered so substring matching is deterministic — atom is
# checked BEFORE vllm/sglang to keep parity with
# ``server_args_env_name``'s ordering convention (atom paths never contain
# vllm/sglang substrings today, but the explicit ordering keeps a future
# framework name like "atom-vllm" from accidentally falling into the
# wrong bucket).
_FRAMEWORK_BUCKETS: tuple[str, ...] = ("atom", "vllm", "sglang", "aiter")


def summarise_framework_root_discovery(roots: str) -> str:
    """Return ``"sglang=ok atom=missing ..."``-style one-line summary.

    Input is the colon-separated string emitted by
    ``probe_framework_source_roots_for_env``. Output is a single
    space-separated line where each framework reports ``=ok`` if any
    discovered root path contains the framework name and ``=missing``
    otherwise. Buckets are emitted in the order declared by
    ``_FRAMEWORK_BUCKETS`` so the line is stable across runs.

    Used by ``install.sh`` to give operators a one-line "did atom get
    picked up" answer; tested via the matching pytest case rather than
    via install.sh shell smoke (the helper is operator-visible
    observability, not a control-flow boundary).
    """
    parts: list[str] = []
    items = [p.strip().lower() for p in (roots or "").split(":") if p.strip()]
    for fw in _FRAMEWORK_BUCKETS:
        token = f"/{fw}/"
        status = "ok" if any(item.endswith(token) for item in items) else "missing"
        parts.append(f"{fw}={status}")
    return " ".join(parts)


__all__ = [
    "probe_framework_source_roots_for_env",
    "resolve_atom_arg_utils_path",
    "resolve_patch_target_roots",
    "resolve_sglang_server_args_path",
    "resolve_source_file_allowlist",
    "resolve_vllm_arg_utils_path",
    "summarise_framework_root_discovery",
]
