"""Framework source-root resolution for PolicyGate and flag discovery.

Containers may ship framework code under ``/sgl-workspace/{aiter,sglang,vllm}``
or under ``site-packages`` when only pip wheels are present. This module
centralises probe order so PolicyGate, AST discovery, and install.sh agree.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_DEFAULT_SGLANG_SERVER_ARGS = Path(
    "/sgl-workspace/sglang/python/sglang/srt/server_args.py"
)
_DEFAULT_VLLM_ARG_UTILS = Path(
    "/sgl-workspace/vllm/vllm/engine/arg_utils.py"
)

_DEFAULT_SOURCE_ROOTS: tuple[str, ...] = (
    "/sgl-workspace/aiter/",
    "/sgl-workspace/sglang/",
    "/sgl-workspace/vllm/",
)


def _normalize_root(path: str) -> str:
    p = str(path or "").strip()
    if not p:
        return ""
    return p if p.endswith("/") else f"{p}/"


def resolve_source_file_allowlist() -> tuple[str, ...]:
    """Return PolicyGate ``source_file`` allowlist roots (default ∪ env)."""
    env = os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", "").strip()
    if not env:
        return _DEFAULT_SOURCE_ROOTS
    extra = tuple(
        _normalize_root(p) for p in env.split(":") if p.strip()
    )
    seen: set[str] = set()
    out: list[str] = []
    for root in (*_DEFAULT_SOURCE_ROOTS, *extra):
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


def probe_framework_source_roots_for_env() -> str:
    """Colon-separated roots for ``INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS``."""
    found: list[str] = []
    seen: set[str] = set()
    for root in _DEFAULT_SOURCE_ROOTS:
        p = Path(root.rstrip("/"))
        if p.is_dir() and root not in seen:
            seen.add(root)
            found.append(_normalize_root(str(p)))
    for mod in ("vllm", "sglang", "aiter"):
        origin = _find_spec_origin(mod)
        if origin is not None:
            r = _normalize_root(str(origin))
            if r and r not in seen:
                seen.add(r)
                found.append(r)
    venv = os.environ.get("VIRTUAL_ENV", "").strip()
    if venv:
        site = Path(venv) / "lib"
        if site.is_dir():
            for pattern in ("python*/site-packages/vllm", "python*/site-packages/sglang",
                            "python*/site-packages/aiter"):
                for match in sorted(site.glob(pattern)):
                    r = _normalize_root(str(match))
                    if r not in seen:
                        seen.add(r)
                        found.append(r)
    return ":".join(found)


__all__ = [
    "probe_framework_source_roots_for_env",
    "resolve_sglang_server_args_path",
    "resolve_source_file_allowlist",
    "resolve_vllm_arg_utils_path",
]
