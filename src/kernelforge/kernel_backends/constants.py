# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared kernel backend constants — a dependency-free leaf module."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# Registry of backend → prompt module.
KERNEL_BACKEND_PROMPT_MODULES = {
    "assembly": "kernelforge.kernel_backends.assembly.prompts",
    "ck": "kernelforge.kernel_backends.ck.prompts",
    "flydsl": "kernelforge.kernel_backends.flydsl.prompts",
    "triton": "kernelforge.kernel_backends.triton.prompts",
    "gluon": "kernelforge.kernel_backends.gluon.prompts",
    "aiter": "kernelforge.kernel_backends.aiter.prompts",
    "hip": "kernelforge.kernel_backends.hip.prompts",
    "hipblaslt": "kernelforge.kernel_backends.hipblaslt.prompts",
    "fusion": "kernelforge.kernel_backends.fusion.prompts",
    "ave": "kernelforge.kernel_backends.ave.prompts",
    "mono": "kernelforge.kernel_backends.mono.prompts",
}
KERNEL_BACKENDS = list(KERNEL_BACKEND_PROMPT_MODULES)

_FLYDSL_LANGUAGE_DIRS = ("flydsl", "assembly")

# The languages/ subdirectories serving each backend, in reading order.
_BACKEND_LANGUAGE_DIRS: dict[str, tuple[str, ...]] = {
    "flydsl": _FLYDSL_LANGUAGE_DIRS,
    "assembly": ("assembly", "flydsl", "triton", "gluon", "hip"),
    "triton": ("triton", "gluon"),
    "gluon": ("gluon", "triton"),
    "mono": (*_FLYDSL_LANGUAGE_DIRS, "hip"),
}

# The package directory every sgl-amd-mono source lives under.
SGL_AMD_MONO_PACKAGE = "sgl_amd_mono"


def includes_sgl_amd_mono(backend: str, source_paths: Sequence[str | Path] | None) -> bool:
    """Whether ``framework/sgl_amd_mono/`` serves the task: a monokernel campaign, or sources in that package."""
    return backend == "mono" or any(SGL_AMD_MONO_PACKAGE in Path(str(path)).parts for path in source_paths or ())


def resolve_language_dirs(backend: str, local_knowledge_root: Path | str) -> tuple[str, ...]:
    """Return the languages/ subdirectories serving ``backend``, in reading order."""
    if not backend:
        return ()
    root = Path(local_knowledge_root)
    names = _BACKEND_LANGUAGE_DIRS.get(backend, (backend,))
    return tuple(name for name in names if (root / "languages" / name).is_dir())
