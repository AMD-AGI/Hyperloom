# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared kernel backend constants — a dependency-free leaf module."""

from __future__ import annotations

from pathlib import Path

# Registry of backend → prompt module.
KERNEL_BACKEND_PROMPT_MODULES = {
    "ck": "kernelforge.kernel_backends.ck.prompts",
    "flydsl": "kernelforge.kernel_backends.flydsl.prompts",
    "triton": "kernelforge.kernel_backends.triton.prompts",
    "gluon": "kernelforge.kernel_backends.gluon.prompts",
    "aiter": "kernelforge.kernel_backends.aiter.prompts",
    "hip": "kernelforge.kernel_backends.hip.prompts",
    "hipblaslt": "kernelforge.kernel_backends.hipblaslt.prompts",
    "fusion": "kernelforge.kernel_backends.fusion.prompts",
}
KERNEL_BACKENDS = list(KERNEL_BACKEND_PROMPT_MODULES)

# The languages/ subdirectories serving each backend, in reading order.
_BACKEND_LANGUAGE_DIRS: dict[str, tuple[str, ...]] = {
    "triton": ("triton", "gluon"),
    "gluon": ("gluon", "triton"),
}


def resolve_language_dirs(backend: str, local_knowledge_root: Path | str) -> tuple[str, ...]:
    """Return the languages/ subdirectories serving ``backend``, in reading order."""
    if not backend:
        return ()
    root = Path(local_knowledge_root)
    names = _BACKEND_LANGUAGE_DIRS.get(backend, (backend,))
    return tuple(name for name in names if (root / "languages" / name).is_dir())


def resolve_language_dir(backend: str, local_knowledge_root: Path | str) -> str | None:
    """Return the PRIMARY languages/ subdirectory serving ``backend``, or None."""
    dirs = resolve_language_dirs(backend, local_knowledge_root)
    return dirs[0] if dirs else None
