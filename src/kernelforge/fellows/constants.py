# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared fellow constants — a dependency-free leaf module."""

from __future__ import annotations

from pathlib import Path

# Registry of backend → prompt module. The iteration loop loads from this
# registry so availability cannot drift between backend validation and prompt
# construction.
FELLOW_PROMPT_MODULES = {
    "ck": "kernelforge.fellows.ck.prompts",
    "flydsl": "kernelforge.fellows.flydsl.prompts",
    "triton": "kernelforge.fellows.triton.prompts",
    "gluon": "kernelforge.fellows.gluon.prompts",
    "aiter": "kernelforge.fellows.aiter.prompts",
    "hip": "kernelforge.fellows.hip.prompts",
    "hipblaslt": "kernelforge.fellows.hipblaslt.prompts",
    "intellikit": "kernelforge.fellows.intellikit.prompts",
    "fusion": "kernelforge.fellows.fusion.prompts",
}
FELLOW_BACKENDS = list(FELLOW_PROMPT_MODULES)

# The languages/ subdirectories serving each backend, in reading order. A
# backend absent from this map serves the folder named after itself, which is
# the ordinary case.
#
# Two backends need more than one. The intellikit fellow writes AMDGCN assembly,
# so its tree is keyed by that language rather than by the fellow. Triton and
# Gluon are one toolchain -- same frontend, same JIT, same
# Triton -> TritonGPU -> TritonAMDGPU -> AMDGCN lowering, same cache -- differing
# only in who assigns layouts, pipeline stages, register budget and the MFMA. So
# each carries the other: a Triton campaign has to know that dropping to Gluon
# is an available move rather than a different project, and a Gluon kernel still
# needs the shared compile-pipeline and ISA-verification cards that live under
# languages/triton/. The pairing is what lets the Gluon tree stay thin instead of
# restating the substrate, and it means a misinferred fellow costs a prompt
# template rather than a whole knowledge layer.
_BACKEND_LANGUAGE_DIRS: dict[str, tuple[str, ...]] = {
    "intellikit": ("asm",),
    "triton": ("triton", "gluon"),
    "gluon": ("gluon", "triton"),
}


def resolve_language_dirs(
    backend: str, local_knowledge_root: Path | str
) -> tuple[str, ...]:
    """Return the languages/ subdirectories serving ``backend``, in reading order.

    Each candidate is filtered on existence, so a checkout missing one folder
    degrades to the folders it has rather than emitting a dead section. Backends
    with no language layer (aiter, hipblaslt) resolve to an empty tuple and the
    knowledge builder skips the layer entirely.
    """
    if not backend:
        return ()
    root = Path(local_knowledge_root)
    names = _BACKEND_LANGUAGE_DIRS.get(backend, (backend,))
    return tuple(name for name in names if (root / "languages" / name).is_dir())


def resolve_language_dir(backend: str, local_knowledge_root: Path | str) -> str | None:
    """Return the PRIMARY languages/ subdirectory serving ``backend``, or None.

    The backend's own language, for callers that need one name rather than the
    whole reading order. Prefer :func:`resolve_language_dirs` when assembling
    knowledge, so a backend that reads a second language does not lose it.
    """
    dirs = resolve_language_dirs(backend, local_knowledge_root)
    return dirs[0] if dirs else None
