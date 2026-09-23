# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Client-side fuzzy identity ranking for KernelForge warm-start reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from packaging.version import InvalidVersion, Version

from kernelforge.knowledge.implementation_identity import canonical_framework_version
from kernelforge.knowledge.kernel_identity import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)

_UNUSABLE = {"", "unknown", "none", "unspecified"}
_GPU_ISA = {
    "mi300x": "gfx942",
    "amd_instinct_mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi355x": "gfx950",
    "amd_instinct_mi355x": "gfx950",
}


@dataclass(frozen=True)
class _RankedIdentity:
    canonical_id: str
    score: int
    version_affinity: int
    gpu_affinity: int
    version_distance: tuple[int, ...]
    updated_at: str


def _release(canonical_version: str) -> Version | None:
    """Parse a canonical version, or ``None`` when it names no release."""
    try:
        return Version(canonical_version)
    except InvalidVersion:
        return None


def _version_affinity(target: Version, candidate: Version) -> int:
    if candidate == target:
        return 3
    if candidate.release[:2] == target.release[:2]:
        return 2
    if candidate.release[:1] == target.release[:1]:
        return 1
    return 0


def _version_distance(target: Version, candidate: Version) -> tuple[int, ...]:
    width = max(3, len(target.release), len(candidate.release))
    target_release = (*target.release, *(0 for _ in range(width - len(target.release))))
    candidate_release = (
        *candidate.release,
        *(0 for _ in range(width - len(candidate.release))),
    )
    return tuple(abs(left - right) for left, right in zip(target_release, candidate_release))


def _gpu_affinity(target: str, candidate: str) -> int:
    target_value = str(target or "").strip().lower()
    candidate_value = str(candidate or "").strip().lower()
    if candidate_value == target_value:
        return 3
    target_isa = _GPU_ISA.get(target_value)
    candidate_isa = _GPU_ISA.get(candidate_value)
    if target_isa and target_isa == candidate_isa:
        return 2
    return 1


def rank_fallback_identities(
    target: KernelRecipeIdentity,
    rows: list[Mapping[str, Any]],
) -> list[str]:
    """Return fuzzy donor identities best-first.

    Producer, kernel name, framework and backend remain exact. Framework
    version and GPU are soft ranking dimensions: known cross-version and
    cross-ISA donors remain eligible. The exact identity is omitted because
    callers probe it first.

    The target's version is resolved by :func:`canonical_framework_version`
    because it arrives raw -- installed distribution metadata says
    ``0.24.0+rocm723`` and an image tag says ``v0.24.0``. A stored dimension is
    not resolved again: every page is written at its canonical address, so a
    stored spelling that needs resolving is a page written by something that
    skipped the address rules, and reading it as the release it names would hide
    that rather than report it. Such a page ranks as unparseable and is dropped.

    Two runs that both failed to observe a version can still reach each other,
    because both resolve to the one word for that. What stays rejected is
    ranking *across* it: how far a known release sits from an unknown one is not
    a question the strings can answer.
    """
    target_version = canonical_framework_version(target.framework_version)
    target_release = _release(target_version)
    target_gpu = target.gpu.strip().lower()
    if target_gpu in _UNUSABLE:
        return []

    exact_id = kernel_recipe_canonical_id(target)
    fixed = {
        "producer": target.producer.strip().lower(),
        "kernel_name": target.kernel_name.strip().lower(),
        "framework": target.framework.strip().lower(),
        "backend": target.backend.strip().lower(),
    }
    ranked: list[_RankedIdentity] = []
    seen = {exact_id}
    for row in rows:
        canonical_id = str(row.get("canonical_id") or "").strip()
        dimensions = row.get("dimensions")
        if not canonical_id or canonical_id in seen or not isinstance(dimensions, Mapping):
            continue
        values = {str(key): str(value or "").strip().lower() for key, value in dimensions.items()}
        if any(values.get(key) != expected for key, expected in fixed.items()):
            continue
        candidate_gpu = values.get("gpu", "")
        if candidate_gpu in _UNUSABLE:
            continue
        candidate_version = values.get("framework_version", "")
        candidate_release = _release(candidate_version)
        if candidate_version == target_version:
            version_affinity = 3
            version_distance: tuple[int, ...] = ()
        elif target_release is None or candidate_release is None:
            continue
        else:
            version_affinity = _version_affinity(target_release, candidate_release)
            version_distance = _version_distance(target_release, candidate_release)
        seen.add(canonical_id)
        gpu_affinity = _gpu_affinity(target_gpu, candidate_gpu)
        ranked.append(
            _RankedIdentity(
                canonical_id=canonical_id,
                score=version_affinity + gpu_affinity,
                version_affinity=version_affinity,
                gpu_affinity=gpu_affinity,
                version_distance=version_distance,
                updated_at=str(row.get("updated_at") or ""),
            )
        )

    # Stable ranking: newest breaks a complete similarity tie.
    ranked.sort(key=lambda item: item.updated_at, reverse=True)
    ranked.sort(key=lambda item: item.version_distance)
    ranked.sort(key=lambda item: item.gpu_affinity, reverse=True)
    ranked.sort(key=lambda item: item.version_affinity, reverse=True)
    ranked.sort(key=lambda item: item.score, reverse=True)
    return [item.canonical_id for item in ranked]


__all__ = ["rank_fallback_identities"]
