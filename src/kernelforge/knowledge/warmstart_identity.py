# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Client-side fuzzy identity ranking for KernelForge warm-start reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from packaging.version import InvalidVersion, Version

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


def _version(value: str) -> Version | None:
    raw = str(value or "").strip().lower()
    if raw in _UNUSABLE:
        return None
    try:
        return Version(raw)
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
    cross-ISA donors remain eligible, while missing/unknown metadata is
    rejected. The exact identity is omitted because callers probe it first.
    """
    target_version = _version(target.framework_version)
    target_gpu = target.gpu.strip().lower()
    if target_version is None or target_gpu in _UNUSABLE:
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
        values = {
            str(key): str(value or "").strip().lower()
            for key, value in dimensions.items()
        }
        if any(values.get(key) != expected for key, expected in fixed.items()):
            continue
        candidate_gpu = values.get("gpu", "")
        candidate_version = _version(values.get("framework_version", ""))
        if candidate_gpu in _UNUSABLE or candidate_version is None:
            continue
        seen.add(canonical_id)
        version_affinity = _version_affinity(target_version, candidate_version)
        gpu_affinity = _gpu_affinity(target_gpu, candidate_gpu)
        ranked.append(
            _RankedIdentity(
                canonical_id=canonical_id,
                score=version_affinity + gpu_affinity,
                version_affinity=version_affinity,
                gpu_affinity=gpu_affinity,
                version_distance=_version_distance(target_version, candidate_version),
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
