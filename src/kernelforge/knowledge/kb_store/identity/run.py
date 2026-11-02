# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve the ``kernel:`` identity a forge-loop run files its experience under."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kernelforge.knowledge.kb_store.identity.source_resolution import (
    KernelSources,
    detect_backend_language,
    infer_source_owner_framework,
    resolve_operation,
)
from kernelforge.knowledge.kb_store.identity.implementation import normalize_operator_name
from kernelforge.knowledge.kb_store.identity.kernel_recipe import KernelRecipeIdentity
from kernelforge.knowledge.kb_store.identity.normalization import (
    UNKNOWN_SEGMENT,
    framework_version,
    segment,
)

#: The system that authored the candidate stream. It partitions forge-loop's
#: records from a FlyDSL port's inside one identity scheme, and is deliberately
#: independent of ``backend``, which names the implementation type produced.
LOOP_PRODUCER = "forge-loop"

#: The cumulative diff travels as an artifact rather than inside the record, so
#: a reader can rank candidates without pulling a patch it may not want. Both
#: sides name it here so a write and a later read cannot disagree.
PATCH_ARTIFACT = "solution.patch"

#: The same run rendered for a reader rather than for a ranker. The record's
#: fields are what a program compares; this is what a person or an agent reads
#: when deciding whether a candidate is worth replaying, so it accompanies the
#: patch instead of being reconstructed from the record at every read.
EXPERIENCE_ARTIFACT = "experience.md"


@dataclass(frozen=True)
class LoopKernel:
    """The kernel a forge-loop run optimizes, the workspace it lives in, and the hints that name its operator."""

    kernel_path: str
    kernel_backend: str
    kernel_source: str = ""
    target_functions: list[str] | None = None
    source_files: list[str] | None = None
    framework: str = ""
    operator_name: str = ""
    workspace: str = ""


def resolve_loop_identity(
    kernel: LoopKernel,
    *,
    gpu_type: str,
    producer: str = "",
) -> tuple[KernelRecipeIdentity, str, str]:
    """Return ``(identity, concrete_op, framework)`` for this run."""
    concrete_op = resolve_operation(kernel.kernel_source, kernel.kernel_path, target_functions=kernel.target_functions)
    operator = normalize_operator_name(kernel.operator_name or concrete_op)
    backend = detect_backend_language(kernel.kernel_backend)
    resolved_framework = infer_source_owner_framework(
        KernelSources(kernel.kernel_path, kernel.kernel_source, kernel.source_files),
        target_functions=kernel.target_functions,
        framework_override=kernel.framework,
        concrete_operation=concrete_op,
    )
    identity = KernelRecipeIdentity(
        producer=producer.strip() or LOOP_PRODUCER,
        kernel_name=segment(operator, fallback=UNKNOWN_SEGMENT),
        gpu=segment(gpu_type, fallback=UNKNOWN_SEGMENT),
        framework=segment(resolved_framework, fallback=UNKNOWN_SEGMENT),
        framework_version=framework_version(resolved_framework),
        # A run whose kernel backend names no language still has to populate the dimension: an empty one would not
        # render as an address at all.
        backend=segment(backend, fallback=UNKNOWN_SEGMENT),
    )
    return identity, concrete_op, resolved_framework


def campaign_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """The run's conditions as its ``campaign_config.json`` records them; empty and unknown ones are left out.

    ``kernel`` is the name Kernel Forge files the run's own KB records under, which it derives from the operator.
    """

    filed, _operation, _framework = resolve_loop_identity(
        LoopKernel(
            kernel_path=str(config.get("kernel_path") or ""),
            kernel_backend=str(config.get("kernel_backend") or ""),
            target_functions=config.get("target_functions"),
            source_files=config.get("source_files"),
            framework=str(config.get("framework") or ""),
            operator_name=str(config.get("operator_name") or ""),
        ),
        gpu_type=str(config.get("gpu_type") or ""),
    )
    identity = {
        "operator": config.get("operator_name"),
        "kernel": filed.kernel_name,
        "framework": config.get("framework"),
        "backend": config.get("kernel_backend"),
        "gpu": config.get("gpu_type"),
        # The benchmark driver bytes. Two runs of one operator on different drivers must not share a repeat group.
        "driver_sha256": config.get("driver_sha256"),
    }
    return {key: value for key, value in identity.items() if value not in (None, "", UNKNOWN_SEGMENT)}


__all__ = [
    "EXPERIENCE_ARTIFACT",
    "LOOP_PRODUCER",
    "LoopKernel",
    "PATCH_ARTIFACT",
    "campaign_identity",
    "resolve_loop_identity",
]
