# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve the ``kernel:`` identity a forge-loop run files its experience under."""

from __future__ import annotations

from kernelforge.knowledge.experience_sink import (
    detect_backend_language,
    infer_source_owner_framework,
    resolve_operation,
)
from kernelforge.knowledge.implementation_identity import normalize_operator_name
from kernelforge.knowledge.kernel_identity import KernelRecipeIdentity

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


def resolve_loop_identity(
    *,
    kernel_path: str,
    kernel_source: str,
    kernel_backend: str,
    gpu_type: str,
    target_functions: list[str] | None = None,
    source_files: list[str] | None = None,
    framework: str = "",
    operator_name: str = "",
    producer: str = "",
) -> tuple[KernelRecipeIdentity, str, str]:
    """Return ``(identity, concrete_op, framework)`` for this run."""
    # Imported here rather than at module scope: reaching the store's identity helpers initializes its package, which
    # imports this package's reader back, and a top-level import would close that cycle.
    from kernelforge.rewrite_by_flydsl.identity import (
        UNKNOWN_SEGMENT,
        framework_version,
        segment,
    )

    concrete_op = resolve_operation(kernel_source, kernel_path, target_functions=target_functions)
    operator = normalize_operator_name(operator_name or concrete_op)
    backend = detect_backend_language(kernel_backend)
    resolved_framework = infer_source_owner_framework(
        kernel_path=kernel_path,
        kernel_source=kernel_source,
        target_functions=target_functions,
        source_files=source_files,
        framework_override=framework,
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


__all__ = [
    "EXPERIENCE_ARTIFACT",
    "LOOP_PRODUCER",
    "PATCH_ARTIFACT",
    "resolve_loop_identity",
]
