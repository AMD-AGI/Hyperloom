# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve the ``kernel:`` identity a forge-loop run files its experience under.

Read and write must agree on every dimension or a warm start resolves to an
address no prior run ever wrote to, so both sides call this one function rather
than each deriving the identity themselves.

The GPU is part of the address rather than a filter applied after reading: a
solution validated on one card is not a candidate for another, and fetching it
only to discard it costs a round trip. It is addressed by hardware model
(``mi355x``) rather than compilation target (``gfx950``) because one target
spans several cards whose memory bandwidth and cache sizes differ, and a recipe
tuned against one of them is not a recommendation for the rest; the target is
kept alongside the metrics, where it describes how the solution was built rather
than where it applies. ``framework_version`` joins the address for a related
reason -- a framework upgrade can rewrite the very source a solution was
authored against.
"""

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
    """Return ``(identity, concrete_op, framework)`` for this run.

    ``concrete_op`` and the resolved framework come back alongside the identity
    because the callers need them for dtype extraction and the implementation
    signature, and resolving them twice risks the two answers drifting apart.

    ``producer`` defaults to the loop's own. A pipeline driving the loop as a
    subprocess overrides it so its records land in an index of their own.
    """
    # Imported here rather than at module scope: reaching the store's identity
    # helpers initializes its package, which imports this package's reader back,
    # and a top-level import would close that cycle. Both sides of the store
    # must fold a value into a dimension identically or they address different
    # records, so these come from the store rather than from a second copy.
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
        # A run whose kernel backend names no language still has to populate the
        # dimension: an empty one would not render as an address at all.
        backend=segment(backend, fallback=UNKNOWN_SEGMENT),
    )
    return identity, concrete_op, resolved_framework


__all__ = [
    "EXPERIENCE_ARTIFACT",
    "LOOP_PRODUCER",
    "PATCH_ARTIFACT",
    "resolve_loop_identity",
]
