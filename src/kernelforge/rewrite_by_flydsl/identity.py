"""Resolve a producer-owned ``kernel:`` recipe identity for a rewrite record."""

from __future__ import annotations

from kernelforge.knowledge.kb_store.writer import (
    infer_source_owner_framework,
    resolve_operation,
)
from kernelforge.knowledge.kb_store.identity.implementation import (
    implementation_signature,
    normalize_operator_name,
)
from kernelforge.knowledge.kb_store.identity.kernel_recipe import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.knowledge.kb_store.identity.normalization import (
    UNKNOWN_SEGMENT,
    framework_version,
    segment,
)
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec

REWRITE_BACKEND = "flydsl"
REWRITE_PRODUCER = "flydsl"


def resolve_identity(
    spec: RewriteSpec,
    *,
    framework: str,
    gpu: str,
    source_text: str,
    producer: str = REWRITE_PRODUCER,
    backend: str = REWRITE_BACKEND,
) -> tuple[KernelRecipeIdentity, str, str, dict]:
    """Return the identity, its canonical id, and the implementation signature."""
    concrete_op = resolve_operation(
        source_text,
        spec.source_kernel,
        target_functions=spec.target_functions,
    )
    operator = normalize_operator_name(spec.op_name or concrete_op)
    resolved_framework = infer_source_owner_framework(
        kernel_path=spec.source_kernel,
        kernel_source=source_text,
        target_functions=spec.target_functions,
        source_files=None,
        framework_override=framework,
        concrete_operation=concrete_op,
    )
    signature, implementation = implementation_signature(
        workspace=spec.workspace,
        kernel_path=spec.source_kernel,
        source_files=None,
        framework=resolved_framework,
    )
    identity = KernelRecipeIdentity(
        producer=segment(producer, fallback=REWRITE_PRODUCER),
        kernel_name=segment(operator, fallback=UNKNOWN_SEGMENT),
        framework=segment(resolved_framework, fallback=UNKNOWN_SEGMENT),
        framework_version=framework_version(resolved_framework),
        backend=segment(backend, fallback=REWRITE_BACKEND),
        gpu=segment(gpu, fallback=UNKNOWN_SEGMENT),
    )
    return identity, kernel_recipe_canonical_id(identity), signature, implementation
