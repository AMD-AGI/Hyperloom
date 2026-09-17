"""Resolve a producer-owned ``kernel:`` recipe identity for a rewrite record."""

from __future__ import annotations

import hashlib
import re
from importlib import metadata

from kernelforge.knowledge.experience_sink import (
    infer_source_owner_framework,
    resolve_operation,
)
from kernelforge.knowledge.implementation_identity import (
    canonical_framework_version,
    implementation_signature,
    normalize_operator_name,
)
from kernelforge.knowledge.kernel_identity import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec

REWRITE_BACKEND = "flydsl"
REWRITE_PRODUCER = "flydsl"

#: Stands in for a dimension that could not be resolved, and is also what
#: ``detect_framework`` returns for a file owned by no framework package.
UNKNOWN_SEGMENT = "unknown"

_DISALLOWED = re.compile(r"[^a-z0-9._+-]+")
_LEADING = re.compile(r"^[^a-z0-9_]+")
#: A dimension may carry characters a session id may not, and may be far longer
#: than the 128 the store allows an id to be.
_UNSAFE_IN_SESSION_ID = re.compile(r"[^A-Za-z0-9._-]+")
_NAME_BUDGET = 48
_FINGERPRINT_LEN = 12


def segment(value: str, *, fallback: str) -> str:
    """Fold a free-form value into one identity dimension."""
    folded = _DISALLOWED.sub("-", str(value or "").strip().lower())
    folded = _LEADING.sub("", folded).strip("-")
    if not folded:
        folded = fallback
    return folded.encode("ascii", "ignore").decode("ascii")[:256] or fallback


def framework_version(framework: str) -> str:
    """Read the release of the framework that owns the source.

    A framework that is not there, one whose distribution is not installed, and
    one whose wheel was built on another machine each used to answer in their
    own words -- ``none``, ``unspecified``, ``0.24.0+rocm723`` -- so one kernel
    accumulated a page per answer. Every one of them resolves here to the
    release, or to the single word for not knowing it.
    """
    name = str(framework or "").strip().lower()
    try:
        installed = metadata.version(name) if name and name != UNKNOWN_SEGMENT else ""
    except metadata.PackageNotFoundError:
        installed = ""
    return segment(canonical_framework_version(installed), fallback=UNKNOWN_SEGMENT)


def session_id(canonical_id: str, kernel_name: str, port_digest: str) -> str:
    """Name one candidate under one identity."""
    name = _UNSAFE_IN_SESSION_ID.sub("-", str(kernel_name or "")).strip("-.")
    legible = name[:_NAME_BUDGET].strip("-.") or UNKNOWN_SEGMENT
    identity_fingerprint = hashlib.sha256(str(canonical_id or "").encode()).hexdigest()[:_FINGERPRINT_LEN]
    port = _UNSAFE_IN_SESSION_ID.sub("", str(port_digest or ""))[:_FINGERPRINT_LEN]
    return f"rewrite-{legible}-{identity_fingerprint}-{port}"


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
