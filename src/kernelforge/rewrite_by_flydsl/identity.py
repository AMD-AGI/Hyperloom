"""Resolve a producer-owned ``kernel:`` recipe identity for a rewrite record.

The rewrite path used to address records by operator and framework alone and
carry the GPU as a filter applied after reading. Here the GPU is part of the
address, because a port validated on one architecture is not a candidate for
another and should not be fetched only to be discarded.

``framework_version`` is the dimension the rewrite path never tracked. It is
read from the installed distribution, so records stop being shared across
framework upgrades that change the very source the port was written against.
"""

from __future__ import annotations

import hashlib
import re
from importlib import metadata

from kernelforge.knowledge.experience_sink import (
    infer_source_owner_framework,
    resolve_operation,
)
from kernelforge.knowledge.implementation_identity import (
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
#: The version of a framework that is not there. A literal keeps the dimension
#: populated without pretending a version was observed.
NO_FRAMEWORK_VERSION = "none"
#: The framework is known but its distribution is not installed here.
UNKNOWN_VERSION = "unspecified"

_DISALLOWED = re.compile(r"[^a-z0-9._+-]+")
_LEADING = re.compile(r"^[^a-z0-9_]+")
#: A dimension may carry characters a session id may not, and may be far longer
#: than the 128 the store allows an id to be.
_UNSAFE_IN_SESSION_ID = re.compile(r"[^A-Za-z0-9._-]+")
_NAME_BUDGET = 48
_FINGERPRINT_LEN = 12


def segment(value: str, *, fallback: str) -> str:
    """Fold a free-form value into one identity dimension.

    Dimensions are lowercase ASCII and colon-free because they are the address:
    a value that cannot be rendered would otherwise silently file the record
    somewhere the next reader will not look.
    """
    folded = _DISALLOWED.sub("-", str(value or "").strip().lower())
    folded = _LEADING.sub("", folded).strip("-")
    if not folded:
        folded = fallback
    return folded.encode("ascii", "ignore").decode("ascii")[:256] or fallback


def framework_version(framework: str) -> str:
    """Read the installed version of the framework that owns the source."""
    name = str(framework or "").strip().lower()
    if not name or name == UNKNOWN_SEGMENT:
        return NO_FRAMEWORK_VERSION
    try:
        return segment(metadata.version(name), fallback=UNKNOWN_VERSION)
    except metadata.PackageNotFoundError:
        return UNKNOWN_VERSION


def session_id(canonical_id: str, kernel_name: str, port_digest: str) -> str:
    """Name one candidate under one identity.

    Artifact keys are partitioned by session id alone, so an id that repeated
    across identities would let two of them collide on any shared artifact
    path. The identity fingerprint is what keeps this id distinct per identity.
    The port digest is what keeps it stable, so re-recording the same port
    updates one candidate instead of accumulating one per run.

    The kernel name is here only to keep the id legible, and is budgeted rather
    than trusted: a dimension may be longer than a whole id is allowed to be.
    """
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
