# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build operation-centric KernelForge knowledge-base identities."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

# Canonical identity schema version; independent of the KB page namespace.
IDENTITY_VERSION = 1
_PAGE_PREFIX = "kernelforge-exp/kernels"
_UNKNOWN_VALUES = {"", "n/a", "none", "null", "unknown", "unlinked", "unresolved"}
_PROJECT_ALIASES = {
    "aiter_meta": "aiter",
}
_GENERIC_REPOSITORY_NAMES = {
    "checkout",
    "code",
    "kernel",
    "kernels",
    "package",
    "repo",
    "repository",
    "source",
    "src",
    "worktree",
}
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Kernel kinds are resolver-produced implementation taxonomies. A
# ``<project>_<implementation>`` kind is therefore stronger evidence than any
# repository or package hint.
_KERNEL_KIND_IMPLEMENTATIONS = (
    ("triton_inductor_generated", ""),
    ("hip_cpp", "hip"),
    ("cuda_cpp", "cuda"),
    ("tilelang", "tilelang"),
    ("flydsl", "flydsl"),
    ("triton", "triton"),
    ("asm", "asm"),
    ("hip", "hip"),
    ("cuda", "cuda"),
    ("ck", "ck"),
    ("py", "py"),
)
_SOURCE_TYPE_TAXONOMY = {
    "ck": ("ck", "native"),
    "asm": ("asm", "native"),
    "triton": ("triton", "py"),
    "flydsl": ("flydsl", "py"),
    "tilelang": ("tilelang", "py"),
    "python": ("py", "py"),
    "py": ("py", "py"),
}
_IMPLEMENTATION_SOURCE_KINDS = {
    "asm": "native",
    "ck": "native",
    "cuda": "native",
    "hip": "native",
    "flydsl": "py",
    "py": "py",
    "tilelang": "py",
    "triton": "py",
}
_PACKAGE_KEYS = (
    "source_package",
    "package_name",
    "package",
    "repository",
    "repo_name",
    "kernel_repo_url",
    "repository_url",
    "repo_url",
)


def _slug_component(value: Any) -> str:
    """Return the page-safe representation of one identity component."""
    component = _SLUG_RE.sub("_", str(value or "").strip().lower()).strip("_")
    return component or "unknown"


def _known(value: str) -> bool:
    normalized = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", value.strip().lower())
    return normalized not in _UNKNOWN_VALUES


def _canonical_project(value: Any) -> str:
    """Normalize a package/repository label without retaining locator syntax."""
    if not isinstance(value, str):
        return ""
    raw = value.strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    locator = parsed.path if parsed.scheme or parsed.netloc else raw
    locator = locator.replace("\\", "/").rstrip("/")
    name = locator.rsplit("/", 1)[-1].removesuffix(".git")
    project = _slug_component(name)
    project = _PROJECT_ALIASES.get(project, project)
    if project in _UNKNOWN_VALUES or project in _GENERIC_REPOSITORY_NAMES:
        return ""
    return project


def _kernel_kind_identity(kernel_kind: Any) -> tuple[str, str]:
    """Resolve project/implementation from a resolver-produced kernel kind."""
    if not isinstance(kernel_kind, str):
        return "", ""
    kind = _slug_component(kernel_kind)
    if kind == "unknown":
        return "", ""
    for suffix, implementation in _KERNEL_KIND_IMPLEMENTATIONS:
        if kind == suffix:
            return "", implementation
        marker = f"_{suffix}"
        if kind.endswith(marker):
            project = kind[: -len(marker)].strip("_")
            project = _PROJECT_ALIASES.get(project, project)
            return (project, implementation) if project and implementation else ("", "")
    return "", ""


def _project_from_package_metadata(candidate: Mapping[str, Any]) -> str:
    for key in _PACKAGE_KEYS:
        project = _canonical_project(candidate.get(key))
        if project:
            return project

    repo_project = _canonical_project(candidate.get("kernel_repo"))
    if repo_project:
        return repo_project

    source_path = str(candidate.get("source_file") or "").replace("\\", "/")
    for marker in ("/site-packages/", "/dist-packages/"):
        if marker not in source_path:
            continue
        package = source_path.split(marker, 1)[1].split("/", 1)[0]
        project = _canonical_project(package)
        if project:
            return project
    return ""


def _implementation_from_source_type(
    source_type: Any,
    *,
    source_kind: str,
) -> str:
    taxonomy = _SOURCE_TYPE_TAXONOMY.get(str(source_type or "").strip().lower())
    if taxonomy is None:
        return ""
    implementation, expected_source_kind = taxonomy
    return implementation if expected_source_kind == source_kind else ""


def _linked_implementation(implementation: str, source_kind: str) -> bool:
    expected_source_kind = _IMPLEMENTATION_SOURCE_KINDS.get(implementation)
    return expected_source_kind == source_kind


def _canonical_operation(operation: str, kernel_project: str) -> str:
    """Remove only a leading namespace that resolves to the owning project."""
    for separator in ("::", "."):
        if separator not in operation:
            continue
        provider, remainder = operation.split(separator, 1)
        if remainder and _canonical_project(provider) == kernel_project:
            return remainder
    return operation


def _identity_digest(identity_fields: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(identity_fields),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()[:12]


def build_kernel_identities(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build a high-confidence shared identity, or return no identity."""
    task_group = candidate.get("task_group")
    if not isinstance(task_group, Mapping):
        return []
    operator_identity = task_group.get("operator_identity")
    if not isinstance(operator_identity, Mapping):
        return []

    raw_operation = operator_identity.get("operation")
    raw_source_kind = operator_identity.get("source_kind")
    if not isinstance(raw_operation, str) or not isinstance(raw_source_kind, str):
        return []
    operation = raw_operation.strip()
    source_kind = raw_source_kind.strip().lower()
    if not _known(operation) or source_kind not in {"native", "py"}:
        return []

    kernel_project, implementation = _kernel_kind_identity(candidate.get("kernel_kind"))
    if not kernel_project:
        kernel_project = _project_from_package_metadata(candidate)
    if not implementation:
        implementation = _implementation_from_source_type(
            candidate.get("source_type"),
            source_kind=source_kind,
        )

    if (
        not _known(kernel_project)
        or not _known(implementation)
        or not _linked_implementation(implementation, source_kind)
    ):
        return []
    operation = _canonical_operation(operation, kernel_project)
    if not _known(operation):
        return []

    identity_fields = {
        "identity_version": IDENTITY_VERSION,
        "kernel_project": kernel_project,
        "implementation": implementation,
        "source_kind": source_kind,
        "operation": operation,
    }
    digest = _identity_digest(identity_fields)
    page_slug = "/".join(
        (
            _PAGE_PREFIX,
            _slug_component(kernel_project),
            _slug_component(implementation),
            f"{_slug_component(operation)}--{digest}",
        )
    )
    return [
        {
            **identity_fields,
            "identity_digest": digest,
            "kernel_page_slug": page_slug,
            "confidence": "high",
        }
    ]


__all__ = ["IDENTITY_VERSION", "build_kernel_identities"]
