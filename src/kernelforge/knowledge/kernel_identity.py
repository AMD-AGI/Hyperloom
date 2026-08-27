# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared identity for producer-owned kernel recipe records."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

DEFAULT_SCHEME_NAME = "kernel"
KERNEL_CANONICAL_DIMENSIONS = (
    "producer",
    "kernel_name",
    "framework",
    "framework_version",
    "backend",
    "gpu",
)
KERNEL_RECIPE_PRODUCERS = frozenset({"flydsl", "forge-loop", "fusion"})

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9._+-]*$")
_IDENTITY_SEGMENT_RE = re.compile(r"^[a-z0-9_][a-z0-9._+-]*$")


@dataclass(frozen=True)
class KernelRecipeIdentity:
    """Identity of one producer's recipe for a final kernel implementation.

    ``producer`` names the system that authored and owns the candidate stream;
    ``backend`` names the final implementation type (for example FlyDSL,
    Triton, or HIP). They are intentionally independent dimensions.
    """

    producer: str
    kernel_name: str
    gpu: str
    framework: str
    framework_version: str
    backend: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"KernelRecipeIdentity.{name} must be a non-empty string")
        if self.producer not in KERNEL_RECIPE_PRODUCERS:
            supported = ", ".join(sorted(KERNEL_RECIPE_PRODUCERS))
            raise ValueError(f"KernelRecipeIdentity.producer must be one of: {supported}; got {self.producer!r}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KernelRecipeIdentity":
        """Build an identity from the current producer-aware record shape."""
        return cls(
            producer=str(value.get("producer") or ""),
            kernel_name=str(value.get("kernel_name") or ""),
            gpu=str(value.get("gpu") or ""),
            framework=str(value.get("framework") or ""),
            framework_version=str(value.get("framework_version") or ""),
            backend=str(value.get("backend") or ""),
        )


def _validate_scheme(value: str) -> str:
    if not isinstance(value, str) or not _SCHEME_RE.fullmatch(value) or len(value.encode("ascii")) > 64:
        raise ValueError(
            "scheme_name/prefix must be 1-64 lowercase ASCII characters, "
            "start with a letter, and contain only letters, digits, '.', '_', '+', or '-'"
        )
    return value


def _resolve_scheme_name(
    *,
    scheme_name: str | None,
    prefix: str | None,
) -> str:
    if scheme_name is not None and prefix is not None and scheme_name != prefix:
        raise ValueError("scheme_name and prefix conflict; pass only one or use the same value")
    selected = scheme_name if scheme_name is not None else prefix if prefix is not None else DEFAULT_SCHEME_NAME
    return _validate_scheme(selected)


def _validate_identity_segment(name: str, value: str) -> str:
    if not isinstance(value, str) or not _IDENTITY_SEGMENT_RE.fullmatch(value) or len(value.encode("ascii")) > 256:
        raise ValueError(
            f"KernelRecipeIdentity.{name} must be 1-256 lowercase ASCII characters and "
            "contain only letters, digits, '.', '_', '+', or '-'"
        )
    return value


def kernel_recipe_canonical_id(
    identity: KernelRecipeIdentity,
    *,
    scheme_name: str | None = None,
    prefix: str | None = None,
) -> str:
    """Encode a recipe identity as a scheme plus six ordered dimensions."""
    scheme = _resolve_scheme_name(scheme_name=scheme_name, prefix=prefix)
    identity_values = asdict(identity)
    dimensions = [_validate_identity_segment(name, identity_values[name]) for name in KERNEL_CANONICAL_DIMENSIONS]
    return ":".join([scheme, *dimensions])


__all__ = [
    "DEFAULT_SCHEME_NAME",
    "KERNEL_CANONICAL_DIMENSIONS",
    "KERNEL_RECIPE_PRODUCERS",
    "KernelRecipeIdentity",
    "kernel_recipe_canonical_id",
]
