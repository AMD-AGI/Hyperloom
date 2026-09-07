# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical id helpers for the local recipe-snapshot KB store."""

from __future__ import annotations

import re
from pathlib import Path

from hyperloom.inference_optimizer.recipe_snapshot_constants import (
    DEFAULT_ARCHITECTURES_SLUG,
    DEFAULT_FRAMEWORK_SLUG,
    DEFAULT_FRAMEWORK_VERSION_SLUG,
    DEFAULT_HARDWARE_SLUG,
    DEFAULT_MODEL_SLUG,
    DEFAULT_MODEL_TYPE_SLUG,
    DEFAULT_PRECISION_SLUG,
    canonical_labels,
    detect_framework_version,
    recipe_canonical_id,
)

# One canonical-id segment: a path component safe to join onto a store root.
SLUG_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


# Documented prefix for recipe-snapshot v2 ids; bumping it is a compatibility break coordinated with the central
# kb-service.
CANONICAL_ID_PREFIX: str = "inference"


# Number of identity dimensions encoded in the canonical id (8 colon-separated
# segments total: 1 prefix + 7 dimensions).
CANONICAL_ID_DIMENSIONS: int = 7


class InvalidCanonicalIdError(ValueError):
    """Raised when a string cannot be parsed as a 7-tuple canonical id."""

    def __init__(self, raw: str, reason: str) -> None:
        """Build the error from the offending id and a reason."""
        super().__init__(f"invalid canonical_id {raw!r}: {reason}")
        self.raw = raw
        self.reason = reason


def cid_to_path_components(
    canonical_id: str,
) -> tuple[str, str, str, str, str, str, str]:
    """Decompose a canonical id into its seven identity slugs."""
    raw = (canonical_id or "").strip()
    if not raw:
        raise InvalidCanonicalIdError(raw, "empty string")
    parts = raw.split(":")
    if len(parts) != 1 + CANONICAL_ID_DIMENSIONS:
        raise InvalidCanonicalIdError(
            raw,
            f"expected {1 + CANONICAL_ID_DIMENSIONS} colon-separated "
            f"segments (prefix + {CANONICAL_ID_DIMENSIONS} dimensions), "
            f"got {len(parts)}",
        )
    if parts[0] != CANONICAL_ID_PREFIX:
        raise InvalidCanonicalIdError(
            raw,
            f"prefix must be {CANONICAL_ID_PREFIX!r}, got {parts[0]!r}",
        )
    model, hardware, framework_name, model_type, architectures, framework_version, precision = parts[1:]
    if any(not seg for seg in parts[1:]):
        raise InvalidCanonicalIdError(
            raw,
            "empty segment(s) detected — every dimension must be non-empty",
        )
    for seg in parts[1:]:
        if not SLUG_PART_RE.fullmatch(seg):
            raise InvalidCanonicalIdError(
                raw,
                f"segment {seg!r} is not a safe path component",
            )
    return (model, hardware, framework_name, model_type, architectures, framework_version, precision)


def canonical_id_from_components(
    *,
    model: str,
    hardware: str,
    framework_name: str,
    model_type: str = "",
    architectures: "str | list[str]" = "",
    framework_version: str,
    precision: str,
) -> str:
    """Inverse of :func:`cid_to_path_components` — pass-through to"""
    return recipe_canonical_id(
        model=model,
        hardware=hardware,
        framework_name=framework_name,
        model_type=model_type,
        architectures=architectures,
        framework_version=framework_version,
        precision=precision,
    )


def canonical_id_for_path(*, root: Path, recipe_dir: Path) -> str:
    """Build the canonical id for the recipe directory at ``recipe_dir``."""
    try:
        rel = recipe_dir.relative_to(root)
    except ValueError as exc:
        raise InvalidCanonicalIdError(
            str(recipe_dir),
            f"path is not under store root {root!r}: {exc}",
        ) from exc
    parts = rel.parts
    if len(parts) != CANONICAL_ID_DIMENSIONS:
        raise InvalidCanonicalIdError(
            str(recipe_dir),
            f"expected {CANONICAL_ID_DIMENSIONS} levels under root, got {len(parts)}: {parts!r}",
        )
    model, hardware, framework_name, model_type, architectures, framework_version, precision = parts
    return canonical_id_from_components(
        model=model,
        hardware=hardware,
        framework_name=framework_name,
        model_type=model_type,
        architectures=architectures,
        framework_version=framework_version,
        precision=precision,
    )


__all__ = [
    "CANONICAL_ID_PREFIX",
    "CANONICAL_ID_DIMENSIONS",
    "DEFAULT_ARCHITECTURES_SLUG",
    "DEFAULT_FRAMEWORK_SLUG",
    "DEFAULT_FRAMEWORK_VERSION_SLUG",
    "DEFAULT_HARDWARE_SLUG",
    "DEFAULT_MODEL_SLUG",
    "DEFAULT_MODEL_TYPE_SLUG",
    "DEFAULT_PRECISION_SLUG",
    "InvalidCanonicalIdError",
    "canonical_id_for_path",
    "canonical_id_from_components",
    "canonical_labels",
    "cid_to_path_components",
    "detect_framework_version",
    "recipe_canonical_id",
]
