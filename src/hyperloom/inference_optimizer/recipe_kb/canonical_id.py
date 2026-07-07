# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Canonical id helpers for the local recipe-snapshot KB store.

Re-exports the 7-tuple builder + auto-detect helper from
:mod:`hyperloom.inference_optimizer.recipe_snapshot_constants` (Commit 1) so
the ``recipe_kb`` package presents a self-contained surface — callers
of ``recipe_kb`` should never need to know about
``recipe_snapshot_constants`` (it gets retired in Commit 3 alongside
the legacy NDJSON pending-queue plumbing).

Adds two local-store-specific helpers:

* :func:`cid_to_path_components` — decompose a canonical id back into
  its seven identity slugs (``model / hardware / framework_name /
  framework_version / precision / model_type / architectures``). The
  local store maps each slug to a directory level, so the round-trip
  ``recipe_canonical_id`` → ``cid_to_path_components`` →
  ``Path(*components)`` must be lossless.
* :func:`canonical_id_for_path` — given a path under the store root,
  derive the canonical id of the recipe that lives there. Used by
  :meth:`LocalRecipeStore.list_recent` / ``search`` so a tree-walk
  result can produce the same cid string the caller would have built
  from a 7-tuple.
"""

from __future__ import annotations

from pathlib import Path

from ..recipe_snapshot_constants import (
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


# ``inference:`` is the documented prefix for recipe-snapshot v2 ids
# (see boundary doc §0). It is a fixed literal here — bumping it is a
# compatibility break that has to be coordinated with the central
# kb-service, not silently forked by the local store.
CANONICAL_ID_PREFIX: str = "inference"


# Number of identity dimensions encoded in the canonical id. Eight
# colon-separated segments total: 1 prefix + 7 dimensions.
CANONICAL_ID_DIMENSIONS: int = 7


class InvalidCanonicalIdError(ValueError):
    """Raised when a string cannot be parsed as a 7-tuple canonical id.

    Carries the offending string and the parse-failure reason so
    callers (the local store's ``walk`` -> ``canonical_id_for_path``
    path, in particular) can log a structured warning rather than
    a bare exception when filesystem garbage drifts into the tree.
    """

    def __init__(self, raw: str, reason: str) -> None:
        """Build the error from the offending id and a reason.

        Args:
            raw (str): The string that failed to parse as a canonical
                id.
            reason (str): Human-readable parse-failure reason; folded
                into the message and stored on ``self.reason``.
        """
        super().__init__(f"invalid canonical_id {raw!r}: {reason}")
        self.raw = raw
        self.reason = reason


def cid_to_path_components(
    canonical_id: str,
) -> tuple[str, str, str, str, str, str, str]:
    """Decompose a canonical id into its seven identity slugs.

    The shape is enforced to exactly eight segments so a malformed id
    cannot quietly route writes to a sibling directory and silently
    shadow a real recipe. Legacy 6-segment ids (5-tuple era) are
    accepted and padded with default slugs for model_type and
    architectures to maintain backward compatibility.

    Returns the tuple in the order
    ``(model, hardware, framework_name, framework_version, precision,
    model_type, architectures)``,
    matching :func:`recipe_canonical_id`'s keyword order so callers
    can splat the result directly into a downstream call.

    Args:
        canonical_id (str): The canonical id to decompose.

    Returns:
        tuple[str, str, str, str, str, str, str]: The seven identity
            slugs.

    Raises:
        InvalidCanonicalIdError: If the id is empty, has a bad prefix,
            or contains an empty segment.
    """
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
    # Order: model, hardware, framework_name, model_type, architectures, framework_version, precision
    model, hardware, framework_name, model_type, architectures, framework_version, precision = parts[1:]
    if any(not seg for seg in parts[1:]):
        raise InvalidCanonicalIdError(
            raw,
            "empty segment(s) detected — every dimension must be non-empty",
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
    """Inverse of :func:`cid_to_path_components` — pass-through to
    :func:`recipe_canonical_id` for symmetry / discoverability.

    Kept here (rather than inlining the upstream helper at every
    call-site) so the recipe_kb package remains self-contained.

    Returns:
        str: The canonical id built from the seven slugs.
    """
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
    """Build the canonical id for the recipe directory at ``recipe_dir``.

    ``recipe_dir`` MUST be exactly seven levels below ``root`` —
    one level per dimension.

    The directory names ARE the canonical_id slugs (no extra slugging
    is applied — they were already slug-clean when
    :func:`recipe_canonical_id` produced them).

    Args:
        root (Path): Store root the recipe directory lives under.
        recipe_dir (Path): Directory of the recipe under ``root``.

    Returns:
        str: The canonical id of the recipe at ``recipe_dir``.

    Raises:
        InvalidCanonicalIdError: If ``recipe_dir`` is not under
            ``root`` or has an unexpected depth.
    """
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
    # Directory order matches canonical_id: model/hw/fw/model_type/arch/fwv/precision
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
