# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local recipe-snapshot KB for the inference optimizer.

Reads and writes through :class:`RecipeKB` use only
:class:`LocalRecipeStore`. Remote Recipe mode is implemented separately by the
KB Store CLOSE writer under :mod:`hyperloom.orchestrator.knowledge.remote_recipe`.

The on-disk layout maps the canonical id
``inference:{model}:{hardware}:{framework_name}:{model_type}:{architectures}:{framework_version}:{precision}``
to a directory tree under :data:`LocalRecipeStore.root`. Each leaf directory
holds ``recipe.json`` (live), ``history/v{N}.json`` (archived prior
versions), ``attempts.ndjson`` (append-only attempts log), and ``.lock``
(flock target).

The ``Recipe`` / ``Attempt`` dataclasses define the local on-disk contract.
"""

from __future__ import annotations

from .canonical_id import (
    CANONICAL_ID_DIMENSIONS,
    CANONICAL_ID_PREFIX,
    DEFAULT_FRAMEWORK_SLUG,
    DEFAULT_FRAMEWORK_VERSION_SLUG,
    DEFAULT_HARDWARE_SLUG,
    DEFAULT_MODEL_SLUG,
    DEFAULT_PRECISION_SLUG,
    InvalidCanonicalIdError,
    canonical_id_for_path,
    canonical_id_from_components,
    canonical_labels,
    cid_to_path_components,
    detect_framework_version,
    recipe_canonical_id,
)
from .dispatcher import RecipeKB
from .local_store import (
    ATTEMPTS_FILENAME,
    HISTORY_DIRNAME,
    LOCK_FILENAME,
    LocalRecipeStore,
    LocalRecipeStoreError,
    RECIPE_FILENAME,
)
from .schema import Attempt, Recipe


__all__ = [
    "ATTEMPTS_FILENAME",
    "Attempt",
    "CANONICAL_ID_DIMENSIONS",
    "CANONICAL_ID_PREFIX",
    "DEFAULT_FRAMEWORK_SLUG",
    "DEFAULT_FRAMEWORK_VERSION_SLUG",
    "DEFAULT_HARDWARE_SLUG",
    "DEFAULT_MODEL_SLUG",
    "DEFAULT_PRECISION_SLUG",
    "HISTORY_DIRNAME",
    "InvalidCanonicalIdError",
    "LOCK_FILENAME",
    "LocalRecipeStore",
    "LocalRecipeStoreError",
    "RECIPE_FILENAME",
    "Recipe",
    "RecipeKB",
    "canonical_id_for_path",
    "canonical_id_from_components",
    "canonical_labels",
    "cid_to_path_components",
    "detect_framework_version",
    "recipe_canonical_id",
]
