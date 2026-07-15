# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Local-first recipe-snapshot KB for the inference optimizer.

* Writes go LOCAL ONLY — never to the central kb-service. The store
  on disk is the single source of truth in degraded / offline mode
  and the authoritative source in healthy mode (the central service
  becomes a read-side cache, not a write target).
* Reads are dispatched at a higher layer (``recipe_kb.dispatcher.RecipeKB``);
  the local store implemented here is the read fallback.

The on-disk layout maps the canonical id
``inference:{model}:{hardware}:{framework}:{framework_version}:{precision}``
to a directory tree under :data:`LocalRecipeStore.root`. Each leaf directory
holds ``recipe.json`` (live), ``history/v{N}.json`` (archived prior
versions), ``attempts.ndjson`` (append-only attempts log), and ``.lock``
(flock target).

The wire shapes (``Recipe`` / ``Attempt`` dataclasses) mirror the central
kb-service v2 contract, so a dispatcher consumer sees identical dicts whether
they come from the local store or a central GET.
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
from .remote_client import RemoteRecipeClientError
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
    "RemoteRecipeClientError",
    "canonical_id_for_path",
    "canonical_id_from_components",
    "canonical_labels",
    "cid_to_path_components",
    "detect_framework_version",
    "recipe_canonical_id",
]
