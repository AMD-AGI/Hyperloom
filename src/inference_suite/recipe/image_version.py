# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The framework version a recipe's image stands for, which the installed framework must reach (D6).

The version is read from the image tag (``lmsysorg/sglang:v0.5.22-rocm720-mi35x``). A tag that names none -- a
nightly, ``latest``, ``dev-*``, a placeholder, or no image at all -- stands for the framework's latest release when
the snapshot was taken, and the base records that release beside the version it gave.
"""

from __future__ import annotations

import re

from inference_suite.contracts.base_launch import ImageFacts, VersionBasis
from inference_suite.recipe.model import RecipeSourceError

_VERSION = r"\d+(?:\.\d+){1,2}(?:\.post\d+)?"
_TAG_VERSION = re.compile(rf"v({_VERSION})(?![.\d])")
_RELEASE_VERSION = re.compile(rf"v?({_VERSION})")

#: The image facts of a launch read from no recipe image: a user file or the default table.
NO_RECIPE_IMAGE = ImageFacts(run=None, recipe=None, framework_version=None, version_basis=VersionBasis.NOT_APPLICABLE)


def _tag(image: str) -> str:
    name = image.partition("@")[0]
    repository, _, tag = name.rpartition(":")
    return tag if repository and "/" not in tag else ""


def tag_version(image: str | None) -> str | None:
    """The framework version ``image``'s tag names (``:v0.5.22-...`` gives ``0.5.22``), else ``None``."""
    found = _TAG_VERSION.match(_tag(image)) if image else None
    return found.group(1) if found else None


def recipe_image_facts(image: str | None, latest_release: str | None) -> ImageFacts:
    """The D6 facts of a recipe launch run from ``image``, where ``latest_release`` is the framework's release tag
    when the snapshot was taken (``None`` if it recorded none)."""
    version = tag_version(image)
    if version is not None:
        return ImageFacts(run=None, recipe=image, framework_version=version, version_basis=VersionBasis.TAG)
    released = _RELEASE_VERSION.fullmatch(latest_release) if latest_release else None
    if released is None:
        raise RecipeSourceError(
            f"the recipe image {image!r} names no framework version, and the snapshot records no release version to "
            f"assume in its place (got {latest_release!r})"
        )
    return ImageFacts(
        run=None,
        recipe=image,
        framework_version=released.group(1),
        version_basis=VersionBasis.ASSUMED_LATEST,
        assumed_release=latest_release,
    )


__all__ = ["NO_RECIPE_IMAGE", "recipe_image_facts", "tag_version"]
