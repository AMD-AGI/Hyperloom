# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The key of a native build: the hash of exactly the inputs that decide what the compiler makes.

The inputs are the image the outputs are laid over, the builder image that compiles them, the component, the GPU
architecture, the recipe version and ``native_digest``. ``native_digest`` hashes the files a patch stack leaves different in
the native section (path, the hash of the base file, the blob, the mode), the way the aiter cache hashes its source tree
(:func:`hyperloom.measure.jit.key.tree_identity`), not the patch text: two stacks that yield the same sources share one
build.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hyperloom.measure.docs import DocumentError, Reader, canonical_json
from hyperloom.measure.jit.key import tree_identity
from hyperloom.measure.jit.store import file_sha256

__all__ = ["BUILD_SCHEMA", "BuildInputs", "native_digest"]

BUILD_SCHEMA = "hyperloom.native-build/v1"
#: A component names a directory of the store, so it is one plain path segment.
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_HEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class BuildInputs:
    """Everything a native build depends on; :attr:`key` is their hash."""

    image_digest: str
    builder_image_digest: str
    component: str
    gfx_arch: str
    native_digest: str
    recipe_version: str

    def __post_init__(self) -> None:
        if not _COMPONENT.fullmatch(self.component):
            raise ValueError(f"{self.component!r} is not a component name")

    def document(self) -> dict[str, str]:
        return {
            "schema": BUILD_SCHEMA,
            "image": self.image_digest,
            "builder_image": self.builder_image_digest,
            "component": self.component,
            "gfx": self.gfx_arch,
            "native_digest": self.native_digest,
            "recipe_version": self.recipe_version,
        }

    @property
    def key(self) -> str:
        return hashlib.sha256(canonical_json(self.document()).encode("utf-8")).hexdigest()

    @classmethod
    def from_document(cls, value: Any, path: str) -> BuildInputs:
        reader = Reader(value, path)
        if reader.text("schema") != BUILD_SCHEMA:
            raise DocumentError(f"{reader.at('schema')}: expected {BUILD_SCHEMA}")
        try:
            inputs = cls(
                image_digest=str(reader.text("image", pattern=_DIGEST)),
                builder_image_digest=str(reader.text("builder_image", pattern=_DIGEST)),
                component=str(reader.text("component", pattern=_COMPONENT)),
                gfx_arch=str(reader.text("gfx")),
                native_digest=str(reader.text("native_digest", pattern=_HEX)),
                recipe_version=str(reader.text("recipe_version")),
            )
        except ValueError as exc:
            raise DocumentError(f"{path}: {exc}") from None
        reader.done()
        return inputs


class ChangedFile(Protocol):
    """One file a patch stack leaves different (:class:`hyperloom.measure.jit.native.needs.EnvEntry`)."""

    path: str
    blob: str | None
    mode: str | None


def native_digest(entries: Iterable[ChangedFile], base: Path) -> str:
    """The identity of the native sources ``entries`` make of the tree ``base``: per changed file its path, the hash of
    the file it replaces (empty when it adds one), its blob and its mode (both empty for a removal)."""
    files = (
        (entry.path, file_sha256(base / entry.path) or "", entry.blob or "", entry.mode or "") for entry in entries
    )
    return tree_identity(files, None, ())
