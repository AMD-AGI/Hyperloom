# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The queue's build driver over the native build store (:mod:`hyperloom.measure.jit.native`).

:class:`StoreBuildDriver` turns what a stack changes into the keys it needs, looks keys up in the store, estimates a
build from the store's own records and drives a build through
:func:`~hyperloom.measure.runner.native.service.ensure_build`, which holds the key's lock and signs the entry with the
service's key in this process. The compile is never run here: ``runner`` hands each build to the CPU build runner (a
sandbox with no key and no view of this process) and returns what it staged.

What a stack changes, ``plan``, is the runtime's to say: per component, the
:func:`~hyperloom.measure.jit.native.key.native_digest` of the native files the stack leaves different. The rest of a
key is the site's: the image, the builder image, the GPU architecture and the component's recipe version.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.contracts.measurement import TestRequest
from hyperloom.measure.docs import DocumentError
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.record import READY, RECORD_FILES, BuildRecord
from hyperloom.measure.jit.native.store import BuildStore
from hyperloom.measure.jit.native.trust import BuildRefused, Signer
from hyperloom.measure.runner.native.recipes import NativeRecipe
from hyperloom.measure.runner.native.service import Build, Ensured, ensure_build

logger = logging.getLogger(__name__)

#: A record larger than this is not one the service wrote; it is skipped when estimating.
_MAX_RECORD_BYTES = 1024 * 1024
_MANIFEST = RECORD_FILES[READY][0]

#: The native digest of each component an accepted request's stack changes, given what acceptance prepared.
Plan = Callable[[TestRequest, object], Mapping[str, str]]


@dataclass(frozen=True)
class BuildSite:
    """What every key of a session shares: the image the outputs lay over, the builder image that compiles them, the
    GPU architecture, and the recipe of each component that can be built."""

    image_digest: str
    builder_image_digest: str
    gfx_arch: str
    recipes: Mapping[str, NativeRecipe]


class StoreBuildDriver:
    """:class:`~hyperloom_multibranch.toolbox.pending_builds.BuildDriver` over a :class:`BuildStore`.

    ``runner`` gives the build of one key; it must compile out of this process.
    """

    def __init__(
        self, store: BuildStore, signer: Signer, site: BuildSite, plan: Plan, runner: Callable[[BuildInputs], Build]
    ) -> None:
        self._store, self._signer, self._site, self._plan, self._runner = store, signer, site, plan, runner

    def needs(self, request: TestRequest, environment: object) -> tuple[BuildInputs, ...]:
        """The key of each component the stack changes; a component without a recipe is a ``KeyError``."""

        site = self._site
        return tuple(
            BuildInputs(
                site.image_digest,
                site.builder_image_digest,
                component,
                site.gfx_arch,
                digest,
                site.recipes[component].version,
            )
            for component, digest in self._plan(request, environment).items()
        )

    def lookup(self, inputs: BuildInputs) -> BuildRecord | None:
        """The verified entry of ``inputs``; ``None`` also for an entry that fails a check, which ``ensure`` then
        refuses by name."""

        try:
            return self._store.read(inputs)
        except BuildRefused as exc:
            logger.warning("the stored build of %s %s is refused: %s", inputs.component, inputs.key, exc)
            return None

    def eta_seconds(self, inputs: BuildInputs) -> float:
        """The median wall time of the ready builds of ``inputs``' component on its image, else the recipe budget."""

        walls = [record.wall_seconds for record in self._prior(inputs)]
        return float(statistics.median(walls)) if walls else float(self._site.recipes[inputs.component].budget_sec)

    async def ensure(self, inputs: BuildInputs) -> Ensured:
        build = self._runner(inputs)
        return await asyncio.to_thread(ensure_build, self._store, inputs, build, self._signer)

    def _prior(self, inputs: BuildInputs) -> Iterator[BuildRecord]:
        """The verified ready records of ``inputs``' component built for its image."""

        component_dir = self._store.entry_dir(inputs).parent
        if not component_dir.is_dir():
            return
        for entry in component_dir.iterdir():
            record = self._verified(entry / _MANIFEST)
            if record is not None and record.inputs.image_digest == inputs.image_digest:
                yield record

    def _verified(self, manifest: Path) -> BuildRecord | None:
        try:
            if not manifest.is_file() or manifest.stat().st_size > _MAX_RECORD_BYTES:
                return None
            return self._store.read(BuildRecord.parse(manifest.read_bytes()).inputs)
        except (OSError, DocumentError, BuildRefused):
            return None  # an entry that cannot be read or proven tells nothing of how long a build takes


__all__ = ["BuildSite", "Plan", "StoreBuildDriver"]
