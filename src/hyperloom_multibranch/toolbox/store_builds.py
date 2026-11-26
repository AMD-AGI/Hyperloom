# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The queue's build driver over the native build store (:mod:`hyperloom.measure.jit.native`).

:class:`StoreBuildDriver` turns what a stack changes into the keys it needs, looks keys up in the store, estimates a
build from the store's own records and drives a build through
:func:`~hyperloom.measure.runner.native.service.ensure_build`, which holds the key's lock and signs the entry with the
service's key in this process. The compile is never run here: ``runner`` hands each build to the CPU build runner (a
sandbox with no key and no view of this process) and returns what it staged.

What a stack needs, ``plan``, is the compile's to say: the inputs of each build it planned at acceptance, whose hash is
the key the build runner builds (a pinned checkout's commit is part of its recipe version there,
:meth:`~hyperloom.measure.runner.native.recipes.NativeRecipe.build_version`), so the driver never keys a build again.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Callable, Iterator, Mapping

from hyperloom.contracts.measurement import TestRequest
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.record import BuildRecord
from hyperloom.measure.jit.native.store import BuildStore
from hyperloom.measure.jit.native.trust import BuildRefused, Signer
from hyperloom.measure.runner.native.recipes import NativeRecipe
from hyperloom.measure.runner.native.service import Build, Ensured, ensure_build

logger = logging.getLogger(__name__)

#: The inputs of each build an accepted request's stack needs, as acceptance planned them.
Plan = Callable[[TestRequest, object], tuple[BuildInputs, ...]]


class StoreBuildDriver:
    """:class:`~hyperloom_multibranch.toolbox.pending_builds.BuildDriver` over a :class:`BuildStore`.

    ``recipes`` is the recipe of each component a session can build. ``runner`` gives the build of one key; it must
    compile out of this process.
    """

    def __init__(
        self,
        store: BuildStore,
        signer: Signer,
        recipes: Mapping[str, NativeRecipe],
        plan: Plan,
        runner: Callable[[BuildInputs], Build],
    ) -> None:
        self._store, self._signer, self._recipes, self._plan, self._runner = store, signer, recipes, plan, runner

    def needs(self, request: TestRequest, environment: object) -> tuple[BuildInputs, ...]:
        """The inputs of each build the stack needs, as planned."""

        return self._plan(request, environment)

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
        return float(statistics.median(walls)) if walls else float(self._recipes[inputs.component].budget_sec)

    async def ensure(self, inputs: BuildInputs) -> Ensured:
        build = self._runner(inputs)
        return await asyncio.to_thread(ensure_build, self._store, inputs, build, self._signer)

    def _prior(self, inputs: BuildInputs) -> Iterator[BuildRecord]:
        """The verified ready records of ``inputs``' component built for its image."""

        return self._store.ready_records(inputs.component, inputs.image_digest)


__all__ = ["Plan", "StoreBuildDriver"]
