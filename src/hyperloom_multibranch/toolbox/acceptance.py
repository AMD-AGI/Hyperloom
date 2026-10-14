# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance of a test request: the checks of design 4.17 that refuse a request before it queues.

Steps: the request validates (3.7, done by its constructor, plus the toolbox's own: no preset ``request_id``, a
``retest_of`` that names an accepted request); the parent recipe is read from the experiment repository; the delta
applies and the environment compiles; the measurement surface matches the canonical one. Compile and surface come from
hosted measurement (``hyperloom.measure.compile`` and ``surface``); they are reached through the two narrow seams below
so the toolbox does not depend on them until they land.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from hyperloom.contracts.measurement import TestRequest
from hyperloom.contracts.recipe import Recipe, RecipeDelta, RecipeError, apply
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import RequestRejected


class SurfaceMismatch(Exception):
    """The measurement surface of a request differs from the canonical one; ``str(error)`` names the field."""


class EnvironmentCompiler(Protocol):
    """Compiles the test environment from a recipe and a delta; raises ``RecipeError`` when it cannot (for example a
    patch that does not apply on the parent). Writes the delta's new files to the blob store as a side effect."""

    def compile(self, recipe: Recipe, delta: RecipeDelta) -> object: ...


class SurfaceGuard(Protocol):
    """Builds the measurement point of a request and compares its surface to the canonical one; raises
    :class:`SurfaceMismatch` when they differ."""

    def check(self, request: TestRequest, target: Recipe) -> None: ...


@dataclass(frozen=True)
class Prepared:
    """What acceptance learned: the compiled environment, handed to the executor with every attempt."""

    environment: object


class Acceptance:
    def __init__(
        self,
        repo: ExperimentRepo,
        compiler: EnvironmentCompiler,
        surface: SurfaceGuard,
        *,
        active_image_digest: str | None,
        is_accepted: Callable[[str], bool],
    ) -> None:
        self._repo = repo
        self._compiler = compiler
        self._surface = surface
        self._image = active_image_digest
        self._is_accepted = is_accepted

    async def prepare(self, request: TestRequest, *, replay: bool = False) -> Prepared:
        """Run the checks; ``replay`` re-prepares an already accepted request after a restart (it has its id)."""

        if not replay and request.request_id:
            raise RequestRejected("request_id is assigned by the toolbox; submit the request without one")
        if not replay and request.retest_of is not None and not self._is_accepted(request.retest_of):
            raise RequestRejected(f"retest_of {request.retest_of!r} names no accepted request", "unknown_retest")
        recipe = await self._read_parent(request)
        self._check_image(recipe)
        target = self._apply(recipe, request)
        environment = self._compile(recipe, request)
        self._check_surface(request, target)
        return Prepared(environment)

    async def _read_parent(self, request: TestRequest) -> Recipe:
        try:
            return await self._repo.read_recipe(request.parent_commit)
        except KeyError:
            raise RequestRejected(
                f"parent commit {request.parent_commit!r} is not in the experiment repository", "unknown_parent"
            ) from None

    def _check_image(self, recipe: Recipe) -> None:
        if self._image is None:
            return
        digests = {recipe.base_image_digest, recipe.runtime_snapshot or recipe.base_image_digest}
        if digests != {self._image}:
            raise RequestRejected(
                f"recipe image {sorted(digests)} is not the active framework image {self._image}", "base_image_mismatch"
            )

    @staticmethod
    def _apply(recipe: Recipe, request: TestRequest) -> Recipe:
        try:
            return apply(recipe, request.delta)
        except RecipeError as error:
            raise RequestRejected(
                f"the delta does not fit the parent recipe: {error}", "delta_not_applicable"
            ) from error

    def _compile(self, recipe: Recipe, request: TestRequest) -> object:
        try:
            return self._compiler.compile(recipe, request.delta)
        except RecipeError as error:
            raise RequestRejected(f"the environment does not compile: {error}", "delta_not_applicable") from error

    def _check_surface(self, request: TestRequest, target: Recipe) -> None:
        try:
            self._surface.check(request, target)
        except SurfaceMismatch as error:
            raise RequestRejected(
                f"measurement surface differs from the canonical one: {error}", "surface_mismatch"
            ) from error
