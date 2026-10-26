# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted-measurement planner: acceptance steps 2 to 4 of design 4.17 through ``hyperloom.measure``.

Every point it plans, a baseline's and a candidate's alike, launches its server with aiter's hit logging on, so a
result can say whether a kernel table was read (``table_hits``) and a parent and its child run under the same
conditions. The point is where the server's environment is made, on the local backend and the hosted one, which run
the same points.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

from hyperloom.common.outside_facts import AITER_LOG_TUNED_CONFIG_ENV
from hyperloom.contracts.measurement import TestRequest
from hyperloom.contracts.recipe import Recipe, RecipeDelta, RecipeError
from hyperloom.measure.compile import CompiledEnvironment, EnvironmentCompiler
from hyperloom.measure.jit.native.needs import NativePlan
from hyperloom.measure.points import AcceptedMeasurement, CanonicalMeasurement, accept_measurement

#: What every measured point sets ``AITER_LOG_TUNED_CONFIG`` to. aiter reads the variable as an integer and logs the hits
#: of its tuned tables for any value but 0; one value is what makes two points comparable.
HIT_LOGGING_ON = "1"


class HitLoggingConflict(RecipeError):
    """A recipe launches with ``AITER_LOG_TUNED_CONFIG`` at a value other than the one every measured point has."""


class _HitLogging:
    """What ``accept_measurement`` compiles with: ``compiler``'s environments, each asking its server for aiter's hits.

    The variable is carried by the environment, which the points are rendered with next to the recipe's own launch
    variables, and is not put into the recipe: the commits and the deltas of the notebook are as the specialists wrote
    them. ``accept_measurement`` calls ``compile`` and nothing else of its compiler.
    """

    def __init__(self, compiler: EnvironmentCompiler) -> None:
        self._compiler = compiler

    @property
    def compiler(self) -> EnvironmentCompiler:
        return self._compiler

    def compile(self, recipe: Recipe, action: RecipeDelta) -> CompiledEnvironment:
        environment = self._compiler.compile(recipe, action)
        own = environment.recipe.launch.envs.get(AITER_LOG_TUNED_CONFIG_ENV)
        if own is not None and own != HIT_LOGGING_ON:
            raise HitLoggingConflict(
                f"the recipe launches with {AITER_LOG_TUNED_CONFIG_ENV}={own}, but every measured point runs with "
                f"{AITER_LOG_TUNED_CONFIG_ENV}={HIT_LOGGING_ON}, so that a parent and its child are measured the same way "
                f"and a result can say whether a kernel table was read. Remove the variable from the recipe, or set it "
                f"to {HIT_LOGGING_ON}"
            )
        return replace(environment, envs={**environment.envs, AITER_LOG_TUNED_CONFIG_ENV: HIT_LOGGING_ON})


class HostedPlanner:
    """Compiles with ``compiler`` and renders each request's points under ``points_root``, one directory per request,
    held to ``canonical``. The returned :class:`AcceptedMeasurement` travels with the queued test to the executor."""

    def __init__(self, compiler: EnvironmentCompiler, canonical: CanonicalMeasurement, points_root: Path) -> None:
        self._compiler = _HitLogging(compiler)
        self._canonical = canonical
        self._root = points_root

    def with_native(self, plan: NativePlan) -> HostedPlanner:
        """This planner compiling with ``plan``, so a stack that changes native sources needs a build instead of being
        refused; the compile cache starts empty, since a native stack compiles differently."""
        return HostedPlanner(replace(self._compiler.compiler, native=plan), self._canonical, self._root)

    def plan(self, request: TestRequest, parent: Recipe) -> AcceptedMeasurement:
        out_dir = self._root / uuid.uuid4().hex
        return accept_measurement(request, parent, self._compiler, self._canonical, out_dir)  # type: ignore[arg-type]
