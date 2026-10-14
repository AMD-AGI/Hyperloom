# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted-measurement planner: acceptance steps 2 to 4 of design 4.17 through ``hyperloom.measure``."""

from __future__ import annotations

import uuid
from pathlib import Path

from hyperloom.contracts.measurement import TestRequest
from hyperloom.contracts.recipe import Recipe
from hyperloom.measure.compile import EnvironmentCompiler
from hyperloom.measure.points import AcceptedMeasurement, CanonicalMeasurement, accept_measurement


class HostedPlanner:
    """Compiles with ``compiler`` and renders each request's points under ``points_root``, one directory per request,
    held to ``canonical``. The returned :class:`AcceptedMeasurement` travels with the queued test to the executor."""

    def __init__(self, compiler: EnvironmentCompiler, canonical: CanonicalMeasurement, points_root: Path) -> None:
        self._compiler = compiler
        self._canonical = canonical
        self._root = points_root

    def plan(self, request: TestRequest, parent: Recipe) -> AcceptedMeasurement:
        out_dir = self._root / uuid.uuid4().hex
        return accept_measurement(request, parent, self._compiler, self._canonical, out_dir)
