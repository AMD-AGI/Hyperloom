# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The concurrencies a conc sweep measures and the order an arm runs them in, for the single-lineage and multi-branch
sweeps alike; imports nothing of either flow's runtime."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from inference_suite.contracts.base_launch import BaseLaunch, SourceKind

log = logging.getLogger(__name__)

SCHEMA_VERSION = "2.0"
SOURCE_INFERENCEX = "inferencex"
SOURCE_OPERATOR = "operator"


@dataclass(frozen=True)
class SweepPoints:
    """The concurrencies the sweep measures, ascending, and where they came from."""

    concs: tuple[int, ...]
    source: str


def sweep_points(base: BaseLaunch | None, conc: int, operator: Sequence[int]) -> SweepPoints | None:
    """The points of the InferenceX arm ``base`` was taken from, else the ``operator``'s, each with the session's
    ``conc``; ``None`` when neither names any."""
    if base is not None and base.source.kind is SourceKind.INFERENCEX and base.point.source_concs:
        if operator:
            log.warning("conc_sweep: --conc-sweep-concs %s ignored; the InferenceX recipe names the points", operator)
        return SweepPoints(tuple(sorted({*base.point.source_concs, conc})), SOURCE_INFERENCEX)
    if operator:
        return SweepPoints(tuple(sorted({*operator, conc})), SOURCE_OPERATOR)
    return None


def ladder(concs: Sequence[int], anchor: int) -> list[int]:
    """The order an arm runs ``concs`` in: the session's CONC first, then low to high."""
    return [anchor, *(conc for conc in concs if conc != anchor)] if anchor in concs else list(concs)


__all__ = ["SCHEMA_VERSION", "SOURCE_INFERENCEX", "SOURCE_OPERATOR", "SweepPoints", "ladder", "sweep_points"]
