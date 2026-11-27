# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The concurrencies a conc sweep measures and the order an arm runs them in, for the single-lineage and multi-branch
sweeps alike; imports nothing of either flow's runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inference_suite.contracts.base_launch import BaseLaunch, SourceKind

SCHEMA_VERSION = "2.1"
SOURCE_INFERENCEX = "inferencex"
SOURCE_OPERATOR = "operator"
#: Why the sweep of an InferenceX base frozen before it recorded a launch per concurrency measures nothing: every point
#: would run the session's own launch. Backfilling ``point.source_launches`` into the base lets it sweep.
NO_SOURCE_LAUNCHES = "no_source_launches"
#: Where a sweep's points take their launch from: the source's own at each concurrency, or the session's at every one.
LAUNCHES_SOURCE = "source"
LAUNCHES_SESSION = "session"


@dataclass(frozen=True)
class SweepPoints:
    """The concurrencies the sweep measures, ascending, and where they came from."""

    concs: tuple[int, ...]
    source: str


def sweep_points(base: BaseLaunch | None, conc: int, operator: Sequence[int]) -> SweepPoints | None:
    """The ``operator``'s points, else those of the InferenceX arm ``base`` was taken from, each with the session's
    ``conc``; ``None`` when neither names any."""
    if operator:
        return SweepPoints(tuple(sorted({*operator, conc})), SOURCE_OPERATOR)
    if base is not None and base.source.kind is SourceKind.INFERENCEX and base.point.source_concs:
        return SweepPoints(tuple(sorted({*base.point.source_concs, conc})), SOURCE_INFERENCEX)
    return None


def launches_missing(base: BaseLaunch | None) -> str | None:
    """:data:`NO_SOURCE_LAUNCHES` for an InferenceX ``base`` that holds no launch per concurrency, else ``None``."""
    if base is not None and base.source.kind is SourceKind.INFERENCEX and not base.point.source_launches:
        return NO_SOURCE_LAUNCHES
    return None


def point_launches(base: BaseLaunch | None) -> str:
    """Where the points of a sweep of ``base`` take their launch from."""
    return LAUNCHES_SOURCE if base is not None and base.point.source_launches else LAUNCHES_SESSION


def ladder(concs: Sequence[int], anchor: int) -> list[int]:
    """The order an arm runs ``concs`` in: the session's CONC first, then low to high."""
    return [anchor, *(conc for conc in concs if conc != anchor)] if anchor in concs else list(concs)


__all__ = [
    "LAUNCHES_SESSION",
    "LAUNCHES_SOURCE",
    "NO_SOURCE_LAUNCHES",
    "SCHEMA_VERSION",
    "SOURCE_INFERENCEX",
    "SOURCE_OPERATOR",
    "SweepPoints",
    "ladder",
    "launches_missing",
    "point_launches",
    "sweep_points",
]
