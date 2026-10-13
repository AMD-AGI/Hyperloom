# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The dispersion to expect before PRELUDE measures its own (Q1): what past sessions on the same task measured.

The archive holds, per past session, its baseline's dispersion over its measurements and the environment it measured
in: multi-branch sessions record the rule gate's baseline dispersion and their sandbox backends; single-lineage
sessions (backfilled) the dispersion of their baseline measurements, the noise band they graded under (empty on
sessions seeded before it was recorded), and one host. A session matches when it ran the same model, TP and GPU.

* Sessions that measured in the requested environment give the prior: the median of their dispersions.
* Single-host measurements are only a floor for hosted measurement: they raise any value below them.
* Without either, the noise bands old sessions graded under stand in.
* With nothing, the global statistics stand: the median (0.38%) or the p90 (3.30%) of every session, whichever the
  caller chooses. Pulse is not read yet.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from hyperloom_multibranch.rsi.archive import SearchTree

GLOBAL_MEDIAN_PCT = 0.38
GLOBAL_P90_PCT = 3.30
SINGLE_HOST = "local"
HOSTED = "hosted"
ANY_ENVIRONMENT = "any"


class PriorSource(str, Enum):
    ARCHIVE = "archive"
    NOISE_BAND = "noise_band"
    GLOBAL = "global"


class GlobalFallback(str, Enum):
    MEDIAN = "median"
    P90 = "p90"


_GLOBAL = {GlobalFallback.MEDIAN: GLOBAL_MEDIAN_PCT, GlobalFallback.P90: GLOBAL_P90_PCT}


@dataclass(frozen=True)
class DispersionPrior:
    """The dispersion to expect, where it comes from, and the measurement environment it describes.

    ``lower_bound_pct`` is the single-host dispersion of matching sessions, when any measured on one host.
    """

    dispersion_pct: float
    source: PriorSource
    environment: str
    samples: int = 0
    sessions: tuple[str, ...] = ()
    lower_bound_pct: float | None = None


def _same(left: str | None, right: str) -> bool:
    return left is not None and left.lower() == right.lower()


def matching(history: Sequence[SearchTree], *, model: str, tp: int, gpu: str) -> list[SearchTree]:
    """The archived sessions that ran ``model`` with ``tp`` on ``gpu``."""

    return [
        tree
        for tree in history
        if _same(tree.task.labels.get("model"), model)
        and _same(tree.task.labels.get("gpu"), gpu)
        and tree.task.sizes.get("tp") == float(tp)
    ]


def _environments(tree: SearchTree) -> set[str]:
    return set(tree.noise.environment.split("+")) if tree.noise.environment else set()


def dispersion_prior(
    history: Sequence[SearchTree],
    *,
    model: str,
    tp: int,
    gpu: str,
    environment: str = HOSTED,
    fallback: GlobalFallback = GlobalFallback.P90,
) -> DispersionPrior:
    """The dispersion to expect measuring ``model`` with ``tp`` on ``gpu`` in ``environment``."""

    matched = matching(history, model=model, tp=tp, gpu=gpu)
    measured = [tree for tree in matched if tree.noise.dispersion_pct is not None]
    here = [tree for tree in measured if environment in _environments(tree)]
    single_host = [tree.noise.dispersion_pct for tree in measured if _environments(tree) == {SINGLE_HOST}]
    floor = statistics.median(single_host) if single_host and environment != SINGLE_HOST else None
    bands = [tree for tree in matched if tree.noise.noise_band_pct is not None]
    if here:
        value = statistics.median(tree.noise.dispersion_pct for tree in here)
        prior = DispersionPrior(value, PriorSource.ARCHIVE, environment, sum(tree.noise.samples for tree in here))
        chosen = here
    elif bands:
        value = statistics.median(tree.noise.noise_band_pct for tree in bands)
        prior = DispersionPrior(value, PriorSource.NOISE_BAND, SINGLE_HOST)
        chosen = bands
    else:
        prior, chosen = DispersionPrior(_GLOBAL[fallback], PriorSource.GLOBAL, ANY_ENVIRONMENT), []
    return DispersionPrior(
        dispersion_pct=max(prior.dispersion_pct, floor or 0.0),
        source=prior.source,
        environment=prior.environment,
        samples=prior.samples,
        sessions=tuple(tree.session_id for tree in chosen),
        lower_bound_pct=floor,
    )


__all__ = [
    "ANY_ENVIRONMENT",
    "GLOBAL_MEDIAN_PCT",
    "GLOBAL_P90_PCT",
    "HOSTED",
    "SINGLE_HOST",
    "DispersionPrior",
    "GlobalFallback",
    "PriorSource",
    "dispersion_prior",
    "matching",
]
