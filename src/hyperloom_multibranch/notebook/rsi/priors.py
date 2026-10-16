# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The dispersion to expect before PRELUDE measures its own (Q1): what past sessions on the same task measured.

The archive holds, per past session, its baseline's dispersion over its measurements and the environment it measured
in: multi-branch sessions record the rule gate's baseline dispersion and where they measured (one host, or the
sandbox backends they used); single-lineage sessions (backfilled) the dispersion of their baseline measurements, the
noise band they graded under (empty on sessions seeded before it was recorded), and one host. A session matches when
it ran the same model, TP and GPU.

* Sessions that measured in the requested environment give the prior: the median of their dispersions.
* Single-host measurements are only a floor for hosted measurement: they raise any value below them.
* Without such sessions, for hosted measurement, the shadow runs' hosted dispersion stands when the caller passes it
  (``shadow``, from yihan's consistency check, :mod:`rsi.shadow`); their hosted-minus-local offset comes with any prior.
* Without either, the noise bands old sessions graded under stand in.
* With nothing, the global statistics stand: the median (0.38%) or the p90 (3.30%) of every session, whichever the
  caller chooses. Pulse is not read yet.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from hyperloom_multibranch.notebook.rsi.archive import SINGLE_HOST, SearchTree
from hyperloom_multibranch.notebook.rsi.shadow import ShadowNoise

GLOBAL_MEDIAN_PCT = 0.38
GLOBAL_P90_PCT = 3.30
HOSTED = "hosted"
ANY_ENVIRONMENT = "any"


class PriorSource(str, Enum):
    ARCHIVE = "archive"
    SHADOW = "shadow"
    NOISE_BAND = "noise_band"
    GLOBAL = "global"


class GlobalFallback(str, Enum):
    MEDIAN = "median"
    P90 = "p90"


_GLOBAL = {GlobalFallback.MEDIAN: GLOBAL_MEDIAN_PCT, GlobalFallback.P90: GLOBAL_P90_PCT}


@dataclass(frozen=True)
class DispersionPrior:
    """The dispersion to expect, where it comes from, and the measurement environment it describes.

    ``lower_bound_pct`` is the single-host dispersion of matching sessions, when any measured on one host;
    ``offset_pct`` how far hosted reads from local in the shadow runs, when the caller passed them. From the shadow
    runs, ``samples`` counts their tests.
    """

    dispersion_pct: float
    source: PriorSource
    environment: str
    samples: int = 0
    sessions: tuple[str, ...] = ()
    lower_bound_pct: float | None = None
    offset_pct: float | None = None


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


@dataclass(frozen=True)
class DispersionTask:
    """The task a prior is for: ``model`` with ``tp`` on ``gpu``."""

    model: str
    tp: int
    gpu: str


def dispersion_prior(
    history: Sequence[SearchTree],
    task: DispersionTask,
    *,
    environment: str = HOSTED,
    fallback: GlobalFallback = GlobalFallback.P90,
    shadow: ShadowNoise | None = None,
) -> DispersionPrior:
    """The dispersion to expect measuring ``task`` in ``environment``.

    ``shadow`` is the shadow runs' noise for this task (:func:`rsi.shadow.shadow_noise`), none by default.
    """

    matched = matching(history, model=task.model, tp=task.tp, gpu=task.gpu)
    measured = [tree for tree in matched if tree.noise.dispersion_pct is not None]
    here = [tree for tree in measured if environment in _environments(tree)]
    single_host = [tree.noise.dispersion_pct for tree in measured if _environments(tree) == {SINGLE_HOST}]
    floor = statistics.median(single_host) if single_host and environment != SINGLE_HOST else None
    bands = [tree for tree in matched if tree.noise.noise_band_pct is not None]
    hosted_shadow = shadow.hosted_cv_pct if shadow is not None and environment == HOSTED else None
    if here:
        value = statistics.median(tree.noise.dispersion_pct for tree in here)
        prior = DispersionPrior(value, PriorSource.ARCHIVE, environment, sum(tree.noise.samples for tree in here))
        chosen = here
    elif shadow is not None and hosted_shadow is not None:
        prior, chosen = DispersionPrior(hosted_shadow, PriorSource.SHADOW, HOSTED, len(shadow.test_ids)), []
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
        offset_pct=shadow.offset_pct if shadow is not None else None,
    )


__all__ = [
    "ANY_ENVIRONMENT",
    "GLOBAL_MEDIAN_PCT",
    "GLOBAL_P90_PCT",
    "HOSTED",
    "SINGLE_HOST",
    "DispersionPrior",
    "DispersionTask",
    "GlobalFallback",
    "PriorSource",
    "dispersion_prior",
    "matching",
]
