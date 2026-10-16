# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The dispersion prior: matching sessions' measurements, single-host floors, old noise bands, global statistics."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.rsi.archive import MeasurementNoise, tree_from_snapshot
from hyperloom_multibranch.notebook.rsi.backfill import tree_from_hyperloom_session
from hyperloom_multibranch.notebook.rsi.priors import (
    ANY_ENVIRONMENT,
    GLOBAL_MEDIAN_PCT,
    GLOBAL_P90_PCT,
    DispersionTask,
    GlobalFallback,
    PriorSource,
    dispersion_prior,
    matching,
)
from hyperloom_multibranch.notebook.rsi.tests.test_backfill import _session
from hyperloom_multibranch.notebook.rsi.tests.trees import TASK, tree
from hyperloom_multibranch.notebook.tests.sessions import METRIC

MODEL, GPU = TASK.labels["model"], TASK.labels["gpu"]
PRIOR_TASK = DispersionTask(MODEL, 8, GPU)


def _measured(session_id: str, dispersion: float | None, environment: str, band: float | None = None) -> object:
    return tree(session_id, noise=MeasurementNoise(dispersion, band, 3, environment))


def test_sessions_measured_in_the_same_environment_give_the_prior_and_one_host_is_a_floor() -> None:
    history = [
        _measured("hosted-a", 0.6, "hosted"),
        _measured("hosted-b", 1.0, "hosted+local"),
        _measured("one-host", 0.3, "local"),
        dataclasses.replace(
            _measured("other-model", 9.0, "hosted"), task=dataclasses.replace(TASK, labels={"model": "x"})
        ),
    ]

    prior = dispersion_prior(history, DispersionTask(MODEL, 8, GPU.lower()))

    assert (prior.dispersion_pct, prior.source, prior.environment) == (
        pytest.approx(0.8),
        PriorSource.ARCHIVE,
        "hosted",
    )
    assert (prior.samples, prior.sessions, prior.lower_bound_pct) == (6, ("hosted-a", "hosted-b"), 0.3)


def test_one_host_alone_only_raises_the_global_statistics() -> None:
    history = [_measured("one-host", 0.5, "local")]

    p90 = dispersion_prior(history, PRIOR_TASK)
    median = dispersion_prior(history, PRIOR_TASK, fallback=GlobalFallback.MEDIAN)
    on_one_host = dispersion_prior(history, PRIOR_TASK, environment="local")

    assert (p90.dispersion_pct, p90.source, p90.environment) == (GLOBAL_P90_PCT, PriorSource.GLOBAL, ANY_ENVIRONMENT)
    assert (median.dispersion_pct, median.lower_bound_pct) == (0.5, 0.5)
    assert (on_one_host.dispersion_pct, on_one_host.source, on_one_host.lower_bound_pct) == (
        0.5,
        PriorSource.ARCHIVE,
        None,
    )


def test_old_noise_bands_stand_in_and_nothing_falls_back_to_the_global_median() -> None:
    banded = dispersion_prior([_measured("old", None, "local", band=2.0)], PRIOR_TASK)
    nothing = dispersion_prior([], PRIOR_TASK, fallback=GlobalFallback.MEDIAN)

    assert (banded.dispersion_pct, banded.source, banded.sessions) == (2.0, PriorSource.NOISE_BAND, ("old",))
    assert (nothing.dispersion_pct, nothing.samples, nothing.sessions) == (GLOBAL_MEDIAN_PCT, 0, ())


def test_both_kinds_of_archived_session_feed_the_prior(snapshot: NotebookSnapshot, tmp_path: Path) -> None:
    ours = tree_from_snapshot(snapshot, METRIC)
    old = tree_from_hyperloom_session(_session(tmp_path / "old-1"))

    assert [item.session_id for item in matching([ours, old], model=MODEL, tp=8, gpu=GPU)] == ["sess-1", "old-1"]
    prior = dispersion_prior([ours, old], PRIOR_TASK, environment="local")
    assert prior.dispersion_pct == pytest.approx((0.8 + old.noise.dispersion_pct) / 2)
    assert prior.sessions == ("sess-1", "old-1")
