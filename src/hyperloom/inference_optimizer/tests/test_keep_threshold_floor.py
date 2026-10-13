# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The grid's KEEP floor tracks the workload's measurement noise, not one global constant."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors._grid_base import (
    DEFAULT_KEEP_THRESHOLD_PCT,
    KEEP_THRESHOLD_PCT_ENV,
    MLPERF_KEEP_THRESHOLD_PCT,
    default_keep_threshold_pct,
)

_BACKEND_ENV = "HYPERLOOM_AGENTIC_BACKEND"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(KEEP_THRESHOLD_PCT_ENV, raising=False)
    monkeypatch.delenv(_BACKEND_ENV, raising=False)


def test_the_synthetic_workload_keeps_its_one_percent_floor():
    assert default_keep_threshold_pct() == pytest.approx(DEFAULT_KEEP_THRESHOLD_PCT)


def test_the_mlperf_backend_raises_the_floor_above_its_own_drift(monkeypatch):
    """Two identical 150-trajectory baselines differed by 7.2% on the axis these sessions grade."""
    monkeypatch.setenv(_BACKEND_ENV, "mlperf")
    assert default_keep_threshold_pct() == pytest.approx(MLPERF_KEEP_THRESHOLD_PCT)
    assert MLPERF_KEEP_THRESHOLD_PCT > 7.2


def test_the_env_overrides_either_floor(monkeypatch):
    monkeypatch.setenv(KEEP_THRESHOLD_PCT_ENV, "4.5")
    assert default_keep_threshold_pct() == pytest.approx(4.5)
    monkeypatch.setenv(_BACKEND_ENV, "mlperf")
    assert default_keep_threshold_pct() == pytest.approx(4.5)


@pytest.mark.parametrize("raw", ["0", "0.5", "-3"])
def test_the_override_cannot_fall_below_the_generic_noise_floor(monkeypatch, raw):
    """A gate under the generic floor cannot separate a win from a re-run."""
    monkeypatch.setenv(KEEP_THRESHOLD_PCT_ENV, raw)
    assert default_keep_threshold_pct() == pytest.approx(DEFAULT_KEEP_THRESHOLD_PCT)


def test_the_explore_executor_resolves_the_floor_rather_than_binding_it_at_import(monkeypatch):
    """A module-level default binds before the backend is known, so the executor must resolve per instance."""
    from hyperloom.orchestrator.actions.executors.explore import ExploreExecutor

    monkeypatch.setenv(_BACKEND_ENV, "mlperf")
    assert ExploreExecutor().keep_threshold_pct == pytest.approx(MLPERF_KEEP_THRESHOLD_PCT)
    monkeypatch.delenv(_BACKEND_ENV)
    assert ExploreExecutor().keep_threshold_pct == pytest.approx(DEFAULT_KEEP_THRESHOLD_PCT)


def test_an_explicit_threshold_still_wins(monkeypatch):
    from hyperloom.orchestrator.actions.executors.explore import ExploreExecutor

    monkeypatch.setenv(_BACKEND_ENV, "mlperf")
    assert ExploreExecutor(keep_threshold_pct=2.0).keep_threshold_pct == pytest.approx(2.0)
