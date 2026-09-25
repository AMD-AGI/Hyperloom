# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The benchmark cap the MLPerf agentic backend needs to finish a measurement."""

from __future__ import annotations

import os

from hyperloom.inference_optimizer.agentx.deploy import (
    MLPERF_CANONICAL_TRAJECTORIES,
    MLPERF_SMOKE_TRAJECTORIES,
    mlperf_benchmark_timeout_sec,
    mlperf_trajectories,
)
from hyperloom.inference_optimizer.cli import (
    BENCHMARK_TIMEOUT_ENV,
    _apply_mlperf_benchmark_timeout,
)

_STOCK_SEC = 7800.0


def test_trajectories_default_per_flow():
    assert mlperf_trajectories({"MLPERF_AGENTIC_FLOW": "smoke_test"}) == MLPERF_SMOKE_TRAJECTORIES
    assert mlperf_trajectories({"MLPERF_AGENTIC_FLOW": "full"}) == MLPERF_CANONICAL_TRAJECTORIES


def test_trajectories_honour_explicit_override():
    assert mlperf_trajectories({"MLPERF_AGENTIC_FLOW": "smoke_test", "AGENTIC_NUM_TRAJECTORIES": "25"}) == 25


def test_full_run_needs_more_than_the_stock_cap():
    """A 613-trajectory run takes ~4.4h measured; the stock 2.17h cap kills it."""
    derived = mlperf_benchmark_timeout_sec({"MLPERF_AGENTIC_FLOW": "full"}, floor=_STOCK_SEC)
    assert derived > _STOCK_SEC
    # Comfortably covers the measured 613 x 26s, with boot and cold-start margin.
    assert derived >= MLPERF_CANONICAL_TRAJECTORIES * 26


def test_small_smoke_never_tightens_the_cap():
    derived = mlperf_benchmark_timeout_sec({"AGENTIC_NUM_TRAJECTORIES": "25"}, floor=_STOCK_SEC)
    assert derived == _STOCK_SEC


def test_cli_raises_the_cap_for_a_full_run(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "mlperf")
    monkeypatch.setenv("MLPERF_AGENTIC_FLOW", "full")
    monkeypatch.delenv(BENCHMARK_TIMEOUT_ENV, raising=False)
    _apply_mlperf_benchmark_timeout()
    assert float(os.environ[BENCHMARK_TIMEOUT_ENV]) > _STOCK_SEC


def test_cli_never_overrides_an_operator_pin(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "mlperf")
    monkeypatch.setenv("MLPERF_AGENTIC_FLOW", "full")
    monkeypatch.setenv(BENCHMARK_TIMEOUT_ENV, "12345")
    _apply_mlperf_benchmark_timeout()
    assert os.environ[BENCHMARK_TIMEOUT_ENV] == "12345"


def test_cli_leaves_the_aiperf_backend_alone(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)
    monkeypatch.delenv(BENCHMARK_TIMEOUT_ENV, raising=False)
    _apply_mlperf_benchmark_timeout()
    assert BENCHMARK_TIMEOUT_ENV not in os.environ
