# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixed benchmark caps remain independent of AgentX client warmup settings."""

from __future__ import annotations

import sys

import pytest

from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor
from hyperloom.orchestrator.actions.executors._workload_envs import apply_agentx_switch


def _clear(monkeypatch):
    for key in (
        "HYPERLOOM_AGENTX",
        "AGENTX_WARMUP_GRACE_PERIOD",
        "AGENTX_WARMUP_GRACE_CONC",
        "CONC",
        "INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("agentx", ["0", "1"])
def test_baseline_cap_does_not_expand_with_workload(monkeypatch, tmp_path, agentx):
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", agentx)
    monkeypatch.setenv("AGENTX_DURATION", "50000")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "20000")
    executor = BaselineExecutor(magpie_python=sys.executable, session_dir=tmp_path)
    assert executor._resolve_timeout({"timeout_sec": 99}) == 7800
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "123.5")
    assert executor._resolve_timeout({}) == 123.5


def test_profile_retains_nonbenchmark_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "99")
    executor = ProfileExecutor(magpie_python=sys.executable, session_dir=tmp_path)
    assert executor.benchmark_watchdog is False
    assert executor._resolve_timeout({}) == 14400
    assert executor._resolve_timeout({"timeout_sec": 15000}) == 15000


def test_agentx_switch_keeps_profile_yaml_cap(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_DURATION", "50000")
    bench = {"framework": "sglang", "timeout_seconds": 14400}
    apply_agentx_switch(bench, active=True)
    assert bench["timeout_seconds"] == 14400


# --- the grace declares which concurrency it was measured at -------------------
