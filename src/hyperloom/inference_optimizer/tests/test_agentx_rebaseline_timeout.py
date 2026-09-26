# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rebaseline uses the benchmark policy rather than a task- or AgentX-derived cap."""

from __future__ import annotations

import sys

import pytest

from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor


@pytest.mark.parametrize("task_cap", [60, 7200, 9000, 50000])
def test_rebaseline_task_cannot_shrink_or_expand_round_cap(monkeypatch, tmp_path, task_cap):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_DURATION", "50000")
    monkeypatch.setenv("AGENTX_BASELINE_TIMEOUT_SEC", "50000")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "7800")
    executor = BaselineExecutor(magpie_python=sys.executable, session_dir=tmp_path)
    assert executor._resolve_timeout({"timeout_sec": task_cap, "accuracy_timeout_sec": task_cap}) == 7800
