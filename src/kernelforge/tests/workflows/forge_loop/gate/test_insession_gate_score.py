# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The in-session gate judges a stop by the campaign's own score, so a session and the loop cannot disagree."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

import kernelforge.workflows.forge_loop.gate.insession_gate as gate_module
from kernelforge.evaluation.scoring import CaseAggregate
from kernelforge.workflows.forge_loop.gate.gate_settings import GateMeasurement, GateScope
from kernelforge.tests.workflows.forge_loop.gate._gate_workspace import committed_task_workspace

# One case 4x faster, one 5x slower: an arithmetic win of 2.1x and a geometric loss of 0.894x.
_CANDIDATE_CASE_MS = {"won": 0.25, "lost": 5.0}


def _edited_gate(tmp_path: Path, monkeypatch, aggregate: CaseAggregate) -> gate_module.InSessionGate:
    workspace, kernel = committed_task_workspace(tmp_path)
    gate = gate_module.InSessionGate(
        GateMeasurement(driver_script=str(workspace / "forge_driver.py"), case_aggregate=aggregate),
        GateScope(kernel_file=str(kernel), target_files=[str(kernel)]),
        baseline_case_times={"won": 1.0, "lost": 1.0},
        best_mean_case_speedup=1.0,
    )
    kernel.write_text("__global__ void kernel() { }\n")

    async def correct(**_kwargs):
        return {"passed": True, "message": ""}

    async def bench(_run, **_kwargs):
        measurement = {"success": True, "case_times": dict(_CANDIDATE_CASE_MS), "unscored_cases": []}
        return {
            "success": True,
            "median_ms": 2.625,
            "case_times": dict(_CANDIDATE_CASE_MS),
            "measurements": [dict(measurement) for _ in range(3)],
        }

    monkeypatch.setattr(gate_module, "test_correctness", correct, raising=False)
    monkeypatch.setattr(gate_module, "measure_wallclock", bench, raising=False)
    monkeypatch.setattr(gate, "_candidate_diff_sha256", lambda: hashlib.sha256(b"").hexdigest())
    return gate


def test_an_arithmetic_campaign_lets_a_session_stop_on_a_large_gain_paid_for_by_one_case(tmp_path, monkeypatch):
    gate = _edited_gate(tmp_path, monkeypatch, CaseAggregate.ARITHMETIC)

    decision = asyncio.run(gate._on_stop({}, None, None))

    assert decision == {}
    assert gate.end_reason == "converged"
    assert gate.last_mean_case_speedup == pytest.approx(2.1)


def test_a_geometric_campaign_sends_the_same_session_back(tmp_path, monkeypatch):
    gate = _edited_gate(tmp_path, monkeypatch, CaseAggregate.GEOMETRIC)

    decision = asyncio.run(gate._on_stop({}, None, None))

    assert decision["decision"] == "block"
    assert "NOT faster than the current best" in decision["reason"]
    assert gate.passed is False
    assert gate.last_mean_case_speedup == pytest.approx(0.8**0.5)
