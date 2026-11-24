# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The in-session gate judges a stop by the campaign's own score, so a session and the loop cannot disagree."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

import kernelforge.workflows.forge_loop.gate.insession_gate as gate_module
from kernelforge.evaluation.scoring import CaseAggregate
from kernelforge.workflows.forge_loop.gate.gate_settings import DeviceAccess, GateMeasurement, GateScope
from kernelforge.tests.workflows.forge_loop.gate._gate_workspace import committed_task_workspace

# One case 4x faster, one 5x slower: an arithmetic win of 2.1x and a geometric loss of 0.894x.
_CANDIDATE_CASE_MS = {"won": 0.25, "lost": 5.0}


def _edited_gate(tmp_path: Path, monkeypatch, aggregate: CaseAggregate, **measurement) -> gate_module.InSessionGate:
    workspace, kernel = committed_task_workspace(tmp_path)
    gate = gate_module.InSessionGate(
        GateMeasurement(driver_script=str(workspace / "forge_driver.py"), case_aggregate=aggregate, **measurement),
        GateScope(kernel_file=str(kernel), target_files=[str(kernel)]),
        baseline_case_times={"won": 1.0, "lost": 1.0},
        best_mean_case_speedup=1.0,
    )
    kernel.write_text("__global__ void kernel() { }\n")

    gate.driver_runs = []

    async def correct(**kwargs):
        gate.driver_runs.append(("correctness", kwargs.get("env"), _waiting(gate)))
        return {"passed": True, "message": ""}

    async def bench(run, **_kwargs):
        gate.driver_runs.append(("benchmark", run.env, _waiting(gate)))
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


def _waiting(gate: gate_module.InSessionGate) -> int:
    """How many holders and waiters the gate's device queue has right now; 0 for a gate with a device of its own."""
    if gate.device is None:
        return 0
    return len(list(Path(gate.device.queue_root).iterdir()))


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


def test_a_lane_beside_others_may_stop_on_a_correct_candidate_that_is_no_faster(tmp_path, monkeypatch):
    """What the lane gained on one case is the merge's to use, so its score does not hold the session back."""
    gate = _edited_gate(tmp_path, monkeypatch, CaseAggregate.GEOMETRIC, gain_required=False)

    decision = asyncio.run(gate._on_stop({}, None, None))

    assert decision == {}
    assert (gate.passed, gate.end_reason) == (True, "converged")
    # The measurement is still taken and kept: the merge reads it case by case.
    assert gate.last_mean_case_speedup == pytest.approx(0.8**0.5)
    assert gate.last_bench_result["case_times"] == _CANDIDATE_CASE_MS


def test_a_lane_beside_others_is_still_sent_back_for_an_incorrect_candidate(tmp_path, monkeypatch):
    gate = _edited_gate(tmp_path, monkeypatch, CaseAggregate.GEOMETRIC, gain_required=False)

    async def incorrect(**_kwargs):
        return {"passed": False, "outcome": "correctness_failure", "message": "SNR 3 dB", "output": "mismatch"}

    monkeypatch.setattr(gate_module, "test_correctness", incorrect, raising=False)

    decision = asyncio.run(gate._on_stop({}, None, None))

    assert decision["decision"] == "block"
    assert "fails correctness" in decision["reason"]


def test_a_lanes_stop_check_holds_the_shared_device_and_measures_in_the_lanes_environment(tmp_path, monkeypatch):
    """The check runs in the loop's process, so it queues like any lane command and carries the lane's build caches."""
    device = DeviceAccess(
        runner=str(tmp_path / "forge_device.py"),
        queue_root=str(tmp_path / "queue"),
        usage_log=str(tmp_path / "usage.jsonl"),
        lane="2",
        environment={"AITER_JIT_DIR": "/lane-2/jit"},
    )
    gate = _edited_gate(tmp_path, monkeypatch, CaseAggregate.ARITHMETIC, device=device)

    asyncio.run(gate._on_stop({}, None, None))

    assert gate.driver_runs == [
        ("correctness", {"AITER_JIT_DIR": "/lane-2/jit"}, 1),
        ("benchmark", {"AITER_JIT_DIR": "/lane-2/jit"}, 1),
    ]
    assert list((tmp_path / "queue").iterdir()) == []
    [record] = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert (record["lane"], record["command"]) == ("2", "stop check")


def test_a_session_with_its_own_device_measures_without_queueing(tmp_path, monkeypatch):
    gate = _edited_gate(tmp_path, monkeypatch, CaseAggregate.ARITHMETIC)

    asyncio.run(gate._on_stop({}, None, None))

    assert gate.driver_runs == [("correctness", None, 0), ("benchmark", None, 0)]
