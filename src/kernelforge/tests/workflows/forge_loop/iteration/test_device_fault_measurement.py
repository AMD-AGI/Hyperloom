# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A GPU hang during a measurement: waiting for the device, measuring once more, and whom the hang is pinned on."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from kernelforge.evaluation.correctness.pipeline import ValidationReport, ValidationResult
from kernelforge.evaluation.device_faults import DEVICE_HANG
from kernelforge.tests.conftest import patch_iteration_name
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _make_loop,
    _measurement_loop,
    _unused_supervisor,
)
from kernelforge.workflows.forge_loop.campaign.run_state import LoopStateStore
from kernelforge.workflows.forge_loop.iteration import attempt_verdict
from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.iteration.iteration_types import _decision_label

HANG = "HW Exception by GPU node-2 (Agent handle: 0x5dac2f79a5b0) reason :GPU Hang"
MEMORY = "Memory access fault by GPU node-2 (Agent handle: 0x55a1) on address 0x7f0000. Reason: Page not present."


def _measured(candidate_ms=1.0 / 1.05):
    return {
        "success": True,
        "median_ms": candidate_ms,
        "case_times": {"small": candidate_ms, "large": candidate_ms},
        "unscored_cases": [],
        "measurement_count": 3,
        "measurements": [
            {"success": True, "case_times": {"small": candidate_ms, "large": candidate_ms}, "unscored_cases": []}
            for _ in range(3)
        ],
        "message": "three measurements",
    }


def _aborted(line):
    return {"success": False, "message": "driver exited with code -6", "output": f"case small ok\n{line}\n"}


def _validation(output=""):
    if not output:
        return ValidationReport([ValidationResult(5, "full", True, "SNR 40 dB", snr_db=40.0)])
    return ValidationReport(
        [ValidationResult(5, "full", False, "driver exited", outcome="driver_error", output=output)]
    )


def _loop(monkeypatch, *, benches, validations=None, recovers=True):
    """A loop whose measurements report ``benches`` and ``validations`` in turn, on a GPU that ``recovers`` or not."""
    loop, bench_calls = _measurement_loop(monkeypatch, _measured())
    benches, validations = iter(benches), iter(validations or [])
    waits, events = [], []

    async def bench(_run, **kwargs):
        bench_calls.append(kwargs)
        return next(benches)

    async def validate(**_kwargs):
        return next(validations, _validation())

    async def wait_for_device_recovery():
        waits.append(recovers)
        return recovers

    patch_iteration_name(monkeypatch, "measure_wallclock", bench)
    monkeypatch.setattr(attempt_verdict, "run_validation_pipeline", validate)
    monkeypatch.setattr(attempt_verdict, "wait_for_device_recovery", wait_for_device_recovery)
    loop.state_store = SimpleNamespace(append_event=events.append)
    return loop, bench_calls, waits, events


def test_a_hang_the_candidate_does_not_repeat_is_ruled_on_by_the_clean_measurement(monkeypatch):
    """A hang ends every process holding the GPU, so the run that reported it may only have been beside its cause."""
    loop, bench_calls, waits, events = _loop(monkeypatch, benches=[_aborted(HANG), _measured()])

    result = asyncio.run(loop.run_one_iteration(1))

    assert _decision_label(result) == "KEEP"
    assert (len(bench_calls), waits) == (2, [True])
    assert [(e["type"], e["kind"], e["node"], e["recovered"], e["repeated"]) for e in events] == [
        ("device_fault", DEVICE_HANG, 2, True, False)
    ]


def test_a_candidate_that_hangs_the_gpu_in_both_measurements_is_reverted_for_it(monkeypatch):
    loop, _bench_calls, _waits, events = _loop(monkeypatch, benches=[_aborted(HANG), _aborted(HANG)])

    result = asyncio.run(loop.run_one_iteration(1))

    assert _decision_label(result) == "REVERT_DEVICE_FAULT"
    assert not result.kept
    assert result.validation_summary.startswith("DEVICE FAULT: the candidate hung the GPU in both of its measurements")
    assert HANG in result.error_output
    assert [e["repeated"] for e in events] == [True]


def test_a_hang_during_the_correctness_suite_is_measured_again_too(monkeypatch):
    loop, bench_calls, waits, _events = _loop(monkeypatch, benches=[_measured()], validations=[_validation(HANG)])

    result = asyncio.run(loop.run_one_iteration(1))

    assert _decision_label(result) == "KEEP"
    assert (len(bench_calls), waits) == (1, [True])


def test_a_memory_access_fault_is_the_candidates_own_and_is_not_measured_again(monkeypatch):
    """Only the faulting process dies on a memory access fault, so the run that reported it caused it."""
    loop, bench_calls, waits, events = _loop(monkeypatch, benches=[_aborted(MEMORY)])

    result = asyncio.run(loop.run_one_iteration(1))

    assert _decision_label(result) == "REVERT_PERF"
    assert (len(bench_calls), waits, events) == (1, [], [])


def test_a_gpu_that_does_not_recover_leaves_the_candidate_unjudged(monkeypatch):
    loop, bench_calls, _waits, events = _loop(monkeypatch, benches=[_aborted(HANG)], recovers=False)

    result = asyncio.run(loop.run_one_iteration(1))

    assert _decision_label(result) == "DEVICE_UNAVAILABLE"
    assert len(bench_calls) == 1
    assert loop.device_unavailable.line == HANG
    assert [(e["recovered"], e["repeated"]) for e in events] == [(False, False)]


def test_no_candidate_is_measured_on_a_gpu_that_did_not_recover(monkeypatch):
    """A round may hold more candidates than the one whose measurement found the GPU gone."""
    loop, bench_calls, waits, events = _loop(monkeypatch, benches=[_aborted(HANG)], recovers=False)
    asyncio.run(loop.run_one_iteration(1))

    result = asyncio.run(loop.run_one_iteration(2))

    assert _decision_label(result) == "DEVICE_UNAVAILABLE"
    assert (len(bench_calls), len(waits), len(events)) == (1, 1, 1)


def test_a_gpu_that_does_not_recover_stops_the_run(tmp_path, monkeypatch):
    """Every measurement after it would fail for the device's reason, and each would be pinned on a candidate."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=5)
    sessions = []

    async def agent(kernel_path, _history, session_sink):
        sessions.append(len(sessions) + 1)
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        session_sink["plan"] = "a candidate"
        session_sink["end_reason"] = "candidate_submitted"
        return "candidate"

    async def validate(**_kwargs):
        return _validation(HANG)

    async def wait_for_device_recovery():
        return False

    monkeypatch.setattr(attempt_verdict, "run_validation_pipeline", validate)
    monkeypatch.setattr(attempt_verdict, "wait_for_device_recovery", wait_for_device_recovery)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=agent, supervisor_fn=_unused_supervisor)))

    events = LoopStateStore(str(workspace)).read_events()
    assert loop.termination_reason == "device_unrecovered"
    assert sessions == [1]
    assert [e["decision"] for e in events if e.get("type") == "iteration_result"] == ["DEVICE_UNAVAILABLE"]
    assert [e["kind"] for e in events if e.get("type") == "device_fault"] == [DEVICE_HANG]
