# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A verified correctness fix may KEEP a bounded throughput drop."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.actions.executors._integrate_attempt import IntegrateAttempt
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.measurement.runtime_findings import persist_runtime_findings, scan_server_log
from hyperloom.orchestrator.state.shared_state import SharedState

ENV_LOG = "WARNING [envs.py:2128] Unknown vLLM environment variable detected: VLLM_FOO\n"
DISABLED_LOG = "WARNING [compilation.py:1183] Disabling fuse_rope_kvcache.\n"
TRACEBACK_LOG = 'Traceback (most recent call last):\n  File "x.py", line 1, in f\nRuntimeError: bad output\n'
CLEAN_LOG = "server ready\n"
FINDING = "vllm.unknown_env:VLLM_FOO"


def _slot(tmp_path: Path, name: str, log_text: str) -> str:
    slot = tmp_path / name
    slot.mkdir()
    (slot / "server.log").write_text(log_text, encoding="utf-8")
    report = scan_server_log(str(slot / "server.log"), "vllm", declared_env=("VLLM_FOO",))
    persist_runtime_findings(report, slot=slot)
    return str(slot / "launch_evidence.json")


def _state(tmp_path: Path) -> SharedState:
    return SharedState(
        baseline_tput=1000.0,
        current_best={"tput": 1000.0},
        current_best_measurement={
            "launch_evidence_path": _slot(tmp_path, "before", ENV_LOG + DISABLED_LOG + TRACEBACK_LOG)
        },
    )


async def _gate(tmp_path: Path, *, after_log: str, tput: float, accuracy_pass: bool | None, finding: str = FINDING):
    state = _state(tmp_path)
    attempt = IntegrateAttempt(
        task_id="t-int",
        specialist_task_id="t-spec",
        shared_state=state,
        done_payload={"payload": {"resolves_finding": finding}},
        output_root=tmp_path / "run",
    )
    return await IntegratePatchExecutor(session_dir=tmp_path / "session")._gate_perf(
        attempt=attempt,
        params={"base_tput": 1000.0, "keep_threshold_pct": 1.0},
        extra={"shared_state": state},
        bench_result={"output_throughput": tput, "launch_evidence_path": _slot(tmp_path, "after", after_log)},
        gate_evidence={"accuracy_pass": accuracy_pass},
    )


@pytest.mark.asyncio
async def test_verified_fix_keeps_a_small_drop(tmp_path):
    out = await _gate(tmp_path, after_log=CLEAN_LOG, tput=985.0, accuracy_pass=True)

    assert (out["status"], out["keep_reason"], out["resolves_finding"]) == ("kept", "correctness_fix", FINDING)
    assert out["reason"] == f"correctness fix {FINDING} verified at throughput delta -1.50%"


@pytest.mark.asyncio
async def test_fix_still_detected_reverts_with_reason(tmp_path):
    out = await _gate(tmp_path, after_log=ENV_LOG, tput=985.0, accuracy_pass=True)

    assert out["status"] == "reverted"
    assert out["reason"] == (
        "throughput delta -1.50% < keep_threshold 1.00%; "
        f"correctness fix {FINDING} refused: {FINDING} is still detected on the candidate run"
    )


@pytest.mark.asyncio
async def test_fix_drop_over_allowance_reverts(tmp_path):
    out = await _gate(tmp_path, after_log=CLEAN_LOG, tput=960.0, accuracy_pass=True)

    assert out["status"] == "reverted"
    assert out["reason"] == (
        "throughput delta -4.00% < keep_threshold 1.00%; "
        f"correctness fix {FINDING} refused: throughput delta -4.00% exceeds the 3.0% correctness-fix allowance"
    )


@pytest.mark.asyncio
async def test_fix_without_accuracy_is_accuracy_unavailable(tmp_path):
    out = await _gate(tmp_path, after_log=CLEAN_LOG, tput=985.0, accuracy_pass=None)

    assert out["status"] == "accuracy_unavailable_reject"


@pytest.mark.asyncio
async def test_restoring_a_disabled_fast_path_keeps_a_small_drop(tmp_path):
    finding = "feature_disabled:fuse_rope_kvcache"
    out = await _gate(tmp_path, after_log=CLEAN_LOG, tput=985.0, accuracy_pass=True, finding=finding)

    assert (out["status"], out["keep_reason"], out["resolves_finding"]) == ("kept", "correctness_fix", finding)


@pytest.mark.asyncio
async def test_traceback_finding_cannot_claim_the_allowance(tmp_path):
    out = await _gate(
        tmp_path, after_log=CLEAN_LOG, tput=985.0, accuracy_pass=True, finding="runtime.traceback:RuntimeError"
    )

    assert out["status"] == "reverted"
    assert out["reason"].endswith("runtime.traceback cannot justify a correctness fix")


def _coord(tmp_path: Path) -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(baseline_tput=1000.0, current_best={"tput": 1000.0}, model_path="/models/m")
    return coord


def test_lift_promotes_a_correctness_fix_below_the_anchor(tmp_path):
    coord = _coord(tmp_path)
    winner = {
        "name": "t-spec",
        "output_throughput": 985.0,
        "keep_reason": "correctness_fix",
        "resolves_finding": FINDING,
        "attribution_eligible": False,
    }

    assert coord.writeback.lift_to_current_best("integrate_patch", 985.0, winner) is True
    entry = coord.shared_state.optimization_stack[-1]
    assert (entry["keep_reason"], entry["resolves_finding"], entry["attribution_eligible"]) == (
        "correctness_fix",
        FINDING,
        False,
    )


def test_lift_still_holds_an_ordinary_winner_below_the_anchor(tmp_path):
    coord = _coord(tmp_path)

    assert (
        coord.writeback.lift_to_current_best("integrate_patch", 985.0, {"name": "t", "output_throughput": 985.0})
        is False
    )
    assert coord.shared_state.optimization_stack == []
