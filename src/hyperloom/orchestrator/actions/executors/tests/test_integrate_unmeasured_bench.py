# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A bench that measured nothing says nothing about the patch, so it is not graded as a REVERT."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._integrate_attempt import IntegrateAttempt
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.state.shared_state import SharedState

PORT_TAKEN = (
    "Benchmark failed: ['AgentX server port 8888 is unavailable; existing servers are not reused', "
    "'inferencex_result.json not found - benchmark may have failed locally']"
)


async def _gate(tmp_path: Path, bench_result: dict) -> dict:
    state = SharedState(baseline_tput=314.0, current_best={"tput": 314.0})
    attempt = IntegrateAttempt(
        task_id="t-int", specialist_task_id="t-spec", shared_state=state, output_root=tmp_path / "run"
    )
    return await IntegratePatchExecutor(session_dir=tmp_path / "session")._gate_perf(
        attempt=attempt,
        params={"base_tput": 314.0, "keep_threshold_pct": 1.0},
        extra={"shared_state": state},
        bench_result=bench_result,
        gate_evidence={"accuracy_pass": None},
    )


@pytest.mark.asyncio
async def test_a_bench_that_never_reached_the_server_is_unmeasured_not_reverted(tmp_path):
    """Smoke 2: an orphaned profile server held port 8888, and two upstream PRs were recorded as REVERT."""
    out = await _gate(
        tmp_path,
        {
            "status": "failed",
            "output_throughput": None,
            "error": PORT_TAKEN,
            "error_class": "magpie_nonzero_invalid_measurement",
        },
    )

    assert (out["status"], out["error_class"], out["failure_attribution"], out["bench_error_class"]) == (
        "failed",
        "bench_unmeasured",
        "harness",
        "magpie_nonzero_invalid_measurement",
    )
    assert out["reason"] == "bench produced no measurement (harness): magpie_nonzero_invalid_measurement"


@pytest.mark.asyncio
async def test_a_bench_the_candidate_broke_is_still_reverted(tmp_path):
    out = await _gate(
        tmp_path,
        {
            "status": "failed",
            "output_throughput": None,
            "error": "error: unrecognized arguments: --enable-foo",
            "error_class": "invalid_argument",
        },
    )

    assert out["status"] == "reverted"
