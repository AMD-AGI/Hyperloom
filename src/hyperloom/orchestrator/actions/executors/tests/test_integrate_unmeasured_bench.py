# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A bench the environment kept from measuring says nothing about the patch, so it is not graded as a REVERT."""

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
ARG_REFUSED = "error: unrecognized arguments: --enable-foo"


async def _gate(tmp_path: Path, bench_result: dict, *, switched: bool, patched: bool = True) -> tuple[dict, list[str]]:
    state = SharedState(baseline_tput=314.0, current_best={"tput": 314.0})
    attempt = IntegrateAttempt(
        task_id="t-int",
        specialist_task_id="t-spec",
        shared_state=state,
        output_root=tmp_path / "run",
        switch_manifest=[{"switch": "SGLANG_FOO", "value": "1"}] if switched else [],
        applied=[tmp_path / "001.patch"] if patched else [],
    )
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    lessons: list[str] = []
    legs: list[str] = []

    async def _record(lesson, **_kwargs):
        lessons.append(lesson.outcome)

    async def _parity(*_args, **_kwargs):
        legs.append("parity")
        return {"ran": True, "ok": False, "inconclusive": True, "reason": "parity leg returned no throughput"}

    executor._maybe_write_framework_kb_record = _record
    executor._switch_off_parity = _parity
    out = await executor._gate_perf(
        attempt=attempt,
        params={"base_tput": 314.0, "keep_threshold_pct": 1.0},
        bench_result=bench_result,
        gate_evidence={"accuracy_pass": None},
    )
    return out, lessons + legs


@pytest.mark.asyncio
async def test_a_bench_whose_server_port_was_taken_is_unmeasured_not_reverted(tmp_path):
    out, recorded = await _gate(
        tmp_path,
        {
            "status": "failed",
            "output_throughput": None,
            "error": PORT_TAKEN,
            "error_class": "round_nonzero_invalid_measurement",
        },
        switched=True,
    )

    assert (out["status"], out["error_class"], out["failure_attribution"], out["bench_error_class"]) == (
        "failed",
        "bench_unmeasured",
        "environment",
        "round_nonzero_invalid_measurement",
    )
    assert recorded == []


@pytest.mark.asyncio
async def test_a_bench_the_candidates_arguments_broke_is_still_reverted(tmp_path):
    out, recorded = await _gate(
        tmp_path,
        {
            "status": "failed",
            "output_throughput": None,
            "error": ARG_REFUSED,
            "error_class": "round_nonzero_invalid_measurement",
        },
        switched=False,
    )

    assert out["status"] == "reverted"
    assert recorded == ["reverted_smoke_fail"]


UNIMPORTABLE = {
    "status": "failed",
    "output_throughput": None,
    "error": "/opt/venv/bin/python cannot import sglang",
    "error_class": "framework_version_unknown",
}


@pytest.mark.asyncio
async def test_a_framework_the_attempts_patch_left_unimportable_is_reverted(tmp_path):
    out, recorded = await _gate(tmp_path, UNIMPORTABLE, switched=False)

    assert out["status"] == "reverted"
    assert recorded == ["reverted_smoke_fail"]


@pytest.mark.asyncio
async def test_an_unimportable_framework_an_attempt_without_a_patch_left_alone_is_unmeasured(tmp_path):
    out, recorded = await _gate(tmp_path, UNIMPORTABLE, switched=False, patched=False)

    assert (out["error_class"], out["failure_attribution"]) == ("bench_unmeasured", "environment")
    assert recorded == []
