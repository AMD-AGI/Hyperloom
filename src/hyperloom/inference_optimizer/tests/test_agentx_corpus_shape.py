# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A native AgentX baseline records the ISL/OSL distribution its round actually replayed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

_ISL = {"unit": "tokens", "avg": 155473.2, "p50": 108378.5, "p75": 184810.0, "p90": 371795.2, "p99": 557163.23}
_OSL = {"unit": "tokens", "avg": 785.9, "p50": 383.0, "p75": 790.75, "p90": 1561.2, "p99": 7288.48}
_MEASURED = {
    "isl": {"avg": 155473, "p50": 108378, "p75": 184810, "p90": 371795, "p99": 557163},
    "osl": {"avg": 785, "p50": 383, "p75": 790, "p90": 1561, "p99": 7288},
    "source": "measured",
}


def _workspace(tmp_path: Path, export: dict | None) -> Path:
    """An AgentX run_dir whose client left ``export`` as aiperf's summary, one level below the run_dir."""
    from hyperloom.common.tests.test_round_measurement import _agentx
    from inference_suite.contracts import RunLayout

    run_dir = tmp_path / "run"
    layout = RunLayout(run_dir)
    artifacts = layout.client_dir / "aiperf_artifacts"
    artifacts.mkdir(parents=True)
    layout.report.write_text(json.dumps(_agentx().to_document()), encoding="utf-8")
    if export is not None:
        (artifacts / "profile_export_aiperf.json").write_text(json.dumps(export))
    return run_dir


def _measure(workspace: Path) -> dict:
    return extract_benchmark_measurement(None, workspace=workspace)


def test_the_rounds_aiperf_export_gives_the_measurement_its_corpus_shape(tmp_path):
    measured = _measure(_workspace(tmp_path, {"input_sequence_length": _ISL, "output_sequence_length": _OSL}))

    assert measured["agentx_corpus_shape"] == _MEASURED


@pytest.mark.parametrize("export", [None, {"input_sequence_length": _ISL}, {"input_sequence_length": "n/a"}])
def test_a_round_without_both_lengths_reports_no_corpus_shape(tmp_path, export):
    assert "agentx_corpus_shape" not in _measure(_workspace(tmp_path, export))


async def test_a_promoted_baseline_carries_the_measured_corpus_shape_into_the_session(session_dir, tmp_path):
    idle = ScriptedPlan(
        turns=[],
        default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"}),
    )
    coord = Coordinator(session_dir, backends={n: MockBackend(idle, name=n) for n in ("orchestration", "critic")})
    try:
        measured = _measure(_workspace(tmp_path, {"input_sequence_length": _ISL, "output_sequence_length": _OSL}))
        task = Task(task_id="t-base", kind="baseline", state="running", params={}, idempotency_key="idem-t-base")

        await coord.writeback.promote_to_shared_state("baseline", {**measured, "output_throughput": 55.0}, task=task)

        assert coord.shared_state.agentx_corpus_shape == _MEASURED
    finally:
        await coord.stop()
