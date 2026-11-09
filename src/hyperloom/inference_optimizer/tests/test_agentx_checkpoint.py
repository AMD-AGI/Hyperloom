# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX checkpoint: its median, the judge that rules on it, and the grid's handling of a stopped round."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom_common.env import EnvValueError
from hyperloom.common.perf_metric import (
    GRADED_INTVTY_P50_CHECKPOINT,
    graded_axes_of,
    perf_snapshot_from_mapping,
)
from hyperloom.common.agentx_early_stop import EARLY_STOP_ENV
from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors._agentx_checkpoint import (
    CHECKPOINT_ARTIFACT_NAME,
    AgentxCheckpoint,
    CheckpointJudge,
    round_checkpoint,
    workspace_checkpoint_p50,
)
from hyperloom.orchestrator.actions.executors._grid_base import SuiteRoundExit
from hyperloom.orchestrator.actions.executors._grid_runner import GridRecipe, GridStop, GridVariant, run_grid
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    AGENTX_EARLY_STOP_ERROR_CLASS,
    AGENTX_EARLY_STOP_RETURNCODE,
    AgentxCheckpointMissed,
)
from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement

from .conftest import write_benchmark_yaml

_T0 = 1_000.0

# (phase, start s, end s, seconds per output token, failed). The profiling window closes at _T0 + 1800 = 2800.
_IN_WINDOW = [
    ("warmup", 500.0, 600.0, 0.5, False),
    ("profiling", _T0, 1_100.0, 0.1, False),
    ("profiling", 1_010.0, 1_500.0, 0.05, False),
    ("profiling", 1_020.0, 2_700.0, 0.04, False),
    ("profiling", 1_030.0, 1_200.0, 0.01, True),
]
_AFTER_WINDOW = ("profiling", 1_040.0, 3_000.0, 0.2, False)
_SUMMARY = "intvty p50 at 1800s 25.0->20.0 (-20.00%; stops below -2.0%)"


def _write_export(root: Path, rows: list[tuple[str, float, float, float, bool]]) -> Path:
    export = root / "aiperf_artifacts" / "profile_export.jsonl"
    export.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (phase, start, end, sec_per_token, failed) in enumerate(rows):
        lines.append(
            json.dumps(
                {
                    "metadata": {
                        "x_request_id": f"r{i}",
                        "benchmark_phase": phase,
                        "request_start_ns": int(start * 1e9),
                        "request_end_ns": int(end * 1e9),
                    },
                    "metrics": {
                        "request_latency": {"value": sec_per_token * 1000 * 1000, "unit": "ms"},
                        "output_sequence_length": {"value": 1000, "unit": "tokens"},
                    },
                    "error": {"code": 500} if failed else None,
                }
            )
        )
    export.write_text("\n".join(lines) + "\n")
    return export


def _ruling(slot: Path) -> dict:
    return json.loads((slot / CHECKPOINT_ARTIFACT_NAME).read_text())


def test_checkpoint_median_reads_only_profiling_requests_done_by_the_checkpoint(tmp_path):
    _write_export(tmp_path / "benchmark_vllm_1", [*_IN_WINDOW, _AFTER_WINDOW])
    assert workspace_checkpoint_p50(tmp_path) == 20.0


def test_checkpoint_median_waits_for_the_window_to_close(tmp_path):
    _write_export(tmp_path, _IN_WINDOW)
    assert workspace_checkpoint_p50(tmp_path) is None


def test_agentx_measurement_carries_the_checkpoint_median(tmp_path):
    _write_export(tmp_path, [*_IN_WINDOW, _AFTER_WINDOW])
    report = {"scenario": "agentx", "framework": "vllm", "model": "m", "agentx_metrics": {}}
    measurement = extract_benchmark_measurement(report, workspace=tmp_path)
    assert measurement[GRADED_INTVTY_P50_CHECKPOINT] == 20.0


def test_an_inference_suite_agentx_round_carries_the_checkpoint_median(tmp_path):
    from hyperloom.common.tests.test_round_measurement import _agentx
    from inference_suite.contracts import RunLayout

    run_dir = tmp_path / "run-1"
    layout = RunLayout(run_dir)
    layout.report.parent.mkdir(parents=True, exist_ok=True)
    layout.report.write_text(json.dumps(_agentx().to_document()), encoding="utf-8")
    _write_export(layout.client_dir, [*_IN_WINDOW, _AFTER_WINDOW])

    measurement = extract_benchmark_measurement(None, workspace=run_dir)

    assert measurement[GRADED_INTVTY_P50_CHECKPOINT] == 20.0


def test_reference_snapshots_carry_the_checkpoint_median():
    measured = {
        "e2e_norm_intvty_p90": 10.0,
        "e2e_norm_intvty_p50": 20.0,
        "total_token_throughput": 1000.0,
        GRADED_INTVTY_P50_CHECKPOINT: 19.0,
    }
    assert perf_snapshot_from_mapping(measured)[GRADED_INTVTY_P50_CHECKPOINT] == 19.0
    assert graded_axes_of(measured)[GRADED_INTVTY_P50_CHECKPOINT] == 19.0


@pytest.mark.parametrize(
    "mode,expected",
    [
        (None, AgentxCheckpoint(25.0, enforce=True)),
        ("Shadow", AgentxCheckpoint(25.0, enforce=False)),
        ("off", None),
    ],
)
def test_round_checkpoint_follows_the_mode(monkeypatch, mode, expected):
    if mode is None:
        monkeypatch.delenv(EARLY_STOP_ENV, raising=False)
    else:
        monkeypatch.setenv(EARLY_STOP_ENV, mode)
    assert round_checkpoint({GRADED_INTVTY_P50_CHECKPOINT: 25.0}) == expected


def test_round_checkpoint_refuses_an_unknown_mode(monkeypatch):
    monkeypatch.setenv(EARLY_STOP_ENV, "shdow")
    with pytest.raises(EnvValueError, match="HYPERLOOM_AGENTX_EARLY_STOP='shdow' is not one of enforce, shadow, off"):
        round_checkpoint({GRADED_INTVTY_P50_CHECKPOINT: 25.0})


def test_round_checkpoint_needs_a_reference_median(monkeypatch):
    monkeypatch.setenv(EARLY_STOP_ENV, "enforce")
    assert round_checkpoint({"e2e_norm_intvty_p50": 25.0}) is None
    assert round_checkpoint(None) is None


def test_judge_rules_once_the_checkpoint_is_due(tmp_path):
    _write_export(tmp_path, [*_IN_WINDOW, _AFTER_WINDOW])
    judge = CheckpointJudge(AgentxCheckpoint(25.0), tmp_path)
    judge.note_phase("measured", 0.0)
    judge.tick(1_799.0)
    assert not (tmp_path / CHECKPOINT_ARTIFACT_NAME).exists()
    judge.tick(1_800.0)
    assert _ruling(tmp_path) == {
        "at_sec": 1800.0,
        "candidate_p50": 20.0,
        "reference_p50": 25.0,
        "gain_pct": pytest.approx(-20.0),
        "max_drop_pct": 2.0,
        "trails": True,
        "stopped": False,
    }


def test_enforced_judge_stops_a_trailing_round(tmp_path):
    _write_export(tmp_path, [*_IN_WINDOW, _AFTER_WINDOW])
    judge = CheckpointJudge(AgentxCheckpoint(25.0, enforce=True), tmp_path)
    judge.note_phase("measured", 0.0)
    with pytest.raises(AgentxCheckpointMissed) as stopped:
        judge.tick(1_800.0)
    assert str(stopped.value) == _SUMMARY
    assert stopped.value.returncode == AGENTX_EARLY_STOP_RETURNCODE
    assert _ruling(tmp_path)["stopped"] is True


def test_enforced_judge_lets_a_round_within_the_drop_run_on(tmp_path):
    _write_export(tmp_path, [*_IN_WINDOW, _AFTER_WINDOW])
    judge = CheckpointJudge(AgentxCheckpoint(20.2, enforce=True), tmp_path)
    judge.note_phase("measured", 0.0)
    judge.tick(1_800.0)
    assert (_ruling(tmp_path)["trails"], _ruling(tmp_path)["stopped"]) == (False, False)


def test_judge_rereads_an_export_whose_window_is_still_open(tmp_path):
    _write_export(tmp_path, _IN_WINDOW)
    judge = CheckpointJudge(AgentxCheckpoint(25.0), tmp_path)
    judge.note_phase("measured", 0.0)
    judge.tick(1_800.0)
    _write_export(tmp_path, [*_IN_WINDOW, _AFTER_WINDOW])
    judge.tick(1_829.0)
    assert not (tmp_path / CHECKPOINT_ARTIFACT_NAME).exists()
    judge.tick(1_830.0)
    assert _ruling(tmp_path)["candidate_p50"] == 20.0


@pytest.mark.asyncio
async def test_grid_records_a_stopped_round_and_moves_on(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(tmp_path / "leaks"))
    base = tmp_path / "base.yaml"
    write_benchmark_yaml(base)
    checkpoint = AgentxCheckpoint(25.0, enforce=True)
    seen: list[AgentxCheckpoint | None] = []

    def _stopped_at_checkpoint(_route, bench_round, **_kwargs):
        seen.append(bench_round.checkpoint)
        _write_export(bench_round.output_dir, [*_IN_WINDOW, _AFTER_WINDOW])
        judge = CheckpointJudge(checkpoint, bench_round.output_dir)
        judge.note_phase("measured", 0.0)
        with pytest.raises(AgentxCheckpointMissed):
            judge.tick(1_800.0)
        return SuiteRoundExit(AGENTX_EARLY_STOP_RETURNCODE)

    monkeypatch.setattr(gr, "suite_route", lambda session_dir: object())
    monkeypatch.setattr(gr, "run_grid_suite_round", _stopped_at_checkpoint)
    results = await run_grid(
        grid=[GridVariant("vA"), GridVariant("vB")],
        output_root=tmp_path / "out",
        recipe=GridRecipe(base_yaml_path=base),
        stop=GridStop(keep_going_on_failure=False, checkpoint=checkpoint),
    )
    assert seen == [checkpoint, checkpoint]
    assert [(r.name, r.error_class, r.error, r.intvty_p50_checkpoint) for r in results] == [
        ("vA", AGENTX_EARLY_STOP_ERROR_CLASS, _SUMMARY, 20.0),
        ("vB", AGENTX_EARLY_STOP_ERROR_CLASS, _SUMMARY, 20.0),
    ]
