# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX checkpoint a sheet carries: only the measured round is judged, shadow records, enforce stops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from hyperloom.common.agentx_early_stop import EarlyStopMode
from hyperloom.measure.checkpoint import CheckpointPlan, read_ruling
from hyperloom.measure.docs import DocumentError
from hyperloom.measure.spec import MeasurementSpec
from hyperloom.measure.tests.bundle_factory import spec_document
from hyperloom.measure.tests.test_local_backend import _call, _Grid, _point, _run, _Server, _agentx_point
from hyperloom.measure.tests.test_spec import _validator
from hyperloom.orchestrator.actions.executors import VariantResult, _agentx_checkpoint
from hyperloom.orchestrator.actions.executors._agentx_checkpoint import CheckpointJudge
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    AGENTX_EARLY_STOP_ERROR_CLASS,
    AGENTX_EARLY_STOP_RETURNCODE,
    AgentxCheckpointMissed,
)
from hyperloom.measure import local

VALIDATOR: Draft202012Validator = _validator("measurement_spec.v2.schema.json")


class _JudgedGrid(_Grid):
    """``run_grid`` whose rounds run under the checkpoint they are given, as its measured round does: the judge rules
    into the round's slot and, enforcing, stops a trailing round with the early stop's return code."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.judged: list[str] = []

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        (result,) = await super().__call__(**kwargs)
        checkpoint = kwargs["stop"].checkpoint
        if checkpoint is None:
            return [result]
        self.judged.append(result.name)
        slot = Path(kwargs["output_root"]) / f"variant_00_{result.name}"
        slot.mkdir(parents=True, exist_ok=True)
        judge = CheckpointJudge(checkpoint, slot)
        judge.note_phase("measured", 0.0)
        try:
            judge.tick(0.0)
        except AgentxCheckpointMissed as stopped:
            return [
                self.result(
                    result.name,
                    status="failed",
                    returncode=AGENTX_EARLY_STOP_RETURNCODE,
                    error_class=AGENTX_EARLY_STOP_ERROR_CLASS,
                    error=str(stopped),
                )
            ]
        return [result]


@pytest.fixture
def grid(monkeypatch, tmp_path) -> _JudgedGrid:
    fake = _JudgedGrid(tmp_path)
    monkeypatch.setattr(local, "run_grid", fake)
    monkeypatch.setattr(_agentx_checkpoint, "AGENTX_CHECKPOINT_SEC", 0.0)
    monkeypatch.setattr(_agentx_checkpoint, "workspace_checkpoint_p50", lambda root: 90.0)
    return fake


def _plan(mode: EarlyStopMode, reference: float = 100.0) -> CheckpointPlan:
    return CheckpointPlan(reference, mode)


@pytest.mark.parametrize("held", [True, False], ids=["held", "unheld"])
def test_a_shadow_checkpoint_records_the_ruling_and_the_test_runs_to_its_end(monkeypatch, grid, tmp_path, held):
    _Server(monkeypatch, eligible=held)
    call = _call(tmp_path, _agentx_point(tmp_path, monkeypatch), checkpoint=_plan(EarlyStopMode.SHADOW))

    outcome = _run(call)

    ruling = read_ruling(call.output_dir)
    assert outcome.returncode == 0 and outcome.failure_class is None
    assert ruling is not None and (ruling.trails, ruling.stopped) == (True, False)
    assert (ruling.reference_p50, ruling.candidate_p50) == (100.0, 90.0)
    assert ruling.gain_pct == pytest.approx(-10.0) and ruling.max_drop_pct == 2.0


def test_an_enforced_checkpoint_stops_a_trailing_test_as_the_candidates_ruling(monkeypatch, grid, tmp_path):
    _Server(monkeypatch)
    call = _call(tmp_path, _agentx_point(tmp_path, monkeypatch), checkpoint=_plan(EarlyStopMode.ENFORCE))

    outcome = _run(call)

    ruling = read_ruling(call.output_dir)
    assert (outcome.failure_class, outcome.failure_reason) == ("candidate", "agentx_early_stop")
    assert ruling is not None and ruling.stopped is True


def test_an_enforced_checkpoint_lets_a_test_that_keeps_up_finish(monkeypatch, grid, tmp_path):
    _Server(monkeypatch)
    call = _call(tmp_path, _agentx_point(tmp_path, monkeypatch), checkpoint=_plan(EarlyStopMode.ENFORCE, 91.0))

    outcome = _run(call)

    ruling = read_ruling(call.output_dir)
    assert outcome.failure_class is None and ruling is not None and (ruling.trails, ruling.stopped) == (False, False)


def test_only_the_measured_round_is_judged_not_the_warmup_or_the_evaluation(monkeypatch, grid, tmp_path):
    _Server(monkeypatch)
    point = _point(tmp_path, "p", run_eval="true")
    call = _call(tmp_path, point, accuracy=True, checkpoint=_plan(EarlyStopMode.SHADOW))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")

    _run(call)

    assert [c["grid"][0].name for c in grid.calls] == ["warmup", "test", "eval"]
    assert grid.judged == ["test"]


def test_a_call_without_a_checkpoint_judges_no_round_and_leaves_no_ruling(monkeypatch, grid, tmp_path):
    _Server(monkeypatch)
    call = _call(tmp_path, _agentx_point(tmp_path, monkeypatch))

    outcome = _run(call)

    assert outcome.failure_class is None and grid.judged == [] and read_ruling(call.output_dir) is None


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        ({"reference_p50": 50.0, "mode": "off"}, "must be one of shadow, enforce"),
        ({"reference_p50": 0.0, "mode": "shadow"}, "positive"),
        ({"reference_p50": 50.0, "mode": "shadow", "extra": 1}, "extra"),
    ],
)
def test_a_sheet_never_carries_an_off_checkpoint_or_one_without_a_reference(document, reason):
    with pytest.raises(DocumentError, match=reason):
        CheckpointPlan.parse(document, "procedure.checkpoint")


def test_the_checkpoint_round_trips_through_the_sheet_and_the_schema_accepts_it():
    document = spec_document()
    document["procedure"]["checkpoint"] = _plan(EarlyStopMode.ENFORCE, 42.5).to_document()

    spec = MeasurementSpec.from_document(document)

    assert spec.procedure.checkpoint == _plan(EarlyStopMode.ENFORCE, 42.5)
    assert spec.procedure.to_document()["checkpoint"] == {"reference_p50": 42.5, "mode": "enforce"}
    assert list(VALIDATOR.iter_errors(document)) == []
    assert list(VALIDATOR.iter_errors({**document, "procedure": {**document["procedure"], "checkpoint": {}}})) != []
