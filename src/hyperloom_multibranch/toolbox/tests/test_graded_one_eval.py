# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A graded test evaluates once, in its warmup round, and the result the rule gate reads is made of that evaluation.

The real local backend runs the test with ``run_grid`` replaced by a stand-in that lays out each round's directory
the way the old flow does (a report in every round, lm-eval's files where the round evaluated); the toolbox then turns
the outcome into the rounds and accuracy of its result.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from hyperloom.common.env import is_truthy
from hyperloom.contracts.measurement import AccuracyValue, Purpose, TestKind
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure import local
from hyperloom.measure.local import LocalMeasurementBackend
from hyperloom.measure.socket import RoundPlan, TestCall
from hyperloom.orchestrator.actions.executors import VariantResult
from hyperloom_multibranch.toolbox.hosted_executor import _measured_values
from hyperloom_multibranch.toolbox.tests.helpers import req

FLEXIBLE = "exact_match,flexible-extract"


class _Grid:
    """``run_grid``: every call is one round with a report in its directory, and lm-eval's files when it evaluates."""

    def __init__(self, root: Path, score: float, throughputs: dict[str, float]) -> None:
        self.root, self.score, self.throughputs = root, score, throughputs
        self.evaluated: list[str] = []
        self.reseeded: dict[str, bool] = {}

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        name = kwargs["grid"][0].name
        self.reseeded[name] = kwargs["launch"].warmup_round
        config = yaml.safe_load(Path(kwargs["recipe"].base_yaml_path).read_text())
        envs = {**config["benchmark"]["envs"], **(kwargs["recipe"].base.extra_envs or {})}
        workspace = self.root / f"variant_00_{name}" / "benchmark_fake_1"
        workspace.mkdir(parents=True)
        report = {
            "success": True,
            "throughput": {"output_throughput": self.throughputs[name], "completed_requests": 8, "num_prompts": 8},
        }
        (workspace / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
        if is_truthy(envs["RUN_EVAL"], default=True):
            self.evaluated.append(name)
            results = workspace / "eval_output"
            results.mkdir()
            document = {"results": {"gsm8k": {FLEXIBLE: self.score}}}
            (results / "results_2026-10-14T10-00-00.json").write_text(json.dumps(document), encoding="utf-8")
        return [VariantResult(name, "", {}, "succeeded", returncode=0, workspace=str(workspace))]


def _graded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **plan: Any) -> tuple[_Grid, TestCall, Path]:
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "magpie")
    monkeypatch.setattr(
        local, "resolve_lifecycle_params", lambda path: {"eligible": True, "port": 8123, "framework": "sglang"}
    )
    monkeypatch.setattr(local, "teardown_lifecycle_server", lambda **kwargs: None)
    point = tmp_path / "point.yaml"
    envs = {"TP": 1, "CONC": 4, "RUN_EVAL": "true"}
    point.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": envs}}), encoding="utf-8")
    grid = _Grid(tmp_path / "out", 0.64, {"warmup": 1111.0, "test": 2222.0})
    monkeypatch.setattr(local, "run_grid", grid)
    call = TestCall(
        points=(point,),
        output_dir=tmp_path / "out",
        timeout_sec=900,
        cwd=str(tmp_path),
        magpie_python="",
        task_id="t-1",
        kind="accuracy",
        accuracy=True,
        rounds=RoundPlan(**plan),
    )
    return grid, call, point


def _result_of(call: TestCall, point: Path):
    request = req(kind=TestKind.ACCURACY, purpose=Purpose.BASELINE, accuracy=True, change=RecipeDelta())
    outcome = LocalMeasurementBackend().run_test(call)
    return outcome, _measured_values(request, SimpleNamespace(points=(point,)), outcome)  # type: ignore[arg-type]


def test_a_graded_test_with_a_warmup_runs_one_evaluation_and_its_score_is_the_results(tmp_path, monkeypatch):
    grid, call, point = _graded(tmp_path, monkeypatch, warmup_rounds=1)

    outcome, (rounds, accuracy) = _result_of(call, point)

    assert grid.evaluated == ["warmup"], "one evaluation, in the warmup round, none after the measured bench"
    assert accuracy == (AccuracyValue("gsm8k", FLEXIBLE, 0.64),)
    assert [(r.phase, r.metrics["output_throughput"]) for r in rounds] == [("warmup", 1111.0), ("measure", 2222.0)]
    assert [r.phase for r in outcome.rounds] == ["warmup", "accuracy", "measure"]


def test_the_warmup_round_draws_other_prompts_than_the_measured_round(tmp_path, monkeypatch):
    """With one seed, the measured round reused the warmup's prompts from the prefix cache whenever the evaluation between
    them did not evict them, and read 6-14 % high at C64 (Ruby, 2026-11-01)."""
    monkeypatch.delenv("HYPERLOOM_WARMUP_OWN_PROMPTS", raising=False)
    grid, call, point = _graded(tmp_path, monkeypatch, warmup_rounds=1)

    _result_of(call, point)

    assert grid.reseeded == {"warmup": True, "test": False}


def test_the_flag_set_false_gives_the_warmup_the_measured_rounds_prompts(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_WARMUP_OWN_PROMPTS", "0")
    grid, call, point = _graded(tmp_path, monkeypatch, warmup_rounds=1)

    _result_of(call, point)

    assert grid.reseeded == {"warmup": False, "test": False}


def test_a_graded_test_without_a_warmup_still_evaluates_in_its_one_round(tmp_path, monkeypatch):
    grid, call, point = _graded(tmp_path, monkeypatch, warmup_rounds=0)
    grid.throughputs["test"] = 3333.0

    outcome, (rounds, accuracy) = _result_of(call, point)

    assert grid.evaluated == ["test"]
    assert accuracy == (AccuracyValue("gsm8k", FLEXIBLE, 0.64),)
    assert [(r.phase, r.metrics["output_throughput"]) for r in rounds] == [("measure", 3333.0)]
