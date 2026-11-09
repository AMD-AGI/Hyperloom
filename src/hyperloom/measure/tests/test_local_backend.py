# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local backend against a stubbed ``run_grid``: the parameter mapping of each kind of test."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml

from hyperloom_common.env import is_truthy
from hyperloom.contracts.measurement import AccuracyValue, AttachmentRef
from hyperloom.measure import farm as farm_module
from hyperloom.measure import local
from hyperloom.measure.local import SUPPORTED_KINDS, LocalMeasurementBackend, TestCallRefused
from hyperloom.measure.points import AMBIENT_SHAPE_ENVS
from hyperloom.measure.repro import read_verdict
from hyperloom.measure.runner.rounds import rounds_document
from hyperloom.measure.socket import RoundPlan, SessionFacts, TestCall
from hyperloom.measure.tests import eval_refusal_logs, test_log_causes
from hyperloom.measure.tests._inferencex import CLIENT, CLIENT_PATH, LIBRARY, LIBRARY_PATH, stock_checkout
from hyperloom.measure.tests.run_dirs import write_run_dir
from hyperloom.orchestrator.actions.executors import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    PatchedServer,
    VariantResult,
)
from hyperloom.orchestrator.actions.executors._server_lifecycle import REUSE_PORT_DEFAULT
from inference_suite.contracts import FailedStep, RunLayout


def _point(directory: Path, name: str, *, conc: int = 4, run_eval: object = "false", **counts: object) -> Path:
    envs: dict[str, object] = {"TP": 1, "CONC": conc, **counts}
    if run_eval is not None:
        envs["RUN_EVAL"] = run_eval
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": envs}}), encoding="utf-8")
    return path


def _call(tmp_path: Path, *points: Path, **kwargs: Any) -> TestCall:
    defaults: dict[str, Any] = {
        "points": points or (_point(tmp_path, "p"),),
        "output_dir": tmp_path / "out",
        "timeout_sec": 900,
        "cwd": str(tmp_path),
        "task_id": "t-1",
    }
    return TestCall(**{**defaults, **kwargs})


class _Grid:
    """A ``run_grid`` stand-in: records each call and answers from a script keyed by the variant's name."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.calls: list[dict[str, Any]] = []
        self.answers: dict[str, VariantResult] = {}
        self.env_seen: list[dict[str, str | None]] = []
        self.configs: list[dict[str, Any]] = []  # what each call's point held while it ran: a copy is gone after
        self.shapes: list[dict[str, str]] = []  # the round's shape run_grid would read from the environment

    def result(self, name: str, **kwargs: Any) -> VariantResult:
        workspace = self.tmp_path / "ws" / name
        workspace.mkdir(parents=True, exist_ok=True)
        fields: dict[str, Any] = {
            "name": name,
            "extra_server_args": "",
            "extra_envs": {},
            "status": "succeeded",
            "returncode": 0,
            "workspace": str(workspace),
        }
        return VariantResult(**{**fields, **kwargs})

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        self.calls.append(kwargs)
        self.configs.append(yaml.safe_load(Path(kwargs["recipe"].base_yaml_path).read_text()))
        self.shapes.append({key: os.environ[key] for key in AMBIENT_SHAPE_ENVS if key in os.environ})
        self.env_seen.append(
            {
                key: os.environ.get(key)
                for key in (
                    "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR",
                    "INFERENCEX_PATH",
                    "HL_TEST_FLAG",
                    "PROFILE_EXTRA_BODY",
                )
            }
        )
        name = kwargs["grid"][0].name
        return [self.answers.get(name) or self.result(name)]


@pytest.fixture
def grid(monkeypatch, tmp_path) -> _Grid:
    fake = _Grid(tmp_path)
    monkeypatch.setattr(local, "run_grid", fake)
    return fake


class _Server:
    """Stands in for the lifecycle helpers: a server that is eligible and records its teardowns."""

    def __init__(self, monkeypatch, *, eligible: bool = True, survivors: tuple[int, ...] = ()) -> None:
        self.teardowns: list[dict[str, Any]] = []
        monkeypatch.setattr(
            local,
            "resolve_lifecycle_params",
            lambda path: {"eligible": eligible, "port": 8123, "framework": "sglang", "reason": "not eligible"},
        )

        def teardown(slot: Path) -> list[int]:
            self.teardowns.append({"pid_dir": slot})
            return list(survivors)

        monkeypatch.setattr(local, "teardown_variant_server", teardown)


@pytest.fixture
def server(monkeypatch) -> _Server:
    return _Server(monkeypatch)


@pytest.fixture
def unheld(monkeypatch) -> _Server:
    """A point whose server the lifecycle cannot hold: the test is one ``run_grid`` call."""
    return _Server(monkeypatch, eligible=False)


def _run(call: TestCall):
    return LocalMeasurementBackend().run_test(call)


def _own_port(config: dict[str, Any]) -> int:
    """The port a round's point carries, taken out of it: what is left is the point the test was asked to run."""
    port = int(config["benchmark"]["envs"].pop("PORT"))
    assert port != REUSE_PORT_DEFAULT, "the default is the port every round without one shares"
    return port


# --- single-point test --------------------------------------------------------------------------------------------


def test_a_single_point_test_is_one_run_grid_call_with_one_variant(grid, unheld, tmp_path):
    call = _call(tmp_path, result_dir="/res", session_deadline_sec=123.0, serving_lease="lease")

    outcome = _run(call)

    (kwargs,) = grid.calls
    assert [v.name for v in kwargs["grid"]] == ["test"]
    assert kwargs["output_root"] == tmp_path / "out"
    assert (kwargs["launch"].result_dir, kwargs["stop"].session_deadline_sec, kwargs["launch"].serving_lease) == (
        "/res",
        123.0,
        "lease",
    )
    assert kwargs["launch"].server_lifecycle is None
    (config,) = grid.configs
    _own_port(config)
    assert config == yaml.safe_load(call.points[0].read_text()), "the point is run as it is, with a port of its own"
    assert [(r.point, r.phase, r.returncode) for r in outcome.rounds] == [(0, "measure", 0)]
    assert (outcome.returncode, outcome.failure_class, outcome.request_id) == (0, None, "t-1")
    assert outcome.sandbox.backend == "local"


@pytest.mark.parametrize(("plan", "expected"), [(None, None), (0, False), (1, True)])
def test_the_warmup_plan_reaches_run_grid_as_it_was_stated(grid, unheld, tmp_path, plan, expected):
    _run(_call(tmp_path, rounds=RoundPlan(warmup_rounds=plan)))

    assert grid.calls[0]["launch"].warmup_before_measure is expected


def test_a_discarded_warmup_round_is_reported_as_its_own_round(grid, tmp_path):
    grid.answers["test"] = grid.result("test", nonfatal_warnings=["run_grid_warmup_discarded_first"])

    outcome = _run(_call(tmp_path))

    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 0), ("measure", 0)]
    assert outcome.rounds[0].output_dir == outcome.rounds[1].output_dir.parent / "warmup_round"


def test_the_warnings_run_grid_wrote_ride_on_the_first_round_of_the_point_once(grid, tmp_path):
    warnings = ["warmup_round_seed:1", "run_grid_warmup_discarded_first", "warmup_round_seed:1"]
    grid.answers["test"] = grid.result("test", nonfatal_warnings=warnings)

    outcome = _run(_call(tmp_path))

    assert [r.nonfatal_warnings for r in outcome.rounds] == [tuple(warnings), ()]
    assert [w for r in outcome.rounds for w in r.nonfatal_warnings] == warnings


def test_a_point_that_run_grid_did_not_warn_about_carries_none(grid, tmp_path):
    outcome = _run(_call(tmp_path))
    assert [r.nonfatal_warnings for r in outcome.rounds] == [()]


def test_an_accuracy_test_reports_the_accuracy_round_with_the_eval_exit_code(grid, unheld, tmp_path):
    result = grid.result("test", returncode=0)
    write_run_dir(Path(result.workspace), eval_code=3)
    grid.answers["test"] = result
    point = _point(tmp_path, "acc", run_eval="true")

    outcome = _run(_call(tmp_path, point, accuracy=True, kind="accuracy"))

    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("measure", 0), ("accuracy", 3)]
    assert outcome.returncode == 3


#: The version every run_dir report of these tests states for its evaluation.
LM_EVAL_VERSION = "lm_eval=0.4.9;gsm8k=3"


def _evaluated(run_dir: Path, score: float) -> None:
    """The round in ``run_dir`` evaluated ``score``, as its report says."""
    write_run_dir(run_dir, scores={"exact_match,flexible-extract": score}, eval_code=0)


def test_the_accuracy_round_carries_the_measured_rounds_scores_not_the_warmups(grid, unheld, tmp_path):
    result = grid.result("test", nonfatal_warnings=["run_grid_warmup_discarded_first"])
    workspace = Path(result.workspace)
    _evaluated(workspace, 0.64)
    _evaluated(workspace.parent / "warmup_round", 0.10)
    grid.answers["test"] = result
    point = _point(tmp_path, "acc", run_eval="true")

    outcome = _run(_call(tmp_path, point, accuracy=True, kind="accuracy"))

    by_phase = {r.phase: r.accuracy for r in outcome.rounds}
    assert by_phase["accuracy"] == (AccuracyValue("gsm8k", "exact_match,flexible-extract", 0.64, LM_EVAL_VERSION),)
    assert by_phase["warmup"] == by_phase["measure"] == ()


def test_the_accuracy_round_carries_the_version_lm_eval_reported(grid, unheld, tmp_path):
    result = grid.result("test")
    _evaluated(Path(result.workspace), 0.64)
    grid.answers["test"] = result
    point = _point(tmp_path, "acc", run_eval="true")

    outcome = _run(_call(tmp_path, point, accuracy=True, kind="accuracy"))

    (value,) = [r.accuracy for r in outcome.rounds if r.phase == "accuracy"][0]
    assert value.version == "lm_eval=0.4.9;gsm8k=3"


def test_an_accuracy_round_with_only_a_warmup_evaluation_carries_no_score(grid, tmp_path):
    result = grid.result("test", nonfatal_warnings=["run_grid_warmup_discarded_first"])
    _evaluated(Path(result.workspace).parent / "warmup_round", 0.10)
    grid.answers["test"] = result
    point = _point(tmp_path, "acc", run_eval="true")

    outcome = _run(_call(tmp_path, point, accuracy=True, kind="accuracy"))

    assert [r.accuracy for r in outcome.rounds if r.phase == "accuracy"] == [()]


def test_a_test_that_did_not_ask_for_accuracy_reports_no_accuracy_round(grid, tmp_path):
    outcome = _run(_call(tmp_path))

    assert [r.phase for r in outcome.rounds] == ["measure"]


# --- every test runs its rounds on one held server, the evaluation after the measured bench -------------------------


def _graded_call(tmp_path: Path, **kwargs: Any) -> TestCall:
    """A graded single-point test whose warmup is planned."""
    point = _point(tmp_path, "graded", run_eval="true")
    return _call(
        tmp_path, point, **{"accuracy": True, "kind": "accuracy", "rounds": RoundPlan(warmup_rounds=1), **kwargs}
    )


def _explore_call(tmp_path: Path, **kwargs: Any) -> TestCall:
    """An explore: the same single-point test, not graded."""
    return _call(tmp_path, **{"rounds": RoundPlan(warmup_rounds=1), **kwargs})


def _envs(kwargs: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    """A round's environment as ``run_grid`` would build it: the round's extra envs laid over the point's own."""
    return {
        key: str(value)
        for key, value in {**config["benchmark"]["envs"], **(kwargs["recipe"].base.extra_envs or {})}.items()
    }


def _evaluates(kwargs: dict[str, Any], config: dict[str, Any]) -> bool:
    return is_truthy(_envs(kwargs, config)["RUN_EVAL"], default=True)


def _benches(kwargs: dict[str, Any], config: dict[str, Any]) -> bool:
    """Whether a round runs a bench: InferenceX's ``run_benchmark_serving`` returns before it on ``EVAL_ONLY``."""
    return not is_truthy(_envs(kwargs, config).get("EVAL_ONLY"), default=False)


def _graded_answers(grid: _Grid, score: float = 0.64) -> VariantResult:
    """The scripted answers of a graded test: the evaluation's score sits in the evaluation round's directory, a score in
    any other round's directory is one no round of the test wrote."""
    warmup, measured, evaluation = (
        grid.result("warmup", output_throughput=1111.0),
        grid.result("test", output_throughput=2222.0),
        grid.result("eval", note="evaluation_only"),
    )
    _evaluated(Path(warmup.workspace), 0.10)
    _evaluated(Path(measured.workspace), 0.20)
    _evaluated(Path(evaluation.workspace), score)
    grid.answers.update(warmup=warmup, test=measured, eval=evaluation)
    return evaluation


@pytest.mark.parametrize(
    ("make_call", "rounds"),
    [(_explore_call, ["warmup", "test"]), (_graded_call, ["warmup", "test", "eval"])],
    ids=["explore", "graded"],
)
def test_every_test_runs_warmup_bench_then_measured_bench_and_a_graded_one_evaluates_after_them(
    grid, server, tmp_path, make_call, rounds
):
    _graded_answers(grid)

    _run(make_call(tmp_path))

    assert [c["grid"][0].name for c in grid.calls] == rounds, (
        "an explore and a graded test share the order to the bench"
    )
    calls = list(zip(grid.calls, grid.configs))
    assert [_benches(c, config) for c, config in calls] == [True, True, False][: len(rounds)]
    assert [_evaluates(c, config) for c, config in calls] == [False, False, True][: len(rounds)], (
        "no bench evaluates; the evaluation follows the measured bench"
    )
    assert [c["launch"].evaluation_only for c in grid.calls] == [False, False, True][: len(rounds)]
    assert [c["launch"].warmup_round for c in grid.calls] == [True, False, False][: len(rounds)], (
        "only the warmup draws prompts of its own"
    )


def test_the_rounds_of_a_graded_test_share_one_server_that_its_last_round_reclaims(grid, server, tmp_path):
    _graded_answers(grid)

    _run(_graded_call(tmp_path))

    warmup, measured, evaluation = grid.calls
    held = {"pid_dir": warmup["launch"].server_lifecycle["pid_dir"], "port": 8123}
    assert [c["launch"].server_lifecycle for c in grid.calls] == [
        {**held, "cleanup": False},
        {**held, "cleanup": False},
        {**held, "cleanup": True},
    ]
    assert [c["launch"].server_already_ready for c in grid.calls] == [False, True, True], "one boot, in the warmup"
    assert all(c["launch"].warmup_before_measure is False for c in grid.calls), "run_grid does not warm up"
    assert not any(c["launch"].lifecycle_boot_only for c in grid.calls), "every round runs its client"
    assert [c["recipe"].base.extra_envs for c in grid.calls] == [
        {"RUN_EVAL": "false"},
        {"RUN_EVAL": "false"},
        {"RUN_EVAL": "true", "EVAL_ONLY": "true"},
    ]
    assert len(server.teardowns) == 1, "and whatever is still up is reclaimed when the test ends"


def test_an_explore_on_a_held_server_runs_its_point_as_it_is_and_its_measured_round_reclaims_the_server(
    grid, server, tmp_path
):
    outcome = _run(_explore_call(tmp_path))

    assert [c["launch"].server_lifecycle["cleanup"] for c in grid.calls] == [False, True]
    assert [c["recipe"].base.extra_envs for c in grid.calls] == [None, None], "an explore's point does not evaluate"
    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 0), ("measure", 0)]
    assert len(server.teardowns) == 1


def test_the_accuracy_of_a_graded_test_is_read_from_its_evaluation_round(grid, server, tmp_path):
    evaluation = _graded_answers(grid, 0.64)

    outcome = _run(_graded_call(tmp_path))

    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 0), ("measure", 0), ("accuracy", 0)]
    warmed, measured, accuracy = outcome.rounds
    assert accuracy.output_dir == Path(evaluation.workspace)
    assert (warmed.output_dir, measured.output_dir) == (
        Path(grid.answers["warmup"].workspace),
        Path(grid.answers["test"].workspace),
    )
    assert accuracy.accuracy == (AccuracyValue("gsm8k", "exact_match,flexible-extract", 0.64, LM_EVAL_VERSION),)
    assert warmed.accuracy == measured.accuracy == ()
    assert outcome.returncode == 0


def test_the_rounds_file_of_a_graded_test_lists_its_rounds_in_the_order_they_ran(grid, server, tmp_path):
    _graded_answers(grid, 0.64)

    outcome = _run(_graded_call(tmp_path))

    document = rounds_document(outcome, tmp_path / "ws")
    assert [(entry["phase"], entry["dir"]) for entry in document] == [
        ("warmup", "warmup"),
        ("measure", "test"),
        ("accuracy", "eval"),
    ]
    assert [entry.get("accuracy") for entry in document] == [
        None,
        None,
        [{"task": "gsm8k", "metric": "exact_match,flexible-extract", "score": 0.64, "version": LM_EVAL_VERSION}],
    ]


def test_the_warmup_round_of_a_held_test_says_what_run_grid_says_of_a_discarded_one(grid, server, tmp_path):
    _graded_answers(grid)
    grid.answers["warmup"].nonfatal_warnings.append("harvested_leaked_artifact:x")
    grid.answers["test"].nonfatal_warnings.append("served_slowly")
    grid.answers["eval"].nonfatal_warnings.append("eval_note")

    outcome = _run(_graded_call(tmp_path))

    assert [r.nonfatal_warnings for r in outcome.rounds] == [
        ("run_grid_warmup_discarded_first", "warmup_round_tput:1111.0", "harvested_leaked_artifact:x"),
        ("served_slowly",),
        ("eval_note",),
    ]


def test_the_eval_exit_code_of_a_graded_test_is_the_one_its_evaluation_round_recorded(grid, server, tmp_path):
    evaluation = _graded_answers(grid)
    write_run_dir(Path(evaluation.workspace), scores={"exact_match,flexible-extract": 0.64}, eval_code=3)

    outcome = _run(_graded_call(tmp_path))

    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 0), ("measure", 0), ("accuracy", 3)]
    assert outcome.returncode == 3


def test_a_server_that_dies_in_the_evaluation_fails_the_accuracy_and_keeps_the_measured_reading(grid, server, tmp_path):
    _graded_answers(grid)
    workspace = Path(grid.answers["eval"].workspace)
    write_run_dir(workspace)  # the evaluation never finished, so the report holds no score and no exit code of it
    (workspace / "server.log").write_text(test_log_causes.ILLEGAL_ADDRESS_LOG, encoding="utf-8")
    grid.answers["eval"] = grid.result(
        "eval",
        status="failed",
        returncode=1,
        error="server died",
        error_class="round_nonzero_invalid_measurement",
        server_log_path=str(workspace / "server.log"),
    )

    outcome = _run(_graded_call(tmp_path))

    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 0), ("measure", 0), ("accuracy", 1)]
    measured, accuracy = outcome.rounds[1:]
    assert measured.output_dir == Path(grid.answers["test"].workspace), "the measured reading stays in the result"
    assert accuracy.accuracy == (), "a score from an evaluation that failed is no score"
    assert outcome.returncode == 1
    assert (outcome.failure_class, (outcome.failure_reason or "").split(":")[0]) == ("candidate", "hip_runtime_error")
    summary = {row["name"]: row["status"] for row in json.loads(outcome.stdout)}
    assert summary == {"warmup": "succeeded", "test": "succeeded", "eval": "failed"}
    assert len(server.teardowns) == 1


@pytest.mark.parametrize("make_call", [_explore_call, _graded_call], ids=["explore", "graded"])
def test_a_warmup_that_fails_ends_the_test_with_nothing_measured_and_no_accuracy(grid, server, tmp_path, make_call):
    _graded_answers(grid)
    grid.answers["warmup"] = grid.result("warmup", status="failed", returncode=7, error="boom", error_class="x")

    outcome = _run(make_call(tmp_path))

    assert [c["grid"][0].name for c in grid.calls] == ["warmup"]
    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 7)] and outcome.returncode == 7
    assert len(server.teardowns) == 1


def test_a_measured_round_that_fails_ends_a_graded_test_before_its_evaluation(grid, server, tmp_path):
    _graded_answers(grid)
    grid.answers["test"] = grid.result("test", status="failed", returncode=5, error="boom", error_class="x")

    outcome = _run(_graded_call(tmp_path))

    assert [c["grid"][0].name for c in grid.calls] == ["warmup", "test"]
    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 0), ("measure", 5)]
    assert outcome.returncode == 5 and len(server.teardowns) == 1


def test_a_round_whose_client_could_not_fetch_its_dependencies_is_an_infrastructure_failure(grid, server, tmp_path):
    """No server started: the git fetch of the replay client stalled, which says nothing of the candidate."""
    stall = (
        "subprocess.TimeoutExpired: Command ['git', '-C', '/w/InferenceX', 'submodule', 'update', '--init', "
        "'--recursive', '--', 'inferencex-e2e/utils/aiperf'] timed out after 300 seconds"
    )
    grid.answers["warmup"] = grid.result("warmup", status="failed", returncode=1, error=stall, error_class="x")

    outcome = _run(_graded_call(tmp_path))

    assert outcome.failure_class == "infra", outcome.failure_reason
    assert outcome.failure_reason.startswith("git_fetch_timeout: subprocess.TimeoutExpired")


def test_a_round_refused_before_it_ran_names_its_class_and_detail_as_the_reason(grid, server, tmp_path):
    """No run_dir, no server log: what the round runner recorded is the only account of the failure."""
    detail = "inferencex is not staged in /tools"
    grid.answers["warmup"] = grid.result(
        "warmup", status="failed", returncode=None, workspace=None, error=detail, error_class="staging_missing"
    )

    outcome = _run(_graded_call(tmp_path))

    assert (outcome.failure_class, outcome.failure_reason) == ("unclear", f"staging_missing: {detail}")


@pytest.mark.parametrize(
    ("step", "failure_class", "reason"),
    [
        (FailedStep.PORT_IN_USE, "infra", "port_in_use"),
        (FailedStep.GPUS_UNAVAILABLE, "infra", "host_gpus_unavailable"),
        (FailedStep.SERVER_SILENT_BEFORE_READY, "infra", "hung_before_ready"),
        (FailedStep.SERVER_STALLED, "unclear", "server_stalled"),
    ],
)
def test_the_step_a_rounds_report_failed_at_classifies_the_test_its_logs_cannot(
    grid, server, tmp_path, step, failure_class, reason
):
    """The suite refuses a taken port itself, so no bind error reaches ``server.log``; a traceback there is no cause."""
    workspace = write_run_dir(grid.tmp_path / "ws" / "warmup", failed_step=step)
    (workspace / "server.log").write_text("Traceback (most recent call last):\nRuntimeError: boom\n", encoding="utf-8")
    grid.answers["warmup"] = grid.result(
        "warmup", status="failed", returncode=1, error=step.value, server_log_path=str(workspace / "server.log")
    )

    outcome = _run(_graded_call(tmp_path))

    assert (outcome.failure_class, outcome.failure_reason) == (failure_class, reason)


def test_a_flag_the_suite_refused_names_the_refusal_its_report_records(grid, server, tmp_path):
    """The suite parses the argv before boot: the refusal is in ``report.json``'s ``failure_detail``, in no log."""
    detail = "the user layer: argument --attention-backend: invalid choice: 'nonexistent'"
    workspace = write_run_dir(
        grid.tmp_path / "ws" / "warmup", failed_step=FailedStep.FLAG_UNKNOWN, failure_detail=detail
    )
    grid.answers["warmup"] = grid.result("warmup", status="failed", returncode=1, error="flag_unknown")

    outcome = _run(_graded_call(tmp_path))

    assert (outcome.failure_class, outcome.failure_reason) == ("candidate", f"serve_flag: {detail}")
    assert [c["grid"][0].name for c in grid.calls] == ["warmup"]


@pytest.mark.parametrize(
    ("framework_changed", "failure_class", "reason"),
    [(False, "infra", "framework_unimportable"), (True, "candidate", "import_error")],
)
def test_a_framework_the_serving_interpreter_cannot_import_is_the_images_unless_the_test_rebuilt_it(
    grid, server, tmp_path, framework_changed, failure_class, reason
):
    detail = "/opt/venv/bin/python cannot import sglang"
    workspace = write_run_dir(
        grid.tmp_path / "ws" / "warmup", failed_step=FailedStep.FRAMEWORK_VERSION_UNKNOWN, failure_detail=detail
    )
    grid.answers["warmup"] = grid.result("warmup", status="failed", returncode=1, error="framework_version_unknown")

    outcome = _run(_graded_call(tmp_path, framework_changed=framework_changed))

    assert (outcome.failure_class, outcome.failure_reason) == (failure_class, f"{reason}: {detail}")


@pytest.mark.parametrize("round_name", ["warmup", "test", "eval"])
def test_a_graded_test_the_run_stops_in_a_round_reports_the_stop_and_runs_no_round_after_it(
    grid, server, tmp_path, round_name
):
    _graded_answers(grid)
    grid.answers[round_name] = grid.result(
        round_name, status="failed", returncode=None, error_class="session_time_exhausted", error="out of time"
    )
    write_run_dir(Path(grid.answers[round_name].workspace))  # the stopped round evaluated nothing

    outcome = _run(_graded_call(tmp_path))

    names = ["warmup", "test", "eval"]
    assert [c["grid"][0].name for c in grid.calls] == names[: names.index(round_name) + 1]
    assert outcome.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE and len(server.teardowns) == 1


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"rounds": RoundPlan(warmup_rounds=0)}, "no warmup is planned"),
        ({"rounds": RoundPlan(warmup_rounds=None)}, "pytest's own default is no warmup"),
    ],
)
def test_a_graded_test_without_a_warmup_boots_in_its_measured_round_and_evaluates_after_it(
    grid, server, tmp_path, kwargs, why
):
    evaluation = _graded_answers(grid)

    outcome = _run(_graded_call(tmp_path, **kwargs))

    assert [c["grid"][0].name for c in grid.calls] == ["test", "eval"], why
    assert [c["launch"].server_already_ready for c in grid.calls] == [False, True]
    assert [c["launch"].server_lifecycle["cleanup"] for c in grid.calls] == [False, True]
    assert [r.phase for r in outcome.rounds] == ["measure", "accuracy"]
    assert outcome.rounds[1].output_dir == Path(evaluation.workspace) and outcome.rounds[1].accuracy[0].score == 0.64


@pytest.mark.parametrize("held", [True, False], ids=["held", "unheld"])
@pytest.mark.parametrize("plan", [None, 1])
def test_an_agentx_test_is_measured_once_with_no_warmup_round_before_it(monkeypatch, grid, tmp_path, held, plan):
    """An AgentX replay is long and stable and warms its own client inside the round: one round, whatever the plan."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")
    _Server(monkeypatch, eligible=held)
    point = _agentx_point(tmp_path, monkeypatch)

    outcome = _run(_call(tmp_path, point, rounds=RoundPlan(warmup_rounds=plan)))

    (kwargs,) = grid.calls
    assert kwargs["grid"][0].name == "test"
    assert kwargs["launch"].warmup_before_measure is False and not kwargs["launch"].server_already_ready
    assert [r.phase for r in outcome.rounds] == ["measure"]


def test_a_graded_test_on_a_framework_that_cannot_hold_a_server_evaluates_in_its_one_round(monkeypatch, grid, tmp_path):
    _Server(monkeypatch, eligible=False)
    result = grid.result("test")
    _evaluated(Path(result.workspace), 0.64)
    grid.answers["test"] = result

    outcome = _run(_graded_call(tmp_path))

    (kwargs,) = grid.calls
    assert kwargs["grid"][0].name == "test" and kwargs["launch"].warmup_before_measure is True
    assert kwargs["launch"].server_lifecycle is None and kwargs["recipe"].base.extra_envs is None
    assert [r.phase for r in outcome.rounds] == ["measure", "accuracy"]
    assert outcome.rounds[1].accuracy[0].score == 0.64


def test_the_default_warmup_of_run_grid_decides_for_a_graded_test_that_states_none(grid, server, tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")
    _graded_answers(grid)

    _run(_graded_call(tmp_path, rounds=RoundPlan(warmup_rounds=None)))

    assert [c["grid"][0].name for c in grid.calls] == ["warmup", "test", "eval"]


# --- refusals -----------------------------------------------------------------------------------------------------


def test_a_point_that_leaves_run_eval_unset_is_refused_because_unset_means_on(grid, tmp_path):
    point = _point(tmp_path, "unset", run_eval=None)

    with pytest.raises(TestCallRefused, match="does not set RUN_EVAL"):
        _run(_call(tmp_path, point))
    assert grid.calls == []


def test_a_point_not_shaped_as_one_is_refused_like_any_call_the_backend_cannot_run(grid, tmp_path):
    point = tmp_path / "shapeless.yaml"
    point.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": "RUN_EVAL=false"}}), encoding="utf-8")

    with pytest.raises(TestCallRefused, match=str(point)):
        _run(_call(tmp_path, point))
    assert grid.calls == []


@pytest.mark.parametrize(("yaml_value", "accuracy"), [("true", False), ("false", True), (True, False), ("0", True)])
def test_a_point_whose_run_eval_contradicts_the_call_is_refused(grid, tmp_path, yaml_value, accuracy):
    point = _point(tmp_path, "mismatch", run_eval=yaml_value)

    with pytest.raises(TestCallRefused, match=f"asks for accuracy={accuracy}"):
        _run(_call(tmp_path, point, accuracy=accuracy))
    assert grid.calls == []


@pytest.mark.parametrize("kind", ["sweep", "bogus"])
def test_a_kind_the_local_backend_does_not_run_is_refused(grid, tmp_path, kind):
    with pytest.raises(TestCallRefused, match="runs kinds"):
        _run(_call(tmp_path, kind=kind))


def test_an_accuracy_kind_without_accuracy_is_refused(grid, tmp_path):
    with pytest.raises(TestCallRefused, match="must ask for accuracy"):
        _run(_call(tmp_path, kind="accuracy"))


def test_a_single_point_test_with_two_points_is_refused(grid, tmp_path):
    with pytest.raises(TestCallRefused, match="exactly one point"):
        _run(_call(tmp_path, _point(tmp_path, "a"), _point(tmp_path, "b")))


def test_env_overrides_may_not_set_a_blocked_name(grid, tmp_path):
    with pytest.raises(TestCallRefused, match="PYTHONPATH"):
        _run(_call(tmp_path, env_overrides={"PYTHONPATH": "/x"}))


@pytest.mark.parametrize("levels", [(4, 4), (4, 0)])
def test_a_sweep_needs_a_distinct_positive_level_on_every_point(grid, server, tmp_path, levels):
    points = [_point(tmp_path, f"c{i}", conc=level) for i, level in enumerate(levels)]

    with pytest.raises(TestCallRefused, match="names its own CONC"):
        _run(_call(tmp_path, *points, prep="conc_sweep"))


# --- the process environment --------------------------------------------------------------------------------------


def test_the_session_is_a_fresh_empty_directory_and_the_environment_is_put_back(grid, tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", "/the/real/session")
    monkeypatch.delenv("HL_TEST_FLAG", raising=False)
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    facts = SessionFacts(reference_unset_envs=("A_PIN",), inferencex_path="/ix")

    _run(_call(tmp_path, env_overrides={"HL_TEST_FLAG": "on"}, unset_envs=("B_PIN",), session_facts=facts))

    seen = grid.env_seen[0]
    assert seen["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] != "/the/real/session"
    assert (seen["HL_TEST_FLAG"], seen["INFERENCEX_PATH"]) == ("on", "/ix")
    assert grid.calls[0]["grid"][0].unset_envs == ["A_PIN", "B_PIN"]
    assert os.environ["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] == "/the/real/session"
    assert "HL_TEST_FLAG" not in os.environ and "INFERENCEX_PATH" not in os.environ
    assert not Path(seen["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] or "x").exists()


def test_the_environment_is_put_back_when_the_test_raises(grid, tmp_path, monkeypatch):
    monkeypatch.delenv("HL_TEST_FLAG", raising=False)
    grid.answers["test"] = grid.result("test", status="failed", error_class="round_timeout")

    with pytest.raises(subprocess.TimeoutExpired):
        _run(_call(tmp_path, env_overrides={"HL_TEST_FLAG": "on"}))

    assert "HL_TEST_FLAG" not in os.environ


def test_the_session_base_the_old_flow_keeps_reads_empty_in_the_test_session(tmp_path, monkeypatch):
    from hyperloom.orchestrator.state.base_launch import ServerBase, session_server_base
    from hyperloom.orchestrator.state.tests._session_base import freeze_session_base

    real = tmp_path / "real"
    real.mkdir()
    freeze_session_base(real, args=("--from-the-old-flow",))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(real))
    seen: list[ServerBase] = []

    async def probe(**kwargs):
        seen.append(session_server_base())
        return [VariantResult(name="test", extra_server_args="", extra_envs={}, status="succeeded", returncode=0)]

    monkeypatch.setattr(local, "run_grid", probe)

    _run(_call(tmp_path))

    assert seen == [ServerBase()]


def test_the_test_session_holds_the_calls_frozen_base_and_never_the_applications(tmp_path, monkeypatch):
    """The base a call carries reaches its rounds through the test's own session, frozen as a session freezes it, so
    the point routes to inference_suite with it; the application's session, with another base, is not read."""
    from hyperloom.measure.tests.bundle_factory import STAGING, base_launch
    from hyperloom.orchestrator.state.base_launch import frozen_session_base
    from hyperloom.orchestrator.state.tests._session_base import freeze_session_base

    real = tmp_path / "real"
    real.mkdir()
    freeze_session_base(real, args=("--from-the-application",))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(real))
    seen = []

    async def probe(**kwargs):
        seen.append(frozen_session_base())
        return [VariantResult(name="test", extra_server_args="", extra_envs={}, status="succeeded", returncode=0)]

    monkeypatch.setattr(local, "run_grid", probe)

    _run(_call(tmp_path, session_facts=SessionFacts(staging=STAGING, base_launch=base_launch())))

    (frozen,) = seen
    assert frozen.launch == base_launch() and frozen.path.parent != real
    assert not frozen.path.exists(), "the test session is removed with the test"


def test_the_per_round_cap_and_the_leak_scope_come_from_the_call(grid, tmp_path, monkeypatch):
    captured: dict[str, str | None] = {}

    async def look(**kwargs):
        captured.update(
            {
                k: os.environ.get(k)
                for k in ("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "INFERENCE_OPTIMIZER_LEAK_ROOTS")
            }
        )
        return [VariantResult(name="test", extra_server_args="", extra_envs={}, status="succeeded", returncode=0)]

    monkeypatch.setattr(local, "run_grid", look)

    _run(_call(tmp_path, timeout_sec=321))

    assert captured == {
        "INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC": "321",
        "INFERENCE_OPTIMIZER_LEAK_ROOTS": str(tmp_path),
    }


def test_two_tests_in_one_process_do_not_overlap(monkeypatch, tmp_path):
    active, peak = [0], [0]
    gate = threading.Lock()

    async def slow(**kwargs):
        with gate:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        await __import__("asyncio").sleep(0.05)
        with gate:
            active[0] -= 1
        return [VariantResult(name="test", extra_server_args="", extra_envs={}, status="succeeded", returncode=0)]

    monkeypatch.setattr(local, "run_grid", slow)
    threads = [threading.Thread(target=_run, args=(_call(tmp_path),)) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak[0] == 1


# --- one arm of a concurrency sweep -------------------------------------------------------------------------------


def _sweep_call(tmp_path: Path, *levels: int, **kwargs: Any) -> TestCall:
    points = [_point(tmp_path, f"c{level}", conc=level) for level in levels]
    return _call(tmp_path, *points, prep="conc_sweep", **kwargs)


def _lifecycle(call: dict[str, Any]) -> dict[str, Any] | None:
    return call["launch"].server_lifecycle


#: What a boot round adds to the point: no evaluation, and one request of the client round that runs after its server
#: phase. Not the phase itself, which no round names (``test_local_sweep_rounds``).
BOOT_ENVS = {"RUN_EVAL": "false", "NUM_PROMPTS": "1"}


def _envs_handed_to_the_round(grid: _Grid, tmp_path: Path, call: dict[str, Any]) -> dict[str, Any]:
    """The ``benchmark.envs`` ``run_grid`` makes of one recorded call: its own layering of the point and the recipe's
    base variables over the round's shape the process environment held while the call ran."""
    from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

    shape = grid.shapes[next(index for index, recorded in enumerate(grid.calls) if recorded is call)]
    with mock.patch.dict(os.environ):
        for key in AMBIENT_SHAPE_ENVS:
            os.environ.pop(key, None)
        os.environ.update(shape)
        candidate = _build_variant_yaml(
            call["recipe"],
            call["grid"][0],
            output_subdir=tmp_path / "candidates" / str(len(list((tmp_path / "candidates").glob("*")))),
        )
    return yaml.safe_load(candidate.read_text())["benchmark"]["envs"]


def test_a_sweep_boots_the_server_once_and_measures_the_levels_from_high_to_low_on_it(grid, server, tmp_path):
    outcome = _run(_sweep_call(tmp_path, 2, 8, 4))

    names = [c["grid"][0].name for c in grid.calls]
    assert names == ["c8", "c8", "c4", "c2"]
    boot, *reuse = grid.calls
    assert boot["launch"].lifecycle_boot_only is True and boot["recipe"].base.extra_envs == BOOT_ENVS
    assert all(c["recipe"].base.extra_envs is None for c in reuse), "the rounds that measure carry nothing of the boot"
    assert _lifecycle(boot)["cleanup"] is False
    assert all(c["launch"].server_already_ready is True for c in reuse)
    assert [_lifecycle(c)["cleanup"] for c in reuse] == [False, False, True]
    assert {(_lifecycle(c)["pid_dir"], _lifecycle(c)["port"]) for c in grid.calls} == {
        (str(tmp_path / "out" / "server_arm"), 8123)
    }
    assert all(c["launch"].warmup_before_measure is False for c in grid.calls)
    assert [r.point for r in outcome.rounds] == [1, 0 + 2, 0]  # c8, c4, c2 by their positions in the call
    assert len(server.teardowns) == 1


def _graded_sweep(tmp_path: Path, *levels: int, run_eval: str = "true") -> TestCall:
    """A sweep whose points hold the request counts the point generator gives them (an accuracy sweep by default)."""
    points = [
        _point(tmp_path, f"g{n}", conc=n, run_eval=run_eval, NUM_PROMPTS=n * 10, NUM_WARMUPS=min(n, 8)) for n in levels
    ]
    return _call(tmp_path, *points, prep="conc_sweep", accuracy=run_eval == "true")


@pytest.mark.parametrize("ambient_conc", [None, "64"], ids=["clean", "conc_left_in_the_process"])
def test_the_boot_of_a_sweep_arm_does_not_evaluate_and_asks_one_request_and_no_bench_evaluates(
    grid, server, tmp_path, monkeypatch, ambient_conc
):
    if ambient_conc is None:
        monkeypatch.delenv("CONC", raising=False)
    else:
        monkeypatch.setenv("CONC", ambient_conc)
    call = _graded_sweep(tmp_path, 8, 4, 2)
    points = [Path(p).read_text() for p in call.points]

    outcome = _run(call)

    boots = [c for c in grid.calls if c["launch"].lifecycle_boot_only]
    measured = [c for c in grid.calls if not c["launch"].lifecycle_boot_only]
    assert [c["grid"][0].name for c in boots] == ["c8"]
    assert [c["grid"][0].name for c in measured] == ["c8", "c4", "c2", "eval"]
    assert all(c["recipe"].base.extra_envs == BOOT_ENVS for c in boots)
    assert [c["recipe"].base.extra_envs for c in measured] == [
        {"RUN_EVAL": "false"},
        {"RUN_EVAL": "false"},
        {"RUN_EVAL": "false"},
        {"RUN_EVAL": "true", "EVAL_ONLY": "true"},
    ]
    assert [Path(p).read_text() for p in call.points] == points, "the points are the ones the call was accepted on"
    envs = _envs_handed_to_the_round(grid, tmp_path, boots[0])
    assert (envs["RUN_EVAL"], envs["NUM_PROMPTS"]) == ("false", "1")
    assert "HYPERLOOM_RUN_PHASE" not in envs and envs["CONC"] == 8
    assert envs["NUM_WARMUPS"] == 8, "a variable outside the reuse check is the point's own"
    for level, round_ in zip((8, 4, 2, 8), measured, strict=True):
        envs = _envs_handed_to_the_round(grid, tmp_path, round_)
        assert (envs["NUM_PROMPTS"], envs["NUM_WARMUPS"], envs["CONC"]) == (level * 10, min(level, 8), level)
        assert "HYPERLOOM_RUN_PHASE" not in envs
    assert [_envs_handed_to_the_round(grid, tmp_path, c)["RUN_EVAL"] for c in measured] == [
        "false",
        "false",
        "false",
        "true",
    ], "no bench evaluates; one evaluation, from the level the server booted on, after the last bench"
    assert [(r.point, r.phase) for r in outcome.rounds] == [
        (0, "measure"),
        (1, "measure"),
        (2, "measure"),
        (0, "accuracy"),
    ]
    assert outcome.returncode == 0
    assert os.environ.get("CONC") == ambient_conc, "the process environment is put back after the test"


def _graded_sweep_answers(grid: _Grid, *levels: int, score: float = 0.64) -> VariantResult:
    """The scripted answers of a graded sweep: each level's directory holds a score no round of the test wrote, and the
    evaluation's score sits in the evaluation round's directory."""
    for level in levels:
        bench = grid.answers[f"c{level}"] = grid.result(f"c{level}", output_throughput=100.0 * level)
        _evaluated(Path(bench.workspace), level / 100.0)
    evaluation = grid.answers["eval"] = grid.result("eval", note="evaluation_only")
    _evaluated(Path(evaluation.workspace), score)
    return evaluation


def test_a_graded_sweep_benches_every_level_and_then_evaluates_once_on_the_server_its_boot_left_up(
    grid, server, tmp_path
):
    """Every level after the first used to bench on the KV pool the evaluation of the level before it left full."""
    evaluation = _graded_sweep_answers(grid, 2, 8, 4)
    call = _graded_sweep(tmp_path, 2, 8, 4)

    outcome = _run(call)

    assert [c["grid"][0].name for c in grid.calls] == ["c8", "c8", "c4", "c2", "eval"]
    calls = list(zip(grid.calls, grid.configs))
    assert [_benches(c, config) for c, config in calls[1:]] == [True, True, True, False]
    assert [_evaluates(c, config) for c, config in calls] == [False, False, False, False, True], (
        "nothing evaluates before the last bench"
    )
    assert [c["launch"].evaluation_only for c in grid.calls] == [False, False, False, False, True]
    assert grid.calls[-1]["recipe"].base_yaml_path == call.points[1], "from the level the server booted on"
    assert [c["launch"].server_already_ready for c in grid.calls] == [False, True, True, True, True], "one boot"
    assert [_lifecycle(c)["cleanup"] for c in grid.calls] == [False, False, False, False, True]
    assert len(server.teardowns) == 1
    assert [(r.point, r.phase, r.returncode) for r in outcome.rounds] == [
        (1, "measure", 0),
        (2, "measure", 0),
        (0, "measure", 0),
        (1, "accuracy", 0),
    ]
    assert [r.accuracy for r in outcome.rounds] == [
        (),
        (),
        (),
        (AccuracyValue("gsm8k", "exact_match,flexible-extract", 0.64, LM_EVAL_VERSION),),
    ]
    assert outcome.rounds[-1].output_dir == Path(evaluation.workspace)
    assert outcome.returncode == 0


def test_the_rounds_file_of_a_graded_sweep_lists_every_bench_and_then_its_one_evaluation(grid, server, tmp_path):
    _graded_sweep_answers(grid, 8, 4)

    outcome = _run(_graded_sweep(tmp_path, 8, 4))

    document = rounds_document(outcome, tmp_path / "ws")
    assert [(entry["point"], entry["phase"], entry["dir"]) for entry in document] == [
        (0, "measure", "c8"),
        (1, "measure", "c4"),
        (0, "accuracy", "eval"),
    ]


@pytest.mark.parametrize(
    ("answer", "returncode"),
    [
        (dict(status="failed", returncode=5, error="boom", error_class="x"), 5),
        (
            dict(status="skipped", returncode=None, error_class="session_time_exhausted"),
            SESSION_TIME_EXHAUSTED_RETURNCODE,
        ),
    ],
    ids=["failed", "stopped"],
)
def test_a_graded_sweep_whose_bench_fails_or_is_stopped_runs_no_evaluation(grid, server, tmp_path, answer, returncode):
    _graded_sweep_answers(grid, 8, 4, 2)
    grid.answers["c4"] = grid.result("c4", **answer)

    outcome = _run(_graded_sweep(tmp_path, 8, 4, 2))

    assert "eval" not in [c["grid"][0].name for c in grid.calls]
    assert not any(r.phase == "accuracy" for r in outcome.rounds)
    assert outcome.returncode == returncode and len(server.teardowns) == 1


def test_a_graded_sweep_whose_evaluation_fails_keeps_every_bench_and_fails_its_accuracy(grid, server, tmp_path):
    _graded_sweep_answers(grid, 8, 4)
    grid.answers["eval"] = grid.result("eval", status="failed", returncode=1, error="server died", error_class="x")
    write_run_dir(Path(grid.answers["eval"].workspace))  # the evaluation never finished

    outcome = _run(_graded_sweep(tmp_path, 8, 4))

    assert [(r.point, r.phase, r.returncode) for r in outcome.rounds] == [
        (0, "measure", 0),
        (1, "measure", 0),
        (0, "accuracy", 1),
    ]
    assert outcome.rounds[-1].accuracy == (), "a score from an evaluation that failed is no score"
    assert outcome.returncode == 1 and len(server.teardowns) == 1


def test_a_graded_sweep_on_a_framework_that_cannot_hold_a_server_evaluates_in_each_levels_own_round(
    monkeypatch, grid, tmp_path
):
    """Each level has a server of its own, which no bench uses after its evaluation."""
    _Server(monkeypatch, eligible=False)
    _graded_sweep_answers(grid, 8, 4)

    outcome = _run(_graded_sweep(tmp_path, 8, 4))

    assert [c["grid"][0].name for c in grid.calls] == ["c8", "c4"]
    assert all(_lifecycle(c) is None and c["recipe"].base.extra_envs is None for c in grid.calls)
    assert [(r.point, r.phase) for r in outcome.rounds] == [
        (0, "measure"),
        (0, "accuracy"),
        (1, "measure"),
        (1, "accuracy"),
    ]


def test_a_sweep_that_does_not_run_accuracy_boots_the_same_way(grid, server, tmp_path):
    _run(_graded_sweep(tmp_path, 8, 4, run_eval="false"))

    boot, *measured = grid.calls
    assert boot["recipe"].base.extra_envs == BOOT_ENVS and all(c["recipe"].base.extra_envs is None for c in measured)


def test_a_boot_does_not_pin_what_the_point_pins_for_a_graded_round(grid, server, tmp_path):
    """The boot's overrides live in its round's environment; the call's own validation reads the point."""
    call = _graded_sweep(tmp_path, 8, 4)

    _run(call)  # an accuracy sweep whose points say RUN_EVAL=true is accepted, and its boots run

    assert all(yaml.safe_load(Path(p).read_text())["benchmark"]["envs"]["RUN_EVAL"] == "true" for p in call.points)
    assert grid.calls[0]["recipe"].base_yaml_path == call.points[0]


def test_an_agentx_boot_pins_nothing(grid, server, tmp_path, monkeypatch):
    """An AgentX point has no evaluation round and no prompt count, and no backend phase is said for a boot."""
    _boot_only(grid)
    _Case(tmp_path, monkeypatch, "exit 0")
    point = _agentx_point(tmp_path, monkeypatch)

    _run(_call(tmp_path, point, kind="repro", attachments=(_CASE,), timeout_sec=30))

    (boot,) = grid.calls
    assert boot["launch"].lifecycle_boot_only
    assert boot["recipe"].base.extra_envs == {}


@pytest.mark.parametrize("levels", [(8, 4, 2), (4,)], ids=["levels", "one-level"])
def test_an_agentx_sweep_boots_a_server_of_its_own_for_each_level(grid, tmp_path, monkeypatch, levels):
    """Every level replays one trace corpus with one seed: on a server a level before it warmed, it reads its cache."""
    torn_down: list[Path] = []
    monkeypatch.setattr(local, "teardown_variant_server", lambda slot: torn_down.append(slot) or [])
    points = [_agentx_point(tmp_path, monkeypatch, conc=level) for level in levels]
    assert local.resolve_lifecycle_params(points[0])["eligible"], "the lifecycle could hold an AgentX server"

    outcome = _run(_call(tmp_path, *points, prep="conc_sweep"))

    assert [c["grid"][0].name for c in grid.calls] == [f"c{level}" for level in levels]
    assert all(_lifecycle(c) is None and not c["launch"].lifecycle_boot_only for c in grid.calls)
    assert [(r.point, r.phase) for r in outcome.rounds] == [(i, "measure") for i in range(len(levels))]
    assert "server restarted per level: an AgentX replay is cache-sensitive" in outcome.stderr
    assert outcome.returncode == 0 and torn_down == []


def test_no_call_of_a_sweep_runs_without_the_shared_server(grid, server, tmp_path):
    _run(_sweep_call(tmp_path, 8, 4, 2))

    assert all(_lifecycle(c) is not None for c in grid.calls)


def test_a_round_the_lifecycle_holds_runs_from_the_point_itself_on_the_lifecycles_port(grid, server, tmp_path):
    call = _sweep_call(tmp_path, 8, 4)

    _run(call)

    assert [c["recipe"].base_yaml_path for c in grid.calls] == [call.points[0], call.points[0], call.points[1]]
    assert all("PORT" not in config["benchmark"]["envs"] for config in grid.configs)
    assert {_lifecycle(c)["port"] for c in grid.calls} == {8123}


def test_a_sweep_that_restarts_per_level_gives_every_level_a_port_of_its_own(monkeypatch, grid, tmp_path):
    _Server(monkeypatch, eligible=False)
    call = _sweep_call(tmp_path, 8, 4)

    _run(call)

    ports = [_own_port(config) for config in grid.configs]
    assert len(ports) == 2 and all(0 < port < 65536 for port in ports)
    assert all(config == yaml.safe_load(point.read_text()) for config, point in zip(grid.configs, call.points[:2]))


@pytest.mark.parametrize("levels", [(8, 4, 2), (8,)], ids=["levels", "one-level"])
def test_a_boot_that_fails_is_the_sweeps_failed_row_and_the_launch_is_not_booted_again(grid, server, tmp_path, levels):
    """A level's CONC never reaches the server's launch: a lower level, or a server per level, boots the same one."""
    grid.answers["c8"] = grid.result(
        "c8", status="failed", returncode=1, error_class="no_benchmark_workspace", error="oom"
    )

    outcome = _run(_sweep_call(tmp_path, *levels))

    (boot,) = grid.calls
    assert boot["grid"][0].name == "c8" and boot["launch"].lifecycle_boot_only
    assert [(r.point, r.phase, r.returncode) for r in outcome.rounds] == [(0, "measure", 1)]
    assert outcome.returncode == 1 and "oom" in outcome.stderr
    assert len(server.teardowns) == 1


def test_a_sweep_takes_its_server_down_through_its_pid_dir_and_names_what_outlived_it(monkeypatch, grid, tmp_path):
    """The arm's server is reaped as every grid caller's is: by its pid_dir, which a suite round's server.pid is in."""
    server = _Server(monkeypatch, survivors=(4242,))
    call = _sweep_call(tmp_path, 8, 4)

    outcome = _run(call)

    assert server.teardowns == [{"pid_dir": Path(call.output_dir) / "server_arm"}]
    assert "server pids [4242] outlived SIGKILL" in outcome.stderr


def test_a_framework_that_cannot_hold_a_server_restarts_per_level_and_says_so(monkeypatch, grid, tmp_path):
    _Server(monkeypatch, eligible=False)

    outcome = _run(_sweep_call(tmp_path, 8, 4))

    assert [c["grid"][0].name for c in grid.calls] == ["c8", "c4"]
    assert all(_lifecycle(c) is None for c in grid.calls)
    assert "server restarted per level: not eligible" in outcome.stderr


def test_a_sweep_stops_when_the_run_stops_it_and_reports_the_stop_code(grid, server, tmp_path):
    grid.answers["c4"] = grid.result(
        "c4", status="skipped", returncode=None, error_class="session_time_exhausted", error="out of time"
    )

    outcome = _run(_sweep_call(tmp_path, 8, 4, 2))

    assert [c["grid"][0].name for c in grid.calls] == ["c8", "c8", "c4"]
    assert outcome.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE
    assert outcome.failure_class is None and outcome.failure_reason == "session_time_exhausted"
    assert len(server.teardowns) == 1


def test_a_cancel_during_the_boot_ends_the_sweep_without_stepping_down(grid, server, tmp_path):
    grid.answers["c8"] = grid.result("c8", status="skipped", returncode=None, error_class="orchestrator_cancelled")

    outcome = _run(_sweep_call(tmp_path, 8, 4))

    assert [c["grid"][0].name for c in grid.calls] == ["c8"]
    assert outcome.returncode == ORCHESTRATOR_CANCELLED_RETURNCODE


# --- a compiled environment ---------------------------------------------------------------------------------------

OLD_SOURCE = "def f():\n    return 1\n"
NEW_SOURCE = "def f():\n    return 10\n"


@pytest.fixture
def installed_framework(tmp_path):
    """A real interpreter with a framework 'sglang' installed in it, as the image has one."""
    import subprocess
    import sys

    from hyperloom.measure.tests._recipes import write_tree

    venv = tmp_path / "image-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, capture_output=True)
    site = next((venv / "lib").glob("python*/site-packages"))
    package = write_tree(site / "sglang", {"__init__.py": "", "srt/__init__.py": "", "srt/a.py": OLD_SOURCE})
    return venv / "bin" / "python", package


def _compile(tmp_path, installed_framework):
    from hyperloom.contracts.recipe import RecipeDelta
    from hyperloom.measure.compile import ComponentLayout, LocalBlobStore, PackageBase, compile_environment
    from hyperloom.measure.tests._recipes import IMAGE, PREFIX, git_patch, put_patch, recipe_with

    _, package = installed_framework
    blobs = LocalBlobStore(tmp_path / "blobs")
    patch = git_patch(tmp_path / "pw", {f"{PREFIX}/srt/a.py": OLD_SOURCE}, {f"{PREFIX}/srt/a.py": NEW_SOURCE})
    recipe = recipe_with((put_patch(blobs, "a.patch", patch),))
    base = PackageBase("sglang", package, "sglang", "0.5.4")

    def locate(layout: ComponentLayout) -> PackageBase:
        return base

    return compile_environment(recipe, RecipeDelta(), blobs=blobs, active_image_digest=IMAGE, locator=locate)


def _point_with_python(tmp_path, python) -> Path:
    path = tmp_path / "patched.yaml"
    envs = {"TP": 1, "CONC": 4, "RUN_EVAL": "false", "HYPERLOOM_FRAMEWORK_PYTHON": str(python)}
    path.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": envs}}), encoding="utf-8")
    return path


def test_a_patched_test_measures_the_rebuilt_framework_that_the_compile_produced(
    monkeypatch, tmp_path, installed_framework
):
    import subprocess

    from hyperloom.measure.tests._recipes import IMAGE

    python, package = installed_framework
    compiled = _compile(tmp_path, installed_framework)
    imported: dict[str, str] = {}

    async def look(**kwargs):
        envs = yaml.safe_load(Path(kwargs["recipe"].base_yaml_path).read_text())["benchmark"]["envs"]
        done = subprocess.run(
            [str(python), "-c", "import sglang.srt.a as a; print(a.f())"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={"PYTHONPATH": envs["PYTHONPATH"], "PATH": ""},
        )
        imported.update(value=done.stdout.strip(), path=envs["PYTHONPATH"].split(":")[0])
        return [VariantResult(name="test", extra_server_args="", extra_envs={}, status="succeeded", returncode=0)]

    monkeypatch.setattr(local, "run_grid", look)

    outcome = _run(_call(tmp_path, _point_with_python(tmp_path, python), environment=compiled))

    assert imported["value"] == "10"  # the server's interpreter imported the patched file, not the image's
    assert (package / "srt" / "a.py").read_text() == OLD_SOURCE  # and the image's was not written
    assert not Path(imported["path"]).exists()  # the farm is removed when the test ends
    assert outcome.sandbox.image_digest == IMAGE


def test_a_patched_test_tells_the_rounds_verdict_that_it_changed_the_framework(
    monkeypatch, tmp_path, installed_framework
):
    python, _ = installed_framework
    seen: list[bool] = []
    ended = local._ended

    def record(run):
        seen.append(run.call.framework_changed)
        return ended(run)

    async def look(**kwargs):
        return [VariantResult(name="test", extra_server_args="", extra_envs={}, status="succeeded", returncode=0)]

    monkeypatch.setattr(local, "run_grid", look)
    monkeypatch.setattr(local, "_ended", record)
    point = _point_with_python(tmp_path, python)

    _run(_call(tmp_path, point, environment=_compile(tmp_path, installed_framework)))
    _run(_call(tmp_path, point))

    assert seen[0] is True and seen[-1] is False


def _agentx_point(tmp_path, monkeypatch, conc: int = 4, **launch_envs: str) -> Path:
    """The point the toolbox renders for an AgentX session at ``conc`` whose recipe launches with ``launch_envs``."""
    from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
    from hyperloom.inference_optimizer.cli.benchmark_config import AgentXWorkload, write_agentx_benchmark
    from hyperloom.orchestrator.actions.executors import LaunchLayer, materialize_benchmark_config

    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.setenv(BENCHMARK_CONFIG_ENV, "")
    source = tmp_path / f"agentx-c{conc}.yaml"
    workload = AgentXWorkload(framework="sglang", model="Qwen/Qwen3-8B", precision="fp8", tp=2, ep=1, conc=conc)
    write_agentx_benchmark(source, workload)
    envs = {**launch_envs, "RUN_EVAL": "false"}
    return materialize_benchmark_config(source, tmp_path / f"agentx-c{conc}", launch=LaunchLayer(extra_envs=envs))


def _agentx_launch(monkeypatch, tmp_path, python) -> dict[str, Any]:
    """What ``run_grid`` hands the round for the point, and what the server it launches then imports from ``a.f()``."""
    import subprocess

    from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

    launched: dict[str, Any] = {}

    async def look(**kwargs):
        candidate = _build_variant_yaml(kwargs["recipe"], kwargs["grid"][0], output_subdir=tmp_path / "candidate")
        launch = yaml.safe_load(candidate.read_text())["benchmark"]["agentx"]["launch_overrides"]
        done = subprocess.run(
            [launch.get("executable") or "python3", "-c", "import sglang.srt.a as a; print(a.f())"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={"PYTHONPATH": launch["env"]["PYTHONPATH"], "PATH": str(Path(python).parent)},
        )
        launched.update(value=done.stdout.strip(), launch=launch)
        return [VariantResult(name="test", extra_server_args="", extra_envs={}, status="succeeded", returncode=0)]

    monkeypatch.setattr(local, "run_grid", look)
    return launched


def test_a_patched_agentx_test_hands_the_rebuilt_framework_to_the_server_the_round_launches(
    monkeypatch, tmp_path, installed_framework
):
    """``benchmark.envs`` never reaches an AgentX server: its environment and interpreter are the launch's."""
    python, package = installed_framework
    launched = _agentx_launch(monkeypatch, tmp_path, python)
    point = _agentx_point(tmp_path, monkeypatch, HYPERLOOM_FRAMEWORK_PYTHON=str(python))

    _run(_call(tmp_path, point, environment=_compile(tmp_path, installed_framework)))

    launch = launched["launch"]
    assert launched["value"] == "10"  # the interpreter the round launches imported the patched file, not the image's
    assert launch["executable"] == str(python) and "HYPERLOOM_FRAMEWORK_PYTHON" not in launch["env"]
    assert launch["env"]["PYTHONPATH"].split(":")[0] in "".join(launch["source_files"]), "rebuilt files are evidence"
    assert (package / "srt" / "a.py").read_text() == OLD_SOURCE
    assert "PYTHONPATH" not in yaml.safe_load(point.read_text())["benchmark"]["envs"]


def test_a_patched_agentx_test_that_names_no_interpreter_is_rebuilt_for_the_python3_the_round_launches(
    monkeypatch, tmp_path, installed_framework
):
    """The round starts the server with ``python3`` unless told otherwise; the farm is built and checked against that."""
    python, _ = installed_framework
    monkeypatch.setenv("PATH", f"{python.parent}{os.pathsep}{os.environ['PATH']}")
    launched = _agentx_launch(monkeypatch, tmp_path, python)
    point = _agentx_point(tmp_path, monkeypatch)

    _run(_call(tmp_path, point, environment=_compile(tmp_path, installed_framework)))

    assert launched["value"] == "10" and "executable" not in launched["launch"]


def test_a_profile_whose_patch_cannot_be_laid_in_the_rebuilt_framework_runs_and_its_record_says_so(
    grid, monkeypatch, tmp_path, installed_framework
):
    """A point rendered with TraceLens's graph-capture patch, whose candidate rebuilt the framework the patch cannot be
    laid in: no flag of the point needs the patch, so the profile runs, and its note and points say it ran without."""
    python, _ = installed_framework
    monkeypatch.setattr(farm_module, "lay_tracelens_patch", lambda package, site: False)
    monkeypatch.setattr(farm_module, "ensure_patched_copy", lambda package, checkout: pytest.fail("a copy was laid"))
    _with_trace(grid)
    point = _profile_point(tmp_path)
    config = yaml.safe_load(point.read_text())
    config["benchmark"]["envs"].update(
        {"HYPERLOOM_FRAMEWORK_PYTHON": str(python), "HYPERLOOM_TRACELENS_PATCH_STATUS": "ok"}
    )
    point.write_text(yaml.safe_dump(config), encoding="utf-8")

    outcome = _run(_call(tmp_path, point, kind="profile", environment=_compile(tmp_path, installed_framework)))

    assert outcome.returncode == 0 and len(grid.calls) == 1
    assert grid.configs[0]["benchmark"]["envs"]["HYPERLOOM_TRACELENS_PATCH_STATUS"] == "unavailable"
    assert [note for note in outcome.notes if note.startswith(f"{farm_module.PATCH_NOT_LAID}: ")] == [
        f"{farm_module.PATCH_NOT_LAID}: TraceLens's patch cannot be laid in the sglang the test rebuilt, so the profiled "
        "server runs it without the patch and its trace lacks what the patch adds, such as the graph-capture tags; "
        "HYPERLOOM_TRACELENS_PATCH_STATUS=unavailable"
    ]


def test_a_test_on_the_image_as_it_is_runs_the_point_unchanged(grid, unheld, tmp_path, installed_framework):
    from hyperloom.contracts.recipe import RecipeDelta
    from hyperloom.measure.compile import LocalBlobStore, compile_environment
    from hyperloom.measure.tests._recipes import IMAGE, recipe_with

    compiled = compile_environment(
        recipe_with(), RecipeDelta(), blobs=LocalBlobStore(tmp_path / "blobs"), active_image_digest=IMAGE
    )
    call = _call(tmp_path, environment=compiled)

    _run(call)

    _own_port(grid.configs[0])
    assert compiled.is_base and grid.configs[0] == yaml.safe_load(call.points[0].read_text())


def test_an_environment_that_cannot_be_rebuilt_is_refused_with_the_reason(grid, tmp_path, installed_framework):
    compiled = _compile(tmp_path, installed_framework)
    point = _point_with_python(tmp_path, "/no/such/python")

    with pytest.raises(TestCallRefused, match="cannot be rebuilt here"):
        _run(_call(tmp_path, point, environment=compiled))
    assert grid.calls == []


def test_a_patch_to_aiters_csrc_is_refused_where_no_jit_session_keeps_the_images_modules_out(
    grid, tmp_path, installed_framework
):
    from hyperloom.measure.tests.test_compile_aiter_meta import aiter_image, compile_aiter_environment

    compiled = compile_aiter_environment(
        tmp_path, aiter_image(tmp_path / "aiter-image"), {"csrc/kernels/m0.cu": "// x\n"}
    )
    point = _point_with_python(tmp_path, installed_framework[0])

    with pytest.raises(TestCallRefused, match="changes aiter's csrc or hsa"):
        _run(_call(tmp_path, point, environment=compiled))
    assert grid.calls == []


# --- profile test -------------------------------------------------------------------------------------------------


def _profile_point(directory: Path, *, profile: object = "1", enabled: bool = True, body: str | None = None) -> Path:
    envs: dict[str, object] = {"TP": 1, "CONC": 4, "RUN_EVAL": "false"}
    if body is not None:
        envs["PROFILE_EXTRA_BODY"] = body
    if profile is not None:
        envs["PROFILE"] = profile
    config = {"benchmark": {"framework": "sglang", "envs": envs, "profiler": {"torch_profiler": {"enabled": enabled}}}}
    path = directory / "profile.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _with_trace(grid: _Grid, name: str = "test", where: str = "torch_trace") -> VariantResult:
    result = grid.result(name)
    trace = RunLayout(Path(result.workspace)).profile_dir / where / "rank0.pt.trace.json.gz"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_bytes(b"trace")
    grid.answers[name] = result
    return result


def test_a_profile_test_is_one_profiled_round_with_no_warmup_before_it(grid, tmp_path):
    _with_trace(grid)
    call = _call(tmp_path, _profile_point(tmp_path), kind="profile", rounds=RoundPlan(warmup_rounds=1))

    outcome = _run(call)

    (kwargs,) = grid.calls
    assert [v.name for v in kwargs["grid"]] == ["test"]
    assert kwargs["launch"].warmup_before_measure is False, "a profiled round is not a statistic: no warmup before it"
    assert outcome.returncode == 0 and [r.phase for r in outcome.rounds] == ["measure"]


def test_a_profile_test_puts_the_points_start_profile_body_in_the_process_environment(grid, tmp_path, monkeypatch):
    """The benchmark client reads PROFILE_EXTRA_BODY from its own environment; the point's envs never reach it."""
    monkeypatch.delenv("PROFILE_EXTRA_BODY", raising=False)
    _with_trace(grid)
    body = '{"start_step": 320, "num_steps": 128, "profile_by_stage": false}'

    _run(_call(tmp_path, _profile_point(tmp_path, body=body), kind="profile"))

    assert [seen["PROFILE_EXTRA_BODY"] for seen in grid.env_seen] == [body]
    assert "PROFILE_EXTRA_BODY" not in os.environ, "the test's environment is put back afterwards"


def _vllm_profile_point(directory: Path) -> Path:
    """A vLLM profile point rendered on another host whose vLLM took TraceLens's patch (a hosted app container)."""
    point = _profile_point(directory)
    config = yaml.safe_load(point.read_text())
    config["benchmark"]["framework"] = "vllm"
    config["benchmark"]["envs"]["EXTRA_VLLM_ARGS"] = (
        "--profiler-config.capture_torch_profiler True --profiler-config.detailed_trace_annotation True"
    )
    point.write_text(yaml.safe_dump(config), encoding="utf-8")
    return point


def _host_copy(monkeypatch, tmp_path: Path) -> tuple[Path, list[str]]:
    """This host's patched copy of the profiled framework, and the packages a copy was laid for."""
    site = tmp_path / "tmp" / "hyperloom-tracelens" / "fw-0123" / "site"
    laid: list[str] = []
    monkeypatch.setattr(
        farm_module, "ensure_patched_copy", lambda package, checkout: laid.append(package) or PatchedServer(site)
    )
    return site, laid


def test_a_profile_asking_for_patched_profiler_fields_runs_this_hosts_patched_copy(grid, tmp_path, monkeypatch):
    site, laid = _host_copy(monkeypatch, tmp_path)
    _with_trace(grid)
    point = _vllm_profile_point(tmp_path)
    accepted = point.read_text()

    _run(_call(tmp_path, point, kind="profile"))

    assert laid == ["vllm"], "the copy is laid for the server's framework, never the installed one patched"
    path = grid.configs[0]["benchmark"]["envs"]["PYTHONPATH"].split(os.pathsep)
    assert path[-1] == str(site), "a point that names no copy gets this host's after what it inherits"
    assert point.read_text() == accepted, "the accepted point is not rewritten"


def test_a_profile_rendered_on_another_host_runs_this_hosts_copy_where_the_other_was(grid, tmp_path, monkeypatch):
    site, laid = _host_copy(monkeypatch, tmp_path)
    _with_trace(grid)
    config = yaml.safe_load(_profile_point(tmp_path).read_text())
    rendered = os.pathsep.join(["/farm/site", "/app/tmp/hyperloom-tracelens/sglang-gc-ab/site", "/tools/kernel_shape"])
    config["benchmark"]["envs"].update({"HYPERLOOM_TRACELENS_PATCH_STATUS": "ok", "PYTHONPATH": rendered})
    point = tmp_path / "rendered.yaml"
    point.write_text(yaml.safe_dump(config), encoding="utf-8")

    _run(_call(tmp_path, point, kind="profile"))

    assert laid == ["sglang"]
    assert grid.configs[0]["benchmark"]["envs"]["PYTHONPATH"] == os.pathsep.join(
        ["/farm/site", str(site), "/tools/kernel_shape"]
    ), "behind a farm, which carries its own patch, and ahead of the installed framework"


def test_a_profile_asking_for_patched_profiler_fields_on_an_unpatchable_vllm_is_refused(grid, tmp_path, monkeypatch):
    monkeypatch.setattr(farm_module, "ensure_patched_copy", lambda package, checkout: None)

    with pytest.raises(TestCallRefused, match="only TraceLens's patch adds, and the patch cannot be applied"):
        _run(_call(tmp_path, _vllm_profile_point(tmp_path), kind="profile"))
    assert grid.calls == []


def test_a_profile_rendered_without_the_patch_runs_as_it_was_rendered(grid, tmp_path, monkeypatch):
    monkeypatch.setattr(farm_module, "ensure_patched_copy", lambda package, checkout: pytest.fail("a copy was laid"))
    _with_trace(grid)
    point = _profile_point(tmp_path)

    _run(_call(tmp_path, point, kind="profile"))

    assert "PYTHONPATH" not in grid.configs[0]["benchmark"]["envs"], "nothing is added to the point"


def test_a_test_that_does_not_profile_leaves_the_checkout_as_upstream_ships_it(grid, tmp_path):
    root = stock_checkout(tmp_path / "ix")
    config = yaml.safe_load(_point(tmp_path, "p").read_text())
    config["benchmark"]["inferencex_path"] = str(root)
    point = tmp_path / "plain.yaml"
    point.write_text(yaml.safe_dump(config), encoding="utf-8")

    _run(_call(tmp_path, point))

    assert (root / LIBRARY_PATH).read_text() == LIBRARY and (root / CLIENT_PATH).read_text() == CLIENT


def test_the_traces_of_a_profile_test_move_into_the_output_profile_directory(grid, tmp_path):
    result = _with_trace(grid)
    sidecar = RunLayout(Path(result.workspace)).profile_dir / "capture_traces" / "graph.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{}")

    _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    profile = tmp_path / "out" / "profile"
    assert (profile / "torch_trace" / "rank0.pt.trace.json.gz").read_bytes() == b"trace"
    assert (profile / "capture_traces" / "graph.json").is_file(), "the capture sidecars keep their directory name"
    assert not RunLayout(Path(result.workspace)).profile_torch_trace.exists(), "the bundle must carry each trace once"


def test_a_profile_test_that_left_no_trace_does_not_end_as_a_success(grid, tmp_path):
    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert outcome.returncode != 0 and "profile_no_trace" in outcome.stderr
    assert not (tmp_path / "out" / "profile").exists()


def test_a_failed_profile_round_moves_no_trace(grid, tmp_path):
    result = _with_trace(grid)
    grid.answers["test"] = VariantResult(**{**result.__dict__, "status": "failed", "returncode": 1})

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert outcome.returncode == 1 and "profile_no_trace" not in outcome.stderr
    assert RunLayout(Path(result.workspace)).profile_torch_trace.exists()


@pytest.mark.parametrize(
    ("point", "kwargs", "match"),
    [
        (dict(profile=None), {}, "does not turn the torch profiler on"),
        (dict(profile="0"), {}, "does not turn the torch profiler on"),
        (dict(enabled=False), {}, "does not turn the torch profiler on"),
        ({}, dict(accuracy=True), "does not run accuracy"),
        ({}, dict(prep="conc_sweep"), "not a sweep"),
    ],
)
def test_a_profile_call_that_would_not_profile_or_asks_for_more_is_refused(grid, tmp_path, point, kwargs, match):
    run_eval = "true" if kwargs.get("accuracy") else "false"
    path = _profile_point(tmp_path, **point)
    if run_eval == "true":
        config = yaml.safe_load(path.read_text())
        config["benchmark"]["envs"]["RUN_EVAL"] = "true"
        path.write_text(yaml.safe_dump(config))

    with pytest.raises(TestCallRefused, match=match):
        _run(_call(tmp_path, path, kind="profile", **kwargs))
    assert grid.calls == []


def test_an_agentx_profile_test_is_one_profiled_round_that_turns_the_profiler_on_in_the_yaml(
    grid, tmp_path, monkeypatch
):
    from hyperloom.orchestrator.actions.executors.agentx_point import enable_profile

    point = _agentx_point(tmp_path, monkeypatch)
    enable_profile(point)
    capture = "0123456789abcdef0123456789abcdef"
    _with_trace(grid, where=f"torch_trace/{capture}")

    outcome = _run(_call(tmp_path, point, kind="profile", rounds=RoundPlan(warmup_rounds=1)))

    (kwargs,) = grid.calls
    _own_port(grid.configs[0])
    assert kwargs["launch"].warmup_before_measure is False and grid.configs[0] == yaml.safe_load(point.read_text())
    assert outcome.returncode == 0
    assert (tmp_path / "out" / "profile" / "torch_trace" / capture / "rank0.pt.trace.json.gz").is_file()


def test_an_agentx_profile_call_whose_point_leaves_the_profiler_off_is_refused(grid, tmp_path, monkeypatch):
    point = _agentx_point(tmp_path, monkeypatch)

    with pytest.raises(TestCallRefused, match="does not turn the torch profiler on"):
        _run(_call(tmp_path, point, kind="profile"))
    assert grid.calls == []


def test_a_plain_test_does_not_gather_traces(grid, tmp_path):
    _with_trace(grid)

    _run(_call(tmp_path))

    assert not (tmp_path / "out" / "profile").exists()


class _Machine:
    """One machine's ports: ``run_grid`` as it is for a round whose server is up, listening on the port the round's own
    config names in ``benchmark.envs.PORT`` (the default when it names none).

    The InferenceX script reads it from the environment it is run with; an AgentX recipe reads it into the ``--port``
    of the server's argv and ``PORT`` is not one of ``launch_overrides.env``, so the AgentX server's port can come from
    nowhere else.

    The listeners stay open to the end of the test, as the servers of tests running at the same time do; a round whose
    port is taken fails the way the real one did: its server dies.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.ports: list[int] = []
        self.launches: list[dict[str, Any]] = []
        self.listeners: list[socket.socket] = []

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

        variant = kwargs["grid"][0]
        slot = self.tmp_path / "slots" / str(len(self.ports))
        launched = _build_variant_yaml(kwargs["recipe"], variant, output_subdir=slot)
        self.launches.append(benchmark := yaml.safe_load(launched.read_text())["benchmark"])
        self.ports.append(port := int((benchmark.get("envs") or {}).get("PORT", REUSE_PORT_DEFAULT)))
        workspace = slot / "suite_run_0"
        RunLayout(workspace).profile_torch_trace.mkdir(parents=True)
        (RunLayout(workspace).profile_torch_trace / "rank0.pt.trace.json.gz").write_bytes(b"trace")
        listener = socket.socket()
        try:
            listener.bind(("127.0.0.1", port))
            listener.listen()
        except OSError as exc:
            listener.close()
            error = f"Server process died during benchmark: {exc}"
            return [
                VariantResult(
                    variant.name, "", {}, status="failed", returncode=1, workspace=str(workspace), error=error
                )
            ]
        self.listeners.append(listener)
        return [VariantResult(variant.name, "", {}, status="succeeded", returncode=0, workspace=str(workspace))]


@pytest.fixture
def machine(monkeypatch, tmp_path):
    fake = _Machine(tmp_path)
    monkeypatch.setattr(local, "run_grid", fake)
    yield fake
    for listener in fake.listeners:
        listener.close()


def _profile_agentx_point(tmp_path: Path, monkeypatch) -> Path:
    from hyperloom.orchestrator.actions.executors.agentx_point import enable_profile

    point = _agentx_point(tmp_path, monkeypatch)
    enable_profile(point)
    return point


@pytest.mark.parametrize("make_point", [_profile_point, _profile_agentx_point], ids=["inferencex", "agentx"])
def test_two_profile_tests_on_one_machine_launch_their_servers_on_ports_of_their_own(
    machine, tmp_path, monkeypatch, make_point
):
    """A profile round is not held by the server lifecycle, so no port was assigned to it: both listened on 8888."""
    point = make_point(tmp_path) if make_point is _profile_point else make_point(tmp_path, monkeypatch)
    shared = point.read_text()

    first = _run(_call(tmp_path, point, kind="profile", output_dir=tmp_path / "first"))
    second = _run(_call(tmp_path, point, kind="profile", output_dir=tmp_path / "second"))

    assert [first.returncode, second.returncode] == [0, 0], "the second server found its port taken"
    assert len(set(machine.ports)) == 2 and REUSE_PORT_DEFAULT not in machine.ports
    server_envs = [((b.get("agentx") or {}).get("launch_overrides") or {}).get("env") or {} for b in machine.launches]
    assert all("PORT" not in env for env in server_envs), "PORT is never among an AgentX server's variables"
    assert point.read_text() == shared, "the point is shared by every test that runs it: it is not written to"


# --- repro test ---------------------------------------------------------------------------------------------------

_CASE = AttachmentRef("case", "dir", "a" * 64)


class _Case:
    """A repro case unpacked where the runner leaves it, and the order the server and the case ran in."""

    def __init__(self, tmp_path: Path, monkeypatch, body: str) -> None:
        script = tmp_path / "attach" / _CASE.id / "repro.sh"
        script.parent.mkdir(parents=True)
        script.write_text(f"{body}\n", encoding="utf-8")
        self.ran = tmp_path / "case-ran"
        self.events: list[str] = []

        def teardown(slot: Path) -> list[int]:
            self.events.append("teardown after the case" if self.ran.exists() else "teardown before the case")
            return []

        monkeypatch.setattr(local, "teardown_variant_server", teardown)


def _repro_point(directory: Path) -> Path:
    path = directory / "repro-point.yaml"
    config = {"benchmark": {"framework": "sglang", "model": "/models/m", "envs": {"TP": 1, "RUN_EVAL": "false"}}}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _repro_call(tmp_path: Path, **kwargs: Any) -> TestCall:
    defaults: dict[str, Any] = {"kind": "repro", "attachments": (_CASE,), "timeout_sec": 30}
    return _call(tmp_path, _repro_point(tmp_path), **{**defaults, **kwargs})


def _boot_only(grid: _Grid) -> None:
    grid.answers["repro"] = grid.result("repro", note="server_lifecycle_boot_only")


def test_repro_is_a_kind_the_local_backend_runs():
    assert "repro" in SUPPORTED_KINDS


@pytest.mark.parametrize(("code", "verdict"), [(0, "not_reproduced"), (1, "reproduced")])
@pytest.mark.usefixtures("server")
def test_a_repro_test_boots_the_server_runs_the_case_against_it_and_reports_the_verdict(
    grid, monkeypatch, tmp_path, code, verdict
):
    _boot_only(grid)
    case = _Case(tmp_path, monkeypatch, f'echo "$HL_BASE_URL $HL_MODEL" > {tmp_path}/case-ran\nexit {code}')

    outcome = _run(_repro_call(tmp_path))

    (kwargs,) = grid.calls
    assert [v.name for v in kwargs["grid"]] == ["repro"]
    assert kwargs["launch"].lifecycle_boot_only is True and kwargs["launch"].warmup_before_measure is False
    assert kwargs["recipe"].base.extra_envs == BOOT_ENVS, "the repro boot is not an evaluation"
    assert kwargs["launch"].server_lifecycle["cleanup"] is False and kwargs["launch"].server_lifecycle["port"] == 8123
    assert (tmp_path / "case-ran").read_text() == "http://127.0.0.1:8123 /models/m\n"
    assert case.events == ["teardown after the case"], "the server stays up for the case and is reclaimed after it"
    assert read_verdict(tmp_path / "out") == verdict
    assert (outcome.returncode, outcome.failure_class, outcome.rounds) == (0, None, ())


def test_a_repro_test_runs_no_warmup_even_when_the_call_plans_one(grid, server, monkeypatch, tmp_path):
    _boot_only(grid)
    _Case(tmp_path, monkeypatch, "exit 0")

    _run(_repro_call(tmp_path, rounds=RoundPlan(warmup_rounds=1)))

    assert [c["launch"].warmup_before_measure for c in grid.calls] == [False]


@pytest.mark.parametrize(
    ("body", "reason"),
    [("exit 2", "exited with 2"), ("sleep 30", "did not finish within"), (None, "no attachment holds repro.sh")],
)
@pytest.mark.usefixtures("server")
def test_a_case_that_cannot_run_to_a_verdict_is_a_failure_of_the_measurement_not_a_verdict(
    grid, monkeypatch, tmp_path, body, reason
):
    _boot_only(grid)
    case = _Case(tmp_path, monkeypatch, body or "")
    if body is None:
        (tmp_path / "attach" / _CASE.id / "repro.sh").unlink()

    outcome = _run(_repro_call(tmp_path, timeout_sec=1))

    assert outcome.failure_class == "infra" and outcome.returncode == local.HOSTED_INFRA_FAILURE_RC
    assert reason in outcome.stderr and "repro_case_error" in outcome.stderr
    assert not (tmp_path / "out" / "repro" / "verdict.json").exists()
    assert len(case.events) == 1, "the server is reclaimed whatever became of the case"


def test_a_server_that_does_not_boot_fails_the_test_like_any_test_and_runs_no_case(grid, server, monkeypatch, tmp_path):
    grid.answers["repro"] = grid.result("repro", status="failed", returncode=1, error="server died")
    case = _Case(tmp_path, monkeypatch, f"touch {tmp_path}/case-ran\nexit 0")

    outcome = _run(_repro_call(tmp_path))

    assert outcome.returncode == 1 and outcome.failure_class in ("candidate", "unclear")
    assert not (tmp_path / "case-ran").exists() and not (tmp_path / "out" / "repro").exists()
    assert case.events == ["teardown before the case"]


def test_a_server_that_cannot_be_held_up_is_a_failure_of_the_measurement_and_boots_nothing(grid, monkeypatch, tmp_path):
    _Server(monkeypatch, eligible=False)
    _Case(tmp_path, monkeypatch, "exit 0")

    outcome = _run(_repro_call(tmp_path))

    assert (
        outcome.failure_class == "infra" and "cannot be held up" in outcome.stderr and "not eligible" in outcome.stderr
    )
    assert grid.calls == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"attachments": ()}, "needs the reproduction case as an attachment"),
        ({"accuracy": True}, "does not run accuracy"),
        ({"prep": "conc_sweep"}, "not a sweep"),
    ],
)
def test_a_repro_call_that_is_not_one_case_against_one_server_is_refused(grid, server, tmp_path, kwargs, match):
    with pytest.raises(TestCallRefused, match=match):
        _run(_repro_call(tmp_path, **kwargs))
    assert grid.calls == []


def _failed_round_with_logs(grid, tmp_path, *, server_log: str, benchmark_stderr: str = "") -> VariantResult:
    workspace = tmp_path / "ws" / "test"
    workspace.mkdir(parents=True)
    (workspace / "server.log").write_text(server_log, encoding="utf-8")
    if benchmark_stderr:
        (workspace / "benchmark_stderr.log").write_text(benchmark_stderr, encoding="utf-8")
    return grid.result("test", status="failed", returncode=1, error="", server_log_path=str(workspace / "server.log"))


def test_a_failed_round_is_classified_from_the_flag_error_in_its_server_log(grid, tmp_path):
    flag_error = "launch_server.py: error: ambiguous option: --chunked-prefill could match --chunked-prefill-size"
    grid.answers["test"] = _failed_round_with_logs(
        grid, tmp_path, server_log=f"usage: launch_server.py [-h]\n{flag_error}\n"
    )

    outcome = _run(_call(tmp_path))

    assert (outcome.returncode, outcome.failure_class) == (1, "candidate")
    assert outcome.failure_reason == f"serve_flag: {flag_error}"


def test_a_round_whose_replay_client_the_hub_refused_is_an_infrastructure_failure(grid, tmp_path):
    """The round's report names only the client's exit code; the Hub's 429 is in the run_dir's AgentX client log."""
    refusal = "429 Too Many Requests: you have reached your 'api' rate limit."
    result = _failed_round_with_logs(grid, tmp_path, server_log="INFO waiting for the server to be ready\n")
    client_log = RunLayout(Path(result.workspace)).agentx_client_log
    client_log.parent.mkdir(parents=True, exist_ok=True)
    client_log.write_text(f"Fetching 9 files\nhuggingface_hub.errors.HfHubHTTPError: \n\n{refusal}\n", encoding="utf-8")
    grid.answers["test"] = result

    outcome = _run(_call(tmp_path))

    assert (outcome.failure_class, outcome.failure_reason) == ("infra", f"hub_rate_limited: {refusal}")


def test_a_failed_round_is_classified_from_the_refused_request_in_its_benchmark_stderr(grid, tmp_path):
    refusal = 'API request failed with error message: {"message":"maximum context length is 5120 tokens","code":400}'
    grid.answers["test"] = _failed_round_with_logs(
        grid, tmp_path, server_log="INFO server booted\n", benchmark_stderr=f"WARNING {refusal}\n"
    )

    outcome = _run(_call(tmp_path))

    assert (outcome.failure_class, outcome.failure_reason) == ("unclear", f"lm_eval_http_400: WARNING {refusal}")


# What the server and the evaluation client print when the evaluation's request is longer than the served context
# (``--context-length 400``): the benchmark served its whole protocol, so run_grid kept the round ``succeeded`` with the
# evaluation's exit code.
_REFUSED_REQUEST = (
    "2000-01-01:00:00:00 WARNING  [models.api_models:521] API request failed! Status code: 400, Response text: "
    '{"object":"error","message":"The input (1000 tokens) is longer than the model\'s context length (400 tokens).",'
    '"type":"BadRequestError","param":null,"code":400}. Retrying...'
)
_SESSION_CLOSED = """\
Task exception was never retrieved
future: <Task finished name='Task-1' coro=<tqdm_asyncio.gather.<locals>.wrap_awaitable() done, defined at /opt/venv/lib/python3.10/site-packages/tqdm/asyncio.py:75> exception=RuntimeError('Session is closed')>
Traceback (most recent call last):
  File "/opt/venv/lib/python3.10/site-packages/lm_eval/models/api_models.py", line 514, in amodel_call
    async with session.post(
  File "/opt/venv/lib/python3.10/site-packages/aiohttp/client.py", line 1756, in __aenter__
    self._resp: _RetType_co = await self._coro
  File "/opt/venv/lib/python3.10/site-packages/aiohttp/client.py", line 615, in _request
    raise RuntimeError("Session is closed")
RuntimeError: Session is closed
"""
_ACCESS_400 = '[2000-01-01 00:00:00] INFO:     127.0.0.1:40000 - "POST /v1/completions HTTP/1.1" 400 Bad Request\n'


def test_a_refused_evaluation_names_the_refusal_that_a_flood_of_tracebacks_pushed_out_of_the_log_tail(
    grid, server, tmp_path
):
    grid.answers["eval"] = _failed_evaluation(
        grid, tmp_path, eval_refusal_logs.server_log(), eval_refusal_logs.benchmark_stderr()
    )

    outcome = _run(_graded_call(tmp_path))

    assert (outcome.returncode, outcome.failure_class) == (1, "candidate")
    reason = outcome.failure_reason or ""
    assert reason.startswith("lm_eval_http_400: "), "the refusal, not the generic end of the evaluation"
    assert "max_completion_tokens is too large: 4096" in reason and "at most 2304 completion tokens" in reason


def _log_with_refusal_at(tmp_path: Path, *, from_end: int, size: int) -> Path:
    """A log of ``size`` bytes of progress lines whose refusal line starts ``from_end`` bytes before its end."""
    line = eval_refusal_logs.REFUSAL + "\n"
    filler = "Requesting API: 100%|##########| 1319/1319\n"
    body = bytearray((filler * (size // len(filler) + 1)).encode()[:size])
    start = size - from_end
    body[start - 1 : start + len(line)] = b"\n" + line.encode()
    path = tmp_path / "benchmark_stderr.log"
    path.write_bytes(bytes(body))
    return path


@pytest.mark.parametrize(
    "from_end",
    [
        3 * (1 << 20) + 500,  # a few blocks up, inside one block
        (1 << 20) + 100,  # the line starts just above the first block's start: it ends inside that block
        (1 << 20) + len(eval_refusal_logs.REFUSAL) // 2,  # and the block boundary cuts the line in two
    ],
)
def test_the_refusal_is_found_wherever_it_sits_in_the_scanned_part_of_a_log(tmp_path, from_end):
    path = _log_with_refusal_at(tmp_path, from_end=from_end, size=5 * (1 << 20))

    assert local._read_refusal(path) == eval_refusal_logs.REFUSAL, "the whole line, not the half a block cut"


def test_a_refusal_older_than_the_scanned_part_of_a_log_is_not_searched_for(tmp_path):
    path = _log_with_refusal_at(
        tmp_path, from_end=local._REFUSAL_SCAN_CHARS + 1000, size=local._REFUSAL_SCAN_CHARS + 2000
    )

    assert local._read_refusal(path) == ""
    assert local._read_refusal(tmp_path / "missing.log") == ""


def test_an_evaluation_that_fails_is_classified_from_the_logs_it_wrote(grid, server, tmp_path):
    grid.answers["eval"] = _failed_evaluation(
        grid, tmp_path, _ACCESS_400 * 4, _REFUSED_REQUEST + "\n" + _SESSION_CLOSED * 40, error=_SESSION_CLOSED * 2
    )

    outcome = _run(_graded_call(tmp_path))

    assert [c["grid"][0].name for c in grid.calls] == ["warmup", "test", "eval"]
    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 0), ("measure", 0), ("accuracy", 1)]
    assert (outcome.returncode, outcome.failure_class) == (1, "unclear")
    reason = outcome.failure_reason or ""
    assert reason.startswith("lm_eval_http_400: "), "the refusal, not the session the client closed after it"
    assert "longer than the model's context length (400 tokens)" in reason
    assert len(server.teardowns) == 1


def _failed_evaluation(grid: _Grid, tmp_path: Path, server_log: str, stderr: str, *, error: str = "") -> VariantResult:
    """An evaluation round whose evaluation failed: it ran no bench, so it has no measurement and its round failed."""
    workspace = tmp_path / "ws" / "eval"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "server.log").write_text(server_log, encoding="utf-8")
    (workspace / "benchmark_stderr.log").write_text(stderr, encoding="utf-8")
    return grid.result(
        "eval",
        status="failed",
        returncode=1,
        error=error,
        error_class="round_nonzero_invalid_measurement",
        server_log_path=str(workspace / "server.log"),
    )


def _warmup_with_logs(grid: _Grid, tmp_path: Path, server_log: str, *, returncode: int) -> None:
    """A warmup round that ran its whole protocol (so its status is ``succeeded``) and left ``server_log``."""
    workspace = tmp_path / "ws" / "warmup"
    workspace.mkdir(parents=True)
    (workspace / "server.log").write_text(server_log, encoding="utf-8")
    grid.answers.update(
        warmup=grid.result(
            "warmup", returncode=returncode, output_throughput=1111.0, server_log_path=str(workspace / "server.log")
        ),
        test=grid.result("test"),
    )


@pytest.mark.parametrize(
    ("log", "failure_class", "reason"),
    [
        (
            test_log_causes.ILLEGAL_ADDRESS_LOG,
            "candidate",
            "hip_runtime_error: Exception: Capture cuda graph failed: HIP error: an illegal memory access was encountered",
        ),
        (
            test_log_causes.FASTSAFETENSORS_LOG,
            "candidate",
            "import_error: ImportError: Please install fastsafetensors via `pip install fastsafetensors`",
        ),
        (
            test_log_causes.ARGPARSE_LOG,
            "candidate",
            "serve_flag: launch_server.py: error: ambiguous option: --chunked-prefill could match",
        ),
        (
            test_log_causes.PYTHON_EXCEPTION_LOG,
            "unclear",
            "python_exception: AssertionError: M must be a multiple of 16, got 63",
        ),
    ],
    ids=["hip_runtime_error", "import_error", "serve_flag", "python_exception"],
)
def test_a_warmup_whose_server_crashed_or_refused_to_serve_ends_the_test_with_no_measured_round(
    grid, server, tmp_path, log, failure_class, reason
):
    _warmup_with_logs(grid, tmp_path, log, returncode=1)

    outcome = _run(_graded_call(tmp_path))

    assert [c["grid"][0].name for c in grid.calls] == ["warmup"], "no measured round boots a server that cannot serve"
    assert [r.phase for r in outcome.rounds] == ["warmup"] and outcome.returncode == 1
    assert outcome.failure_class == failure_class
    assert (outcome.failure_reason or "").startswith(reason)
    assert len(server.teardowns) == 1


@pytest.mark.parametrize(
    ("log", "failure_class", "reason"),
    [
        (test_log_causes.PLAIN_LOG, "unclear", "no_known_signature"),
        ("OSError: [Errno 98] Address already in use\n", "infra", "port_in_use"),
        (
            'Traceback (most recent call last):\n  File "triton/runtime/cache.py", line 1\n'
            "OSError: [Errno 122] Disk quota exceeded: '/work/.cache/triton/x.json'\n",
            "infra",
            "disk_full: OSError: [Errno 122] Disk quota exceeded: '/work/.cache/triton/x.json'",
        ),
    ],
    ids=["no_known_signature", "port_in_use", "disk_full_in_the_server_log"],
)
def test_a_warmup_that_failed_for_a_cause_no_log_names_keeps_the_measured_round_as_before(
    grid, server, tmp_path, log, failure_class, reason
):
    _warmup_with_logs(grid, tmp_path, log, returncode=1)

    outcome = _run(_graded_call(tmp_path))

    assert [c["grid"][0].name for c in grid.calls] == ["warmup", "test"], "a test that failed is not evaluated"
    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 1), ("measure", 0)]
    assert (outcome.failure_class, outcome.failure_reason) == (failure_class, reason)


def test_a_warmup_whose_server_log_is_healthy_goes_on_to_the_measured_round(grid, server, tmp_path):
    _warmup_with_logs(grid, tmp_path, test_log_causes.SERVER_LOG_WITH_BAD_REQUEST, returncode=0)
    evaluation = grid.result("eval")
    _evaluated(Path(evaluation.workspace), 0.64)
    grid.answers["eval"] = evaluation

    outcome = _run(_graded_call(tmp_path))

    assert [c["grid"][0].name for c in grid.calls] == ["warmup", "test", "eval"]
    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 0), ("measure", 0), ("accuracy", 0)]
    assert (outcome.returncode, outcome.failure_class, outcome.failure_reason) == (0, None, None)
